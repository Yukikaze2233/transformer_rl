"""Private continuous task sampling and nonrefundable curriculum bookkeeping.

Sampling covers requested domains. No sampled request is asserted physically
reachable; asset/actuator admission and independent validation remain external.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from enum import IntEnum
import math
import re

import torch

from .geometry import TerrainKind


class TaskFamily(IntEnum):
    HEIGHT = 0
    MOTION = 1
    ROTATION = 2
    TERRAIN = 3
    REQUEST = 4
    RECOVERY = 5


class RotationMode(IntEnum):
    """Separate rotation buckets so the full command box is not one bucket."""

    NORMAL = 0
    SPIN_LOW = 1
    SPIN_MID = 2
    DASH = 3


ROTATION_NORMAL_YAW_RAD_S = 2 * math.pi
ROTATION_SPIN_LOW_RAD_S = (2 * math.pi, 3.25 * math.pi)
ROTATION_SPIN_MID_RAD_S = (3.25 * math.pi, 6 * math.pi)
ROTATION_DASH_FORWARD_M_S = (2.0, 2.7)


STAGE_NAMES = ('warmup', 'rough', 'hard')
STAGE_BUDGETS = (5000, 20000, 5000)
STAGE_WEIGHTS = (
    (0.60, 0.20, 0.10, 0.00, 0.00, 0.10),
    (0.15, 0.25, 0.15, 0.35, 0.00, 0.10),
    (0.15, 0.15, 0.15, 0.35, 0.00, 0.20),
)
TOTAL_BUDGET = sum(STAGE_BUDGETS)
JUMP_BUDGET = 5000
JUMP_WEIGHTS = (0.20, 0.10, 0.10, 0.10, 0.50, 0.00)
COMPLETE_STAGE_NAMES = (*STAGE_NAMES, 'jump')
COMPLETE_STAGE_BUDGETS = (*STAGE_BUDGETS, JUMP_BUDGET)
COMPLETE_STAGE_WEIGHTS = (*STAGE_WEIGHTS, JUMP_WEIGHTS)
PHASE_NAMES = STAGE_NAMES
PHASE_BUDGETS = STAGE_BUDGETS
PHASE_WEIGHTS = STAGE_WEIGHTS


@dataclass(frozen=True)
class TaskDomain:
    height_m: tuple[float, float] = (0.17, 0.40)
    forward_m_s: tuple[float, float] = (-2.7, 2.7)
    yaw_rad_s: tuple[float, float] = (-6 * math.pi, 6 * math.pi)
    jump_m: tuple[float, float] = (0.03, 0.40)
    slope_rad: tuple[float, float] = (-math.pi / 12, math.pi / 12)
    roughness_m: tuple[float, float] = (0.0, 0.02)
    step_up_m: tuple[float, float] = (0.0, 0.20)
    step_down_m: tuple[float, float] = (0.0, 0.15)
    nominal_height_m: float = 0.32
    nominal_fraction: float = 0.20

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ('nominal_height_m', 'nominal_fraction'):
                continue
            if (not isinstance(value, (tuple, list)) or len(value) != 2
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                    or value[0] >= value[1]):
                raise ValueError(f'{field.name} requires two ordered finite bounds')
            object.__setattr__(self, field.name, tuple(value))
        if (type(self.nominal_height_m) not in (int, float) or not math.isfinite(self.nominal_height_m)
                or not self.height_m[0] <= self.nominal_height_m <= self.height_m[1]
                or type(self.nominal_fraction) not in (int, float) or not 0 < self.nominal_fraction < 1
                or not self.forward_m_s[0] < 0 < self.forward_m_s[1]
                or not self.yaw_rad_s[0] < 0 < self.yaw_rad_s[1]
                or self.height_m[0] <= 0 or self.jump_m[0] <= 0
                or any(getattr(self, name)[0] < 0 for name in ('roughness_m', 'step_up_m', 'step_down_m'))):
            raise ValueError('inconsistent task anchors or physical dimensions')


# The first exposure is deliberately narrower than the final task domain. The
# formal course expands every range continuously; it never switches from a
# flat-only world to the full course at one update boundary.
INITIAL_DOMAIN = TaskDomain(
    height_m=(0.28, 0.32),
    forward_m_s=(-0.5, 0.5),
    yaw_rad_s=(-1., 1.),
    jump_m=(0.03, 0.10),
    slope_rad=(-math.pi / 60, math.pi / 60),
    roughness_m=(0.0, 0.005),
    step_up_m=(0.0, 0.05),
    step_down_m=(0.0, 0.03),
)


def interpolate_domain(start: TaskDomain, end: TaskDomain, fraction: float) -> TaskDomain:
    """Linear continuous domain expansion with the declared endpoint values."""
    if (not isinstance(start, TaskDomain) or not isinstance(end, TaskDomain)
            or type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 <= fraction <= 1):
        raise ValueError('domain interpolation requires parsed endpoints and a finite [0,1] fraction')

    def blend(left, right):
        return (float(left[0] + (right[0] - left[0]) * fraction),
                float(left[1] + (right[1] - left[1]) * fraction))

    return replace(
        end,
        height_m=blend(start.height_m, end.height_m),
        forward_m_s=blend(start.forward_m_s, end.forward_m_s),
        yaw_rad_s=blend(start.yaw_rad_s, end.yaw_rad_s),
        jump_m=blend(start.jump_m, end.jump_m),
        slope_rad=blend(start.slope_rad, end.slope_rad),
        roughness_m=blend(start.roughness_m, end.roughness_m),
        step_up_m=blend(start.step_up_m, end.step_up_m),
        step_down_m=blend(start.step_down_m, end.step_down_m),
    )


class _FairQuota:
    """Carry fractional deficits across small reset batches without starving a pool."""

    def __init__(self, weights):
        if not weights or any(v < 0 or not math.isfinite(v) for v in weights) or abs(sum(weights) - 1) > 1e-9:
            raise ValueError('quota weights must be nonnegative and sum to one')
        self.weights = tuple(weights)
        self.credit = [0.0] * len(weights)

    def set_weights(self, weights):
        weights = tuple(weights)
        if (len(weights) != len(self.weights) or any(not math.isfinite(v) or v < 0 for v in weights)
                or abs(sum(weights) - 1) > 1e-9):
            raise ValueError('dynamic quota weights must be finite, nonnegative and sum to one')
        self.weights = weights
        active = [index for index, weight in enumerate(weights) if weight > 0]
        disabled = [index for index, weight in enumerate(weights) if weight == 0]
        if not active:
            self.credit = [0.0] * len(weights)
            return
        # A family disabled at a stage boundary cannot carry an old fractional
        # deficit into a checkpoint. Retire it and redistribute the deficit to
        # the still-active quotas so the row ledger remains conserved.
        removed = sum(self.credit[index] for index in disabled)
        for index in disabled:
            self.credit[index] = 0.0
        if removed:
            active_total = sum(weights[index] for index in active)
            for index in active:
                self.credit[index] += removed * weights[index] / active_total
        drift = sum(self.credit[index] for index in active)
        if drift:
            active_total = sum(weights[index] for index in active)
            for index in active:
                self.credit[index] -= drift * weights[index] / active_total

    def allocate(self, count):
        allocation = [0] * len(self.weights)
        active = [index for index, weight in enumerate(self.weights) if weight > 0]
        # This loop is scalar bookkeeping per resampled episode, not a per-tick
        # environment loop or a device synchronization. Six pools remain fixed.
        for _ in range(count):
            for index in active:
                self.credit[index] += self.weights[index]
            selected = max(active, key=lambda index: self.credit[index])
            allocation[selected] += 1
            self.credit[selected] -= 1.0
        return allocation


@dataclass(frozen=True)
class TaskDraw:
    family: torch.Tensor
    stratum: torch.Tensor
    requested_command: torch.Tensor  # body vx, body vy=0, yaw rate, height; SI units
    jump_height_m: torch.Tensor
    slope_rad: torch.Tensor
    roughness_m: torch.Tensor
    step_up_m: torch.Tensor
    step_down_m: torch.Tensor
    terrain_kind: torch.Tensor       # TerrainKind; stairs and two absolute platforms remain distinct
    recovery_posture: torch.Tensor   # references the frozen 16-case posture catalogue
    trajectory_kind: torch.Tensor    # hold, ramp, sine, chirp, smoothstep, start/stop, bounded step
    nominal: torch.Tensor
    rotation_mode: torch.Tensor      # RotationMode; separate normal/spin/dash buckets


class ContinuousTaskSampler:
    """Full-width continuous support, endpoint emphasis and an exact-zero mass.

    A dedicated generator is checkpointed with quota deficits. Calls with one
    reset row retain all family/stratum support over time; Latin-hypercube draws
    cover the interior within each family block, then rows are shuffled.
    """

    def __init__(self, seed, device='cpu', domain=TaskDomain(), *, stage_weights=STAGE_WEIGHTS,
                 continuous=False):
        if type(seed) is not int or not 0 <= seed < 2**63 or not isinstance(domain, TaskDomain):
            raise ValueError('sampler requires a nonnegative seed and parsed task domain')
        if type(continuous) is not bool:
            raise ValueError('sampler continuous mode must be explicit bool')
        self.device, self.domain = torch.device(device), domain
        self.stage_weights = tuple(tuple(row) for row in stage_weights)
        self.continuous = continuous
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._families = [_FairQuota(weights) for weights in self.stage_weights]
        self._strata = [[_FairQuota((0.70, 0.15, 0.15)) for _ in TaskFamily] for _ in self.stage_weights]
        self._dynamic_family = _FairQuota(self.stage_weights[0])
        self._dynamic_strata = [_FairQuota((0.70, 0.15, 0.15)) for _ in TaskFamily]

    def _random(self, count):
        return torch.rand(count, device=self.device, generator=self.generator)

    def _integer(self, maximum, count):
        return torch.randint(maximum, (count,), device=self.device, generator=self.generator)

    def _scalar(self, count, stratum, bounds, anchor):
        low, high = bounds
        span = high - low
        order = torch.randperm(count, device=self.device, generator=self.generator)
        interior = low + span * (order + self._random(count)) / count
        upper = self._integer(2, count).bool()
        inward = torch.where(self._random(count) < 0.5, 0, self._random(count) * span * 0.05)
        boundary = torch.where(upper, high - inward, low + inward)
        near = (anchor + (self._random(count) * 2 - 1) * span * 0.05).clamp(low, high)
        near = torch.where(self._random(count) < 1 / 3, anchor, near)
        return torch.where(stratum == 1, boundary, torch.where(stratum == 2, near, interior))

    def sample(self, count, stage, *, weights=None, domain=None):
        if type(count) is not int or count < 1 or type(stage) is not int or not 0 <= stage < len(self.stage_weights):
            raise ValueError('sampling requires positive row count and a valid curriculum stage')
        if domain is not None and not isinstance(domain, TaskDomain):
            raise ValueError('sampling domain must be a parsed TaskDomain')
        d = self.domain if domain is None else domain
        if weights is None:
            family_quota = self._families[stage]
            stratum_quotas = self._strata[stage]
        else:
            if not self.continuous:
                raise ValueError('dynamic weights require a continuous sampler')
            weights = tuple(weights)
            if (len(weights) != len(TaskFamily) or abs(sum(weights) - 1) > 1e-9
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in weights)):
                raise ValueError('dynamic curriculum weights must be finite, nonnegative and sum to one')
            self._dynamic_family.set_weights(weights)
            family_quota = self._dynamic_family
            stratum_quotas = self._dynamic_strata
        allocation = family_quota.allocate(count)
        blocks = []
        for family, size in enumerate(allocation):
            if not size:
                continue
            strata = stratum_quotas[family].allocate(size)
            stratum = torch.cat([torch.full((n,), index, dtype=torch.long, device=self.device)
                                  for index, n in enumerate(strata)])
            stratum = stratum[torch.randperm(size, device=self.device, generator=self.generator)]
            forward = self._scalar(size, stratum, d.forward_m_s, 0.0)
            normal_yaw_limit = min(ROTATION_NORMAL_YAW_RAD_S,
                                   -d.yaw_rad_s[0], d.yaw_rad_s[1])
            yaw = self._scalar(size, stratum, (-normal_yaw_limit, normal_yaw_limit), 0.0)
            height = self._scalar(size, stratum, d.height_m, d.nominal_height_m)
            rotation_mode = torch.zeros(size, dtype=torch.long, device=self.device)
            trajectory_kind = self._integer(7, size)
            nominal = self._random(size) < d.nominal_fraction
            if family in (TaskFamily.HEIGHT, TaskFamily.RECOVERY):
                forward.zero_()
                yaw.zero_()
            elif family == TaskFamily.MOTION:
                yaw.zero_()
            elif family == TaskFamily.ROTATION:
                # Normal rotation stays inside the wheel-envelope-friendly
                # +/-2pi bucket. High-rate spin and high-speed dash are separate
                # modes introduced later by the sampled domain and use immediate
                # hold commands, matching the reference implementation's split.
                draw = self._random(size)
                rotation_mode.copy_(torch.where(
                    draw < .70, torch.full_like(rotation_mode, int(RotationMode.NORMAL)),
                    torch.where(draw < .85, torch.full_like(rotation_mode, int(RotationMode.SPIN_LOW)),
                                torch.where(draw < .95, torch.full_like(rotation_mode, int(RotationMode.SPIN_MID)),
                                            torch.full_like(rotation_mode, int(RotationMode.DASH))))))
                # Do not introduce a bucket before its command range exists.
                max_yaw = min(-d.yaw_rad_s[0], d.yaw_rad_s[1])
                max_forward = min(-d.forward_m_s[0], d.forward_m_s[1])
                rotation_mode.masked_fill_(
                    ((rotation_mode == int(RotationMode.SPIN_LOW))
                       & (max_yaw <= ROTATION_SPIN_LOW_RAD_S[0]))
                    | ((rotation_mode == int(RotationMode.SPIN_MID))
                       & (max_yaw <= ROTATION_SPIN_MID_RAD_S[0]))
                    | ((rotation_mode == int(RotationMode.DASH))
                       & (max_forward < ROTATION_DASH_FORWARD_M_S[0])),
                    int(RotationMode.NORMAL))
                normal = rotation_mode == int(RotationMode.NORMAL)
                sign = torch.where(self._random(size) < .5, -1., 1.)
                low = torch.full_like(yaw, ROTATION_SPIN_LOW_RAD_S[0])
                high = torch.full_like(yaw, min(ROTATION_SPIN_LOW_RAD_S[1], max_yaw))
                spin = sign * (low + (high - low) * self._random(size))
                low.fill_(ROTATION_SPIN_MID_RAD_S[0])
                high.fill_(min(ROTATION_SPIN_MID_RAD_S[1], max_yaw))
                mid = sign * (low + (high - low) * self._random(size))
                yaw = torch.where(normal, yaw, torch.where(
                    rotation_mode == int(RotationMode.SPIN_LOW), spin,
                    torch.where(rotation_mode == int(RotationMode.SPIN_MID), mid, yaw)))
                dash_limit = max(ROTATION_DASH_FORWARD_M_S[0],
                                 min(ROTATION_DASH_FORWARD_M_S[1], max_forward))
                forward = torch.where(
                    rotation_mode == int(RotationMode.DASH),
                    sign * (ROTATION_DASH_FORWARD_M_S[0]
                            + (dash_limit - ROTATION_DASH_FORWARD_M_S[0])
                            * self._random(size)),
                    torch.where(normal, torch.where(self._random(size) < .35,
                                                    torch.zeros_like(forward), forward),
                                torch.zeros_like(forward)))
                # Hold preserves the high-rate request across the episode.
                # It does not claim an instantaneous physical response.
                trajectory_kind = torch.where(normal, trajectory_kind,
                                              torch.zeros_like(trajectory_kind))
                nominal &= normal
            command = torch.stack((forward, torch.zeros_like(forward), yaw, height), -1)
            if family == TaskFamily.TERRAIN:
                terrain = self._integer(len(TerrainKind) - 1, size) + 1
                if self.continuous:
                    # Keep a flat-terrain rehearsal component in the terrain
                    # family instead of making flat and non-flat mutually
                    # exclusive curriculum phases.
                    terrain = torch.where(self._random(size) < 0.20,
                                          torch.zeros_like(terrain), terrain)
            else:
                terrain = torch.zeros_like(stratum)
            block = TaskDraw(
                family=torch.full_like(stratum, family), stratum=stratum, requested_command=command,
                jump_height_m=(self._scalar(size, stratum, d.jump_m, d.jump_m[0])
                               if family == TaskFamily.REQUEST else torch.zeros_like(height)),
                slope_rad=self._scalar(size, stratum, d.slope_rad, 0),
                roughness_m=self._scalar(size, stratum, d.roughness_m, 0),
                step_up_m=self._scalar(size, stratum, d.step_up_m, 0),
                step_down_m=self._scalar(size, stratum, d.step_down_m, 0),
                terrain_kind=terrain, recovery_posture=self._integer(16, size),
                trajectory_kind=trajectory_kind, nominal=nominal,
                rotation_mode=rotation_mode,
            )
            blocks.append(block)
        order = torch.randperm(count, device=self.device, generator=self.generator)
        return TaskDraw(*(torch.cat([getattr(block, field.name) for block in blocks], 0)[order]
                          for field in fields(TaskDraw)))

    def state_dict(self):
        result = {'domain': asdict(self.domain), 'rng': self.generator.get_state().clone(),
                'family_credit': [list(quota.credit) for quota in self._families],
                'stratum_credit': [[list(quota.credit) for quota in stage] for stage in self._strata]}
        if self.stage_weights != STAGE_WEIGHTS:
            result['stage_weights'] = self.stage_weights
        if self.continuous:
            result.update({
                'continuous': True,
                'dynamic_weights': list(self._dynamic_family.weights),
                'dynamic_family_credit': list(self._dynamic_family.credit),
                'dynamic_stratum_credit': [list(quota.credit) for quota in self._dynamic_strata],
            })
        return result

    def load_state_dict(self, state):
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or state['domain'] != expected['domain']
                or (self.continuous and state['continuous'] is not True)):
            raise ValueError('sampler checkpoint domain or fields differ')
        prepared_families = deepcopy(self._families)
        prepared_strata = deepcopy(self._strata)
        self._restore_credits(prepared_families, state['family_credit'])
        if state.get('stage_weights', STAGE_WEIGHTS) != self.stage_weights:
            raise ValueError('sampler checkpoint curriculum differs')
        if not isinstance(state['stratum_credit'], list) or len(state['stratum_credit']) != len(self.stage_weights):
            raise ValueError('sampler checkpoint has invalid stratum stages')
        for stage, credit in zip(prepared_strata, state['stratum_credit']):
            self._restore_credits(stage, credit)
        dynamic_family = deepcopy(self._dynamic_family)
        dynamic_strata = deepcopy(self._dynamic_strata)
        if self.continuous:
            dynamic_family.set_weights(state['dynamic_weights'])
            # Older checkpoints may contain a fractional deficit for a family
            # that is now disabled. Retire and redistribute it before applying
            # the checkpoint; never accept an inactive credit as live state.
            raw = state['dynamic_family_credit']
            if (not isinstance(raw, list) or len(raw) != len(dynamic_family.credit)
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           or abs(value) > 2 for value in raw)
                    or abs(sum(raw)) > 1e-6):
                raise ValueError('sampler checkpoint dynamic credit differs')
            dynamic_family.credit = list(raw)
            if any(weight == 0 and value != 0
                   for weight, value in zip(dynamic_family.weights, raw)):
                dynamic_family.set_weights(state['dynamic_weights'])
            self._restore_credits([dynamic_family], [dynamic_family.credit])
            if (not isinstance(state['dynamic_stratum_credit'], list)
                    or len(state['dynamic_stratum_credit']) != len(dynamic_strata)):
                raise ValueError('sampler checkpoint has invalid dynamic stratum stages')
            for quota, credit in zip(dynamic_strata, state['dynamic_stratum_credit']):
                self._restore_credits([quota], [credit])
        generator = torch.Generator(device=self.device)
        generator.set_state(state['rng'].detach().cpu().clone())
        self.generator, self._families, self._strata = generator, prepared_families, prepared_strata
        self._dynamic_family, self._dynamic_strata = dynamic_family, dynamic_strata

    @staticmethod
    def _restore_credits(quotas, values):
        if not isinstance(values, list) or len(values) != len(quotas):
            raise ValueError('sampler checkpoint quota dimensions differ')
        for quota, credit in zip(quotas, values):
            if (not isinstance(credit, list) or len(credit) != len(quota.credit)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 2 for v in credit)
                    or abs(sum(credit)) > 1e-6
                    or any(weight == 0 and value != 0 for weight, value in zip(quota.weights, credit))):
                raise ValueError('sampler checkpoint quota credit is invalid')
            quota.credit = list(credit)


@dataclass(frozen=True)
class AuditedStageDecision:
    """Input from an independent validator; this type grants no physical qualification."""

    stage: str
    update: int
    checkpoint_sha256: str
    validation_source_sha256: str
    validation_seeds: tuple[int, int]
    passed: bool
    coverage_passed: bool
    all_introduced_cases_present: bool
    heldout_used: bool = False


class CurriculumBudget:
    """Continuous 30000-update budget, preserving consumption after failed PPO.

    Exposure scheduling is the training default; manual evaluation never
    blocks it or grants qualification. Audited promotion remains an explicit
    separate mode for callers retaining the earlier research contract.
    """

    def __init__(self, *, promotion='exposure', stage_names=None, stage_budgets=None):
        if promotion not in ('exposure', 'audited'):
            raise ValueError('curriculum promotion must be exposure or explicitly audited')
        self.promotion = promotion
        # Resolve at call time so tests and explicit schedules do not get a
        # stale default captured at import time.
        self.stage_names = tuple(STAGE_NAMES if stage_names is None else stage_names)
        self.stage_budgets = tuple(STAGE_BUDGETS if stage_budgets is None else stage_budgets)
        if (not self.stage_names or len(self.stage_names) != len(self.stage_budgets)
                or len(set(self.stage_names)) != len(self.stage_names)
                or any(type(value) is not int or value <= 0 for value in self.stage_budgets)):
            raise ValueError('curriculum requires named positive stage budgets')
        self.stage = 0
        self.consumed = [0] * len(self.stage_names)
        self.transferred_to_consolidation = 0
        self.decisions = []
        self.consecutive_passes = 0
        self.status = 'running'

    @property
    def total_updates(self):
        return sum(self.consumed)

    @property
    def stage_limit(self):
        return self.stage_budgets[self.stage] + (self.transferred_to_consolidation if self.stage == len(self.stage_names) - 1 else 0)

    def charge_rollout(self):
        self.prepare_rollout()
        if self.status != 'running' or self.consumed[self.stage] >= self.stage_limit:
            raise RuntimeError('curriculum cannot spend beyond a stopped or exhausted stage')
        self.consumed[self.stage] += 1
        if self.stage == len(self.stage_names) - 1 and self.total_updates == sum(self.stage_budgets):
            self.status = ('training_budget_complete' if self.promotion == 'exposure'
                           else 'training_budget_complete_evaluation_pending')
        elif self.promotion == 'audited' and self.consumed[self.stage] == self.stage_limit:
            self.status = 'stage_budget_complete_validation_pending'

    def prepare_rollout(self):
        """Advance only between rollouts,never during the last old-stage sample."""
        if (self.promotion == 'exposure' and self.status == 'running' and self.stage < len(self.stage_names) - 1
                and self.consumed[self.stage] == self.stage_limit):
            self.stage += 1
        return self.stage

    def submit_stage_validation(self, decision: AuditedStageDecision):
        if self.promotion != 'audited':
            raise RuntimeError('manual evaluation does not change the exposure training schedule')
        if self.stage == len(self.stage_names) - 1 or self.status not in ('running', 'stage_budget_complete_validation_pending'):
            raise RuntimeError('stage promotion is unavailable in the current curriculum state')
        if not isinstance(decision, AuditedStageDecision):
            raise TypeError('stage decisions require the independent audited-decision contract')
        if (decision.stage != self.stage_names[self.stage] or type(decision.update) is not int
                or decision.update != self.total_updates
                or any(type(getattr(decision, name)) is not bool for name in
                       ('passed', 'coverage_passed', 'all_introduced_cases_present', 'heldout_used'))
                or decision.heldout_used or tuple(decision.validation_seeds) != (44101, 44102)
                or any(not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None
                       for sha in (decision.checkpoint_sha256, decision.validation_source_sha256))):
            raise ValueError('stage decision has the wrong stage, checkpoint, seeds or source identity')
        if self.consumed[self.stage] < math.ceil(self.stage_budgets[self.stage] * 0.25):
            raise ValueError('promotion validation requires at least 25 percent of stage exposure')
        previous = [item for item in self.decisions if item.stage == decision.stage]
        if len(previous) >= 4 or any(item.update == decision.update or item.checkpoint_sha256 == decision.checkpoint_sha256
                                     for item in previous):
            raise ValueError('stage candidate limit or distinct-checkpoint requirement violated')
        self.decisions.append(decision)
        passed = decision.passed and decision.coverage_passed and decision.all_introduced_cases_present
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if self.consecutive_passes >= 2:
            self.transferred_to_consolidation += self.stage_budgets[self.stage] - self.consumed[self.stage]
            self.stage += 1
            self.consecutive_passes = 0
            self.status = 'running'
        elif len(previous) + 1 == 4 or self.status == 'stage_budget_complete_validation_pending':
            self.status = 'failed_qualification'

    def state_dict(self):
        result = {'promotion': self.promotion, 'stage': self.stage, 'consumed': list(self.consumed),
                'transferred_to_consolidation': self.transferred_to_consolidation,
                'decisions': [asdict(item) for item in self.decisions],
                'consecutive_passes': self.consecutive_passes, 'status': self.status}
        if self.stage_names != STAGE_NAMES or self.stage_budgets != STAGE_BUDGETS:
            result['schedule'] = {'names': self.stage_names, 'budgets': self.stage_budgets}
        return result

    def load_state_dict(self, state):
        """Reconstruct the budget by replaying consumption and accepted decisions."""
        if (not isinstance(state, dict) or state.keys() != self.state_dict().keys()
                or state['promotion'] != self.promotion
                or state.get('schedule') != self.state_dict().get('schedule')
                or not isinstance(state['consumed'], list) or len(state['consumed']) != len(self.stage_names)
                or any(type(value) is not int or value < 0 for value in state['consumed'])
                or sum(state['consumed']) > sum(self.stage_budgets)
                or not isinstance(state['decisions'], list) or len(state['decisions']) > 20):
            raise ValueError('invalid curriculum checkpoint fields or consumption')
        restored = CurriculumBudget(promotion=self.promotion, stage_names=self.stage_names, stage_budgets=self.stage_budgets)
        try:
            for item in state['decisions']:
                decision = AuditedStageDecision(**item)
                if type(decision.update) is not int or decision.update <= restored.total_updates:
                    raise ValueError('curriculum decisions must advance consumption monotonically')
                while restored.total_updates < decision.update:
                    restored.charge_rollout()
                restored.submit_stage_validation(decision)
            while restored.total_updates < sum(state['consumed']):
                restored.charge_rollout()
            # An entry checkpoint can be published after the stage boundary
            # advances and before that stage spends its first rollout.
            if state['stage'] != restored.stage:
                restored.prepare_rollout()
        except (TypeError, RuntimeError) as error:
            raise ValueError('curriculum checkpoint cannot be replayed') from error
        if restored.state_dict() != state:
            raise ValueError('curriculum checkpoint is inconsistent with its decision history')
        self.__dict__.update(restored.__dict__)
