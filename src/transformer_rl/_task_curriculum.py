"""Private continuous task sampling and nonrefundable curriculum bookkeeping.

Sampling covers requested domains. No sampled request is asserted physically
reachable; asset/actuator admission and independent validation remain external.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from enum import IntEnum
import math
import re

import torch


class TaskFamily(IntEnum):
    HEIGHT = 0
    MOTION = 1
    ROTATION = 2
    TERRAIN = 3
    REQUEST = 4
    RECOVERY = 5


STAGE_NAMES = ('foundation', 'motion', 'rotation', 'terrain', 'requests', 'consolidation')
STAGE_BUDGETS = (3000, 3500, 3500, 4500, 3000, 2500)
STAGE_WEIGHTS = (
    (0.80, 0.00, 0.00, 0.00, 0.00, 0.20),
    (0.20, 0.70, 0.00, 0.00, 0.00, 0.10),
    (0.10, 0.10, 0.70, 0.00, 0.00, 0.10),
    (0.10, 0.10, 0.10, 0.65, 0.00, 0.05),
    (0.10, 0.10, 0.10, 0.05, 0.60, 0.05),
    (0.15, 0.20, 0.20, 0.20, 0.15, 0.10),
)


@dataclass(frozen=True)
class TaskDomain:
    height_m: tuple[float, float] = (0.17, 0.40)
    forward_m_s: tuple[float, float] = (-5.0, 5.0)
    yaw_rad_s: tuple[float, float] = (-6 * math.pi, 6 * math.pi)
    jump_m: tuple[float, float] = (0.03, 0.30)
    slope_rad: tuple[float, float] = (-math.pi / 12, math.pi / 12)
    roughness_m: tuple[float, float] = (0.0, 0.02)
    step_up_m: tuple[float, float] = (0.0, 0.25)
    step_down_m: tuple[float, float] = (0.0, 0.15)
    nominal_height_m: float = 0.305
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


class _FairQuota:
    """Carry fractional deficits across small reset batches without starving a pool."""

    def __init__(self, weights):
        if not weights or any(v < 0 or not math.isfinite(v) for v in weights) or abs(sum(weights) - 1) > 1e-9:
            raise ValueError('quota weights must be nonnegative and sum to one')
        self.weights = tuple(weights)
        self.credit = [0.0] * len(weights)

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
    terrain_kind: torch.Tensor       # flat, slope, rough, up, down, two-platform stairs, material
    recovery_posture: torch.Tensor   # references the frozen 16-case posture catalogue
    trajectory_kind: torch.Tensor    # hold, ramp, sine, chirp, smoothstep, start/stop, bounded step
    nominal: torch.Tensor


class ContinuousTaskSampler:
    """Full-width continuous support, endpoint emphasis and an exact-zero mass.

    A dedicated generator is checkpointed with quota deficits. Calls with one
    reset row retain all family/stratum support over time; Latin-hypercube draws
    cover the interior within each family block, then rows are shuffled.
    """

    def __init__(self, seed, device='cpu', domain=TaskDomain()):
        if type(seed) is not int or not 0 <= seed < 2**63 or not isinstance(domain, TaskDomain):
            raise ValueError('sampler requires a nonnegative seed and parsed task domain')
        self.device, self.domain = torch.device(device), domain
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._families = [_FairQuota(weights) for weights in STAGE_WEIGHTS]
        self._strata = [[_FairQuota((0.70, 0.15, 0.15)) for _ in TaskFamily] for _ in STAGE_NAMES]

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

    def sample(self, count, stage):
        if type(count) is not int or count < 1 or type(stage) is not int or not 0 <= stage < len(STAGE_NAMES):
            raise ValueError('sampling requires positive row count and a valid curriculum stage')
        allocation = self._families[stage].allocate(count)
        blocks = []
        d = self.domain
        for family, size in enumerate(allocation):
            if not size:
                continue
            strata = self._strata[stage][family].allocate(size)
            stratum = torch.cat([torch.full((n,), index, dtype=torch.long, device=self.device)
                                  for index, n in enumerate(strata)])
            stratum = stratum[torch.randperm(size, device=self.device, generator=self.generator)]
            forward = self._scalar(size, stratum, d.forward_m_s, 0.0)
            yaw = self._scalar(size, stratum, d.yaw_rad_s, 0.0)
            height = self._scalar(size, stratum, d.height_m, d.nominal_height_m)
            if family in (TaskFamily.HEIGHT, TaskFamily.RECOVERY):
                forward.zero_()
                yaw.zero_()
            elif family == TaskFamily.MOTION:
                yaw.zero_()
            elif family == TaskFamily.ROTATION:
                forward = torch.where(self._random(size) < 0.35, 0, forward)
            command = torch.stack((forward, torch.zeros_like(forward), yaw, height), -1)
            terrain = self._integer(6, size) + 1 if family == TaskFamily.TERRAIN else torch.zeros_like(stratum)
            block = TaskDraw(
                family=torch.full_like(stratum, family), stratum=stratum, requested_command=command,
                jump_height_m=(self._scalar(size, stratum, d.jump_m, d.jump_m[0])
                               if family == TaskFamily.REQUEST else torch.zeros_like(height)),
                slope_rad=self._scalar(size, stratum, d.slope_rad, 0),
                roughness_m=self._scalar(size, stratum, d.roughness_m, 0),
                step_up_m=self._scalar(size, stratum, d.step_up_m, 0),
                step_down_m=self._scalar(size, stratum, d.step_down_m, 0),
                terrain_kind=terrain, recovery_posture=self._integer(16, size),
                trajectory_kind=self._integer(7, size), nominal=self._random(size) < d.nominal_fraction,
            )
            blocks.append(block)
        order = torch.randperm(count, device=self.device, generator=self.generator)
        return TaskDraw(*(torch.cat([getattr(block, field.name) for block in blocks], 0)[order]
                          for field in fields(TaskDraw)))

    def state_dict(self):
        return {'domain': asdict(self.domain), 'rng': self.generator.get_state().clone(),
                'family_credit': [list(quota.credit) for quota in self._families],
                'stratum_credit': [[list(quota.credit) for quota in stage] for stage in self._strata]}

    def load_state_dict(self, state):
        expected = self.state_dict()
        if not isinstance(state, dict) or state.keys() != expected.keys() or state['domain'] != expected['domain']:
            raise ValueError('sampler checkpoint domain or fields differ')
        prepared_families = deepcopy(self._families)
        prepared_strata = deepcopy(self._strata)
        self._restore_credits(prepared_families, state['family_credit'])
        if not isinstance(state['stratum_credit'], list) or len(state['stratum_credit']) != len(STAGE_NAMES):
            raise ValueError('sampler checkpoint has invalid stratum stages')
        for stage, credit in zip(prepared_strata, state['stratum_credit']):
            self._restore_credits(stage, credit)
        generator = torch.Generator(device=self.device)
        generator.set_state(state['rng'].detach().cpu().clone())
        self.generator, self._families, self._strata = generator, prepared_families, prepared_strata

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
    """Continuous 20000-update budget, preserving consumption after failed PPO.

    Only already-audited decisions may be supplied. Evaluating case metrics,
    source identities and feasible-domain coverage is a separate acceptance job.
    """

    def __init__(self):
        self.stage = 0
        self.consumed = [0] * len(STAGE_NAMES)
        self.transferred_to_consolidation = 0
        self.decisions = []
        self.consecutive_passes = 0
        self.status = 'running'

    @property
    def total_updates(self):
        return sum(self.consumed)

    @property
    def stage_limit(self):
        return STAGE_BUDGETS[self.stage] + (self.transferred_to_consolidation if self.stage == 5 else 0)

    def charge_rollout(self):
        if self.status != 'running' or self.consumed[self.stage] >= self.stage_limit:
            raise RuntimeError('curriculum cannot spend beyond a stopped or exhausted stage')
        self.consumed[self.stage] += 1
        if self.stage == 5 and self.total_updates == sum(STAGE_BUDGETS):
            self.status = 'training_budget_complete_evaluation_pending'
        elif self.consumed[self.stage] == self.stage_limit:
            self.status = 'stage_budget_complete_validation_pending'

    def submit_stage_validation(self, decision: AuditedStageDecision):
        if self.stage == 5 or self.status not in ('running', 'stage_budget_complete_validation_pending'):
            raise RuntimeError('stage promotion is unavailable in the current curriculum state')
        if not isinstance(decision, AuditedStageDecision):
            raise TypeError('stage decisions require the independent audited-decision contract')
        if (decision.stage != STAGE_NAMES[self.stage] or type(decision.update) is not int
                or decision.update != self.total_updates
                or any(type(getattr(decision, name)) is not bool for name in
                       ('passed', 'coverage_passed', 'all_introduced_cases_present', 'heldout_used'))
                or decision.heldout_used or tuple(decision.validation_seeds) != (44101, 44102)
                or any(not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None
                       for sha in (decision.checkpoint_sha256, decision.validation_source_sha256))):
            raise ValueError('stage decision has the wrong stage, checkpoint, seeds or source identity')
        if self.consumed[self.stage] < math.ceil(STAGE_BUDGETS[self.stage] * 0.25):
            raise ValueError('promotion validation requires at least 25 percent of stage exposure')
        previous = [item for item in self.decisions if item.stage == decision.stage]
        if len(previous) >= 4 or any(item.update == decision.update or item.checkpoint_sha256 == decision.checkpoint_sha256
                                     for item in previous):
            raise ValueError('stage candidate limit or distinct-checkpoint requirement violated')
        self.decisions.append(decision)
        passed = decision.passed and decision.coverage_passed and decision.all_introduced_cases_present
        self.consecutive_passes = self.consecutive_passes + 1 if passed else 0
        if self.consecutive_passes >= 2:
            self.transferred_to_consolidation += STAGE_BUDGETS[self.stage] - self.consumed[self.stage]
            self.stage += 1
            self.consecutive_passes = 0
            self.status = 'running'
        elif len(previous) + 1 == 4 or self.status == 'stage_budget_complete_validation_pending':
            self.status = 'failed_qualification'

    def state_dict(self):
        return {'stage': self.stage, 'consumed': list(self.consumed),
                'transferred_to_consolidation': self.transferred_to_consolidation,
                'decisions': [asdict(item) for item in self.decisions],
                'consecutive_passes': self.consecutive_passes, 'status': self.status}

    def load_state_dict(self, state):
        """Reconstruct the budget by replaying consumption and accepted decisions."""
        if (not isinstance(state, dict) or state.keys() != self.state_dict().keys()
                or not isinstance(state['consumed'], list) or len(state['consumed']) != len(STAGE_NAMES)
                or any(type(value) is not int or value < 0 for value in state['consumed'])
                or sum(state['consumed']) > sum(STAGE_BUDGETS)
                or not isinstance(state['decisions'], list) or len(state['decisions']) > 20):
            raise ValueError('invalid curriculum checkpoint fields or consumption')
        restored = CurriculumBudget()
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
        except (TypeError, RuntimeError) as error:
            raise ValueError('curriculum checkpoint cannot be replayed') from error
        if restored.state_dict() != state:
            raise ValueError('curriculum checkpoint is inconsistent with its decision history')
        self.__dict__.update(restored.__dict__)
