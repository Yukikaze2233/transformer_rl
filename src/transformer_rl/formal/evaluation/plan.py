"""Private evaluation requests and costs, separate from physical qualification.

The catalogue retains every original request. A scene/reset/protocol resolver
must bind each case before execution; its presence here is not a runnable asset
recipe. Held-out requests require a frozen selection identity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re


FAMILIES = ('stand_height', 'motion', 'rotation_combinations', 'terrain', 'recovery', 'requests')
VALIDATION_SEEDS = (44101, 44102)
HELDOUT_SEEDS = (44901, 44902)
TRAINING_SEEDS = (2101, 2102, 2103)
INTRODUCED_AT = dict(zip(FAMILIES, (0, 1, 2, 3, 0, 4)))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    family: str
    duration_s: int
    parameters_json: str

    def __post_init__(self):
        if (not isinstance(self.name, str) or re.fullmatch('[a-z0-9_]+', self.name) is None
                or self.family not in FAMILIES or type(self.duration_s) is not int
                or self.duration_s != (60 if self.family in ('recovery', 'requests') else 30)):
            raise ValueError('evaluation case identity/family/full horizon differs')
        value = json.loads(self.parameters_json)
        if not isinstance(value, dict) or not value:
            raise ValueError('evaluation case requires explicit task parameters')
        object.__setattr__(self, 'parameters_json', json.dumps(value, sort_keys=True,
                                                              separators=(',', ':'), allow_nan=False))

    @property
    def parameters(self):
        return json.loads(self.parameters_json)

    def request(self):
        return {'name': self.name, 'family': self.family, 'duration_s': self.duration_s,
                'parameters': self.parameters, 'first_episode_only': True,
                'automatic_reset_does_not_replace_first_outcome': True,
                'geometry_and_runtime_resolution_required': True}


def formal_cases():
    """The complete 144-case design, in its original authored order."""
    cases = []

    def add(name, family, **parameters):
        cases.append(EvaluationCase(name, family, 60 if family in ('recovery', 'requests') else 30,
                                    json.dumps(parameters, allow_nan=False)))

    for i, height in enumerate((.17, .20, .23, .26, .29, .32, .36, .40)):
        for profile in ('nominal', 'payload_com_stress', 'effective_delay_stress'):
            add(f'stand_height_{i:02d}_{profile}', 'stand_height', height_m=height, vx_m_s=0., wz_rad_s=0.,
                profile=profile, stress_probability_is_not_measured=profile != 'nominal')
    for i, forward in enumerate((-5., -3., -1., -.3, .3, 1., 3., 5.)):
        for j, height in enumerate((.20, .26, .32, .38)):
            add(f'motion_{i:02d}_height_{j:02d}', 'motion', vx_m_s=forward, wz_rad_s=0., height_m=height)
    for i, yaw in enumerate((-6 * math.pi, -4 * math.pi, -2 * math.pi, -1., 1., 2 * math.pi,
                              4 * math.pi, 6 * math.pi)):
        for j, forward in enumerate((0., .5, -.5, 1.)):
            add(f'rotation_{i:02d}_motion_{j:02d}', 'rotation_combinations', vx_m_s=forward,
                wz_rad_s=yaw, height_m=.30, command_frame='explicit_task_contract_required')
    terrains = (
        ('slope_up', {'angle_deg': 15.}), ('slope_down', {'angle_deg': -15.}),
        ('cross_slope', {'angle_deg': 15.}), ('rough', {'amplitude_m': .02}),
        ('step_up', {'rise_m': .25}), ('step_down', {'drop_m': .15}),
        ('stairs', {'rise_m': .04, 'tread_m': .30}),
        ('two_platforms', {'absolute_top_heights_m': [.20, .25], 'second_rise_m': .05, 'tread_m': .60}),
    )
    for name, dimensions in terrains:
        for i, forward in enumerate((.5, 1., 2.)):
            add(f'terrain_{name}_speed_{i:02d}', 'terrain', terrain=name, **dimensions,
                vx_m_s=forward, height_m=.30, entry_request='manual_skill_request_when_required')
    postures = ('upright', 'crouched', 'upright_release', 'crouched_release', 'left_side_90', 'right_side_90',
                'pitch_forward_45', 'pitch_backward_45', 'front_down_90', 'back_down_90',
                'front_down_left_forward_right_back_90', 'front_down_left_back_right_forward_90',
                'back_down_left_forward_right_back_90', 'back_down_left_back_right_forward_90',
                'prone_belly_legs_above', 'prone_back_legs_above')
    for pose in postures:
        add('recovery_' + pose, 'recovery', pose=pose, pose_table_requires_selected_asset=True,
            policy_and_scripted_outcomes_separate=True, script_intervals_excluded_from_ppo=True)
    protocols = ('stand_jump', 'walking_jump', 'brake_jump_resume', 'repeat_before_takeoff', 'repeat_airborne',
                 'cancel_before_takeoff', 'cancel_airborne', 'repeat_after_landing')
    for protocol in protocols:
        for i, direction in enumerate((-1, 1)):
            add(f'request_{protocol}_direction_{i:02d}', 'requests', protocol=protocol,
                vx_m_s=0. if protocol == 'stand_jump' else .5 * direction,
                initial_heading_rad=math.pi if direction < 0 else 0., requested_jump_m=.15,
                cancel_before_takeoff_requires_no_jump=protocol == 'cancel_before_takeoff',
                airborne_cancel_requires_safe_landing=protocol == 'cancel_airborne')
    return tuple(cases)


# Two cases per family, sealed before any formal model exists. The probe cannot
# grant promotion and never substitutes its 12 cases for the complete catalogue.
PROBE_CASE_NAMES = (
    'stand_height_01_nominal', 'stand_height_06_effective_delay_stress',
    'motion_02_height_01', 'motion_05_height_02',
    'rotation_02_motion_00', 'rotation_05_motion_01',
    'terrain_slope_up_speed_00', 'terrain_two_platforms_speed_00',
    'recovery_left_side_90', 'recovery_prone_back_legs_above',
    'request_stand_jump_direction_01', 'request_repeat_after_landing_direction_00',
)


@dataclass(frozen=True)
class EvaluationSelection:
    training_seed: int
    checkpoint_sha256: str
    validation_report_sha256: str
    selection_record_sha256: str

    def __post_init__(self):
        if (self.training_seed not in TRAINING_SEEDS or type(self.training_seed) is not int
                or not all(_sha(v) for v in (self.checkpoint_sha256, self.validation_report_sha256,
                                            self.selection_record_sha256))):
            raise ValueError('held-out selection needs a concrete training seed and frozen content identities')


@dataclass(frozen=True)
class EvaluationTrial:
    case_name: str
    family: str
    evaluation_seed: int
    repetition: int
    duration_s: int
    challenge_index: int | None = None


@dataclass(frozen=True)
class EvaluationPlan:
    tier: str
    stage: int
    training_seed: int
    checkpoint_sha256: str
    trials: tuple[EvaluationTrial, ...]
    selection_record_sha256: str | None
    challenge_manifest_sha256: str | None

    @property
    def planned_task_transitions(self):
        return sum(trial.duration_s * 100 for trial in self.trials)

    @property
    def sha256(self):
        return _digest({'tier': self.tier, 'stage': self.stage, 'training_seed': self.training_seed,
                        'checkpoint_sha256': self.checkpoint_sha256,
                        'trials': [vars(trial) for trial in self.trials],
                        'selection_record_sha256': self.selection_record_sha256,
                        'challenge_manifest_sha256': self.challenge_manifest_sha256})


def evaluation_plan(tier, training_seed, checkpoint_sha256, *, stage=5, selection=None,
                    challenge_manifest_sha256=None):
    """Pure plan creation; durable reservation and native recipe binding are external."""
    if (tier not in ('probe', 'stage', 'final_validation', 'heldout', 'challenges')
            or type(stage) is not int or not 0 <= stage <= 5 or type(training_seed) is not int
            or training_seed not in TRAINING_SEEDS or not _sha(checkpoint_sha256)):
        raise ValueError('evaluation plan needs its declared tier/stage/training seed/checkpoint')
    if tier in ('heldout', 'challenges'):
        if (not isinstance(selection, EvaluationSelection) or selection.training_seed != training_seed
                or selection.checkpoint_sha256 != checkpoint_sha256 or stage != 5):
            raise ValueError('held-out use requires the matching final selected model;not a model-selection probe')
    elif selection is not None:
        raise ValueError('validation and selection-sealed evaluation have different provenance')
    if tier in ('final_validation', 'heldout', 'challenges') and stage != 5:
        raise ValueError('final evaluation cannot silently remove not-yet-introduced cases')
    if tier == 'challenges' and not _sha(challenge_manifest_sha256):
        raise ValueError('continuous challenges need the frozen full command/terrain/request manifest')
    if tier != 'challenges' and challenge_manifest_sha256 is not None:
        raise ValueError('fixed catalogue trials cannot be relabeled continuous challenges')
    trials = []
    if tier == 'challenges':
        for seed in (*VALIDATION_SEEDS, *HELDOUT_SEEDS):
            for family in FAMILIES:
                for index in range(20):
                    trials.append(EvaluationTrial(f'continuous_{family}_{index:02d}', family, seed, 0, 120, index))
    else:
        seeds = HELDOUT_SEEDS if tier == 'heldout' else VALIDATION_SEEDS
        count = 2 if tier == 'probe' else 4 if tier == 'stage' else 20
        selected = [c for c in formal_cases() if INTRODUCED_AT[c.family] <= stage
                    and (tier != 'probe' or c.name in PROBE_CASE_NAMES)]
        for case in selected:
            for seed in seeds:
                for repetition in range(count):
                    trials.append(EvaluationTrial(case.name, case.family, seed, repetition, case.duration_s))
    return EvaluationPlan(tier, stage, training_seed, checkpoint_sha256, tuple(trials),
                          None if selection is None else selection.selection_record_sha256,
                          challenge_manifest_sha256)


def evaluation_cost_bound():
    """Full-task sample caps only; setup/physics/I/O/failed attempts are measured separately."""
    checkpoint = '0' * 64
    selection = EvaluationSelection(TRAINING_SEEDS[0], checkpoint, '1' * 64, '2' * 64)
    values = {
        'probes': 100 * evaluation_plan('probe', 2101, checkpoint).planned_task_transitions,
        'stage_candidates': 4 * sum(evaluation_plan('stage', 2101, checkpoint, stage=i).planned_task_transitions
                                   for i in range(6)),
        'final_validation': evaluation_plan('final_validation', 2101, checkpoint).planned_task_transitions,
        'heldout': evaluation_plan('heldout', 2101, checkpoint, selection=selection).planned_task_transitions,
        'challenges': evaluation_plan('challenges', 2101, checkpoint, selection=selection,
                                      challenge_manifest_sha256='3' * 64).planned_task_transitions,
    }
    return {**values, 'total_task_transitions': sum(values.values()), 'policy_dt_s': .01,
            'preparation_and_discarded_vector_rows_counted_separately': True,
            'physical_qualification': False}
