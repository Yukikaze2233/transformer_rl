"""Private partial-row source planning around one published native boundary.

Only requested rows consume draws. Issuing a plan does not prove its native
application: a repeated boundary reuses the exact proposal, and an unapplied
proposal blocks further draws. Different colliders still require native
replacement; no atlas, global simulation reset or physical qualification is
introduced by this cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import torch

from ..tasks.trajectory import CommandTrajectory
from ..simulation.episode import NativeEpisodeLifecycle
from ..tasks.reset_plan import NativeTaskResetPlan
from ..control.actuation import OutputFeedbackController, PreparationHold
from ..tasks.reset_program import PreparedTaskReset, TaskResetPreparer, _tensor_versions
from ..tasks.commands import TaskCommandPlan
from ..tasks.curriculum import STAGE_NAMES, TaskDraw


@dataclass(frozen=True)
class ResetProposal:
    sequence: int
    reset_mask: torch.Tensor
    next_episode_id: torch.Tensor
    requested_at_s: torch.Tensor
    curriculum_stage: int
    requested_draw: TaskDraw
    prepared_rows: PreparedTaskReset | None
    plan: NativeTaskResetPlan | None
    preparation_error: str | None
    _versions: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_versions', _tensor_versions(self))


class ResetSourceNotReady(RuntimeError):
    def __init__(self, proposal):
        super().__init__('reset source proposal is retained but not admitted: ' + str(proposal.preparation_error))
        self.proposal = proposal


class ResetBatchCoordinator:
    def __init__(self, lifecycle, preparer, *, stage_source):
        if (not isinstance(lifecycle, NativeEpisodeLifecycle) or not isinstance(preparer, TaskResetPreparer)
                or not callable(stage_source) or lifecycle.device != preparer.device):
            raise ValueError('reset coordinator requires one native lifecycle, selected preparer and stage source')
        plant, geometry = lifecycle.plant, preparer.geometry
        scene, binding = plant.scene, geometry.kinematics.binding
        if (scene.observation.binding != binding
                or scene.readback.get('selected_asset_manifest_sha256') != geometry.kinematics.manifest_sha256
                or scene.readback.get('selected_model_spec_sha256') != geometry.kinematics.model_sha256
                or not isinstance(plant.controller, OutputFeedbackController)):
            raise ValueError('reset source geometry differs from the actual selected native model')
        self.lifecycle, self.preparer, self.stage_source = lifecycle, preparer, stage_source
        self.num_envs, self.device = lifecycle.num_envs, lifecycle.device
        self._scene, self._controller = scene, plant.controller
        self._pair_domain = self._controller.pair_domain
        self._controller_versions = _tensor_versions(self._controller_tensors())
        self._origins = scene.environment_origins_m.double().detach().clone()
        self._issued, self._sequence = None, 0
        self._native_installer = None

    def bind_native_installer(self, installer):
        self._identity()
        if (self._issued is not None or self._native_installer is not None or not callable(installer)
                or installer.coordinator is not self or installer.lifecycle is not self.lifecycle):
            raise ValueError('native terrain installer must bind its one source before the first proposal')
        self._native_installer = installer

    def _identity(self):
        self.lifecycle._identity()
        if (self.lifecycle.plant.scene is not self._scene or self.lifecycle.plant.controller is not self._controller
                or self._controller.pair_domain is not self._pair_domain
                or self._controller_versions != _tensor_versions(self._controller_tensors())
                or not torch.equal(self._origins, self._scene.environment_origins_m.double())
                or self._issued is not None and self._issued._versions != _tensor_versions(self._issued)):
            raise RuntimeError('reset coordinator owner, controller, origins or issued numeric plan changed')

    def _controller_tensors(self):
        return (self._controller.nominal, self._controller.action_bounds, self._controller.effort_bounds,
                getattr(self._controller, '_difference', None), getattr(self._controller, '_signs', None))

    def _applied(self):
        p = self._issued
        return (p is not None and p.plan is not None
                and torch.equal(self.lifecycle.episode_id, p.next_episode_id)
                and self.lifecycle.course_sha256 == p.plan.course_sha256)

    def _subcontroller(self, count):
        value = OutputFeedbackController(self._controller.nominal.tolist(), count,
                                         pair_domain=self._pair_domain, device=self.device)
        value.action_bounds.copy_(self._controller.action_bounds)
        value.effort_bounds.copy_(self._controller.effort_bounds)
        return value

    @staticmethod
    def _merge(previous, fresh, rows):
        def tensor(old, new):
            value = old.detach().clone()
            value[rows] = new
            return value

        attrs = ('start', 'end', 'duration_s', 'kind', 'frequency_start_hz', 'frequency_end_hz')
        trajectory = CommandTrajectory(*(tensor(getattr(previous.commands.trajectory, name),
                                                 getattr(fresh.commands.trajectory, name)) for name in attrs))
        attrs = ('base_mode', 'task_horizon_s', 'jump_height_m', 'request_interval_s')
        commands = TaskCommandPlan(trajectory, *(tensor(getattr(previous.commands, name),
                                                         getattr(fresh.commands, name)) for name in attrs))
        courses, hashes = list(previous.source_courses), list(previous.course_sha256)
        for row, course, sha in zip(rows.tolist(), fresh.source_courses, fresh.course_sha256):
            courses[row], hashes[row] = course, sha
        values = [tensor(getattr(previous, name), getattr(fresh, name)) for name in
                  ('root_pose_xyzw', 'root_com_velocity', 'joint_position', 'joint_velocity')]
        if ((previous.preparation_issued_action is None) != (fresh.preparation_issued_action is None)
                or (previous.preparation_hold is None) != (fresh.preparation_hold is None)):
            raise ValueError('partial reset cannot silently switch its preparation control representation')
        action = (None if previous.preparation_issued_action is None
                  else tensor(previous.preparation_issued_action, fresh.preparation_issued_action))
        hold = None
        if previous.preparation_hold is not None:
            if previous.preparation_hold.position_tolerance_rad != fresh.preparation_hold.position_tolerance_rad:
                raise ValueError('partial reset cannot change untouched hold comparison semantics')
            hold = PreparationHold(*(tensor(getattr(previous.preparation_hold, f.name),
                                            getattr(fresh.preparation_hold, f.name))
                                     for f in fields(PreparationHold) if f.name != 'position_tolerance_rad'),
                                     previous.preparation_hold.position_tolerance_rad)
        return NativeTaskResetPlan(*values, tuple(hashes), commands, action, tuple(courses), hold)

    @torch.no_grad()
    def __call__(self, reset_mask, next_episode_id, requested_at_s):
        self._identity()
        life = self.lifecycle
        life._mask(reset_mask)
        if (not reset_mask.any() or not isinstance(next_episode_id, torch.Tensor)
                or next_episode_id.shape != (self.num_envs,) or next_episode_id.dtype != torch.long
                or next_episode_id.device != self.device
                or not torch.equal(next_episode_id, life.episode_id + reset_mask.long())
                or not isinstance(requested_at_s, torch.Tensor) or requested_at_s.shape != (self.num_envs,)
                or requested_at_s.dtype != torch.float64 or requested_at_s.device != self.device
                or not torch.isfinite(requested_at_s).all() or requested_at_s.lt(0).any()
                or not torch.equal(requested_at_s, self._scene.observation.timestamp_s())):
            raise ValueError('reset proposal requires the exact next episode identities and published native clock')
        if self._issued is not None:
            same = (torch.equal(reset_mask, self._issued.reset_mask)
                    and torch.equal(next_episode_id, self._issued.next_episode_id)
                    and torch.equal(requested_at_s, self._issued.requested_at_s))
            if same:
                if self._issued.plan is None:
                    raise ResetSourceNotReady(self._issued)
                return self._issued.plan
            if not self._applied():
                raise RuntimeError('previous reset proposal has not been applied; never resample its pending requests')
        elif not reset_mask.all():
            raise ValueError('initial reset proposal requires every native row')
        if (reset_mask & life._last_reset_tick.eq(self._scene.observation.physical_tick)).any():
            raise RuntimeError('native rows already reset at this same physical boundary; do not draw another task')
        rows = reset_mask.nonzero().flatten()
        local_com = self._scene.observation._local_com_m
        if (not isinstance(local_com, torch.Tensor) or local_com.shape != (self.num_envs, 3)
                or not torch.isfinite(local_com).all()):
            raise ValueError('reset proposal requires actual per-row native base COM readback')
        stage = self.stage_source()
        if type(stage) is not int or not 0 <= stage < len(STAGE_NAMES):
            raise ValueError('reset proposal needs an actual declared curriculum stage')
        controller = self._subcontroller(len(rows))
        draw = self.preparer.tasks.sample(len(rows), stage)
        previous = self._issued.plan if self._issued is not None else None
        self._sequence += 1
        args = (self._sequence, reset_mask.detach().clone(), next_episode_id.detach().clone(),
                requested_at_s.detach().clone(), stage, draw)
        # Seal the draw before geometric preparation. Even an interrupted or
        # rejected preparation must not silently draw another task on retry.
        self._issued = ResetProposal(*args, None, None, 'source_preparation_in_progress')
        error, plan, prepared = None, None, None
        try:
            prepared = self.preparer.prepare_draw(draw, self._origins[rows], controller)
            fresh, _ = prepared.encode_plan(self.preparer.geometry, controller,
                                             base_com_body_m=local_com[rows].double())
            plan = fresh if previous is None else self._merge(previous, fresh, rows)
        except Exception as problem:
            error = type(problem).__name__ + ': ' + str(problem)
        self._issued = ResetProposal(*args, prepared, plan, error)
        if error is not None:
            raise ResetSourceNotReady(self._issued)
        if self._native_installer is not None:
            self._native_installer(plan, reset_mask)
        return plan

    def state_dict(self):
        """Source RNG disposition only; never a complete learner commit.

        The outer learning journal, parent stop receipt, optimizer and model
        acceptance remain independent requirements for any cold resume.
        """
        self._identity()
        pending = self._issued is not None and not self._applied()
        return {'source_manifest_sha256': self.preparer.geometry.kinematics.manifest_sha256,
                'source_model_sha256': self.preparer.geometry.kinematics.model_sha256,
                'num_envs': self.num_envs, 'program': self.preparer.state_dict(), 'sequence': self._sequence,
                'sensors': (None if self.lifecycle.plant.sensor_model is None
                            else self.lifecycle.plant.sensor_model.checkpoint_state_dict()),
                'status': 'pending_not_resumable' if pending else 'ready_for_explicit_cold_setup',
                'pending_proposal': self._pending_snapshot() if pending else None,
                'cold_episode_disposition': 'abort active physical episodes;retain consumed requests and sampling RNG'}

    def checkpoint_state_dict(self):
        state = self.state_dict()
        if state['status'] != 'ready_for_explicit_cold_setup':
            raise RuntimeError('an unapplied reset proposal cannot become a resumable learning snapshot')
        return state

    def _pending_snapshot(self):
        p = self._issued
        snapshot = {'sequence': p.sequence, 'reset_mask': p.reset_mask.clone(),
                    'next_episode_id': p.next_episode_id.clone(), 'requested_at_s': p.requested_at_s.clone(),
                    'curriculum_stage': p.curriculum_stage, 'preparation_error': p.preparation_error,
                    'requested_draw': {f.name: getattr(p.requested_draw, f.name).clone() for f in fields(p.requested_draw)},
                    'source_courses': None, 'prepared_geometry': None}
        if p.prepared_rows is not None:
            value = p.prepared_rows
            snapshot['source_courses'] = tuple({'request': asdict(c.request), 'geometry_seed': c.geometry_seed}
                                                for c in value.courses)
            snapshot['prepared_geometry'] = {'tree_position': value.recovery.closure.tree_position.clone(),
                'requested_root_xyzw': value.recovery.requested_root_orientation_xyzw.clone(),
                'placed_root_pose_xyzw': value.placement.root_pose_xyzw.clone(),
                'source_ready': value.source_ready.clone()}
        return snapshot

    def load_after_explicit_cold_setup(self, state):
        self._identity()
        expected = self.state_dict()
        if (not isinstance(state, dict) or state.keys() != expected.keys()
                or any(state[k] != expected[k] for k in ('source_manifest_sha256', 'source_model_sha256',
                                                         'num_envs', 'cold_episode_disposition'))
                or state['status'] != 'ready_for_explicit_cold_setup' or state['pending_proposal'] is not None
                or type(state['sequence']) is not int or state['sequence'] < 0
                or self._issued is not None or self.lifecycle._last is not None
                or not self.lifecycle.episode_id.eq(-1).all()):
            raise ValueError('reset cache restore needs an empty cold lifecycle and a complete nonpending boundary')
        sensors = self.lifecycle.plant.sensor_model
        if (sensors is None) != (state['sensors'] is None):
            raise ValueError('cold source sensor implementation differs from the frozen run')
        self.preparer.load_state_dict(state['program'])
        if sensors is not None:
            sensors.load_after_explicit_cold_setup(state['sensors'])
        self._sequence = state['sequence']
