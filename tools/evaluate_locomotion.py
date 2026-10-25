"""Deterministic native evaluation of one structured checkpoint; never touches its session.

Run with the Lab interpreter. All robot/task imports come from the session's
frozen ``source``; the session's recipe/runtime bind the asset, response and
atlas. The actor's deterministic mean is clipped and decoded as in training;
no optimizer is built and no budget is charged. Suites:

- flat: the eight CP4400-comparable cases (standing at four heights, forward
  and reverse steps, height sequence) plus 2 m/s steps, yaw steps, +/-2 pi
  spins and forward turning; metrics over the 5-35 s window.
- ground: every row is a ground birth (0.2 s zero torque), then a 0.32 m stand.
- step: RMUC road step (0.20 m) and central steps (0.20 + 0.15 m) at 1 m/s,
  requests from the simulated forward ToF and the policy's own estimate.

``--timing nominal`` (default, as the CP4400 report) disables sensor noise and
delay, the target latch, communication and episode randomization; ``trained``
keeps the session's timing and randomization. Only the first physical episode
of each row is scored.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

FLAT_CASES = ('stand_h028', 'stand_h030', 'stand_h032', 'stand_h035', 'forward_reverse_05', 'forward_reverse_10',
              'height_sequence', 'reverse_forward_05', 'forward_reverse_20', 'yaw_sequence_15', 'spin_2pi',
              'forward_turn')
PHASE_S = 5.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, document):
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True, help='Session directory with source/recipe/runtime')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh evaluation directory')
    parser.add_argument('--suite', choices=('flat', 'ground', 'step'), required=True)
    parser.add_argument('--timing', choices=('nominal', 'trained'), default='nominal')
    parser.add_argument('--num-envs', type=int)
    parser.add_argument('--seconds', type=float, default=40.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=44101)
    args = parser.parse_args()
    defaults = {'flat': len(FLAT_CASES), 'ground': 64, 'step': 16}
    args.num_envs = args.num_envs or defaults[args.suite]
    if args.suite == 'flat' and args.num_envs % len(FLAT_CASES):
        parser.error(f'flat suite needs a multiple of {len(FLAT_CASES)} environments')
    if args.suite == 'step' and args.num_envs % 2:
        parser.error('step suite splits rows between the two RMUC courses')
    return args


def flat_command(case, t):
    """[vx, vy, yaw, h] requested for ``case`` at time ``t`` (5 s phases)."""
    phase = int((t + 1e-8) // PHASE_S)
    sign = 1. if phase in (1, 5) else -1. if phase == 3 else 0.
    name = FLAT_CASES[case]
    vx, wz, height = 0., 0., .32
    if name.startswith('stand_h'):
        height = int(name[-3:]) / 100
    elif name in ('forward_reverse_05', 'forward_reverse_10', 'reverse_forward_05', 'forward_reverse_20'):
        amplitude = {'forward_reverse_05': .5, 'forward_reverse_10': 1., 'reverse_forward_05': -.5,
                     'forward_reverse_20': 2.}[name]
        vx = amplitude * sign
    elif name == 'height_sequence':
        sequence = (.32, .28, .30, .32, .35, .32, .28, .32, .32)
        height = sequence[min(phase, len(sequence) - 1)]
    elif name == 'yaw_sequence_15':
        wz = 1.5 * sign
    elif name == 'spin_2pi':
        wz = 2 * math.pi * sign
    elif name == 'forward_turn':
        vx, wz = (1., sign) if sign else (0., 0.)
    return vx, 0., wz, height


def step_windows(trace, row, t, valid, signal, admitted, np):
    """Per 5 s phase: rise times, overshoot and tracking of one command axis."""
    windows = []
    for start in np.arange(PHASE_S, 35. - 1e-6, PHASE_S):
        mask = valid & (t >= start - 1e-8) & (t < start + PHASE_S - 1e-8)
        index = np.flatnonzero(mask)
        if not len(index):
            continue
        target = float(admitted[index[-1]])
        values, times = signal[index], t[index]
        item = {'from_s': float(start), 'admitted_end': target,
                'tracking_rmse': float(np.sqrt(np.mean((values - admitted[index]) ** 2))),
                'settled_error_mean': float(np.mean(values[len(values) // 2:] - admitted[index][len(values) // 2:]))}
        if abs(target) > 1e-3:
            progress = values / target
            reach = [np.flatnonzero(progress >= level) for level in (.1, .9)]
            item['rise_10_90_s'] = (float(times[reach[1][0]] - times[reach[0][0]])
                                    if all(len(r) for r in reach) else None)
            item['overshoot'] = float(max(0., np.max(np.sign(target) * values) - abs(target)))
        windows.append(item)
    return windows


def summarize_flat(trace, end, np):
    t = trace['time_s']
    rows = []
    for row, status in enumerate(end):
        case = row % len(FLAT_CASES)
        valid = trace['alive'][:, row]
        steady = valid & (t > 5. + 1e-8) & (t <= 35. + 1e-8)
        entry = dict(status, row=row, case=FLAT_CASES[case], steady_seconds=float(steady.sum() * .02))
        if steady.any():
            admitted = trace['admitted'][:, row]
            height_error = trace['height_m'][steady, row] - admitted[steady, 3]
            rpy = trace['rpy_rad'][steady, row]
            fork = np.abs(trace['fork_dx_m'][steady, row])
            entry.update(height_bias_m=float(height_error.mean()),
                         height_rmse_m=float(np.sqrt(np.mean(height_error ** 2))),
                         roll_rms_rad=float(np.sqrt(np.mean(rpy[:, 0] ** 2))),
                         pitch_rms_rad=float(np.sqrt(np.mean(rpy[:, 1] ** 2))),
                         fork_abs_mean_m=float(fork.mean()), fork_abs_p95_m=float(np.quantile(fork, .95)))
            name = FLAT_CASES[case]
            if name.startswith('stand_h'):
                position = trace['position_world_m'][steady, row, :2]
                anchor = np.flatnonzero(valid & (t <= 5. + 1e-8))
                start = trace['position_world_m'][anchor[-1], row, :2] if len(anchor) else position[0]
                distance = np.linalg.norm(position - start, axis=1)
                heading = np.unwrap(rpy[:, 2])
                entry['standing'] = {'net_displacement_m': float(distance[-1]),
                                     'maximum_displacement_m': float(distance.max()),
                                     'path_length_m': float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
                                     'yaw_change_rad': float(heading[-1] - heading[0])}
            if name in ('forward_reverse_05', 'forward_reverse_10', 'reverse_forward_05', 'forward_reverse_20',
                        'forward_turn'):
                entry['forward_windows'] = step_windows(trace, row, t, valid, trace['velocity_body_m_s'][:, row, 0],
                                                        admitted[:, 0], np)
            if name in ('yaw_sequence_15', 'spin_2pi', 'forward_turn'):
                entry['yaw_windows'] = step_windows(trace, row, t, valid, trace['yaw_rate_rad_s'][:, row],
                                                    admitted[:, 2], np)
        rows.append(entry)
    return rows


def summarize_ground(trace, end, np):
    """Stand-up time: |h-0.32|<0.05 m, tilt<20 deg, both wheels >5 N held for 0.5 s."""
    t = trace['time_s']
    rows = []
    for row, status in enumerate(end):
        valid = trace['alive'][:, row]
        upright = (valid & (np.abs(trace['height_m'][:, row] - .32) < .05)
                   & (trace['tilt_rad'][:, row] < math.radians(20)) & (trace['wheel_load_n'][:, row] > 5).all(-1))
        held, stood_at = 0., None
        for index in range(len(t)):
            held = held + .02 if upright[index] else 0.
            if held >= .5 - 1e-9:
                stood_at = float(t[index] - .5)
                break
        rows.append(dict(status, row=row, stood_up_at_s=stood_at, stood_within_5s=bool(stood_at is not None
                                                                                      and stood_at <= 5.)))
    return rows


def summarize_step(trace, end, courses, np):
    rows = []
    for row, status in enumerate(end):
        valid = trace['alive'][:, row]
        gain = trace['ground_m'][:, row] - trace['ground_m'][0, row]
        top = .35 if courses[row] == 'central_steps' else .20
        reached = np.flatnonzero(valid & (gain >= top - .02))
        rows.append(dict(status, row=row, course=courses[row], climbed_m=float(gain[valid].max()) if valid.any() else 0.,
                         reached_top=bool(len(reached)),
                         reached_top_at_s=float(trace['time_s'][reached[0]]) if len(reached) else None,
                         stair_modes=sorted({int(v) for v in trace['mode'][valid, row]})))
    return rows


def aggregate(suite, rows):
    if suite == 'ground':
        times = [row['stood_up_at_s'] for row in rows if row['stood_up_at_s'] is not None]
        return {'rows': len(rows), 'stood_within_5s': sum(row['stood_within_5s'] for row in rows),
                'failed': sum(row['kind'] == 'true_failure' for row in rows),
                'median_stand_up_s': sorted(times)[len(times) // 2] if times else None}
    if suite == 'step':
        result = {}
        for course in sorted({row['course'] for row in rows}):
            chosen = [row for row in rows if row['course'] == course]
            result[course] = {'rows': len(chosen), 'reached_top': sum(row['reached_top'] for row in chosen),
                              'failed': sum(row['kind'] == 'true_failure' for row in chosen)}
        return result
    by_case = {}
    for row in rows:
        by_case.setdefault(row['case'], []).append(row)
    summary = {}
    for case, chosen in by_case.items():
        item = {'rows': len(chosen), 'failed': sum(row['kind'] == 'true_failure' for row in chosen)}
        for key in ('height_rmse_m', 'pitch_rms_rad', 'roll_rms_rad', 'fork_abs_mean_m'):
            values = [row[key] for row in chosen if key in row]
            if values:
                item[key] = sum(values) / len(values)
        if any('standing' in row for row in chosen):
            item['net_displacement_m'] = max(row['standing']['net_displacement_m'] for row in chosen if 'standing' in row)
        for axis in ('forward_windows', 'yaw_windows'):
            values = [w['tracking_rmse'] for row in chosen for w in row.get(axis, [])]
            if values:
                item[axis.replace('_windows', '_tracking_rmse')] = sum(values) / len(values)
        summary[case] = item
    return summary


def main():
    args = parse_args()
    session = args.session.resolve()
    recipe = json.loads((session / 'recipe.json').read_text())
    runtime = json.loads((session / 'runtime.json').read_text())
    if any(name == 'transformer_rl' or name.startswith('transformer_rl.') for name in sys.modules):
        raise RuntimeError('a non-frozen transformer_rl module was imported before the source binding')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(session / 'source'))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from transformer_rl.formal.lab_env.sdk import LabSimulation

    torch.manual_seed(args.seed)
    provenance = {'format': 'locomotion_checkpoint_evaluation_v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
                  'session': str(session), 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
                  'script_sha256': sha(__file__), 'suite': args.suite, 'timing': args.timing,
                  'num_envs': args.num_envs, 'seed': args.seed, 'horizon_s': args.seconds,
                  'policy': 'deterministic actor mean, clipped to the action bounds and decoded as in training',
                  'formal_budget_charge': 0, 'scope': 'first physical episode per row'}
    write_json(output / 'provenance.json', provenance)
    launcher = env = None
    trace_rows = {}
    started = time.perf_counter()
    try:
        launcher = LabSimulation(args.device, gui=False)
        launcher.configure_settings()
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.init()
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        from transformer_rl.common.training_control import verify_pin
        from transformer_rl.formal.control.actuation import PairTargetDomain, ResponseParameters
        from transformer_rl.formal.control.observation import CommandMode
        from transformer_rl.formal.control.sensors import SensorModelConfig
        from transformer_rl.formal.lab_env.birth import BirthRandomizationConfig
        from transformer_rl.formal.lab_env.cfg import build_env_cfg
        from transformer_rl.formal.lab_env.contract import FlatTaskSpec
        from transformer_rl.formal.lab_env.env import WheeledLabEnv
        from transformer_rl.formal.lab_env.formal_curriculum import LabTaskSample
        from transformer_rl.formal.lab_env.formal_reward import FormalRewardConfig
        from transformer_rl.formal.lab_env.sections import optional_sections, termination_limits
        from transformer_rl.formal.lab_env.settings import load_asset_description
        from transformer_rl.formal.policy import StructuredPolicy, policy_config_from_dict
        from transformer_rl.formal.simulation.dynamics import StartupDynamicsConfig
        from transformer_rl.formal.tasks.curriculum import TaskDraw, TaskFamily
        from transformer_rl.formal.tasks.geometry import TerrainKind
        from transformer_rl.formal.tasks.protocol import CompletedEstimate

        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        policy_config = policy_config_from_dict(payload['recipe']['policy'])
        policy = StructuredPolicy(policy_config)
        prefix = 'actor.policy.'
        policy.load_state_dict({k[len(prefix):]: v for k, v in payload['model'].items() if k.startswith(prefix)},
                               strict=True)
        policy = policy.to(args.device).eval()
        provenance['checkpoint_update'] = payload.get('update', payload.get('budget_update'))

        n, device = args.num_envs, torch.device(args.device)
        courses = ['road_step' if row % 2 == 0 else 'central_steps' for row in range(n)]

        class EvalSampler:
            stage = 2 if args.suite == 'step' else 0

            def __init__(self):
                self.curriculum = SimpleNamespace(budget=SimpleNamespace(total_updates=35000),
                                                  terrain_level_cap=lambda levels: levels - 1)

            def __call__(self, ids):
                count = len(ids)
                zeros = torch.zeros(count, device=device)
                longs = torch.zeros(count, dtype=torch.long, device=device)
                command = torch.zeros(count, 4, device=device)
                command[:, 3] = .32
                terrain, family, mode, step = longs.clone(), longs.clone(), longs.clone(), zeros.clone()
                if args.suite == 'step':
                    central = (ids % 2 == 1)
                    terrain = torch.where(central, int(TerrainKind.TWO_PLATFORMS), int(TerrainKind.STEP_UP))
                    family.fill_(int(TaskFamily.TERRAIN))
                    mode.fill_(int(CommandMode.STAIR))
                    step.fill_(.2)
                    command[:, 0] = 1.
                draw = TaskDraw(family, longs.clone(), command.clone(), zeros.clone(), zeros.clone(), zeros.clone(),
                                step, zeros.clone(), terrain, longs.clone(), longs.clone(),
                                torch.zeros(count, dtype=torch.bool, device=device), longs.clone())
                return LabTaskSample(draw, command, zeros.clone(), terrain.clone(), mode, longs.clone())

        class EvaluationEnv(WheeledLabEnv):
            def __init__(self, *positional, **keyword):
                self._eval_enabled = False
                super().__init__(*positional, **keyword)
                self._eval_desired = self._command.clone()
                self._eval_last_flags = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

            def _reset_idx(self, env_ids):
                # Keep the failure bits of rows the step is about to reset.
                if self._eval_enabled:
                    ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
                    self._eval_last_flags[ids] = self._failure_latch.failure_reasons[ids]
                super()._reset_idx(env_ids)

            def desired_command(self, now):
                if args.suite == 'flat':
                    case = torch.arange(self.num_envs) % len(FLAT_CASES)
                    return torch.tensor([flat_command(int(c), now) for c in case], device=self._device)
                command = torch.zeros(self.num_envs, 4, device=self._device)
                command[:, 3] = .32
                if args.suite == 'step':
                    command[:, 0] = 1. if now >= 1. else 0.
                return command

            def _update_skill(self):
                if self._eval_enabled:
                    tasks = self._terrain_course.tasks
                    self._eval_desired = self.desired_command(float(self._clock()[0]))
                    tasks.trajectory.start.copy_(self._eval_desired)
                    tasks.trajectory.end.copy_(self._eval_desired)
                    tasks.trajectory.kind.zero_()
                    tasks.heading_enabled.zero_()
                super()._update_skill()

        asset = Path(runtime['asset_directory'])
        description = load_asset_description(asset, expected_manifest_sha256=recipe['asset_manifest_sha256'])
        response = ResponseParameters.from_recorded_response(
            json.loads(verify_pin(runtime['response_source_pin']).read_text())['actuator_response'])
        atlas = verify_pin(runtime['terrain_atlas_pin'])
        pair = None if runtime['pair_domain'] is None else PairTargetDomain(**runtime['pair_domain'])
        sections = optional_sections(runtime)
        sections['command_pool'] = None  # the evaluation owns the commands
        sections['birth_randomization'] = (
            BirthRandomizationConfig(enabled=True, ground_probability=1., roll_pitch_rad=0., wheel_velocity_rad_s=0.)
            if args.suite == 'ground' else None)
        sensors = dynamics = None
        if args.timing == 'trained':
            sensors = SensorModelConfig(**runtime['sensors'])
            dynamics = StartupDynamicsConfig(**runtime['dynamics'])
        else:
            for name in ('communication', 'episode_randomization', 'action_latch', 'wheel_material'):
                sections[name] = None
        cfg = build_env_cfg(str(asset), num_envs=n, env_spacing_m=runtime['environment_spacing_m'],
                            device=args.device, response=response, pair_domain=pair,
                            task=FlatTaskSpec(**runtime['task']),
                            formal_reward=FormalRewardConfig.from_manifest(
                                description.manifest, normal_load_source='native_contact_patch_normal_forces'),
                            termination_limits=termination_limits(runtime['termination_limits']), sensors=sensors, dynamics=dynamics,
                            **sections, terrain_profile='course', terrain_atlas_manifest=str(atlas),
                            terrain_scan=True, forward_scan=True, declarative_scene=launcher.declarative_scene,
                            expected_manifest_sha256=description.manifest_sha256, seed=args.seed)
        cfg.episode_length_s = args.seconds + 5.
        physics = dict(runtime['physics'], gpu_found_lost_pairs_capacity=2**18,
                       gpu_found_lost_aggregate_pairs_capacity=2**20, gpu_total_aggregate_pairs_capacity=2**18,
                       gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**18,
                       gpu_collision_stack_size=2**27)
        cfg.sim.enable_scene_query_support = False
        cfg.sim.physics = PhysxCfg(enable_external_forces_every_iteration=True, enable_scene_query_support=False,
                                  **physics)
        if launcher.declarative_scene:
            from transformer_rl.formal.lab_env.ga_isolation import configure_ga_isolation
            object.__setattr__(cfg, '_native_clone_binding', configure_ga_isolation(
                cfg.sim.physics, cfg.scene.robot, n, args.device))
        env = EvaluationEnv(cfg)
        env.set_task_sampler(EvalSampler())
        if args.suite == 'step':
            router = env._terrain_course.router
            router.level.fill_(router.manifest['levels'] - 1)
            # Route step and platform rows only to the authored RMUC courses.
            procedural = torch.tensor([tile['name'] not in ('road_step', 'central_steps')
                                       for tile in router.manifest['tiles']], device=router.kinds.device)
            stepped = (router.kinds == int(TerrainKind.STEP_UP)) | (router.kinds == int(TerrainKind.TWO_PLATFORMS))
            router.kinds[procedural & stepped] = -1
        env.reset()
        if args.suite == 'step':
            names = [env._terrain_course.router.manifest['tiles'][int(tile)]['name']
                     for tile in env._terrain_course.router.tile.tolist()]
            provenance['step_tiles'] = names
            if names != courses:
                raise RuntimeError(f'step suite routed to {names}, expected the RMUC courses')
        write_json(output / 'provenance.json', provenance)
        env._eval_desired = env._command.clone()
        env._eval_enabled = True
        bounds = torch.tensor(recipe['collection']['action_bounds'], device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        end = [{'kind': 'horizon', 'at_s': args.seconds, 'failure_flags': 0} for _ in range(n)]
        estimate = None
        for step in range(round(args.seconds / .02)):
            frames = env.obs_buf['policy'].reshape(n, 5, 35)
            with torch.inference_mode():
                mean, raw_state, _ = policy.outputs(frames)
            if not torch.isfinite(mean).all():
                raise RuntimeError('policy produced a nonfinite action mean')
            env._completed_estimate = estimate
            _, _, terminated, truncated, _ = env.step(mean.clamp(-bounds, bounds))
            done = terminated | truncated
            for row in (alive & done).nonzero().flatten().tolist():
                end[row] = {'kind': 'true_failure' if bool(terminated[row]) else 'truncated',
                            'at_s': float(env._clock()[row]), 'failure_flags': int(env._eval_last_flags[row])}
            alive &= ~done
            estimate = CompletedEstimate._from_actor_observation(
                env.obs_buf['policy'].reshape(n, 5, 35)[:, -1].float(), raw_state.float(), env._clock().clone())
            state, sample = env._last_state, env._last_sample
            q = sample.root_link_orientation_xyzw
            x, y, z, w = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
            rpy = torch.stack((torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                               torch.asin((2 * (w * y - z * x)).clamp(-1, 1)),
                               torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))), -1)
            wheels = env._wheel_position_base()
            gravity = state.gravity_body
            values = {'time_s': env._clock()[0:1].float(), 'alive': alive, 'admitted': env._command,
                      'position_world_m': sample.root_link_position_world_m, 'rpy_rad': rpy,
                      'velocity_body_m_s': state.velocity_origin_body_m_s,
                      'yaw_rate_rad_s': state.angular_velocity_body_rad_s[:, 2],
                      'height_m': state.height_above_ground_m, 'ground_m': env._ground_at_root,
                      'tilt_rad': torch.atan2(gravity[:, :2].norm(dim=-1), -gravity[:, 2]),
                      'wheel_load_n': state.wheel_ground_force_n,
                      'fork_dx_m': wheels[:, 0, 0] - wheels[:, 1, 0], 'mode': env._command_mode}
            for name, value in values.items():
                trace_rows.setdefault(name, []).append(value.detach().cpu().numpy().copy())
            if step % 250 == 249:
                print(json.dumps({'evaluation_s': (step + 1) * .02, 'alive': int(alive.sum()), 'num_envs': n}),
                      flush=True)
            if not bool(alive.any()):
                break
        trace = {name: np.stack(values) for name, values in trace_rows.items()}
        trace['time_s'] = trace['time_s'][:, 0]
        np.savez_compressed(output / 'trace.npz', **trace)
        if args.suite == 'flat':
            rows = summarize_flat(trace, end, np)
        elif args.suite == 'ground':
            rows = summarize_ground(trace, end, np)
        else:
            rows = summarize_step(trace, end, courses, np)
        report = {'format': 'locomotion_checkpoint_report_v1', 'suite': args.suite, 'timing': args.timing,
                  'checkpoint_update': provenance['checkpoint_update'], 'elapsed_wall_s': time.perf_counter() - started,
                  'summary': aggregate(args.suite, rows), 'rows': rows, 'trace_sha256': sha(output / 'trace.npz')}
        write_json(output / 'report.json', report)
        print(json.dumps({'report': str(output / 'report.json'), 'summary': report['summary']}), flush=True)
    except BaseException as error:
        write_json(output / 'failure.json', {'error': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc(), 'formal_budget_charge': 0})
        if launcher is not None:
            launcher.defer_fatal_shutdown(env)
        raise
    else:
        env.close()
        launcher.close()


if __name__ == '__main__':
    main()
