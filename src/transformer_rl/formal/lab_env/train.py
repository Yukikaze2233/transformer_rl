"""LabEnv 的直接采样/PPO诊断入口。

PPO/auxiliary 数学复用 StructuredPPOTrainer。这里的 lab_ppo_v1 端点不具备
完整正式 journal/control 合同；正式长训使用 lab_env.prepare 和 trainctl。
阶段覆盖产生的验证偏移不能被解释为学习消费或用于正式恢复。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def _cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _invocation_scope(args):
    diagnostic = getattr(args, 'diagnostic_only', False)
    stage = getattr(args, 'diagnostic_stage', None)
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError('resume and warm start are distinct; choose only one checkpoint source')
    if type(diagnostic) is not bool or (stage is not None and type(stage) is not int):
        raise ValueError('diagnostic scope must be an explicit flag and integer stage')
    if diagnostic != (stage is not None):
        raise ValueError('diagnostic-only and diagnostic-stage must be supplied together')
    if diagnostic and (not args.formal_curriculum or args.resume is not None
                       or not 0 <= stage < (4 if args.include_jump else 3)):
        raise ValueError('diagnostic stage requires a fresh formal curriculum and an available stage')
    return {'validation_only': diagnostic, 'diagnostic_stage': stage,
            'budget_scope': 'diagnostic_stage_offset' if diagnostic else 'lab_learning_updates',
            'initialization': ('resume' if args.resume is not None else
                               'warm_start' if args.init_checkpoint is not None else 'fresh')}


def _require_resume_scope(payload):
    if not isinstance(payload, dict) or payload.get('format') != 'lab_ppo_v1':
        raise ValueError('Lab resume requires lab_ppo_v1; managed learning resumes through trainctl')
    if (payload.get('validation_only', False) is not False or payload.get('diagnostic_stage') is not None
            or payload.get('budget_scope') == 'diagnostic_stage_offset'):
        raise ValueError('validation-only stage checkpoints cannot become learning resume points')
    update = payload.get('update')
    if type(update) is not int or update < 0:
        raise ValueError('Lab resume requires its nonnegative completed update count')
    if payload.get('formal_curriculum'):
        curriculum = payload.get('curriculum')
        budget = curriculum.get('budget') if isinstance(curriculum, dict) else None
        consumed = budget.get('consumed') if isinstance(budget, dict) else None
        if (not isinstance(consumed, (list, tuple)) or len(consumed) not in (3, 4)
                or any(type(value) is not int or value < 0 for value in consumed)
                or sum(consumed) != update):
            raise ValueError('Lab curriculum consumption differs from actual completed PPO updates')


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="M1 LabEnv PPO")
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--env-spacing", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", type=int, default=None, choices=range(4),
                        help="M1 command stage 0-3；正式 3-phase 课程 fresh run 只能从 0 开始")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--asset-directory", type=str)
    parser.add_argument("--formal-reward", action="store_true",
                        help="使用正式 PhysicalRewardComposer")
    parser.add_argument("--formal-curriculum", action="store_true",
                        help="使用正式 TaskDraw 课程；默认包含 jump 巩固段")
    parser.add_argument("--include-jump", action="store_true", default=True,
                        help="正式课程包含第 4 段 jump（默认开启）")
    parser.add_argument("--no-jump", action="store_false", dest="include_jump",
                        help="只跑 30k 三段主课程")
    parser.add_argument("--recover", action="store_true",
                        help="使用随机翻倒 recover task：随机翻倒 reset + action hold + 5s episode")
    parser.add_argument("--recover-hold-s", type=float, default=.5)
    parser.add_argument("--recover-height", type=float, default=.25)
    parser.add_argument("--height-scanner", action="store_true",
                        help="启用 base_link 地面 raycast；terrain/正式任务建议开启")
    parser.add_argument("--terrain-scan", action="store_true",
                        help="启用 11x7 地形扫描；185D critic 自动开启")
    parser.add_argument("--forward-scan", action="store_true",
                        help="启用左右轮轮下/轮前 raycast 和 StairFSM")
    parser.add_argument("--terrain-profile", choices=("plane", "rough", "course"), default="plane")
    parser.add_argument('--terrain-atlas', type=Path, help='Prepared terrain atlas manifest; never a raw source USD')
    parser.add_argument('--profile-collection', action='store_true',
                        help='Diagnostic GPU fences and per-stage collection logs; disabled by default')
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    scope = _invocation_scope(args)
    if args.terrain_profile == 'course' and (args.terrain_atlas is None or not args.formal_curriculum):
        raise ValueError('course requires --terrain-atlas and --formal-curriculum')
    if args.terrain_atlas is not None and args.terrain_profile != 'course':
        raise ValueError('--terrain-atlas requires --terrain-profile course')
    if args.updates < 1 or args.num_envs < 1 or args.rollout_steps < 1 or args.checkpoint_interval < 1:
        raise ValueError("updates/num_envs/rollout_steps/checkpoint_interval 必须为正")
    if "cuda" in args.device and "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("WARN: LD_LIBRARY_PATH 不含 /usr/lib/wsl/lib，PhysX 可能回退 CPU", flush=True)
    if args.formal_curriculum:
        # 正式课程必须使用正式 reward，不能退回 M1 flat_reward。
        args.formal_reward = True
    if args.recover:
        args.formal_reward = True
    if args.formal_curriculum and args.recover:
        raise ValueError("formal-curriculum 与 recover 是不同任务，不能同时启用")
    if args.recover and args.stage not in (None, 0):
        raise ValueError("recover task 不使用 command curriculum stage")
    if not args.formal_curriculum and args.stage is not None and args.stage > 3:
        raise ValueError("M1 command 课程只有 stage 0-3；完整课程请用 --formal-curriculum")
    if args.formal_curriculum and args.stage not in (None, 0) and args.resume is None:
        raise ValueError("正式课程 fresh run 必须从 stage 0 开始；跨 stage 恢复请用 --resume")

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device=args.device, enable_cameras=False, livestream=0)
    health = None
    try:
        import torch
        import omni.kit.app
        import omni.physx
        from isaaclab.app.settings_manager import SettingsManager
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        from ..control.actuation import PairTargetDomain, ResponseParameters
        from ...common.training_control import verify_pin
        from ...common.random_state import capture_rng, restore_rng
        from ..learning.optimization import StructuredPPOConfig, StructuredPPOTrainer
        from ..policy import StructuredActorCritic, StructuredPolicyConfig
        from ..simulation.health import NativePhysicsHealth
        from .adapter import LabLabeledEnv
        from .cfg import build_env_cfg
        from .collector import LabCollector
        from .curriculum import LabCommandCurriculum
        from .env import WheeledLabEnv
        from .formal_curriculum import (
            FormalCurriculumCapabilities, FormalTaskCurriculum, LabTaskDrawSampler,
        )
        from .formal_reward import FormalRewardConfig
        from .recover import RecoverTaskSpec
        from .settings import load_asset_description

        settings = SettingsManager.instance()
        settings.initialize_carb_settings()
        if "cuda" in args.device:
            settings.set_int("/physics/cudaDevice", int(args.device.split(":")[-1]))
            settings.set_bool("/physics/suppressReadback", True)
            extensions = omni.kit.app.get_app().get_extension_manager()
            if not extensions.is_extension_enabled("omni.physx.gpu"):
                extensions.set_extension_enabled_immediate("omni.physx.gpu", True)
            torch.cuda.set_device(int(args.device.split(":")[-1]))
            torch.cuda.init()

        torch.manual_seed(args.seed)
        runtime = json.loads(args.runtime_json.read_text())
        response_path = verify_pin(runtime["response_source_pin"])
        response = ResponseParameters.from_recorded_response(
            json.loads(response_path.read_text())["actuator_response"])
        pair_raw = runtime.get("tuning", {}).get("pair_domain")
        pair = PairTargetDomain(**pair_raw) if pair_raw is not None else None
        asset = args.asset_directory or runtime["asset_directory"]

        source = args.resume or args.init_checkpoint
        payload = None
        if source is not None:
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if args.resume is not None:
                _require_resume_scope(payload)
        expected_asset = None
        if payload is not None:
            expected_asset = payload.get("asset_manifest_sha256")
            if expected_asset is None:
                expected_asset = payload.get("recipe", {}).get("asset_manifest_sha256")
            if (args.resume is not None and payload.get("response_source_sha256") is not None
                    and payload["response_source_sha256"] != runtime["response_source_pin"]["sha256"]):
                raise ValueError("resume 的电机响应 bag pin 与当前 runtime 不一致")
        if payload is None:
            policy = StructuredPolicyConfig()
            ppo = StructuredPPOConfig()
            update_index = 0
        elif "policy" in payload and "ppo" in payload:
            policy = StructuredPolicyConfig(**payload["policy"])
            ppo = StructuredPPOConfig(**payload["ppo"])
            update_index = int(payload.get("update", 0)) if args.resume else 0
        else:
            recipe = payload.get("recipe", {})
            policy = StructuredPolicyConfig(**recipe["policy"])
            ppo = StructuredPPOConfig(**recipe["ppo"])
            update_index = 0
        model = StructuredActorCritic(policy).to(args.device)
        if payload is not None:
            model.load_state_dict(payload["model"], strict=True)
        trainer = StructuredPPOTrainer(model, ppo)
        trainer.update_index = update_index
        stage = args.stage
        if stage is None and args.resume is not None:
            stage = int(payload.get("stage", 0))
        if stage is None:
            stage = 0
        if args.resume is not None:
            trainer.actor_optimizer.load_state_dict(payload["actor_adam"])
            trainer.critic_optimizer.load_state_dict(payload["critic_adam"])

        formal_reward = None
        if args.formal_reward:
            description = load_asset_description(asset, expected_manifest_sha256=expected_asset)
            formal_reward = FormalRewardConfig.from_manifest(description.manifest)
        recover_spec = (RecoverTaskSpec(policy_dt_s=.01, action_hold_s=args.recover_hold_s,
                                        height_command_m=args.recover_height)
                        if args.recover else None)
        cfg = build_env_cfg(asset, num_envs=args.num_envs, env_spacing_m=args.env_spacing,
                            device=args.device, response=response, pair_domain=pair,
                            formal_reward=formal_reward, recover=recover_spec,
                            height_scanner=args.height_scanner,
                            terrain_scan=(args.terrain_scan or policy.critic_input_dim == 185),
                            forward_scan=args.forward_scan,
                            terrain_profile=args.terrain_profile,
                            terrain_atlas_manifest=(None if args.terrain_atlas is None else str(args.terrain_atlas)),
                            expected_manifest_sha256=expected_asset)
        health = NativePhysicsHealth.for_native_process(args.output.parent)
        cfg.sim.enable_scene_query_support = True
        cfg.sim.physics = PhysxCfg(
            enable_external_forces_every_iteration=True,
            enable_scene_query_support=True,
            gpu_found_lost_pairs_capacity=2**22,
            gpu_found_lost_aggregate_pairs_capacity=2**25,
            gpu_total_aggregate_pairs_capacity=2**22,
        )
        env = WheeledLabEnv(cfg)
        formal_task_sampler = None
        if args.recover:
            stage = 0
        elif args.formal_curriculum:
            capabilities = FormalCurriculumCapabilities(
                recovery=True, requests=True, terrain=args.terrain_profile == 'course')
            formal_curriculum = FormalTaskCurriculum(
                args.num_envs, device=args.device, seed=args.seed,
                promotion="exposure", capabilities=capabilities,
                include_jump=args.include_jump,
            )
            # Fail before spending warmup when a later stage is not wired.
            for index in range(1, len(formal_curriculum.stage_names)):
                formal_curriculum.require_supported(index)
            if args.resume is not None:
                state = payload.get("curriculum")
                if state is None:
                    raise ValueError("正式课程 resume 缺少 curriculum checkpoint")
                formal_curriculum.load_state_dict(state)
            formal_task_sampler = LabTaskDrawSampler(formal_curriculum)
            env.set_task_sampler(formal_task_sampler)
            stage = formal_task_sampler.stage
        else:
            command_curriculum = LabCommandCurriculum(
                args.num_envs, device=args.device, stage=stage, seed=args.seed)
            env.set_command_sampler(command_curriculum.sample)
        collector = LabCollector(LabLabeledEnv(env), model, profile=args.profile_collection)
        if args.resume is not None and env._terrain_course is not None:
            if payload.get('terrain_course') is None or payload.get('global_rng') is None:
                raise ValueError('course resume requires terrain/request state and global RNG; no implicit migration')
            env._terrain_course.restore_after_physical_reset(payload['terrain_course'])
        args.output.mkdir(parents=True, exist_ok=True)
        metrics_path = args.output / "metrics.jsonl"
        collector.reset()
        if args.resume is not None and payload.get('global_rng') is not None:
            restore_rng(payload['global_rng'])
        print(f"LAB PPO start update={trainer.update_index} envs={args.num_envs} steps={args.rollout_steps} stage={stage} formal_reward={args.formal_reward} formal_curriculum={args.formal_curriculum} recover={args.recover}", flush=True)
        if args.formal_curriculum:
            print("FORMAL curriculum", formal_task_sampler.curriculum.snapshot(), flush=True)
        if args.formal_reward:
            print(f"FORMAL reward missing terms={env._formal_reward.missing_terms}", flush=True)

        for run_update in range(args.updates):
            if formal_task_sampler is not None:
                if formal_task_sampler.curriculum.status != 'running':
                    break
                stage = formal_task_sampler.begin_update()
                print(f"FORMAL stage update={trainer.update_index + 1} stage={stage} "
                      f"consumed={formal_task_sampler.curriculum.snapshot().consumed}", flush=True)
            collection_started = time.perf_counter()
            batch = collector.collect(args.rollout_steps)
            health.check()
            collected = time.perf_counter()
            update_metrics = trainer.update(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            updated = time.perf_counter()
            record = {
                **scope,
                "update": trainer.update_index,
                "collection_s": collected - collection_started,
                "optimization_s": updated - collected,
                "stage": stage,
                "formal_reward": args.formal_reward,
                "recover": args.recover,
                "recover": args.recover,
                "collector": collector.last_metrics,
                "ppo": update_metrics,
            }
            if env._terrain_course is not None:
                record['terrain_course'] = env._terrain_course.metrics()
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
            print("LAB PPO update", json.dumps({
                "update": trainer.update_index,
                "collection_s": round(record["collection_s"], 3),
                "optimization_s": round(record["optimization_s"], 3),
                "reward_mean": collector.last_metrics.get("reward_mean"),
                "kl": update_metrics.get("full_rollout_kl_after_guard"),
                "actor_steps": update_metrics["actor_optimizer_steps"],
                "critic_steps": update_metrics["critic_optimizer_steps"],
                "rolled_back": update_metrics.get("whole_actor_update_rolled_back"),
                "stage": stage,
                "formal_reward": args.formal_reward,
                "recover": args.recover,
            }, ensure_ascii=False), flush=True)
            course_finished = (formal_task_sampler is not None
                               and formal_task_sampler.curriculum.status != 'running')
            if (trainer.update_index % args.checkpoint_interval == 0
                    or run_update == args.updates - 1 or course_finished):
                checkpoint = args.output / f"update_{trainer.update_index:05d}.pt"
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({
                    **scope,
                    "format": "lab_ppo_v1",
                    "policy": asdict(policy),
                    "ppo": asdict(ppo),
                    "model": _cpu(model.state_dict()),
                    "actor_adam": _cpu(trainer.actor_optimizer.state_dict()),
                    "critic_adam": _cpu(trainer.critic_optimizer.state_dict()),
                    "update": trainer.update_index,
                    "stage": stage,
                    "formal_reward": args.formal_reward,
                    "formal_curriculum": args.formal_curriculum,
                    "recover": args.recover,
                    "curriculum": (None if formal_task_sampler is None
                                   else _cpu(formal_task_sampler.state_dict())),
                    'global_rng': _cpu(capture_rng()),
                    'terrain_course': (None if env._terrain_course is None else _cpu(env._terrain_course.state_dict())),
                    "formal_reward_available_terms": (
                        None if env._formal_reward is None else list(env._formal_reward.config.available_terms)),
                    'formal_reward_missing_terms': ({} if env._formal_reward is None
                                                    else env._formal_reward.missing_term_reasons),
                    "asset_manifest_sha256": cfg.asset_manifest_sha256,
                    "response_source_sha256": runtime["response_source_pin"]["sha256"],
                }, temporary)
                temporary.replace(checkpoint)
                (args.output / "latest.txt").write_text(checkpoint.name)
        env.close()
        print("LAB PPO done", flush=True)
    finally:
        try:
            if health is not None:
                health.close()
        finally:
            launcher.app.close()


if __name__ == "__main__":
    main()
