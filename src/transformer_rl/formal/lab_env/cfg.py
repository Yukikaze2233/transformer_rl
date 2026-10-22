"""IsaacLab DirectRLEnv 配置工厂。

只负责把选中资产、执行器分组、接触传感器和地形拼成 DirectRLEnvCfg；
策略时钟、任务公式和历史窗口仍由 contract/settings 约束。
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field
import math

from ..contracts import ACTION_ORDER
from ..control.actuation import PairTargetDomain, ResponseParameters
from .contract import ACTION_DIM, ACTOR_DIM, CRITIC182_DIM, HISTORY_LENGTH, POLICY_DT_S, PHYSICS_DT_S
from .contract import FlatTaskSpec
from .action_transport import ActionTransportConfig
from ..control.communication import CommunicationConfig, EpisodeRandomizationConfig
from .materials import WheelMaterialConfig
from .formal_reward import FormalRewardConfig
from .recover import RecoverTaskSpec
from .settings import AssetDescription, load_asset_description
from ..tasks.termination import NativeTaskLimits
from ..control.sensors import SensorModelConfig
from ..simulation.dynamics import StartupDynamicsConfig

STEP_PLATFORM_DEPTH_M = 0.80

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import (
    HfPyramidSlopedTerrainCfg, HfPyramidStairsTerrainCfg,
    HfRandomUniformTerrainCfg, MeshPlaneTerrainCfg, TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils.configclass import configclass
import isaaclab.sim as sim_utils


@dataclass(frozen=True)
class _ActuatorGroup:
    names: tuple[str, ...]
    damping: float
    effort_limit: float


def _actuator_groups(asset: AssetDescription) -> dict[str, _ActuatorGroup]:
    manifest = asset.manifest
    springs = tuple(manifest["spring_joint_names"])
    passive = tuple(name for name in manifest["tree_joint_names"]
                    if name not in (*ACTION_ORDER, *springs))
    damping = manifest["passive_damping"]
    if springs != ("L_spring_slide", "R_spring_slide"):
        raise ValueError("资产弹簧关节必须与当前契约一致")
    return {
        "legs": _ActuatorGroup(ACTION_ORDER[:4], 0.0, 40.0),
        "wheels": _ActuatorGroup(ACTION_ORDER[4:], 0.0, 4.5),
        "passive": _ActuatorGroup(passive, float(damping["passive_hinge_Nm_s_per_rad"]),
                                  100.0),
        "pressure": _ActuatorGroup(springs, float(damping["slider_N_s_per_m"]),
                                   1000.0),
    }


def _response_fields(response: ResponseParameters | None) -> dict:
    if response is None:
        return {name: {} for name in ("armature", "friction", "dynamic_friction", "viscous_friction")}
    native = response.native_parameters()
    return {name: {joint: values[name] for joint, values in native.items()}
            for name in ("armature", "friction", "dynamic_friction", "viscous_friction")}


def build_robot_cfg(asset: AssetDescription, *,
                    prim_path: str = "/World/envs/env_.*/Robot",
                    response: ResponseParameters | None = None) -> ArticulationCfg:
    """按资产 manifest 构造显式零增益执行器。

    PD 由 WheeledControlCore 在 200Hz 反馈中计算，USD/执行器只保留
    被动阻尼、惯量和输出上限；不在此处偷偷加入位置增益。
    """
    groups = _actuator_groups(asset)
    fields = _response_fields(response)
    velocity_limits = asset.joint_velocity_limits()
    actuators = {}
    for name, group in groups.items():
        values = {
            field_name: {joint: fields[field_name].get(joint, 0.0) for joint in group.names}
            for field_name in fields
        }
        # Some USD exports omit maxJointVelocity. Use the pinned model source
        # rather than an unbounded override or an SDK-dependent fallback.
        actuators[name] = IdealPDActuatorCfg(
            joint_names_expr=list(group.names),
            stiffness=0.0,
            damping=group.damping,
            effort_limit=group.effort_limit,
            effort_limit_sim=group.effort_limit,
            velocity_limit_sim={joint: velocity_limits[joint] for joint in group.names},
            **values,
        )
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(asset.usd_path),
            activate_contact_sensors=True,
            copy_from_source=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=asset.manifest["nominal_joint_pos"],
            joint_vel={".*": 0.0},
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )


def build_contact_sensor_cfg() -> ContactSensorCfg:
    """地形过滤后的每物理步净接触力；不保留 air-time 历史。"""
    # 机器人自碰撞已关闭；跨环境互撞由 _setup_scene 的碰撞组隔离。
    # 保留整机 contact sensor，以维持机身前部/腿部接触语义。
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=0,
        update_period=0.0,
        track_air_time=False,
        debug_vis=False,
    )


def build_height_scanner_cfg() -> RayCasterCfg:
    """机体 base_link 正上方的地面高度射线；平地/terrain 共用。"""
    return RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.05, 0.05)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def build_terrain_scan_cfg() -> RayCasterCfg:
    """11x7 body-frame ground grid for the 77D privileged critic block."""
    return RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 5.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.6)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=POLICY_DT_S,
    )


def build_forward_scanners_cfg() -> tuple[RayCasterCfg, RayCasterCfg, RayCasterCfg, RayCasterCfg]:
    """Wheel-current and wheel-forward ground rays for the stair/jump FSM."""
    def one(prim: str, x_m: float) -> RayCasterCfg:
        return RayCasterCfg(
            prim_path=f"/World/envs/env_.*/Robot/{prim}",
            offset=RayCasterCfg.OffsetCfg(pos=(x_m, 0.0, 0.20)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.01, 0.01)),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
            update_period=0.0,
        )
    return one("L_link3", 0.0), one("L_link3", 0.25), one("R_link3", 0.0), one("R_link3", 0.25)


def build_terrain_cfg(profile: str = "plane", *, atlas_manifest=None) -> TerrainImporterCfg:
    if profile not in ("plane", "rough", "course"):
        raise ValueError("terrain profile 只支持 plane/rough/course")
    if profile == 'course':
        from pathlib import Path
        if atlas_manifest is None:
            raise ValueError('course requires a prepared terrain atlas manifest')
        return TerrainImporterCfg(
            prim_path='/World/ground', terrain_type='usd', collision_group=-1,
            usd_path=str(Path(atlas_manifest).resolve().parent / 'terrain.usda'), debug_vis=False)
    if profile == "rough":
        generator = TerrainGeneratorCfg(
            size=(9.0, 9.0), border_width=5.0, num_rows=10, num_cols=14,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            difficulty_range=(0.0, 1.0), use_cache=False,
            sub_terrains={
                "flat": MeshPlaneTerrainCfg(proportion=0.25),
                "rough": HfRandomUniformTerrainCfg(
                    proportion=0.35, noise_range=(0.01, 0.04), noise_step=0.01,
                    border_width=0.25),
                "slope": HfPyramidSlopedTerrainCfg(
                    proportion=0.2, slope_range=(0.0, 0.35), platform_width=1.0,
                    border_width=0.25),
                "stairs": HfPyramidStairsTerrainCfg(
                    proportion=0.2, step_height_range=(0.05, 0.14),
                    step_width=STEP_PLATFORM_DEPTH_M, platform_width=1.0, border_width=0.25),
            },
        )
        return TerrainImporterCfg(
            prim_path="/World/ground", terrain_type="generator",
            terrain_generator=generator, collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            debug_vis=False,
        )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )


@configclass
class WheeledLabSceneCfg(InteractiveSceneCfg):
    """GA 在 scene 构造期登记机器人；共享地形仍由环境绑定。"""

    robot: ArticulationCfg = MISSING


@configclass
class WheeledLabEnvCfg(DirectRLEnvCfg):
    """M1 平地 DirectRLEnv 配置；资产相关字段由 build_env_cfg 填充。"""

    decimation = 2
    episode_length_s = 30.0
    action_space = ACTION_DIM
    observation_space = ACTOR_DIM * HISTORY_LENGTH
    state_space = CRITIC182_DIM
    history_length = HISTORY_LENGTH
    asset_directory: str = ""
    declarative_scene: bool = False
    loop_telemetry_enabled: bool = False

    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT_S,
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=4.0,
        replicate_physics=True,
    )
    terrain: TerrainImporterCfg = field(default_factory=build_terrain_cfg)
    robot_cfg: ArticulationCfg = MISSING
    contact_sensor: ContactSensorCfg = field(default_factory=build_contact_sensor_cfg)
    height_scanner: RayCasterCfg | None = None
    terrain_scanner: RayCasterCfg | None = None
    forward_scanners: tuple[RayCasterCfg, ...] = field(default_factory=tuple)

    def __post_init__(self):
        super().__post_init__()
        # 配置对象必须显式表达 100Hz 策略/200Hz 物理，不依赖后端默认。
        if self.decimation != 2 or abs(self.sim.dt - PHYSICS_DT_S) > 1e-12:
            raise ValueError("M1 固定 decimation=2、sim.dt=0.005")
        if abs(self.decimation * self.sim.dt - POLICY_DT_S) > 1e-12:
            raise ValueError("decimation*sim.dt 必须等于 0.01s")
        if self.observation_space != ACTOR_DIM * HISTORY_LENGTH or self.state_space != CRITIC182_DIM:
            raise ValueError("LabEnv 观测空间固定 5x35 actor / 182 critic")
        if self.action_space != ACTION_DIM:
            raise ValueError("M1 动作空间固定 6")
        if self.history_length != HISTORY_LENGTH:
            raise ValueError("M1 固定 5 帧历史")


def build_env_cfg(asset_directory: str, *, num_envs: int = 8192,
                  env_spacing_m: float = 4.0, device: str = "cuda:0",
                  response: ResponseParameters | None = None,
                  pair_domain: PairTargetDomain | None = None,
                  task: FlatTaskSpec | None = None,
                  formal_reward: FormalRewardConfig | None = None,
                  termination_limits: NativeTaskLimits | None = None,
                  sensors: SensorModelConfig | None = None,
                  dynamics: StartupDynamicsConfig | None = None,
                  action_transport: ActionTransportConfig | None = None,
                  communication: CommunicationConfig | None = None,
                  episode_randomization: EpisodeRandomizationConfig | None = None,
                  wheel_material: WheelMaterialConfig | None = None,
                  recover: RecoverTaskSpec | None = None,
                  height_scanner: bool = False,
                  terrain_scan: bool = False,
                  forward_scan: bool = False,
                  terrain_profile: str = "plane",
                  terrain_atlas_manifest: str | None = None,
                  declarative_scene: bool = False,
                  seed: int | None = 0,
                  expected_manifest_sha256: str | None = None) -> WheeledLabEnvCfg:
    """从资产目录构造可直接交给 WheeledLabEnv 的配置。"""
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs 必须是正整数")
    if (type(env_spacing_m) not in (int, float) or not math.isfinite(env_spacing_m)
            or env_spacing_m <= 0):
        raise ValueError("env_spacing_m 必须是有限正数")
    if formal_reward is not None and not isinstance(termination_limits, NativeTaskLimits):
        raise ValueError('formal Lab tasks require explicit original physical termination limits')
    if (sensors is not None and not isinstance(sensors, SensorModelConfig)
            or dynamics is not None and not isinstance(dynamics, StartupDynamicsConfig)
            or action_transport is not None and not isinstance(action_transport, ActionTransportConfig)
            or communication is not None and not isinstance(communication, CommunicationConfig)
            or episode_randomization is not None and not isinstance(episode_randomization, EpisodeRandomizationConfig)
            or wheel_material is not None and not isinstance(wheel_material, WheelMaterialConfig)):
        raise ValueError('Lab randomization requires its explicit sensor, transport and mechanical configurations')
    asset = load_asset_description(asset_directory,
                                   expected_manifest_sha256=expected_manifest_sha256)
    if communication is not None and communication.enabled and sensors is None:
        raise ValueError('shared communication requires the actual physical-clock sensor history')
    cfg = WheeledLabEnvCfg()
    object.__setattr__(cfg, "asset_manifest_sha256", asset.manifest_sha256)
    cfg.asset_directory = str(asset.directory)
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = env_spacing_m
    cfg.sim.device = device
    cfg.seed = seed
    # 冻结任务/响应对象不放进 configclass 字段，避免其 post-init 递归改写。
    object.__setattr__(cfg, "task_spec", task if task is not None else FlatTaskSpec())
    object.__setattr__(cfg, "response_spec", response)
    object.__setattr__(cfg, "pair_domain_spec", pair_domain)
    object.__setattr__(cfg, "formal_reward_spec", formal_reward)
    object.__setattr__(cfg, 'termination_limits', termination_limits)
    object.__setattr__(cfg, 'sensor_model_spec', sensors)
    object.__setattr__(cfg, 'startup_dynamics_spec', dynamics)
    object.__setattr__(cfg, 'action_transport_spec', action_transport)
    object.__setattr__(cfg, 'communication_spec', communication)
    object.__setattr__(cfg, 'episode_randomization_spec', episode_randomization)
    object.__setattr__(cfg, 'wheel_material_spec', wheel_material)
    object.__setattr__(cfg, "recover_spec", recover)
    object.__setattr__(cfg, "course_manifest", terrain_atlas_manifest)
    object.__setattr__(cfg, 'static_plane_height_m', 0. if terrain_profile == 'plane' else None)
    if recover is not None:
        cfg.episode_length_s = recover.episode_length_s
    cfg.robot_cfg = build_robot_cfg(asset, response=response)
    if type(declarative_scene) is not bool:
        raise ValueError('declarative_scene must be an explicit bool')
    cfg.declarative_scene = declarative_scene
    if declarative_scene:
        cfg.scene = WheeledLabSceneCfg(
            num_envs=num_envs, env_spacing=env_spacing_m, replicate_physics=True,
            filter_collisions=False, robot=cfg.robot_cfg)
        # 我们在每个物理步显式计算电机力矩，不启用 GA 新的 host actuator 路径。
        cfg.sim.use_newton_actuators = False
    cfg.contact_sensor = build_contact_sensor_cfg()
    if formal_reward is not None and formal_reward.normal_load_source == 'native_contact_patch_normal_forces':
        if terrain_profile != 'course':
            raise ValueError('native patch-load binding currently requires the frozen static course atlas')
        from .terrain_collision import NATIVE_COLLISION_PATTERN
        cfg.contact_sensor.filter_prim_paths_expr = [NATIVE_COLLISION_PATTERN]
        cfg.contact_sensor.max_contact_data_count_per_prim = 16
    # The 11x7 scanner already contains the center ground ray; do not build a
    # second single-ray sensor in large rough-terrain batches.
    cfg.height_scanner = (build_height_scanner_cfg()
                          if height_scanner and not terrain_scan else None)
    cfg.terrain_scanner = build_terrain_scan_cfg() if terrain_scan else None
    cfg.forward_scanners = build_forward_scanners_cfg() if forward_scan else ()
    cfg.terrain = build_terrain_cfg(terrain_profile, atlas_manifest=terrain_atlas_manifest)
    if terrain_profile in ('course', 'plane'):
        # Query the fixed plane analytically, or the course's exact triangles.
        # Neither path needs a second RayCaster/BVH or an exported Warp view.
        cfg.height_scanner = cfg.terrain_scanner = None
        cfg.forward_scanners = ()
    return cfg
