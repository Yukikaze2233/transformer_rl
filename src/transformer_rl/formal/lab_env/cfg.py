"""IsaacLab DirectRLEnv 配置工厂。

只负责把选中资产、执行器分组、接触传感器和地形拼成 DirectRLEnvCfg；
策略时钟、任务公式和历史窗口仍由 contract/settings 约束。
"""
from __future__ import annotations

from dataclasses import MISSING, dataclass, field
import math

from ..contracts import ACTION_ORDER
from ..control.actuation import PairTargetDomain, ResponseParameters
from .contract import ACTION_DIM, ACTOR_DIM, CRITIC_DIM, HISTORY_LENGTH, POLICY_DT_S, PHYSICS_DT_S
from .contract import FlatTaskSpec
from .formal_reward import FormalRewardConfig
from .recover import RecoverTaskSpec
from .settings import AssetDescription, load_asset_description

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
    velocity_limit: float


def _actuator_groups(asset: AssetDescription) -> dict[str, _ActuatorGroup]:
    manifest = asset.manifest
    springs = tuple(manifest["spring_joint_names"])
    passive = tuple(name for name in manifest["tree_joint_names"]
                    if name not in (*ACTION_ORDER, *springs))
    damping = manifest["passive_damping"]
    if springs != ("L_spring_slide", "R_spring_slide"):
        raise ValueError("资产弹簧关节必须与当前契约一致")
    return {
        "legs": _ActuatorGroup(ACTION_ORDER[:4], 0.0, 40.0, 1e9),
        "wheels": _ActuatorGroup(ACTION_ORDER[4:], 0.0, 4.5, 1e9),
        "passive": _ActuatorGroup(passive, float(damping["passive_hinge_Nm_s_per_rad"]),
                                  100.0, 1e9),
        "pressure": _ActuatorGroup(springs, float(damping["slider_N_s_per_m"]),
                                   1000.0, 100.0),
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
    actuators = {}
    for name, group in groups.items():
        values = {
            field_name: {joint: fields[field_name].get(joint, 0.0) for joint in group.names}
            for field_name in fields
        }
        actuators[name] = IdealPDActuatorCfg(
            joint_names_expr=list(group.names),
            stiffness=0.0,
            damping=group.damping,
            effort_limit=group.effort_limit,
            effort_limit_sim=group.effort_limit,
            velocity_limit_sim=group.velocity_limit,
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
    # 机器人自碰撞已关闭，平地阶段地面是唯一碰撞体；不设置 filter 可避免
    # IsaacLab GPU 对全局 ground prim 的 filter 警告和额外建视图开销。
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


def build_terrain_cfg(profile: str = "plane") -> TerrainImporterCfg:
    if profile not in ("plane", "rough"):
        raise ValueError("terrain profile 只支持 plane/rough")
    if profile == "rough":
        generator = TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=4, num_cols=4,
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
                    step_width=0.3, platform_width=1.0, border_width=0.25),
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
class WheeledLabEnvCfg(DirectRLEnvCfg):
    """M1 平地 DirectRLEnv 配置；资产相关字段由 build_env_cfg 填充。"""

    decimation = 2
    episode_length_s = 30.0
    action_space = ACTION_DIM
    observation_space = ACTOR_DIM * HISTORY_LENGTH
    state_space = CRITIC_DIM
    history_length = HISTORY_LENGTH
    asset_directory: str = ""

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

    def __post_init__(self):
        super().__post_init__()
        # 配置对象必须显式表达 100Hz 策略/200Hz 物理，不依赖后端默认。
        if self.decimation != 2 or abs(self.sim.dt - PHYSICS_DT_S) > 1e-12:
            raise ValueError("M1 固定 decimation=2、sim.dt=0.005")
        if abs(self.decimation * self.sim.dt - POLICY_DT_S) > 1e-12:
            raise ValueError("decimation*sim.dt 必须等于 0.01s")
        if self.observation_space != ACTOR_DIM * HISTORY_LENGTH or self.state_space != CRITIC_DIM:
            raise ValueError("M1 观测空间固定 5x35 / 81")
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
                  recover: RecoverTaskSpec | None = None,
                  height_scanner: bool = False,
                  terrain_profile: str = "plane",
                  seed: int | None = 0,
                  expected_manifest_sha256: str | None = None) -> WheeledLabEnvCfg:
    """从资产目录构造可直接交给 WheeledLabEnv 的配置。"""
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs 必须是正整数")
    if (type(env_spacing_m) not in (int, float) or not math.isfinite(env_spacing_m)
            or env_spacing_m <= 0):
        raise ValueError("env_spacing_m 必须是有限正数")
    asset = load_asset_description(asset_directory,
                                   expected_manifest_sha256=expected_manifest_sha256)
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
    object.__setattr__(cfg, "recover_spec", recover)
    if recover is not None:
        cfg.episode_length_s = recover.episode_length_s
    cfg.robot_cfg = build_robot_cfg(asset, response=response)
    cfg.contact_sensor = build_contact_sensor_cfg()
    cfg.height_scanner = build_height_scanner_cfg() if height_scanner else None
    cfg.terrain = build_terrain_cfg(terrain_profile)
    return cfg
