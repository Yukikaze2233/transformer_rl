"""正式 3-phase reward/PPO 课程的任务调度层。

这里只负责三件事：按冻结的 STAGE_WEIGHTS 采样 TaskDraw、维护不可退款的
CurriculumBudget、把 stage 状态纳入 checkpoint。PPO 数学仍由
StructuredPPOTrainer 负责，正式 reward 仍由 PhysicalRewardComposer 负责。
没有 terrain/request 能力时显式拒绝对应 stage，不用平地任务冒充。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch

from ..control.observation import CommandMode
from ..tasks.curriculum import (
    COMPLETE_STAGE_BUDGETS, COMPLETE_STAGE_NAMES, COMPLETE_STAGE_WEIGHTS,
    INITIAL_DOMAIN, JUMP_BUDGET, JUMP_WEIGHTS, STAGE_BUDGETS, STAGE_NAMES,
    STAGE_WEIGHTS, TOTAL_BUDGET, ContinuousTaskSampler, CurriculumBudget, TaskDomain,
    TaskDraw, TaskFamily, interpolate_domain,
)
from ..tasks.geometry import TerrainKind
from ..tasks.reward import RewardPhase


_FAMILY_NAMES = {
    TaskFamily.HEIGHT: "height",
    TaskFamily.MOTION: "motion",
    TaskFamily.ROTATION: "rotation",
    TaskFamily.TERRAIN: "terrain",
    TaskFamily.REQUEST: "request",
    TaskFamily.RECOVERY: "recovery",
}


class UnsupportedCurriculumTask(RuntimeError):
    """当前 LabEnv 能力不足以执行该 stage 的冻结任务族。"""


@dataclass(frozen=True)
class FormalCurriculumCapabilities:
    """显式声明 LabEnv 已具备的任务能力；缺省全部关闭。"""

    terrain: bool = False
    requests: bool = False
    recovery: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in vars(self).values()):
            raise ValueError("课程能力开关必须是显式 bool")

    def supports(self, family: TaskFamily) -> bool:
        if family in (TaskFamily.HEIGHT, TaskFamily.MOTION, TaskFamily.ROTATION):
            return True
        if family == TaskFamily.TERRAIN:
            return self.terrain
        if family == TaskFamily.REQUEST:
            return self.requests
        if family == TaskFamily.RECOVERY:
            return self.recovery
        raise ValueError("未声明的任务族")

    def missing(self, stage: int, weights=STAGE_WEIGHTS) -> tuple[str, ...]:
        return tuple(_FAMILY_NAMES[family] for family, weight in enumerate(weights[stage])
                     if weight > 0 and not self.supports(TaskFamily(family)))

    def to_dict(self):
        return {name: getattr(self, name) for name in ("terrain", "requests", "recovery")}


@dataclass(frozen=True)
class FormalCurriculumState:
    """只读快照，便于日志和测试；不是 checkpoint 格式。"""

    stage: int
    stage_name: str
    budget: int
    consumed: int
    total_updates: int
    weights: tuple[float, ...]
    status: str


class FormalTaskCurriculum:
    """精确 3-phase 调度器；采样与名额记账共用一条 RNG/预算状态。"""

    _FORMAT = "lab_formal_curriculum_v1"

    def __init__(self, num_envs: int, device="cpu", *, seed: int = 0,
                 promotion: str = "exposure",
                 capabilities: FormalCurriculumCapabilities | None = None,
                 domain: TaskDomain | None = None, include_jump: bool = False,
                 continuous: bool = False, initial_domain: TaskDomain | None = None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs 必须是正整数")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed 必须是非负整数")
        if promotion not in ("exposure", "audited"):
            raise ValueError("promotion 必须是 exposure 或 audited")
        if capabilities is None:
            capabilities = FormalCurriculumCapabilities()
        if not isinstance(capabilities, FormalCurriculumCapabilities):
            raise TypeError("capabilities 必须是 FormalCurriculumCapabilities")
        if domain is None:
            domain = TaskDomain()
        if not isinstance(domain, TaskDomain):
            raise TypeError("domain 必须是 TaskDomain")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.capabilities = capabilities
        self.domain = domain
        if type(include_jump) is not bool or type(continuous) is not bool:
            raise TypeError('include_jump and continuous must be explicit bool')
        if initial_domain is None:
            initial_domain = INITIAL_DOMAIN
        if not isinstance(initial_domain, TaskDomain):
            raise TypeError('initial_domain must be a parsed TaskDomain')
        self.include_jump = include_jump
        self.continuous = continuous
        self.initial_domain = initial_domain
        self.stage_names = COMPLETE_STAGE_NAMES if include_jump else STAGE_NAMES
        self.stage_budgets = COMPLETE_STAGE_BUDGETS if include_jump else STAGE_BUDGETS
        self.stage_weights = COMPLETE_STAGE_WEIGHTS if include_jump else STAGE_WEIGHTS
        self.sampler = ContinuousTaskSampler(seed, device=device, domain=domain, stage_weights=self.stage_weights,
                                             continuous=continuous)
        self.budget = CurriculumBudget(promotion=promotion, stage_names=self.stage_names, stage_budgets=self.stage_budgets)

    @property
    def stage(self) -> int:
        return self.budget.stage

    @property
    def stage_name(self) -> str:
        return self.stage_names[self.stage]

    @property
    def weights(self) -> tuple[float, ...]:
        if not self.continuous or self.stage == 0:
            return self.stage_weights[self.stage]
        consumed = self.budget.consumed[self.stage]
        transition = max(1, self.stage_budgets[self.stage] // 4)
        fraction = min(consumed / transition, 1.0)
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        previous, target = self.stage_weights[self.stage - 1], self.stage_weights[self.stage]
        return tuple(float(a + (b - a) * fraction) for a, b in zip(previous, target))

    def _effective_domain(self) -> TaskDomain:
        if not self.continuous:
            return self.domain
        total = sum(self.stage_budgets)
        fraction = min(self.budget.total_updates / total, 1.0)
        fraction = fraction * fraction * (3.0 - 2.0 * fraction)
        domain = interpolate_domain(self.initial_domain, self.domain, fraction)
        # Jump tasks are absent before the final stage. Their envelope must
        # therefore follow the jump stage's own exposure, not total progress;
        # otherwise the first sampled jump is already near the final ceiling.
        if self.stage == len(self.stage_names) - 1:
            exposure = min(self.budget.consumed[self.stage] / self.stage_budgets[self.stage], 1.0)
            exposure = exposure * exposure * (3.0 - 2.0 * exposure)
            start, end = self.initial_domain.jump_m, self.domain.jump_m
            domain = replace(domain, jump_m=(
                float(start[0] + (end[0] - start[0]) * exposure),
                float(start[1] + (end[1] - start[1]) * exposure),
            ))
        return domain

    @property
    def status(self) -> str:
        return self.budget.status

    def missing_capabilities(self, stage: int | None = None) -> tuple[str, ...]:
        if stage is None:
            weights = self.weights
        else:
            if type(stage) is not int or not 0 <= stage < len(self.stage_names):
                raise ValueError("stage 超出正式课程范围")
            weights = self.stage_weights[stage]
        return tuple(_FAMILY_NAMES[TaskFamily(family)] for family, weight in enumerate(weights)
                     if weight > 0 and not self.capabilities.supports(TaskFamily(family)))

    def require_supported(self, stage: int | None = None) -> None:
        stage = self.stage if stage is None else stage
        missing = self.missing_capabilities(stage)
        if missing:
            raise UnsupportedCurriculumTask(
                f"stage {stage} ({self.stage_names[stage]}) 需要未接入能力: {', '.join(missing)}")

    def sample(self, env_ids) -> TaskDraw:
        """为一个 reset 批次采样；不推进 budget，不改变 stage。"""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or ids.numel() < 1:
            raise ValueError("env_ids 必须是非空一维整数张量")
        if ids.numel() > self.num_envs:
            raise ValueError("采样行数不能超过环境数")
        self.require_supported()
        if self.continuous:
            return self.sampler.sample(ids.numel(), self.stage,
                                       weights=self.weights, domain=self._effective_domain())
        return self.sampler.sample(ids.numel(), self.stage)

    def begin_update(self) -> int:
        """在完整 rollout 前推进 stage 边界并扣一次预算。"""
        self.require_supported()
        if self.budget.status != "running":
            raise RuntimeError("课程已结束或等待人工验证，不能继续扣预算")
        stage = self.budget.prepare_rollout()
        self.require_supported(stage)
        self.budget.charge_rollout()
        return stage

    def state_dict(self):
        state = {
            "format": 'lab_formal_curriculum_v2' if self.include_jump else self._FORMAT,
            "num_envs": self.num_envs,
            "promotion": self.budget.promotion,
            "capabilities": self.capabilities.to_dict(),
            "domain": self.domain,
            "sampler": self.sampler.state_dict(),
            "budget": self.budget.state_dict(),
        }
        if self.continuous:
            state.update({'continuous': True, 'initial_domain': self.initial_domain})
        return state

    def load_state_dict(self, state):
        expected_keys = {"format", "num_envs", "promotion", "capabilities", "domain", "sampler", "budget"}
        if self.continuous:
            expected_keys |= {"continuous", "initial_domain"}
        if (not isinstance(state, dict) or set(state) != expected_keys
                or state["format"] != ('lab_formal_curriculum_v2' if self.include_jump else self._FORMAT)
                or state["num_envs"] != self.num_envs
                or state["promotion"] != self.budget.promotion
                or state["capabilities"] != self.capabilities.to_dict()
                or state["domain"] != self.domain
                or (self.continuous and (state["continuous"] is not True
                                          or state["initial_domain"] != self.initial_domain))):
            raise ValueError("正式课程 checkpoint 的身份/能力/域不一致")
        sampler = ContinuousTaskSampler(0, device=self.device, domain=self.domain,
                                        stage_weights=self.stage_weights, continuous=self.continuous)
        budget = CurriculumBudget(promotion=self.budget.promotion, stage_names=self.stage_names, stage_budgets=self.stage_budgets)
        sampler.load_state_dict(state["sampler"])
        budget.load_state_dict(state["budget"])
        self.sampler, self.budget = sampler, budget

    def snapshot(self) -> FormalCurriculumState:
        return FormalCurriculumState(
            self.stage, self.stage_name, self.stage_budgets[self.stage],
            self.budget.consumed[self.stage], self.budget.total_updates,
            tuple(float(value) for value in self.weights), self.status,
        )


@dataclass(frozen=True)
class LabTaskSample:
    """一次 reset 的完整任务行；env 只消费已声明的字段。"""

    draw: TaskDraw
    command: torch.Tensor
    request_height_m: torch.Tensor
    terrain_kind: torch.Tensor
    mode: torch.Tensor
    phase: torch.Tensor

    def __post_init__(self):
        n = len(self.draw.family)
        if (self.command.shape != (n, 4) or self.request_height_m.shape != (n,)
                or self.terrain_kind.shape != (n,) or self.mode.shape != (n,)
                or self.phase.shape != (n,) or self.mode.dtype != torch.long
                or self.terrain_kind.dtype != torch.long or self.phase.dtype != torch.long
                or self.command.dtype != torch.float32
                or self.request_height_m.dtype != torch.float32):
            raise ValueError("LabTaskSample 的 command/request/terrain/mode/phase 布局不一致")


class LabTaskDrawSampler:
    """把正式 TaskDraw 转成 LabEnv 命令、请求高度、地形类别和模式。

    环境是否已经具备 terrain/jump 路由由环境层显式检查；这里不把未知
    family 降级成平地，也不把请求高度写入普通高度命令。
    """

    def __init__(self, curriculum: FormalTaskCurriculum):
        if not isinstance(curriculum, FormalTaskCurriculum):
            raise TypeError("curriculum 必须是 FormalTaskCurriculum")
        self.curriculum = curriculum

    def map_draw(self, draw: TaskDraw) -> LabTaskSample:
        if not isinstance(draw, TaskDraw):
            raise TypeError("draw 必须是 TaskDraw")
        family = draw.family
        recovery = family == int(TaskFamily.RECOVERY)
        request = family == int(TaskFamily.REQUEST)
        terrain = family == int(TaskFamily.TERRAIN)
        command = draw.requested_command.clone()
        command[:, 1] = 0
        command[recovery, 0] = 0
        command[recovery, 2] = 0
        stairs = torch.isin(draw.terrain_kind, torch.tensor(
            [int(TerrainKind.STEP_UP), int(TerrainKind.STEP_DOWN),
             int(TerrainKind.STAIRS), int(TerrainKind.TWO_PLATFORMS)],
            dtype=torch.long, device=draw.terrain_kind.device))
        slope = draw.terrain_kind == int(TerrainKind.SLOPE)
        terrain_mode = torch.where(slope, int(CommandMode.SLOPE),
                                   torch.where(stairs, int(CommandMode.STAIR), int(CommandMode.NORMAL)))
        mode = torch.where(recovery, int(CommandMode.RECOVERY),
                           torch.where(request, int(CommandMode.JUMP),
                                       torch.where(terrain, terrain_mode, int(CommandMode.NORMAL))))
        phase = torch.where(
            recovery, int(RewardPhase.RECOVERY),
            torch.where(request, int(RewardPhase.TAKEOFF), int(RewardPhase.SUPPORT)))
        request_height = torch.where(request, draw.jump_height_m,
                                     torch.zeros_like(draw.jump_height_m))
        return LabTaskSample(draw, command, request_height, draw.terrain_kind, mode, phase)

    def __call__(self, env_ids) -> LabTaskSample:
        return self.map_draw(self.curriculum.sample(env_ids))

    def begin_update(self):
        return self.curriculum.begin_update()

    @property
    def stage(self):
        return self.curriculum.stage

    def state_dict(self):
        return self.curriculum.state_dict()

    def load_state_dict(self, state):
        self.curriculum.load_state_dict(state)


@dataclass(frozen=True)
class FormalPPOPlan:
    """完整 35k 课程的预算/权重绑定，不复制 PPO 数学。"""

    promotion: str = "exposure"
    stage_budgets: tuple[int, ...] = COMPLETE_STAGE_BUDGETS
    stage_weights: tuple[tuple[float, ...], ...] = COMPLETE_STAGE_WEIGHTS

    def __post_init__(self):
        if self.promotion not in ("exposure", "audited"):
            raise ValueError("promotion 必须是 exposure 或 audited")
        if (tuple(self.stage_budgets) != COMPLETE_STAGE_BUDGETS
                or tuple(self.stage_weights) != COMPLETE_STAGE_WEIGHTS):
            raise ValueError("正式课程预算/权重必须与冻结的 35k 定义一致")
        if sum(self.stage_budgets) != sum(COMPLETE_STAGE_BUDGETS):
            raise ValueError(f"正式课程总预算必须是 {sum(COMPLETE_STAGE_BUDGETS)} updates")
        for index, weights in enumerate(self.stage_weights):
            if (len(weights) != len(TaskFamily) or abs(sum(weights) - 1) > 1e-9
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in weights)):
                raise ValueError(f"stage {index} 权重必须是有限非负且和为一")

    def stage(self, index: int) -> dict:
        if type(index) is not int or not 0 <= index < len(COMPLETE_STAGE_NAMES):
            raise ValueError("stage 超出正式课程范围")
        return {"index": index, "name": COMPLETE_STAGE_NAMES[index],
                "budget": self.stage_budgets[index],
                "weights": self.stage_weights[index]}

    def jump(self) -> dict:
        """跳跃巩固是第 4 段，继承同一模型和 Adam，不重置。"""
        index = len(COMPLETE_STAGE_NAMES) - 1
        return {"index": index, "name": COMPLETE_STAGE_NAMES[index],
                "budget": COMPLETE_STAGE_BUDGETS[index],
                "weights": COMPLETE_STAGE_WEIGHTS[index]}


__all__ = [
    "FormalCurriculumCapabilities", "FormalCurriculumState", "FormalPPOPlan",
    "FormalTaskCurriculum", "LabTaskDrawSampler", "LabTaskSample",
    "UnsupportedCurriculumTask",
]
