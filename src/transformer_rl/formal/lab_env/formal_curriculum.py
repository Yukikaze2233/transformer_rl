"""正式 3-phase reward/PPO 课程的任务调度层。

这里只负责三件事：按冻结的 STAGE_WEIGHTS 采样 TaskDraw、维护不可退款的
CurriculumBudget、把 stage 状态纳入 checkpoint。PPO 数学仍由
StructuredPPOTrainer 负责，正式 reward 仍由 PhysicalRewardComposer 负责。
没有 terrain/request 能力时显式拒绝对应 stage，不用平地任务冒充。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..control.observation import CommandMode
from ..tasks.curriculum import (
    JUMP_BUDGET, JUMP_WEIGHTS, STAGE_BUDGETS, STAGE_NAMES, STAGE_WEIGHTS,
    TOTAL_BUDGET, ContinuousTaskSampler, CurriculumBudget, TaskDomain, TaskDraw,
    TaskFamily,
)
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

    def missing(self, stage: int) -> tuple[str, ...]:
        return tuple(_FAMILY_NAMES[family] for family, weight in enumerate(STAGE_WEIGHTS[stage])
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
                 domain: TaskDomain | None = None):
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
        self.sampler = ContinuousTaskSampler(seed, device=device, domain=domain)
        self.budget = CurriculumBudget(promotion=promotion)

    @property
    def stage(self) -> int:
        return self.budget.stage

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage]

    @property
    def weights(self) -> tuple[float, ...]:
        return STAGE_WEIGHTS[self.stage]

    @property
    def status(self) -> str:
        return self.budget.status

    def missing_capabilities(self, stage: int | None = None) -> tuple[str, ...]:
        stage = self.stage if stage is None else stage
        if type(stage) is not int or not 0 <= stage < len(STAGE_NAMES):
            raise ValueError("stage 超出正式课程范围")
        return self.capabilities.missing(stage)

    def require_supported(self, stage: int | None = None) -> None:
        stage = self.stage if stage is None else stage
        missing = self.missing_capabilities(stage)
        if missing:
            raise UnsupportedCurriculumTask(
                f"stage {stage} ({STAGE_NAMES[stage]}) 需要未接入能力: {', '.join(missing)}")

    def sample(self, env_ids) -> TaskDraw:
        """为一个 reset 批次采样；不推进 budget，不改变 stage。"""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.ndim != 1 or ids.numel() < 1:
            raise ValueError("env_ids 必须是非空一维整数张量")
        if ids.numel() > self.num_envs:
            raise ValueError("采样行数不能超过环境数")
        self.require_supported()
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
        return {
            "format": self._FORMAT,
            "num_envs": self.num_envs,
            "promotion": self.budget.promotion,
            "capabilities": self.capabilities.to_dict(),
            "domain": self.domain,
            "sampler": self.sampler.state_dict(),
            "budget": self.budget.state_dict(),
        }

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {
                "format", "num_envs", "promotion", "capabilities", "domain", "sampler", "budget"}
                or state["format"] != self._FORMAT or state["num_envs"] != self.num_envs
                or state["promotion"] != self.budget.promotion
                or state["capabilities"] != self.capabilities.to_dict()
                or state["domain"] != self.domain):
            raise ValueError("正式课程 checkpoint 的身份/能力/域不一致")
        sampler = ContinuousTaskSampler(0, device=self.device, domain=self.domain)
        budget = CurriculumBudget(promotion=self.budget.promotion)
        sampler.load_state_dict(state["sampler"])
        budget.load_state_dict(state["budget"])
        self.sampler, self.budget = sampler, budget

    def snapshot(self) -> FormalCurriculumState:
        return FormalCurriculumState(
            self.stage, self.stage_name, STAGE_BUDGETS[self.stage],
            self.budget.consumed[self.stage], self.budget.total_updates,
            tuple(float(value) for value in self.weights), self.status,
        )


@dataclass(frozen=True)
class LabTaskSample:
    """一次 reset 的完整任务行；env 只消费已声明的字段。"""

    draw: TaskDraw
    command: torch.Tensor
    mode: torch.Tensor
    phase: torch.Tensor

    def __post_init__(self):
        n = len(self.draw.family)
        if (self.command.shape != (n, 4) or self.mode.shape != (n,)
                or self.phase.shape != (n,) or self.mode.dtype != torch.long
                or self.phase.dtype != torch.long):
            raise ValueError("LabTaskSample 的 command/mode/phase 布局不一致")


class LabTaskDrawSampler:
    """把正式 TaskDraw 转成 LabEnv 可执行的命令/模式。

    当前 LabEnv 只实现 height/motion/rotation/recovery reset；terrain 和
    jump/request 仍未接入，遇到这些 family 会显式报错，不降级成平地。
    """

    def __init__(self, curriculum: FormalTaskCurriculum):
        if not isinstance(curriculum, FormalTaskCurriculum):
            raise TypeError("curriculum 必须是 FormalTaskCurriculum")
        self.curriculum = curriculum

    def __call__(self, env_ids) -> LabTaskSample:
        draw = self.curriculum.sample(env_ids)
        family = draw.family
        if (family == int(TaskFamily.TERRAIN)).any() or (family == int(TaskFamily.REQUEST)).any():
            raise UnsupportedCurriculumTask("LabEnv 尚未接入 terrain/jump request，不能执行该 TaskDraw")
        command = draw.requested_command.clone()
        command[:, 1] = 0
        recovery = family == int(TaskFamily.RECOVERY)
        command[recovery, 0] = 0
        command[recovery, 2] = 0
        mode = torch.where(recovery, int(CommandMode.RECOVERY), int(CommandMode.NORMAL))
        phase = torch.where(recovery, int(RewardPhase.RECOVERY), int(RewardPhase.SUPPORT))
        return LabTaskSample(draw, command, mode, phase)

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
    """正式 reward/PPO 课程的超参绑定，不复制 PPO 数学。"""

    promotion: str = "exposure"
    stage_budgets: tuple[int, ...] = STAGE_BUDGETS
    stage_weights: tuple[tuple[float, ...], ...] = STAGE_WEIGHTS

    def __post_init__(self):
        if self.promotion not in ("exposure", "audited"):
            raise ValueError("promotion 必须是 exposure 或 audited")
        if tuple(self.stage_budgets) != STAGE_BUDGETS or tuple(self.stage_weights) != STAGE_WEIGHTS:
            raise ValueError("正式课程预算/权重必须与冻结定义一致")
        if sum(self.stage_budgets) != TOTAL_BUDGET:
            raise ValueError(f"正式课程总预算必须是 {TOTAL_BUDGET} updates")
        for index, weights in enumerate(self.stage_weights):
            if (len(weights) != len(TaskFamily) or abs(sum(weights) - 1) > 1e-9
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                           for value in weights)):
                raise ValueError(f"stage {index} 权重必须是有限非负且和为一")

    def stage(self, index: int) -> dict:
        if type(index) is not int or not 0 <= index < len(STAGE_NAMES):
            raise ValueError("stage 超出正式课程范围")
        return {"index": index, "name": STAGE_NAMES[index],
                "budget": self.stage_budgets[index],
                "weights": self.stage_weights[index]}

    def jump(self) -> dict:
        """独立的跳跃巩固段；不计入三段主课程预算。"""
        return {"name": "jump", "budget": JUMP_BUDGET,
                "weights": JUMP_WEIGHTS, "after_main_budget": TOTAL_BUDGET}


__all__ = [
    "FormalCurriculumCapabilities", "FormalCurriculumState", "FormalPPOPlan",
    "FormalTaskCurriculum", "LabTaskDrawSampler", "LabTaskSample",
    "UnsupportedCurriculumTask",
]
