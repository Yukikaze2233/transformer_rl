# 研究结论与归档

收尾日期：2026-09-19。

## 工程决定

**结束本轮Transformer训练，当前低层实时控制优先采用短历史MLP／轻量状态估计器。**
现有实验没有观察到足以抵偿Transformer额外计算与实现复杂度的控制收益。
已有训练代码、配置、模型和原始评估保留，后续预算不继续执行。

这是当前证据下的工程选型，并不证明Transformer在所有机器人任务中更差，也不表示
首轮MLP已经达到部署要求。完整的估计器配对长训、历史长度消融和目标MiniPC验证均未完成。

## 已完成与终止范围

| 工作 | 最终状态 |
|---|---|
| 首轮7种网络 × 3个seed | 21个最终模型、147份评估完成并回收 |
| 每项首轮预算 | 977次更新，16,007,168条完整rollout样本 |
| 六种估计器／直接控制网络 | 训练、双优化器恢复及导出路径已实现 |
| 接线：direct MLP、direct Transformer、velocity MLP | 三项训练与评估完成，各80次更新 |
| 接线：velocity Transformer | 80次更新完成，评估中断 |
| 接线：context MLP、context Transformer | 未启动 |
| 16M特权诊断、18项64M主对照、可选延长及轻量消融 | 未执行，收尾后取消继续安排 |

估计器队列原本已于北京时间2026-09-18 03:49:28停止。9月19日再次执行
`~/trainctl pause estimator --wait 180`并复核，`active=false`、
`supervisor_active=false`、`orphaned_workers=[]`。相关进程扫描没有匹配项。
这次是确认停止并结束后续计划，不是又完成了一轮训练。
核查记录见[研究收尾收据](evidence/research-closure.json)。

## 首轮实际表现

下表为三个训练seed的均值。高度列是`stand_mid`场景中实际高度减目标高度的偏差；
速度列为前进／后退场景的全区间vx MAE，指令分别为+0.5／−0.5 m/s。

| 网络 | 高度偏差 mm | 前进vx MAE m/s | 后退vx MAE m/s |
|---|---:|---:|---:|
| Last-token Transformer | −86.95 | 0.5024 | 0.4979 |
| Time Transformer | −71.25 | 0.4971 | 0.4849 |
| Index Transformer | −81.48 | 0.5017 | 0.4772 |
| Gated Transformer | −89.09 | 0.5033 | 0.4966 |
| Supervised Transformer | −103.69 | 0.5029 | 0.4966 |
| History MLP | −81.95 | 0.5051 | 0.4942 |
| History GRU | −90.04 | 0.5025 | 0.4970 |

各网络的速度跟踪均未合格，高度存在明显偏差。部分策略的波动很小，但不能据此
认定控制成功或选出架构赢家。Time Transformer的这项高度均值较好，也不足以支持
整体优势结论；跨seed差异大，且任务本身未解决。

数据来源：本地`artifacts/analysis/training-results-20260918/results.json`，SHA-256：
`58a7fe4efbb0c888906afd4d3e3b08080e57563fa49560fdcff54d5373402c7b`。
原始模型与评估的回收验证见[首轮完成记录](KAISER_ARCHITECTURE_COMPLETE.md)。
两项supervised任务是分段续训，保留其环境、历史及随机数流重置的解释边界。

新估计器接线只有单seed短训，其中10秒评估不足一个20秒完整episode。
这些结果验证数据与优化接口，不用于推断长期控制质量或估计器优劣。

## 实时控制为什么优先短历史

低层平衡与运动控制通常依赖当前姿态、角速度、关节状态和可由短历史估计的速度。
当这些信息已经充分表征控制所需状态时，更旧观测的新增价值有限；增加历史长度
并不会自动改善控制。短窗口MLP与轻量估计器因此是当前更直接的工程选择。

实时性本身不排斥长历史：因果历史都是过去的数据，不需要等未来帧，也不会天然产生
一个窗口长度的等待。真正需要控制的是采样到动作应用的端到端延迟、最坏执行时间、
调度抖动和计算／内存开销。历史较长还可能使模型依赖已经过时的状态，但不是必然。

对于载荷、摩擦、温漂、打滑、未知延迟等隐含或慢变化因素，长时信息仍可能有用。
这种需求可以由递归估计器、滤波器、GRU或Transformer满足，不能仅由“需要记忆”
推出“需要attention”。视觉、接触历史和高层任务上下文也不在本轮结论覆盖范围内。

本轮固定16帧、100 Hz策略、200 Hz物理，额外传输延迟为零且无推扰。
**没有历史长度消融，因此尚未实验证明长历史无效。** 目标200 Hz控制链路和MiniPC
尚未验收；本机推理吞吐不能替代硬件端到端时延。
另外，统一配方用于架构筛选，不代表每种网络都已独立调到最优。
其他仓库的Round4长训或V5闭链混合任务具有不同预算／任务条件，不用于补充成同条件优势证明。

## 代码与证据入口

| 边界 | 主要入口 |
|---|---|
| 网络、历史、时间语义 | `src/transformer_rl/{model,history,timing}.py` |
| PPO、rollout、训练循环 | `src/transformer_rl/{ppo,storage,runner}.py` |
| 独立估计器与辅助优化 | `src/transformer_rl/estimation.py` |
| 环境接口与Isaac示例 | `src/transformer_rl/adapters.py`、`examples/isaaclab_task.py` |
| 评估、稳定性与控制质量 | `src/transformer_rl/{evaluation,stability,control_quality}.py` |
| 检查点与ONNX | `src/transformer_rl/{checkpoint,export}.py` |
| 实验计划与队列 | `src/transformer_rl/{experiments,experiment_cli}.py`、`tools/run_estimator_study.py` |
| 进程控制与回收 | `tools/{trainctl,_managed_study,recover_study}.py` |
| 冻结研究规格 | `configs/`、`docs/protocols/` |

实现与配置保持可复查，历史计划顶部标注终止状态，不改写成已执行的结果。
大工件保留在Git忽略的`artifacts/`中：

- 首轮完整归档：`recovered/architecture-16m-20260915T0540/recovery-complete-20260917T1820Z/`；
- 接线阶段归档：`recovered/estimator-results-20260918T1010Z/`，260个文件已核验；
- 物理指标及分析脚本：`analysis/training-results-20260918/`。

具体机制和复现说明保留在[训练方法](KAISER_ESTIMATOR_RUN.md)、
[训练控制](TRAINING_CONTROL.md)、[估计器设计](ESTIMATOR_TRAINING_PLAN.md)中。
