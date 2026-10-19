# 优化与动作分布的观测口径

这些统计用于新的冻结源码。正在运行的历史任务继续使用各自封存源码；不能把新增字段回填为旧运行的实测值。结构、课程、初始探索、实际优化量和闭环控制表现应分别记录。

## PPO 参数变化

`PPOTrainer.update(..., diagnostics=True)` 在成功应用的每个优化步记录统计；KL 拒绝的 minibatch 不进入分母。`diagnostics=False` 不创建参数快照或额外梯度观测。

| 字段 | 定义和分母 |
|---|---|
| `actor_grad_norm`、`critic_grad_norm` | 优化器实际优化参数中，两网络各自独占参数的裁剪前 L2 梯度范数；成功步等权平均 |
| `shared_grad_norm`、`other_grad_norm` | 交叠参数只进入 shared；其余优化参数进入 other，不重复归因 |
| `grad_clip_coef` | 原全局裁剪实际系数 `min(1, max_grad_norm / (grad_norm + 1e-6))` 的步平均，保留原运算 dtype |
| `grad_clip_step_fraction` | 系数小于 1 的成功步数 / 成功步数，包含阈值相等时 epsilon 导致的微小裁剪 |
| `[group_]param_update_l2` | 每个成功 `optimizer.step()` 前后真实参数差的 L2，成功步等权平均 |
| `[group_]param_update_relative_l2` | 上述差 / 该步之前参数的 L2；只平均分母大于零的步 |
| `[group_]param_update_relative_step_count` | 相对变化的实际有效步数；零范数不填成零变化 |
| `[group_]param_count` | 对应优化参数组的元素数，去重且排除冻结、未优化及 detached estimator 参数 |

`group` 为 actor、critic、shared、other；无前缀为这些优化参数的并集。没有成功优化步时连续统计为 null、有效步数为 0。空参数组在已应用步中的绝对范数为 0，相对范数不可定义。全局裁剪仍沿用原 policy 参数范围；外部替换优化器子集时，组梯度不一定构成该全局裁剪范数的完整分解。

梯度来自带系数的总 loss，actor 可包含熵、辅助监督及保留项，critic 包含 value loss 系数；这不是各项独立 loss 的梯度干扰测量。参数坐标的绝对梯度和变化不能直接比较不同宽度、参数化的模型，需同时看参数规模、相对变化、固定 rollout 的 KL 与闭环控制。真实 Adam 更新不能用学习率乘梯度近似。损失、KL 和 PPO likelihood-ratio `clip_fraction` 继续按成功优化的样本使用次数加权，与这里的步平均不同。

## 采样动作与发出动作

`FrameCollector` 从实际返回的 rollout 读取 raw action、issued action 和行为高斯均值，使用训练控制合同声明的各通道 bounds。完整及正常返回的部分 rollout 都统计；终止端点和 reset 后采样不删除。失败而未返回有效 batch 的采集不制造动作统计，空采集也不沿用上一次数据。

- `action_sample_count` 是每个通道的实际端点数。
- `raw_action_outside_count_i` 和 `raw_action_clip_fraction_i` 统计 `abs(raw_i) > bound_i`。刚好在边界的 raw action 未被截断。
- `action_mean_outside_count_i` 和 `action_mean_outside_fraction_i` 描述行为高斯均值是否越界，与随机采样越界分开。
- `issued_action_at_bound_count_i` 和 `issued_action_at_bound_fraction_i` 包含正好落在边界的动作，与实际截断比例不同。
- `raw_action_{mean,rms,std}_i`、`issued_action_{mean,rms,std}_i` 是该 rollout 的通道分布矩，std 使用总体分母。它们不是去均值后的回合内抖动，也不代表力矩或物理速度。

通道顺序来自 `control.action_names`。动作的到达、应用和最终动态力矩包络另由环境及控制评估记录；这里不能评定电机饱和。均值、RMS、std 使用 actor 输出坐标，比例无量纲。无端点的直接统计调用返回 null 矩和比例、计数 0。

`collection.action_statistics_elapsed_s` 包括动作统计和标量回读时间，`collection.elapsed_s` 包含这段开销。既有 logger 自动将数值写入 `ppo/*`、`rollout/*`；null 保留在 JSON，不能写成零。端到端训练耗时还包含优化、日志、checkpoint 等，采集耗时不能代替总训练耗时。

单元验证使用小型真实 CPU tensor、SGD/Adam、完整状态及 RNG 对照，不授予真实机器人、GPU 性能、教师资格或 sim2real 结论。新观测不修改 loss、全局裁剪算法、采样策略、动作约束或当前远端训练。
