# 100 Hz 运控：时序策略与奖励参考

## 范围与结论

用户于 2026-10-18 明确保留 **100 Hz 策略、200 Hz 物理**，主要参考正式 MLP v7 的 reward 等设计。本次为源码和文献审查，不迁移 v7 的 50 Hz 配方、78D critic、训练预算、机械资产或运行状态。现有 35D actor、81D critic 和冻结研究队列不变。

对当前单一六动作轮腿底盘，优先验证小型、固定窗口的因果时序 Transformer，历史编码结果与当前帧拼接，再由 MLP 输出动作。门控残差和当前帧 query readout 是需要分别验证的选项。这个优先级是结合任务约束作出的工程判断，不是已经证明的架构排名。

普通 MLP 和 history MLP 继续作为完整对照；保留原 34 候选、三训练 seed、H1/11/31/61、完整 LR 和连续训练/遗忘研究范围。本文不新增已调度单元，不缩小原范围，不重跑已开始的格子，也不将旧结果改标成 v7 实验。

## 文献证据与适用边界

| 方法 | 已发表的方法或开源内容 | 对本任务的启示与限制 |
|---|---|---|
| 因果时序 Transformer | [Real-World Humanoid Locomotion with Reinforcement Learning](https://arxiv.org/html/2303.03381)：用观测与过去动作组成时序 token，并有实际 Digit 运控部署。该版本采用 teacher 监督结合 PPO。 | 支持“历史有助于处理部分可观测动力学”的方向；不能据此断言从零 PPO 训练的小 Transformer 必然胜过 MLP。 |
| GTrXL | [ICML 2020 论文](https://proceedings.mlr.press/v119/parisotto20a.html)：调整归一化和门控改善 RL 优化稳定性，主要证据来自记忆任务。 | 借鉴 Pre-LN、偏向恒等映射的残差门控；当前固定窗口实现不引入 XL 的跨段记忆，不能称完整 GTrXL 复现。 |
| Body Transformer | [论文](https://proceedings.mlr.press/v270/sferrazza25a.html)、[作者代码](https://github.com/carlosferrazza/BodyTransformer)：按身体传感器/执行器图组织 token，通过邻接 attention mask 传播信息，有在线 RL 和 A1 实机路径。 | 是明确值得研究的空间结构方向；空间图不能替代时间历史。当前动作少、身体固定，新增拆分与图映射成本是否值得仍需实测。 |
| LocoFormer | [CoRL 2025 论文](https://arxiv.org/html/2509.23745v1)：Transformer-XL、跨 trial 记忆、多机型大规模随机化与 PPO，也覆盖轮式机器人。 | 长时间适应与跨机型研究有参考价值；跨段隐藏状态超出当前无时间递归、固定窗口范围，不能只复制网络宽度便声称复现。 |
| Decision Transformer | [原论文](https://arxiv.org/abs/2106.01345)、[作者代码](https://github.com/kzl/decision-transformer)：以 return-to-go、状态、动作作条件的离线序列建模。 | 与当前在线 PPO 的问题和训练过程不同，优先级较低；并不意味着它不能用于任何运控问题。 |

开源实现的使用边界：

- Body Transformer 的作者仓库同时提供 `reinforcement_learning` 和 `a1_walk`，适合参考身体图和实际机器人接线方式。
- [episodic-transformer-memory-ppo](https://github.com/MarcoMeter/episodic-transformer-memory-ppo) 是其作者提供的 TrXL/GTrXL 与 PPO 参考实现，但仓库的动作接口是离散或多离散；不能当作六维连续机器人策略的直接运行包。
- [lucidrains/locoformer](https://github.com/lucidrains/locoformer) 是第三方实现，不能称原论文作者的正式代码或等价复现。本次没有运行这些仓库。

## 对当前实现的具体判断

已有 `FramePolicy` 的主路径符合小型时序策略方向：

```text
35D observation history -> projection + position -> causal attention -> latent
current 35D frame ------------------------------------------------------+
                                      concatenate -> ELU MLP -> 6 means
```

critic、Gaussian 探索、PPO、历史缓冲区和执行器处理保持独立。当前帧直连是一条直接使用最新信息的计算路径，不是额外物理控制器，也不能保证系统闭环稳定。

现有 `transformer_gated` 使用 128D、2 层、4 头、512D FFN 和 last readout；`transformer_query` 使用相同主干规格、普通残差和当前帧 query readout。两者可以分别研究门控和读出方式，不能将二者的单独局部优势自动相加成“组合必胜”。门控中的 GRU 形式沿网络深度作用，不是跨时间隐藏状态，符合当前无 RNN 的约束。

推荐的起始规格为 96–128D、2 层、4 头，使用当前帧相对位置；它是实验起点，不是论文给出的本机最优配置。H11、H31、H61 在 100 Hz 下最早到最新观测的跨度分别为 0.10、0.30、0.60 s。H1 保留为无历史对照。

先用固定窗口明确重算语义，不默认启用流式 KV-cache。多层因果模型的缓存可能包含窗口外上下文；窗口滑动、位置编码、episode reset、PPO 参数更新后的缓存失效都需要单独证明等价。PPO 应针对保存的原始观测窗口重算当前参数的 log probability，不能将旧参数的隐藏表示冒充当前 encoder 输出。

35D 中已有上一条 policy action。它是上一条已发出的命令，不是传输到达、控制器实际应用或实际力矩。不得重复添加未来动作或把这些不同信号混用。更长历史也不会自动创造可观测的绝对世界速度；纯编码器/IMU 在滑移等条件下仍有信息边界。

## 正式 MLP v7 的奖励参考

只读源码基准为 [isaac_wheeled_rl_train main](https://github.com/Yukikaze2233/isaac_wheeled_rl_train/tree/cdd6339548ef2d86ff850f04305b3a1343e47eae)，commit `cdd6339548ef2d86ff850f04305b3a1343e47eae`。本次没有读取另一个聊天的 native 进程、服务或正式 checkpoint。

审查 `configs/v7_formal_20261012/mixed_tracking_repair_063.json`，并沿 `env.py`、`scut_reward_selection.py`、`scut_flat_reward_runtime.py`、`scut_flat_reward.py` 与 `tracking_reward_repair.py` 核对分发。该配方选择 `scut0730_rough_v1`，部分平地 rehearsal 使用 `scut0719_flat_v1`，并启用 `v7_exact_command_tracking_v1`。这不是把开发分支中所有 reward 都一起打开；未选中的 `v7_final_rough_v1` 及其 31 项权重不能当成该配方的实际付款。

最值得借鉴的是目标语义和分项归因：

1. 最新 tracking repair 使用 actor 可见的原始 vx/yaw 指令，取消原奖励的 0.1 指令死区；standing gate 改为 vx 恰为零。yaw 非零时不应被奖励端悄悄当作零。
2. 前向速度平方误差项权重由 −0.1 改为 −2.0。该项的原始量为 `(0.25 * e_v)^2`，所以不能把 −2.0 当成直接乘 `e_v²`。其中 `e_v = command_vx - body_com_vx * cos(pitch)`，也不是世界 XY 漂移误差。
3. 高度同时有窄域指数项与未截断平方代价：`exp(-clamp(e_h, -.15, .15)^2 / .001)` 和 `-(10 * e_h)^2`。指数项饱和后，平方项仍区分大偏差。
4. posture、静止速度、undesired contact、实际力矩、轮功率和关节速度/加速度分别收费；不能只看总 reward 或动作是否平滑。
5. 该实现的纯站立惩罚使用 body 水平速度的 L1 范数，没有由此定义世界 XY 位置锚点或完整静止资格。

平地速度跟踪还乘高度 gate：高度误差不超过 50 mm 时为 1，到 100 mm 线性降至 0；rough 分支不使用该 gate。height datum 也有区别：平地使用 root link z 相对环境原点，rough 使用 base ground scan；当前研究的 wheel-support 相对高度不能只换名称后直接代入。旧研究约 111 mm 的高度 RMSE 不是正式 v7 MLP 的结果，不能用来给当前 v7 下结论。

### 保留 100 Hz 时应如何改写

以下是供新配方验证的设计，不是已部署的奖励变更：

| 项目 | 100 Hz 处理原则 | 必须保留的区别 |
|---|---|---|
| 连续状态、速度、姿态、力矩/功率代价 | 先定义 reward/s 密度，统一乘一次 `dt = 0.01 s` | 不在调用端再乘 dt；不直接照搬 50 Hz 的单步 reward |
| 动作一阶/二阶差分 | 若保留原 20 ms 上的导数正则强度，系数分别乘 `(0.02/dt)^2` 和 `(0.02/dt)^4` | 在 100 Hz 分别为 4 和 16；不等于所有 reward 权重翻倍 |
| 终止事件 | 先决定要保持每次失败的事件成本还是源的每步系数 | 源 `−200 * 0.02 = −4`/事件；直接改 dt 会变成 −2。选择保持 −4 是新设计，不能称源实现逐字等价 |
| 接触历史 | 按物理时间维护 3 个 5 ms 接触样本 | 不是 3 个策略 tick；不能用轮接触 bool 替代有害部件的接触力 |
| 终止确认、command hold、recovery 等持续时间 | 由秒换算 tick，并验证真实 runtime | 不能将原 50 Hz 的 tick 数不加说明地复制到 100 Hz |
| episode/reset 与 scripted interval | 明确终止/超时和可学习区间 | 源 scripted/hold 区间排除 PPO；当前 collector 未验证相同语义，不移植脚本恢复后仍把动作算成 policy rollout |

上面的差分换算只针对同一平滑连续动作曲线的离散近似。采样、噪声、零阶保持、clip 前后动作历史和优化分布都可能变化，不能承诺严格频率不变。正式参考使用 clip 前的 raw actor history；旧研究的 normalized clipped action 正则不是同一个物理定义。

将来接入新的 reward profile 时，要先核验完整 PRE-reset 物理输入、COM/link 参考点、joint/action 排序、接触部件及传感器、实际施加力矩与加速度；缺失字段不补零、不从已有误差 CSV 伪造。当前数据投影不足以离线重算完整 v7 奖励。

## 下一轮研究应如何判断收益

reward 配方和网络结构是两个实验变量，先分清再做交叉对照。每种架构采用相同版本的 reward、任务采样、物理/控制时序和预算，才能解释网络差异。文献中的 teacher 辅助或其他训练流程不能悄悄只加给 Transformer 一组。

保留原完整范围；在其中优先分析普通残差、门控、query 读出，以及 H1/11/31/61 的结果。至少三个独立训练 seed，开发评估 seed 不代替训练重复；报告高度/速度/姿态的 bias、RMSE、段内去均值波动、世界 XY 静止漂移、接触、失败、饱和、实际 torque 变化，并列出真实训练和部署成本。

100 Hz 给整个策略链路 10 ms 周期，部署资格需要在目标硬件实测单环境端到端 p50/p95/p99、超期比例及内存，包括数据准备、历史维护、推理和输出；不以 GPU 大 batch 吞吐量推算。本文未给出任何新架构的硬实时资格或架构胜出结论。

本次验证范围为配置解析、源码调用链与论文/仓库核对；没有执行模型、单元、SDK 或仿真测试，没有修改生产源码或活跃冻结 release。原队列继续使用原已认证配方，新的 reward 或架构提案均未自动调度。
