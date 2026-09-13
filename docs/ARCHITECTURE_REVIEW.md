# Time-aware Transformer 与控制时序架构评审

日期：2026-09-13。接口依据为本仓库 [`INTERFACES.md`](INTERFACES.md)、`config.py`、`types.py` 及评审时可读实现；研究背景与公开来源见文末。本轮仅实现代码、执行定向测试与静态算术核对，没有运行机器人训练、仿真或推理 benchmark。

## 1. 结论与证据边界

**推荐将小型、固定窗口、无 KV cache 的 time-aware Transformer 作为在线 PPO 的可检验候选，同时保留 MLP 和 GRU 对照。**最值得检验的收益是：在有限可观测性、异步传感器、变化的延迟和动力学下，历史信息能否改善鲁棒性。没有证据承诺其在 V40 上比 MLP 更强、更省样本或更快。

- 本仓库是独立 PyTorch 实现，算法与时序组件不依赖 RSL-RL。Transformer 可以直接接受在线 PPO 梯度，teacher 不是必需条件。
- 旧 `isaac_wheeled_rl_deploy` 已由用户明确为 **demo**；不能作为可用部署链路、真实硬件频率、反馈协议、PID 参数或电机工作模式的证据。
- 前研究中的旧训练代码配置仅提供历史比较背景。其 `proprio25`、MLP history5、候选 policy100 Hz / physics200 Hz 都不是本仓库已经连接实机的证明。前研究中的部署建议也不能升级为硬件事实。
- 本仓库默认 `frame30`、左 padding + valid mask、当前命令 query，是新的接口选择；不能与前研究的 `frame25`、repeat-first padding、last-observation head 混称为同一模型或 checkpoint 合约。
- 真实 V40 资产、下位机 PID 和真实通讯链路尚未接入本时序组件，具体待接入范围见第 8 节。

## 2. 与 MLP、复旦速度 encoder、GRU、DT 的取舍

| 路线 | 优势与可检验收益 | 代价与适用边界 | 公平比较要求 |
|---|---|---|---|
| 现有短窗 MLP | 结构简单、低计算量、易导出；短历史已能表达一定有限差分和局部动力学 | 固定 flatten 对更长窗口和非等间隔时间缺少专门归纳偏置；增加窗口通常增加第一层参数 | 先做原输入/动作/奖励合约的 trainer parity；再给 MLP 相同 frame30、时间信息与窗口，区分信息增加和 attention 的效果 |
| 同长度长窗 MLP | 无隐状态陈旧问题，完整输入明确，强而便宜的对照 | 对窗口位置和长度绑定较强；参数匹配不等于算力匹配 | 同时报告同历史、同参数近似、同 wall-clock 三种预算口径 |
| 复旦历史速度估计 MLP | 已核查plane分支：125→128→64→3估计线速度；当前25维＋latent3→128→64→32→6控制网络；速度监督有物理意义 | 三维速度瓶颈未必保留接触/延迟/执行器内部信息；监督域偏差和不可观测滑移不会自动消失 | 参考commit `8204e85`；actor使用`latent.detach()`，encoder由独立速度监督更新。应对比监督目标与表示结构，不能仅归因为网络大小 |
| 有限窗 GRU | 用门控压缩历史；同固定窗从初态重算时，PPO 输入和当前参数语义容易说明 | 窗内计算依赖时间递推；压缩可能遗失细节；每次完整重算也有成本 | 同窗口、同时间特征、同初始态；独立报告吞吐与鲁棒性 |
| Streaming GRU | 每次只处理新帧，隐状态容量不随 episode 时长线性增长，可保留更长记忆 | reset、连续序列 minibatch、BPTT/burn-in 更复杂；旧参数生成的 hidden state 在更新后存在 staleness，detach 不能修复 | 明确是精确重建还是有限 burn-in 近似；不把每个 PPO update 当作 episode reset |
| 固定窗 time-aware Transformer | query 可按当前命令读取不同历史线索；显式时间 age 能区分“相隔一帧”和“相隔多少秒”；无参数化历史缓存 | 完整窗口重复编码，显存与计算明显高于短窗 MLP；短窗、简单任务未必受益；可能学成带滞后的平滑器 | 与同窗 MLP/GRU 使用相同 actor 信息、critic、随机化、奖励及训练预算 |
| 原始 Decision Transformer（DT） | 适合以 desired return / return-to-go 条件化的离线轨迹序列建模，可利用现有高质量数据 | 依赖数据覆盖及回报条件的定义；没有数据质量证据时，不是纯在线 PPO 的直接替代 | 记录离线数据和生成 teacher 的成本。仅使用 Transformer 网络不构成 DT；本仓库 PPO 奖励进入 advantage，不是 return-to-go token |

前研究核查了：Digit 的 Transformer student 主方法包含在线 PPO 与退火 teacher KL；GTrXL 原论文使用 V-MPO；LocoTransformer 官方实现支持直接 PPO，但 attention 主要跨空间/模态。它们支持研究可行性，不能直接证明本仓库的时间 query、V40 观测或无 teacher 训练会获得同样收益。若加入速度辅助监督或 teacher KL，应在各 backbone 上给出同等监督预算，避免把监督优势误归因于 Transformer。

## 3. 默认 input30：每个字段的可得性

`ModelConfig` 只定义维数；`pack_frame()` 负责按顺序拼接与有限值检查，不知道具体机器人关节命名、坐标系或缩放。以下是与 V40 任务意图一致的**待由资产/观测合约绑定的字段解释**：

| 索引（Python slice） | 维数 | 字段 | actor 侧来源与限制 |
|---|---:|---|---|
| `0:3` | 3 | 机身角速度 | 通常由 IMU/状态估计提供；需明确轴向、单位、滤波及时间戳，本轮没有实机验证 |
| `3:6` | 3 | projected gravity | 由姿态估计和坐标变换得到，不是独立、无延迟的真值传感器；确认是否与角速度共享采样时间 |
| `6:10` | 4 | 腿关节相对参考位置 | 编码器读数与已验证 nominal/符号/wrap 合约；不是任意四个资产 DOF |
| `10:16` | 6 | 四腿与两轮关节速度 | 硬件测量或估计输出；滤波/差分 age 可能不同于关节位置，不能无条件共享一个 timestamp |
| `16:19` | 3 | 历史 `command`，意图为目标 vx / yaw rate / height | 指令发生时的软件可得量；目标高度不是实测高度，目标 vx 不是机身真实线速度 |
| `19:25` | 6 | `previous_issued_action` | 软件上一次实际发出的动作可得；必须使用 env 约定的 issued 动作域。不是 ACK、不是 applied target、不是 torque，也不是当前尚未发出的 raw sample |
| `25:27` | 2 | `sensor_age_s` | 每个传感器组在该帧可用时刻的 `availability_time - sample_time`；要求可比较的时钟。默认仅两个组，不能为异步的多个流捏造共同 sample time |
| `27:29` | 2 | `sensor_age_known` | 协议/时钟映射确实支持 age 才置真；未知 age 在 pack 后值为 0 且 known=false，不能将未知延迟编码为已知零延迟 |
| `29:30` | 1 | 实际 `policy_dt_s` | 连续 policy 事件可用时刻之差；由 float64 timestamp 先做差再转 feature dtype。reset 首帧采用显式初始化约定，不能伪称已测得一个前序间隔 |

总计 `16 + 3 + 6 + 2 + 2 + 1 = 30`。`HistoryBatch.times / now` 单独保存 policy 事件时间，`command` 单独提供当前 query 的命令，`valid` 区分真实帧与 padding；这些不是再塞入 input30 的隐藏特征。

**可得性规则：**

1. 真实 body linear velocity、实际 base height、仿真摩擦/载荷、未来随机 delay、真实队列计划到达时间均不因“训练时方便读取”而成为 actor 输入。默认 critic29 的具体组成仍须 env 明确；29 维本身不证明状态充分。
2. `sensor_age_s` 在历史帧构造后保持当时的观测事实；query 时该传感器样本的总 age 可理解为“历史帧 age + 当时 sensor age”（known 时）。传感器时间未知，只能知道接收/可用时间及其不确定性。
3. 两个传感器组可在验证后对应 IMU 与 joints，但这是分组选择，不是硬件事实。若姿态、gyro、q、qdot 的采样/滤波时间不一致，需要更细 schema 或明确近似；不应把 frame30 当作永远足够。
4. `previous_issued_action` 保留软件可知的动作历史。有真实应用反馈时，可另定义带 known flag 和反馈可用时间的 applied-action 扩展；不得偷偷替换第 `19:25` 维语义。

## 4. 时间 query、可观测性与无 KV 的意义

当前模型将历史 frame 投影后加上真实 `now - frame_time` 的固定 Fourier 编码，按因果顺序编码，再附加 `command_projection(current_command) + query_embedding`，从最后的 query 输出动作。历史 token 保留当时的命令，不被当前命令覆盖；历史 token 不反向读取末尾 query。L 个历史 token 加 query，实际 attention 长度为 **L+1**。

这一设计允许：相同本体历史在新命令下被不同地读取；不等间隔、传感器 age 和 policy jitter 可以成为模型条件。它不是显式前向动力学积分器，也没有自动给出 delay 补偿公式。没有时间戳/反馈就不能从 token 序号反推出真实通信时序；纯本体历史下不可辨识的匀速轮滑也不能靠更深 attention 变成可观测。

首版完整窗口重算的理由：

- 原始观测与 issued 动作是已经发生的事实，可跨 PPO update 保留；KV、embedding、GRU hidden 是参数相关表示，不能作为当前参数的精确表示直接复用。
- 即使权重固定，移出最老 token 会改变后续 token 的深层因果上下文；这里 query 时刻改变还会改变历史 token 的 age 编码。简单移除旧 KV、追加新 KV 不等价于完整窗口重算。
- 全窗口快照使 rollout/update/export 的条件分布容易逐 endpoint 核对；代价是重复计算和存储。未来 prefix/gather、SDPA 或 cache 优化必须证明行为等价，不能只证明没有未来泄漏。
- 默认 L16、候选 policy100 Hz 时，16 个等间隔真实样本首尾跨度为 150 ms；L32 为 310 ms。真实跨度以 timestamp 为准，reset 初期有效窗口更短；这些不是推理时延，也不是硬件工作频率。

## 5. env 侧延迟指令通道接口

实现：[`../src/transformer_rl/timing.py`](../src/transformer_rl/timing.py)。目标是有界、可核对的 batched Torch **控制侧目标锁存模型**，不改变 actuator 语义。

```python
from transformer_rl.timing import DelayedCommandChannel, TimingProfile

channel = DelayedCommandChannel(initial_target, capacity=queue_capacity, now_s=start_s)
receipt = channel.submit(issued_target, now_s=issue_s, delay_s=per_env_delay_s)
state = channel.advance(controller_tick_s)
applied_target = state.applied_target

# At an episode boundary, with a full [N, A] initial target and bool [N] mask:
state = channel.reset(done_mask, initial_target, now_s=reset_s)
```

### 5.1 调用与时间语义

- `initial_target / issued_target` 为 `[N,A]` 浮点 tensor，shape/device/dtype 一致且值有限。组件不 clip、不做六动作 decoder、不转换腿位置/轮速度/力矩单位；这些合约由 env/已验证控制层负责。
- `now_s / delay_s` 接受标量秒或同 device 的浮点 `[N]` tensor；内部用 float64。每次提交的 delay 可按 env、按包变化，非负且有限。应在生产者侧就生成 float64 绝对时间，事后转换不能恢复 float32 丢掉的毫秒精度。
- `submit()` 同时向每个 env 提交一个完整 target。返回 `CommandSubmission(sequence, issued_at_s, scheduled_at_s)`，其中计划到达时间为 `issue + delay`。返回值只证明此模拟队列接受了提交，**不表示 transport ACK，更不表示 applied**。
- `advance(now_s)` 选择所有已经到达的包中**序号最大且比已应用序号新**的目标；没有新到达目标则保持旧值。较旧包晚到不会回滚控制目标。此为明确选择的 **latest-sequence-wins** 模型，并非所有硬件 FIFO 的共同语义。
- 未到达旧包仍占容量；已到达但未胜出的包消费后计入 `superseded_count`。一次 advance 不依次向 actuator 施加所有中间包。reset 是另一种显式清队列事件。
- 零延迟也需在同一时刻调用 `advance()` 才生效。到达时刻与控制器采样时刻分离：例如 delay=2.5 ms、控制器在 5 ms 才 advance，`scheduled_at_s=2.5 ms`，`applied_at_s=5 ms`。通道不将延迟量化为 policy ticks，也不会把实际生效时间回填成更早的到达时刻。
- env 必须按物理/控制事件的时间顺序调用。若只每个 policy tick advance，就建模成只在 policy tick 更新目标；不能据此声称已有 substep delay fidelity。相同边界处“旧包处理、发新包、控制反馈”的顺序应由 env 固定并测试。
- 所有操作的时间按 env **在通道整个生命周期内单调不减**。reset 只检查选中行、只更新选中行的时间；若 observation 使用 episode-relative timestamp，env 要保留独立的单调通道时钟。
- sequence 从 0 开始、按 env 单调分配，reset 不复用旧 sequence。reset 清选中行全部 pending 包并设置其 initial target；其他行连同目标、队列、统计和时钟均保持不变。

### 5.2 有界容量、所有权与设备

`capacity` 必须由调用者显式指定，是每个 env 的 in-flight slot 上限。任意一行满时，`submit()` 抛 `BufferError`，**整批拒绝且 sequence、时钟、目标、队列都不变**；不靠静默覆盖旧包“腾空间”。已到期但尚未 advance 的包仍占 slot；应显式处理到期事件，再重试，或在配置阶段调整容量。该错误不是硬件丢包模型。

所有输入写入内部 owned storage；返回 dataclass 中的 tensor 也是独立快照、无 autograd graph。修改原输入或返回值不会改通道状态。队列选择完全 tensorized，没有 per-env Python loop 和整批 `.cpu()` / `.tolist()`。有限值、单调性和溢出检查有标量 reduction 导致的 CUDA 同步；这是易审计参考实现，不是已优化或已测 GPU 吞吐的路径。

### 5.3 `AppliedCommandState` 元数据

| 字段 | shape / dtype | 含义 |
|---|---|---|
| `applied_target` | `[N,A]`，原 target dtype | 当前被模拟控制侧锁存的目标；不是实际 motor position/velocity/torque |
| `applied_sequence` | `[N]` int64 | 选中指令序号；`-1` 表示 initial/reset target |
| `issued_at_s` | `[N]` float64 | 当前选中包的发出时刻；initial/reset 为 NaN |
| `scheduled_at_s` | `[N]` float64 | 当前选中包在模拟传输模型中的计划到达时刻；initial/reset 为 NaN |
| `applied_at_s` | `[N]` float64 | 实际锁存的 advance 时刻，或设置 initial target 的构造/reset 时刻 |
| `now_s` | `[N]` float64 | 各 env 最近一次成功操作的通道时钟 |
| `pending_count` | `[N]` int64 | 尚未消费的包数 |
| `next_scheduled_at_s` | `[N]` float64 | pending 包中最早的计划到达时刻；空队列为 `+inf`，未 advance 时可早于 now |
| `superseded_count` | `[N]` int64 | 本 episode 已到达但未成为当次新目标的包数；partial reset 只清选中行 |

env 可将快照放进 `StepResult.info["command_channel"]` 供诊断，或通过版本化 critic schema 提供所需特征。NaN/inf 是诊断哨兵，**不能原样拼入要求有限值的 critic/actor**；转换为 age 时须有 known/valid mask 和明确定义的初始化值。

真实硬件即使报告 ACK，也需区分“主机入队、下位机收到、目标寄存器更新、控制周期使用、机械响应”。只有真实提供的应用反馈及其反馈可用时间，才支持对应的 actor 特征；仿真中采样的未来计划到达时间不自动可供 actor 读取。默认 input30 不因本组件存在而新增 privileged 信息。

默认 critic29 不含完整在途队列、随机执行器隐状态或 policy history；相同 critic29 可能对应不同后续演化。上述诊断摘要也不等价于完整队列状态。它可作为非充分状态上的 value approximation，不能称为精确 Markov critic；若这一缺失影响 value 拟合，独立比较扩展 critic 或 history-conditioned critic，并保持各 actor 对照信息预算一致。

### 5.4 `TimingProfile` 只验证可表达的同步时钟

`TimingProfile(policy_period_s, controller_period_s, physics_period_s)` 无硬件频率默认值，验证：

- 三个 period 为有限正数；`controller_period_s >= physics_period_s`。
- `controller / physics` 与 `policy / controller` 均为正整数比（仅容忍 `1e-9` 的浮点比值误差）。因此本 profile 选择的是嵌套、整数 substeps 的同步调度。
- `controller_substeps` 表示每个 controller tick 间的 physics 步数；`controllers_per_policy` 表示每个 policy tick 间的 controller tick 数；`policy_substeps` 是两者之积。

例如 `(0.010, 0.005, 0.005)` 可表达；`(0.010, 0.001, 0.005)` 被拒绝。在 5 ms 同一份 q/qdot 上重复计算五次，不能冒充有新反馈的 1 kHz controller。若真实硬件内部确为 1 kHz，应由经验证的更细 physics 或执行器内部状态积分模型表达，并验证耦合时序；本类不提供该模型，也不将未整除的异步周期偷偷四舍五入。delay 不需要是任何 period 的整数倍。

## 6. 正确性验收：alias、时间、PPO 与控制边界

以下是跨模块集成验收要求；本轮只执行 `tests/test_timing.py`，其他模块的通过状态以主集成报告为准。

1. **Tensor alias / episode 隔离**：env 下一步 in-place 更新、调用者修改 receipt/state、history ring 滚动，都不改变已存 rollout；partial reset 不动其他 env，不泄露旧 episode 的历史/队列；同 timestamp 重复 history query 幂等或明确报错；PPO update 边界不清 history。
2. **时序参考轨迹**：零 delay、半 controller period、不同 env 的 lag/时钟、同时到达、乱序到达、迟到旧包、独立 slot 复用、partial reset、队列溢出、负 delay/非有限数、倒退时间、大 uptime 的毫秒精度均应有手算断言。overflow/非法输入必须在任何状态变更前失败。
3. **采样率与时间 aliasing**：时间特征只能标记 age/jitter，不能重建未采到的高频运动。区分 policy observation、传感器采样/滤波、controller 和 physics 的频率；改变 substeps 不应只是重复读取同 state。评估抖动、掉包、滤波群延迟时，记录有效时间分布与动作频谱。
4. **因果与 privileged 隔离**：未来帧、未来奖励、当前 raw action、critic 真值、未来 channel schedule 不进入当前 actor；改变 query 命令不应改历史 token 的既有因果表示。padding 不可产生全 `-inf` softmax 的 NaN，首次 reset 后的短有效窗口须可计算。
5. **PPO likelihood 一致**：rollout 固定 behavior 权重和 preprocessing，保存当时 noisy frame/timestamps/valid/current query；raw Gaussian sample 用于 log-prob，issued action 用于实际命令历史，applied target 只影响 env transition。首次更新前重算 old log-prob，ratio≈1、analytic KL(old||new)≈0；dropout=0，old statistics/targets 不在 epoch 内刷新。
6. **窗口与预算一致**：每个 endpoint 只计算一个策略 loss；shuffle endpoint 不能打乱窗内顺序，也不能把跨 env 的相邻内存当时间。未来省存储的 prefix/gather 需逐 endpoint 等价，不能将整条 rollout 的长上下文冒充采样时固定窗口。
7. **GAE 两个 mask**：`delta = r + gamma * (1-terminated) * V(final_or_next) - V(old)`；trace 用 `1-(terminated|truncated)`。真正 terminal 不 bootstrap；timeout 用 reset 前 final critic 值但不串下个 episode；两者同时为真时 terminal 优先。rollout 切片末尾只截 trace，仍 bootstrap。returns 必须在 advantage 标准化前形成，不能把非有限终态先送 critic 再乘零。
8. **真实秒与折扣语义**：目前 PPO 配置的 gamma/lambda 是每 transition 的标量。将 policy_dt 加入 actor 不会自动把 GAE 变成连续时间折扣。若未来显著随机化 policy_dt，需明确保留“按事件折扣”，还是采用 `gamma_t = gamma_ref ** (dt_t / dt_ref)` 等显式定义，并一起处理 trace 衰减、奖励时间积分和 rollout next-value；本次不改该算法合约。
9. **actuator / 导出一致**：channel 输出目标不等于执行器输出；decoder 顺序、限幅、模式、反馈刷新时刻与 checkpoint 合约必须一致。batch1 导出回放比较 mean，验证冷/热启动、partial reset 与 command 切换；模型 forward p99 和含拼窗/通信的端到端 p99 分别测量。

未来算法验收至少比较原 MLP parity、同窗 MLP、有限窗 GRU、不同历史长度的 Transformer；对照使用相同 frame、时间特征、critic、奖励/随机化、动作语义、teacher/监督预算。配对 seed，建议至少三个独立训练 seed 起步，同时报告同 unique transitions 与同 wall-clock/GPU-hours 的结果；记录速度/转向/高度误差、失败率、恢复时间、动作变化/频谱、限幅与控制时延，不能只看 return 最好的一次。此处是后续验收定义，本轮未开展这些实验。

## 7. 单卡 RTX 4090：计算与内存仅为估算

**以下是对当前默认结构的静态算术，不是 RTX 4090 实测，不给出 FPS、训练时长、收敛概率或部署时延承诺。**默认配置 `F30 / d64 / 2 layers / 4 heads / FFN128`，frame projection 与当前命令 projection 分开，输出为直接 `Linear(64,6)`，包含 learnable query 和 Gaussian log_std。该结构与前研究带额外 head MLP 的参数表不同。

令 `S=L+1`，每 block 参数为 `8d²+11d`；主要 MAC 为 `4Sd² + 2Sd*ffn_dim + 2S²d`。加上 `L*F*d + command_dim*d + d*action_dim`，得到：

| 项目 | L16 | L32 |
|---|---:|---:|
| actor 可训练参数 | 69,772 | 69,772 |
| batch1 完整窗口主要 MAC | 1,219,392 | 2,503,488 |
| `N2048,T48` 的 FP32 完整 frame 历史快照 | 180 MiB | 360 MiB |
| 相应 float64 历史时间戳 | 12 MiB | 24 MiB |
| 相应 bool valid mask | 1.5 MiB | 3 MiB |
| `B=2048*48/4=24,576` 的单层 FP32 QKV tensor | 306 MiB | 594 MiB |
| 同一 B、4 heads 的**单份** attention score tensor | 108.375 MiB | 408.375 MiB |

MAC 仅包括线性层与 attention 两次矩阵乘，不含 LN/GELU/softmax/mask、内存与调度。历史 MLP5 的 `125→256→128→64→6` 同口径约 73,344 MAC；TF-L16 的约 16.6 倍 MAC **不等于** 16.6 倍端到端耗时，也不是公平同输入信息比较。

默认 critic29 参数 48,897；actor+critic FP32 权重约 0.453 MiB，粗计权重、梯度、Adam 两份状态约 1.81 MiB，尚不含 allocator/optimizer 临时张量。小权重不等于低总显存：

- 当前 reference storage 保存完整历史，`finish()` stack 时原快照与新 batch 可能同时存在；还需 critic/actions/old statistics、query、minibatch gather 等空间，不能将表中单份 input 当作总 storage。
- 当前显式 attention 会有 score、softmax weights 和反向所需中间量；上表只列单份单层 tensor，完整两层 autograd 峰值更高。大的 PPO minibatch 可能比 actor 权重昂贵几个数量级。
- 模拟器、接触/资产、GPU scratch、PyTorch reserved memory 与训练相互竞争；尚无真实 V40 资产/仿真总显存测量。24 GB 4090 是容量预算背景，不是“任意 N 都可训练”的保证。
- 后续若内存受限，可先减少候选 N/T 或采用保持有效 optimizer batch 的 microbatch accumulation；当前接口未因此自动实现 accumulation。修改 minibatch 个数导致更多 optimizer steps，不应宣称与原 PPO 配置等价。
- 本时序队列主存储约 `capacity*N*(A*target_element_size + 24)` 字节，24 来自 int64 sequence 与两个 float64 时间；另有 O(N*A) 的锁存/快照及临时选择 tensor。它也没有 GPU 吞吐实测。

真正的 4090 评估需分别记录 collect/update/wall-time、unique policy transitions/s、physics substeps/s、peak allocated/reserved VRAM、达到固定指标的 env steps 与 GPU-hours；compile/warm-up 单列。部署在目标机器上另测 batch1、p50/p95/p99、长期 jitter 与端到端路径。推理算术小不保证小 batch 的调度延迟，也不能据此承诺比 MLP 强。

## 8. 真实 V40 资产与 PID 尚未接入的范围

当前时序模块不加载 USD/URDF，不读取真实机器人，不连接通讯，也不实现 PID。`TimingProfile` 是调度约束，`DelayedCommandChannel` 是目标延迟/选择模型；二者都不构成已验证的 V40 environment。

后续接入需要真实、可追踪的资产与控制合约：关节命名/顺序/方向/零位、运动与惯性/碰撞参数、限位及 nominal、腿位置与轮速度目标映射、电机控制模式/饱和/速率限制、消息序号/乱序/FIFO 或 overwrite 规则、真实可得反馈和时钟映射。不得从 demo 或旧 checkpoint 的维度相等推导这些事实。

PID/PD 的实际实现、增益、是否有 I 项、积分状态/饱和行为、真实内部更新周期目前未核实；**本次不发明 I 项、参数或扭矩公式**。若后续有经验证控制器，只在约定控制时刻交付 target 并使用约定的新反馈，其实际力矩/电流输出属于该控制器模型。真实 ACK 的含义、是否有目标应用确认、主机与 MCU 时钟是否可比较，均需原协议或实测支持。

精确 episode/恢复也属于集成边界：reset 除清 history/channel 外，还需按真实控制器合约重置其状态；保存 actor 权重不能恢复未保存的仿真、队列和执行器隐状态。不能只恢复旧历史而宣称轨迹精确续接。

## 9. 本轮定向验证

命令（仓库根目录）：

```bash
/tmp/opencode/v40-sim60-ci/bin/python -m pytest tests/test_timing.py -q
```

覆盖零 delay、5 ms 延迟、半 5 ms controller period 的到达/应用分离、多 env 独立 lag/clock、乱序与 simultaneous/latest-wins、旧包迟到不回滚、独立空 slot 复用、partial/empty reset、序号不复用、单 env 满导致整批原子溢出、倒退时间/非有限数拒绝、输入及返回快照 alias 隔离、float64 大 uptime、整数 substeps 与假 1 kHz 拒绝。测试含 CUDA 参数化；无可用 CUDA 时显式 skip，不将 CPU 通过写成 GPU 通过。

本轮结果：**33 passed, 21 skipped，0.71 s**。21 个跳过项均为 CUDA 参数化测试，当前测试进程 `torch.cuda.is_available()` 为 false；CPU 定向测试通过，CUDA 执行未验证。其他模块的测试与整体集成结果由主集成报告汇总。

上述数字是时序模块最初的CPU验证记录，后续全仓CPU检查及补充CUDA检查见[验证记录](VALIDATION.md)。

## 10. 公开研究来源

- [复旦plane配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py)与[历史编码器](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py)：仅作为已核查基线的引用，不作为本仓库依赖。
- [Learning Humanoid Locomotion with Transformers](https://arxiv.org/abs/2303.03381v1)：教师辅助的Transformer在线RL。
- [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)：GTrXL与V-MPO；并非本仓库的固定窗PPO实现。
- [Decision Transformer](https://arxiv.org/abs/2106.01345)与[Online Decision Transformer](https://arxiv.org/abs/2202.05607)：回报条件化轨迹学习路线，区别于本仓库的在线PPO。
