# 奖励目标与运动控制指标的对齐

奖励设计会影响策略学到的稳态，但控制指标仍需独立记录。奖励权重、误差尺度、启用条件、任务采样、观测信息和优化过程共同影响最终行为。观察到偏差或漂移，不能单独证明奖励缺项，也不能直接归因于网络结构。

本文分析已封存的 Gated H31 课程对照配方，区别于普通 MLP 的原生 v6 训练；两者的动力学、控制频率和训练预算不能直接混为架构对照。以下记录不改变运行中的奖励、课程或选择规则。

## 如何区分准确性与平滑性

在一个固定指令、单环境、单回合的统计窗口内，令误差为 $e_t$。使用总体二阶矩时：

$$
\operatorname{RMSE}^2 = \bar e^2 + \frac{1}{T}\sum_t(e_t-\bar e)^2.
$$

去均值后的波动只反映第二项。不同窗口聚合时，还需处理窗口均值之间的差异和样本权重，不能把跨回合均值差当作抖动。高度波动小，可能只是稳定地停在错误高度。

对多个有效保持段，令第 $g$ 段含 $n_g$ 个样本、均值为 $\mu_g$、总体方差为 $\sigma_g^2$，且 $N=\sum_g n_g$、$\mu=\sum_g n_g\mu_g/N$。按样本数加权的分解为：

$$
\operatorname{RMSE}^2
=\mu^2+\frac{1}{N}\sum_g n_g\sigma_g^2
+\frac{1}{N}\sum_g n_g(\mu_g-\mu)^2.
$$

当前控制统计的 `within_group_std` 对应第二项的平方根；`group_mean_std` 则对各段均值等权计算标准差，采用 $G^{-1}\sum_g(\mu_g-\bar\mu_{\mathrm{groups}})^2$，其中 $\bar\mu_{\mathrm{groups}}=G^{-1}\sum_g\mu_g$。段长不同时，后者不等于上述第三项，不能直接声称 `RMSE² = bias² + within_group_std² + group_mean_std²`。例如三个零误差样本组成一段，一个误差为 4 的样本组成另一段：总体 bias 为 1、RMSE 为 2、段内波动为 0；按样本加权的段间方差为 3，等权段间方差为 4。

历史报告没有保存逐段均值和长度时，可以从 `RMSE² − bias² − within_group_std²` 推导按样本加权的段间方差，并明确标为派生值；需要考虑浮点舍入，不能将它冒充原报告的 `group_mean_std`。比较检查点时同时列出有效样本数、保持段数、剔除的短段和失败回合。采用同一稳态筛选规则，不代表保留下来的样本群体相同。

静止运动同时记录世界 XY 速度范数、RMS、净位移、最大偏离和角速度。有符号平均速度可能相互抵消，净位移也可能掩盖持续运动。世界 XY 指标以 base_link 原点的位置变化定义；不将它未经核验地解释为整机质心速度。

## 实际使用的奖励

原课程 manifest 的 canonical SHA 为 `917e35a0a85e16f2212a3ceb1775911a70898a4959610fe55251d920e18c7b28`，learner 源来自 commit `7491e9f`。父环境 snapshot 为 `174bc6fb53bef34198139ff480f72b4b6f6bac23256efbbb5e17a8aea61357ab`，课程生成的 snapshot 为 `c939b38a649e048d071369c1edccac958445eb7355024ecdf3d73a64c4c46331`。

已用原生成器重建小配置并逐项匹配 manifest：mixed phase2 canonical SHA `95a9eb6c9928ac9ab9053336e1eb8175cd0533b315ed410bfa8c71623d82220e`，配置字节 SHA `eeba42a2620608c874cc84d8ed92508ef6d52c7fa5c89d9d8f0369ff7ec8ccca`。公式对应 snapshot 中 `src/wheeled_tasks/chassis/rewards.py` 的 SHA `fad8d22be6af7213a8fb76145a86d13b48ceb06bcafa1f5e3519b40fecc29c4e`，环境调用对应 `env.py` 的 SHA `6e4dbe17f6a951a833e98906b4901ad070b1888257e608c7bf3da3f7507ca68b`。封存评估 receipt 绑定了同一 manifest、原 worker 和 checkpoint，不能用当前开发分支替代这份配方解释历史结果。

实际启用的 `DENSE_TRACKING_V1` 将跟踪误差转换为宽域代价和精细代价之和：

$$
c(e)=-w_b\left(\sqrt{1+(e/s_b)^2}-1\right)
+w_f\left(\exp[-(e/s_f)^2]-1\right).
$$

宽域项不截断；精细项在大误差时饱和，但宽域项继续区分大误差。连续项返回奖励每秒的密度，环境统一乘策略步长 $\Delta t=0.01\,\mathrm{s}$。离散成功、失败事件另计，不乘步长。

| 误差项 | 宽域尺度 $s_b$ | 权重 $w_b$ | 精细尺度 $s_f$ | 权重 $w_f$ |
|---|---:|---:|---:|---:|
| 高度（m） | 0.04 | 3.0 | 0.015 | 1.5 |
| 前向速度（m/s） | 0.50 | 2.0 | 0.10 | 0.6 |
| 侧向速度（m/s） | 0.15 | 0.5 | 0.08 | 0.25 |
| 机体 z 轴角速度（rad/s） | 1.0 | 1.0 | 0.20 | 0.35 |
| 站立平移速率（m/s） | 0.10 | 1.5 | 0.04 | 0.75 |

启用条件同样重要：

- 高度代价乘 `support * not jumping`；站立速率还乘普通运动标记和可见指令决定的 quiet 权重。
- quiet 为 `clamp(1 - max(abs(command_vx)/0.1, abs(command_wz)/0.2), 0, 1)`，再乘普通运动、非跳跃标记。前向、侧向跟踪使用互补的普通运动权重。
- 角速度误差代价全时启用，不因没有支撑或进入跳跃而消失。
- 冻结配置没有启用 `stationary_tracking`，因此没有 XY 位置驻留锚点或绝对航向锚点奖励。已有站立速率代价不能解释为奖励回到初始位置。
- 速度跟踪峰值和姿态指数项仍保留，没有独立的 alive bonus。原任务成功事件为 +30，源 `terminated` 事件为 −200；其语义仍由源任务诊断定义。

若支撑、非跳跃条件成立，将 87.026 mm 高度误差代入实际函数，代价密度约为 −5.683/s；将 7.278 rad/s 角速度误差代入，约为 −6.696/s。它们分别对应约 −0.05683 和 −0.06696 的单步代价。这只是代表性误差上的函数值，**不是评估中实际奖励分项的均值**：非线性函数的均值不能由误差均值替代，旧轨迹也没有完整保存这些 reward mask。

平滑性成本使用截断、归一化动作的一阶和二阶差分。环境按参考步长 0.02 s 对系数作二次、四次重标定；在 100 Hz 下，系数分别为动作一阶差分 −0.04、腿部二阶差分 −0.8、轮部二阶差分 −0.16，再统一乘策略步长。不能只读底层默认系数，就认定提高频率削弱了平滑性成本。它们也不等同于评价中的腿目标 rad/s、轮目标 rad/s² 或实际力矩 N·m/s。

物理力矩成本为 $-10^{-4}\sum_i\tau_i^2$，此配方的 effort scale 为 1。两腿持续输出 40 N·m 的示例代价密度为 −0.32/s，尚未包括其他轴和成本。该示例不证明完整目标中的成本竞争关系。

## 奖励与评价信号的对应关系

高度误差和机体 z 轴角速度误差的源信号一致：高度是 base_link 原点相对轮下平均地面的高度，以 m 计；角速度以 rad/s 计，不能解释成世界 Euler yaw 角的导数。

前向跟踪评价采用原始机体 vx，奖励采用经过俯仰投影的 `vx * sqrt(max(1-gx², 0))`。站立奖励用这个前向分量和机体侧向速度组成范数，世界 XY 评价则用 PRE-reset 位置差分。因此不能把世界速度范数直接当作该奖励项的实际输入或贡献。

冻结配置的 `reward_velocity_reference=base_link_origin` 使 reward 路径明确选择 `root_link_lin_vel_b`，而世界位置差分使用 link 原点。已检查本机 IsaacLab 实现包含根刚体 COM 到 link 原点的点速度转换；原 Kaiser campaign 没有绑定这些外部 SDK 文件，仍需同步采样才能核验远端运行的一致性。当前没有找到足以判定点速度实现错误的证据，也不能从这些汇总值反推整机质心运动或实际旋转中心。

力矩统计和物理 applied torque 一致，轴序为四个腿轴、两个轮轴；第 3、4 通道是右侧两个腿轴。旧报告中的力矩 RMS 包含完整采样区间，不能称为去掉 reset 后前 200 步的稳态 RMS。存在传输延迟时，旧 target 诊断是延迟前的解码目标，不是实际到达 PD 的目标。

35D actor 输入包含命令、IMU、关节状态、上一动作和命令上下文，没有直接提供实际 base velocity、高度、XY、绝对 yaw、地面或接触状态。历史可能帮助估计局部运动；打滑和接触变化下，单帧不能保证唯一恢复真实平移速度。这是比较历史网络的理由，但不证明任何 Transformer 已经取得收益。

## 新封存课程对照的观察

同一 Gated H31、训练 seed 1102、同为 1,200 次更新，使用评估 seed 8701 的名义 0.305 m 站立场景；策略 100 Hz、物理 200 Hz。两组 reward 配方相同，指令暴露不同。稳态统计按每环境、每指令段去掉 reset 后前 200 步。

| 指标 | mixed | stationary |
|---|---:|---:|
| 高度平均偏差（mm） | −87.026 | −83.519 |
| 高度 RMSE（mm） | 87.055 | 83.519 |
| 去均值高度波动（mm） | 2.237 | 0.089 |
| 站立世界 XY 速度范数均值（m/s） | 0.219 | 0.312 |
| 机体 z 轴角速度平均偏差（rad/s） | −7.278 | +6.104 |

两组各有 25,600 个稳态样本、32 个统计段。stationary 的高度波动更小，仍有约 84 mm 平均偏差和明显持续运动。它记录的每回合净位移均值约 0.077 m，mixed 约 1.180 m；这是包含部分回合的区间统计，不是统一完整时长的位置保持结果。净位移改善不等于速度或自旋消失，不能用单个指标概括整体改善。

mixed 的另一个评估 seed 9701 已完成，名义站立指标逐数相同；两个评估 seed 不是两个训练 seed。课程对照改变了任务分布和暴露量，且这里只比较一个训练 seed，不能据此证明课程顺序效应、reward 因果效果或架构优胜。

## 下一次如何区分原因

先记录每个 reward 项的真实均值、分位数、启用比例和任务条件统计，包括 support/quiet/jumping 条件下的高度、速度和角速度误差。奖励分项保持奖励每秒与事件奖励分开，避免用总 reward 或代表性误差的曲线代替实际贡献。物理评价继续独立记录未被 reward mask 过滤的误差。

网络与奖励诊断使用交叉对照：当前帧 MLP / Gated × 当前冻结奖励 / 一个明确的奖励干预。两种奖励使用完全相同的任务池及采样剂量、观测与动作约定、100/200 Hz 时钟、reset 与延迟分布、PPO 配方、fresh-rollout 预算和训练 seeds。干预前先确认具体失配机制，再冻结公式、mask 和源码身份，不同时改变课程。

每种网络的奖励效应为 $M_{a,\mathrm{new}}-M_{a,\mathrm{old}}$；两种网络之间这些变化量的差可用于检查架构与奖励的交互。先按训练 seed 配对，再报告跨训练 seed 的差异；多个评估环境或同模型的评估 seed 不能冒充多个训练重复。

选型仍使用高度、速度、角速度偏差/RMSE，速度范数和位移，回合内波动，目标与实际力矩变化、饱和，固定请求首回合的完整时长/健康完整时长统计，以及样本和计算成本。课程干预单独比较；旧成功标记不追补为新的健康存活率。新实验仍需满足原队列闭合与资源准入，本文没有启动新的训练。

## 首回合健康与稳态统计的区别

健康存活率使用每个声明环境的首次回合，包含 reset 后的起始阶段。首回合失败后，即使自动 reset 的后续回合成功，也不会覆盖首次结果或增加分母。健康条件为高度低于 0.20 m 或倾斜超过 0.60 rad 连续达到 0.20 s；完整健康回合还要求达到该场景的时间上限，且没有物理失败、任务成功提前结束、越界或阻塞。它不要求高度、速度或角速度跟踪准确。

当前冻结任务池的首次回合时长为 34 个 10 s 场景、8 个 16 s 场景和 8 个 28 s 场景。整轮 4,001 步、100 Hz 的采样窗口为 40.01 s，可能包含多个自动 reset 回合，不能把完整首回合解释为连续存活 40.01 s。稳态统计的前 200 步剔除只用于误差和波动分析，不用于跳过健康检查。短回合、缺失结果和有效稳态样本覆盖量分别保留。

新的历史控制 trace 保存 PRE-reset 的明确回合 ticks、场景时间上限、结束原因和存活任务标记，以及原精度的绝对高度、倾斜。验证流程复用 `EpisodeOutcomeStatistics` 重算整体和每场景的首回合结果，并与报告逐项核对，再保留到验证后的 cell 指标。场景时间上限来自冻结 exact-case contract，按实际仿真行的场景标签绑定；不假定 PhysX 的行顺序与配置块顺序相同。旧 trace 缺少这些明确证据时只能标为不可用，不能从 `done` 或旧成功率补造健康结果。

这一接入保留已有 `success_rate` 门槛。旧成功率仍对所有已完成的自动 reset 回合统计，后续成功可能稀释首次失败；它不等于首回合健康存活率，也不证明候选已经通过健康筛选。奖励诊断和网络比较应同时查看两种统计及跟踪指标。

新增数组会增加存储开销：两项 int64、六项 bool 和两项 float32 状态共增加 30 B/采样点；float64 状态则为 38 B/点。这是字段大小计算，不是 SDK 存储校准。完整矩阵的分母保持不变，原来的 448 B/点条件预算不能直接沿用到新增字段后的 trace。

## 整段 rollout 的奖励分项记录

后续使用新源码的 frame collector 会启用独立的 `RewardComponentStatistics`。adapter 在源环境 step 返回后、手动 reset 之前，读取逐环境 `native_reward_components`，按整段采集窗口累计实际总奖励与各连续项。统计包括 terminal 和 reset 转移，不是去掉前 200 步的稳态统计，也没有按 PPO learning mask 过滤。物理稳态指标继续独立计算。

每项保留样本数、和、均值、RMS、最小值、最大值；连续密度与乘以 0.01 s 后的单步贡献分别标明 `reward/s`、`reward/step`。没有保存样本，因而不提供分位数。每次 collection 边界 drain，包括提前停止、空采集和异常退出；partial window 不会混进下次 collection。普通训练保留原有非空 partial batch 行为，continuation 仍丢弃不足完整 rollout 的训练尾段，报告中的 `last_collection_reward_components` 只表示最后一次采集，不保证已经用于优化。

显式事件只能来自逐环境 `native_reward_events`。当前冻结环境未提供这个字段，因此事件分项标为不可用，不从 `done` 或总奖励余量推断。余量定义为实际单步奖励减去已报告连续项的单步贡献与显式事件；它可能同时包含未记录的连续项和事件，不自动解释为记账错误。源未提供实际 gate 权重，启用比例同样标为不可用；零贡献不能恢复 support/quiet/jumping coverage。

`metrics.jsonl` 在 `collection.reward_components` 保存整段报告。新增 TensorBoard 标签如下；旧 `environment/reward/*` 曲线继续表示最后一步的跨环境均值，历史日志不会回填或改变含义。

| 标签 | 统计内容 |
|---|---|
| `reward_components/total_step_mean` | 整段所有返回样本的实际单步奖励均值 |
| `reward_components/density_per_s/<term>` | 连续奖励密度均值，reward/s |
| `reward_components/step_contribution/<term>` | 连续项单步贡献均值，reward/step |
| `reward_components/event_step_contribution/<term>` | 源显式提供的事件单步贡献均值 |
| `reward_components/unattributed_step_mean` | 未归属的单步奖励均值 |

组件 shape、device、有限值、窗口内字段一致性验证失败时，adapter 先保留错误，并让物理 step 正常返回供 collector 计数；collector 在采集边界、PPO 开始前拒绝该窗口。失败记录分别保留有效前缀统计、已观察物理步数与 collector 返回样本数，不能把前缀统计称为完整窗口。原有采集异常保持为主异常。若 drain 本身失败，collector 停止接受后续采集，即使显式 reset 也不能重用；需要重新创建环境和 collector，防止未清空窗口混入下一批。

累计使用设备上的 float64 归约，不保留源 tensor alias；每个 observer 调用有一次合并有限值检查的主机同步，drain 再复制归约结果。报告记录 observer 与 drain 的主机耗时；collector 的总采集时间包含这些开销。尚未用实际 SDK/CUDA 验证开销，不将 CPU 接口测试解释成仿真训练或 100 Hz 部署证明。新功能没有热改已经冻结、正在运行的 worker；它也不修复 dense 配方与旧 `diagnostic_logging` 的兼容问题。

## 本地证据索引

以下 audit 是本机运行证据，不作为模型或运行数据提交进 Git：

- `outputs/audits/frozen-reward-alignment-oct09-20261008T193619Z-6e8cdb80/reward_alignment_audit.json`：17 项配置/源码匹配、实际纯函数的 CPU 计算；SHA `ef8d5e081ced648c186554dec72232afc0c68e50f12014268f5949d1eb53bc36`。
- 同目录 `original_runtime_binding.json`：7 项封存 worker、manifest、checkpoint、评估 artifact 绑定；SHA `498325473686da3cf34a0946facb3252c88defbb6c52e68200e21b37385ec8e2`。
- `outputs/audits/runtime-original-queues-20261008T193304Z-c503aa05/paired_eval_seed_comparison.json`：新 mixed 9701 与已有 8701 的配对，SHA `5b64c7440805a886ed9c5e782f277ddad8b45b9969a3a87b49c7e39f4717e8a4`。
- `outputs/audits/runtime-stationary-phase2-newcontrol-20261008T193617Z-4bb70bd1/paired_curriculum_comparison.json`：新 stationary control 与已有 mixed 投影的对比，SHA `8f3bf508f625914d6da2b3e9046a348d055a1c533a9d85d0493a70f5ad7b03c8`。

全部读取均未启动 SDK 或 CUDA，没有热改原训练，也没有重新读取 checkpoint、NPZ 或 events。
