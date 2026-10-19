# 网络选择：训练预算与独立评估设计

**本轮研究于2026-09-19收尾。** 本文保留历史设计；实际完成范围及停止决定以
[研究结论与归档](RESEARCH_CONCLUSION.md)为准，不继续执行剩余预算。

本方案依据原论文、现有训练历史与Kaiser吞吐记录。迭代数、梯度步数和环境样本数分开统计；短测诊断不等于架构结论。

## 文献依据

- [What Matters In On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990)，§3.2/3.5：在其MuJoCo任务中，缩小策略最后一层权重、合理选择初始std和重复利用经验对样本效率有显著影响。其建议的小输出初始化不等于std越小越好，也不直接给出本机器人最优参数。
- [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764)：GTrXL以pre-LN/恒等路径、GRU-type残差门控改善优化。原算法为V-MPO；门控偏置初始化与动作输出层缩放是不同因素，不能混为同一种初始化。
- [Learning Humanoid Locomotion with Transformers](https://arxiv.org/abs/2303.03381v1)：teacher/student PPO＋退火教师KL，而非纯从零Transformer PPO。v1表IV列出6000迭代、teacher/student 8192/4096环境、24步rollout、4张A100，但global/per-device统计口径不明确，不能把这些迭代数直接移到单卡。v2将详细超参表移至补充材料，本次未独立核实其精确预算。
- [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)：少量训练runs及选择最大测试分数可能改变排名；大量evaluation episodes不能替代独立training seeds。3/5 seeds只是资源约束下的起点，不是充分性保证。

## 第一阶段：优化敏感性

固定time-attention骨干、任务、观测、history、critic与样本预算，围绕基线分别改变：

| 因素 | 基线 | 单因素候选 |
|---|---:|---|
| learning_rate | 1e-4 | 3e-5、1e-5 |
| mean_init_scale | 1 | 0.1、0.01 |
| initial_std | 0.2 | 0.1、0.4 |

mean_init_scale仅乘初始化时的动作均值头weight/bias，不改变hidden/critic初始化、RNG流或运行时动作映射。初始std是归一化动作的标准差；相同均值变化下，较小std会放大KL中的均值项。

首先用一个筛选seed运行单因素检查，再用新训练seeds确认有希望的配方或组合。记录完整rollout上第一次optimizer.step后的KL及均值/方差分解、实际梯度步利用率、初始输出幅值、最终分布及独立物理指标。低KL本身不是成功标准：完全不学习也会低KL。

诊断开关不改变训练轨迹，但增加额外前向计算；不能把开诊断的短测耗时直接作为关闭诊断的长训速度。

## 第二阶段：足够展开的学习曲线

先冻结各架构的优化配方及选择规则，再比较共同的环境任务与样本预算。

当前首档规格为`configs/learning_curves.json`：七种网络（加入标准last-token读出）、三个新的训练seeds1011/1022/1033，512环境×32 rollout×977 updates，每项16,007,168 transitions、全组336,150,528 transitions；共同采用LR3e-5、mean_init_scale0.1、std0.2、2 epochs/4 minibatches，关闭额外诊断前向。该共同配方是结构对照的工作基线，不宣称它是MLP/GRU等各架构各自最优的超参。

最终checkpoint分别评估低/中/高站立、前进、后退、左右旋转七个命名场景；每场景2000步、eval seed301，统计按场景独立保留。20s左右的任务timeout仍可能reset，不能冒称连续60s站立。中间checkpoint每100 updates保存，但不自动对每个checkpoint启动全场景评估，避免把阶段评估预算放大十倍。

| 累计预算/训练seed | 用途 | 解释边界 |
|---|---|---|
| 16M transitions | 至少3 seeds，检查是否持续学习 | 不据早期reward淘汰整个Transformer家族 |
| 64M transitions | 相同任务下的正式初筛 | 同时看物理指标和学习曲线，不只看终点回报 |
| 128M transitions | 保留MLP、最佳Transformer及竞争候选 | 接近旧MLP的平台样本量，不是普适收敛门槛 |
| 256M/512M，如有必要 | 核查慢热或长期退化 | 由验证曲线决定，不机械追加 |

旧MLP的参考点：1024×48×2500≈122.88M transitions；累计20k更新约0.983B。新512×32每update仅16,384个transitions，所以不能拿两边“1000次更新”作相等预算。

在512×32下，16M/64M/128M分别约需要977/3907/7813次updates。更改环境数或rollout后按真实transitions重新计算；不乘PPO epochs或Transformer context长度。KL提前停止造成的实际optimizer steps和样本复用次数另列。

## 统计与选型

- 调参使用的seed/场景与最终held-out测试分开。最终两个候选建议各5个新的training seeds；如果区间仍宽或失败率差异不明确，增加runs或保留“证据不足”。
- 同一个训练run的多高度/多场景结果具有相关性。统计以training run为cluster，不能把所有帧、episodes或场景当作独立训练样本。
- 同时比较相同新增环境样本数与相同GPU-hours。网络参数量相近不等于计算成本相同。
- 预先固定checkpoint选择规则；完整保留失败seed、timeout和partial结果，不只挑最优seed或训练过程中的最高测试分数。
- 调参预算应对各架构公平。统一超参初筛与各架构经过同等搜索预算后的比较是两种问题，应分别报告。

## 独立场景矩阵

| 任务族 | 场景 | 当前状态 |
|---|---|---|
| 高低静止 | 0.28/0.30/0.32 m，vx=wz=0 | 已有固定命令接口；长时验收尚待执行 |
| 行驶/旋转 | 各高度正反行驶、纯旋转、组合命令 | 已有固定命令接口；统一矩阵尚待执行 |
| 指令转换 | 起步/刹停、正反切换、低高切换 | 需明确时序脚本与评价窗口 |
| 扰动恢复 | 不同高度、前后左右扰动、恢复时间 | 待接入并验证，不把零扰动结果算作覆盖 |
| 通信/控制差异 | 观测与动作延迟、jitter、闭环参数差异 | 核心时序组件已实现；研究任务当前额外延迟为零，未完成此项物理对照 |

评价优先看失败、非预期接触、速度/高度误差、零速漂移、恢复时间及执行器负荷。净接触力没有对手身份时不称为地面支撑。现有研究任务20s timeout会reset，累计60s不能表述为连续站立60s。真实PID、传动和传感时序仍需独立标定。

## 计算预算

Kaiser先前六模型短测的总训练样本/完整调度时间约3135 transitions/s，包含启动和评估，不能直接代表所有优化配方的稳态速度。按该量级，1B样本约88.6小时；一个全部展开至多seed确认的3.512B示例约311小时（约13天），完整held-out测试另计。

因此采用逐级分配和更新吞吐实测，不一次盲目提交全部最大预算。新配方有效梯度步数增加后，计算成本可能变化；诊断关闭后也会减少额外前向。先给出实际速度与候选稳定性，再决定长期并发预算。
