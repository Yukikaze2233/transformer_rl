# 历史估计器与控制网络：训练及后续评估方案

**归档状态：2026-09-19终止后续执行。** 接线阶段仅部分完成，16M特权诊断和64M主对照
未启动；下文保留设计与实现记录，不表示仍在排队。最终状态见[研究结论](RESEARCH_CONCLUSION.md)。

初版日期：2026-09-17。当前为实现与分阶段运行版（protocol revision 3）。
目标是分别回答“状态估计是否有帮助”和“同样的估计任务是否需要Transformer”。

机器可读设计见 [estimator_comparison.json](protocols/estimator_comparison.json)。
它是研究协议，**不是当前CLI可以执行的完整实验配置**。
已有两种直接控制网络的参考规格为 [estimator_reference.json](../configs/estimator_reference.json)，
可以用现有CLI生成plan；环境factory、任务路径和identity仍需在运行环境绑定。

**实现更新：六种网络均已具备训练、检查点和导出路径。** 完整CLI规格为
[`configs/estimator_comparison.json`](../configs/estimator_comparison.json)，
分阶段入口为 `tools/run_estimator_study.py`。它先执行接线和16M特权诊断，诊断满足
预先固定的任务门槛后才自动进入18项64M主对照；未通过则写入`needs_review`收据。
可选64M诊断延长、轻量消融和128M确认不由本次队列自动提交。

## 1. 六种主对照

| 名称 | 历史的用途 | 进入控制头的信息 | 训练方式 | 支持状态 |
| --- | --- | --- | --- | --- |
| `direct_mlp` | 16帧展平，直接输出动作 | 完整历史、当前指令 | PPO | 已有 |
| `direct_transformer` | 标准Last-token，直接输出动作 | 时序表征 | PPO | 已有 |
| `velocity_mlp` | MLP估计三维机体线速度 | 当前观测＋估计速度 | PPO＋独立速度监督 | CPU验证通过 |
| `velocity_transformer` | Transformer估计相同速度 | 与上一行相同 | 与上一行相同 | CPU验证通过 |
| `context_mlp` | HIM式MLP估计速度/高度及16维上下文 | 当前观测＋4维状态＋16维latent | PPO＋状态监督＋swap loss | CPU验证通过 |
| `context_transformer` | Transformer替换HIM式历史encoder | 与上一行相同 | 与上一行相同 | CPU验证通过 |

`velocity`是复旦思想的受控移植，`context`是华南虎HIM思想的受控移植。
两者统一到我们的机器人、30维输入和训练器，不称为原系统的逐项复现。
DreamWaQ的随机latent、重建与KL属于后续单独实验，不与HIM混为一种估计器。

**主要因果比较是同一行组中的MLP与Transformer。** direct到velocity同时改变信息路径
和监督，velocity到context同时增加高度标签、latent和预测目标；这些只能说明整套配方
的收益，不能直接把收益归因于attention或某个单独损失。

当前Kaiser的7变体16M结果是历史参考。新研究使用新seed，不把既有最优seed当新的独立证据。

## 2. 固定网络与任务条件

- 所有主实验：512环境、32步rollout、策略100Hz、物理200Hz，历史16帧，critic29。
- 沿用冻结研究任务的资产、奖励、动作顺序、缩放与限幅；额外传输延迟为零、无推扰。
- 指令、本体量和上一条issued action沿用现有frame30；padding为左侧零填充加valid mask。
- Critic保持独立 `29 → 256 → 128 → 64 → 1`。
- PPO工作配方沿用 `optimized_control.json`：LR3e-5、2 epochs、4 minibatches、
  initial_std=0.2、mean_init_scale=0.1、target_kl=0.01；action_clip=100。
- 这是**共同优化配方**下的结构对照，不声称每个模型已达到各自最优超参。

两个估计器家族的共同编码输入是每帧
`[scaled_frame30, elapsed_age/0.1s, valid]`，共32维，padding payload先消除。
MLP展平后使用 `[128,64]`；Transformer使用2层、64维、4头、FFN128、固定index编码、
普通残差、Last-token读出、dropout=0。同一家族的输出维数完全相同。
32维估计器入口已在专用`HistoryEstimator`中实现，原直接控制Actor仍使用自己的预处理路径。

估计器控制头统一 `[128,64,32] → 6`：velocity输入33维，context输入50维。
保持6维联合Gaussian及现有环境动作解码。当前帧已含command，控制头不额外重复拼接。
direct两项保持现有实现，作为直接控制锚点，不与估计器对照混称只改变了一个结构因素。

参数量要分别记录动作头、部署encoder、训练专用target/prototype和critic；
不能只数一个动作MLP就声称整套系统更小。近似参数匹配仍不等于计算量匹配。

### 2.1 固定部署参数预算

主档限制为 **100,000个部署参数以内**，轻量档限制为 **60,000以内**。
计数包含部署必需的encoder、状态预测头及动作控制头；不含6个训练探索参数、
48,897个critic参数，以及context家族训练专用的11,984个target/prototype参数。
ONNX中的非参数常量、buffer与图元数据单独统计，不能用文件字节数代替参数量。

| 方案 | 主档部署参数 | 轻量档部署参数 | 核算依据 |
| --- | ---: | ---: | --- |
| direct MLP | 74,694 | 37,574 | 实例化已有Actor |
| direct Transformer | 69,446 | 39,798 | 实例化已有Actor |
| 速度估计MLP＋控制头 | 89,001 | 52,073 | 实例化真实估计器Actor |
| 速度估计Transformer＋控制头 | 84,265 | 54,633 | 同上 |
| HIM式MLP＋控制头 | 92,282 | 55,354 | 同上 |
| HIM式Transformer＋控制头 | 87,546 | 57,642 | 同上 |

主档Transformer：**d64、2层、4头、FFN128**；轻量档：**d48、2层、4头、FFN96**。
主档MLP历史主干/估计器为 `[128,64]`，轻量档为 `[64,64]`。
估计器家族的控制头始终为 `[128,64,32]`，输出目标和latent维数不随容量改变。
普通残差、固定窗口、无KV缓存，不在本矩阵加入Gated或更大宽度。

主档纯FP32参数约0.28–0.37MB；文件体积、运行时内存和推理时间另外实测。
以上容量已经通过真实可执行Actor重新核算，训练专用target/prototype独立计数。
参考配置显式固定层宽，避免以后修改共同base时静默改变本实验容量。

### 2.2 200Hz部署的计算准入

候选MiniPC上的完整部署网络以 **batch1、FP32、ONNX Runtime CPU**作为基线，
目标为 **p99≤2ms、p99.9尽量≤3ms**，并单独测输入处理至命令提交的完整5ms周期。
完整网络必须包含估计器，不能只测动作头。记录最大延迟、超时率和连续超时次数；
p99合格不代表每次都能满足截止时间。FP16需有后端支持及数值/闭环验证后另作对照。

至少30分钟、100,000次调用，在预热并达到稳定温度后，按候选控制频率调度，
带代表性的ROS/通信负载，记录CPU、功耗、线程、亲和性与调频条件。比较1与2线程。
目标MiniPC型号尚未绑定，当前状态是 **target-device latency未验证**；
桌面CPU短测不能作为此项通过依据。参数量达标与延迟达标分别判定。

本设计的学习对照仍明确是 **100Hz策略/16帧**。若部署要求实际200Hz闭环，必须另行
冻结并验证200Hz任务，包括奖励时间积分、动作变化惩罚、折扣/GAE时域、物理/控制
节拍及历史跨度。200Hz的16帧跨度75ms，31/32帧才约150/155ms；扩大窗口后必须
重新计数并计时，尤其MLP第一层会变大。不能因前向低于5ms就宣称100Hz策略已经
成为合格的200Hz控制器，也不把两种频率的训练结果混合排名。

## 3. 估计器损失、时间与PPO顺序

### 标签与目标

- velocity：当前决策时刻的真实机体COM线速度3维，参考现有critic列25–27的环境语义。
- context：在此基础上增加当前base_height，定义与环境 `_base_height()` 一致。
- 速度以1m/s归一化，高度以 `(h−0.30m)/0.10m` 归一化；只使用固定常数。
- 状态损失为归一化分量的平均MSE，训练日志另报SI单位MAE/RMSE/p95。
- context另以同一episode内的下一帧proprio16为预测表征目标，排除command、issued action、
  时间字段，降低直接复制确定性字段的捷径。target MLP `[128,64] → 16`，32个prototype，
  temperature=3、Sinkhorn epsilon=0.05、3 iterations，swap loss权重1。
- next-state表征目标只进入训练损失；Actor只使用当前及过去信息。
- terminated/truncated的跨reset配对一律排除。当前状态监督仍可用于这些endpoint；
  没有有效未来配对的minibatch跳过swap部分并记录有效计数，不能对空张量求均值。

这是明确的新目标布局，不照搬上游列号；上游代码中的不同时间切片不能直接迁移。
`EstimatorBatch`扩展了独立、owned的下一帧目标和有效mask；旧`PPOBatch`字段保持原样。

### 更新顺序

1. 冻结估计器版本，采集rollout，保存原history、行为分布与监督标签。
2. 使用原严格moment/log-prob一致性检查；完成全部PPO epochs，期间估计器保持冻结。
3. 固定控制头，独立Adam更新估计器；LR1e-3、2 epochs、4 minibatches、grad clip1。
4. 以PPO结束、辅助更新开始前的完整策略为参考，在相同rollout端点计算辅助更新引入的
   全rollout平均analytic KL，初始限制为0.0025。
5. 超限时回滚整个辅助模型、Adam、target/prototype和随机状态，依次用原LR的0.5、0.25
   重试；仍超限则保留辅助更新前的状态。本次缩放只作用于候选事务，下一rollout仍从
   配置LR开始。接受/拒绝次数、尝试与接受梯度步、实际计算时间分开记录。
6. 另外记录从原behavior到最终完整策略的KL。辅助KL限制不等于完整策略的总KL上界。

这项约束作用于两种encoder，而不是只给Transformer使用。它属于受控移植的新增规则，
不是复旦/HIM原始配方。已通过回滚与重放测试；`detach()`本身不能防止估计器
更新改变动作分布。保存检查点时必须包含两个优化器以及辅助网络全部训练状态。

## 4. 分阶段预算

统一每次更新16,384条环境transition，以977次更新为一个保存单位。
使用整倍数是为了复用当前checkpoint_interval，避免丢失精确阶段检查点。

| 档位 | 累计updates | 实际训练transitions/seed |
| --- | ---: | ---: |
| 约16M | 977 | 16,007,168 |
| 约32M | 1,954 | 32,014,336 |
| 约64M | 3,908 | 64,028,672 |
| 约128M | 7,816 | 128,057,344 |

### A. 接口与任务可行性

六种网络各用seed2003做80updates有界接线验证：前向、辅助标签时序、PPO一致性、
优化器隔离、reset、checkpoint与导出。这不是选型训练，6项合计7,864,320条采样。

另设两个诊断seed2309/2311，使用当前frame30加真速度/高度的MLP，先检查16M，
必要时预算最多64M/seed。它提供“状态直接给出时能否学会任务”的诊断，**不进入部署排名**。
结合指令响应、奖励分项、动作饱和与物理指标判断；诊断失败不自动证明任务不可学。
若仍存在全模型共性的错误，应先处理任务/优化问题，避免直接投入完整18项训练。

### B. 主对照：六种×三个seed

训练seed：**2101、2113、2129**。每项连续训练到3,908 updates，保存977、1,954、
2,931、3,908检查点；预先指定评估977、1,954、3,908，2,931只作恢复参考。
不为检查点观察人为重启仿真，不以16M reward较低为由淘汰某一模型。

主对照共18项，训练预算 **1,152,516,096 transitions**，不是18次16M独立训练相加。
评估不可变检查点可在该训练段完成后串行进行；如以后支持在线阶段评估，需要独立进程
和冻结文件，不修改正在训练的模型。

### C. 条件性轻量容量消融

主档结果出来后，最多选择一个MLP与Transformer都有合格结果的家族，同时测试两种
backbone的轻量档。每种使用相同三个screening seeds、从头训练到64M，复用既有主档
结果作配对对照；不只给Transformer额外调参，也不把所有轻量候选一开始加入主矩阵。

最多增加 **6项训练、384,172,032条训练样本、252份阶段评估**。
两种轻量配置各有80updates接线，额外2,621,440条样本。
两档只改变历史主干的宽度，保持历史长度、估计目标、控制头和优化配方相同。

轻量档须通过同样任务门槛，逐配对seed/场景比较tracking分数，增加不超过5%；
同时高度jitter增加≤0.2mm、vx jitter增加≤0.005m/s、wz jitter增加≤0.02rad/s，
站立最大xy偏离p95增加≤1cm，健康episode比例不下降，才优先选择更小模型。
这些是训练前冻结的研究容忍度，不是统计等价证明；报告全部退化与额外搜索成本。

如果主档在长训前已被目标设备延迟测试否决，应修订并重新冻结成对实验，
而不是运行中静默换成轻量网络。原始主档结果与新规格仍保留各自来源。

### D. 新seed确认

在64M验证集上各选一个任务合格的MLP家族候选和Transformer家族候选，
若执行了轻量消融，则按上述冻结规则确定相应容量，
以新seed **2203、2213、2221、2237、2243** 从头训练至128M。
共10项、**1,280,573,440 transitions**，只有最终固定检查点用于held-out确认。
中间checkpoint可保留，但不能在held-out集上挑峰值。

若只有一个家族合格，报告另一家族未达标；不为凑足决赛候选选一个不合格网络。
若都未达标，报告没有赢家，回到共同任务/优化诊断。

主对照与确认合计约2.433B训练样本；加接线及可行性上限约2.569B。
如执行完整轻量消融及其接线，总训练上限约 **2.956B**，轻量阶段不是默认全部执行。
按**假设有效吞吐4000 transitions/s**，仅主对照训练约80小时、确认约89小时，
均未计评估与启动成本，也不是Kaiser交付时间承诺。先用实际模型吞吐更新预算。
接线、可行性、主对照、轻量消融、确认分别设预算，不自动一次提交全部阶段。

## 5. 固定评估协议

### 验证集

每checkpoint使用8环境×4,000 vector steps、独立eval seeds **3101、3113**，
确定性动作均值；每回合丢前200步，最小稳态段200样本。
7场景沿用站立0.28/0.30/0.32m、前后0.5m/s、左右1rad/s。

18项×3个checkpoint×7场景×2个eval seeds＝**756份评估报告**，
对应24,192,000条评估transition，与训练样本分开。
4,000步约40s，但环境仍有20s timeout，不能说成连续40s站立。

### Held-out确认集

eval seeds **3203、3217、3229**；包含上述7场景，以及6个额外场景：

- 0.29/0.31m站立；
- `[vx,wz,h]=[0.25,0,0.29]`、`[-0.25,0,0.31]`；
- `[0.25,0.5,0.29]`、`[-0.25,-0.5,0.31]`组合运动。

这些是未用于本轮模型选择的测试设置，不声称位于训练命令分布之外。
若两名候选进入确认，共10项×13场景×3个eval seeds＝390份报告。

### 指标与原始数据

每步保存时间/env/episode/指令、真实速度/高度/倾角/世界坐标、动作均值与issued action、
腿轮目标、力矩、估计值和标签、终止/截断标记。大轨迹留ignored artifacts，报告引用SHA。
direct模型的估计字段为不适用，不因没有估计头被判评估失败。

- 全区间：跟踪误差、物理失败、非轮净力、终止/timeout、episode完成与censoring。
- 稳态：MAE、signed bias、逐episode中心化std、derivative RMS及覆盖率。
- 漂移：从每个episode暖机后首个世界xy位置计端点位移和最大偏离，重置后独立计算；
  不把body-frame vx积分伪称世界坐标位移。
- 估计器：同一时刻SI单位MAE/RMSE/p95，按高度和运动场景分别统计。
- 成本：采集/更新/评估/启动时间、尝试及接受优化器步、显存、部署参数、batch1
  forward p50/p95/p99；完整控制链路时延单独记录。

指令阶跃、推扰、非零延迟/掉包、无timeout连续站立是后续套件，需要对应环境能力后
单独冻结。当前固定命令实验不冒充覆盖这些能力。

## 6. 任务门槛与选择规则

以下是首轮**研究筛选阈值**，不是用户指定的实机精度要求。应在新训练启动前冻结，
不能看完新结果再放宽。机器可读阈值在研究协议中。

物理失败：环境terminated，或高度低于0.20m/倾角超过0.60rad持续0.20s；
暖机期同样计失败。每episode一旦失败即锁存，直到reset，不因之后不动而恢复合格。
每份报告要求完成episode中的健康timeout比例≥90%，稳态样本覆盖≥80%，
评估截止导致的censored episode样本数占总样本≤10%。censored段单独统计，不能计成功。

| 场景 | 稳态高度MAE | 稳态vx MAE | 稳态wz MAE |
| --- | ---: | ---: | ---: |
| 站立 | ≤15mm | ≤0.03m/s | ≤0.10rad/s |
| 运动 | ≤20mm | ≤0.10m/s | ≤0.20rad/s |

所有场景、eval seeds和三个training seeds都通过，才进入主对照的任务合格候选集合。
统计不按成功挑选低抖动片段；失败段与短段仍保留全区间结果。
stability已新增`mean_abs`，**不能拿signed bias替代MAE**；新门槛还使用独立`control_quality`报告。

选型规则：

1. 每training seed计算各高度/vx/wz稳态MAE除以对应阈值，取所有场景和eval seeds的最大值。
2. 在合格候选内，比较该最差场景分数的跨training-seed中位数。
3. 相对差异在5%内视为工程近似平手，依次比较最差training seed分数、站立episode
   最大xy偏离的p95、batch1 p99和部署参数量。5%不是统计显著性阈值。
4. 同时展示抖动、动作/力矩变化与Pareto取舍，不因排名分数较小隐藏其他指标退化。
5. 全部失败/中止seed进入结果表；frames和8个评估环境不能当作独立training seeds。
   新seed确认报告配对差异、区间和失败率；五个seed仍可能证据不足，允许“无明确优势”。

## 7. 实现与执行顺序

1. 按容量表实现独立估计器Actor、辅助优化器/事务回滚及checkpoint/导出；补reset与next-target合约。
2. 以CPU合成序列验证未来/特权泄漏、梯度隔离、old-logp、KL回滚和恢复后一致性。
3. 补逐episode失败、稳态MAE、世界位移、估计误差及原始轨迹输出，验证常值大偏差、
   正负误差抵消、reset跳变、短段和censored段等反例。
4. 增加成对family校验、任意已保存checkpoint的评估编排及跨阶段汇总。
5. 核验完整部署参数预算与目标CPU的batch1时间预算，在新的source与runroot冻结代码/
   资产/配置、环境identity、依赖与设备信息，执行有界接线。未绑定设备时保留延迟未验证标志。
6. 通过可行性检查后按阶段排队；本批默认单并发，训练与评估单独记时。

现有CLI只能自动评估最终checkpoint，不能靠修改JSON就实现三档评估。
现有reference规格产生6个直接控制任务、84份**最终checkpoint**评估；另外两档评估由
后续编排负责。环境factory保持null，所以该规格目前只用于规划验证。

```bash
python -m transformer_rl.experiment_cli plan \
  --spec configs/estimator_reference.json --root /tmp/estimator-reference-plan
```

主训练单项软上限6h、含评估job上限12h；确认分别12h/24h。
全队列timeout按实际任务数和并发另行推导，不能套用旧队列36h导致后续任务被截断。
中断后恢复模型和所有优化器，但环境重置要标明；累计采集/实际用于更新/丢弃样本分别记录，
不因中断后再采样而把样本量强行改成理论值。禁止覆盖旧root或只重试失败seed而删除原失败。

六种Actor、估计器训练、双优化器检查点、导出和新增评估量已实现并完成CPU验证。
真实仿真接线与任务可行性仍由分阶段队列产生独立证据。

## 8. 规划校验

参数预算检查（只实例化CPU模块，不启动仿真）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python tools/check_model_budget.py \
  --protocol docs/protocols/estimator_comparison.json
```

工具同时校验主档与轻量档的12项参数清单、参考配置层宽以及条件轻量阶段的预算。
核算证据见 [model-capacity.json](evidence/model-capacity.json)。

revision 3改为实例化真实估计器Actor核算，最新证据见
[model-capacity-implemented.json](evidence/model-capacity-implemented.json)。
全量CPU验证 **1085 passed、46 skipped、10 subtests passed**；覆盖双优化器续训、
估计器更新拒绝后的参数/Adam/RNG恢复、ONNX八类输入验证和物理失败/漂移反例。

revision 2校验完成：12项容量清单全部符合声明与上限；参数超限和声明计数错误的
反例均被拒绝。新reference plan为6 jobs、16份配置，SHA为
`2f3bf4436b47d162e75a4fa424b67d82bd774d6a73eee31b17b5f53131d1fbcb`。
相关实验/场景编排CPU回归 **89 passed**。这不是目标MiniPC的延迟验收。

以下保留revision 1首次规划校验，revision 2显式固定容量后应生成新的plan/source身份：

- 现有CLI成功生成reference plan：6 jobs、16份冻结配置、84份预期最终评估报告。
- 核对研究协议与reference的seed、场景、预算、action clip及保存间隔一致；
  两个估计器家族内除backbone/name外的实验因素一致，各用途seed集合互不重叠。
- 核算18项主训练、756份阶段评估、390份条件性确认评估及各阶段真实样本量。
- 当前工作树生成的plan SHA：
  `6eb2f249dc1f1df65d55f07eb32901d5e608ecf2d6857292dffd2860891706af`。
  此值只对应本次规划检查；后续实现/环境绑定后必须重新生成plan。
- CPU执行 `tests/test_experiments.py` 与 `tests/test_scenario_experiments.py`：
  **89 passed**。这些测试验证已有编排能力，不是待实现估计器的测试或物理效果验证。
