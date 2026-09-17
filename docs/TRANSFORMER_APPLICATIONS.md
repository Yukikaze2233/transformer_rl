# Transformer应用方向与MLP基线复核

日期：2026-09-17。本次为源码核查与文献调研，以下候选结构尚未实现或训练。

## 结论

优先研究“小型历史编码器/状态估计器＋当前观测直连＋MLP控制头”，
将MLP和Transformer作为可替换的历史编码器。在同样输入、监督目标和梯度边界下
比较，才能判断注意力是否有增益。

现有 `history_mlp` 是历史直接控制的有效对照，但不是复旦默认方案的等价复现。
四种已完成Transformer在16M档普遍缺乏有效高度/速度跟踪；Last-token只是该快照下
平均训练reward较高的候选，并非已通过任务的赢家。

## 1. 复旦：默认配置也有显式估计器

核查仓库 `yly-true/fudan_rl_wheel_leg`，本地HEAD：
`8204e853dfd2ed06d85a322e1a998c3d20a3be2c`。
所读tracked源码无本地修改；本地其他未跟踪运行工件未用于推断默认结构。

`plane`和`jump`默认 `policy_class_name` 都是 `ActorCriticSequence`。
尽管类名含Sequence，历史编码器实际是前馈MLP：

- 单帧25维，历史5帧，展平为125维。
- 历史编码器：`125 → 128 → 64 → 3`，估计缩放后的机体三维线速度。
- 动作网络：`当前25维＋估计3维 → 128 → 64 → 32 → 6`。
- `actor_critic_sequence.py:181–184`显式使用 `latent.detach()`。
- PPO优化器更新actor、critic、std；独立extra optimizer更新encoder。
- encoder目标由 `ppo.py:263–278`对齐critic前三列；
  环境 `legged_robot.py:374–388`确认前三列为缩放后的机体线速度。

用原类在本地CPU实例化核算，未创建仿真环境：

| 部分 | 参数量 |
| --- | ---: |
| 动作均值MLP | 14,246 |
| 历史速度encoder | 24,579 |
| 两者合计，不含critic/std | 38,825 |
| 加6维Gaussian std | 38,831 |

`125 → 128 → 64 → 32 → 6`确实有26,662个均值网络参数，
但这是假定“历史直接进入动作MLP”的结构，不是上述默认Sequence配置。
具体公开权重或实机演示如使用其他配置，需要绑定其配置、checkpoint和部署导出物，
不能用默认配置替代工件身份。

主要源码：

- [plane配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py#L270-L310)
- [历史encoder与动作路径](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py#L70-L116)
- [梯度边界](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/modules/actor_critic_sequence.py#L181-L199)
- [独立估计器优化](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/rsl_rl/algorithms/ppo.py#L255-L283)

## 2. 华南虎：应区分普通PPO、DreamWaQ和HIM

核查仓库 `scutrobotlab/wheeled-legged_RL`，本地HEAD：
`b8ff79f3df855faf9dc92f4a282bd80c42649466`，所读tracked源码无本地修改。

- 普通V14 Flat PPO配置为MLP `[256,128,64]`；对应基础Flat配置
  `use_frame_stack=False / num_obs_hist=1`。因此不能概括所有配置都用了短历史拼接。
- DreamWaQ与HIM特定配置都是28维单帧、5帧历史；额外估计状态为4维。
  环境critic特权块先拼机体线速度3维，再拼 `obs_height` 1维；高度定义以该环境为准。
- DreamWaQ的CENet接收展平历史，生成显式状态与潜变量，动作MLP接收当前观测与code。
  配置encoder `[256,128,64]`、decoder `[64,128,256]`，默认动作头 `[512,256,128]`。
  训练包含状态估计MSE、通常对下一观测的重建MSE、潜变量KL正则。
- HIM历史encoder输出显式估计量和16维latent；动作路径通过no_grad/detach隔离估计器。
  估计器独立训练，使用显式状态MSE及prototype/Sinkhorn的swap loss。

这些结构支持“用历史推断当前不可直接获得的状态/上下文”的判断。
但潜变量没有逐维对应接触、摩擦或外扰的语义保证；不能把这些都说成有真值监督的
显式输出。HIM也不是VAE，DreamWaQ的潜变量采样/重建与HIM的对比目标不能混称。

主要源码（相对上述仓库）：

- `source/agent_tasks/agent_tasks/direct/wheelbipe/agents/rsl_rl_ppo_cfg.py`
- `source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/{cfg_utils,env_cfg}.py`
- `source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py:3262–3306`
- `source/agent_rl/agent_rl/rsl_rl/modules/{actor_critic_dreamwaq,actor_critic_him,him_estimator}.py`
- `source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_dreamwaq.py:313–337`

## 3. 对当前设计的修正

### 拼接MLP也能建模时间关系

固定位置上的输入权重可以表达有限差分、趋势、延迟反馈和非线性历史关系。
准确区别是：拼接MLP没有显式递推/注意力结构以及跨时间共享的序列归纳偏置，
不是“没有任何序列建模”。在短固定窗口上，这种简单结构可能已足够。

### 目前缺的是强估计器基线

我们的 `history_mlp` 为 `(30+2)×16+3=515 → 128 → 64 → 6`，
额外2维是帧年龄和valid，额外3维是当前指令。它没有速度估计瓶颈或独立估计器损失。
它可以检验“相同窗口与输入下，端到端注意力是否优于flatten”，
不能代替“复旦式MLP估计器＋反应式控制”的对照。

当前 `supervised_attention` 的速度头只通过MSE塑造共享query表征；
预测速度没有直接进入动作头，且PPO与辅助损失共同更新编码器。
这与复旦的独立估计器、detach边界、当前观测直连是三个不同因素。

### 信息路径比简单增大模型更值得先检验

当前Transformer由64维readout经单线性层直接产生动作均值。
当前观测和指令已存在于网络输入，但必须经过时序主干；没有独立的当前观测直连路径。
候选结构保留历史表征，同时将当前观测/指令直接提供给非线性控制头。
这是优化和归纳偏置假设，不是已证实的失败根因，也不是补入新的传感器信息。

### 同信息量对照不等于原系统复现

原系统的历史长度、padding、动作定义、critic特权信息、奖励、随机化和PPO预算均不同。
例如复旦默认critic含更丰富的高度扫描和动力学信息，我们的critic是29维。
需要区分：原配方复现用来验证实现；同任务、同信息预算实验用来归因架构。
实机成功支持该完整控制系统的可行性，不单独证明某个MLP的因果优势，也不自动满足
本项目的静止漂移、毫米级高度误差和抖动目标。

## 4. 推荐的应用方向

### 优先级一：估计器与反应式策略解耦

令历史编码器为 `E(H_t)`，动作策略为 `P(o_t, c_t, E(H_t))`。
先复现MLP encoder和速度监督、detach边界，再只将encoder替换为小Transformer。
两者使用完全相同的当前观测路径、估计输出、控制头、辅助标签和优化规则。
先比较三维线速度；高度监督作为另一个消融因素加入，必须统一高度参考定义。

与现有辅助头相比，显式估计量使误差可诊断、状态可复用。
它仍是历史的函数，没有新增信息；纯本体输入下不可辨识的匀速滑移不会被网络自动解决。
如引入latent以补足速度瓶颈，需给MLP和Transformer相同维数、相同学习目标。

### 优先级二：教师辅助的Transformer策略

先建立通过多高度/速度检查的特权MLP teacher，再在student自身访问的状态上提供
教师动作或分布，联合PPO训练并逐步降低模仿权重。
这样可以研究是否改善从零训练的样本效率。生成teacher和额外采样的成本单独记录。
旧MLP的静止漂移尚未达到目标，不能直接当作合格专家。

### 优先级三：身体结构token与时间尺度

轮腿系统可研究机身、左腿、右腿和轮组的分组表示，以及空间/时间分解注意力。
腿位置目标和轮速度目标使用不同非线性输出分支，但共享融合表征与联合PPO目标。
仅将一个线性6维头拆成线性4维和2维头在表达上等价，不构成新的有效结构。

窗口先单独比较5、16、32帧；100Hz下首尾跨度分别40、150、310ms。
必要时再研究近历史密采样、远历史稀疏采样，并保留真实时间戳。
同样16帧在文献50Hz控制器中跨度约300ms，不能只按帧数照搬。

### 后续：预测式预训练、多模态与统一教师学生

下一观测预测可作为历史表征的辅助任务；完整轨迹数据需要包含时间、动作域和reset
语义，标量metrics与checkpoint本身不是离线轨迹集。
深度/地形输入具备可靠来源后，再考虑跨模态attention。
ULT式统一teacher/student可以研究，但其特权隔离和混合行为策略需要重新核验PPO
likelihood合约，工程复杂度高于替换历史encoder。

## 5. 下一轮最小实验

| 对照 | 回答的问题 |
| --- | --- |
| 当前history MLP/GRU与Last-token | 同窗口端到端控制是否需要attention？ |
| MLP历史估计器＋当前观测MLP控制 | 物理估计与控制解耦能否改善跟踪？ |
| Transformer估计器＋相同控制头 | 保持估计任务不变，attention是否真正有增益？ |
| 上述两种encoder的detach/联合梯度对照 | 收益来自编码器还是梯度耦合方式？ |

不要一次叠加更长历史、更多层、更多辅助标签和新奖励。
统一任务、动作、critic信息、监督预算与预先固定的评估场景，按新增环境样本和
GPU-hours分别比较。统一超参是第一步，各架构同等调参预算是另一个问题。
16M只是首档；延长预算依据学习曲线和物理表现，而不是依据模型名称。

先检查指令输入/缩放/动作响应、奖励分项和可行性，再把共同的跟踪失败归因于模型。
命令反事实探针须同步修改Last-token当前帧中的指令和外部command，不篡改历史帧；
动作响应检查不能替代闭环评估。

验收同时报告状态估计误差、高度/速度bias、回合内抖动、漂移、失败、目标/力矩变化、
命令切换响应和batch1推理p95/p99。网络forward时延与完整控制链路时延分别测量。
实现估计器时必须固定rollout期间预处理/编码器版本；detach不意味着编码器不会在独立
优化后改变动作分布。dropout或随机latent需明确采样重放语义，不能破坏old-logp一致性。

## 6. 文献核对

以下为原论文方法描述，支持设计可行性，不是本机器人上的效果证明。

| 工作 | 核对到的方法 | 对本项目的启发 |
| --- | --- | --- |
| [Learning Humanoid Locomotion with Transformers](https://arxiv.org/html/2303.03381v2) | 先训练特权teacher，student联合PPO与退火teacher KL；16帧、50Hz，4层192维 | 训练范式与物理时间跨度需要一起比较，不能只复制Transformer层 |
| [State Estimation Transformers](https://arxiv.org/html/2410.13496v1) | 显式估计高度与速度；以非特权/特权状态序列训练自回归估计器 | Transformer可以服务于估计而非直接产生动作；本文候选是简化设计，不是SET逐项复现 |
| [Terrain Transformer](https://arxiv.org/html/2212.07740v2) | teacher轨迹离线预训练，再在student轨迹上由teacher标注在线纠正 | 只拟合teacher轨迹存在分布偏移；原文去掉return-to-go |
| [Unified Locomotion Transformer](https://arxiv.org/html/2503.08997v2) | 末尾特权token、因果隔离、下一状态动作预测、动作模仿和混合探索 | 估计/预测/控制可统一，但监督优势与网络优势要分别归因 |
| [Robust Locomotion Transformer](https://arxiv.org/html/2507.04039v1) | 身体模块token、地形patch、多模态融合及rollout/update一致dropout | 可借鉴结构归纳偏置；其含真线速度/地形输入的任务并不等同于我们的纯本体任务 |
| [Humanoid Locomotion as Next Token Prediction](https://arxiv.org/html/2402.19469v1) | 对传感器与动作序列作模态对齐预测，利用多来源完整/缺失模态轨迹 | 为轨迹预训练提供路线，不代表低质量动作自动成为专家标签 |
| [RMA](https://arxiv.org/abs/2107.04034) | 基础策略与在线适应模块分离 | 模块划分依据；RMA本身不是Transformer论文 |
| [LocoTransformer](https://arxiv.org/abs/2107.03996) | 本体与深度观测融合 | 传感与地形接口具备后再研究多模态扩展 |

原始[Decision Transformer](https://arxiv.org/abs/2106.01345)是回报条件化离线序列学习，
与本项目当前的在线PPO Transformer属于不同训练范式。

## 7. 本次验证范围

完成两仓默认配置/调用链/梯度路径核对、复旦原类CPU参数计数和上述论文阅读。
2026-09-17 18:30的Kaiser状态为监督式1011进行reverse场景评估，
MLP/GRU尚未启动；当前七变体任务的最终对照结论仍待产生。
本文只新增设计记录，后续实验应建立独立配置和运行收据。
