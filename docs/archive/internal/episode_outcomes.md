# 完整回合与物理健康统计

`evaluate_frame_policy` 在全局报告和各场景分组的 `episode_outcomes` 中，保存每个已声明环境行 reset 后**首回合**的结果。每行只贡献一个请求；失败后反复自动 reset 不增加这项统计的分母。原 reward、跟踪误差、漂移、动作变化与稳态统计继续覆盖各自原本声明的采样范围。

## 连续控制与离散任务

只有场景合同明确声明 `task == "survive"` 的行参与存活率。站立、速度和高度指令等连续控制使用相同的固定请求分母：

\[
S=\frac{N_{\mathrm{full\ horizon}}}{N_{\mathrm{requested,\ survive}}},\qquad
S_{\mathrm{healthy}}=\frac{N_{\mathrm{healthy\ full\ horizon}}}{N_{\mathrm{requested,\ survive}}}.
\]

完整存活同时要求：回合已结束、环境明确标记 timeout、PRE-reset `episode_ticks >= episode_horizon_ticks`，且没有环境失败、任务成功、越界或受阻原因。每行使用自己的时限，不能用一个统一的20秒阈值替代不同场景的合同。

健康完整存活还要求整个首回合没有高度低于0.20 m或倾角高于0.60 rad、连续达到0.20 s的异常。异常一旦达到持续时间就保留，即使之后恢复；reset 后暖身段也计入。这个健康条件不要求目标高度、速度、角速度、漂移或动作平滑达标，相关控制指标仍须分别评估。

跳跃和台阶等 `jump`、`traverse` 行的存活指标为 `null/not_applicable`，使用固定首回合的 `task.task_success_rate`。任务成功可以在学习接口中表现为 `terminated=True`；失败终态使用环境独立的 `environment_failure`，不能把任务成功误计为失败。连续控制的最低高度规则也不应用于离散任务的预期腾空阶段。

`environment_failure` 保存环境定义的失败，可能包含起跳或任务结果超时等原因，不能把这个计数全部解释为机械跌倒。

## 原始来源与未完成回合

`ChassisFrameAdapter` 仅在评估显式启用后输出独立的 `evaluation_episode`，不改变训练收集器、reward、环境 outcome 或策略网络：

- `episode_ticks`、`episode_horizon_ticks`：PRE-reset int64计数与场景时限；
- `time_out`、`environment_failure`、`task_success`：原环境的独立终止事实；
- `boundary`、`blocked`：明确的截断原因；
- `survival_applicable`：来自场景任务声明，而非网络表现。

这些数据在 reset 前复制。缺少任务声明、时限或必要原因时不能根据时间戳猜测成功：缺少整个接口时统计为 `unavailable`，启用接口后数据非法则拒绝报告。没有启用 step assistance 的环境可明确将缺少的 blocked 原因设为False；启用后缺少原因则拒绝。健康检测使用同一PRE-reset端点的高度和倾角。

报告列出 `requested/started/completed/censored/not_started`、互斥结束原因和每行首回合结果。尚未结束或未启动的请求不计为成功，分母保留；存在这类请求时，比例只能解释为已观察成功的下界。`all_requested_accounted` 表明所有请求都有终态。`censored_sample_fraction` 仅统计这个首回合cohort中的样本，不能与全采样区间的样本比例混用。

这项统计不覆盖首回合结束后才发生的扰动；全区间控制与扰动评估仍单独保留。每个训练seed也必须单独评估，不将重复评估seed当作独立训练重复。

## 与既有指标的关系

既有 frame 入口的 `success_rate` 仍为 adapter 的 `episode_success / completed_episodes`，覆盖自动 reset 后的全部已结束回合，并通过 `success_rate_scope` 明确口径。它可能包含普通任务的越界、受阻等提前截断，不能当作严格完整存活率。

旧通用 `evaluation.py` 的 `ControlQuality.healthy_timeout_fraction` 是另一套接口：已结束回合中没有终止、没有持续物理异常且以 truncated 结束的比例。它既不检查完整时限，也不使用固定请求分母；不能称为当前 frame 入口已经记录的同一指标。

新协议标识为 `transformer_rl.first_episode_outcomes.v1`。场景报告保存各自统计；suite的control汇总同时保留全局 `episode_outcomes` 和 `group_episode_outcomes`。原控制trace布局保持原定义，每行首回合的终止原因与时限证据保存在新报告中。旧报告缺少明确时限及截断原因时，不追补严格存活率。已有冻结训练和评估继续按其冻结源码执行，不追改历史数据或选择协议。
