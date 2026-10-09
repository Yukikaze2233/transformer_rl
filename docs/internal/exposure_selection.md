# 完整 validation 矩阵的不可变选型

`exposure_selection` 在全部原始训练 job 及 validation worker 关闭后，重新核验原协议、实际训练链、完整物理轨迹并封存控制质量选型。controller 可以仍然存活并持有原来的两个锁，以便后续在同一授权下运行 held-out。该选型是 **provisional**：未通过确切导出产物及目标部署运行时的真实 latency 门槛，不能称为可部署资格或硬件验证。

## 输入与封存接口

`freeze_selection(protocol_path, expected_protocol_sha256=...)` 要求外部提供协议**原文件** SHA，生成一次性的 `output_root/selection/choice.json`，返回 choice 与实际文件收据。目录已存在时不允许重写、续签或自动重试。`verify_selection(selection_receipt)` 不信任 choice 的摘要分数，而是从原始输入重建整份选择并比较全部内容。`authorize_heldout_batch(selection_receipt, endpoint_receipt, cells, directory)` 仅授权原协议完整 role/seed batch 及已固定的确切 endpoint。

封存包含原协议 raw receipt、代码与运行时身份、实际 storage 合同及校准收据、每个 job 的完整 reservation、所有实际阶段 endpoint/checkpoint/训练 completion/optimizer 日志、每个 validation batch 的 request/worker/完整 NPZ/report/completion、每个 cell 的实际成绩及全历史样本门槛、所有候选和训练 seed 的分母、最佳 eligible Transformer、MLP 对照，以及原矩阵**全部候选×训练 seed×阶段×场景×held-out seed** 的固定 endpoint/checkpoint。每个冻结输入同时保存实际 device、inode、size、mtime_ns、ctime_ns、nlink，构造收据前后与重放后均核对元数据稳定、当前用户拥有的普通文件及无符号链接路径。

held-out 的快速授权只接受调用方保留的、前置实际 `freeze_selection` 返回的**确切外部收据**；核验 choice raw SHA、原协议/source、所有冻结输入的实际元数据签名、原始 missing 缺项以及当前完整 batch 的确切 endpoint/CP。这个 stat 快检不是独立内容重放，不能接受 CSV 或自行补签的摘要作为前置证据。它避免每个 held-out batch 反复读取全部大轨迹；封存和最终 `verify_selection` 仍执行全 hash 和完整实际 replay。任何签名变化立即拒绝；恢复原字节也不能恢复 ctime/原 inode，不允许自动重新封存。

缺失或合法失败保留在原始矩阵中。数值失败仍收费整个原 job 预算，不更换 seed，不补跑阶段，不替换 checkpoint。失败更新的 optimizer 子步骤只有实际记录可确知时才报告整数；`optimizer_update_may_be_partial=True` 且计数为空时保留未知，不宣称完整 optimizer 工作量。没有 endpoint 的 held-out cells 保留空 endpoint 和原因，不能转用另一个模型。自动 campaign 遇到未知错误或评价 worker 非正常终止时停止整个流程。单独核验已有选择材料时，具有合法终态记录的失败评价 batch 保留 cells missing，使相应候选不合格；它不生成正常成绩，也不重跑该 batch。未知训练错误、运行中的 worker、身份漂移或证据篡改会拒绝封存。

## 最终控制质量与遗忘

所有阶段均保存独立评估数据，但最终排名只使用**最后阶段**。对于候选 \(a\)、训练 seed \(s\)、最终阶段 \(K\)，各场景和 validation seed 等权：

\[
q_{a,s}=\frac{1}{|\mathcal C||\mathcal V|}
\sum_{c\in\mathcal C}\sum_{v\in\mathcal V}q_{a,s,K,c,v},\qquad
Q_a=\bar q_a+\lambda\,\mathrm{SD}_{\mathrm{sample}}(q_{a,s}).
\]

单 cell 的 score/gates 使用原 `protocol.selection` 与 `frame_study.grade_report`。标准差使用 \(n-1\) 分母，表示跨训练 seed 波动，不是置信区间。只要一个原始 seed 的最终目标缺失，候选 mean/std/rank_score 保留为空，不能用其余 seed 均值替代；`min_training_seeds` 是下限，不能用来丢掉坏 seed。

### 按场景使用物理控制指标

objective 可以用可选的 `scenarios` 声明适用的完整场景名称；省略时仍用于全部场景。名称必须来自原 spec，列表不能为空或重复，并且每个场景至少有一个适用 objective。比如零指令站立的世界 XY 速度范数可以参与该类场景评分，不应把前进指令下的实际运动当成静止漂移。单 cell 只累加其适用项，场景间仍保持原等权；没有适用项或适用项缺失时 score 为空，不能填零或把该 cell 从分母中删除。

`path` 的点号片段支持字典字段和列表的十进制下标，如 `control.actuation.effort_rate.channels.2.rms`、`control.actuation.actual_bound_fraction.2`。列表下标从零开始，不接受负数、前导零或超出实际通道的索引；缺失、布尔值和非有限数值均不能评分。gates 使用同一取值规则，因此实际力矩及其变化率可以同时作为评分项和约束。通道顺序、物理单位与控制配置一起冻结。

固定指令的跟踪偏差应与回合内去均值波动分开：`control.steady.axes.height.rmse` 描述高度是否正确，`within_group_std` 描述保持段内部的波动。动态指令场景可以另行声明 `control.full_interval` 项；不能用不存在的稳态窗口、零值或静止漂移目标替代。`control.planar_motion.stationary_steady.rms_speed_m_s` 是世界 XY 的 base-link 原点速度范数，和 body vx 误差不是同一个指标。`control.actuation` 的目标变化率、轮目标加速度及力矩变化率采用采样间隔统计，不能解释为去均值稳态量或下位机电流环。

上述接口只增加可表达的指标与适用范围；原来的 objectives、权重和 gates 不自动改变。新评价规则须先写入独立 spec，并在训练与 validation 前冻结尺度、权重及适用场景；不根据结果改规则或强求 Transformer 胜出。修改源码后重新 prepare/freeze，旧封存计划继续保留原身份。

所有原训练 seed 必须正常完成、全部 validation cells 必须有合法数据、最后阶段所有场景必须通过任务 gates。实际 `history_control` 的 H、`minimum_full_age=H-1`、policy_dt、settle_steps 和 min_steady_samples 必须与原 config 一致。full-history 样本至少达到 `min_steady_samples × cell.num_envs`；要求稳态的场景，其 full-history steady_tracking 样本也必须达到该门槛。不足时保留实际分数并判为不合格，不能用 reset-filled 样本填充。

遗忘只在技能已获得后判断：同一场景、同一 validation seed 首次通过 gates 后获得技能，后续阶段必须继续通过 gates，score 不能比此前已获得阶段的最佳 score 恶化超过原 retention_score_tolerance。早期尚未学会的新技能不被视为遗忘，也不混入最后阶段排名；获得后退步则影响 eligibility。完整早期曲线仍保留在选择证据中。

最佳 Transformer 从全部 eligible Transformer 中按 \(Q_a\) 最小选择，另保留 best_overall 与所有 MLP/history-MLP 对照。候选代表训练 seed 选最接近该候选最终 score 中位数者，平分时取 seed 数值较小者，checkpoint 固定为该 seed 的最后阶段。所有网络均不合格时返回 `no_eligible`，不假定总能选出胜者。

## Held-out 与部署

封存前要求全部 held-out 输出目录不存在。封存后完整 held-out 比较只观察原矩阵和已固定的全部阶段 checkpoint；不允许根据 held-out 指标调换网络、训练 seed 或阶段。held-out 输出不参与 `verify_selection` 的选择重建，validation 证据变化则会使 seal 失效。每个实际 worker 的一次性目录和原协议 role/seed 身份防止重复观测和缩小矩阵。

choice 记录原协议要求的 latency_p99、latency_max 和 deadline miss 上限，但明确 `latency_gate_applied=False`、`deployment.status=unverified`。必须对选定的确切 CP 执行真实 export/数值一致性校验，并在目标运行时重新 benchmark 后，才可讨论 100 Hz 部署。控制轨迹的 policy-rate 力矩和机械功率 proxy 不证明高频电流环、真实能耗或传输到达时刻。

## CPU 验证范围

`tests/test_exposure_selection.py` 的算术测试使用显式 logical grade fixtures，仅验证最终阶段排名、样本标准差、完整 seed 分母、技能获得后的 retention 和无 eligible 结果。独立 OS 集成使用最小合法完整矩阵，执行真实 PPO/Adam、原始 checkpoint、实际 `-m` worker、父进程身份、两个继承的 flock、异步 reset、完整磁盘 NPZ 与逐场景重放。默认自动流程先关闭全部 validation，再实际封存与重建 choice，最后完整运行所有可用 CP 的 held-out 比较；只读 phase observer 验证封存时原父进程和两个 FD 仍活跃、held-out 目录均尚未创建。测试拒绝补签摘要、改 CP、缩小 cells 及伪造 report。

CPU 集成的 SDK-pinned `sitecustomize.py` **只替代原队列定义/闭合 provider 与 Isaac factory resolver**；测试启动环境额外注入 SDK PYTHONPATH，以启用这个显式 fixture，生产 `_worker_environment` 白名单不改变。六电机运动、场景成功标志与 permissive gates 均为合成物理 fixture。原始 Counter 与 report 标签属于代码固定的 owned-worker attribution，不能称为独立 PhysX 场景行身份验证。实际 storage 校准是 CPU fixture 的四类真实文件/NPZ/缓存清单，不能据此声称生产 Isaac 内存、缓存或磁盘上限已校准。未启动生产 campaign，未证明原生产队列关闭，未授予硬件资格。
