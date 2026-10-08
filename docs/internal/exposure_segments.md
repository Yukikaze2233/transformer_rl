# 跨操作系统进程的固定阶段训练

`exposure_training.train_exposure_segment()` 每次只执行一个固定阶段，允许外层执行器为每阶段启动独立 OS 子进程。原有 `train_exposure_job()` 的公共参数、fresh 任务格式和默认行为保持不变；两种入口共用完整 rollout、优化、日志与端点发布循环。

```python
train_exposure_segment(
    stage, env_factory, environment_reference, output_root,
    job_id=..., rollout_steps=..., training_seed=..., retention_seed=...,
    evaluation_seeds=..., device=..., expected_initial_model_sha256=...,
    max_seconds=..., parent_endpoint=None,
    should_stop=None, protected_paths=(),
)
```

`stage` 为 `{name, config, updates}`。第一个阶段使用 `parent_endpoint=None`，其阶段索引为零。后续阶段必须提供上一个成功端点的实际文件凭据：

```json
{"path":"/absolute/previous/stage_0000_stage_a/endpoint.json","sha256":"64 lowercase hex characters","bytes":1234}
```

示例中的 SHA 和长度必须替换为实际文件值。下一阶段索引严格为父端点索引加一。接口继承完整模型、Adam、全局 RNG、私有 RNG 与累计学习时钟，显式使用 `resume=True, environment_transition=True`。环境和历史重新 reset，历史采用 repeat-first；不恢复物理 episode。

## 输入证明与输出保护

创建新输出目录和调用环境工厂之前，接口核验父端点的 canonical JSON、实际字节长度与 SHA、端点 schema、checkpoint 与 sidecar 的实际 SHA，以及 CPU 加载的完整学习状态。

同一父输出目录中的 request、reservation、completion 和逐行 metrics 都参与证明。父 completion 必须清洁完成全部声明预算，端点必须是最后一个且唯一匹配的已封存端点。关闭失败、同步失败、未完整发布、部分 rollout、未封存 PPO 更新、教师或其它任务 checkpoint 都不能作为父输入。一个已完成多阶段 fresh 任务只允许使用其最后端点，避免隐式回滚。

父请求、checkpoint metadata 与新请求必须绑定同一 job identity、训练和私有 seed、评价 seeds、环境工厂、实际 package 文件 SHA、设备、rollout 长度及原始初始化 guard。model、PPO、control 保持相同，仅 environment 可以变化。初始化 guard 始终指向原 seed 构造的初始模型，不以当前已学习的模型 SHA 代替。

对分阶段入口生成的父请求，接口继续核验之前的实际端点、request、reservation、completion、metrics 凭据。阶段索引逐次减一；当前学习状态的 `continuation_parent`、环境转换 SHA 和累计时钟起点必须匹配真实父端点。全部输入链的目录加入保护树，输出不得覆盖、包含或位于其中。

兼容已有多阶段 fresh job 时，接口读取每个阶段实际保存的 endpoint、checkpoint 和 sidecar，而不只检查 completion 中的端点标签。第一个 checkpoint 必须没有 continuation 来源；之后每个 checkpoint 的真实父路径、SHA、更新边界和 environment transition 都须匹配上一个实际封存端点。整个输入链的实际文件在检查结束时再次核验。

checkpoint 内实际 Adam 的完整 parameter group 必须与同一冻结 PPO 配方新建的 Adam 一致，包括学习率、betas、eps、weight decay 和更新选项；这里没有改变配方的 scheduler。每个参与 PPO 的参数都必须具有与已核验全链优化日志相符的累计 Adam step，不能只以“Adam 状态非空”证明学习发生。累计优化步为零时，只有真正尚未创建参数状态的 Adam 才符合边界。

逐行优化样本量也按真实 `torch.tensor_split` 顺序验证。对一个完整 rollout 的 `N` 个样本，令 `C=min(num_minibatches,N)`，已应用优化步 `S=qC+r`，则其样本使用数必须为：

```text
q × N + r × floor(N / C) + min(r, N mod C)
```

这包含完整 epoch 和最后一个 epoch 的 minibatch 前缀，允许 KL 提前停止及零个应用步。优化日志与 completion 同时改成另一组自洽数字，也不能覆盖实际 checkpoint 的 Adam 计数。

这份证明核验学习状态和记录的连续性。实际环境源码、资产与运行时 contract 仍由完整会话在环境构造后核验；CPU 输入检查不代替模拟器或实机验收。

## 单次总预算与阶段分项

外层执行器一次性预留整个 job 的更新和样本预算。本入口写入的 reservation 是该预算中本阶段的分项，不再创建整任务收费：

```text
charge_scope = segment_itemization_of_external_whole_job_reservation
whole_job_reservation_created = false
```

每阶段的完整样本预算为 `updates × rollout_steps × num_envs`。阶段环境数可以不同，但必须与实际 collector 一致。低 reward 不改变暴露预算，不执行成绩 gate、回滚、自动重试或退款。

分阶段记录使用独立格式：`exposure_segment_request`、`exposure_segment_reservation`、`exposure_segment_endpoint` 和 `exposure_segment_completion`，均位于 `transformer_rl` format 命名空间。新 request 保存父输入的实际凭据，端点保存本阶段与累计成功边界，completion 分开记录局部和累计时钟。

`successful_updates`、`attempted_updates`、`actual_collected_transitions` 只表示本段实际发生的数量；`cumulative_successful_updates`、`cumulative_attempted_updates`、`cumulative_collected_transitions` 包含真实父时钟。优化步和优化样本使用量也只计本段成功返回的 PPO 更新。不能将各段累计值相加作为总样本量。

## 失败处理

部分采集、deadline、调用者停止和用户中断保留实际计数及全部预留分项，不封存当前阶段端点。PPO 可能已修改参数后抛错时，尝试更新与实际采集仍记入本段及累计时钟，失败更新的优化步记为未知。日志发布失败不抹掉已经返回的成功学习更新。

文件发布失败可能只留下部分目录或单独 checkpoint。输出目录不允许复用；真实 `.pt` 文件本身不证明父阶段可续训。外层必须保留总预算账本、准确核验子进程结果，并在全部阶段完成后独立评估。该 API 的 `completed` 仅证明本段固定训练预算完成，不表示运动控制合格或已经选出最佳网络。
