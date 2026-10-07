# 历史窗口架构研究准备接口

`transformer_rl.history_study` 把一份完整的 FrameStudy 规格扩展为新的历史窗口对比。接口只准备和验证，不启动环境、训练、评估或队列。准备成功表示配置和源文件已冻结，不能据此认为训练已完成或已获得合格教师。

输入必须显式提供原 spec、它实际引用的 base 配置、新输出路径、历史长度列表以及 `position_reference=current`。长度必须是至少两个互不重复的正整数，并包含 H1。原规格须保留单帧 MLP、历史编码 MLP 和 Transformer 三类比较；每种原始网络宽度、门控方式或 query 读出分别保留自己的身份。单帧 MLP 仅生成 H1；每个历史 MLP、每个 Transformer 变体都生成全部 H 候选。不会加入 RNN 或直接堆帧 MLP，不裁剪 H31 模型，不复用其他 H 的 checkpoint。

例如显式选择 H1、H11、H31、H61，在 100 Hz 策略频率下，最老帧到当前帧的跨度分别为 0、0.10、0.30、0.60 秒。跨度使用输入中的实际策略周期计算，而非固定写死 100 Hz。采样步长固定为每次策略步一帧，窗口从旧到新；reset 使用首帧重复填充。只有回合年龄 `age >= H - 1` 才拥有完整的真实历史；不足龄窗口必须单独记录，不能混为完整历史收益。Transformer 使用相对当前帧的整数位置，当前帧为零，使不同 H 的共有帧年龄对齐。这里不增加时间降采样、KV cache 或跨回合记忆。

除了 H 和 Transformer 时间参考，完整有效架构保持原来的 actor、encoder、latent、readout、层宽、层数、head 数及门控设置；观察和动作维度、初始均值缩放、初始探索标准差、critic、PPO、控制接口、环境参数、全部阶段、门槛和种子保持原值。它不修改学习率、课程或 reward，也不强行匹配参数量。历史 MLP 第一层参数量随 H 增加，Transformer 参数量通常保持相同但注意力计算量变化；实际时延和显存需由后续测量获得。

研究要求至少三个训练 seed、两个 held-out 评估 seed，selection 至少需要三个训练 seed；anchor、validation、训练及 held-out 种子沿用 FrameStudy 的非空、唯一、两两互斥校验。不会因为某候选失败而删除候选或缩小分母。全部声明的候选×训练 seed 都属于最终比较分母；缺失、提前拒绝和未完成项必须保留。

## 冻结与验证

新目录通过独占创建获得，拒绝复用或覆盖已有目录，拒绝 symlink 及与输入树、learner 源树重叠的输出。原 spec 和 base 的实际字节、长度及 SHA256 保存为收据，并在 `inputs/` 保留原样副本。完整展开规格保存在 `expanded_spec.json`，`study/` 是既有 `plan_study` 和 `validate_study` 直接消费的完整 packed study。learner 包包含本规划器在内的实际源文件由 FrameStudy 冻结；`history_plan.json` 同时记录本模块、完整包源身份、plan、全部固定输出文件的实际 SHA 和长度。

`history_plan.json` 最后写入并同步。中断留下的目录没有有效封存 manifest，验证不能报告准备成功；重试需另选新路径。验证重新读取原始输入及冻结文件，从真实 spec 和 base 重建候选、全部不变项、预算和执行语义，调用原 `validate_study(..., source=True)`，核验真实目录清单，拒绝缺项、额外固定文件、篡改或仅重签 JSON 的预算更改。输入或当前 learner 源变更后必须准备新研究。`study/jobs/` 是未来执行器的可变产物区域，不纳入准备时的静态文件收据；原 runner 的精确 `study/.run.lock` 也不纳入静态收据。仅在 `study/policy_source/transformer_rl/` 内，按原 `validate_study` 的源文件规则排除 `__pycache__` 下的缓存及 `.pyc`，允许真实 worker 生成可重建 bytecode。其它位置、配置目录或陌生根文件不获得此豁免；被排除的路径仍检查 symlink 和不支持的特殊文件。准备 manifest 的 `preparation_started_training=false` 仅声明本次准备器没有启动训练。

验证返回值和 CLI 另行实际查看 `study/jobs/`：空目录报告 `execution_state=unobserved`，有条目报告 `job_artifacts_present`。这个动态观察字段不写入封存 manifest，也不参与静态 SHA；它既不能证明训练从未发生，也不能凭 artifact 或其中的 status 声称进程仍在运行、已完成或产物合格。后续使用原 runner 生成 job 产物后，静态计划仍能验证，但不会输出恒定的“训练未发生”结论。

此接口冻结本项目的 learner 源与输入配置。外部环境工厂、SDK、机器人资产、环境声明中的外部文件及硬件时序仍须由执行协议绑定和验证；静态规划没有加载它们，也不授予实际 sim2real 证据。

## 预算与执行语义

每个实际展开的训练及评估环境必须显式声明正整数 `environment.num_envs`；缺失或非法值拒绝准备，不默认 1024。每阶段的新 rollout 上限为：

`stage_updates × rollout_steps × resolved_num_envs × candidate_count × training_seed_count`。

这是完整新 rollout batch 的请求上限，不是 PPO 重复使用样本数，也不是已完成更新。部分 rollout、失败或未封存 worker 的真实采集量需另用运行时账本记录，不能由本公式证明。报告保留每阶段实际环境数、全部 job/update/sample 上限、累计需要评估的场景、每个 checkpoint 的 validation 单元、promotion 的 anchor 单元以及 stage endpoint 的 held-out 单元。一个评估单元指候选×训练 seed×场景×评估 seed；环境 replica、完成回合和稳态有效采样数不能与之混用。最终 held-out 数量按所有场景计算，完整统计不得删去失败候选。

现有 FrameStudy runner 会在 checkpoint 检查累计场景，阶段间使用 `initialize-from`，因此会重置 Adam 和学习 RNG；阶段内 resume 继承状态，而 retention 回归可能恢复 protected checkpoint。promotion 门槛、rollback 限制和提前拒绝会改变实际后续暴露。runner 只在完成全课程后执行最终 held-out 评估；本计划列出的阶段 endpoint held-out 单元是后续研究的完整义务，并非现 runner 已提供的产物。checkpoint 数量字段是按完整 chunk 计算的计划值，失败或部分更新需用真实账本记录。

因此 manifest 明确标记 `equal_exposure_comparison_ready=false`、`needs_independent_equal_exposure_executor=true`。最终固定样本量的 H 架构比较需另行执行器确保所有候选的相同暴露预算，并记录真实请求/尝试/完成更新、新 rollout、优化器与 RNG 恢复方式、checkpoint 父节点、promotion/rollback/rejection、全部缺失单元、时间和计算开销。本接口不能用于声称固定 K/λ 的公平完整状态续训，也不能直接给出教师资格、遗忘率、记忆容量收益或架构胜者。

## 调用

所有路径需换成真实绝对路径；输出的父目录必须已存在，且位于输入树之外：

```bash
python -B -m transformer_rl.history_study prepare \
  --spec /absolute/input/spec.json \
  --base-config /absolute/input/base.json \
  --output-root /absolute/output/new_history_study \
  --history-lengths 1 11 31 61 \
  --position-reference current

python -B -m transformer_rl.history_study validate \
  --output-root /absolute/output/new_history_study
```

API 为 `prepare_history_study(spec_path, base_config_path, output_root, *, history_lengths, position_reference)` 和 `validate_history_study(output_root)`，返回完整封存 manifest 的内容及独立的动态 `execution_state` 观察字段。这里的单元测试只验证 CPU 配置、真实模型前后向、原 study 接口及文件完整性，不运行机器人仿真或远程训练。
