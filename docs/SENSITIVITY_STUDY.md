# 学习率、输出初始化与探索标准差敏感性实验

本实验先核查 LR、动作均值头初始化和探索 std，再以充分样本、多训练 seed 验证候选。
接口见 [SENSITIVITY_INTERFACES.md](SENSITIVITY_INTERFACES.md)，通用编排和产物协议见
[EXPERIMENTS.md](EXPERIMENTS.md)。本节为实验设计和可复现方法；实际训练结果另行追加。

## 1. 先核查优化行为

当前 `configs/control.json` 对应 time_attention：Transformer、elapsed 时间编码、add 残差，
history_length=16，d_model=64，2 层，4 heads，ffn_dim=128；critic_hidden=[256,128,64]。
canonical 默认 LR=1e-4、mean_init_scale=1.0、initial_std=0.2。
mean_init_scale 只作用于构造时动作均值输出层的 weight/bias，不是 forward 输出缩放，
不改变 action decoder、动作裁剪或力矩边界。

对同一固定 rollout 检查初始均值幅度、实际 std、首个 Adam step 后的完整 rollout KL，
以及 PPO update 结束后的完整 rollout KL。结合均值项/方差项分解、归一化均值变化、
early-stop 和实际 optimizer steps，判断问题是否来自第一步变化过大、后续累积变化或探索尺度。
`optimization.kl` 是已有 minibatch 聚合量，`stop_kl` 是触发 gate 的 minibatch 值，
都不能替代 `first_step_kl` / `final_kl` 的固定全 rollout 诊断。

**不能自动选择 KL 最低的候选。** 零 LR、不更新权重同样可能产生很低的 KL，
但并未学习；KL 只用于解释更新行为，还需学习曲线、独立评估收益和任务物理指标。

## 2. 七项筛选配置与公平性

`configs/sensitivity.json` 以 control.json 为共同 base：

| 变体 | 相对 base 的唯一变化 |
| --- | --- |
| baseline | 无 |
| lr_3e5 | ppo.learning_rate=3e-5 |
| lr_1e5 | ppo.learning_rate=1e-5 |
| init_scale_01 | model.mean_init_scale=0.1 |
| init_scale_001 | model.mean_init_scale=0.01 |
| std_01 | model.initial_std=0.1 |
| std_04 | model.initial_std=0.4 |

`sensitivity` 只允许改变上述三个字段，固定 backbone、数据/观测、历史、critic、
辅助监督设置、其余 PPO 超参，以及统一训练/评估预算。共享 environment、action_clip、
rollout_steps、updates 和评估 seed，完成后再次核查实际 transitions。
`architecture` 和 `supervision` 保持原有白名单，不能借此次扩展改变 LR、初始化或 std。

允许后续组合确认，例如降低 LR 同时减小初始化；summary JSON/CSV 中：

- `factor_changes` 为相对 canonical base 的实际配置差异，格式为
  `{"ppo.learning_rate": {"base": 0.0001, "value": 0.00003}}`。
  未显式填写的字段取配置类真实默认，显式重复同值（含 1 与 1.0）不计为变化。
- `single_factor=true` 仅表示恰好一个实际字段变化；baseline 为 false 且变化为空。
- 多字段变化是组合确认，不能将效果归给某一个因素；单因素结论仍需对照、多 seed 和充足样本。

## 3. 预算与训练 seed

示例默认 **seed 11 仅用于筛选**，不是最终结论。正式比较每项至少 **3 个独立训练 seed，
建议 5 个**；例如筛选后锁定候选，用 `[22,33,44,55,66]` 做独立确认，
另用评估 seed `[101,102,103]`。同一组候选使用同一训练 seed 集合，
评估 seed 的重复测量先在每个训练 seed 内聚合，不能当成额外独立训练样本。

必须区分三种计量：

1. **Transitions**：在固定并行环境数 N、完整 rollout 下，约为 `updates × rollout_steps × N`；
   以 completion 中实际 `collected_transitions` 为准。
2. **PPO updates**：每次采样后的一轮优化，不等于一个 Adam step。
3. **Optimizer steps**：每次实际 Adam 更新；KL early-stop 会使其低于 planned steps。
   不为追平 actual steps 而给某个候选额外 rollout，避免改变样本预算。

示例给出 1000 updates × 48 rollout steps，即每环境 48,000 transitions，
名义每 update 为 5 epochs × 4 minibatches；实际 planned/actual steps 以日志为准。
这只是起始预算，不保证所有任务已经收敛。先根据学习曲线覆盖启动期、持续学习期、
稳定期/平台期，再对所有候选统一增加样本；应在分析结果前约定阶段预算和评估点。
不能用一个 seed 的 20 updates 比较 architecture，也不能把此次同 backbone 调参当架构优胜证据。
max_seconds=3600、job_timeout_seconds=4200 是示例墙钟上限；实际硬件上需给足完成预算的时间，
超时/少样本会明确保留为不完整实验。

筛选后比较架构时，使用独立的 architecture 规格和正式多 seed 预算，
组内继续共享初始化、std、LR 和数据协议。若分别为每个架构调参，应另行报告等价搜索预算及选择协议，
不能混称为固定优化配置的纯架构消融。

## 4. Plan、诊断与来源

示例 `environment_factory=null`，只可 plan/summarize，run 会在启动 worker 前拒绝。
以下命令只冻结计划和生成空结果汇总：

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m transformer_rl.experiment_cli plan --spec configs/sensitivity.json --root /tmp/opencode/sensitivity-plan
/home/yukikaze/isaacsim60-venv/bin/python -m transformer_rl.experiment_cli summarize --root /tmp/opencode/sensitivity-plan
```

实际实验须使用已经核查的环境工厂和环境配置，在新 spec/root 中固定其来源和预算。
`execution.max_parallel` 或 run 的 `--max-parallel` 控制并发；每 job 独立训练/优化器，
同 job 内训练、checkpoint 核验、独立评估依序进行。并发数需按实际显存与吞吐设置。

`training.diagnostics` 是可选 boolean，缺省/false 均不传参数，保持旧 spec 的 argv 行为；
true 仅给训练 worker 添加 `--diagnostics`，评估不传该 flag。
示例开启诊断；正式实验组内保持一致设置，并计入诊断重算的墙钟开销。

源码 SHA 基于实际导入包的相对文件路径与内容，包含未提交修改。
相同代码位于本机/远端不同目录仍产生同一 hash；未执行 plan 可以整体迁移后运行。
run 强制匹配源码，summarize 允许本机包源码与远端记录不同，但仍验证冻结 plan/spec/config
和产物 SHA，配置语义也必须兼容。不同源码 hash 不会因汇总而被改写成本机来源。
已执行产物中的绝对 checkpoint 路径限制仍适用；不能假定搬移训练结果后可以直接汇总。
外部工厂、仿真依赖和资产不在 package SHA 内，应在实测记录中单独固定。

## 5. 读取 summary

逐 seed 的 `optimizer_diagnostics` 尽可能从 `train/metrics.jsonl` 提取：

- `first_update.*`、`last_update.*`：首/末可读 optimization 记录中的初始动作分布、
  first_step KL 及分解、final KL 及分解、均值变化、std、gate KL 和 step 数。
  “初/末”是训练日志端点；每个 update 自身的 first_step/final 才表示固定 rollout 上的两次诊断。
- `optimizer_steps`、`planned_optimizer_steps`：可读 updates 的各自总计，
  `optimizer_step_utilization=sum(actual)/sum(planned)`，不是每 update 比例的简单平均。
  任意可读 update 缺少有效 step 数时不编造对应总量；planned 缺失/总计为零，或 actual 超过 planned，
  利用率为 null。旧日志只有 actual 时仍可汇总 actual，不用配置估算 planned。
- `early_stopped_fraction`、`updates_observed`、`invalid_metric_rows`：早停比例和日志覆盖情况。
  不完整末行跳过并计数；诊断只描述可读部分，不能证明全部更新已完成。

没有执行首步时 first_step 字段为 null；没有新诊断的旧日志保持缺失，不以零替代。
变体级同名指标按独立训练 seed 计算 n/mean/样本 std，CSV 以
`optimizer_diagnostics.*` 为 metric。失败/超时 job 的现有训练诊断也可展示，
其 seed 状态仍显式保留；这与仅使用完整 checkpoint+评估报告的 evaluation 聚合不同。

`comparison_available` 仅表示组内配置、完整性和实际样本预算检查通过，
不保证训练 seed 数达到正式标准或模型已充分学习。必须结合 requested/completed、
训练与评估 transitions、诊断 n、学习曲线和任务指标解释结果，不挑最佳 seed，不自动给 winner。

## 6. 接口验证

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest tests/test_experiments.py tests/test_sensitivity_experiments.py -q
```

覆盖三因素白名单及旧分组公平性、canonical 因素归因、组合标记、diagnostics true/false/缺省、
空工厂禁止启动、实际样本预算/缺 seed、优化日志端点和总 step 利用率、
本机/远端路径无关源码 SHA 及源码不匹配的 run/summarize 行为。执行载荷仅为 fake runner。
