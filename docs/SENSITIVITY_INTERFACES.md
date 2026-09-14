# 优化敏感性实验接口

目标：区分学习率、动作均值头初始化和探索标准差对Transformer早期KL/控制行为的影响。短测只筛选优化配置，不据此判断架构优胜；后续以相同样本预算、多训练seed和独立场景比较。

## 初始化配置

`ModelConfig.mean_init_scale: float = 1.0`：有限非负，仅在构造时乘到动作均值输出层的weight和bias；默认1.0完全保留原参数和RNG行为。不在forward再乘一次，不改变动作decoder或力矩边界。所有actor实现一致语义，MLP使用最后一层，GRU/Transformer使用mean_head。describe/export需明确这是初始化因子。

旧schema 1/2 checkpoint通过精确字段迁移补mean_init_scale=1.0；新保存格式显式记录新配置和来源schema。旧权重、参数顺序、下一次Adam更新须保持兼容。

## 可选优化诊断

`PPOTrainer.update(batch, *, diagnostics=False)`。诊断不能改变loss、KL early-stop、梯度更新顺序或RNG。开启时，在首次optimizer.step后及update结束时，分块重算整个固定rollout的分布并报告：

- `initial_mean_abs`、`initial_std_mean`、`initial_std_min`、`initial_std_max`。
- `first_step_kl`、`first_step_mean_kl`、`first_step_std_kl`。
- `first_step_mean_change_rms`、`first_step_normalized_mean_change_rms`（除old std）。
- `final_kl`、`final_mean_kl`、`final_std_kl`、`final_mean_change_rms`、`final_std_mean`。
- `planned_optimizer_steps`、已有actual optimizer_steps/stop_kl/early_stopped。

first-step未执行则相应字段为null。分块结果按真实样本数加权，KL(old||new)分解为均值项及方差项；诊断数值可用float64，但不改原PPO gate的数值路径。CLI增加`--diagnostics`，run.json记录是否开启，默认关闭保留旧性能和结果。

## 实验规格

在现有groups之外增加`sensitivity`。该group固定骨干/观测/历史/critic/环境/任务预算，只允许`model.mean_init_scale`、`model.initial_std`和`ppo.learning_rate`改变；可包含后续组合确认，但必须在summary记录每项相对base的`factor_changes`和`single_factor`，不能把组合效果归给一个因素。

training可选`diagnostics: bool`，传入worker CLI。architecture/supervision已有公平性限制保留，不能为了调参放宽它们。默认敏感性配置以当前time_attention为base，baseline、lr=3e-5/1e-5、mean_init_scale=0.1/0.01、initial_std=0.1/0.4共7项。

正式训练设计需明确transitions、PPO updates和optimizer steps三个不同计量，按学习曲线阶段增加样本；配置筛选seed与最终评估seed分开，不以一个seed的20次更新比较网络。
