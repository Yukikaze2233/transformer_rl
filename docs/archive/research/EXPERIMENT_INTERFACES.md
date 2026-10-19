# 多架构实验接口

本功能保持原有默认网络行为，增加可对照的架构和批量实验流程。实际机器人任务由显式环境工厂提供；Kaiser研究任务的真实pilot与合成测试分别记录，均不代替硬件一致性验证。

## 配置扩展

`ModelConfig`新增：

- `actor_type: str = "transformer"`，可选`transformer / mlp / gru`。
- `time_encoding: str = "elapsed"`，Transformer可选`elapsed / index`；index仅消融历史位置编码，frame中的sensor age和policy interval保留。
- `residual_type: str = "add"`，Transformer可选`add / gated`。
- `auxiliary_indices: tuple[int, ...] = ()`，可选从critic取出的监督目标列，必须唯一、合法、显式配置；只在Transformer使用。列号本身不代表线速度，环境必须声明其语义。
- `baseline_hidden: tuple[int, ...] = (128, 64)`，历史MLP隐藏层。
- `gru_hidden: int = 64`，有限窗口GRU状态宽度；每个窗口重新编码，不引入跨调用隐藏状态。

`PPOConfig`新增`auxiliary_coef: float = 0.0`。大于零必须有显式auxiliary_indices。辅助loss是有效样本/目标维的MSE，单列记录，不与纯架构对照混淆。

`ActorCritic(config)`根据actor_type选择actor。默认TimeAwareActor的state key和参数注册顺序保持兼容。所有actor维持forward/act/evaluate/forward_tensors五输入接口；共同输入信息含历史frame、valid和真实age、当前command。各架构提供可序列化的`describe()`字典，供inspect/export显示真实语义。

带辅助头的actor提供`predict_auxiliary(history)->[B,len(auxiliary_indices)]`；head接受query表示，不把critic真值拼入actor。没有辅助头时调用应拒绝。原始检查点schema必须通过显式默认迁移继续可读；新配置和各actor的导出元数据不能冒称都使用Fourier attention。

## 实验规格

实验由`experiments.py`管理，独立CLI为`python -m transformer_rl.experiment_cli`，子命令`plan / run / summarize`。规格路径中的base_config相对规格文件解析。

```json
{
  "base_config": "control.json",
  "environment_factory": null,
  "seeds": [11, 22, 33],
  "variants": [
    {"name": "time_attention", "model": {}, "ppo": {}, "group": "architecture"}
  ],
  "training": {"updates": 1000, "rollout_steps": 48, "max_seconds": 3600, "checkpoint_interval": 100, "action_clip": null},
  "execution": {"devices": ["cuda:0"], "max_parallel": 1, "job_timeout_seconds": 4200},
  "evaluation": {"steps": 2000, "seeds": [101, 102, 103], "environment": {}}
}
```

默认规格可规划但environment_factory=null时禁止run，不能用占位任务假装机器人已训练。每个variant/seed独立进程、run目录和优化器；并发上限不是硬件容量的测量。plan冻结合并后配置与规范化规格SHA，重复输出目录拒绝覆盖。调度器不能用shell拼接用户命令，应使用argv、独立process group、有界timeout、停止并回收其创建的子进程。只有成功且完成全部更新的训练run才进入评估；已有训练不自动覆盖或重复执行。

训练命令复用`python -m transformer_rl train`。评估命令为`python -m transformer_rl evaluate --checkpoint PATH --config MERGED_CONFIG --env-factory MODULE:CALLABLE --steps N --seed N --device DEVICE --output PATH [--action-clip X]`。execution可选`worker_module`替换这两条命令的`-m`模块，供需要显式SDK进程所有权的环境使用。评估config保留checkpoint的model配置，其environment是common base environment与统一evaluation.environment合并后的配置。

## 独立评估

`evaluate_policy(checkpoint_path, env_factory, environment_config, steps, seed, device, action_clip=None)->dict`位于`evaluation.py`。env_factory为callable，签名与训练一致；环境必须自行以配置实现固定命令/确定性评估场景。

返回包含`checkpoint_sha256`、`seed`、`vector_steps`、`transitions`、`reward_mean`、`terminated_count`、`truncated_count`、`environment`、`action_clip`、`policy="deterministic_mean"`和`metrics`。可选的`StepResult.info["evaluation_metrics"]`为`dict[str, Tensor[N]]`，是环境提供的PRE-reset物理指标，必须有限；core按指标返回`mean / rms / min / max / count`，不伪造p95或接触对象身份。没有物理指标时明确只报告reward和终止计数。评估无optimizer更新。

summarize汇总JSON/CSV，按variant统计完成/失败/超时/缺失seed，训练收益与独立评估分开；评估先对每个训练seed的evaluation seeds取均值，再跨训练seed统计mean/std。不能把帧数当成独立实验样本，不能只挑成功seed或只凭训练reward声明赢家。环境、预算或evaluation协议不一致时不能混合排名。
