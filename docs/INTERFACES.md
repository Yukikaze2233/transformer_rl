# 内部接口与实现边界

本仓库独立于旧训练仓库、RSL-RL及部署demo。当前任务是代码实现和定向验证，不运行机器人训练。

## 网络侧

- `ModelConfig` / `PPOConfig`：`config.py`；冻结配置。
- `pack_frame(config, proprio, command, previous_issued_action, sensor_age_s, sensor_age_known, policy_dt_s)`：生成 `[N,F]`，F默认30。输入已有明确的观测/控制量缩放；时间使用秒，模型侧按time_scale_s编码。未知age以known=false表示，不能冒充已知零延迟。
- `HistoryBuffer(config, num_envs, device)`，`reset(mask)`、`append(VectorObservation) -> HistoryBatch`。每env独立，左侧padding、valid mask，当前帧最后；相同timestamp重复query必须幂等或对不同内容报错；真正reset才能清记忆。
- `TimeAwareActor(config)`：`forward(HistoryBatch) -> mean[B,A]`，`act(history, deterministic=False) -> ActionSample`，`evaluate(history, raw_action) -> PolicyEvaluation`。无RSL、无仿真import、无dropout、无可变KV状态。
- `ValueCritic(config)`：`forward(critic[B,S]) -> value[B]`。
- `ActorCritic(config)`：含`actor`、`critic`子模块。
- 原始policy事件timestamp优先float64，先作时间差再转换到网络dtype，避免系统uptime较大时丢失毫秒精度。
- 当前命令query追加到历史序列末尾，从query输出动作；历史帧保留当时命令。各历史token加真实时间age的固定Fourier编码，因果注意力屏蔽未来和padding。

## 训练侧

- 仅普通PyTorch；`PPOTrainer(model: ActorCritic, config: PPOConfig)`，`update(PPOBatch)->dict`。
- `RolloutBuffer(capacity)`，`add(history, critic, raw_action, issued_action, old_log_prob, old_mean, old_std, old_value, reward, next_value, terminated, truncated)`；每项均detach/clone，不能别名环境tensor。
- `finish(gamma, gae_lambda)->PPOBatch`，T/N展平前沿时间算GAE。PPOBatch含history、critic、raw/issued actions、old logprob/mean/std/value、advantages和returns；提供batch index能力。
- 初版保存完整history快照作为正确性参考，不做有上下文差异的packed sequence优化。
- delta用`1-terminated` bootstrap；GAE trace用`1-(terminated|truncated)`；returns在advantage标准化前形成。critic/loss/logp约定全部用明确shape，无隐含广播。
- 每个endpoint只计算一个策略loss。PPO第一次更新前，检查同权重重算logprob与old logprob一致。actor dropout0，完整数据预处理与history不在epoch内重采。

## 集成侧

`VectorEnv` / `VectorObservation` / `StepResult`见`types.py`。runner为唯一采集负责人，使用最新frame推进history；done行的next observation已经reset，truncation须提供真正pre-reset final_critic。

runner给env的是issued action，actual applied target、通信FIFO、下位机PID和电机响应属于env/控制模型。真实硬件模式/周期未知，不填造测量值。policy100Hz是候选，非实机证明。

网络、训练器、采集器、控制时序和导出各自通过这些接口协作，不从旧demo推导硬件事实。

## 采集、检查点和导出接口

- `RolloutCollector(env, model, ppo_config, action_clip=None)`位于`runner.py`。`reset(seed=None)`；`collect(steps, should_stop=None) -> PPOBatch | None`，内部每次创建buffer。终止预算前已采数据可形成partial rollout，空则None。每次env.step前保存critic/action/statistics等独立快照，避免auto-reset或in-place更新覆盖o_t。`total_steps`计向量步，`total_transitions=total_steps*num_envs`；`last_metrics`记录本次采集统计。
- `TensorEnvAdapter(env, model_config, encode_observation)`位于`adapters.py`。包装torch形式Gymnasium/IsaacLab五返回值API，encoder签名`(raw_obs, info)->VectorObservation`。不导入Isaac、不启动App、不猜测obs字段。时间限制需要info中的`final_critic`/`final_critic_valid`；缺失则明确拒绝，不能使用reset后状态替代。raw env必须自行auto-reset。
- `save_checkpoint(path, model, trainer, update, metadata)->dict`、`load_checkpoint(path, device='cpu')->(model, trainer, update, metadata)`位于`checkpoint.py`。包含model/ppo配置、weights、optimizer、累计update、JSON元数据；weights_only读取、严格state keys/shape/finite和格式标识校验；原子新文件写入，不覆盖已存在结果。恢复优化状态后从新episode开始，不声称保存仿真/通信/PID状态的精确续接。
- `export_policy(checkpoint_path, output_path)->dict`位于`export.py`。输出deterministic mean的ONNX和JSON sidecar，绑定checkpoint/model配置/输入布局及文件SHA，使用ORT多种合法history/padding/time条件核对。五输入frames/times/valid/command/now，固定history长度，batch轴可动态。
- CLI直接组合collector和trainer的显式collect/update循环，管理有界run、信号、JSONL指标和checkpoint；只有显式train子命令与用户提供env_factory时才执行环境交互。
