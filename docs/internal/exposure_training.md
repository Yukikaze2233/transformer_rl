# 固定阶段暴露的训练任务层

`exposure_training.train_exposure_job()` 执行一个架构、一个训练种子的全部预声明阶段。每阶段必须完成指定数量的完整 rollout；成绩不控制晋级，不执行回滚、提前淘汰、重置 Adam 或自动重试。单帧 MLP、历史编码 MLP 与 Transformer 使用同一个接口，窗口长度在任务内保持不变。

这是学习任务层。操作系统 worker、旧队列闭合、原资源锁、预算总账、磁盘预留、独立评价和选择由外层执行器负责；本接口不申请这些资源，也不凭自己的 `completed` 声称完整对比研究已经执行。

## 配置与初始状态

`stages` 是非空列表，每项为 `{name, config, updates}`。`config` 可为解析后的 `FrameTrainConfig` 或配置字典；接口一律重新解析为私有副本，防止调用者后续修改其它阶段的配置。model、PPO 和 control 在各阶段必须完全相同，仅 environment 可变。工厂不得修改实际会话配置；启动、每次更新及封存前都会重新对照声明配置。

`initial_model_sha256(config, training_seed)` 在 CPU 上构造实际模型并计算命名 tensor 内容 SHA。它保留调用者 CPU RNG，不改 Python、NumPy 或 CUDA 流。`expected_initial_model_sha256` 必须显式提供，错误在创建输出和环境之前拒绝。此哈希包含 actor、critic、固定 buffer 与分布参数，不能以参数总量或另一窗口的权重代替。

接口另需显式指定 job identity、训练 seed、独立私有 seed、评价 seed、rollout 长度、设备及总截止时间。训练 seed 不得与评价 seed 相同；私有 seed 不得与上述 seed 相同。设备为 `cpu` 或带明确编号的 `cuda:N`。这些校验不等于实际 CUDA/模拟器或实机已通过验证。

fresh 使用 `FrameContinuation.start()`，λ 为零，没有教师或 anchors。首次环境 reset 完成后固定学习随机流的规则见 [完整学习会话](frame_continuation.md)。这个新 fresh 配方的初始权重与原初始化 guard 一致；不能据此宣称它与旧入口消耗环境随机数后的所有学习轨迹相同。

## 阶段链

每阶段的真实完整边界保存 `endpoint.pt`、sidecar 与 `endpoint.json`。下一阶段严格从这份文件的实际 SHA 打开，显式指定 `resume=True, environment_transition=True`，继承模型、Adam、全局及私有 RNG、成功更新、已尝试更新与累计 transition。环境和历史重新 reset，历史规则为 repeat-first；物理 episode 没有恢复。

评价不在训练进程内运行，不消费学习随机数，也没有评价 gate 回调影响任务暴露。成绩差的模型仍接受预声明的全部阶段；后续外层评价应覆盖所有候选及每个真正封存的阶段端点。

## 预算和失败

任务启动前写入 `request.json` 和整个任务的 `reservation.json`。阶段完整样本预算为

\[
B_s=U_s\,T_{\mathrm{rollout}}\,N_s,\qquad B=\sum_s B_s.
\]

各阶段环境数可不同，以实际 collector 再核验。reservation 是预留与收费，不是已经采集的样本；失败不退款，不自动换 seed 或重用目录。

`FrameContinuation.step()` 仅优化完整 rollout。部分采集计入真实 transition，不优化也不封存该阶段端点；采集或 PPO 异常令会话失效，不能把部分修改的优化器保存成完整状态。任务保留之前确实完成的阶段端点，剩余阶段记为 missing。

账本分别记录：成功返回的学习更新、已尝试更新、实际采集、成功完整 rollout 样本、已记录的 metric 行、真实优化步及重复优化样本。日志写入失败不抹掉此前已经完成的学习；失败 PPO 可能已修改部分参数时，其优化步数记为未知，不伪装为零。未封存成功更新、未发布端点文件、主要错误、关闭错误和中断原因各自保存。

`completion.json.status=completed` 仅表示全部训练阶段端点及实际预算吻合，并不表示运动任务合格、遗忘实验完成或已经选出最佳 Transformer。部分目录、单独 `.pt` 或不存在 completion 都不能授权重新运行。磁盘或发布失败可能只留下 reservation，外层审计必须保守保留整个收费，核验实际子进程终止后处理。

## 外层执行器的义务

生产执行仍需将此接口接入固定授权的 worker/controller：校验新源码下的 history plan、实际环境源码/资产/contract、完整 runtime；等待原课程、诊断、学习率队列真正闭合；获取原两把锁并向精确子进程继承 FD；在每次启动前预留磁盘和全部样本预算。

每个候选、训练 seed、阶段、场景、评价 seed 都应出现在完整分母中。选择只使用 validation，封存架构和 exact checkpoint 后才读取 held-out 确认。低成绩和缺失均保留；没有同时满足运动和实时要求的候选时，不强行宣布赢家。K/λ 分支另外依赖真正获取旧技能的教师资格，不以本训练任务层的存在替代该资格。
