# 固定阶段暴露的训练任务层

`exposure_training.train_exposure_job()` 执行一个架构、一个训练种子的全部预声明阶段。每阶段必须完成指定数量的完整 rollout；成绩不控制晋级，不执行回滚、提前淘汰、重置 Adam 或自动重试。单帧 MLP、历史编码 MLP 与 Transformer 使用同一个接口，窗口长度在任务内保持不变。

这是学习任务层。操作系统 worker、旧队列闭合、原资源锁、预算总账、磁盘预留、独立评价和选择由外层执行器负责；本接口不申请这些资源，也不凭自己的 `completed` 声称完整对比研究已经执行。

## 配置与初始状态

`stages` 是非空列表，每项为 `{name, config, updates}`。`config` 可为解析后的 `FrameTrainConfig` 或配置字典；接口一律重新解析为私有副本，防止调用者后续修改其它阶段的配置。model、PPO 和 control 在各阶段必须完全相同，仅 environment 可变。工厂不得修改实际会话配置；启动、每次更新及封存前都会重新对照声明配置。

每项可另外显式提供 `checkpoint_updates`，完整形式为 `{name, config, updates, checkpoint_updates}`。该列表必须非空、严格递增、无重复，只含不超过本阶段 `updates` 的正整数，并以最终更新数结束。例如本阶段 `updates=1200`、`checkpoint_updates=[400,800,1200]` 是一个连续阶段的三个模型观测点，不是三次恢复训练。省略列表时只保存阶段末模型。[exposure 协议](exposure_protocol.md) 的 v1 使用这种阶段末行为；v2 由显式 `freeze(..., checkpoint_interval=400)` 构造并冻结列表，外层传给本接口，而非隐式采用 history 配置的间隔。

`initial_model_sha256(config, training_seed)` 在 CPU 上构造实际模型并计算命名 tensor 内容 SHA。它保留调用者 CPU RNG，不改 Python、NumPy 或 CUDA 流。`expected_initial_model_sha256` 必须显式提供，错误在创建输出和环境之前拒绝。此哈希包含 actor、critic、固定 buffer 与分布参数，不能以参数总量或另一窗口的权重代替。

接口另需显式指定 job identity、训练 seed、独立私有 seed、评价 seed、rollout 长度、设备及总截止时间。训练 seed 不得与评价 seed 相同；私有 seed 不得与上述 seed 相同。设备为 `cpu` 或带明确编号的 `cuda:N`。这些校验不等于实际 CUDA/模拟器或实机已通过验证。

fresh 使用 `FrameContinuation.start()`，λ 为零，没有教师或 anchors。首次环境 reset 完成后固定学习随机流的规则见 [完整学习会话](frame_continuation.md)。这个新 fresh 配方的初始权重与原初始化 guard 一致；不能据此宣称它与旧入口消耗环境随机数后的所有学习轨迹相同。

## 阶段链

同进程任务入口要求环境工厂能够完整关闭并重新创建环境。Isaac chassis 的多阶段执行使用每阶段独立 OS worker 与 [分阶段续训接口](exposure_segments.md)，由外层一次性预留整个任务预算，避免在一个进程中重建应用或重复安装奖励包装。

每阶段的真实完整边界保存 `endpoint.pt`、sidecar 与 `endpoint.json`。下一阶段严格从这份文件的实际 SHA 打开，显式指定 `resume=True, environment_transition=True`，继承模型、Adam、全局及私有 RNG、成功更新、已尝试更新与累计 transition。环境和历史重新 reset，历史规则为 repeat-first；物理 episode 没有恢复。

显式声明中途保存时，worker 在完整 rollout、成功优化及日志发布后封存 `checkpoint_00000400.pt`、sidecar 和 `checkpoint_00000400.json` 等文件；最终模型仍使用 `endpoint.pt` 与 `endpoint.json`。保存继续使用同一环境、collector、历史、actor/critic、Adam 与全局及私有 RNG，不触发环境或历史 reset，不执行 close/open。封存的 `sealed_checkpoints` 将模型绑定到实际成功更新、fresh transition、优化步和优化样本使用次数，以及当时完整日志前缀的 SHA、字节数与行数。保存的墙钟与磁盘开销仍属于实际执行成本。

评价不在训练进程内运行，不消费学习随机数，也没有评价 gate 回调影响任务暴露。阶段关闭后，外层独立 worker 评价每份真实封存模型的全部声明场景。成绩差的模型仍接受预声明的全部阶段；中途模型仅提供学习和取得后退化的观测，不能作为跨阶段恢复 parent，下一阶段仍只接受上阶段完整最终端点。固定场景与 validation seed 上先取得、后退化的能力才记为遗忘；始终未取得属于未学会。最终排名和代表模型固定为最后阶段最终 checkpoint，不用更好的中途模型替换。细则见 [独立执行器](exposure_campaign.md#连续训练的中间模型) 和 [选择证据](exposure_selection.md)。

## 预算和失败

任务启动前写入 `request.json` 和整个任务的 `reservation.json`。阶段完整样本预算为

\[
B_s=U_s\,T_{\mathrm{rollout}}\,N_s,\qquad B=\sum_s B_s.
\]

各阶段环境数可不同，以实际 collector 再核验。reservation 是预留与收费，不是已经采集的样本；失败不退款，不自动换 seed 或重用目录。中途保存不增加请求更新或 fresh rollout 预算，但独立评价和磁盘分母按全部声明观测点扩展。34 候选 × 3 个训练 seed、每 job 一个连续 1,200 次更新的阶段仍为 102 jobs、122,400 次更新；每更新完整采集 48 步 × 1,024 个训练环境时，共请求 6,016,204,800 条 fresh 样本。若每阶段保存 `[400,800,1200]`、每模型评价全部 50 场景和两 validation/两 held-out seed，则两类各有 30,600 格，共 61,200 格。该例仅说明完整请求分母，不是已完成训练或评价结果，也不替代学习率、记忆容量、Sim2Real 与部署研究。

`FrameContinuation.step()` 仅优化完整 rollout。部分采集计入真实 transition，不优化也不封存该阶段端点；采集或 PPO 异常令会话失效，不能把部分修改的优化器保存成完整状态。任务保留之前确实封存的模型；后续失败前的合法中间模型可由外层独立核验和评价，未达到的观测点与剩余阶段仍为 missing，整个原 job 预留不退款。包含中途保存的 completion 使用 schema v2，列出 `sealed_checkpoints`、`missing_checkpoints` 与 `last_sealed_learning_checkpoint`，孤立文件不能代替这些完整封存证据。

账本分别记录：成功返回的学习更新、已尝试更新、实际采集、成功完整 rollout 样本、已记录的 metric 行、真实优化步及重复优化样本。日志写入失败不抹掉此前已经完成的学习；失败 PPO 可能已修改部分参数时，其优化步数记为未知，不伪装为零。未封存成功更新、未发布端点文件、主要错误、关闭错误和中断原因各自保存。

`completion.json.status=completed` 仅表示全部训练阶段端点及实际预算吻合，并不表示运动任务合格、遗忘实验完成或已经选出最佳 Transformer。部分目录、单独 `.pt` 或不存在 completion 都不能授权重新运行。磁盘或发布失败可能只留下 reservation，外层审计必须保守保留整个收费，核验实际子进程终止后处理。

## 外层执行器的义务

已实现的独立 worker/controller 负责校验新源码下的 history plan、实际环境源码/资产/contract、完整 runtime，等待原课程、诊断、学习率队列真正闭合，获取原两把锁并向精确子进程继承 FD，在每次启动前预留磁盘和全部样本预算。实际目标主机仍须满足准入条件，并提供真实运行时校准与执行证据；接口集成不代表这些条件已经完成。

每个候选、训练 seed、阶段、声明 checkpoint、场景、评价 seed 都应出现在相应协议的完整分母中。storage 按全部模型、尚未闭合评价 trace、独立 worker namespace/cache、日志、在途副本及余量预留。源变更需重新 prepare history，保存 schedule 变更需重新 freeze 协议；生产空间依据以实际目标主机的 [运行时校准](RUNTIME_CALIBRATION.md) 核验，storage 合同重新绑定新协议。旧阶段末总预算或 CPU 合成检查不能代替实际 SDK 峰值。选择只使用 validation，封存架构和 exact checkpoint 后才读取 held-out 确认。低成绩和缺失均保留；没有同时满足运动和实时要求的候选时，不强行宣布赢家。K/λ 分支另外依赖真正获取旧技能的教师资格，不以本训练任务层的存在替代该资格。
