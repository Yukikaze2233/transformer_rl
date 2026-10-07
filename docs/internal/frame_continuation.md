# 完整学习会话与记忆分支恢复

`FrameContinuation` 是独立的单会话组件，提供 `start/open → step → save`。它复用 packed policy、rollout collector 和 PPO，不修改旧 `train_frame_policy`、旧锚点损失或已冻结的课程与学习率队列。它不判定教师资格、不收集锚点、不选择 K/λ，也不自动启动模拟器实验。[记忆分支执行器](retention_campaign.md) 将旧 checkpoint 分支接入原始资格与准备 manifest 的重新核验、资源锁和完整分支预算；真实分支仍需满足全部输入资格。新会话接口也可由独立的等任务暴露执行器调用，但这个组件本身不执行架构对比协议。

## 从零初始化

`start()` 必须显式提供 `config`、环境 factory/reference、`training_seed`、独立的 `retention_seed`、`rollout_steps`，以及 `expected_initial_model_sha256`；评估种子与设备也可显式提供。接口没有教师或锚点参数，λ 固定为零。初始化 actor、critic 和架构固定 buffer 的内容哈希使用 `sorted_named_tensor_contents_v1`，与旧训练入口的 initial-model guard 相同。

首先重新解析配置并验证种子、设备格式和哈希格式，然后在 CPU 的 `fork_rng(devices=[])` 内按训练 seed 构造完整模型并校验内容。失败时不构造环境、不调用全局种子函数、不初始化 CUDA，调用者的 CPU/Python/NumPy RNG 保持不变。只有校验成功后才允许构造环境或设备。

环境构造与 reset 前设置零 consumed update、零 collected transition；历史以首帧重复填满。reset 后重新设置全局学习 seed，并恢复 CPU 初始化模型消耗后的随机数前缀，隔离环境构造及 reset 对学习随机流的影响。Python/NumPy 从训练 seed 开始，CUDA 在环境应用、设备与 reset 完成后由同一训练 seed 明确设置。元数据 `initial_rng` 记录这个顺序。这里与旧训练入口共用的是初始权重哈希；旧入口会保留 fresh 环境构造消耗的随机数，不能宣称两个 fresh 训练入口在任意环境下的后续随机轨迹相同。

初始 Adam 无历史状态，私有采样器使用独立 CPU generator，λ=0 不推进采样计数。成功启动后可在 update=0 保存完整配置、模型、Adam、全局/私有 RNG、source、环境 provenance、初始化校验和时钟；它是完整学习 checkpoint，可以显式恢复。CUDA 随机数设置的代码路径尚需实际 GPU 运行验证，CPU 合成测试不证明 GPU 数值或真实机器人行为一致。

## 分支与恢复

新分支必须绑定原始 checkpoint 的路径、文件 SHA、完整 actor/critic、Adam、全局 Python/NumPy/Torch/CUDA RNG、successful update、实际累计 transition 及外部累计 consumed update。model、PPO 配方、观测与控制契约必须一致，只允许环境配置切换。checkpoint 中的训练种子及 factory 身份也必须匹配。旧 checkpoint 未保存外部 consumed update 时，调用者须从已审计账本提供，不能凭 checkpoint 文件名猜测。

构造过程先在 CPU 核验 checkpoint 和锚点，再构造环境、设置累计课程时钟、执行 reset，最后恢复全局学习 RNG。这样环境构造与 reset 消耗的随机数不会改变 PPO 的初始随机序列。checkpoint 没有 CUDA RNG 时拒绝 CUDA 续训；CUDA RNG 恢复仍要求实际设备数量兼容。

已有 continuation checkpoint 必须显式 `resume=True`。默认要求配置、源代码实际内容、设备、rollout 长度、时钟和记忆目标完全一致，并恢复私有采样器状态。保存了评估种子的会话还核验评估种子集合。默认新分支入口不能悄悄重置已有采样器，也不支持隐式代码迁移。

跨阶段切换环境必须在 `open()` 同时显式指定 `resume=True, environment_transition=True`，且父 checkpoint 含有完整 continuation 状态。它仅允许 `config.environment` 不同；model、PPO、控制契约、factory、训练 seed、source、设备、累计 consumed/transition 时钟、rollout 长度、私有 seed、λ、batch size、K/锚点实际身份与私有采样状态仍须通过完整恢复校验。布尔以外的 transition 标志、未指定 resume 或普通旧 checkpoint 都不能走这个入口。

`continuation_parent` 保持旧协议的四字段 `path/sha256/update/resume`。只有本次显式切换才额外写入独立的 `stage_transition`，内容为 `environment_transition=true`、父环境配置 SHA 与新环境配置 SHA。普通 exact resume 会移除父 metadata 中的 `stage_transition`，避免把上次切换误记为本次发生；历史切换仍能通过实际父 checkpoint SHA 链追溯。这个扩展不放宽旧记忆分支的训练证明校验。

阶段切换直接继承完整 actor/critic、Adam 和全局/私有随机数；环境构造与 reset 完成后恢复父全局学习 RNG。它不会把权重迁移、重新创建空 Adam 或重置私有 generator 冒充完整继承。所有打开方式均重置环境和历史；checkpoint 标明 `episode_state_restored=False`、`history_reset=repeat_first`，不能据此声称连续恢复了物理轨迹。factory、provenance 或 reset 抛出异常时会使会话失效并关闭环境；关闭时另有异常会附加到原异常，失效会话不能 step/save。

## 私有采样器

`PrivateAnchorRegularizer` 保持原有采样分布：先均匀选择文件，再从该文件均匀有放回抽取 `min(batch_size,N)` 个 endpoint。标签仍为 clamp 前 Gaussian mean/std，损失仍是每个样本对动作求和、再对样本取平均的 `KL(teacher || student)`，乘以显式 λ。固定窗口 H、锚点池 K 和 λ 是不同变量。

采样只使用私有 CPU `torch.Generator`。λ=0 可使用空锚点列表，不 forward、不抽样、不推进私有状态；λ>0 必须提供池。私有种子必须为 uint32，不能与调用者提供的训练和评估种子重合。实际实验的完整种子集合与教师资格仍由冻结协议负责。

状态为小型 JSON，包含算法版本、Torch 版本、seed、λ、batch size、有序池路径/SHA/行数、抽样计数和编码后的 generator bytes。恢复检查整体与 generator SHA，并重演私有抽样以核对计数和下一状态；不 forward actor，不使用全局 RNG。调用者控制 replay 工作上限，默认最多 1,000,000 次采样调用，超过上限拒绝。该检查成本随调用次数增长，不是常数时间恢复。

## 计数与失败

`step()` 只优化完整 rollout。中途停止的部分样本计入实际累计 transition，同时记录 discarded transition，不能按完整 batch 报数，也不执行 PPO。完整 rollout 开始优化时计入 consumed update；成功完成后才增加 successful update。PPO 或采集抛出异常后，组件拒绝继续训练及发布新的学习 checkpoint，避免将可能部分修改的 Adam 冒充完整更新；外部控制器须保存失败账本并重新加载封存 checkpoint。

`save()` 通过已有独占 checkpoint 发布方法保存完整学习状态和私有采样状态，不覆盖已有文件。组件不替代外部锁、预算、SIGTERM/deadline、TensorBoard、失败账本或资格重验；控制器需要独立管理这些事项。

当前测试使用小型 CPU 合成环境，验证初始 guard、零时钟保存恢复、三段显式环境变化下真实 Adam/全局与私有 RNG 的继承、手工完整状态恢复与组件一致、环境随机数扰动隔离、下一次采样序列一致、严格身份拒绝和失败路径。测试同时覆盖 MLP、history MLP、last/query Transformer；它们不是机器人训练、GPU 一致性或硬件部署验证。
