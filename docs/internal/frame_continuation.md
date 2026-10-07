# 记忆分支的完整学习状态恢复

`FrameContinuation` 是独立的单分支组件，提供 `open → step → save`。它复用 packed policy、rollout collector 和 PPO，不修改旧 `train_frame_policy`、旧锚点损失或已冻结的课程与学习率队列。它不判定教师资格、不收集锚点、不选择 K/λ，也不自动启动模拟器实验；完整 campaign executor 仍需接入原始资格和准备 manifest 的重新核验。

## 分支与恢复

新分支必须绑定原始 checkpoint 的路径、文件 SHA、完整 actor/critic、Adam、全局 Python/NumPy/Torch/CUDA RNG、successful update、实际累计 transition 及外部累计 consumed update。model、PPO 配方、观测与控制契约必须一致，只允许环境配置切换。checkpoint 中的训练种子及 factory 身份也必须匹配。旧 checkpoint 未保存外部 consumed update 时，调用者须从已审计账本提供，不能凭 checkpoint 文件名猜测。

构造过程先在 CPU 核验 checkpoint 和锚点，再构造环境、设置累计课程时钟、执行 reset，最后恢复全局学习 RNG。这样环境构造与 reset 消耗的随机数不会改变 PPO 的初始随机序列。checkpoint 没有 CUDA RNG 时拒绝 CUDA 续训；CUDA RNG 恢复仍要求实际设备数量兼容。

已有 continuation checkpoint 必须显式 `resume=True`。它要求配置、源代码实际内容、设备、rollout 长度、时钟和记忆目标完全一致，并恢复私有采样器状态。默认新分支入口不能悄悄重置已有采样器，也不支持隐式代码迁移。所有打开方式均重置环境和历史；checkpoint 标明 `episode_state_restored=False`、`history_reset=repeat_first`，不能据此声称连续恢复了物理轨迹。

## 私有采样器

`PrivateAnchorRegularizer` 保持原有采样分布：先均匀选择文件，再从该文件均匀有放回抽取 `min(batch_size,N)` 个 endpoint。标签仍为 clamp 前 Gaussian mean/std，损失仍是每个样本对动作求和、再对样本取平均的 `KL(teacher || student)`，乘以显式 λ。固定窗口 H、锚点池 K 和 λ 是不同变量。

采样只使用私有 CPU `torch.Generator`。λ=0 可使用空锚点列表，不 forward、不抽样、不推进私有状态；λ>0 必须提供池。私有种子必须为 uint32，不能与调用者提供的训练和评估种子重合。实际实验的完整种子集合与教师资格仍由冻结协议负责。

状态为小型 JSON，包含算法版本、Torch 版本、seed、λ、batch size、有序池路径/SHA/行数、抽样计数和编码后的 generator bytes。恢复检查整体与 generator SHA，并重演私有抽样以核对计数和下一状态；不 forward actor，不使用全局 RNG。调用者控制 replay 工作上限，默认最多 1,000,000 次采样调用，超过上限拒绝。该检查成本随调用次数增长，不是常数时间恢复。

## 计数与失败

`step()` 只优化完整 rollout。中途停止的部分样本计入实际累计 transition，同时记录 discarded transition，不能按完整 batch 报数，也不执行 PPO。完整 rollout 开始优化时计入 consumed update；成功完成后才增加 successful update。PPO 或采集抛出异常后，组件拒绝继续训练及发布新的学习 checkpoint，避免将可能部分修改的 Adam 冒充完整更新；外部控制器须保存失败账本并重新加载封存 checkpoint。

`save()` 通过已有独占 checkpoint 发布方法保存完整学习状态和私有采样状态，不覆盖已有文件。组件不替代外部锁、预算、SIGTERM/deadline、TensorBoard、失败账本或资格重验；控制器需要独立管理这些事项。

当前测试使用小型 CPU 合成环境，验证手工完整状态恢复与组件一致、环境随机数扰动隔离、下一次采样序列一致、严格身份拒绝和失败路径。它们不是机器人训练、GPU 一致性或硬件部署验证。
