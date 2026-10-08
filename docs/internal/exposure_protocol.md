# 完整架构对比的输入与分母绑定

`exposure_protocol.freeze()` 为历史窗口准备目录重建一份固定暴露定义。它读取实际文件，构造真实 CPU 模型的初始化摘要，并核验原队列与资源锁身份；不会导入环境工厂、启动模拟器、申请锁或执行训练。`validate_protocol()` 从原始输入重新构造整个定义，不能用重新计算 JSON 的自签名掩盖缺失候选、预算更改或输入替换。

这份定义尚需固定授权的 OS controller 与 worker 消费。`needs_fixed_authorization_OS_controller=true`、`execution_status=unexecuted` 均保留；存在协议不代表训练、独立评价、选择或硬件验证已经完成。

## 完整训练与评价身份

每个候选和训练 seed 都有独立 job 身份，保存完整阶段配置、实际配置文件收据、原 seed 下 CPU 初始化的 named-tensor SHA、各阶段请求样本及累计完整边界。初始化摘要包含 actor、critic、分布参数和固定 buffer，不以参数量或另一个 H 的检查点代替。

所有 job 从 fresh 开始，λ 为零。私有 retention seed 必须显式指定，并与训练、验证、独立确认及原 anchor/capture seed 分离。原来的模型、PPO、控制、环境、课程与 reward 沿用冻结配置，不强行匹配参数量，不引入 RNN。

每个真实阶段端点都评价**全部声明场景**，包括该阶段尚未训练过的场景。完整单元身份为：

`candidate × training seed × stage endpoint × scenario × evaluation seed`。

validation 与 held-out 单独标记。两类矩阵都预声明，但选择器只读取 validation；封存候选和 exact checkpoint 后才读取 held-out 确认。预声明 held-out 单元不授予选择器使用其成绩的权限。原 scenario gate 保留，合法完整但成绩差的评价仍应保存指标；未完成、无有效窗口和不存在端点的单元保留 missing，不能当作零或缩小分母。

目前 H1/H11/H31/H61、34 候选、三训练 seed 的单阶段输入对应 102 jobs、122,400 次请求更新、6,016,204,800 条完整新 rollout 样本上限。50 个场景、两 validation seed 与两 held-out seed，分别产生 10,200 个单元，共 20,400 个。它们是计划上限与完整身份，不是已完成样本或结果；replica 与训练 seed 不混用。

## 真实输入与运行环境

定义绑定新的 history manifest、packed study、完整当前 learner 源以及每份实际配置文件。原始 spec/base 所在目录、历史目录、环境 snapshot、SDK、原队列与 worker 输入树均受保护，新的输出不得重叠或复用旧目录。

每个环境必须声明实际 snapshot 和其身份、具体 contract 与原始字节 SHA。验证读取 snapshot 的全部声明文件及实际目录清单，核验每个源文件、资产和合同；缺项、额外固定文件、不同合同、symlink 或 source 漂移都会拒绝。Git 目录和可重建 Python bytecode 缓存的普通文件字节不属于 SDK 树收据，但其目录及后代仍接受路径和类型检查；缓存名称不能豁免 symlink、FIFO 等特殊文件。

`runtime_identity()` 在 CPU 上读取解释器实际文件、Python/version/prefix、Torch/NumPy 的导入文件及 native extension 字节，并遍历调用者明确声明的外部 SDK 树。SDK 树不得与历史、learner、原 spec/base 或 snapshot 树重叠。它不会初始化 CUDA 或 Isaac 应用，也没有覆盖整个操作系统动态库、GPU 驱动、传感器或硬件时序；`hardware_and_complete_system_runtime_verified=false` 保留。后续 worker 仍需使用新空 bytecode cache 路径，核验实际启动 runtime 与环境 provenance。

冻结前后重新检查 history、learner、snapshot/contract、声明 runtime、原队列关联和两把锁的实际 inode，避免把构造初始化摘要期间发生的输入更改封存为完整定义。定义保存原课程、诊断与学习率队列的原始关联、helper/file 收据及原 controller 身份。冻结允许原队列仍在运行；真正执行必须另用 `check_dependency(..., require_complete=True)` 核验三队列完整闭合、controller/worker 终止，再获取原锁并向精确 child 继承 FD。

## 阶段进程与预算

一个 job 在第一个 child 前一次预留全部阶段预算，不退款、不换 seed、不重用目录。每阶段用独立 OS 进程：阶段零 fresh，其后从上阶段真实封存检查点完整恢复模型、Adam、全局及私有 RNG 与累计时钟，显式 `resume=True, environment_transition=True`。环境和历史重新 reset。

这种阶段隔离是实际环境生命周期所需：当前 chassis factory 每次都会新建 Isaac AppLauncher，adapter `close()` 只停 sim，奖励时间缩放包装也未恢复。同进程反复调用该工厂不能凭合成 CPU 检查宣称安全；阶段独立进程避免沿用旧 application/context 与套叠奖励包装，但实际模拟器运行仍需单独验证。

后续 controller 还须实现完整预算总账、固定授权请求、精确 child/kernel handle、两锁 FD 继承、剩余未压缩 trace/在途文件/checkpoint 的磁盘预留，以及全部端点的独立物理证明。数值失败保持该 job 的缺失并继续其他候选；输入/源/协议/锁漂移、磁盘不足及人工中断应停止 campaign。未知错误不能凭异常文本当作可继续的数值失败。当前模块只声明这些执行义务，不代替其运行证据。

## 调用

在实际执行主机重新准备 H study，声明同一主机真实可读的原队列和 SDK。不要把本机路径提案冒充 Kaiser 可执行输入。

```bash
python -B -m transformer_rl.exposure_protocol freeze \
  --history-root /absolute/path/new_history_preparation \
  --output-root /absolute/path/new_exposure_campaign \
  --protocol-output /absolute/path/exposure_protocol.json \
  --curriculum-summary /absolute/original/curriculum/summary.json \
  --diagnostic-summary /absolute/original/diagnostics/summary.json \
  --learning-summary /absolute/original/learning/summary.json \
  --resource-lock /absolute/original/shared/.run.lock \
  --runtime-roots /absolute/actual/external_sdk \
  --retention-seed 91001 --device cuda:0

python -B -m transformer_rl.exposure_protocol validate \
  --protocol /absolute/path/exposure_protocol.json
```

路径和私有 seed 均须换成真实输入。协议文件不得写入任何原输入、SDK、learner 或未来 campaign 输出树。直接从源码 checkout 调用时设置 `PYTHONPATH` 为当前 `src`；使用实际目标 Python 环境，不能把另一个解释器的静态定义当成相同 runtime。
