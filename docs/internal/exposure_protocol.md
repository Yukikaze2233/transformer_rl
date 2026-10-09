# 完整架构对比的输入与分母绑定

`exposure_protocol.freeze()` 为历史窗口准备目录重建一份固定暴露定义。它读取实际文件，构造真实 CPU 模型的初始化摘要，并核验原队列与资源锁身份；不会导入环境工厂、启动模拟器、申请锁或执行训练。`validate_protocol()` 从原始输入重新构造整个定义，不能用重新计算 JSON 的自签名掩盖缺失候选、预算更改或输入替换。

这份定义尚需固定授权的 OS controller 与 worker 消费。`needs_fixed_authorization_OS_controller=true`、`execution_status=unexecuted` 均保留；存在协议不代表训练、独立评价、选择或硬件验证已经完成。

## 完整训练与评价身份

每个候选和训练 seed 都有独立 job 身份，保存完整阶段配置、实际配置文件收据、原 seed 下 CPU 初始化的 named-tensor SHA、各阶段请求样本及累计完整边界。初始化摘要包含 actor、critic、分布参数和固定 buffer，不以参数量或另一个 H 的检查点代替。

所有 job 从 fresh 开始，λ 为零。私有 retention seed 必须显式指定，并与训练、验证、独立确认及原 anchor/capture seed 分离。原来的模型、PPO、控制、环境、课程与 reward 沿用冻结配置，不强行匹配参数量，不引入 RNN。

协议 v1 未显式传入 `checkpoint_interval`，仅在真实阶段端点保存并评价模型；它不会自动使用 history 输入中的训练日志间隔。每个端点都评价**全部声明场景**，包括该阶段尚未训练过的场景。完整单元身份为：

`candidate × training seed × stage endpoint × scenario × evaluation seed`。

显式调用 `freeze(..., checkpoint_interval=400)` 则产生协议 v2：`execution.checkpoint_interval` 固定间隔，每阶段的 `checkpoint_updates` 固定递增的本阶段成功更新数，并始终包含最终更新。例如单阶段 1,200 次更新声明 `[400,800,1200]`，不拆成三个训练阶段。v2 的完整评价单元身份为：

`candidate × training seed × stage × cumulative checkpoint update × scenario × evaluation seed`。

中间模型在完整 rollout、成功优化及日志发布后封存，继续使用同一环境、collector、历史、Adam 与全局及私有 RNG；保存本身不 close/open 或 reset。阶段训练关闭后，独立评价 worker 按每份真实封存模型、role 与评价 seed 运行全部场景。中途保存与跨阶段恢复的生命周期不同，见 [执行器的连续模型保存](exposure_campaign.md#连续训练的中间模型)。

validation 与 held-out 单独标记。两类矩阵都预声明，但选择器只读取 validation；封存候选和 exact checkpoint 后才读取 held-out 确认。预声明 held-out 单元不授予选择器使用其成绩的权限。原 scenario gate 保留，合法完整但成绩差的评价仍应保存指标；未完成、无有效窗口和不存在端点的单元保留 missing，不能当作零或缩小分母。

H1/H11/H31/H61、34 候选、三训练 seed、每 job 单阶段 1,200 次更新的输入，对应 102 jobs、122,400 次请求更新；每更新完整采集 48 步 × 1,024 个训练环境时，fresh rollout 样本上限为 6,016,204,800。50 个场景、两 validation seed 与两 held-out seed，在 v1 下分别产生 10,200 个单元，共 20,400 个。若 v2 显式固定间隔 400，则保存 306 份模型，两类评价各 30,600 个单元，共 61,200 个；每单元 8 个 replica、4,001 个策略步时，评价采样上限为 1,958,889,600。中途保存不增加训练更新或 fresh rollout 预算。以上是声明分母和请求上限，不是完成样本或结果；replica 与训练 seed 不混用。

学习曲线按固定场景及 validation seed 的 checkpoint 时间顺序分析。只有先满足预声明 gate 且具有有效 score 的能力，后续不再通过 gate、score 不可用或比此前已取得的最佳 score 恶化超过冻结的 `retention_score_tolerance`，才计入取得后的遗忘；从未取得的能力仍标为未学会，缺失评价保留为未观测。最终排名和代表模型固定使用最后阶段的最终 checkpoint，较好的中间模型不能代替最终模型。保存间隔、历史窗口 H 与教师 anchor 池 K/正则系数 λ 是不同变量；本协议的 λ 仍为零。新增观测点不取消学习率、记忆容量、Sim2Real 或部署研究的完整范围。

## 真实输入与运行环境

定义绑定新的 history manifest、packed study、完整当前 learner 源以及每份实际配置文件。原始 spec/base 所在目录、历史目录、环境 snapshot、SDK、原队列与 worker 输入树均受保护，新的输出不得重叠或复用旧目录。

每个环境必须声明实际 snapshot 和其身份、具体 contract 与原始字节 SHA。验证读取 snapshot 的全部声明文件及实际目录清单，核验每个源文件、资产和合同；缺项、额外固定文件、不同合同、symlink 或 source 漂移都会拒绝。Git 目录和可重建 Python bytecode 缓存的普通文件字节不属于 SDK 树收据，但其目录及后代仍接受路径和类型检查；缓存名称不能豁免 symlink、FIFO 等特殊文件。

`runtime_identity()` 在 CPU 上读取解释器实际文件、Python/version/prefix、Torch/NumPy 的导入文件及 native extension 字节，并遍历调用者明确声明的外部 SDK 树。SDK 树不得与历史、learner、原 spec/base 或 snapshot 树重叠。它不会初始化 CUDA 或 Isaac 应用，也没有覆盖整个操作系统动态库、GPU 驱动、传感器或硬件时序；`hardware_and_complete_system_runtime_verified=false` 保留。后续 worker 仍需使用新空 bytecode cache 路径，核验实际启动 runtime 与环境 provenance。

除了上述 Git/bytecode 例外，SDK 默认冻结普通输入，包括名为 cache 的目录、`extscache` 与随安装提供的 shader。只有调用者显式提供 `runtime_identity(runtime_roots, mutable_paths=[...])`，或 `freeze(..., runtime_mutable_paths=[...])` 时，才可声明实际 Isaac 安装根下的精确 `kit/cache`、`kit/data`、`kit/logs`。识别读取该根的 Isaac bootstrap、SimulationApp 源与实际 native Kit app plugin；目录名称本身不构成识别依据。每个 mutable 路径必须是存在的绝对 canonical 目录、当前用户拥有且与 SDK 根同一文件系统。所有后代仍遍历并检查普通类型、UID、文件系统及无 symlink，不能用排除名掩盖 FIFO 或逃逸路径。

protocol 保存排序后的 `runtime_mutable_paths`，每份 SDK tree 保存精确相对 `mutable_subdirectories`；它们属于调用者固定的原协议字节。只有这些目录的普通可写文件字节不计入 SDK 输入摘要；`extscache`、kernel/plugin、代码及 shipped shader 仍冻结。环境 snapshot 的 `_tree` 调用不接受此例外，额外 cache 名称仍会改变 snapshot 成员。这项拆分不校准 SDK 峰值，也不授权生产运行。不同源需要新 prepare/freeze，不能续签旧协议后启动。

冻结前后重新检查 history、learner、snapshot/contract、声明 runtime、原队列关联和两把锁的实际 inode，避免把构造初始化摘要期间发生的输入更改封存为完整定义。定义保存原课程、诊断与学习率队列的原始关联、helper/file 收据及原 controller 身份。冻结允许原队列仍在运行；真正执行必须另用 `check_dependency(..., require_complete=True)` 核验三队列完整闭合、controller/worker 终止，再获取原锁并向精确 child 继承 FD。

## 阶段进程与预算

一个 job 在第一个 child 前一次预留全部阶段预算，不退款、不换 seed、不重用目录。每阶段用独立 OS 进程：阶段零 fresh，其后从上阶段真实封存检查点完整恢复模型、Adam、全局及私有 RNG 与累计时钟，显式 `resume=True, environment_transition=True`。环境和历史重新 reset。

这种阶段隔离是实际环境生命周期所需：当前 chassis factory 每次都会新建 Isaac AppLauncher，adapter `close()` 只停 sim，奖励时间缩放包装也未恢复。同进程反复调用该工厂不能凭合成 CPU 检查宣称安全；阶段独立进程避免沿用旧 application/context 与套叠奖励包装，但实际模拟器运行仍需单独验证。

独立 [controller](exposure_campaign.md) 负责完整预算总账、固定授权请求、精确 child/kernel handle、两锁 FD 继承、剩余未压缩 trace/在途文件/checkpoint 的磁盘预留，以及全部声明模型的独立物理证明。数值失败保持该 job 的缺失并继续其他候选；输入/源/协议/锁漂移、磁盘不足及人工中断应停止 campaign。未知错误不能凭异常文本当作可继续的数值失败。当前模块只冻结这些执行义务，代码接口存在不代替真实运行证据。

v2 的磁盘预留按全部声明 checkpoint 及所有尚未闭合评价单元重建：同一连续阶段只占一个训练 worker namespace，每个 checkpoint × role × seed 的评价 batch 各占独立 namespace。模型、未压缩 trace、日志、runtime/cache、在途发布副本与可用空间余量都保留；失败或模型缺失不能缩小原矩阵。源变更后必须在实际目标主机重新 prepare history；保存间隔变更必须重新 freeze 协议。生产空间依据需由 [实际运行时校准](RUNTIME_CALIBRATION.md) 核验，并重建绑定新协议原始字节的 storage 合同。不能沿用 v1 的总空间预算、假定压缩比例，或将 CPU fixture 当作真实 SDK 峰值测量；静态冻结也不授予 SDK/GPU 或实机验证资格。

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
  --retention-seed 91001 --device cuda:0 \
  --checkpoint-interval 400

python -B -m transformer_rl.exposure_protocol validate \
  --protocol /absolute/path/exposure_protocol.json
```

该例显式选择 v2 的连续保存间隔；省略 `--checkpoint-interval` 则保留 v1 的阶段末行为。路径和私有 seed 均须换成真实输入。协议文件不得写入任何原输入、SDK、learner 或未来 campaign 输出树。直接从源码 checkout 调用时设置 `PYTHONPATH` 为当前 `src`；使用实际目标 Python 环境，不能把另一个解释器的静态定义当成相同 runtime。

如实际识别的 SDK 存在需拆分的可写目录，可在 freeze 命令逐项追加 `--runtime-mutable-path /absolute/actual/isaacsim/kit/cache`、`--runtime-mutable-path /absolute/actual/isaacsim/kit/data` 或 `--runtime-mutable-path /absolute/actual/isaacsim/kit/logs`。未声明时完全冻结；不存在、布局不符、任意 cache 目录或 `extscache` 均拒绝。CPU 布局 fixture 只验证识别和排除机制，不证明真实 Isaac 路径或硬件行为。
