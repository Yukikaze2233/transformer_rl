# 固定授权的独立进程训练执行器

`exposure_campaign.run_protocol()` 消费完整 `exposure_protocol`，为每个架构和训练 seed 执行全部预声明阶段。每阶段使用独立 OS 进程，下一阶段只接收紧邻上一阶段的真实封存端点，完整恢复 actor、critic、Adam、全局与私有 RNG 和累计更新/样本时钟。它不根据 reward 升级课程、不回滚、不补训、不换 seed，也不复用旧输出目录。

这份执行器提供训练和独立评价执行能力。没有独立评价 provider 时，最终状态为 `training_completed_evaluation_pending`，所有评价单元保留 missing。接入当前 provider 后，默认 `confirm_heldout=True`：先关闭全部 validation 单元并封存不可变 validation choice 与实际 checkpoint 身份，再执行 held-out 全矩阵；最终状态为 `comparison_closed_deployment_qualification_pending`。数值失败对应的缺失端点和评价单元仍保留在原分母中，不以状态名称代替完整性检查。

显式传入 `confirm_heldout=False`，或 CLI 使用 `--validation-only` 时，仅执行 validation，最终状态为 `validation_closed_heldout_selection_pending`。held-out 成绩不能修改封存的 validation choice。训练、validation 或比较流程闭合都不会自动授予部署资格；实际推理延迟、模拟器证据和硬件验证仍按各自真实材料判定。

## 连续训练的中间模型

`exposure_protocol.freeze(..., checkpoint_interval=400)` 显式冻结保存间隔。以单阶段 1,200 次更新为例，协议 v2 声明 `checkpoint_updates=[400,800,1200]`，并为三个模型分别生成全部场景、validation/held-out seeds 的评价单元。更新数和 fresh rollout 预算保持原定义，评价和磁盘预算按新增模型扩展。未指定该参数时保留 v1 的阶段末保存行为；不会隐式采用 history 配置中的日志间隔。

训练 worker 在完整 rollout、成功优化与日志发布之后保存模型。400、800 次更新的保存操作继续使用同一环境、collector、历史、actor/critic、Adam 和 RNG；不 close/open、不 reset、不在学习进程中运行评价。最终模型仍使用 `endpoint.pt` 与 `endpoint.json`，中间模型使用 `checkpoint_00000400.pt`、sidecar 和独立封存记录。新增 `sealed_checkpoints` 将每个模型绑定到实际更新数、样本数、优化次数和封存时的日志前缀。

阶段关闭后，独立评价 worker 分别读取这些模型。每个评价 batch 的身份包含累计 `checkpoint_update`，输出目录为 `stage_0000/checkpoint_00000400/evaluation_validation_701` 等，不能把最后模型替换为中间模型，也不能把同阶段的不同时间点合并成一个 batch。评价环境重置不影响此前连续学习轨迹。最终选型仍只使用每个训练 seed 最后阶段的最终模型；中间模型用于学习趋势及已学能力退化的证据，不能以更好的中间分数替代最终模型。

只有已经满足预声明任务条件的能力才记为 acquired；其后退化才作为遗忘证据。始终未满足条件属于未学会。中间模型缺失仍保留原评价分母。后续发生已核验的数值失败时，失败前完整封存的模型可以独立评价，未完成模型保持 missing，原整项训练预算仍收费、不重试。中间模型不能作为跨阶段 continuation parent；下一阶段只接受紧邻上一阶段的完整最终端点。

保存会消耗真实墙钟时间和磁盘空间，不代表提高了训练样本量。连续性由真实 CPU 学习状态与保存前后环境/历史检查验证；生产 SDK 的保存开销、缓存峰值与控制表现仍需要实际运行记录。

## 原始字节授权与一次总预算

入口必须传入调用者已固定的 **protocol 文件原始 SHA**，不是 JSON 内部的自签名：

```python
run_protocol(
    protocol_path,
    expected_protocol_sha256=externally_fixed_raw_sha,
    storage_contract_path=actual_storage_contract,
    expected_storage_sha256=externally_fixed_storage_raw_sha,
    evaluation_provider=None,  # or "transformer_rl.exposure_evaluation"
    confirm_heldout=True,  # False runs validation only when a provider is present
)
```

改变候选、seed、课程、配置、源、环境 snapshot、runtime、两锁身份或原队列绑定后重新计算 JSON 签名，不能自动取得执行权限。controller 和 worker 都核验真实文件、实际 SHA/长度和完整协议重建。新的源必须重新 prepare/freeze，不能把本机路径定义当作 Kaiser 的可执行输入。

在 job 的第一个 child 前，controller 写入一次 `exposure_whole_job_reservation`，预留该 job 的**全部阶段**更新和 fresh transition 预算。阶段内部 reservation 只是同一总预算的分项。失败或后续阶段未执行时，总预算仍收费，不退款。summary 同时保存完整候选和评价分母，封存端点的实测成功更新/样本，以及数值失败的实际采集和未知优化记账范围；预留样本不表示已采样。

## 实际 OS 身份与资源占用

三个原始队列都必须通过原 `check_dependency(..., require_complete=True)` 完整闭合证明，且其原 controller 和 worker 句柄已终止。controller 随后获取两把**原 inode** 的独占 flock，重新确认闭合状态，并把相同锁 FD 继承给 child。worker 再核验直接父进程 PID、start tick、完整 argv、canonical controller 字节收据、父/子 FD 的 inode、双方真实 FLOCK 条目及另一次独立 open 的锁冲突。

启动前实际探测 `pidfd_open` 和 `pidfd_send_signal(..., 0)`。超时或中断只向当前 child 的内核 pidfd 发信号，不按复用 PID 或整个进程组猜测目标。获取 child pidfd 失败时停止执行，不冒险给一个未绑定句柄发信号。controller 关闭自己的 FD 而不执行 `LOCK_UN`，仍存活的 orphan child 会继续持有继承的锁，直到它退出。

每个 child 使用新的空 `PYTHONPYCACHEPREFIX`，启动前不存在、创建后为空；`-B` 配合该路径避免读取此前 checkout 的陈旧 bytecode。controller 另独占创建该 worker 的 `runtime/` 和 `runtime.profile.json`，固定临时目录、XDG、Warp、Torch、CUDA 与 OmniClient 的输出位置，以及 Kit portable root、tokens 与 settings。profile 的原始 SHA 通过继承环境传入，process/completion/outcome 使用同一份实际收据；用户的 `worker_env` 不能覆盖受保护的路径或 profile 绑定。profile 与所有固定目录的 UID、inode、类型和文件系统在实际启动及等待期间持续核验。

实际 runtime origin、SDK、snapshot 和 contract 在完整边界验证；rollout 回调继续检查源、协议/请求/父身份、两锁、runtime profile 和磁盘。完整模型重建不放在每个物理 step 的回调里，否则会改变控制工作量。实际 chassis 启动还读取 Kit 的解析结果，再核对声明 tokens/settings；合成 CPU worker 只能证明 profile 继承与检查，不具有真实 Kit readback。

`exposure_process` 使用已有应用注册表，在自己的独立进程中关闭该阶段创建的应用。实际 Isaac、CUDA、部署延迟及完整操作系统动态库仍需真实运行证据，CPU 检查不代表这些内容已经验证。

## 磁盘预留与实测大小合同

storage contract 是单独固定的 canonical JSON，绑定 protocol 原始 SHA 和当前源，包含六个正整数 cap：

- `checkpoint_bytes`：一个完整 actor/critic/Adam/RNG checkpoint 的上限；
- `trace_bytes_per_policy_sample`：未压缩物理 trace 每个 policy sample 的上限；
- `metric_bytes_per_update`：优化日志每次更新的上限；
- `inflight_bytes`：请求、报告和其它在途出版文件的额外空间；
- `runtime_cache_bytes`：worker 输出目录内完整 `runtime/` 与 `empty_python_cache` 的额外空间；
- `free_margin_bytes`：保留可用空间。

`measurements` 必须具有 checkpoint、trace、metric、runtime_cache 四种实测材料，每项为 `{kind, receipt, units}`；实际文件凭据和单位重新检查，cap 不能低于观测大小。测量材料是大小依据，不具有教师、模型性能或安全资格。

runtime_cache 材料不是一个小标记文件。它是格式为 `transformer_rl.exposure_runtime_cache_measurement` 的完整目录清单，字段为 `format/schema_version/root/files/total_bytes`，其中 `files` 为所有实际文件收据，按绝对路径排序。材料核验会重新遍历其声明的 `root`、核对成员和实际总字节，按**目录总字节**核对 cap。这证明清单完整覆盖该声明目录，不表示已经确认生产 SDK 的真实缓存路径；测试中的小 CPU cache 只证明检查机制。

当前运行时的类别 cap 扫描范围限于本 worker 输出目录：完整 `runtime/` 与 `empty_python_cache` 计入 runtime-cache，profile JSON 及其它目录内 SDK 输出计入 inflight。输出根、全部子目录和文件均检查当前 UID、普通类型、无 symlink 与同一 `st_dev`；跨文件系统的目录或文件会拒绝，不能假设它们消耗或释放了输出所在文件系统的预算。runtime-cache 不获得后续 job 的抵扣。

隔离保留调用者 HOME 与只读 SDK 输入。声明路径和实际 Kit readback 不证明 SDK helper、GPU 驱动或所有第三方库均没有其它写入；输出目录外、其它文件系统的增长不属于该类别 cap。生产启动前仍需用真实冷启动、运行、checkpoint/trace 发布及关闭过程的峰值占用校准预算，并检查实际写入范围。声明目录清单和 CPU 检查不替代这些峰值证据，也不表示已经对全系统缓存施加 cap。

每个边界按剩余阶段全部预声明 checkpoint、更新日志及**全部尚未闭合评价单元**的原始 trace 预留空间。独立 worker 的 namespace 不共享，也不在关闭后自动删除；每个尚待训练的阶段与尚待评价的完整 role/seed batch 都分别预留 `runtime_cache_bytes + inflight_bytes`。只要仍有待执行 worker，边界再增加一份相同大小的活动 worker 安全头寸。实际 worker 的 `worker_storage_guard(..., active_worker=True)` 固定去掉这一份头寸，让合法的当前 namespace 写入能够使用它，并记录 `active_runtime_headroom_bytes`；这是启动前确定的常量，不依据已写 cache/log 抵扣预算，底部仍守住完整 cache、inflight 和 free margin。一个场景闭合而同 batch 仍有未闭合场景时，不能提前释放该 worker 的 namespace 预留。已闭合目录仍在盘上，其实际占用已反映在 `statvfs` 可用空间中，不能抵扣未来 job 的上限。

出版并发峰值只另外计算一次最大剩余评价 batch 的临时 memmap/未压缩 NPZ 共存空间，以及一份 checkpoint 发布副本与可用空间余量，不能把最大 batch 乘以全部 worker 数。这里只使用实际校准合同的 cap 和完整原矩阵计数；还未测量的 SDK 冷启动/关闭峰值不具有数值证据。不假设压缩能省空间。trace 测量使用真实 NPZ 的全部未压缩成员字节，样本量由 `episode_id` 的实际 steps × rows 核对。

child 只从未花费预算中扣减本 worker 明确声明的 checkpoint、metrics 和 trace 字节，并分别核对类别 cap。抵扣额取逻辑字节与 `st_blocks × 512` 实际已占用字节的较小值；稀疏 memmap 尚未写入的空洞仍占用未来写入预算。该 checkpoint 同目录的 `.endpoint.pt.<随机名>.tmp` 随发布文件计入同一个 checkpoint cap，`.endpoint.pt.json.<随机名>.tmp` 是 sidecar 临时文件，计入 inflight 且不获得 checkpoint 抵扣；同类别真实 hardlink 只计一次空间，跨预算类别 hardlink 拒绝。本 worker 输出目录中的 stdout、SDK 输出、JSON 和普通临时文件合计不得超过 inflight cap，不获得后续 job 的预算抵扣；其中完整 `runtime/` 和 `empty_python_cache` 单独受 runtime-cache cap 限制。v1 训练 worker 只允许该阶段唯一的 `endpoint.pt`；v2 按冻结 schedule 精确允许多个文件，同时检查每份模型及总模型字节上限，不遍历或抵扣其他阶段目录。同一阶段保存多个模型只预留一个训练 worker namespace，独立评价 namespace 则按模型、role 和 seed 分别预留。跨模型的 hardlink 拒绝，不能共享 inode 绕过每份上限。每次采样检查、最后一次优化后的 checkpoint 封存前，以及关闭 worker 资源后都重新核对实际文件用量。任何超限都停止，不缩小评价分母，也不删除其他训练文件。

controller 还在等待 child 期间按最长一秒的等待间隔检查该 worker 的实际用量，并在 child 退出后再检查。物理调用或优化器尚未返回时，本 worker 输出目录内的 SDK/日志增长也受上述限制；停止只通过该 child 的真实 pidfd，保留已写输出与进程终态，不重试。

进程句柄优先使用 Python 的 `os.pidfd_open` 与 `signal.pidfd_send_signal`，随后使用 libc 的对应封装。两层均缺少封装时，Linux x86_64/aarch64 的 LP64 运行时可通过 `libc.syscall` 调用同一个内核 pidfd ABI；不改用可复用的数值 PID 发送信号。调用号由 [Linux x86_64 syscall 表](https://github.com/torvalds/linux/blob/master/arch/x86/entry/syscalls/syscall_64.tbl)及 [asm-generic ABI](https://github.com/torvalds/linux/blob/master/include/uapi/asm-generic/unistd.h)确认。未知 ABI 或内核/权限拒绝仍在 Popen 前失败，不创建 worker。这只解决用户空间封装兼容性，实际 kernel handle、信号零探测、运行时与空间 guard 保持原规则。

## 失败分类

只有 disposable worker 在真实 `env.step()` 或 `PPOTrainer.update()` 中捕获到 **built-in `FloatingPointError` 对象**，且失败记录位于 collect/optimize、关闭正常、实际计数不超过原预算，才作为可继续其它 job 的数值失败。worker 内拦截仅记录类型和发生位置，随后重新抛出原异常；不改变优化算法。错误文字包含 `nan` 或 completion 自称数值错误，均不足以授权继续。

这种失败保留该 job 未完成阶段和全部相应评价单元的 missing，并转到下一个原候选/seed。源、输入、协议、锁、学习链或字节漂移，磁盘不足、人工中断、deadline 和未知错误停止整个 campaign，保留已发生结果和已收费预算。没有自动 retry。

## 独立评价接口

provider 被限定为当前源中的 `transformer_rl.exposure_evaluation`：

```python
plan_request(
    protocol, endpoint_receipt, cells, directory, leases, controller_receipt,
    selection_receipt=None,  # required immutable validation choice for held-out
    storage_progress=None,  # actual completed_stages and closed_cells from controller
)
# returns {"command": [actual Python, ...], "request": actual_file_receipt}

verify_result(protocol, endpoint_receipt, cells, directory, request_receipt)
# returns {"status": "completed", "cells": {cell_id: ...}, "artifacts": ...}
```

同一 batch 只包含同一 job、stage、checkpoint（v2）、role 和评价 seed 的全部场景。validation 请求不能依赖之后的选择；held-out 请求必须绑定已封存 choice 的实际收据。`storage_progress` 为 controller 的真实预算进度，字段为 `completed_stages` 和 `closed_cells`，不能添加未声明项或重复收费项。provider 创建新目录；controller 用同样的内核/锁机制启动独立 child。完整合法但成绩差的场景仍保存真实指标和 grade；场景缺失不能减少原分母。公共 `verify_segment_endpoint(protocol, job, stage_index, endpoint_receipt)` 先验证完整实际学习链，再返回实际端点，可供评价 worker 复用。

CLI 示例中的原始 SHA 与路径都必须换成实际目标主机的固定值：

```bash
python -B -m transformer_rl.exposure_campaign \
  --protocol /absolute/frozen_protocol.json \
  --expected-protocol-sha256 FIXED_RAW_SHA \
  --storage-contract /absolute/measured_storage_contract.json \
  --expected-storage-sha256 FIXED_STORAGE_RAW_SHA \
  --evaluation-provider transformer_rl.exposure_evaluation
```

上述 CLI 默认在 validation choice 封存后执行 held-out。仅需 validation 时追加 `--validation-only`；未接 provider 的调用始终保留评价 pending，不会因 `confirm_heldout=True` 进行选择或宣称完整比较。

CPU 集成检查实际启动独立 OS worker、继承真实 Linux flock 和 pidfd、执行四种 actor 模式的真实 Adam 更新并核验跨阶段全学习链。测试替代原队列闭合 provider 的部分只具有 fixture 范围，不能授予原课程、诊断或学习率队列闭合证明，也不能证明模拟器或实机性能。
