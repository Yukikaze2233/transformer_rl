# 独立运行时存储校准

完整架构矩阵启动前，需要实际 checkpoint、优化日志、完整控制 trace 与运行时缓存的测量材料。`transformer_rl.calibration` 提供独立入口；校准使用自己的输出目录、小规模整项训练预算和私有评估 seed，不产生正式 validation、held-out 或选型成绩。

校准保留已冻结协议中的模型、初始化、训练 seed、PPO、控制周期、环境、rollout 长度和第一阶段配置。只缩短更新数，随后用另一个新进程评估全部原场景、全部副本和原评估步数。默认覆盖协议中全部候选，每个候选使用一个原训练 seed；显式子集会记入覆盖范围，不能代表未测模型或完整训练期间的峰值。

## 入口

先在目标主机准备新的完整 `exposure_protocol`，确认外部原始文件 SHA。源码变化后必须重新准备历史矩阵和协议；旧冻结副本不能用于新源码。

```sh
python -B -m transformer_rl.calibration plan \
  --protocol /absolute/protocol.json \
  --expected-protocol-sha256 <actual-raw-sha256> \
  --plan /absolute/calibration.plan.json \
  --output-root /absolute/new-calibration \
  --training-seed <original-training-seed> \
  --updates 2 --evaluation-seed 92001 \
  --max-owned-bytes <explicit-prior-whole-calibration-ceiling> \
  --free-margin-bytes <explicit-free-space-margin>

python -B -m transformer_rl.calibration run \
  --plan /absolute/calibration.plan.json \
  --expected-plan-sha256 <actual-raw-plan-sha256>
```

`plan` 不启动 SDK，也不创建校准输出目录。执行入口必须等原 curriculum、diagnostics、learning 三个队列真实闭合，并持有原来的两个内核 flock。校准控制器使用独立的授权格式，不伪造正式生产 storage contract。

## 生命周期与产物

每个候选先运行独立训练 worker，正常关闭并核验真实 checkpoint、初始化、私有 seed、更新、完整 rollout、Adam 配置及步数、RNG、日志和不可退款的整项预算，再运行独立评估 worker。worker 使用新的字节码目录、私有 runtime profile，以及原适配器的实际 SDK 路径读回。原始请求、父子 PID/start/argv、继承描述符、运行 profile、关闭结果和文件 SHA 都需一致。

控制器在启动、采样、优化、原子文件发布和关闭期间采样整个自有目录。它分别记录 runtime、checkpoint、trace、metric 和其他文件的逻辑及 allocated 字节；目录、采样日志和终态元数据也计入限额。超限或后台观察错误会停止这次执行；没有重试、退款或目录复用。

实际观察到的 helper 以 PID/start/UID 持续跟踪。调用者及观察者祖先被排除，已登记 helper 即使释放目录引用，仍继续核验其自身及后续子进程。控制器在原锁内等待这些已观察进程终态，不按 PID 猜测发送信号。不可读 `/proc` 字段保留为观测盲区。

关闭期间持续核验输入、源码和存储。观察、进度写入或中断发生异常时，校准记为失败，记录有界错误摘要，并继续持锁核验已观察进程的退出；这不构成训练重试。锁退出、结果发布或最终完整协议核验失败，也不能记录为完成。

`result.json` 保存 worker 和测量材料，`storage.final.json` 保存采样到结果发布后的事实，`completion.json` 给出终态状态。正常结束还需在所有终态文件写入后实际检查目录限额和剩余空间。完整 trace 必须绑定实际 checkpoint、私有 seed、控制周期、原场景顺序和全部行；其控制统计由 trace 重新计算，不只检查压缩文件大小。

写入失败时尽力发布失败终态；若最后检查失败，会先移除自有目录中的临时成功终态，避免持续 I/O 故障留下成功标记。自有目录身份发生变化时不向替换目录写入。缺少正常终态或返回的终态错误都不能作为成功校准证据。

终态 runtime 文件清单是已关闭 worker 的文件字节测量；observer 的分类 maxima 是离散采样观察值，二者不能混用。采样不能证明连续峰值、整个系统的缓存上限、未观察到的 helper 闭合或硬件延迟。校准结果始终保持 `production_storage_authorized=false`、`formal_architecture_selection=false`、`hardware_verified=false`。正式矩阵仍需依据实际测量、明确余量和完整剩余预算独立制定并核验存储合同。
