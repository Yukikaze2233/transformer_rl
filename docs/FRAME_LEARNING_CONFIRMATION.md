# 固定学习率后的独立确认执行器

`tools/run_learning_confirmation.py` 消费 `select_learning_rate.py prepare-confirmation` 已封存的定义和 requests。它只执行确认评估，不训练、恢复优化器、调整学习率，也不再次比较三个学习率。普通 MLP、历史编码 MLP 和各类 Transformer 使用同一个确认规则；原选择中 `no_eligible_rate` 的架构保留这个身份，不为它额外挑模型或发请求。

每个已选架构固定使用所选学习率下的三个原训练 seed，各执行 `11701`、`12701` 两个噪声流。每个请求仍是原来的 50 case、8 环境、4,001 个策略步，100 Hz 策略、200 Hz 物理，稳态统计去掉 reset 后前 200 步、至少 200 个稳态样本。两组 seed 仅确认留出的噪声流；38 个确定性 case 和 12 个 noise/combined case 的设置保持不变。这不构成新初态或新扰动域测试。

## 原准备定义与执行定义

原 confirmation `manifest.json` 的 `status=prepared_not_queued`、`execution_implemented=false` 属于 selector 的准备模块，始终保留原字节。新的独立目录存放 runner `manifest.json`：

```text
format: transformer_rl.learning_confirmation_execution
schema_version: 1
status: prepared_not_queued
confirmation_manifest: {path: 绝对路径, sha256: 原文件 SHA}
confirmation_sha256 / choice_sha256 / campaign_sha256: 原 canonical SHA
helpers: runner、selector、learning controller 及其原 helpers 的文件 SHA
runtime: 原 Python executable 身份及实际 CPU checkpoint runtime
cache: {path: 独占空目录绝对路径, device, inode}
locks: {原 resource/study/learning lock 和新 runner lock 的绝对路径: {device,inode}}
learning_controller: {receipt: 绝对路径文件 SHA, pid, start}
input_closure: {原输入绝对路径: 文件 SHA}
protocol: 原请求数、50×8×4001、200/200、两个 noise seed、单次 attempt 和时间预算
execution_implemented: true
training_implemented: false
no_reselection: true
sha256: 完整 execution definition 的 canonical SHA
```

`execution_implemented=true` 表示这个模块具有实际调度能力；准备成功不代表已经排队、启动或完成确认。`prepare`、`validate`、`audit` 都不启动 worker。只有显式 `run` 会执行原请求。源代码、prepared grid、原学习 campaign、选择、原确认 manifest 都只读；评估产物写入原请求规定的确认目录，runner 元数据写入其独立目录。

准备时，完整 90 cell 的训练和四个开发评估 seed 必须已封存，原 completed summary 必须与重新核验的 grid audit 一致。选择和 assessment 使用原 selector 全量校验：原 checkpoint、source/config/snapshot、开发报告与 trace、CPU 延迟收据及 graph 仍须一致。新定义同时冻结这些证据的 artifact closure，之后发生字节变更会拒绝执行。

## 资源和进程门

runner 使用已有共享 resource、study 和 learning 锁，逐个非阻塞取得，核对路径与打开文件的 device/inode。遇到任意锁竞争就释放本轮已取得的锁；不会替换文件。独立 runner 锁阻止同一定义并发运行。

等待门还检查原 learning `controller.json` 的具体 PID/start 已消失、它所有已记录的 worker 不再运行或处于未知所有权状态、原三个 predecessor campaign 已封存并退出、整个 90 cell grid 为 `development_complete`。取得全部锁之后再次审核同一证据和进程状态。原 learning controller 退出后仍留下 `status=running`，因此该字段不能单独证明存活或退出；实际判断使用 `/proc` start identity。

实际命令严格等于原 request 的 bare `evaluate-suite` argv。不会加 learner flags、替换 checkpoint、重采 seed 或缩减 case。受控环境增加 `PYTHONDONTWRITEBYTECODE=1` 和每个请求独占的空 `PYTHONPYCACHEPREFIX`，避免读取冻结源旁边的旧 bytecode。CPU runtime 审核使用原 CPU helper；真实评估环境继承当前 CUDA 可见性，不隐藏 GPU。收据只记录受控环境项，不写入继承环境中的凭据。

每个 worker 保存实际 PID/start、PGID、`/proc` argv 和受控环境观察值，并在原 leader 尚可核验时记录 group 成员的 PID/start。成功端点要求原 worker 已退出、返回值严格为整数 0、没有超时、进程组没有剩余 worker。异常或超时回收时，必须由仍活的原已观测成员锚定同组，发 TERM 和 KILL 前分别核验；数字 PGID 本身不证明所有权。无法证明时不发送信号，保留 `ownership_unresolved` 和原 handle，停止后续 launch，即使 leader 已经退出也不当作资源空闲。worker 输出、请求、过程收据、50 个报告、control 和 trace 均有文件 SHA；独立 audit 同时验证 runner 执行证据和原 selector 的内容/指标审核。

## 单次 attempt 与删失

一个 request 只有原 `attempt_0000`。目录一旦创建就已经使用这次机会。失败、超时、输出缺失、被中断、已有未封存目录均不重试、不换 seed、不退款，也不创建 `attempt_0001`。后续 `run` 仅跳过这些已尝试目录并继续尚未开始的原请求；存在额外目录或原封存产物被修改会拒绝执行。

成功的请求先在原目录通过完整诊断校验，再在 `.validation` 私有子目录用报告和 trace 硬链接构建候选收据。候选 control 使用独立副本，仅把内嵌的 `trace.path` 指向验证目录；原 control 字节和 SHA 始终保留。候选再经过真实诊断与 selector 内容审核，最后以原子、独占方式发布原 request 路径下的 `receipt.json`，以及独立 `execution.receipt.json`。校验期间或在发布前骤停时，原路径的 ready receipt 始终不存在。发布之后骤停仍可能缺少 runner 最终 seal，runner audit 会保留缺项，不能据此称为完整确认。失败仅保留独立故障收据和已有的部分输出；不会发布一个可被当作完整评估的 selector receipt。未封存目录也不会被补写成成功记录。

完整确认执行后调用 `selector.audit_confirmation`，同时重新核验实际执行 request、worker 身份/命令/环境和所有输出 SHA：

| 状态 | 含义 |
|---|---|
| `not_ready` | 尚未开始、失败、未封存、缺报告、真实必要指标缺失或确认证据不完整 |
| `confirmed` | 原选模型的完整两个噪声流均通过原逐 case 联合门 |
| `not_confirmed` | 评估完整，但原门未通过；原学习率选择保留，不重新选型 |

完整生产报告可以有 0 个 ended episode 和 `success_rate=null`。它是有效的删失结果，不能补成 0 或 1；低于 8 个 ended episode 会使原门失败。真正缺失的必要指标依旧是 `not_ready`。success 的口径沿用原 selector：所有 ended episode 是分母，合格的 nonterminated ended episode 是分子；它不是单独的“任务全完成”或“每一步均跟踪准确”判据。最终结果不宣称架构 winner 或硬件部署就绪。

## 命令与产物

```sh
python tools/run_learning_confirmation.py prepare \
  --confirmation-manifest /absolute/original-confirmation/manifest.json \
  --output-root /absolute/separate-confirmation-execution \
  --worker-timeout-seconds 21600 --max-wait-seconds 604800 --poll-seconds 30
python tools/run_learning_confirmation.py validate --manifest /absolute/separate-confirmation-execution/manifest.json
python tools/run_learning_confirmation.py audit --manifest /absolute/separate-confirmation-execution/manifest.json
```

以上命令不调度评估。实际执行需要使用同一 Python runtime 显式调用 `run --manifest ...`；时间预算已在定义中封存，不能在 run 时改变。runner 会保存每次控制器 invocation 的独立过程记录、各 attempt 收据、summary，以及每次完整结束后的独立 audit 文件。重复 validate/audit 不写入目录；重复 run 不覆盖任何已有 attempt。

本模块的作者验证仅包含 CPU 合成协议与故障测试。合成 90 cell/50 case fixture 用于测试接口、完整分母、锁和失败处理，不是新的真实训练或物理评估数据。本次没有真实 choice 可供原生调度，没有运行 GPU、Kit、PPO 或真实确认请求。
