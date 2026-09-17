# Kaiser：七变体正式训练启动记录

## 状态与实际启动时间

正式队列实际启动于 **2026-09-15 05:44:19.296248 +08:00**，9月15日按用户要求暂停。
9月17日已在新root恢复剩余任务，见 [暂停后续训](KAISER_ARCHITECTURE_RESUME.md)。
以下进度与操作说明保留首次启动时的历史记录。
`requested_not_before=2026-09-15T00:00:00+08:00` 是最早允许时间，**不是实际开跑时间**；
`start_reason="user clarified visual-only and requested continue"` 已写入远端 `launch.json`。

初始观察约 7 分钟，首两项 last_token_attention 均已超过 100 updates，并各自保存通过
schema-4 loader 校验的 checkpoint。机器可读收据见
[architecture-run-start.json](evidence/architecture-run-start.json)。

本次目标是固定任务与预算下比较 Transformer 形式，同时保留 MLP/GRU 对照。
**16M 是学习曲线首档，不是收敛保证或架构赢家结论；统一优化配方不代表每种模型各自最优超参。**
辅助监督变体仍属于独立 supervision 组。

底盘修改仅作视觉参考；训练沿用原 task source、reward、驱动假设和资产。
没有替换训练 USD，也没有引入 15-body 动力学重建或改变质量、碰撞、惯量。
旧预跑 `learning-curves-16m-20260914T1912Z` 的 update 460/465 checkpoint 保留，
本队列使用新初始化、新 source 和新 runroot，不 resume、不计入旧预跑样本或额外 seeds。

## 部署与冻结身份

| 项目 | 值 |
| --- | --- |
| Git 来源 | `https://github.com/Yukikaze2233/transformer_rl.git`，独立 clone、detached checkout |
| Commit | `80a45b40c9afc7e6e3356ccbb6fa15f8933a3970` |
| Source | `/home/kaiser/robot-rl-sim60/transformer-architecture-20260915` |
| Study | `/home/kaiser/robot-rl-sim60/architecture-16m-20260915T0540` |
| tmux | `transformer-architecture-20260915` |
| runtime | `/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh` |
| spec / base | Study 下的 `derived-spec.json` / `derived-base.json` |
| plan / runroot | Study 下的 `run/plan.json` / `run/` |

`PYTHONPATH=SOURCE/src:SOURCE`，实际导入包来自新 checkout 的 `src/transformer_rl`。
部署和 checkpoint 校验时工作树均 clean，运行记录 Python=3.12.13、Torch=2.11.0+cu128。
CLI help、plan/run help、实际 plan 和七个 variant inspect 全部通过；inspect 没有启动环境。
共冻结 **56 份配置：7 份训练、49 份最终场景评估**。

| SHA-256 类型 | 值 |
| --- | --- |
| 规范化 spec | `63f74d22e5bd8bcc24ad7c720b116788ccb91510b66579a848bb32ecbf229e19` |
| plan | `7a37cad280f7676b4103c0ec883bf1c8ee16e41adf72a9ae9c843f03ea90ed57` |
| 实际 package source | `e6ef86cae8177052999b773d2615af5edaa2c46774a9942548fa0f47242af5c7` |

`freeze.json` 另含 adapter、worker、runtime、外部 task Python 文件和 contract 文件的 SHA。
实际 SDK 环境 identity 记录在训练 `environment.json`、checkpoint 和探针报告中；
package SHA 本身不覆盖外部仿真器或资产。

## 正式预算与评估协议

- variants：last_token_attention（index + last）、time_attention（elapsed + query）、
  index_attention（index + query）、gated_attention、supervised_attention、history_mlp、history_gru。
- training seeds：**1011、1022、1033**，沿用原规格的 variant-major 调度顺序。
- 共同配置：d64 / L2 / heads4 / FFN128 / history16，mean_init_scale=0.1、initial_std=0.2；
  LR=3e-5、epochs=2、minibatches=4、target_kl=0.01，其余继承 optimized_control。
  MLP `[128,64]` 和 GRU hidden64 保持原配置，baseline 不执行 Transformer attention 层。
- 每 job **512 envs × 32 rollout × 977 updates = 16,007,168 transitions**；
  **21 jobs 合计 336,150,528 transitions**，完成预算以实际 completion 为准。
- `diagnostics=false / action_clip=100.0 / checkpoint_interval=100`。
- factory=`examples.isaaclab_task:make_env`，worker=`examples._isaaclab_process`，保证先写 CLI 产物再退出 SDK。
- training environment：taskroot=`/home/kaiser/robot-rl-sim60/v40-live-snapshot03`，
  contract=`contracts/own_v40_v2.json`，`num_envs=512 / mode=train / stage=locomotion`。
- evaluation common 覆盖 `mode=evaluation / num_envs=8`；seed=`[301]`，每场景 **2000 steps**，
  `settle_steps=200 / min_steady_samples=200`。

| 场景 | fixed_command [vx, wz, height] |
| --- | --- |
| stand_low / stand_mid / stand_high | [0,0,0.28] / [0,0,0.30] / [0,0,0.32] |
| forward / reverse | [0.5,0,0.30] / [-0.5,0,0.30] |
| turn_left / turn_right | [0,1,0.30] / [0,-1,0.30] |

中间 checkpoint 每100 updates保存，**仅最终 update 977 自动评估**：每 job 7 次，全队列 **147 次**，
预期每次16,000 transitions、总2,352,000 evaluation transitions。
正式评估保留17个 PRE-reset signals：height/vx/wz signed errors、angular_velocity_x/y、
4个 leg targets、2个 wheel targets、6个 efforts。统计逐 env/episode 中心化，
没有可用稳态窗口时保留计数和 `null`，不把 unavailable 补成0。
它是 **100Hz policy 边界采样**，不是 MCU 高频 PID 或完整200Hz physics trace。
任务可 reset，不能将2000 steps或多个高度拼成连续存活证据。

## 唯一随机策略接口探针

正式队列前执行一次 **random-policy / wiring-only** 探针：8 envs × 500 steps，
零 optimizer updates，随机初始化 seed9001，eval seed301、stand_mid，稳定性窗口200/200。
先由真实 factory 创建 SDK 环境并取得 metadata，再保存随机 checkpoint；
调用当前 `evaluate_policy` 时复用同一个真实环境，metadata identity 由 evaluator 正常校验。
独立 probe owner 复用 SDK worker 的 app registry，在写出报告后关闭 SDK，实际进程 exit=0。

**05:44:05 +08:00** 验证通过：17个 signal 全部存在，4000输入样本/信号，
1600 settle、2400 retained、8个 partial segments；last-token 当前帧/time/command 匹配检查通过，
没有绕过检查。该探针有可用窗口；空窗口的 null 语义由冻结实现保留，不宣称本次物理探针覆盖了空窗口。
产物为 `random-policy.pt`、`random-policy-evaluation.json` 和 `random-policy-wiring-receipt.json`。
随机策略的统计仅证明接口已接通，**不作为控制效果、架构排名或正式样本**。

## 初始真实进度与 checkpoint

监控时间 **2026-09-15 05:51:21.304464 +08:00**；CPU 校验探针开始于05:51:25，
完整结果保留在 Study 的 `initial-observation.json`。

| Job | PID | updates | 新 optimizer steps / 计划 | 实际 samples | 含启动吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| last_token_attention/1011 | 235168 | 118 | 944 / 944 | 1,933,312 | 4,617/s |
| last_token_attention/1022 | 235169 | 118 | 944 / 944 | 1,933,312 | 4,614/s |

总计已记录 **3,866,624 samples**；两项 early-stop=0%、optimizer利用率100%，
最近10 updates吞吐约5,739/s、5,726/s。已记录指标均有限，未见 matching、NaN、OOM 或 traceback。
当前 **2 running、19 queued、0 completed、0 failed/timedout**，正式最终评估尚未启动。
排队为 last_token_attention/1033，以及其余六个 variants 各自的三个 seeds。

两项 `train/checkpoints/checkpoint_000100.pt` 均实际加载验证：schema4、weights_only、CPU，
update100、samples1,638,400、Adam step800、source精确匹配且dirty=false、环境identity一致。

- 1011 SHA：`bfa6c20ca6f884e598802ce1292fbc1c6c653bd3821976f0118fb705308251e6`
- 1022 SHA：`3549b0ba888b55b93d4ab8985ab189dc2c3330f9cb0fe97902c4b0f814a5c15b`

初始正式训练监控：全机GPU显存峰值 **10,069 MiB**、util峰值 **97%**，
WSL最低MemAvailable **6,452.3 MiB**，两个worker RSS合计峰值 **9,492.6 MiB**（共享页可重复计数）。
这些峰值覆盖正式队列初始观察，不包含此前随机策略接口探针。

收尾采样 **05:56:42.831234 +08:00**：两项均达到 **231 updates / 1,848 optimizer steps /
3,784,704 samples**，合计 **7,569,408 samples**，early-stop仍为0%；含启动吞吐分别5,109/s、5,108/s。
队列仍为2 running / 19 queued，GPU峰值仍10,069 MiB / 97%，MemAvailable最低更新为
6,442.5 MiB，worker RSS合计峰值9,503.8 MiB。远端另存 `handoff-observation.json`。
冻结文件哈希、初始采样与随机探针计数交叉核验通过；旧预跑460/465 checkpoint SHA也确认未变。

## 长期监控与操作命令

`max_parallel=2 / cuda:0`；每train软预算7200秒，每job总timeout10800秒（含全部最终评估）。
外层 GNU timeout **36h**，TERM 后 kill-after=120s，覆盖21×3h/2=31.5h及grace。
无旧deadline复用，不因交接返回停止队列。

约每10秒追加 `resources.jsonl` 并原子刷新 `status.json`：全机GPUutil/mem、WSL MemAvailable、
本批worker PID/PGID/RSS、逐job updates/optimizer steps/实际samples及队列状态。
MemAvailable连续30秒低于1GiB，监控仅终止本批scheduler/所属进程并有界清理；不按名称清理其他用户进程。
`events.jsonl` 记录 started/job_state/queue_exited/completed/failed；
结束后自动summarize，`execution-receipt.json` 和 `master-exit.json` 保存真实退出码。
`run/execution.json` 仅是执行marker，不代表训练完成。

连接：

```bash
ssh -S /tmp/opencode/kaiser-training-control -oBatchMode=yes -p2222 kaiser@192.168.64.234
```

在Kaiser的Bash中查看：

```bash
STUDY=/home/kaiser/robot-rl-sim60/architecture-16m-20260915T0540
SNAPSHOT=/home/kaiser/robot-rl-sim60/transformer-architecture-20260915
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT"
python -m json.tool "$STUDY/status.json"
OMP_NUM_THREADS=1 python "$STUDY/learning_progress.py"
tmux attach -t transformer-architecture-20260915
```

需要停止本批时使用以下命令（本次交接未执行）：

```bash
python "$STUDY/request_stop.py"
```

该命令核对 `launch.json` 的monitor PID/start-time及精确脚本路径，再通过pidfd发送SIGTERM；
monitor会停止队列并只清理所属进程。它是有界中断，不承诺保全正在采集的rollout；
已保存checkpoint留在原root，确认停止应查看真实exit收据和本批PID是否退出。

每job日志位于 `run/jobs/VARIANT/seed_SEED/train.log`，优化指标位于 `train/metrics.jsonl`；
最终评估为 `evaluation_SCENARIO_301.json`。结束后查看 `run/summary.json`、`run/summary.csv`、
`execution-receipt.json` 和 `master-exit.json`，或显式刷新：

```bash
python -m transformer_rl.experiment_cli summarize --root "$STUDY/run"
```

本地仅新增本记录和小型JSON收据，保留LF；大模型/原始日志留在远端study，未加入仓库。
本次未改core、未commit/push，交由主agent统一提交。
