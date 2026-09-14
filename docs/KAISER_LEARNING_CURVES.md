# Kaiser：16M 学习曲线首档启动记录

> **本预跑已于2026-09-14保存停止。** 两个checkpoint保留归档，原16项排队任务未启动。用户随后明确机械改动只是视觉修复，正式网络对照恢复，但使用新队列，不恢复或混入本预跑；实际开始时间另行记录，不冒称午夜准时启动。下文均为本预跑的历史收据。

## 结论与范围

2026-09-14 已启动 **6 variants × 3 新 training seeds 的完整长训练队列**。
初始观察约 7.2 分钟，两项真实训练均超过 100 updates，并各自产生通过 CPU loader 校验的
`checkpoint_000100.pt`。队列持续在独立 tmux 中运行；这里是启动收据，尚非完整实验结果。
机器可读记录见 [learning-curves-start.json](evidence/learning-curves-start.json)。

**16M 是学习曲线第一档，不用于宣布架构赢家；共同配方并非每种网络各自最优超参。**
supervised_attention 保持独立 supervision 分组。训练沿用名义 locomotion 合同中的随机命令，
最终评估使用七个固定命令场景；没有非零通信延迟或推扰训练。
这是固定名义场景对照，不构成扰动恢复、硬件鲁棒性或真实部署验收。
本次队列仅含首档 16M，后续 64M/128M 档需要后续决策。

## 独立部署与冻结身份

- GitHub：`https://github.com/Yukikaze2233/transformer_rl.git`，独立 clone 后 detached checkout。
- 精确提交：`e0f97a6719f19f20324323873dbe22ffd65a6e47`。
- 源码目录：`/home/kaiser/robot-rl-sim60/transformer-learning-curves`。
- Study：`/home/kaiser/robot-rl-sim60/learning-curves-16m-20260914T1912Z`。
- Derived spec/base：study 下的 `derived-spec.json`、`derived-base.json`。
- Plan/runroot：study 下的 `run/plan.json`、`run/`；此前 study/runroot 未复用。
- tmux：`kaiser-learning-curves-16m-20260914`。
- runtime：`/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh`。
- `PYTHONPATH=/home/kaiser/robot-rl-sim60/transformer-learning-curves/src:/home/kaiser/robot-rl-sim60/transformer-learning-curves`。
- 启动前及 checkpoint 验证时远端 Git 工作树均 clean，实际包导入来自上述 checkout 的 `src/transformer_rl`。
- study 目录名只是唯一标签；实际启动时间以收据中的 **11:09:27 UTC / 19:09:27 UTC+8** 为准。

| 身份 | SHA-256 |
| --- | --- |
| 规范化 derived spec | `09ecd6afbd1a003af1b6b65676a47f232f3a3950aef161481652f49f1da30bb9` |
| plan | `967bfdccc5027ad95f7377835df40401ef5b00a3ea7804271f92880b398d8a47` |
| 实际导入 package source | `65e6449703a75046e3fe26f8c79ce839e1bb8bad96bfc0038d1f0b75b6426a27` |

`freeze.json` 额外保留外部 task Python 源文件、contract 文件、runtime、adapter 和 worker 文件哈希。
实际环境启动后记录的 contract/asset manifest/task source identity 与两份 checkpoint metadata 一致，
详见 JSON 收据。package SHA 不包含外部仿真，不能仅凭它声称冻结了全部 SDK/资产。

## 精确预算与执行配置

| 项目 | 冻结值 |
| --- | --- |
| variants | time_attention、index_attention、gated_attention、supervised_attention、history_mlp、history_gru |
| training seeds | 1011、1022、1033，原 variant-major 顺序 |
| 共同模型配置 | d_model=64、num_layers=2、num_heads=4、ffn_dim=128、history_length=16 |
| 初始化 | mean_init_scale=0.1、initial_std=0.2 |
| PPO | LR=3e-5、epochs=2、num_minibatches=4、target_kl=0.01，其余继承 optimized_control |
| 每 job 训练 | 512 envs × 32 rollout × 977 updates = **16,007,168 transitions** |
| 全队列训练 | 18 jobs = **288,129,024 transitions**，以最终实际 completion 核实 |
| 诊断 / action clip | diagnostics=false；action_clip 显式 100.0，与任务 contract 相同 |
| 并发 / 设备 | max_parallel=2，两个 slot 共用 cuda:0 |
| 训练软预算 / job timeout | 7200 秒 / 10800 秒；job timeout 包含训练及全部七次最终评估 |
| checkpoint | 每 100 updates 保存；完整训练保存 100、200、…、900、977 |
| 外层预算 | GNU timeout 30h，TERM 后 kill-after=120s；覆盖 18×3h/2=27h 及 grace |

MLP 仍使用原 `[128,64]` hidden，GRU hidden=64；d64/L2/heads4 是共同配置中的 Transformer 参数，
不把它解释成所有 baseline 都执行两层 attention。

训练环境：`examples.isaaclab_task:make_env`，worker 为 `examples._isaaclab_process`；
`task_root=/home/kaiser/robot-rl-sim60/v40-live-snapshot03`，
`contract=contracts/own_v40_v2.json`，`num_envs=512 / mode=train / stage=locomotion`。
evaluation common environment 在此基础上覆盖 `mode=evaluation / num_envs=8`。

每 job **只评估最终 update 977 的 checkpoint**，eval seeds=`[301]`；每场景 2000 vector steps：

| 场景 | fixed_command [vx, wz, height] |
| --- | --- |
| stand_low | [0, 0, 0.28] |
| stand_mid | [0, 0, 0.30] |
| stand_high | [0, 0, 0.32] |
| forward | [0.5, 0, 0.30] |
| reverse | [-0.5, 0, 0.30] |
| turn_left | [0, 1, 0.30] |
| turn_right | [0, -1, 0.30] |

共 **126 次最终 checkpoint 评估**；预期每场景 16,000 transitions，全组评估 2,016,000 transitions，
实际数量由评估报告核实。中间 checkpoint 不自动评估。
任务 policy_dt=0.01s，2000 steps 名义为 20s；20s episode timeout 或失败可能 reset，
不能把多个 episode 或三种站立高度拼成连续 60s 站立证据。

## 初始真实进度

监控采样 **2026-09-14 11:16:39.671512 UTC（19:16:39 UTC+8）**，
CPU checkpoint 验证探针于 **11:16:45 UTC** 开始。

| 运行 job | PID | updates | 新 optimizer steps / 计划 | 实际 transitions | 含启动吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time_attention/seed_1011 | 673104 | 110 | 880 / 880 | 1,802,240 | 4,198/s |
| time_attention/seed_1022 | 673103 | 111 | 888 / 888 | 1,818,624 | 4,240/s |

两项均为新训练；累计已记录 **3,620,864 transitions**。截至该采样各自 KL early-stop 比例为 0%，
optimizer 利用率 100%。验证探针读取最近十次更新吞吐约 5,642/s、5,738/s；
这是首两个 time_attention jobs 的局部速度，不外推成所有架构的完成时间。
已记录指标均有限，日志未发现 traceback、数值一致性异常或 CUDA OOM。

- running=2，completed=0，failed=0，timedout=0，queued=16；独立评估尚未启动。
- 排队：time_attention/1033，以及其余五个 variants 各自的 1011/1022/1033。
- 两份 `train/checkpoints/checkpoint_000100.pt` 均以当前包的
  `load_checkpoint(device="cpu")`（weights_only、schema 3）实际加载；
  update=100、transitions=1,638,400、Adam step=800，source commit 精确匹配且 dirty=false。
- checkpoint SHA：1011 为 `9471c0462eef96de414959dece17de7d5760aecafe54091ee35715e68eea9d4e`；
  1022 为 `ae251b512c6ba23befef90ddfe8f38737e751facecf4a49a16b12ded173a6568`。
- 初始观察的全机 GPU 显存峰值 **12,452 MiB**、util 峰值 **95%**；WSL MemAvailable 最低
  **6,371.7 MiB**；两个 worker 的 RSS 合计峰值 **9,449.8 MiB**（RSS 含共享页重复计数）。
  当前采样 GPU=11,307 MiB / 91%，MemAvailable=6,372.9 MiB。

收尾复查 **2026-09-14 11:21:11 UTC（19:21:11 UTC+8）**：队列仍为 2 running / 16 queued，
1011 达到 **204 updates / 1,632 optimizer steps / 3,342,336 transitions**，
1022 达到 **206 updates / 1,648 optimizer steps / 3,375,104 transitions**；
两者 early-stop 均为 0%，含启动吞吐分别为 4,765/s、4,822/s。
合计已记录 **6,717,440 transitions**。截至此时 GPU 峰值仍为 12,452 MiB / 95%，
MemAvailable 最低 6,363.6 MiB，worker RSS 合计峰值 9,461.0 MiB。
再次核对全部冻结 artifact/checkpoint 哈希、48 份 model/PPO 配置及初始监控原始行均一致。

## 监控、退出语义与后续查看

启动前完成 CLI `--help`、experiment plan/run help、实际 plan 以及六个 variant 的 inspect；
inspect 均为 `environment_started=false`，没有追加真实 smoke。
48 份冻结配置（六份训练、42 份场景评估）及 18 个 job 路由通过 plan/source 校验。
监控脚本通过 Python 编译检查，master 通过 `bash -n`。

独立 study 中 `learning_master.sh` → 30h timeout → `learning_monitor.py` → experiment CLI。
约每 10 秒追加 `resources.jsonl`，并原子刷新 `status.json`：包含全机 GPU util/mem、WSL
MemAvailable、本次 worker PID/PGID/RSS、逐 job 更新数/optimizer 利用率/样本数及队列状态。
MemAvailable 连续低于 1 GiB 达 30 秒才触发终止；先 TERM 本次 scheduler，由其清理所属 stage
进程组，20 秒 grace 后对仍存活且 PID start-time 身份匹配的本次进程组 TERM，5 秒后升级 KILL。
不按进程名或 GPU 占用批量清理用户进程。

`events.jsonl` 记录 started、job_state、queue_exited 以及最终 completed/failed；
`execution-receipt.json` 保存真实 queue exit code、summarize exit code、终止原因和最终逐 job 状态；
外层 `master-exit.json` 保留 timeout 的真实退出码。队列退出后自动执行 summarize（有界 90 秒）。
运行期间 terminal receipt 尚不存在是正常状态；`run/execution.json` 仅是一次性执行 marker，
不能当作 completed。完整 job 需要训练达标且七份独立评估报告全部验证通过。

连接（现有 ControlMaster 可用时）：

```bash
ssh -S /tmp/opencode/kaiser-sensitivity-control -oBatchMode=yes -p2222 kaiser@192.168.64.234
```

在 Kaiser 的 Bash 中查看：

```bash
STUDY=/home/kaiser/robot-rl-sim60/learning-curves-16m-20260914T1912Z
SNAPSHOT=/home/kaiser/robot-rl-sim60/transformer-learning-curves
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
export PYTHONPATH="$SNAPSHOT/src:$SNAPSHOT"
python -m json.tool "$STUDY/status.json"
OMP_NUM_THREADS=1 python "$STUDY/learning_progress.py"
tmux attach -t kaiser-learning-curves-16m-20260914
```

实时训练数据在 `run/jobs/VARIANT/seed_SEED/train/metrics.jsonl`，SDK 日志在同 job 的 `train.log`；
队列日志为 `queue.log`，监控异常为 `monitor.log`。结束后查看 `master-exit.json`、
`execution-receipt.json`、`run/summary.json` 和 `run/summary.csv`；需要刷新汇总时：

```bash
python -m transformer_rl.experiment_cli summarize --root "$STUDY/run"
```

本地仅新增本记录与小型 JSON 收据，交由主 agent 统一 commit/push。完整运行产物留在 Kaiser。

## 停止注记：提前运行归档，正式训练等待午夜

**本 study 已停止，以上“持续运行”描述仅保留为启动历史。** 用户最新要求正式训练不得早于
`2026-09-15T00:00:00+08:00`。这两项作为预跑归档，不作为额外 training seeds，
也不通过续训混入午夜正式实验。后续由主 agent 冻结最终 source/spec，在新 root 重新 plan，
从原 seeds 的初始化开始。本次停止操作未建立 timer、未启动或恢复任何训练。

停止收据：[learning-curves-stop.json](evidence/learning-curves-stop.json)。
以下时间均为 **2026-09-14 UTC+8**，来自实际信号收据、文件 mtime 和进程退出观察：

1. **19:33:46.688497**：核对 scheduler PID 673098、start_ticks=16970053、精确 CLI/root 后，
   通过 pidfd 发送 SIGSTOP，并确认其全部线程停止，阻止领取 queued jobs。
2. **19:33:46.688764 / .688771**：核对 workers 673104/673103 的 start_ticks=16970058、
   精确 worker argv 后，通过 pidfd 分别发送 SIGTERM。CLI 自行保存最后完整 optimizer update。
3. **19:33:47.494792 / .618791**：两个新 checkpoint 的文件 mtime；completion 随后写出
   `status=stopped / stop_reason=SIGTERM`。
4. **19:34:22.521240 / .521297**：确认两个 SDK worker 均已退出，读取 zombie 的原始 wait status=0，
   即实际 exit code=0；这是退出确认时间，不冒称精确进程死亡时间。
5. **19:34:22.521803**：直接 SIGKILL 仍冻结的 scheduler，**从未 SIGCONT**，因此没有重新领取任务。
   monitor 自行收尾并自动 summarize；**19:34:25.557942** master 退出。

仅上述三个经 PID/start-time/argv 核对的进程收到信号，未按名称 pkill 或操作其他用户会话。
monitor/timeout/master/tmux 随队列终止自然退出，没有将 worker SIGSTOP 挂到午夜占用资源。

### 最终保存与采样边界

| Job | 最后完整 update | optimizer / Adam steps | 已优化 rollout transitions | completion 累计采样 | 丢弃的未优化采样 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time_attention/seed_1011 | 460 | 3,680 | 7,536,640 | 7,539,200 | **2,560（5 vector steps）** |
| time_attention/seed_1022 | 465 | 3,720 | 7,618,560 | 7,618,560 | **0** |

checkpoint 均位于原 study 下：

- `run/jobs/time_attention/seed_1011/train/checkpoints/checkpoint_000460.pt`
  — SHA-256 `a0fb76e88a10606f92fc31c0eb91e99e9489aefbf3e96736c7cdca71bfc296b8`
- `run/jobs/time_attention/seed_1022/train/checkpoints/checkpoint_000465.pt`
  — SHA-256 `cbbff610e64e8409e717bf6d20ff5b17822cda5c3fa3909a3381d3943be93b0c`

两份均通过原冻结源码的 `load_checkpoint(device="cpu")`、weights_only/schema 3 校验。
update、Adam state step、source commit `e0f97a6719f19f20324323873dbe22ffd65a6e47`、dirty=false、
environment identity、action_clip=100、diagnostics=false 和文件 SHA 与原始产物一致；
两项所有中间 checkpoint 的 SHA 也重新核实通过。package source SHA 仍为
`65e6449703a75046e3fe26f8c79ce839e1bb8bad96bfc0038d1f0b75b6426a27`。

1011 的 checkpoint metadata 中 collected_transitions 包含终止前已经采集但尚未优化的 2,560 条，
不能将其算成进入模型更新的训练样本。该 rollout 被丢弃，checkpoint 没有保存环境/collector 状态，
因此这里仅保证最后完整 optimizer update 的模型和 Adam 状态保存，不宣称无损保存未优化 rollout。

### 队列、退出码与资源释放

- **2 stopped、16 never started、0 completed、0 evaluation**。其余 16 项 job 目录均未创建。
- scheduler 原始退出码 **-9（人为 SIGKILL）**；monitor/master **exit=1、status=failed**；
  summarize **exit=0** 只表示汇总执行成功，不代表训练实验成功。
  monitor 原始 `abort_reason=null` 保留不改，人为停止原因由独立 stop 收据补充。
  冻结 scheduler 后未生成正常的 job/result 收据，原汇总可能呈 missing；逐训练 completion 的 stopped
  与本停止收据是预跑保存状态的依据。
- **19:36:15 UTC+8 验证**：本批 master/timeout/monitor/scheduler/workers 六个 PID 均已不存在，
  指定 tmux session 不存在，本批 worker RSS=0。
- 停止前 **19:33:45** 全机 GPU=12,068 MiB、util=90%、WSL MemAvailable=6,359.6 MiB；
  释放后采样 GPU=**3,527 MiB**、util=24%、MemAvailable=**13,869.8 MiB**。
  GPU 数值含其他应用，不将剩余占用归于本批；本批进程已退出并释放资源。
- 原始 study、配置、日志、checkpoints、startup JSON 全部保留。
  `start-evidence.json` SHA 仍为 `aba53a5b755e90107a296d7109dc85aac06d6c016502cd726ecb8c1e1b5efa9d`。
  远端新增 `stop-request.json`、`stop-drain.json`、`stop-validation.json`；原 monitor 正常补写其终止收据与汇总。

当前查看归档应使用上述 stop 收据和 `train/completion.json`，不要根据旧的 start/status 字段判断仍在训练。
