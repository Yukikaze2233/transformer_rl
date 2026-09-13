# Kaiser 外部 Isaac Lab 研究任务

## 六变体正式工程 pilot（2026-09-14）

**六项 train + 独立 evaluate 全部 completed，session exit 0。** 每项 512 env、
20 updates × 32 rollout steps、327680 transitions；合计 120 次 PPO 更新、
456 次实际 optimizer.step、1966080 个训练 transitions。每项独立评估 8 env × 300 steps，
合计 14400 个评估 transitions。仅一个 training seed=11、evaluation seed=101。

精简可提交报告：[pilot-summary.json](evidence/pilot-summary.json)、
[pilot-summary.csv](evidence/pilot-summary.csv)。模型、原始日志和 checkpoint 留在 Kaiser。
JSON 包含冻结 spec/base config、六模型参数配置/参数量、checkpoint SHA、环境 identity、
所有物理指标的 mean/RMS/min/max/count，以及两次并发配置尝试的审计信息。

### 源快照与执行协议

- 有效目录：`/home/kaiser/robot-rl-sim60/transformer-comparison-pilot-20260914T0353`。
- 完整脏源归档：`/home/kaiser/robot-rl-sim60/transformer-pilot-complete-source.tar.gz`；
  SHA256 `21b7aeb1cba2591be87d094236ebac1f5e7c851cfddd7d52dcabd6fdbeb4d5a0`。
- 包源 SHA256：`afc83b7a6346b9dd724cb13217fa6f78bbf64297668a66bcf2cb654ac535adf4`。
- 有效 plan SHA256：`c1f5d5c65e56212f29957979b294f8f0cdd1ea7ef76af3ac876a54c2062fa57d`。
- 配置从 `configs/control.json` / `configs/comparison.json` 派生；epochs=2、num_minibatches=4，
  learning_rate=0.0001、target_kl=0.01、action_clip=100，六项保持相同环境与样本预算。
  `supervised_attention` 的 auxiliary_coef=0.1、indices=[25,26,27]，单列 supervision group。
- 训练使用 locomotion 原命令采样；评估固定 `[0,0,0.30]`，关闭 actor noise/root reset velocity。
  六项 train/evaluate 的 identity 全部匹配，包含 contract/source/asset/adapter/worker SHA。
- `evaluate-help.txt`、六份 `inspect-*.json`、`source-freeze.json` 在启动前生成。
  正式调度和验收为 `experiment_cli plan/run/summarize`；原始正式报告在 `plan/summary.json`、
  `plan/summary.csv`，每个 job 的 `result.json`、`train/completion.json`、`evaluation_101.json` 均保留。
- 每 job timeout=900s，train max_seconds=600；独立 tmux 外层 timeout=1200s，内部全局预算=1140s。
  所有 train/evaluate 通过 `examples._isaaclab_process` 完成产物写入和真实进程退出。

### 并发与资源实测

首先按用户授权尝试 3×512-env，独立目录 `transformer-comparison-pilot-20260914T0348`，
plan SHA `97d5eb79a37318f4ba9848d21f5e73e6d25528e16f239a69844ee681b5130f30`。
physics startup 阶段 WSL 可用内存降到 **588.6MiB**，监控在连续两次低于 1GiB 后
停止本次调度及其子进程；约 97.9s、尚无策略更新。GPU 总显存峰值 15390MiB，
瓶颈是 WSL 主机内存。该中止尝试保留且不计入六变体结果；没有修改已有用户进程或系统配置。

随后以**相同源码、相同 512-env 和训练/评估预算**新建 max_parallel=2 的上述有效 plan：

| 实测量 | 结果 |
|---|---:|
| 完整调度耗时（含训练/评估启动） | 627.174s |
| 实际峰值 worker / train worker 数 | 2 / 2 |
| 观测到的独立 train / evaluate 进程数 | 6 / 6 |
| 时间加权平均并发 worker 数 | 1.966 |
| GPU 总显存峰值 | 15370MiB |
| GPU 总利用率峰值 / 采样均值 | 98% / 68.78% |
| WSL 最低 available memory | 2543.6MiB |
| 总训练 transitions / 完整调度秒数 | 3134.83/s |

`resources.jsonl` 保存约每秒一次、共 600 次原始采样，包括 PID、argv、RSS、GPU、MemAvailable。
GPU 数字是含原 r3a 任务的全机值；WSL nvidia-smi 无单进程显存，报告中该字段明确为 null。
下表 RSS 是逐进程采样峰值，不是 GPU memory。

### 六任务训练收据

全部状态 completed、每项 20 updates / 327680 transitions。
train elapsed 包含 SDK/场景启动；collection throughput 只使用 rollout collection 时间之和。

| variant | group | train elapsed (s) | optimizer steps | collection transitions/s | RSS peak (MiB) |
|---|---|---:|---:|---:|---:|
| time_attention | architecture | 169.913 | 20 | 4672.18 | 4736.5 |
| index_attention | architecture | 165.233 | 20 | 4683.09 | 4729.2 |
| gated_attention | architecture | 167.412 | 116 | 4740.23 | 4766.6 |
| supervised_attention | supervision | 171.345 | 20 | 4561.88 | 4729.0 |
| history_mlp | architecture | 158.809 | 120 | 5001.97 | 4713.0 |
| history_gru | architecture | 162.347 | 160 | 4892.98 | 4691.8 |

统一 epochs/minibatches 不意味着实际梯度步数相等：KL early stop 导致明显差异，
尤其 time/index/supervised 每次 update 仅实际执行 1 个 optimizer step。
正式 summarize 的样本预算公平性检查均通过，但不能据此声称梯度计算预算相同。

### 正式评估 PRE-reset 指标

下表均为 2400 transitions 的 mean，保留启动瞬态；每项 terminated=0、truncated=0。

| variant | vx abs error (m/s) | wz abs error (rad/s) | height abs error (m) | planar speed (m/s) | non-wheel net force (N) |
|---|---:|---:|---:|---:|---:|
| time_attention | 0.080189 | 0.152212 | 0.090985 | 0.095197 | 52.780919 |
| index_attention | 0.043228 | 0.164883 | 0.080620 | 0.068579 | 46.495874 |
| gated_attention | 0.057815 | 0.093003 | 0.050221 | 0.080685 | 26.531330 |
| supervised_attention | 0.087356 | 0.189933 | 0.092677 | 0.104128 | 74.562354 |
| history_mlp | 0.038230 | 0.075257 | 0.035332 | 0.050818 | 36.215355 |
| history_gru | 0.032339 | 0.103598 | 0.051305 | 0.053413 | 37.347864 |

这是 **20 次更新、一个 seed 的工程/初期对照**，没有收敛或赢家结论。
高度误差和非轮净接触力均明显非零，termination=0 不等于姿态/接触通过。
net force 没有 ground-pair 身份；旧研究资产短杆归属、低位接触与硬件合同仍不能由此次训练证明。

结束时原 `r3a-train-35f06fa3504049f8924f9ccd1ed02b93`、`r3a-post-eval-waiter` 均保留，
所有本次 worker / tmux 已结束；GPU 回到 6080MiB，WSL available 10354MiB。
3 并发尝试后 swap 使用约 275MiB，2 并发有效 pilot 中未继续增加。
本轮未修改 adapter/core/CLI/experiments，也未提交或推送。

## 此前接入验收记录

`examples.isaaclab_task:make_env(model_config, environment_config, device)` 已实现；纯函数测试 2 项通过。

正式 pilot 之前，Kaiser 接入实测完成 **3 个成功训练进程、6 次真实 PPO 更新、16416 transitions**：
一个 2-env smoke，加上两个并行 256-env 容量短测。每进程均完成 2 updates，
每次 update 的 optimizer_steps=1（后续 minibatch 触发 KL early stop），最终 exit 0。
另有一个初次 shutdown 失败的训练，不计成功；一个故意无效合同的失败验收，exit 1。

真实 timeout final-state、command resampling 边界、加载 checkpoint 的 8-env × 200-step
确定性物理探针已验证。此前阶段未使用正式 evaluation schema；主 agent 随后补齐
`evaluation.py` 与下述 worker_module，已用于本页开头的正式 pilot。

已提供 optional backend worker `python -m examples._isaaclab_process train ...` / `evaluate ...`。
它调用原 CLI main，显式持有 App，env.close 只释放任务，CLI 写完产物后由 worker
以真实 exit code 调用 SDK fast shutdown。experiments execution schema
支持可选 `worker_module`（默认 `transformer_rl`；此 backend 设为
`examples._isaaclab_process`），用于 train/evaluate subprocess argv 的 `-m`。
这能保持 core 不依赖 Isaac，且不使用 atexit 伪造成功、不更改全局 Python executable。
当前 factory 遇到未受此 worker 管理的进程会在启动 App 前报错，避免已知的 shutdown 139。
worker 成功退出和 App 启动后故意传 v1 合同的失败退出，均已实测验证。
本次没有修改 core / CLI / experiments，没有提交或推送。

## 外部研究合同与时序

- task_root：`/home/kaiser/robot-rl-sim60/v40-live-snapshot03`，不覆盖旧源。
- contract：`contracts/own_v40_v2.json`，保留研究机械界、奖励、locomotion commands。
- actor：取 noisy policy125 最后 scalar25；顺序为 omega3、gravity3、command3、q4、qdot6、旧 applied6。
  重排为 proprio16 + command3 + **previous issued6**，追加 sensor age2、known2、policy_dt1。
- 两组 sensor ages 明确为当前仿真 fresh state 的 known zero；并非硬件事实。
- 独立 float64 单调 policy event 时钟，episode reset 不回退；0.01s policy / 0.005s physics，
  现有 joint-space feedback 每 physics substep 更新；额外 transport delay 为零。
- critic29 保留 clean25 + true body COM linear3 + height1。
  auxiliary_indices 25/26/27 分别是 body COM vx/vy/vz，单位 m/s。
- 实例级 rewards hook 在 reward 之后、reset 之前调用纯 build_observation/build_critic。
  command 为刚结束 transition 使用的 command；计时器已递减但尚未采样下一 command。
  实例级 reset hook 仅在 step 期间标记 timeout final 有效；startup/reset 不捕获。
- evaluation：`mode="evaluation"`，`fixed_command=[0,0,0.30]`，使用原 `set_evaluation_command`，
  原合同关闭 actor noise 和 root reset velocity。物理指标全部 PRE-reset。
- non_wheel_net_force 为 rigid-body net-force history peak，**没有 ground-pair 身份**。
- 每进程独立 USD cache；stdout 的 `ISAACLAB_TASK_METADATA` 包含研究源文件、合同、资产 manifest hash。

## 首次部署收据

目录：`/home/kaiser/robot-rl-sim60/transformer-comparison-smoke-20260914`。
脏源归档：`/home/kaiser/robot-rl-sim60/transformer-adapter-smoke-source.tar.gz`。
SHA256：`fee4eaff12f08b6798aa9a8b933243083f2d174e28fa82cd3ebcb35f9c099b55`。
该归档早于后续 adapter seed / 生命周期修复，不能冒称最终代码快照。

`train/checkpoints/checkpoint_000002.pt` SHA256：
`3aa39a554ec27605a534a43385ac7e047ddc4c21e0abe7d53ac38f0d5c004ce2`。
32 transitions，completion elapsed 22.13s；最终 `exit-code.txt=139`。

生命周期修复后的 `train-owned/completion.json`：32 transitions，11.426s，
`owned-exit-code.txt=0`，checkpoint SHA 与初次运行相同。
该机器 `fast_shutdown=False` 虽能让 close 返回，但 interpreter teardown 发生 segfault；
worker 使用 fast shutdown，并把关闭推迟到 CLI 写完产物之后，失败码也传给 SDK。

`final-probe-receipt.json` 与 `final-boundary-receipt.json` 是额外真实物理诊断。
仅第 0 行被置于原合同 timeout 边界，`truncated=[true,false]`、`final_valid=[true,false]`；
PRE-reset height=0.31944552m，返回 reset height=0.31999999m。
final critic 真速度与源 PRE-reset evaluation snapshot 逐张量相等；动作 echo 在 reset 行归零，
在非 reset 行保留 issued；固定 command、零 root reset velocity、无 RSL learner 导入均断言通过。
command 到期探针断言 final critic 保留旧 command、返回 observation 使用新 command，
每个 step 只调用一次原 `_get_observations`。

## 并行容量快照与真实训练

目录：`/home/kaiser/robot-rl-sim60/transformer-comparison-capacity-20260914`。
归档：`/home/kaiser/robot-rl-sim60/transformer-adapter-capacity-source.tar.gz`。
脏源 SHA256：`80533ac78b97fa190b517a950b07e4b51affa3f3fd475031cef5b192466948f1`。
这份归档包含容量验收使用的最终 adapter / worker 源码；研究源仍由独立 task_root 读取。

| 验收运行 | envs | updates / optimizer steps | transitions | completion elapsed | 纯 collection transitions/s | exit |
|---|---:|---:|---:|---:|---:|---:|
| train-11 | 256 | 2 / 2 | 8192 | 34.242s | 2181 | 0 |
| train-22 | 256 | 2 / 2 | 8192 | 34.697s | 2149 | 0 |

collection 吞吐使用 8192 / 两次 collection elapsed 之和，不包括模型更新和初始化；
总 elapsed 包括 scene startup。两次容量训练都是默认 Transformer，不是两个架构变体。

- seed11 checkpoint：`06a027952d98d26857e95a23ed7d865b2f856c98ad2f22e03d9be6a4527bfcbf`。
- seed22 checkpoint：`5e13d4124b608201d59a7b8f665f3b8b3bf3a4c205bfbf850a89239864620cc1`。
- 并行 startup 实测 GPU 总使用 6984MiB、GPU util 35%，每新增进程 RSS 约 2.83GB，
  WSL available 5984MiB。采样未覆盖稳态显存峰值，**不能报告单 job 显存峰值或据此升到 3 并发**。
- `resources.log` 的周期采样开始偏晚，只记录任务结束后 6080MiB；启动阶段值来自 SSH 实测输出。
- `invalid-contract/failure.json` + `invalid-exit.txt=1`：v1 合同被拒绝，无 completion；不会被 fast shutdown 改成成功。

## 加载 checkpoint 的独立物理探针

`policy-probe-receipt.json` / `policy-probe-exit.txt=0`：加载 seed11 update2 checkpoint，
8 env，200 vector steps，1600 transitions，seed101，deterministic mean，无 optimizer 更新；
固定 command `[0,0,0.30]`，14.179s（不含初始化），terminated=0、truncated=0。
它是等待官方 evaluation 入口期间的 adapter 验证脚本，**不冒充 experiments evaluation schema**。

| PRE-reset 指标 | mean | RMS | max |
|---|---:|---:|---:|
| vx_abs_error (m/s) | 0.047139 | 0.105759 | 0.424226 |
| wz_abs_error (rad/s) | 0.145516 | 0.279279 | 1.144554 |
| height_abs_error (m) | 0.048688 | 0.050329 | 0.054455 |
| planar_speed (m/s) | 0.060203 | 0.123701 | 0.498517 |
| non_wheel_net_force (N) | 61.049134 | 97.796322 | 574.643066 |

非轮刚体净接触力明显非零；既不能把 termination=0 解释为姿态/接触通过，
也不能把该力称为指定 ground-pair 力。这只是两次更新后的接口短测。

## 此前容量短测复现入口

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
ROOT=/home/kaiser/robot-rl-sim60/transformer-comparison-capacity-20260914
export PYTHONPATH="$ROOT/src:$ROOT"
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python -m examples._isaaclab_process train \
  --config "$ROOT/transformer-capacity.json" \
  --env-factory examples.isaaclab_task:make_env --device cuda:0 \
  --updates 2 --rollout-steps 16 --max-seconds 210 \
  --run-dir "$ROOT/new-unique-run" --seed 11
```

原执行脚本 `transformer-capacity.sh` 每个 seed 写 exit receipt；由独立 tmux 中的
`timeout --kill-after=15s 240s` 启动两个进程，未复用已有 run 目录。

正式 pilot 使用本页开头的新完整快照和 512-env 冻结规格；此前两次更新的容量收据
与正式六变体结果分别记录。有效快照中的 `prepare-transformer-pilot.py`、
`run-transformer-pilot.py`、`transformer-pilot-session.sh` 保留完整部署和监控命令。
规划/执行/验收入口为：

```bash
ROOT=/home/kaiser/robot-rl-sim60/transformer-comparison-pilot-20260914T0353
export PYTHONPATH="$ROOT/src:$ROOT"
python -m transformer_rl.experiment_cli plan --spec "$ROOT/pilot-spec.json" --root "$ROOT/new-plan"
timeout --kill-after=30s 1200s python -m transformer_rl.experiment_cli run --root "$ROOT/new-plan"
python -m transformer_rl.experiment_cli summarize --root "$ROOT/new-plan"
```

本次实际执行由独立 tmux 和 `run-transformer-pilot.py` 增加每秒资源采样与内存余量保护；
以上命令说明调度接口，不能覆盖已存在的 `plan` 或 job 目录。

已有 tmux `r3a-train-35f06fa3504049f8924f9ccd1ed02b93` 与 `r3a-post-eval-waiter` 保留。
初始 GPU 使用 6071MiB / 24564MiB。所有新测试使用独立 tmux 和有界 timeout。

这些短测只验证优化、环境和评估接口，不能证明收敛或实机可用，也不能据此排名架构。
