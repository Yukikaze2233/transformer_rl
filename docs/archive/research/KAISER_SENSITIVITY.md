# Kaiser Transformer 优化敏感性实测

> 本页保留首次实验及其真实失败事实。行为统计数值阻塞现已复现、修复，并在同一新源码下完成完整3-vs-3确认；最新结果见[修复后复验](KAISER_SENSITIVITY_RECHECK.md)，不要把本页旧失败改写成成功。

本页记录 2026-09-14 的有界真实训练。源码、训练 checkpoint、原始日志和资源采样留在 Kaiser；精简证据由 `docs/evidence/sensitivity-summary.json` / `.csv` 汇总。

**结论：完成七项筛查与冻结组合的新 seed 尝试，共 11 项训练完整成功、2 项被行为统计校验拒绝；候选 3/3、baseline 2/3，完整 3-vs-3 确认受阻。** 组合在三个新 seed 上都达到 100% optimizer-step 利用率，但物理姿态/净接触力仍不理想且分散大，不能宣布架构 winner、配方物理通过或已经收敛。现有五个完整 B checkpoint 的十次额外高度评估全部成功；全轮 11354112 training transitions（含失败采样）、3602 个实际梯度步、100800 evaluation transitions。所有本次 tmux/worker 已结束。

## 范围与冻结来源

- 独立快照：`/home/kaiser/robot-rl-sim60/transformer-sensitivity-20260914T071345Z`。
- 源归档：其父目录的 `transformer-sensitivity-source-20260914T071345Z.tar.gz`。
- 归档 SHA256：`feae56a3ac00362fc0d0ae8086eea7c17c364e7524ba65a630860818fab500bd`。
- 新包源 SHA256：`d91b0d7c450d4e3c6d0386b3368e8b350d8323e7cf4b46c1e3495407a39adadc`。
- 阶段 A plan SHA256：`6d76267590d1a0ebc9abdebdda088162a365f7250a7910bcd3f35fe1ac6282c0`。
- 阶段 B plan SHA256：`50643f16f1a33dbb1430fc49a10a8d5ff90a57dfa59cced598c5731063c29459`。
- task_root：`/home/kaiser/robot-rl-sim60/v40-live-snapshot03`，合同 `contracts/own_v40_v2.json`。
- 工厂 `examples.isaaclab_task:make_env`，worker `examples._isaaclab_process`；由 SDK shutdown 返回真实进程状态。
- runtime：`/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh`；Python 为 `env/bin/python`，`PYTHONPATH=快照/src:快照`。

本轮没有修改核心、模型、callback 或身份校验。A/B 使用同一新包源，baseline 是同版本复测；旧六架构 pilot 的不同预算/评估长度不混入本轮统计。

共同配置：time_attention Transformer、history=16、d_model=64、2 层、4 heads、FFN=128；critic=[256,128,64]；512 training env、rollout=32、epochs=2、minibatches=4、target_kl=0.01、action_clip=100、diagnostics=true。原 locomotion 训练命令，无额外 transport delay。训练软预算 1200s/job，train+eval timeout 1500s/job。

每次独立评估固定 `[0,0,h]`，8 env × 600 steps、deterministic mean，所有物理指标为 PRE-reset。**每环境只有 6s，含启动瞬态，不是长时站稳实验。** `height_abs_error` 是绝对误差，不能还原有符号高度轨迹；`non_wheel_net_force` 没有 ground-pair 身份，低位净接触力来源仍未知。

## 阶段 A：单训练 seed 的七项筛查

训练 seed=11，40 updates，即每完整项 655360 transitions；评估 seed=101、h=0.30。六项 train/checkpoint/evaluate 完整成功，一项真实失败；调度 exit=1，汇总 exit=0。由于缺 `std_01`，整个七项组的完整性比较不可用，不能把完整的六项伪装成七项成功。

下表 first KL 是 **update 1 首个 Adam step 后**的全 rollout KL，final KL 是 **update 40 结束**的全 rollout KL，两者不是同一次 update。完整首/末 update 的两种诊断及 mean/std 分解均保存在 JSON。

| 变体 | actual / planned steps | step 利用率 | early-stop updates | initial mean abs | first KL | final KL |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 40 / 320 | 12.50% | 100% | 0.403065 | 0.181067 | 0.026250 |
| LR=3e-5 | 296 / 320 | 92.50% | 12.5% | 0.403065 | 0.016307 | 0.006434 |
| LR=1e-5 | 320 / 320 | 100% | 0% | 0.403065 | 0.001812 | 0.007142 |
| headscale=0.1 | 319 / 320 | 99.69% | 2.5% | 0.040136 | 0.004092 | 0.007900 |
| headscale=0.01 | 317 / 320 | 99.06% | 5% | 0.004014 | 0.000929 | 0.010772 |
| std=0.1 | 0 / 320¹ | — | — | — | — | — |
| std=0.4 | 129 / 320 | 40.31% | 80% | 0.404207 | 0.038211 | 0.015499 |

¹ 320 是完整配置的名义预算；此项首轮采样后、任何 optimizer step 前退出，没有可读 optimization 行，不能编造诊断 planned 总量或 early-stop 比例。失败收据记录 16384 transitions、0 completed updates、真实 exit=1。报错：`old_log_prob mismatch before first optimizer step (max absolute error 1.23978e-05)`。这是行为分布一致性检查拒绝，不能据此认定低 std 的物理效果；本轮未修改 `rtol=1e-5, atol=1e-5` 或绕过检查。是否属于不同 batch shape 下的浮点差异需主 agent 独立排查。

首步 KL 几乎全部由均值项构成：baseline mean/std KL 为 0.181066610 / 6.001e-8，LR=3e-5 为 0.016307438 / 5.400e-9，headscale=0.1 为 0.004092061 / 5.999e-8。baseline 每次仅走一步，说明此次大 KL 主要在第一步发生，而不是多个 epoch 累积；相同 epochs/minibatches 并不构成相同实际梯度预算。

### A 的独立物理评估

以下为各项 4800 transitions 的均值，完整六项均 terminated=0、truncated=0；只有一个独立训练 seed，不给跨 seed 标准差。

| 变体 | height abs error (m) | vx abs error (m/s) | wz abs error (rad/s) | planar speed (m/s) | non-wheel net force (N) |
|---|---:|---:|---:|---:|---:|
| baseline | 0.117205 | 0.023292 | 0.052271 | 0.029989 | 66.0754 |
| LR=3e-5 | 0.067386 | 0.016415 | 0.056307 | 0.026887 | 44.5889 |
| LR=1e-5 | 0.065691 | 0.014598 | 0.046294 | 0.020443 | 42.4910 |
| headscale=0.1 | 0.039715 | 0.023027 | 0.035656 | 0.027358 | 112.1234 |
| headscale=0.01 | 0.077619 | 0.018122 | 0.029754 | 0.021325 | 53.3088 |
| std=0.4 | 0.125600 | 0.010998 | 0.050566 | 0.021029 | 91.3145 |

不能以低 planar speed/vx error 把低姿态解释为良好站立；也不能以未触发 termination 代替物理通过。headscale=0.1 的高度误差较低，但净接触力明显较高，没有全面优胜项。

## 配方选择与阶段 B 冻结

选择文件为快照内 `selection.json`，它在 B plan/训练前冻结，并引用保留的 `stage-a-selection-summary.json` SHA。**配方选择使用了 A 的 seed11 训练诊断以及 eval seed101、h=0.30 的物理数据**；它们属于 validation，不是确认集。

冻结组合 `lr3e5_head01`：LR=3e-5、mean_init_scale=0.1、initial_std=0.2。理由：

1. headscale=0.1 同时降低初始均值幅度、首步 KL，提高实际梯度利用率，并改善筛查高度误差。
2. 降低 LR 的两个单因素改善了高度与净接触力；1e-5 的物理收益相对 3e-5 较小，选择内部水平 3e-5 与已减敏感的 head 组合，避免仅追求最低 KL 或沿搜索边界扩张。
3. headscale=0.01 的高度误差反而更大；std=0.4 的高度/净接触力更差；std=0.1 无有效结果，保留 std=0.2。
4. 组合是否改善姿态/接触取舍是待检验假设，不能把两个单因素的收益直接相加。`sensitivity.factor_changes` 记录 LR 与 headscale 两项，`single_factor=false`。

B 固定 baseline 与此组合，各用新训练 seeds `[22,33,44]`，80 updates × 32 × 512，即每 seed 1310720 transitions、名义 640 optimizer steps。独立 eval seed=201、h=0.30；B 后同一 checkpoint 在 h=0.28/0.32 各独立运行相同协议，配置、checkpoint SHA、输出 SHA 与 exit 证据另行保留。确认后不根据 B 结果重新选择配方。

### B 的实际完成率与数值阻塞

**候选 3/3 完整完成；baseline 2/3 完整完成。** baseline seed44 在第 52 次更新的首个 optimizer step 前触发同一检查，最大 log-prob 差 `1.2517e-05`，真实 exit=1；failure.json 记录 51 completed updates、851968 collected transitions。没有 update80 checkpoint，不能对其做完整预算的高度复评，也没有补种、替换 seed 或继续训练。

因此 B 的 `comparison_available=false`：其余完整 seed 的样本预算一致，但无法构成完整 3-vs-3 确认。该失败发生在 std=0.2 的 baseline，说明先前 std=0.1 的拒绝不能简单归因于“低 std 不适合”。应先解决行为统计一致性检查的复现与根因，再重新冻结公平确认实验。

| 配方 / seed | completed updates | actual / readable planned steps | step 利用率 | early-stop updates | initial mean abs | first KL | final KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline / 22 | 80 | 92 / 640 | 14.38% | 100% | 0.516111 | 0.089819 | 0.023177 |
| baseline / 33 | 80 | 102 / 640 | 15.94% | 98.75% | 0.442314 | 0.085958 | 0.005848 |
| baseline / 44，失败 | 51 | 67 / 408 | 16.42% | 96.08% | 0.514111 | 0.091317 | 0.039579¹ |
| LR3e-5 + head0.1 / 22 | 80 | 640 / 640 | 100% | 0% | 0.052608 | 0.000338 | 0.004155 |
| LR3e-5 + head0.1 / 33 | 80 | 640 / 640 | 100% | 0% | 0.046669 | 0.000237 | 0.003008 |
| LR3e-5 + head0.1 / 44 | 80 | 640 / 640 | 100% | 0% | 0.050653 | 0.000328 | 0.004611 |

¹ 失败项 final KL 是 update51 的最后可读诊断，不能与 update80 作为相同训练终点。该项完整预算仍为 640 planned steps，表中 408 仅对应 51 个已记录 update。其他行 first/final 列仍分别表示 update1 首步与 update80 末尾。

候选三 seed 的首步 mean KL 分别为 0.000338154、0.000237164、0.000327594，std KL 为 5.409e-9、5.418e-9、5.382e-9；baseline 三 seed 的首步 std KL 均约 6.003e-8。**组合降低了首步均值敏感性，且三个新 seed 都充分使用了梯度预算**，这一优化现象有真实重复证据；它不自动转化为任务物理成功。

### B 在 h=0.30 的逐训练 seed 物理指标

| 配方 / seed | height error (m) | vx error (m/s) | wz error (rad/s) | planar speed (m/s) | net force (N) |
|---|---:|---:|---:|---:|---:|
| baseline / 22 | 0.098908 | 0.016902 | 0.019871 | 0.021594 | 107.811 |
| baseline / 33 | 0.061287 | 0.163684 | 0.951005 | 0.180950 | 784.500 |
| baseline / 44 | 缺失 | 缺失 | 缺失 | 缺失 | 缺失 |
| candidate / 22 | 0.062736 | 0.016158 | 0.048750 | 0.030923 | 779.584 |
| candidate / 33 | 0.068203 | 0.011225 | 0.041295 | 0.024688 | 509.757 |
| candidate / 44 | 0.086381 | 0.010440 | 0.037416 | 0.016957 | 31.120 |

候选没有一致的物理支配关系：seed22 的高度误差改善，但净接触力由 107.8N 升至 779.6N；seed33 的速度/yaw 明显改善，但高度误差略增。候选 seed44 净接触力较低，同时高度误差在其三个 seed 中最大。均值收益不能掩盖这一取舍。

### B checkpoint 的三高度汇总

额外高度计划 SHA256：`312e6fe821996a2fd9cb7e7b09ae7cd506908e5f854d892eba4c4d83f2c5ded9`。h=0.28/0.32 使用同一 eval seed201，五个完整 checkpoint 各两次，共 10 次真实 exit=0；请求的另外两次因 baseline seed44 无完整 checkpoint 而不可用。没有使用较早 checkpoint 补位。

表中为**跨独立训练 seed 的均值 ± 样本标准差（ddof=1）**；baseline 只含可用 seeds22/33，candidate 含22/33/44，两个集合不相等。这是描述性汇总，不是完整配对检验；8 个 env、600 个 step、3 个高度均不增加独立训练 seed 数。

| 配方 | h (m) | n | height error (m) | vx error (m/s) | wz error (rad/s) | planar speed (m/s) | net force (N) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline，可用子集 | 0.28 | 2 | 0.06022 ± 0.02673 | 0.09086 ± 0.10886 | 0.50014 ± 0.68848 | 0.10106 ± 0.11828 | 445.49 ± 476.08 |
| baseline，可用子集 | 0.30 | 2 | 0.08010 ± 0.02660 | 0.09029 ± 0.10379 | 0.48544 ± 0.65841 | 0.10127 ± 0.11268 | 446.16 ± 478.49 |
| baseline，可用子集 | 0.32 | 2 | 0.10081 ± 0.02547 | 0.09697 ± 0.11215 | 0.53843 ± 0.72975 | 0.11160 ± 0.12621 | 416.66 ± 431.75 |
| candidate | 0.28 | 3 | 0.05285 ± 0.01247 | 0.01272 ± 0.00300 | 0.04229 ± 0.00612 | 0.02423 ± 0.00688 | 437.17 ± 376.88 |
| candidate | 0.30 | 3 | 0.07244 ± 0.01238 | 0.01261 ± 0.00310 | 0.04249 ± 0.00576 | 0.02419 ± 0.00700 | 440.15 ± 379.06 |
| candidate | 0.32 | 3 | 0.09229 ± 0.01227 | 0.01289 ± 0.00319 | 0.04267 ± 0.00772 | 0.02445 ± 0.00705 | 444.40 ± 381.47 |

三高度所有完整评估的 terminated/truncated 都为 0。候选的高度绝对误差随目标高度近乎同步增加，速度指标却变化不大：与良好高度命令跟踪不相符，需更长轨迹/有符号高度进一步核查，不能把 h=0.28 较小误差当作解决了高度控制。

JSON 的 `paired_complete_subset` 另列共同完整 seeds22/33 的逐 seed 差值及均值/分散。h=0.30 的 candidate−baseline 为：height error **−0.01463 ± 0.03047 m**，net force **+198.51 ± 669.29 N**；h=0.28/0.32 的 net force 差也分别为 +194.79/+234.52 N。不能把含低净力 seed44 的三 seed candidate 均值与缺44的 baseline 均值简单相减来声称净力改善。

## 资源与验收协议

所有长进程由新 tmux + timeout 执行，max_parallel=2。监控约每秒记录 GPU 全机 memory/utilization、各本次 worker PID/PGID/RSS、MemAvailable 和 swap；一旦 MemAvailable 低于 1 GiB，只停止本次调度和所属 worker 进程组。A/B 各外层 timeout=3000s、内部预算=2900s；全阶段共享 7100s 的 deadline。

A 启动时另有 `a-evaluation-recovery-070110` 用户会话，保留其进程；GPU 指标包含其他进程/Windows 应用，不能归因成本轮独占显存。A 实测耗时 904.158s、最大并发 2、GPU 峰值 17963 MiB、util 峰值/采样均值 98%/57.71%、WSL available 最低 3026.27 MiB；没有触发资源保护。完整训练 transitions=3932160，另有失败项采样 16384；六项实际 optimizer steps 合计 1421/1920。

| 阶段 | elapsed (s) | peak workers | GPU peak (MiB) | GPU util peak / sample mean | min MemAvailable (MiB) | session exit |
|---|---:|---:|---:|---:|---:|---:|
| A | 904.158 | 2 | 17963 | 98% / 57.71% | 3026.27 | 1，单项数值失败 |
| B | 1027.310 | 2 | 14342 | 97% / 60.53% | 6255.31 | 1，单项数值失败 |
| 高度复评 | 263.950 | 2 | 11936 | 93% / 73.52% | 7061.60 | 0，可用 checkpoint 全完成 |

三个调度的实际运行时间合计 **2195.42s（36.59min）**，不含阶段间人工分析/计划时间；全阶段在共享 7100s deadline 内结束，资源保护未触发。A/B 共保存 1852 次资源采样，高度阶段另 253 次。B 完整训练 6553600 transitions，失败项另采样 851968；可读日志共 2181/3608 个 actual/planned optimizer steps，其中失败 seed 的 51 个 update 仅计入诊断，不计入完整物理聚合。

标准 `experiment_cli run/summarize` 验证冻结源/plan/spec/config、真实进程退出、完成更新数、checkpoint SHA 与独立 evaluation JSON。stock runner 非零 returncode 必须报错，成功 job 的 train/evaluate exit=0 由该受检执行路径重建，原 CLI 未单独写逐子进程 exit 文件；session exit、failed job 的明确 returncode、result.json 与原始日志保留。高度扩展使用同一 evaluate CLI 和原 `_process`/`_report` 校验，不放宽训练/评估 identity。

## 吞吐与下一阶段样本量预算

完整训练 transitions / 完整调度秒数：A **4348.98/s**，B **6379.38/s**。分母包含初始化、评估及失败任务开销，分子不包含失败任务采样。这是本轮作业级实测，不是独占 GPU 稳态吞吐。A/B 的外部进程竞争、cache/startup 与并发重叠不同，不能把吞吐差解释为某配方固有性能收益。

| B 完整任务 | train elapsed (s) | collection transitions/s | train end-to-end transitions/s |
|---|---:|---:|---:|
| baseline / 22 | 325.250 | 6016.23 | 4029.88 |
| baseline / 33 | 324.361 | 6038.58 | 4040.93 |
| candidate / 22 | 285.448 | 7689.22 | 4591.80 |
| candidate / 33 | 268.569 | 8468.37 | 4880.39 |
| candidate / 44 | 279.625 | 7992.53 | 4687.43 |

纯 collection 不含 optimizer/diagnostics；end-to-end train 包含场景启动和优化。所有数字为 diagnostics=true 实测；不能当作关闭诊断后的精确速度。

后续预算与 [TRAINING_EVALUATION_PLAN.md](TRAINING_EVALUATION_PLAN.md) 的 16M/64M/128M 累计台阶对齐。以下只是用于审批/排程的**未执行预算**，先解决本轮数值阻塞并冻结新源/新 plan：

| 每配置每训练 seed 的目标 | 512×32 下向上取整 updates | 实际计划 transitions/seed | 名义 optimizer steps/seed（2×4） | 2 配置 × 5 新 seeds 的累计 transitions | 按本轮作业级吞吐粗估累计时间 |
|---|---:|---:|---:|---:|---:|
| 16M | 977 | 16007168 | 7816 | 160071680 | 7.0–10.2h |
| 64M | 3907 | 64012288 | 31256 | 640122880 | 27.9–40.9h |
| 128M | 7813 | 128008192 | 62504 | 1280081920 | 55.7–81.8h |

各行是累计台阶，不应相加；增加训练 seed 数或架构数时按完整 run 数扩展，epochs/minibatches 改变则需重新列 optimizer 预算。实际 steps 因 KL gate 可低于名义值，不通过额外 rollout 追平。启动/评估摊销、diagnostics、仿真负载和失败重跑都会改变墙钟，表中不是完成时限承诺。

网络选择至少先安排每候选 **5 个新的独立训练 seeds**，相同 seed 集、任务/样本预算和 checkpoint 选择规则；若失败率或配对差值区间仍宽，按预先声明的规则扩展至 8–10 或保留证据不足，不能把“5”当作充分性保证。本轮 candidate 只有 3 个 seed，baseline 只有 2 个完整 seed，且都只有 1.31M transitions，远不足以排除慢热、长期退化或姿态局部解。高低命令跟踪、接触来源和更长评估协议需在下一阶段独立固定；原任务 20s timeout 后 reset，长累计评估也不等于连续不重置站立。

本轮只比较同一个 Transformer 的优化配方。后续 MLP/GRU/Transformer 的架构选择必须另行采用公平的优化搜索/训练预算及 held-out 数据；本页不能给出架构 winner。核心数值检查的排错与论文/正式预算文档由主 agent 接续，本轮没有启动这些长训，也没有 commit/push。
