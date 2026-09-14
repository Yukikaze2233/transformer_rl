# Kaiser 敏感性：行为 guard 修复后的同源码复验

## 结论与完成状态

**修复版原 payload 离线 GPU 验收通过；新源码下7/7项训练、19/19次独立评估全部完成，真实 worker exits 均为0。** A 的 std0.1 完成40次更新；B 的 baseline/candidate 各3/3完成80次更新及三高度评估，完整配对比较恢复可用。

候选三 seed 均达到640/640 optimizer steps，baseline分别为92/640、102/640、96/640。**数值阻塞修复已通过此次有界验收，优化利用率优势得到同协议重复；物理表现仍有明显取舍，不能宣布配方全面胜出、架构 winner 或已经收敛。**

- 新增训练 transitions：**8519680**；实际 optimizer steps：**2250/4160**；评估 transitions：**91200**。
- A 补验1 train + 1 eval；B复验6 train + 6次h=.30 eval + 12次h=.28/.32 eval。
- 本次没有新训练/评估错误，也未触发资源保护。所有本次 tmux/worker 已结束。
- 主数据：[evidence/sensitivity-recheck.json](evidence/sensitivity-recheck.json)；长表：[evidence/sensitivity-recheck.csv](evidence/sensitivity-recheck.csv)，**LF，585条数据行**。

这是 **新source补验与同seed复验**。旧 [KAISER_SENSITIVITY.md](KAISER_SENSITIVITY.md) 和 `sensitivity-summary.json/.csv` 保留原来的11项完成、2项失败事实；A新单项不与旧六项拼接成“旧A通过”，B新六项不混入旧source结果。同seed跨源码重复不增加独立训练seed数，B每配方仍只有3个seed。

## 新源码身份与预先冻结的协议

独立快照：`/home/kaiser/robot-rl-sim60/transformer-sensitivity-recheck-20260914T0920`。

| 身份 | SHA256 |
|---|---|
| 源归档 `transformer-sensitivity-recheck-20260914T0920-source.tar.gz` | `3455c1fc7c33ac02473a77013e804089a37f442ee0d6bc28b8bfc8bc8e0ae435` |
| 新 package source | `f182eaeb41a9f4a6bed0c6aff6aca9be01f9364cc4c707b7848a071f4516c209` |
| 旧 package source | `d91b0d7c450d4e3c6d0386b3368e8b350d8323e7cf4b46c1e3495407a39adadc` |
| 新 `ppo.py` | `8994d71c41ebb51feeb568d9a8830c36ca73829a89818836c7dbde88c6e238a2` |
| `source-freeze.json` 文件 | `a95bed526495a9e832f83a43a885453d24cf20e904c2b1cd884ef0c294e7b23f` |
| 新 A 单项 plan | `6b269a7f6c8af5f5e91e9e4abdd9933b37b230941e43de36e830fc253a0572df` |
| 新 B 六项 plan | `cfd7e503d88d4bbafeb23ffa4c68a1af1e9ff96baadd81e14c3f2965a6d824e7` |
| B 额外高度 plan | `f0d381314894ba12d6bed2ef26e528317c143bcb7c5cc96f2ec6330fbed0870c` |

部署的是完整 `src` package、`examples`、配置与测试的工作树快照；没有覆盖旧实验。在 `src/examples/configs` 的旧冻结清单范围内，**内容变化仅为主agent已经修复的 `src/transformer_rl/ppo.py`**。本次没有进一步修改核心/例子或容差。每阶段以及最终汇总再次核对实际import的包源、计划和配置；额外高度评估引用同一个新B的update80 checkpoint。

runtime：`/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh`，Python为该安装的`env/bin/python`，`PYTHONPATH=新快照/src:新快照`。工厂/SDK owner仍为`examples.isaaclab_task:make_env` / `examples._isaaclab_process`，task_root仍是`/home/kaiser/robot-rl-sim60/v40-live-snapshot03`、合同`contracts/own_v40_v2.json`。

共同网络和训练设置：time-attention Transformer，history16、d_model64、2层、4 heads、FFN128，critic=[256,128,64]；512 env、rollout32、epochs2、minibatches4、target_kl=.01、action_clip100、diagnostics=true，无额外transport delay。

| 阶段 | 配方 | train seeds | updates | eval seed / h |
|---|---|---|---:|---|
| A补验 | LR1e-4、headscale1、std.1 | 11 | 40 | 101 / .30 |
| B baseline | LR1e-4、headscale1、std.2 | 22/33/44 | 80 | 201 / .28/.30/.32 |
| B candidate `lr3e5_head01` | LR3e-5、headscale.1、std.2 | 22/33/44 | 80 | 201 / .28/.30/.32 |

每次评估均为8 env × 600 steps、deterministic mean、固定`[0,0,h]`，指标均为PRE-reset。A/B训练计划在新训练开始前同时冻结；候选直接沿用旧`selection.json`，其文件SHA为`b71b81e7985155ecfc8583b5a8bc8579ed8d8c7765661f1823d594112d2a0980`。本次不根据A补验或新B结果重选配方。三高度协议亦在开始时声明；B完成后冻结带实际checkpoint SHA的12项高度清单。

## 原 payload 离线修复验收

使用原`payload.pt`（SHA `9add976532474297239e560dd42c8f6d92434641fdbcbd8466ed28c2b8155d2d`），**零采样、零optimizer steps**。RTX4090、Torch`2.11.0+cu128`、CUDA12.8；沿用FP32 moments/action、FP64 timestamps和原backend设置：matmul TF32=false、float32 matmul precision=highest、CUDA autocast=false。

八项预期全部吻合：

1. 新guard对原16384条、4×4096扫描通过，结束后再次扫描仍通过。
2. 直接加载冻结旧guard，仍拒绝并复现`max absolute error 1.23978e-05`。
3. 分别篡改最后endpoint的old_logp、old_mean、old_std、raw_action，均拒绝；不是只检查最初chunk。
4. current moments正确而current logp错误，仍以`current_log_prob mismatch`拒绝。

所有比较保持`rtol=1e-5, atol=1e-5`。原batch/history、model state、Python/Torch CPU/CUDA RNG和原payload SHA均未变；Adam state为空。完整正反例、误差与源SHA保存在JSON的`offline_acceptance`，更详细的契约说明见[BEHAVIOR_PRECISION.md](BEHAVIOR_PRECISION.md#修复版-kaiser-原-payload-离线-gpu-验收)。

## 优化结果：逐训练 seed

**first KL是update1首个Adam step之后的全rollout解析KL；final KL是最终update结束的全rollout KL。** 二者不处于同一训练时刻。不是把`optimization.kl`这一minibatch聚合值当作首步全rollout诊断。

| 阶段 / 配方 / seed | completed updates | actual / planned steps | step利用率 | early-stop updates | initial mean abs | first KL | final KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| A std.1 / 11 | 40 | 40 / 320 | 12.50% | 40/40 | .402839 | .801794709 | .078014232 |
| B baseline / 22 | 80 | 92 / 640 | 14.38% | 80/80 | .516111 | .089818836 | .023176911 |
| B baseline / 33 | 80 | 102 / 640 | 15.94% | 79/80 | .442314 | .085957815 | .005848249 |
| B baseline / 44 | 80 | 96 / 640 | 15.00% | 78/80 | .514111 | .091316878 | .014039080 |
| B candidate / 22 | 80 | 640 / 640 | 100% | 0/80 | .052608 | .000338159 | .004155263 |
| B candidate / 33 | 80 | 640 / 640 | 100% | 0/80 | .046669 | .000237170 | .003008342 |
| B candidate / 44 | 80 | 640 / 640 | 100% | 0/80 | .050653 | .000327599 | .004610802 |

A首步KL中，mean项为`.80179464946`，std项为`5.98915e-8`。std.1的原数值阻塞解除后仍每次只走一个Adam step；这不是低std配方优化稳定或物理有效的证据。

B baseline首步std KL均约`6.00289e-8`；candidate seeds22/33/44的首步mean KL分别为`.000338153541` / `.000237164275` / `.000327593815`，std项约`5.4e-9`。首步差主要由均值项主导，既定LR+headscale组合在三个seed上都显著减小首步分布移动，并完成全部名义梯度步。实际梯度预算依旧不同，不能因为epochs/minibatches相同就认为optimizer steps相同。

### 与历史同 seed 的只读核对

旧B中五个完整训练的各80条optimization记录，与本次对应新训练的记录逐值相等；collection reward mean也逐值相等。baseline44前51条共同可读记录同样相等，随后本次继续完成update52–80。合计比较451条共同记录。

这为原先可执行区段的优化行为保持一致提供了额外实测支持；比较的是日志字段，不声称仅凭日志证明所有模型中间tensor/RNG轨迹相同。旧记录只作只读对照，**没有用于新source的任何完成收据、checkpoint或评估**。

## h=.30 的逐 seed 物理结果

每行4800 evaluation transitions。所有19次完整评估的terminated和truncated计数均为0。

| 阶段 / 配方 / seed | height abs error (m) | vx abs error (m/s) | wz abs error (rad/s) | planar speed (m/s) | non-wheel net force (N) |
|---|---:|---:|---:|---:|---:|
| A std.1 / 11 | .107059 | .222803 | .252581 | .233224 | 54.027 |
| B baseline / 22 | .098908 | .016902 | .019871 | .021594 | 107.811 |
| B baseline / 33 | .061287 | .163684 | .951005 | .180950 | 784.500 |
| B baseline / 44 | .092670 | .011452 | .016369 | .012946 | 45.105 |
| B candidate / 22 | .062736 | .016158 | .048750 | .030923 | 779.584 |
| B candidate / 33 | .068203 | .011225 | .041295 | .024688 | 509.757 |
| B candidate / 44 | .086381 | .010440 | .037416 | .016957 | 31.120 |

- seed22：candidate高度误差减小约.03617m，但净接触力增加约671.77N，yaw误差也更大。
- seed33：candidate速度/yaw误差和净力明显减小，但高度误差增加约.00692m。
- seed44：candidate高度误差和净力较小，但yaw误差与planar speed较大。

因此候选没有逐seed、逐指标的一致支配关系。A std.1的速度误差明显，不应将本次train/eval正常退出解读成任务控制成功。

## B 的额外高度：逐 seed 原值

下表补足各checkpoint的h=.28/.32；h=.30见上表。三高度使用同一update80 checkpoint及相同独立eval seed201，没有重新训练或选checkpoint。

| 配方 / seed | h (m) | height error (m) | vx error (m/s) | wz error (rad/s) | planar speed (m/s) | net force (N) |
|---|---:|---:|---:|---:|---:|---:|
| baseline / 22 | .28 | .079127 | .013887 | .013314 | .017424 | 108.850 |
| baseline / 22 | .32 | .118823 | .017670 | .022416 | .022356 | 111.369 |
| baseline / 33 | .28 | .041322 | .167839 | .986973 | .184702 | 782.135 |
| baseline / 33 | .32 | .082805 | .176272 | 1.054444 | .200841 | 721.960 |
| baseline / 44 | .28 | .073740 | .009482 | .016052 | .010884 | 47.732 |
| baseline / 44 | .32 | .111934 | .011372 | .015303 | .014119 | 44.485 |
| candidate / 22 | .28 | .043018 | .016163 | .048920 | .030877 | 775.443 |
| candidate / 22 | .32 | .082721 | .016557 | .051204 | .031309 | 782.459 |
| candidate / 33 | .28 | .048644 | .011292 | .041080 | .024681 | 505.120 |
| candidate / 33 | .32 | .088028 | .011381 | .040609 | .024800 | 519.913 |
| candidate / 44 | .28 | .066875 | .010690 | .036862 | .017144 | 30.933 |
| candidate / 44 | .32 | .106121 | .010730 | .036188 | .017227 | 30.822 |

### 完整三对 seed 的均值与配对差

下表均为跨3个训练seed的均值±样本标准差（ddof=1）。3个高度、8个env及600个step不增加独立训练seed数。

| 配方 | h | height error (m) | vx error (m/s) | wz error (rad/s) | net force (N) |
|---|---:|---:|---:|---:|---:|
| baseline | .28 | .06473 ± .02045 | .06374 ± .09018 | .33878 ± .56135 | 312.91 ± 407.51 |
| baseline | .30 | .08429 ± .02016 | .06401 ± .08636 | .32908 ± .53860 | 312.47 ± 409.99 |
| baseline | .32 | .10452 ± .01912 | .06844 ± .09344 | .36405 ± .59791 | 292.60 ± 373.33 |
| candidate | .28 | .05285 ± .01247 | .01272 ± .00300 | .04229 ± .00612 | 437.17 ± 376.88 |
| candidate | .30 | .07244 ± .01238 | .01261 ± .00310 | .04249 ± .00576 | 440.15 ± 379.06 |
| candidate | .32 | .09229 ± .01227 | .01289 ± .00319 | .04267 ± .00772 | 444.40 ± 381.47 |

配对方向为candidate−baseline，以下标准差来自**三个逐seed差值**，不是两组标准差简单相减：

| h | height error差 (m) | vx error差 (m/s) | wz error差 (rad/s) | net force差 (N) |
|---|---:|---:|---:|---:|
| .28 | −.01188 ± .02215 | −.05102 ± .09139 | −.29649 ± .56245 | +124.26 ± 487.36 |
| .30 | −.01185 ± .02208 | −.05140 ± .08752 | −.28660 ± .53965 | +127.68 ± 488.90 |
| .32 | −.01223 ± .02140 | −.05555 ± .09469 | −.32139 ± .59969 | +151.79 ± 459.48 |

完整配对后，候选平均高度和速度误差较小，但净力均值更大且差值分散很宽；速度均值改善主要由baseline seed33的大误差贡献。旧source缺baseline44时的不等样本均值不能替代此表。

两配方高度绝对误差都随目标h从.28增到.32近乎同步增加；候选约.05285→.09229m，而速度指标基本不变，与良好高度命令跟踪不符。每环境只有6秒且包含启动瞬态；无termination也不能证明长时站稳。净力无ground-pair身份，不能据此确认接触来源。此轮完成的是数值修复的公平有界确认，未建立长训收敛或部署结论。

## 退出证据、资源与验证

所有训练/评估均在新tmux会话的有界调度下运行；stock `experiments._process` 启动并回收真实SDK worker，非零returncode会抛错。独立runner只追加`*.process.json`收据，不修改原执行/优化实现；SDK仍由`examples._isaaclab_process`显式owner关闭。

每个新训练目录同时要求：train process exit0、completed更新数匹配、32×512 transitions/update匹配、最后checkpoint更新数/SHA匹配。每次评估要求：真实process exit0、checkpoint SHA/update、factory环境配置、环境provenance identity、seed、高度、steps600、env8、transitions4800、deterministic policy和action clip均匹配。最终汇总再次验证冻结包源/例子/配置；没有仅凭日志出现completion就宣告成功。

| 阶段 | elapsed (s) | max workers | GPU全机峰值 (MiB) | min WSL available (MiB) | 资源采样数 | exit |
|---|---:|---:|---:|---:|---:|---:|
| 离线验收 | 4.149 | 0¹ | 4806 | 13076.23 | 4 | 0 |
| A单项补验+主评估 | 198.510 | 1 | 8229 | 10250.29 | 191 | 0 |
| B六项复验+主评估 | 1206.702 | 2 | 12491 | 6359.57 | 1157 | 0 |
| B额外高度12项 | 340.152 | 2 | 9986 | 7035.93 | 326 | 0 |

¹ worker计数仅统计SDK worker；离线验收是一个独立GPU Python进程，没有SDK环境。

A/B/高度作业执行时间合计**1745.363s（29.09min）**，不含本地准备/分析/收尾；加离线验收约29.16min。共享deadline3500s，外层timeout3500s；训练软预算1200s/job、train+主eval期限1500s/job，额外高度240s/job。max_parallel=2，约每秒记录本次worker PID/PGID/RSS、MemAvailable及全机GPU使用；若available<1GiB只停止本次任务所属进程组。本次全程最低6359.57MiB，未触发。

训练worker采样RSS峰值约4717–4739MiB/进程，逐项保存在JSON；RSS含共享页，不能简单累加为独占占用。GPU数值包含原有Windows应用，不能作为本次独占显存；没有清理用户进程。

本地交付验证通过：八项离线预期、7项训练、19项评估、完整配对seed、共同source、预算计数、checkpoint与协议一致性、资源余量和CSV LF。脚本语法检查与Markdown diff检查通过。本次使用真实GPU/SDK验收；主agent此前的95项单元测试记录见行为精度文档，未将它们冒充本轮重新运行的结果。

原始日志、每worker退出收据、checkpoint、评估JSON、资源日志和完整冻结计划均留在新快照。源码/例子未在执行期间修改，原失败证据未改写。没有commit/push，没有启动16M或多日长训。
