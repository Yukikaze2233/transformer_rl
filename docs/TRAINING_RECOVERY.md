# 训练结果回收与本地验收

## 最新回收状态

首轮七变体已完成并于2026-09-18完整回收：21个最终模型、147份评估，
852份文件逐项哈希通过，21个最终模型严格CPU加载通过。
详见 [首轮完成与完整回收](KAISER_ARCHITECTURE_COMPLETE.md)。
下面保留首次暂停前快照的历史分析；其中“尚缺补传”和原loader失败不代表最新状态。

## 首次暂停回收的历史结论

本批训练已按用户要求暂停。**本地已回收 12 个完整训练 job、它们的 84 份独立场景评估，以及 2 个未完成 job 的中间 checkpoint。** 重新核对归档与全部 729 个文件，大小和 SHA-256 均匹配。

物理质量方面，**当前没有足够证据确认一个合格的多高度站立／运动候选**。四类完整 Transformer 的高度普遍偏低，前后速度跟踪误差接近指令本身的 0.5 m/s；高度抖动很低，经常只是停在错误姿态。不能由这些结果推出“Transformer 最优”，也不能比较尚未训练的 MLP/GRU。

本地 checkpoint 验收有一个必须保留的失败：14 份 checkpoint 均可用 `weights_only=True` 在 CPU 反序列化，模型 tensor 与 Adam schema、有限值、来源、配置和更新数检查通过；但**原严格 loader 为 0/14 通过**，原因是固定 `actor.time_frequencies` 缓冲区重建的两个 float32 元素不逐位相等，详见下文。不能把结构验收写成“可直接加载运行”。

## 文件位置与暂停边界

仓库：`/home/yukikaze/Documents/workspace/transformer_rl`。

- 原始回收包及收据：`artifacts/recovered/architecture-16m-20260915T0540/20260915T075822Z-8f22/`。
- 训练与评估：上述目录下的 `extracted/study/run/jobs/`。
- 冻结计划／配置：`extracted/study/run/plan.json`、`extracted/study/run/configs/`。
- 源码、任务、资产和运行环境证据：`extracted/{source,task,runtime,collection,collection-tools}/`。
- 可复核分析工具：[`tools/analyze_recovered_study.py`](../tools/analyze_recovered_study.py)。
- 小型证据：[`evidence/training-recovery.json`](evidence/training-recovery.json) 保存验证结果、14 个 checkpoint SHA、全部失败和 28 个架构×场景汇总；[`evidence/training-recovery.csv`](evidence/training-recovery.csv) 保存 84 行训练 seed×场景原始统计派生值，使用 LF 换行。

| 时间／证据 | 含义 |
|---|---|
| 2026-09-15 07:58:13.487549–07:58:20.347381 UTC，即北京时间 15:58 | 本地包的快照窗口；逐文件固定长度前缀，并非全局原子快照 |
| 快照内 `12 completed / 2 running / 7 queued` | 只描述暂停前的历史状态 |
| 北京时间 16:02:56，主 agent 的暂停交接 | execution receipt：`abort_reason=signal_SIGTERM`；12 完成、2 人工停止、7 从未启动，本批 PID 全部退出 |

暂停后的事实来自主 agent 已核验的收据交接，本次只读本地快照，不把旧 `running` 当成当前仍在训练，也不将交接信息伪装为快照内部证据。

| 未完成 job | 本地最新已回收 checkpoint | 暂停后远端最后 saved，尚缺补传 |
|---|---:|---:|
| supervised_attention / 1011 | 700；11,468,800 transitions | 789 |
| supervised_attention / 1022 | 600；9,830,400 transitions | 732 |

主 agent 已确认 789/732 对应真实 `stopped / SIGTERM` completion 及 SHA，但 SSH connection reset 阻止了补传。本地验收没有验证这两个尚未到达的文件，**不能宣称所有最后停点均已回收**。

后 7 项从未训练：`supervised_attention/seed_1033`；`history_mlp` 的 1011/1022/1033；`history_gru` 的 1011/1022/1033。它们与两个 supervised 中间结果均不进入同预算质量比较。

## 完整性和模型验收

| 项目 | 本次实际结果 |
|---|---|
| 归档 | 226,433,945 bytes；归档 SHA、729 个 tar member 内容 SHA 均重新核对 |
| 已提取文件 | 729 files，313,976,774 bytes；逐文件大小、SHA、路径边界及文件集合匹配 |
| 回收 manifest | 与固定 SHA 锚点及原 recovery receipt 匹配；未改动原始收据 |
| 完整 job | 12/12 的 result 为 completed，completion 为 updates_completed |
| 每完整 job 预算 | 977 updates；512×32×977 = 16,007,168 transitions |
| checkpoint | 完整 job 的所有 completion checkpoint 引用均映射后验 SHA；12 final + 2 latest partial 做 CPU 权重/schema 检查 |
| 评估 | 84/84：7 场景×12 job；eval seed 301，2000 vector steps，8 envs，16,000 transitions，deterministic mean，action_clip=100 |
| 评估 schema | 复用仓库 `_report` / `_validate_stability`，核对 metric 样本数、稳态协议、final checkpoint SHA 和环境身份 |
| 来源 | 训练 commit `80a45b4`；本机 `804f47c` 的实际 package 文件哈希与冻结 plan 一致；回收 package 同样逐文件一致 |
| 外部环境 | contract 文件及规范化 SHA、资产 manifest／资产文件、task source files／aggregate SHA、adapter／worker SHA 与模型和评估 identity 相符 |

关键锚点：

```text
archive  b798cdfe296da6ee2524549cad0e3d0cdefcd85a991d955e93197f6949d1eb92
manifest 4d8ba084fb983eb4c1e4c89df1c2851d24339fc4cddd393fc02919628f393b4a
plan     7a37cad280f7676b4103c0ec883bf1c8ee16e41adf72a9ae9c843f03ea90ed57
package  e6ef86cae8177052999b773d2615af5edaa2c46774a9942548fa0f47242af5c7
```

checkpoint 原路径是 `/home/kaiser/robot-rl-sim60/architecture-16m-20260915T0540/...`。工具只做这个**明确远端 prefix → 本地 `extracted/study`** 的路径映射，拒绝相似前缀、越界和 `..`，再核验所属 job 目录及 SHA；raw JSON 中的路径保持原样。

### 原 loader 的本地兼容性失败

本机与训练记录均为 PyTorch `2.11.0+cu128`；本机 CPU dispatch 为 AVX2。14 个 checkpoint 的差异一致：

| `actor.time_frequencies` 索引 | checkpoint 值 | 本地按配置重建值 |
|---:|---:|---:|
| 3 | 0.4216965138912201 | 0.4216964840888977 |
| 23 | 0.0013335214462131262 | 0.0013335213297978044 |

最大绝对差 `2.9802322387695312e-8`，最大相对差 `8.729918241696692e-8`；均为相邻 float32 值。其余预处理 buffer 未发现差异。固定频率来自 `torch.exp`；现象符合跨 CPU 数值重建差异，但本次没有远端 CPU 对照，**具体根因尚未证实**。本机分别使用默认、显式 AVX2 和 DEFAULT dispatch 计算这两个元素，结果一致，未解决差异。

本次未修改 checkpoint、未替换 buffer、未放宽 loader，也未执行策略推理。严格加载失败和独立 tensor/Adam schema 检查结果同时保存在 JSON；工具完成输出后返回 1，表示验收仍有未解决失败，而不是统计文件生成失败。已保存的历史评估仍可按其原始 SHA、配置和环境身份分析，这不等于证明本机可重跑。

## 统计口径：先资格，再看抖动

只比较 `last_token_attention / time_attention / index_attention / gated_attention`，每类完整保留训练 seeds 1011/1022/1033。

1. 每个训练 seed、每个场景先取其独立评估文件的指标，详见 84 行 CSV。
2. 下表 `mean ± std` 是这 **3 个独立 training seeds** 的均值与样本标准差（ddof=1），不是 8 个 env 作为 8 个 seed，也不是置信区间。
3. 高度 signed bias、vx/wz signed error 来自每 episode 去掉 200 settle steps 后、至少 200 retained samples 的稳态窗口；误差符号为实际值减命令。高度 `within_episode_std` 已按 episode 中心化，换算为 mm；它不是跨训练 seed 的 std。
4. planar speed、tilt、nonwheel netforce 与跟踪 MAE 是**全时段**指标，含 transient 与 reset；现有信号不足以将它们冒充稳态值。
5. 84 份评估均为 terminated=0、truncated=8、done=8。每份保留 14,392/16,000=89.95% 稳态样本、8 usable segments；另有 8 个短尾 segment，short retained count=0。稳态 coverage 是统计窗口覆盖，不是物理有效站姿比例；20 s 总窗口含 timeout，不能称为长时间无 reset 站立。

### 高度与站姿

low/mid/high 的目标分别是 0.28/0.30/0.32 m。下表各列均为跨训练 seeds 的 mean ± std。

| 架构 | 场景 | 高度 signed bias (mm) | episode 内高度 std (mm) | vx signed drift (m/s) | planar speed (m/s) | tilt (rad) | nonwheel netforce (N) |
|---|---|---:|---:|---:|---:|---:|---:|
| last-token | low | −66.97 ± 57.21 | 0.026 ± 0.021 | −0.0015 ± 0.0042 | 0.0085 ± 0.0031 | 0.133 ± 0.145 | 76.1 ± 43.2 |
| last-token | mid | −86.95 ± 57.27 | 0.027 ± 0.021 | −0.0014 ± 0.0042 | 0.0084 ± 0.0032 | 0.132 ± 0.145 | 75.6 ± 43.5 |
| last-token | high | −106.84 ± 57.33 | 0.036 ± 0.034 | −0.0016 ± 0.0039 | 0.0079 ± 0.0021 | 0.132 ± 0.143 | 76.4 ± 43.2 |
| time | low | −50.94 ± 43.53 | 0.167 ± 0.255 | +0.0024 ± 0.0022 | 0.0107 ± 0.0080 | 0.259 ± 0.215 | 652.2 ± 896.8 |
| time | mid | −71.25 ± 43.30 | 0.279 ± 0.252 | −0.0024 ± 0.0086 | 0.0124 ± 0.0106 | 0.267 ± 0.212 | 628.4 ± 856.2 |
| time | high | −91.37 ± 43.29 | 0.189 ± 0.267 | −0.0095 ± 0.0206 | 0.0169 ± 0.0183 | 0.269 ± 0.211 | 616.9 ± 837.3 |
| index | low | −61.02 ± 40.67 | 0.030 ± 0.026 | +0.0007 ± 0.0018 | 0.0129 ± 0.0090 | 0.188 ± 0.139 | 602.0 ± 924.3 |
| index | mid | −81.48 ± 40.38 | 0.028 ± 0.044 | −0.0107 ± 0.0180 | 0.0187 ± 0.0182 | 0.194 ± 0.136 | 619.3 ± 952.9 |
| index | high | −101.91 ± 40.37 | 0.032 ± 0.053 | −0.0092 ± 0.0157 | 0.0180 ± 0.0171 | 0.196 ± 0.133 | 505.8 ± 754.3 |
| gated | low | −69.17 ± 32.73 | 0.206 ± 0.171 | −0.0001 ± 0.0004 | 0.0076 ± 0.0023 | 0.187 ± 0.074 | 56.0 ± 11.5 |
| gated | mid | −89.09 ± 32.79 | 0.203 ± 0.171 | −0.0001 ± 0.0004 | 0.0076 ± 0.0022 | 0.187 ± 0.074 | 55.8 ± 11.5 |
| gated | high | −108.99 ± 32.91 | 0.243 ± 0.213 | −0.0001 ± 0.0005 | 0.0075 ± 0.0023 | 0.187 ± 0.075 | 56.6 ± 11.4 |

**资格判断先于 jitter 排名：**在 mid 高度，12 个模型的 signed bias 全部低于 −20 mm（最好 −21.11 mm）；low 只有 last-token/1011 接近目标，其余 11 个也均低于 −20 mm。这里的 20 mm 只是事后说明偏差量级，不是训练前冻结的验收阈值。没有模型能同时展示三个目标高度的有效跟踪。

尤其 last-token/1033 的 mid bias 为 −125.24 mm，实际稳态高度约 0.1748 m，episode 内 std 却只有 0.0023 mm；这是“低高度静止”反例，不能称为站得最好。各模型从 low 到 high 的 bias 大致再减 40 mm，说明目标升高时实际高度大多没有相应变化。

netforce 的原语义是 `rigid_body_net_force_history_peak_not_ground_pair`。time/1033 与 index/1033 在站立场景出现约 1.4–1.7 kN 非轮净力，需要保留为异常线索；它没有接触对手身份，**不能直接断言身体触地或把它当作地面支撑力**。

### 前后行驶和左右旋转

前后指令 vx=±0.5 m/s，旋转指令 wz=±1 rad/s，目标高度均为 0.30 m。下表为对应轴的全时段 MAE，mean ± training-seed std；每个场景的 vx/wz signed error 与两轴 MAE 都完整保留在 JSON/CSV。

| 架构 | forward vx MAE (m/s) | reverse vx MAE (m/s) | turn_left wz MAE (rad/s) | turn_right wz MAE (rad/s) |
|---|---:|---:|---:|---:|
| last-token | 0.5024 ± 0.0011 | 0.4979 ± 0.0020 | 0.9987 ± 0.0099 | 0.9915 ± 0.0195 |
| time | 0.4971 ± 0.0012 | 0.4849 ± 0.0285 | 0.6789 ± 0.5451 | 0.7980 ± 0.3488 |
| index | 0.5017 ± 0.0231 | 0.4772 ± 0.0328 | 0.8388 ± 0.2711 | 0.8808 ± 0.2210 |
| gated | 0.5033 ± 0.0040 | 0.4966 ± 0.0025 | 0.9959 ± 0.0027 | 1.0054 ± 0.0128 |

多数模型的稳态 vx error 在 forward 接近 −0.5、reverse 接近 +0.5，主要表现为未跟随速度命令。time/1033 的左转 wz MAE 为 0.0494 rad/s，但同时 vx 稳态漂移 +0.1947 m/s、高度偏低 46.78 mm、tilt 0.354 rad、非轮净力约 1516 N；这不能作为合格的纯旋转。右转 MAE 仍为 0.3952 rad/s，vx 漂移 −0.2771 m/s。time 的跨 seed 左转均值因此被单个 seed 拉低，并不说明三个 seed 都学会旋转。

### 动作目标与 effort 导数

每个评估文件先在同单位、相同 retained support 的通道内取 `sqrt(mean(derivative_rms²))`，再对三个训练 seeds 取 mean ± sample std。腿目标为 4 个关节位置通道，轮目标为 2 个速度通道；effort 按 contract 的腿索引 `[0,1,3,4]`、轮索引 `[2,5]` 分开。这里没有混加 rad、rad/s 与 Nm。

以 stand_mid 为例（其余场景见证据）：

| 架构 | 腿目标导数 RMS (rad/s) | 轮目标导数 RMS (rad/s²) | 腿 effort 导数 RMS (Nm/s) | 轮 effort 导数 RMS (Nm/s) |
|---|---:|---:|---:|---:|
| last-token | 0.01589 ± 0.01658 | 0.10487 ± 0.13635 | 1.724 ± 1.776 | 1.354 ± 1.140 |
| time | 0.01718 ± 0.02832 | 0.20755 ± 0.35249 | 7.413 ± 11.966 | 8.318 ± 13.820 |
| index | 0.00375 ± 0.00592 | 0.05147 ± 0.08259 | 1.378 ± 1.790 | 1.194 ± 1.397 |
| gated | 0.00110 ± 0.00086 | 0.01452 ± 0.01526 | 1.059 ± 0.314 | 7.830 ± 5.169 |

effort 是 policy boundary 处最后一个 physics substep 的 effort target；100 Hz 采样不能替代完整 200 Hz 力矩时序或硬件 PID 观测。gated 的动作目标变化较小并没有解决高度或运动跟踪问题，轮 effort 导数也并非最小，不能只凭某一 jitter 列选型。

## 哪个候选更值得后续关注

目前没有依据直接推荐某一架构扩大训练。若后续只选一个模型做**站姿问题诊断对照**，优先保留 `last_token_attention/seed_1011`：它的 low 高度误差 −1.20 mm，mid/high 分别 −21.11/−40.94 mm，实际高度始终约 0.279 m，便于排查“保持某个高度但不跟随高度命令”。它仍有约 0.30 rad 倾斜，而且前后／旋转不跟踪；同架构另两个 seed 明显更低，不能提升为架构结论。

`time_attention/seed_1033` 可作为“出现旋转响应但伴随漂移和异常净力”的第二个诊断样本。其单场景表现不能抵消资格缺陷，也不能代表 time-attention 已优于其他架构。

本阶段只完成回收、CPU 验收与历史数据分析；训练保持暂停。最重要的后续缺口是补齐 789/732、定位严格 loader 的跨机重建差异，以及先解释高度不响应和非轮净力异常，再讨论新增训练预算。MLP/GRU 尚无本批同预算证据。

## 复核方式与限制

输出文件必须是新路径，工具拒绝覆盖已有文件或写入回收根目录。例如：

```bash
CUDA_VISIBLE_DEVICES='' /home/yukikaze/isaacsim60-venv/bin/python \
  tools/analyze_recovered_study.py \
  --root artifacts/recovered/architecture-16m-20260915T0540/20260915T075822Z-8f22 \
  --json /tmp/opencode/training-recovery-recheck.json \
  --csv /tmp/opencode/training-recovery-recheck.csv \
  --print-tables
```

本次执行已产出上述仓库证据文件；因保留 14 项 strict loader failure，退出码为 1。另做了 5 项定向 CPU 检查：prefix 映射、相似前缀／越界拒绝、样本 std 与 missing 处理、同单位通道 RMS／support 拒绝、84 行 CSV 与 JSON 跨 seed 汇总一致性及 LF；全部通过。工具 `py_compile` 通过；当前 venv 未安装 Ruff，未完成该 lint 检查。未重复运行全仓测试。

只有 3 个训练 seeds、1 个 eval seed，20 s 场景窗口且含 timeout；没有扰动恢复、指令转换或硬件验证。因此本报告描述已保存的研究仿真结果，不提供部署资格判断。原始失败、缺失、历史日志、manifest 和 receipt 均保留，大文件继续由 `.gitignore` 排除。
