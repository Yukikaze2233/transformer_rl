# 旧 MLP 稳态基线：已有证据与六网络评价口径

研究日期：2026-09-14。机器证据：[mlp-steady-state.json](evidence/mlp-steady-state.json)。

## 结论与完成范围

**旧 MLP 已有明确的零速位置漂移证据；高度波动小、未倒下，都不等于原地站稳。它适合启发评价指标，不能直接与今夜全新配方训练的网络排名。**

- 五高度固定零速试验均为 6000 步、100 Hz、单环境、seed42；每项有三个约 20 秒 timeout 回合，以及最后 3 步的部分回合。每个完整回合剔除前 2 秒后有 1799 个样本。
- 0.29 m：逐回合高度 signed bias 约 +0.87～+0.91 mm，但回合净位移 0.58～0.77 m。0.30 m：逐回合约漂移 3.92 m。0.32 m：高度变化范围很小，却偏高约 6.25 mm，回合净位移约 0.99 m。
- 0.28 m：稳态全部 5397 帧非轮净力诊断异常；缺碰撞对手身份，支撑有效性仍不明确，不能让它凭低波动进入“合格站立”排名。
- **经用户澄清允许本地 CPU 分析，现已完成 30,000 行原 CSV 的 SHA256、时间步、命令、reset 检查，以及 15 个完整回合的 signed bias/mean、demeaned std 和世界 XY 位移复算。** 按各回合样本数加权合并 within-episode variance，回合均值差另存，未混入 jitter。
- 0.32 m 的合并高度 std 为 **0.00393574 mm**、vx std 为 **0.0000918766 m/s**，但 mean vx 为 **−0.0501457 m/s**：极小波动伴随持续后退。0.28 m 高度 std 为 **3.80241 mm**、vx std 为 **0.159094 m/s**，且支撑有效性存疑。
- 五份 CSV 哈希均重新计算并与旧证据一致。分析仅使用 Python 标准库读取 CSV 和历史哈希 JSON，没有加载模型或网络推理，没有访问 Kaiser、训练、仿真、GPU 作业或调度。正式训练仍由主 agent 按用户“今晚 0 点后”的要求处理。

## 1. 数据身份与适用范围

下文路径前缀均指本机已有资料：

| 简称 | 本机路径 |
|---|---|
| `R` | `/home/yukikaze/Documents/workspace/robot_rl` |
| `F` | `R/reports/current/cqa1_round2_20260912/final` |
| `H` | `R/isaac_wheeled_rl_train-60/reports/round2-height-kaiser-20260912T155346Z` |
| `E` | `R/isaac_wheeled_rl_train-60/docs/evidence/round2-height-kaiser-20260912T155346Z.json` |

### 最终策略，不混入早期或其他轮次

`F/agent_config.json`、`run_manifest.json`、`policy.onnx.json`：RSL ActorCritic，actor/critic hidden dims 均为 `[256,128,64]`，ELU；actor 输入为 125 维历史观测，critic29，action6。这是历史输入 MLP，不应当作单帧 MLP。PPO 为初始 LR1e-4、adaptive schedule、5 epochs、4 minibatches、initial std0.2，无 empirical normalization。

`F/completion.json` 记录本次追加 19980 updates 完成；`HANDOFF.md` 记录从 20-update pilot 恢复，计划累计 20000。`complete=true`、`export_status=verified` 同时伴随 `policy_quality_verified=false`，训练完成不代表行为通过。

关键 SHA256（均为旧记录中的身份值）：

| 对象 | SHA256 |
|---|---|
| 最终 `policy.onnx` | `7d4a93f0d0692f914dc769fb94c88550de62e44a0b3553d95b20d80d406b700e` |
| 最终 `model_final.pt` | `f94919f8f7ae1cfee95e45b922a9939e21c9f6ad9dff5656ff6f29dc0f4e6360` |
| `contract.json` 文件 | `5b97bf2935b5bb04357737e2a54046ab891c15bc15d6e4c2ad9d4d6150e3e8cc` |
| contract semantic digest | `667865de8352e67cff6720af9a142d00ff9b980132b27e4993a25c8e09d3d22e` |
| asset manifest | `df5ca7693022b7a4c68cb1dc823482291f265e3addc8c4870fb3b0b2ab364886` |
| 训练 `env.py`（`F/source_hashes.json`） | `66e26b470a128c7504d3df17111cd9cc0cf3183a0daec1888b702dacea6f0ca0` |
| 五高度评估 `env.py`（`H/provenance.json`） | `e94314e55133884106349481929a80fe937dafdfd7a4ca313aa927c964fb07ad` |
| 五高度评估 `scripts/play_v40_onnx.py` | `91064e8a1b120d355d79ce5a49b2489e2a83c4bb854f32683ccb68ec2a50bb72` |

训练源码短版本由旧 `HANDOFF.md` 记为 `453cd1c`；不把当前工作树 HEAD 当作旧运行版本。五高度历史执行源为 `/home/kaiser/robot-rl-sim60/v40-live-snapshot03`，这里仅引用本机收据中的字符串。

**跨版本 scope：**训练为 Sim5.1 / Lab2.3 / RSL3.0.1 / Torch2.7.0+cu128；评估为 Sim6.0.0.1 / isaaclab 包6.1.14 / RSL5.5.1 / Torch2.11.0+cu128 / ORT1.30.0，确定性 ONNX raw mean、CPUExecutionProvider。评估 Lab 源码为 `ffff603eafc6b74264a5261cc0183d6a65390d78`，tag `v3.0.0-beta2.patch1`。训练与评估 `env.py` 哈希也不同。资产一致不意味着运行栈或全部行为数值等价。

`stand_0.records.json` 在旧 HANDOFF 中属于第一轮策略检查，`review4200/` 是中间 checkpoint；其记录即使含关节姿态，也不能补成最终 ONNX 五高度的逐 joint 数据。`isaac_wheeled_rl_deploy` 是用户明确的 demo；不作为可用部署、硬件控制或真实 PID 证据。

## 2. 已核对的 telemetry 字段

已读取 `H/h028` 至 `h032/telemetry.csv` 的表头，以及 native teleop / 两个 exploratory CSV 表头。**这些 CSV 均没有 quaternion。**

| 原列或数据 | 能评价什么 | 边界 |
|---|---|---|
| `x,y,z` | root-link 世界 XY 位移；平地高度 signed error | 五高度 env origin z=0；不是轮轴中心，也不是 COM 位置 |
| `vx,vy,vz,wz` | body COM 速度均值与波动，`hypot(vx,vy)` 平移速率 | `wz` 是 body angular z，不严格等于姿态 Euler yaw 的时间导数；不能直接积分得到精确世界 XY |
| `gravity_z` | 总倾角 `acos(clamp(-gravity_z,-1,1))` | 不足以恢复 signed roll/pitch/yaw；其 std 不能替代 signed 姿态 jitter |
| `action_cmd_vx/wz/height`、`reward_cmd_*` | 识别该 transition 的命令及一致性 | `action_cmd_*` 是任务命令，不是电机动作；不能用 `requested_*` 或 `next_cmd_*` 替代 |
| `step,sim_s,policy_tick` | 策略 tick / 采样时轴检查 | CSV 行号比 `step` 多 1（表头） |
| `sample_kind,episode_step,episode_time_s` | 五高度 pre-reset 所属回合及前 2 秒剔除 | native teleop / exploratory 旧表头缺这三列；也都没有显式 `env_id,episode_id` |
| `terminated,timeout`；五高度 `diagnostic_*`,`termination_*` | 区分失败和 timeout；诊断与终止分别统计 | terminal pre-reset 样本仍归旧回合，下一样本才归新回合 |
| `non_wheel_contact_n`、五高度 `non_wheel_net_force_max_n` | 非轮净力 history peak，阈值 >1 N 的帧占比 | 无 pair identity、轮地支撑分解；不是独立碰撞次数 |
| `clearance_m` | base visual bounds clearance 下界诊断 | 不等价于精确碰撞几何或接触对身份 |
| `action_norm` | raw 六维 action 的 L2 norm | 不能恢复各维符号、关节动作波动、限幅比例或实际力矩；等范数动作也可剧烈变化 |

源码映射可见旧工作树 `scripts/play_v40_onnx.py:570–658`：`root_link_pos_w_m`→`x,y,z`；`root_com_lin_vel_b_m_s`→`vx,vy,vz`；`root_com_ang_vel_b_rad_s[...,2]`→`wz`。当前文件仅用于交叉理解字段；旧运行的 source 身份以上述 provenance 哈希为准，本次未验证当前文件与旧哈希相等。

### 为什么不直接用 native replay 或训练曲线算站立 jitter

- `R/reports/server_final_onnx_teleop_20260912T2013` 的 summary 虽写初始 `[0,0,0.30]`，但 tick20 已应用 `[1,0,0.30]`，tick22 又变零；`handoff_verification.json` 还记录前后行驶及正反转向。不能把整个 6000 步当零速稳态。其 1375 行 handoff 验证只是运行中快照。
- `server_final_onnx_exploratory_20260912T203531`、`server_final_onnx_exploratory_20260912T204509-dd2e79` 都是 keyboard/exploratory，允许上限 5 m/s、120 rpm，CSV 多了 OOD 标志。允许上限不证明实际越界；需要逐 transition 命令筛选。它们缺五高度版的显式 pre-reset/episode 诊断列，本次不为其推定合格恒命令回合或重用后来的诊断零计数。三个 native 数据集均未纳入下表，CSV 哈希也未冒充已核验。
- `R/reports/current/cqa1_round2_20260912/curves/train-scalars.csv` 列为 `tag,step,wall_time,value`。manifest 有 `Tracking/height_m`、`Tracking/vx_m_s`、`Tracking/wz_rad_s` 和各 abs_error 标签，但没有逐 env/episode 原序列；训练混合命令、环境与 updates。这些曲线可看学习进展，不能用曲线 std、`Policy/mean_noise_std` 或 `Reward/action_rate` 当机器人稳态 jitter。

## 3. 分回合统计定义与实际窗口

以 `(source, training_seed, eval_seed, env_id, episode_id, fixed_command)` 为键。五高度无 env 列，依据单环境运行配置赋逻辑 `env_id=0`，不是读取了不存在的列。

1. `sample_kind=pre_reset`；`step/policy_tick` 连续，`sim_s≈step*0.01`，数值有限。先分回合，不先删帧后拼轨迹。`episode_step` 回落应与前一行 `terminated/timeout` 一致；reset 标志行属于结束的回合。
2. 整个纳入回合的实际 action/reward 命令都应为 `(0,0,h)`；高度一致性使用绝对容差 1e-7 m 处理 FP32 命令表示。不能把 `0.30000001192092896` 与 `0.3` 当切换，也不能将真实不同高度合并。若日后研究切换后的站立段，另设 segment_id 和切换后等待窗，不冒充固定命令回合。
3. 每次 reset 后仅保留 `episode_time_s > 2.0`。短回合和失败都保留计数，不用“删掉不稳定帧”获得低 jitter。
4. 在每个回合保留窗口内，对带符号值用 FP64 两遍算法或 Welford 计算 population std（ddof=0）：

   ```text
   e_h[i] = z[i] - h
   height_bias = mean(e_h)
   height_demeaned_std = sqrt(mean((e_h - height_bias)^2))
   vx_mean_drift = mean(vx)
   vx_demeaned_std = sqrt(mean((vx - vx_mean_drift)^2))
   wz_mean_drift = mean(wz)
   wz_demeaned_std = sqrt(mean((wz - wz_mean_drift)^2))
   ```

   单位分别为 m、m/s、rad/s；报告高度时可乘 1000 为 mm。mean drift 是 body 速度 DC 分量，与世界位置漂移分列。去均值不去趋势，慢变化也属于该窗口波动；“稳态”仅为窗口名，不声称已经收敛。
5. 世界 XY 分别计算整回合和稳态窗口的 `dx=x_last-x_first`、`dy=y_last-y_first`、`D=hypot(dx,dy)`；另记路径长度和最大离起点距离，避免绕圈净位移小。每回合使用自身首个已记录位置，绝不跨 reset 相减。没有 t=0 样本时不虚构它。

不能从 `std(abs_error)` 恢复 signed jitter。即使全程都有 RMS 和 MAE，`sqrt(RMS²-MAE²)` 也只是绝对误差的 std；既丢符号，又混入回合间均值。

本次严格先逐回合算矩，再在**同一高度、env0、同一训练/评估 seed**内合并。对每个信号，令回合样本数为 `n_e`、均值为 `μ_e`、population variance 为 `v_e`：

```text
N = sum(n_e)
pooled_mean = sum(n_e * μ_e) / N
within_variance = sum(n_e * v_e) / N
within_std = sqrt(within_variance)
between_variance = sum(n_e * (μ_e - pooled_mean)^2) / N
```

`within_std` 是合并的 jitter；不是 std 的算术平均，也不是相对全局均值计算的 std。直接拼接的总方差会等于 `within_variance + between_variance`；JSON 分别保存两项，**绝不将 between 加入 jitter**。分母为 N（ddof=0），不是 N−回合数。不同高度不合并；最后 3 帧 partial 回合保留记录，但没有稳态样本，不参与合并。

### 五高度共同窗口（本次全 CSV 复核）

| episode | 全回合 policy step | 稳态 policy step | 稳态 sim_s 首末 | episode_time_s 首末 | 全/稳态样本数 |
|---|---|---|---|---|---:|
| 1 | 1–1999 | 201–1999 | 2.01–19.99 | 约 2.01–19.99 | 1999 / 1799 |
| 2 | 2000–3998 | 2200–3998 | 22.00–39.98 | 约 2.01–19.99 | 1999 / 1799 |
| 3 | 3999–5997 | 4199–5997 | 41.99–59.97 | 约 2.01–19.99 | 1999 / 1799 |
| 4（partial） | 5998–6000 | 无 | 无 | 约 0.01–0.03 | 3 / 0 |

每个稳态窗样本支持时长 `N*dt=17.99 s`，首末点跨度 `(N-1)*dt=17.98 s`，两者不可混用；位移若除以时间，应使用首末点跨度。前三回合各去掉 200 帧，末回合去掉 3 帧，总计去掉 603、保留 5397（53.97 样本秒），**不是无 reset 连续 60 秒**。原始采集北京时间为 2026-09-12 23:55:45 至 2026-09-13 00:14:43。

本次抽查 `h030/telemetry.csv` 第 2000–2001 行：step1999 为 `timeout=True,episode_step=1999`，step2000 为新回合 `episode_step=1`，世界 x 从约 -3.9247 回到 -0.000043 m。跨这两行相减会伪造约 3.925 m 的反向运动。

## 4. 原 CSV 复算结果

分析器：[tools/analyze_reference_trace.py](../tools/analyze_reference_trace.py)。Python float（FP64），两遍去均值、`math.fsum` 求和，直接使用 `z−h`、`vx`、`wz` 和 `x,y`。以下 signed bias 与位移均重新计算；与旧 E 的相应指标在浮点精度内一致。每格三个值依次为 env0 episode1/2/3；全精度结果在 JSON。

| h (m) | 稳态 signed height bias (mm) | 整回合世界 XY 净位移 (m) | 仅稳态窗口世界 XY 净位移 (m) |
|---|---|---|---|
| 0.28 | +4.116025 / +4.296977 / +4.140894 | 0.834524 / 0.777875 / 0.822828 | 0.838670 / 0.783786 / 0.836975 |
| 0.29 | +0.868093 / +0.900954 / +0.914106 | 0.579059 / 0.720451 / 0.765622 | 0.476774 / 0.615823 / 0.661057 |
| 0.30 | +5.975342 / +5.982346 / +5.972474 | 3.925263 / 3.923728 / 3.926954 | 3.569715 / 3.576009 / 3.572434 |
| 0.31 | +8.190640 / +8.264118 / +8.195636 | 2.720240 / 2.747287 / 2.714095 | 2.440364 / 2.507777 / 2.442264 |
| 0.32 | +6.252699 / +6.252601 / +6.252733 | 0.991362 / 0.991612 / 0.991189 | 0.901706 / 0.902000 / 0.901600 |

### 逐 episode 的 signed mean 与 demeaned std

每行 **N=1799**，均在该回合 `episode_time_s > 2.0` 后统计。高度单位 mm，vx 单位 m/s，wz 单位 rad/s；std 列均已减去本回合均值，**不是 abs_error std**。

| h (m) | ep | height bias | height std | vx mean | vx std | wz mean | wz std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.28 | 1 | +4.116025 | 3.873787 | +0.060191 | 0.158165 | −0.112212 | 0.113227 |
| 0.28 | 2 | +4.296977 | 3.715273 | +0.054637 | 0.160026 | −0.109015 | 0.108126 |
| 0.28 | 3 | +4.140894 | 3.816474 | +0.061276 | 0.159087 | −0.113120 | 0.115249 |
| 0.29 | 1 | +0.868093 | 0.317384 | −0.026496 | 0.014301 | +0.002514 | 0.005392 |
| 0.29 | 2 | +0.900954 | 0.330055 | −0.034201 | 0.018694 | +0.001987 | 0.005232 |
| 0.29 | 3 | +0.914106 | 0.322657 | −0.036708 | 0.018612 | +0.002484 | 0.005341 |
| 0.30 | 1 | +5.975342 | 0.528087 | −0.198125 | 0.009757 | +0.003179 | 0.017624 |
| 0.30 | 2 | +5.982346 | 0.532374 | −0.198354 | 0.009943 | +0.003379 | 0.017910 |
| 0.30 | 3 | +5.972474 | 0.530992 | −0.198241 | 0.009851 | +0.003369 | 0.017780 |
| 0.31 | 1 | +8.190640 | 0.363133 | −0.135863 | 0.016510 | +0.000207 | 0.006024 |
| 0.31 | 2 | +8.264118 | 0.436299 | −0.139396 | 0.020647 | +0.000481 | 0.006266 |
| 0.31 | 3 | +8.195636 | 0.338077 | −0.135856 | 0.016497 | +0.000078 | 0.005467 |
| 0.32 | 1 | +6.252699 | 0.003957 | −0.050142 | 0.0000912982 | +0.001104 | 0.001019 |
| 0.32 | 2 | +6.252601 | 0.003769 | −0.050159 | 0.0000897749 | +0.001160 | 0.000874 |
| 0.32 | 3 | +6.252733 | 0.004075 | −0.050136 | 0.0000944937 | +0.001089 | 0.001023 |

### 同高度合并的 within-episode 波动

每行 **3 episodes、N=5397**。mean 是各回合均值的样本数加权均值；三个 std 都是 `sqrt(sum(n_e*v_e)/N)`。0.28 m 保留为异常行为诊断，不能作为有效站立候选。

| h (m) | height bias (mm) | height within std (mm) | vx mean (m/s) | vx within std (m/s) | wz mean (rad/s) | wz within std (rad/s) |
|---|---:|---:|---:|---:|---:|---:|
| 0.28 | +4.184632 | 3.802410 | +0.058701 | 0.159094 | −0.111449 | 0.112241 |
| 0.29 | +0.894384 | 0.323407 | −0.032468 | 0.017324 | +0.002328 | 0.005322 |
| 0.30 | +5.976721 | 0.530487 | −0.198240 | 0.009851 | +0.003309 | 0.017772 |
| 0.31 | +8.216798 | 0.381453 | −0.137038 | 0.017991 | +0.000255 | 0.005929 |
| 0.32 | +6.252678 | 0.003935739 | −0.050146 | 0.0000918766 | +0.001118 | 0.000974610 |

例如 0.29 m 的 vx `within_variance=0.00030012416094400403 (m/s)²`，`between_variance=0.000018881378158927144 (m/s)²`。如果把三个回合均值的差异加入，会把 jitter 人为增大；本表只使用前者。

**行为解释：**0.29 m 高度偏差和高度波动均较小，但三个回合的后退速度均值从约 −0.0265 到 −0.0367 m/s，并有 0.58～0.77 m 位移；0.30 m 后退 DC 分量约 −0.1982 m/s，不能将“vx std 比 0.29 更小”解释为更好地站稳。0.28 m 的 vx/wz 波动明显，但与持续净力异常同时出现，不能据此单独归因为 MLP 结构或周期振荡。

辅助证据（同高度三个稳态窗合计，各 5397 帧；这是旧阶段汇总，不是跨回合 jitter）：

| h (m) | 平移速率均值 (m/s) | 稳态非轮净力 >1 N 帧数 | 解读 |
|---|---:|---:|---|
| 0.28 | 0.150012 | 5397 | 净力均值 449.69 N、峰值 889.99 N，支撑性质不明 |
| 0.29 | 0.032988 | 0 | 高度准确仍有慢漂；全程有 287 帧初始化接触异常 |
| 0.30 | 0.200596 | 0 | 高度附近保持，但位置明显漂移 |
| 0.31 | 0.137416 | 0 | 高度正偏差与漂移同时存在 |
| 0.32 | 0.050161 | 0 | 很小的高度变化范围也没有保证位置保持 |

上述均值转录自 `E/heights[*]/phases/steady/planar_speed_mean_m_s`，舍入至六位小数。各高度均记录 failure termination=0、timeout=3；五高度版 instantaneous tilt、low height、knee limit 等诊断计数为零，但这些阈值诊断不构成完整 signed 姿态或关节裕量测量。

0.32 m 现在已有实际逐回合高度 std：**0.00395665 / 0.00376933 / 0.00407519 mm**，合并 within std 为 **0.00393574 mm**。此前从旧 min/max 得出的 `<0.019223 mm` 仅为上界，JSON 保留其独立出处，不与本次实算值混淆。如此小的仿真高度波动仍伴随 +6.25 mm 偏置和每回合约 0.99 m 净漂移，足以反驳“只按 jitter 最小选赢家”；不将微米量级的仿真状态变化解释为实机测量精度。

### 原始 CSV 哈希

下列值由**同一份读取字节同时用于 SHA256 和 CSV 解析**，全部与 `E/heights[*]/csv_sha256` 一致。本次状态为 `recomputed_all_five_csvs_match_historical_evidence`。

| `H` 下原文件 | SHA256 |
|---|---|
| `h028/telemetry.csv` | `7dc9236243a629d68cd3d87d72defb43d9c16a591c8cfec28c0543619775224e` |
| `h029/telemetry.csv` | `fe1b2b7c03f4da55838868b9d7244835df8b753a6fa5e53f2ffb6b63d080b9cf` |
| `h030/telemetry.csv` | `befc486daaaed7aa40eccf01bddb89a847daa9ba6b815aac0dcd994e4c5b48cb` |
| `h031/telemetry.csv` | `9339c382bc2f5e6dfe993fb635b8f642f929da4d2cabd47b7aa1c25ec0975951` |
| `h032/telemetry.csv` | `2c1b55c44721759811db768965175d1f152b19d41da7eef5994c238e41d0cac0` |

对应文件大小依次为 **2,825,907 / 2,720,307 / 2,665,011 / 2,698,476 / 2,703,113 bytes**。

### 时间步与异常行审计

- 每文件 6000 条数据行（不含表头），`step=policy_tick=1..6000`，全部 `sample_kind=pre_reset`；action/reward/next/requested 四组命令都通过零 vx/wz、固定 h 检查。数值列全部有限，布尔列严格为 `True/False`。
- 每文件 5999 个 `sim_s` 相邻差均在 `[0.00999999999999801,0.010000000000005116] s`，相对 0.01 s 最大偏差 **5.116e-15 s**，容差 1e-9 s。策略 tick 没有缺帧、重复或乱序。
- 每文件 5996 个**回合内** `episode_time_s` 相邻差在 `[0.009998321533203125,0.010000228881835938] s`；最大步差误差 **1.678467e-6 s**，相对 `episode_step*0.01` 最大误差 **1.373291e-6 s**，均在为 FP32 episode 时钟设置的 2e-6 s 容差内。因此是**均匀策略采样、episode 时间浮点舍入**，不是声称 CSV 小数时间戳严格逐位等间隔。未插值或重采样。
- timeout 仅在 step1999/3998/5997；其 pre-reset 行留在旧回合。下一行 episode_step 重回 1。每高度 failure termination=0、timeout=3；最后 3 行为 partial episode4，完整保留位移记录，稳态指标为 null（N=0）。
- **格式/数值/命令/时序异常行=0，丢弃行=0。** 共排除初始化 3015 行，保留稳态 26985 行。初始化剔除不是坏行删除。
- 非轮接触异常是**物理诊断，不是格式异常**：0.28 m 全程5905帧、稳态5397帧；0.29 m 全程287帧、稳态0帧；其余0。这些帧全部参与相应窗口统计，支撑有效性单独标注。
- 分析器遇到缺列、错宽、空行、nonfinite、命令/时钟/reset 不一致时报告 CSV 行号并拒绝整个数据集；SHA 不匹配在解析前拒绝。不删掉异常行继续出“更稳定”结果。

## 5. 今夜六网络统一评价建议

研究目的仍是网络结构对照。当前 `configs/learning_curves.json` 列出 time_attention、index_attention、gated_attention、supervised_attention、history_mlp、history_gru；其中 supervised_attention 同时改变辅助监督，应标注为“结构＋监督”对照，不归因为纯结构效应。

统一 task/backend/资产/命令/噪声/初始状态/动作映射/采样率/训练预算/评估 seeds、确定性 mean 策略及 checkpoint 选择规则。当前配置是训练 seeds1011/1022/1033，977×32×512=16,007,168 transitions/run；旧 MLP 约 0.983B transitions，配方、历史输入和运行栈也不同。**旧数据是诊断参照，新配方下重新训练的 history_mlp 才是六网络同组基线。**同时报真实 transitions、有效 optimizer steps、GPU-hours 与学习曲线，不能以 update 数直接对齐旧训练。

### 先判断任务有效，再比较波动

1. **有效性层：**每个 episode 记录 nonfinite、任务失败原因、timeout、完成时长；height MAE/P95/max、姿态倾角、关节限位和接触证据。不能剔除失败后只展示幸存帧。为本轮研究预先冻结宽松的基本站立门槛，例如稳态高度绝对误差 P95≤20 mm、最大≤30 mm，倾角 P95≤10°、最大≤20°，无持续失稳/膝硬界越界；这些是待主 agent 统一冻结的建议值，不是已通过的测试或硬件标准。接触必须检查非轮异常及预期轮支撑；只有净力而没有对手身份时标注“支撑未确认”，不能将未知当通过。
2. **任务精度层：**在有效回合中报告 signed height bias、height MAE/P95，signed vx/wz mean drift、mean planar speed，以及世界 XY 净位移/路径长度/最大离起点距离。旧研究的精度参考为 height MAE≤5 mm、P95≤10 mm，静止速率≤0.02 m/s；位置保持应另定固定时长位移目标，不能由速率门槛替代。
3. **波动层：**按第 3 节逐回合报告 height/vx/wz demeaned std；有完整姿态后加 signed roll/pitch 波动与 yaw drift。小波动但高度错误、漂移或支撑无效者不能成为“站稳最佳”。呈现有效率→精度/漂移→波动的分层结果，不把所有量加权成能奖励趴地不动的单一分数。
4. **重复性层：**先按同一 training run 内 env/episode 汇总，再按独立 training seeds 汇总中位数、范围及不确定性；多帧/多 env/多回合不等于多个独立训练 seed。有效回合比例及缺失指标与波动结果一起展示；全失败时不得选出 jitter 赢家。

当前配置站立覆盖 0.28/0.30/0.32 m，建议统一补充 0.29/0.31 m，以暴露旧 MLP 已显示的高度依赖；这只是评价矩阵建议，本次未修改配置。每场景 2000 步不等于多个完整稳态回合，需以真实 reset 记录计算各自 N 和窗口。行驶、旋转、切换任务各自独立评价，不能混入零速 std。

### 对 Transformer 的具体启发

- 要检验的是历史建模能否在**高度/姿态/接触有效且漂移受控**时降低带符号状态波动；不能预设 attention 必然比历史 MLP 平滑。
- 区分 DC 偏置、位置漂移和窗口内波动。0.32 m 的案例表明，再降低高度 std 未必改善站立；0.29 m 则说明精确高度跟踪没有自动带来位置保持。
- 共同的动作平滑约束、观测信息或奖励调整如需采用，应统一应用到六网络并固定版本；只给 Transformer 修改配方会混淆结构因果。额外 temporal/action regularization 可留作独立消融。

## 6. 给主 agent 的评价接口缺口

首次只读接口检查时，新仓库 `examples/isaaclab_task.py:33–41` 仅发布 `vx_abs_error,wz_abs_error,height_abs_error,planar_speed,non_wheel_net_force`。`src/transformer_rl/evaluation.py:18–35,94–119` 将全部 env、初始段和 auto-reset 回合累加为 mean/rms/min/max/count。即使由这些值派生 std，也不是本报告定义的稳态 signed jitter；主 agent 后续补接口应以当前源码为准。

建议复用 adapter 已有 PRE-reset 捕获点（`isaaclab_task.py:62–81`），由 task 提供 SI 原值、坐标系与时序，由 evaluator 按 env/episode/command 管理窗口；不要让网络类承担物理指标语义。现有标量 `[N]` 指标接口若不能承载向量和身份信息，应由主 agent 设计独立 trace/episode 记录契约。

| 待补记录 | 原因 |
|---|---|
| `env_id,episode_id,episode_step,episode_time_s,policy_tick,sim_time_s,sample_kind`；terminated/truncated 和原因 | 正确归属 terminal pre-reset 样本、每 reset 排除 2 秒、防跨 env/reset 差分 |
| transition 的 actual command、command segment id、signed height error 和 signed vx/vy/wz | 避免 next observation 命令错配及绝对值丢符号 |
| root-link world position、env origin/地面参考，root quaternion（明确 wxyz/xyzw 和坐标方向）、完整 projected gravity | 世界位置保持、signed 姿态；现有 CSV 的 `gravity_z` 不足 |
| 每关节 `q,dq`、joint names/order/units、有限关节上下界 | 有限膝裕量；continuous wheels 用 dq/角增量，不将累计 q std 当轮抖动 |
| 逐维 raw policy mean/action、issued action、到达/应用 action 及相应时间、物理 q/dq target | 区分策略输出、运输、控制器应用；`action_norm` 无法补出这些量 |
| 每关节 commanded/applied torque、限幅标志和 effort limits；控制周期 | 关节动作变化率、饱和率、负荷；现有字段不能重建 torque 或电流 |
| 按 body/pair 标识的轮地与非轮接触、力/冲量定义和 history 长度 | 排除非预期支撑，避免把 history peak 当瞬时力或独立碰撞事件 |

若最终只有 abs_error 汇总而没有 signed 原值，字段状态必须是 `missing_signed_signal`。本次旧五高度的 signed 原值存在且已复算，状态为 `recomputed_from_signed_raw_columns`；缺失的 quaternion、逐 joint/action/torque/pair 字段继续保持缺失，不由汇总或模型推理补造。

## 7. 频域与验证边界

本次未做 FFT 或图。今后只有逐回合、恒命令、连续且足够长、确认无缺帧的均匀原序列才能做 FFT/PSD；不可拼 reset 或命令片段。100 Hz 数据只讨论 `0<f<50 Hz`，17.99 s 窗的原始频率间隔约 0.0556 Hz；短窗、泄漏和混叠须按实际数据解释。这里的 body 高度/速度采样及缺失的电流/torque telemetry，不能支持任何下位机高频电流 PID 抖动归因。

### 可复现 CPU 命令与验证

从新仓库根目录执行（输出完整计算 JSON 到 stdout，脚本不写任何文件）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tools/analyze_reference_trace.py \
  --root /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train-60/reports/round2-height-kaiser-20260912T155346Z \
  --reference-evidence /home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_train-60/docs/evidence/round2-height-kaiser-20260912T155346Z.json
```

实际运行 Python **3.14.7**，exit0；脚本 SHA256：`168865ebeda4046680c8fb1b4b89261fa16a1832d5d2b635bfb399cabeb5cdee`。输入仅上述五个 CSV 及历史证据 JSON，原策略权重与模型输出未读取。输出为逐 episode 时间窗、signed 六指标、全/稳态 XY dx/dy/net/path/max，以及 pooled mean/within variance/between variance；小 JSON 按共同时间窗和列语义紧凑保存这些主要结果，完整输出可由同一命令复现。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p test_reference_trace.py -v
```

**4 tests passed**：覆盖带符号值与 abs 值差异、不同样本数/不同均值回合的方差分解、terminal 行所属回合、每次 reset 后 warmup、跨 reset 位移反例，以及 nonfinite/命令/时间/reset/布尔/哈希/空行异常拒绝。标准库验证，不依赖 Torch、GPU 或模拟器。

写入小 JSON 后再次进行数值校验：90 个逐回合 signed 数值、所有保存的 XY 位移/路程/时间字段和 5 个 CSV 哈希均与分析器重算一致；45 个 std 与独立的 `statistics.pstdev` 比较，最大绝对差 **3.469447e-18**。15 个信号组的 `statistics.pvariance` 均满足总方差＝within＋between。旧 E 中全部完整回合的全程/稳态净位移精确复现；height bias 最大差 **8.881784e-16 mm**，仅为浮点求和和单位转换次序差异。

原 CSV 完整扫描与历史哈希复验、所有新指标复算完成。原 metadata 中的模型/源码版本哈希仍是旧运行记录，不宣称本次重新哈希模型。此次仅新增 CPU 分析脚本与对应测试，更新本文和证据 JSON；提交交给主 agent。
