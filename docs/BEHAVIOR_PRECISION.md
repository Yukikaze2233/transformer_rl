# 行为统计精度：Kaiser 固定历史复现证据

## 结论

**已复现 A 的 `std=0.1, seed=11` 首轮行为检查报错；对本次保存的 rollout，证据支持 batch shape 导致的 FP32 均值计算差异，经 Gaussian log-prob 放大后触发近零容差，而非 history/alias/权重版本漂移。**

原 gate 仍拒绝，报错文本中的 chunk 最大差仍为 `1.23978e-05`。没有调用 optimizer update，没有修改核心 gate/PPO 或 `rtol=atol=1e-5`。B 的 baseline seed44 update52 未重跑，不能把本次结论直接当作 B 的逐样本根因证明，也不能将历史失败训练改记为成功。

精简机器证据：[evidence/behavior-precision.json](evidence/behavior-precision.json)。本页及该 JSON 是本次仅新增的仓库文件。

## 复现范围与数据所有权

- 2026-09-14，使用冻结快照 `/home/kaiser/robot-rl-sim60/transformer-sensitivity-20260914T071345Z`；`source-freeze.json` 中所有文件 SHA256 在采样和离线重放前核验通过。
- 直接加载原 `stage-a/configs/std_01.json`，SHA256 为 `3d0a4abeb127a1b9a68823ea280c0517af0682a99d1a7e30e5832852f6709288`。time-attention Transformer，history16、d64、2 layers、4 heads、FFN128；std0.1、mean scale1；原 task/backend、seed11、action clip100。
- 初始化顺序遵循原 CLI：设置 Python/Torch seed → 环境构造 → 模型及 trainer 构造 → reset(seed11) → collect。仅 **1 次真实 collect：32 steps × 512 env = 16384 transitions，0 optimizer steps**。
- 在 `actor.act` 前独立复制五项 HistoryBatch 输入到 CPU，在返回后复制 raw action、old mean/std/logp；包装仅采证，不改变原 forward/sample。随后与 time-major rollout 全字段比较。
- `payload.pt` 保存原始 HistoryBatch 字段、完整 PPOBatch tensor 字段、32 个调用现场输入/输出、采样前后 model state、model/PPO/environment config 及 backend 信息。仅 primitive containers/tensors；已用 `torch.load(..., map_location="cpu", weights_only=True)` 读取并递归检查，无需自定义类反序列化。

| 完整性检查 | 结果 |
|---|---|
| act 现场 history 与 rollout：frames/times/valid/command/now | 全部逐位相等 |
| act 输出与 rollout：raw_action/old_mean/old_std/old_log_prob | 全部逐位相等 |
| collect 前后 state_dict（参数及 buffer） | 全部逐位相等 |
| 原 collector 参数 identity/version 与 eval-mode guard | collect 正常完成，未触发 |
| 离线重放结束后的 state_dict、history | 与原 payload 逐位相等 |
| 原存储 moments + raw action，在 GPU FP32 重算 logp | 最大差 **0** |

### 执行收据中的探针错误

采样进程在 `payload.pt` 成功保存并 `weights_only` 读回后，构造收据时误把 `stat().st_size` 属性当作函数调用，真实 **exit=1**。`env.close()` 和 `examples._isaaclab_process.main` 的 SDK `finally` 关闭路径均执行。原脚本、错误日志和 payload 原封保留。

随后新 tmux 中的独立脚本从该 payload 恢复模型/历史，显式恢复已记录的 backend 设置并核验一致，完成 GPU 重放，**exit=0**；离线汇总 **exit=0**。没有第二次采样。原训练 gate 的拒绝作为诊断结果被记录，重放进程 exit0 只表示采证完成。

## Backend 与重放方法

GPU 为 RTX4090；Torch `2.11.0+cu128`、CUDA12.8、cuDNN91900。`matmul.allow_tf32=false`、`cudnn.allow_tf32=true`、`float32_matmul_precision="highest"`，CUDA autocast=false、deterministic algorithms=false、cuDNN benchmark=true。未为了获得不同结果调整这些设置。

模型/frames/command/action/moments/logp 为 FP32，times/now 为 FP64，valid 为 bool。模型始终 eval；分别检查 no_grad、enable_grad 和 inference_mode。

重放使用同一固定 time-major history、同一 raw action、同一模型权重：

1. 512 顺序分块，重建每个原 collect 调用的完整 batch。
2. 4096 顺序分块，与原 `_check_old_log_prob(batch, 4)` 的索引分块一致。
3. 再次 512 重放，检查是否存在前次 forward 改变状态。
4. 4096 的 grad/inference 对照；256/1024 的 batch-shape 对照。
5. 相同保存权重提升至 FP64，GPU batch512 forward，作为离线数值参考；analytic KL 和 Gaussian 差值分解使用 FP64。

## 主结果：真正的 mean/std 变化与 logp 变化

下表相对于存储行为统计，覆盖全部 16384 条。mean 差按 98304 个 action 分量统计，KL 每条对 6 个 action 求和后再聚合。

| 重放 | max abs Δmean | RMS Δmean | max abs Δstd | max abs Δlogp | mean KL64 | logp 超容差条数 |
|---|---:|---:|---:|---:|---:|---:|
| FP32，512，no_grad | 0 | 0 | 0 | 0 | 0 | 0 |
| FP32，4096，no_grad | 5.066395e-7 | 1.066113e-7 | 0 | 1.680851e-5 | 3.409788e-12 | **1** |
| FP32，再次512 | 0 | 0 | 0 | 0 | 0 | 0 |
| FP32，4096，grad / inference | 与4096 no_grad逐位相同 | 同左 | 0 | 同上 | 同上 | 1 |
| FP32，256 | 5.960464e-7 | 1.200602e-7 | 0 | 1.525879e-5 | 4.324337e-12 | 0 |
| FP32，1024 | 3.576279e-7 | 5.941312e-8 | 0 | 9.298325e-6 | 1.058976e-12 | 0 |

4096 重放的最大单条 KL64 为 **2.204903e-11**；均值差对 old std 归一化后的 RMS 约 **1.066113e-6**。这都是同权重的数值输出差，不能称为一次学习造成的策略移动。

4096 的差值分布：

| 绝对差 | p50 | p90 | p99 | p99.9 | 非零数 |
|---|---:|---:|---:|---:|---:|
| Δmean | 5.960464e-8 | 1.788139e-7 | 2.682209e-7 | 3.576279e-7 | 78709 / 98304 |
| Δlogp | 1.430511e-6 | 4.291534e-6 | 7.629395e-6 | 1.068318e-5 | 14924 / 16384 |

以 old mean 的局部 `nextafter` 间距定义 ULP，绝对差的中位数是 2 ULP、p90=18、p99=160，最大98304。接近零的输出局部 ULP 很小，**不能笼统声称所有分量只差一两个 ULP**；应同时报告上表的绝对量与按 std 归一化的量。

## 唯一失败样本与近零容差

time-major index **171**，即第0步、env171（均为零基索引）：

- old logp：`-0.06449198722839355`
- new logp：`-0.06448066234588623`
- Δlogp：`+1.1324882507324219e-5`
- 原阈值 `1e-5 + 1e-5*abs(old_logp)`：`1.0644919872283937e-5`
- 差/阈值：**1.0638767**，超过约6.39%。
- 六维实际 Δmean：`[+1.192093e-7, -1.788139e-7, +2.682209e-7, -5.960464e-8, +1.341105e-7, +1.192093e-7]`。
- std 每维均为 `0.10000000149011612`，Δstd每维均为0。
- 单条 analytic KL64：`7.693845331357874e-12`。

完整 raw action、old/new mean 在本地 JSON 和远端原始 tensor 中保存。

**报错中的 `1.23978e-05` 是第一个被拒绝的4096 chunk内的最大绝对差，不是失败样本自身差，也不是整个 rollout 的最大差。** 原实现一旦某个 chunk 的 allclose 失败便报告该 chunk 的 max；全 rollout 的最大值为 `1.680851e-5`，对应样本仍可因相对容差项更大而通过。

| abs(old logp) 区间 | 样本数 | 超容差数 |
|---|---:|---:|
| [0,0.01) | 3 | 0 |
| [0.01,0.1) | 25 | **1** |
| [0.1,1) | 261 | 0 |
| [1,5) | 5652 | 0 |
| [5,100) | 10443 | 0 |

因而最大绝对 logp 差本身不能预测检查结果；本次失败落在相对容差项接近消失的区域。256 重放的最大差也大于1e-5但全部通过，进一步体现这一点。

## 用 FP64 分解验证放大机制

对相同 std、固定 raw action，令 `δμ = μ_new − μ_old`，精确代数关系为：

```text
Δlogp_moments = Σ_a ((action − μ_old) * δμ − 0.5 * δμ²) / σ²
KL(old || new) = 0.5 * Σ_a (δμ / σ)²
```

这不是从误差幅度猜测：将 **实测 old/new moments** 提升到 FP64 后重算 Gaussian logp，并与上述 Δmean 公式逐条比较，最大残差仅 **3.159705e-15**。

- 全 rollout，mean-shift 公式的 Δlogp RMS：`2.582358e-6`；实际 FP32 Δlogp RMS：`2.599684e-6`。
- FP32 实测差减去 FP64 moments 差，残差 RMS 为 `3.324098e-7`，最大 `1.690633e-6`，反映 logp 求值/求和本身的 FP32 舍入。
- 唯一失败样本，mean-shift 公式预测 `+1.1687995123136083e-5`，实际 `+1.1324882507324219e-5`，剩余约 `−3.631126e-7`。实测均值变化足以解释该次放大。
- 仅用原存储 moments 在 GPU FP32 重算 logp，差为0；提升原 moments 到 FP64 后与存储 FP32 logp 的最大差为 `1.681330e-6`，说明即便 moments 完全固定，改变 logp 算术精度也不会严格重建原 FP32 数字。

FP64 GPU forward 参考中，512/4096 两种 FP32 mean 相对参考的 RMS 分别为 `1.073046e-7` / `1.064066e-7`，最大差分别为 `6.723008e-7` / `5.141750e-7`，均在相近范围。FP64 forward 的 std 还因 `exp(log_std)` 精度改变产生 `4.687660e-9` 差；它不是新的行为真值，也不宜拿来直接替换存储 logp。此参考没有 CPU 网络 forward；CPU FP64 用于离线 moments/KL 分解。

## 归因边界与检查协议交接

本次归因依据是受控改变 batch shape、两次512逐位重建、4096不同 grad mode 逐位相同、输入/输出所有权核验、权重前后相等，以及逐样本 FP64 代数闭合。**对这份 payload，可以排除可观测的 history/alias/参数漂移作为该拒绝的原因。** 未做 CUDA kernel 级 profiling，不能进一步指认具体 GEMM/attention kernel。

给主 agent 的协议依据：行为数据自洽性、同采样 shape 的重建一致性、跨 shape 的数值误差、按 std 归一化的分布差异，应有各自清楚的语义。本次材料足以作为协议设计和回归工件，但不提供一个新的通用 tolerance 数字，也没有将任意 gate 改绿。正式修改仍需验证对真实权重/history/raw_action 破坏的检出能力。

旧 A 失败当时没有保存 rollout，因此“报错数字一致”不等于已证明两次仿真所有 tensor 相同。B 的 update52 同样需要自己的 payload 才能完成同等级归因。

## 工件、资源与结束状态

独立远端目录：`/home/kaiser/robot-rl-sim60/behavior-precision-20260914`。

| 工件 | SHA256 |
|---|---|
| `payload.pt`，74329203 bytes | `9add976532474297239e560dd42c8f6d92434641fdbcbd8466ed28c2b8155d2d` |
| `replays.pt`，所有重放输出 | `9cbc8015d64acef0b5b656a3e72b270b23c6737dfc51c46d0258cd411c933a83` |
| `analysis.json`，完整分布、失败/极值样本及相等性检查 | `657c1b1791e1720ae54fef7a21f809e08447a921b94ee385c466f273e4f66ff2` |

该目录还保存原采样脚本、恢复重放脚本、汇总脚本、worker/replay 日志、各 exit 文件和采样资源日志。所有作业均由新 tmux + 有界 timeout 执行：采样 worker480s/session530s，恢复重放180s，汇总30s；实际 GPU重放分析约2.92s。首次采样会话08:49 UTC启动，08:51左右退出，08:53完成恢复与汇总，执行窗口小于10分钟。

采样期间46次资源采样：WSL available 最低 **10466 MiB**，GPU全机占用峰值 **9111 MiB**，包含原有 Windows 进程。离线 GPU 重放未单独采资源峰值，不能将前述数值称为全任务峰值。全部本次 tmux/worker 已结束；没有触碰其他进程，没有 commit/push。

## 后续修复：优化前的行为一致性契约

本节记录采证之后的代码修复。**以上原 probe 的失败收据、原 gate 拒绝结果和 JSON 均保持原样**；前文“没有修改核心 gate/PPO”的表述对应当时的采证快照。本节的本地单元验证不构成修复版在完整 Kaiser payload 上的 GPU 验收，也不补足 B seed44 update52 的独立归因。

修复位置：`src/transformer_rl/ppo.py` 的 `_check_old_log_prob`。作用域仍是每次 `update` 的 **before-any-optimizer scan**：按原顺序、原 chunk 划分扫描全部 endpoint（含 remainder），所有 chunk 通过后才进入优化循环。

### 两个契约、四项比较

1. **当前策略与存储行为分布的 moments 接近**：先比较 `current_mean` 与 `old_mean`，再比较 `current_std` 与 `old_std`。保留 `old_mean mismatch` / `old_std mismatch` 错误消息，继续检出超出原参数容差的策略输出漂移。
2. **每份 logp 与自己的 moments、同一 raw action 自洽**：

   ```text
   reconstructed_old = Normal(old_mean, old_std).log_prob(raw_action).sum(-1)
   reconstructed_current = Normal(current_mean, current_std).log_prob(raw_action).sum(-1)

   allclose(reconstructed_old, stored_old_log_prob, rtol=1e-5, atol=1e-5)
   allclose(reconstructed_current, evaluation.log_prob, rtol=1e-5, atol=1e-5)
   ```

   分别报告 `old_log_prob mismatch` / `current_log_prob mismatch`。两次重算各自使用被校验 logp 的 dtype，将对应 mean/std/raw action 转到该 dtype；当前真实采证路径因此仍使用 FP32 `Normal` 算术。FP64 仅用于数值参考与测试中的解析闭合，不代替原 FP32 密度。重算结果也需有限。

**四项比较全部保持 `rtol=1e-5, atol=1e-5`。** 不再对两个近似 moments 所产生的跨 batch-shape logp 施加相同的直接比较：前文公式已经证明，`δμ` 经 `(action−μ)/σ²` 放大，不能把该 logp 差单独当作状态或权重改变的证据。参数比较与密度自洽分别保护分布一致性和数据/求值完整性，且与现有对角 Gaussian 解析 KL 假设一致。

### 优化语义与本地验证

PPO ratio 仍为 `exp(evaluation.log_prob - minibatch.old_log_prob)`，分母始终引用原存储行为密度；检查不重写 old moments/logp。优化循环中的 loss、KL early-stop gate、actor 调用次数和顺序、随机 minibatch 排列保持原实现。检查本身在 `no_grad` 下重算密度，不采样。

新增 `tests/test_behavior_consistency.py`，复用现有可微 Synthetic Gaussian actor，并用真实 `Normal.log_prob` 构造反例：

- FP32/FP64 下，受控 batch-size mean 差 `5e-7` 在原 moments 容差内，`std≈0.1`、old logp 近零，cross-logp 超出旧容差；旧检查拒绝、修复检查接受，FP64 mean-shift 公式闭合。
- 新接受的数据实际执行一次更新，核对原存储分母对应的 actor loss/梯度，以及输入 batch 全字段未改写。
- remainder 中的 mean/std 大漂移、有效 history 改变、stored logp 篡改、正确 moments 配错误 current logp、stored/current/both 使用 issued action、raw 被 issued 替换，均在任何 optimizer step 前拒绝，参数、Adam 状态及 RNG 未推进。
- 在原检查合法的数据上，与冻结的旧检查执行两轮更新，覆盖已初始化 Adam、正常多 epoch/remainder、KL early stop、diagnostics 开关；梯度、模型、Adam、metrics、actor 调用轨迹和 Torch/Python RNG 精确一致。

本地验证命令：

```bash
/home/yukikaze/isaacsim60-venv/bin/python -m pytest -q tests/test_behavior_consistency.py tests/test_ppo.py tests/test_auxiliary.py tests/test_optimization_diagnostics.py
```

结果：**95 passed**，包含新增16项测试及原 padding/auxiliary 回归；原 `test_ppo.py`、`test_auxiliary.py` 无需修改。本轮验证使用本地 synthetic tensors，没有 SSH 重放或训练环境运行。完整真实 payload 的修复版离线 GPU 验证可据此接口继续执行；历史失败训练不追溯改记成功。

## 修复版 Kaiser 原 payload 离线 GPU 验收

2026-09-14，独立新快照 `/home/kaiser/robot-rl-sim60/transformer-sensitivity-recheck-20260914T0920` 完成真实原 payload 的离线验收，进程 **exit=0**。使用本页保存的 `payload.pt`，SHA256 仍为 `9add976532474297239e560dd42c8f6d92434641fdbcbd8466ed28c2b8155d2d`；保留其 FP32 数据及原 backend 设置，不进行环境采样或 optimizer update。

- 新 package SHA256：`f182eaeb41a9f4a6bed0c6aff6aca9be01f9364cc4c707b7848a071f4516c209`。
- 新 `ppo.py` SHA256：`8994d71c41ebb51feeb568d9a8830c36ca73829a89818836c7dbde88c6e238a2`。
- 旧 `ppo.py` SHA256：`76d01590a934dfa874344fe3fd3bc7ded03ba90c9599935c66fecc1a4f091b9b`。
- 对照旧 snapshot 的 `src/examples/configs` 冻结清单，文件内容差异仅为 `src/transformer_rl/ppo.py`。

| 验收输入 / guard | 预期 | GPU 实际结果 |
|---|---|---|
| 原16384条，新 guard，4×4096 | 通过 | 通过；全部四个 chunk 完成 |
| 原16384条，直接加载冻结旧 guard | 拒绝 | 原报错 `max absolute error 1.23978e-05` |
| 最后 endpoint 的 old_log_prob 加0.1 | 拒绝 | `old_log_prob mismatch`，误差约0.0999999 |
| 最后 endpoint 的 old_mean 第0维加0.01 | 拒绝 | `old_mean mismatch`，误差约0.0100001 |
| 最后 endpoint 的 old_std 第0维加0.01 | 拒绝 | `old_std mismatch`，误差约0.01 |
| 最后 endpoint 的 raw_action 第0维加0.2 | 拒绝 | `old_log_prob mismatch`，误差约2.27709 |
| 正确 current moments 配 logp+0.1 | 拒绝 | `current_log_prob mismatch`，误差约0.1 |
| 恢复原输入后再次运行新 guard | 通过 | 通过 |

反例均通过克隆构造，未改原 tensor；最后 endpoint 的篡改同时覆盖末 chunk 扫描。验收后核对 model state、原 PPOBatch 全字段、HistoryBatch 全字段逐位不变，Python/Torch CPU/CUDA RNG 不变，Adam state 为空，原 payload SHA256 不变。容差仍为 `rtol=1e-5, atol=1e-5`。

原始验收脚本、八项实际结果及完整性检查保留于新快照的 `recheck_offline.py`、`offline-acceptance.json`、`offline-runner.log`、`offline-execution-receipt.json`。对应新 source 的训练补验与完整配对复验另见 [KAISER_SENSITIVITY_RECHECK.md](KAISER_SENSITIVITY_RECHECK.md) 及 [evidence/sensitivity-recheck.json](evidence/sensitivity-recheck.json)；本节验收不改写旧训练失败事实。
