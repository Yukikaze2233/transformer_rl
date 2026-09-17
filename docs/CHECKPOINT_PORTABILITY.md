# Checkpoint 固定频率跨机兼容性

## 结论与范围

2026-09-17 修复 `src/transformer_rl/checkpoint.py` 的确定性预处理验证：仅对
`actor.time_frequencies` 接受 **float32 recipe 的相邻可表示值，再转换到存储 dtype**。
这是已有 loader 的跨平台数值表示误拒修复。文件结构没有变化，当前 checkpoint schema
仍为 4，已有 schema 1–3 的迁移规则继续适用。

post-stop 回收包的 **14/14** 真实 checkpoint 已用 `weights_only=True`、CPU 严格加载通过。
全部模型 state（包括 saved buffers）和 Adam state tensor 都与原文件逐字节相等；
metadata、update 一致，CPU RNG、文件 SHA 均未改变。
详见 [`evidence/checkpoint-portability.json`](evidence/checkpoint-portability.json)。

原 [`TRAINING_RECOVERY.md`](TRAINING_RECOVERY.md) 的 0/14 是修复前的历史结果。
本次验证对象为补传后的 12 个 final 977 和 supervised 最终停点 789/732，
不是原包 supervised 的中间停点 700/600。12 个 final 文件的 SHA 也已与 before-stop
原包副本重新比对，全部一致。

## 为什么原逐位重建检查会失败

频率由模型先执行以下 CPU float32 recipe 构造，再随模型 `.to(dtype=...)` 转换：

```python
torch.exp(
    -math.log(10000.0)
    * torch.arange(0, d_model, 2, dtype=torch.float32)
    / d_model
)
```

这与直接用 float64 计算数学公式不是同一个数值过程。乘法、除法及 `exp` 的结果包含
float32 舍入；float64 checkpoint 仍继承该 float32 recipe，不能用 double ULP 判定。

真实文件与本机重建的差异都在索引 3、23，均相差一个 float32 可表示间隔：

| 索引 | 保存值 | 本机重建值 |
|---:|---:|---:|
| 3 | 0.4216965138912201 | 0.4216964840888977 |
| 23 | 0.0013335214462131262 | 0.0013335213297978044 |

最大绝对差为 `2.9802322387695312e-8`。本机为 Torch `2.11.0+cu128`，
git `70d99e998b4955e0049d13a98d77ae1b14db1f45`，CPU dispatch 为 AVX2。
用户交接已验证：同一 Torch 版本的 Kaiser 在默认 AVX512 和强制
`ATEN_CPU_CAPABILITY=avx2` 下都与 checkpoint 精确一致，原始源码哈希也已核对。
因此，**不能把具体根因断言为 AVX2 与 AVX512 的区别**。此次未做新的远端实验。

## 精确定义：离散候选集合

令 `r[k]` 是本机按上述 recipe 得到的 float32 值，`C_D` 是到 checkpoint dtype `D`
的 PyTorch 转换。对于 `k > 0`，保存值必须逐元素属于：

```text
{ C_D(nextafter_f32(r[k], -inf)), C_D(r[k]), C_D(nextafter_f32(r[k], +inf)) }
```

- 首元素单独要求精确为 `1`，因为 `exp(0)=1` 不需要近似。
- 保留原有 key、shape、dtype、dense layout、finite 验证；所有频率必须正值、非递增。
  较低精度可能把相邻频率舍入为相同值，故允许等值；可观察到的乱序仍拒绝。
- **float32：**±1 相邻值可接受，距离本地 recipe 为 2 个及以上 representable steps 的值拒绝。
  `nextafter` 分别定义上下界，避免在二次幂边界错误假设两侧间距相同。
- **float64：**只接受上述三个 float32 值的精确 widening；即使某个 double 落在上下界之间，
  只要不是候选之一也拒绝。直接 double 重新生成频率也不是声明的 recipe。
- **float16 / bfloat16：**先在 float32 找候选，再 cast；不能以 half ULP 宽泛放行，
  也不能把已舍入的 half 值转回 float32 当参考。候选转换后可能只剩一个或两个不同值；
  在候选之外，哪怕仅差一个存储 dtype 的 representable step，也拒绝。
- **低精度可观测性边界：**若源 float32 曾被改动 2+ ULP，但 cast 后恰好与合法候选完全相同，
  文件没有保留该差异，无法据此拒绝。这里严格验证的是已存储值的候选集合成员资格，
  不声称能恢复转换前被丢弃的信息。

这是比连续 `[lower, upper]` 区间更窄的 bounded 检查，尤其能拒绝区间内非法 double 值。
未使用 `allclose` 或宽泛 `atol`。围绕任意高精度数学参考建立误差界需要同时界定原始
float32 中间舍入及不同 `exp` 实现的误差；当前证据不足以证明覆盖所有平台的统一界。
因此选用与实际构造契约一致、可直接审核的有限集合。它只保证上述兼容范围，
不保证任意 Torch 版本或任意两台机器都通过，也不宣称是跨平台统一的 canonical rounding。

容许的 1-ULP 差异本身不能区分平台舍入和有意修改；该检查不是文件真实性认证。
真实回收文件的完整性仍由可信来源的整文件 SHA-256 锚点证明。
本次没有新增 metadata buffer hash，也没有假定旧文件含有这样的 hash。

## 保存值、导出与其他契约

验证只读取 tensor。加载仍以 `load_state_dict(..., strict=True)` 恢复实际保存值，
不会把频率或其他 buffer 替换为本地重建值。重新保存也继续保存这些实际值。
`actor.frame_scale`（含由 `time_scale_s` 派生的项）以及其他 buffer 继续要求精确匹配。
模型、物理配置、参数、Adam 状态与 PPO behavior-consistency gate 均沿用原实现。

现有 `export_policy` 直接使用 loader 返回的 actor，所以 ONNX 使用已存储频率。
合成导出测试直接核对 ONNX initializer 的字节，确认合法 1-ULP 差异保留。
现有 sidecar 的公式字段描述构造 recipe，不代表每次导出会重新生成频率；
精确存储值以 checkpoint / ONNX 和它们绑定的 SHA 为准。本次不修改导出文件结构。

## CPU 验证与复核

新增测试位于 `tests/test_frequency_portability.py`，覆盖四种 dtype 的 float32 构造与转换、
双向 1-ULP 接受、2/3/100-step 篡改拒绝、低精度 cast 边界、非法 double、首元素篡改、
零/负数/非有限数/乱序拒绝、derived frame_scale 单步篡改拒绝、load/save 精确保留、
RNG、非空 Adam state 和合成 ONNX 常量。

相关回归（包括已有篡改、旧 schema 迁移和 PPO gate 测试）实际结果：
**435 passed, 23 skipped**；其中 14 项是默认关闭的真实文件审计，9 项需要 CUDA。
真实文件审计单独执行为 **14 passed**。没有运行全仓测试。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yukikaze/isaacsim60-venv/bin/python -m pytest -q -rs \
  tests/test_frequency_portability.py tests/test_checkpoint.py \
  tests/test_preprocessing_checkpoint.py tests/test_variants.py tests/test_export.py \
  tests/test_readout.py tests/test_initialization.py tests/test_behavior_consistency.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TRANSFORMER_RL_RECOVERY_ROOT='artifacts/recovered/architecture-16m-20260915T0540/recovery-post-stop-20260917T0900Z/extracted/study' \
  /home/yukikaze/isaacsim60-venv/bin/python -m pytest -q -s \
  tests/test_frequency_portability.py -k recovered_checkpoint_state_only
```

真实审计通过字节视图比对全部模型与 optimizer tensor，并拦截 forward、PPO update 和
Adam step，逐文件打印 `PORTABILITY_AUDIT` JSON。没有在真实模型上推理、更新参数或导出；
没有启动训练、仿真、GPU，也没有 SSH。合成测试允许梯度与 CPU ONNX 等价验证。

| 真实模型 | seeds | updates | 严格加载及全部 state 逐字节比对 |
|---|---|---|---|
| last_token_attention | 1011 / 1022 / 1033 | 977 / 977 / 977 | 3/3 通过 |
| time_attention | 1011 / 1022 / 1033 | 977 / 977 / 977 | 3/3 通过 |
| index_attention | 1011 / 1022 / 1033 | 977 / 977 / 977 | 3/3 通过 |
| gated_attention | 1011 / 1022 / 1033 | 977 / 977 / 977 | 3/3 通过 |
| supervised_attention | 1011 / 1022 | 789 / 732 | 2/2 通过 |

两个最终停点的实际 SHA 与用户提供的锚点完全匹配：

```text
789  2510b96c79f7c9e935edcbe1dc86461a882b2c544dc24ae4e792be2e39b17f23
732  2c6a28106dd88913d3a83e0fc2cb44d3d24f3c953fe6541fd8a1022fddc3f0fc
```
