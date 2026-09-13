# Checkpoint 与策略导出

本模块属于独立包 `transformer_rl`。checkpoint 保存模型与优化器；导出产物是确定性的 actor raw mean。合成输入对齐测试仅验证实现与数值一致性，不证明控制效果或真实硬件适用性。

## Checkpoint API

```python
from transformer_rl.checkpoint import load_checkpoint, save_checkpoint

report = save_checkpoint(
    "artifacts/checkpoint.pt", model, trainer,
    update=42, metadata={"experiment": "baseline", "seed": 2026},
)
model, trainer, update, metadata = load_checkpoint(
    "artifacts/checkpoint.pt", device="cpu",
)
```

目标父目录须已存在。`report` 包含 `path`、`sha256`、`update`；`update` 是调用方维护的累计更新次数，不由优化器 step 推算。返回的 trainer 绑定返回的 model，恢复后可继续调用 `trainer.update(batch)`。

### 格式与校验

- 格式标识为 `transformer_rl.checkpoint`，内部 `schema_version=1`，文件名不要求带版本号。
- 保存完整 `ModelConfig`、`PPOConfig`、模型 dtype、actor/critic weights 与 buffers、Adam 状态、累计 update 和 JSON metadata。
- metadata 必须为字符串键 JSON object；嵌套值仅接受 object、list、string、有限 number、boolean、null。tuple、tensor、非字符串键及循环引用被拒绝。
- 读取显式使用 `torch.load(..., weights_only=True, map_location="cpu")`，不回退到非受限 pickle。
- 严格检查顶层和配置 keys、配置类型/约束、模型 state keys/shape/dtype/finite，以及 Adam 参数 ID/顺序、state keys/shape/dtype/finite、非负二阶矩和整数 step。
- 支持 float32、float64、float16、bfloat16 checkpoint。各参数和 buffer 的 dtype 必须与该模型 dtype 一致；导出目前仅接受 float32。
- 优化器须为 PPOTrainer 使用的单参数组 ordinary Adam，覆盖完整模型参数。保存实际 group 学习率，可与配置初始学习率不同；不保存外部 scheduler 对象。支持普通 Adam/AMSGrad 状态；拒绝 capturable、differentiable、fused 或 decoupled-weight-decay 模式。
- 加载先把模型放到目标 device，再构造 optimizer，避免设备转换更换 Parameter 后 optimizer 仍引用旧参数。由 Adam 自己恢复 moment/counter 的设备位置。
- 格式/数据不合法抛出 `ValueError`，API 对象类型不符可抛出 `TypeError`，已存在目标抛出 `FileExistsError`；文件系统错误保留原异常。

模型初始化所消耗的 CPU RNG 被局部保存和恢复，保存/加载不推进调用方全局 CPU RNG。checkpoint 不保存全局随机流、环境、通信队列、collector history 或 PID 状态。恢复采集须从新 episode 开始；不承诺 bitwise 续训或外部控制状态的精确续接。

## ONNX API

```python
from transformer_rl.export import export_policy

report = export_policy("artifacts/checkpoint.pt", "artifacts/policy.onnx")
print(report["path"], report["sidecar_path"], report["sha256"])
```

生成 `policy.onnx` 与 `policy.onnx.json`。返回值包括这两个路径、ONNX/sidecar/checkpoint 的 SHA-256 和 CPU ORT validation 摘要。sidecar 绑定实际读取的 checkpoint 字节，而不是导出结束后再次读取可能已经变化的源文件。

依赖：PyTorch >= 2.7、ONNX、ONNX Runtime；ONNX/ORT 仅在导出调用时使用。本轮验证环境为 PyTorch 2.11 CPU。使用 opset 17 与 tensor-only `actor.forward_tensors`；legacy exporter 的弃用 warning 会记录在 sidecar，不视为导出失败。tensor-dependent `TracerWarning` 会拒绝发布。

### 输入和输出

`B >= 1` 为动态 batch；`L = ModelConfig.history_length` 静态固定。默认 `F=30`、`C=3`、`A=6`。

| 名称 | dtype | shape | 含义 |
|---|---|---|---|
| `frames` | float32 | `[B,L,F]` | 从旧到新的历史帧，左侧 padding |
| `times` | float64 | `[B,L]` | 各事件观测可用时间，秒 |
| `valid` | bool | `[B,L]` | 有效完整历史帧标志 |
| `command` | float32 | `[B,C]` | 当前 query 命令；历史命令不被重写 |
| `now` | float64 | `[B]` | 当前 query 时间，秒，与 times 同时钟 |
| 输出 `mean` | float32 | `[B,A]` | 确定性 raw Gaussian mean |

默认 frame 布局，区间为左闭右开：

| 区间 | 特征 | 单位/语义 |
|---|---|---|
| `[0,16)` | proprio | 调用方预先约定的缩放 |
| `[16,19)` | historical command | 当时事件命令 |
| `[19,25)` | previous issued action | 上一条实际提交的指令；不是 raw sample 或 applied target |
| `[25,27)` | sensor age | 秒；未知值编码 0，必须配合 known=false |
| `[27,29)` | sensor age known | float32 的 0/1 标志 |
| `[29,30)` | policy dt | 秒；reset 首帧可为 0 |

sidecar 根据实际 ModelConfig 生成布局，维度改变时不硬编码默认偏移。传感器 age 与 policy dt 在网络内除以 `time_scale_s`。真实历史 age 先计算 `float64(now - times)`，然后 cast 到 float32，再使用固定 Fourier 编码；生产端应直接提供 float64 timestamp，不能先转 float32 再期望恢复 uptime 下的毫秒精度。

有效 timestamp 须严格递增且 `<= now`。所有有意义的输入须有限；无效 padding 的 frame/time 可含 NaN/Inf，不参与模型计算。空历史合法；partial reset 由调用方清理对应环境的 history，图内部没有可变 KV/cache。导出路径不执行 Python 数据校验，调用方须在 ONNX 调用前满足这些约束。

输出没有 tanh、clipping、动作抽样或 actuator conversion；图不包含 critic、optimizer 或 Gaussian log_std。外部命令限幅/缩放由调用方负责，随后写回 history 的应为 **issued action**。该导出不包含 applied action、传输 FIFO、PID、执行器响应或真实硬件时序模型。

### 发布前验证

实际导出图通过 ONNX checker 和 `CPUExecutionProvider` 核对：

- 五输入的精确名称、dtype，以及固定 L/动态 B；单一 mean 输出。
- B=1/2/3、完整历史、左 padding、partial reset、带 NaN/Inf 的 padding。
- 大 uptime、当前 command 改变、不规则事件间隔。
- 所有输出有限，逐元素满足 `rtol=1e-5, atol=1e-6` 的 PyTorch/ORT 一致性。

每种条件的最大绝对误差及运行库版本记录在 sidecar。验证不采样环境、不启动训练或仿真。CUDA checkpoint 恢复另有条件测试，CPU 环境会跳过；本轮 ORT 验证仅覆盖 CPU。

## 文件发布语义

checkpoint、ONNX 和 sidecar 都拒绝覆盖既存文件、目录或 dangling symlink。写入使用目标目录内临时文件，完成 flush/fsync 后通过 exclusive hard link 发布；目标目录所在文件系统须支持 hard link。并发写入同一路径时不会覆盖胜出者。

ONNX 与 sidecar 全部在验证成功后暂存，sidecar 最后发布作为完整产物标记；捕获到发布错误时回滚本次新增链接并清理临时文件，保留原有/并发产物。两个目录项不能保证掉电或 SIGKILL 下的跨文件原子事务：读取端应要求二者同时存在并核对 SHA，孤立 ONNX 不视为完整导出。已有孤立文件也不会被自动覆盖。

## 定向测试

```bash
PYTHONPATH=src /tmp/opencode/v40-sim60-ci/bin/python -m pytest \
  tests/test_checkpoint.py tests/test_export.py -q -rs
```

测试使用合成模型参数和 fake gradients 验证 Adam 状态往返及下一次更新的一致性，覆盖损坏格式、非法配置/张量/metadata、全局 RNG 保持、拒绝覆盖、staging 失败和 sidecar 发布竞争。所有测试均为实现验证。
