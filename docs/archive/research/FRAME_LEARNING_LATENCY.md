# 学习率实验：独立 CPU 导出与延迟证据

`tools/run_learning_latency.py` 为已封存的 90 个训练单元生成独立 CPU 证据，供中立学习率 selector 审核。输入必须是 `run_learning_campaign.py` 的执行 manifest；不能用旧实验的检查点、导出图或延迟报告填补新网格。工具不训练、不启动仿真，也不选择最优网络。

执行之前，重新计算真实训练账本审核，要求全部 90 个单元同时具备 CP1200、58,982,400 个新采样端点，以及 50 场景 × 四个开发 seed 的完整评估。缺任一单元时返回 `waiting`；准备与只读审核仍显示完整 90 格的 `not_ready` 索引，缺失值不是零延迟。

## 冻结输入与独立输出

`prepare` 冻结执行 manifest 的文件 SHA 与 canonical SHA、producer/selector/执行器及辅助工具文件 SHA、CPU executable、Torch/NumPy/ONNX/ONNX Runtime 版本、CPU 名称与 affinity、协议和独占锁 inode。输出目录与原训练、prepared、source 及前驱目录独立，不能覆盖已有目录。

```text
output_root/
  manifest.json
  .latency.lock
  index.json
  summary.json
  cells/<rate>/<variant>/seed_<training_seed>/attempt_0000/
    request.json
    receipt.json
    bundle/{manifest.json,policy.pt,policy.onnx}
    benchmark.json
    parity.request.json
    parity.json
    export/{worker.process.json,worker.log,.bytecode-cache/}
    benchmark/{worker.process.json,worker.log,.bytecode-cache/}
    parity/{worker.process.json,worker.log,.bytecode-cache/}
```

每个阶段使用新的独占空 `PYTHONPYCACHEPREFIX`，同时设置 `PYTHONDONTWRITEBYTECODE=1`、`CUDA_VISIBLE_DEVICES=''`、`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。公开 CLI 命令保持原样，不修改冻结的 learner 或原目录缓存；前缀出现任何文件即拒绝继续。CPU 依赖与 parity probe 明确核对 CUDA 未初始化。

## 导出、计时与 raw mean 校验

按每个训练单元串行执行三个阶段：

1. 用冻结 source 的公开 `export` CLI 读取该单元 CP1200，导出真实 TorchScript 与 ONNX 图；原 exporter 对五组输入逐元素检查两者与 PyTorch raw mean 的一致性。
2. 用公开 `benchmark` CLI 执行 ONNX、单线程、精确 1000 次测量。冻结 source 的真实默认 warmup 为 **50 次**；CLI 没有 warmup 参数。不能把它写成 500 次，或通过改命令悄悄改变协议。
3. 运行 selector 公开定义的 CPU parity probe，使用种子 `9271`，构造 zeros、random windows、repeat-first windows 共五组输入，batch 为 1 和 7。测量 clipping 与 target mapping 之前 raw mean 的最大绝对值；它是独立实测量，不以动作限幅充当上界。

每个实际 worker 都记录完整命令、PID/start、终态、退出码、deadline/timeout 和 CPU 环境。完成 receipt 绑定配置、控制协议、source、检查点 SHA；封存导出 manifest、两种图文件、benchmark、parity request/process/output 的真实文件 SHA。producer 与 selector 都复核 closure、运行时版本和实际 CPU machine。所有 90 个单元使用同一 CPU identity，不能混合本机与 Kaiser 的计时。

延迟范围为：历史更新、mean 推理和目标映射；不包含传感器 I/O、传输、下位机电流环或力矩反馈控制。P99 ≤ 8 ms、max ≤ 10 ms、deadline misses = 0 是后续 selector 的开发门槛；本工具如实保留超过门槛的已完成测量，不用重试挑选更好计时。CPU evidence 完整也不等于硬件实时部署资格。

## 命令与异常处理

使用执行 manifest 所冻结的 Python 环境和同位置的工具文件：

```sh
python tools/run_learning_latency.py prepare \
  --campaign-manifest /absolute/execution/manifest.json \
  --output-root /absolute/new-latency \
  --worker-timeout-seconds 14400

python tools/run_learning_latency.py validate --manifest /absolute/new-latency/manifest.json
python tools/run_learning_latency.py audit --manifest /absolute/new-latency/manifest.json
python tools/run_learning_latency.py run --manifest /absolute/new-latency/manifest.json
```

`prepare` 仅运行 CPU 依赖身份探针；`audit` 只读取并重验已有证据。`run` 在原训练网格与四个开发 seed 都完成后执行。每阶段 deadline 在准备时统一冻结，超时只回收本控制器拥有的 PID/start 与进程组，保留真实失败回执。

存在原 live 或 unresolved handle 时返回 `waiting`，不杀死或重启它。已经失败或未封存的 job 保留同一个 attempt 并成为 `incomplete`；不复用半截输出，不新增有利重试。完成的 job 再次运行仅做只读复核。任何缺失或失败都保留在完整 90 格索引中，因此索引仍是 `not_ready`。

`index.json` 是 selector 的独立输入。selector 必须重新审核训练、开发与这些原始 CPU 文件，才能封存每架构的学习率选择；[确认执行器](FRAME_LEARNING_CONFIRMATION.md) 只消费原封存选择，最终架构/硬件认定仍需后续证据。本机小 tensor 检查点测试和 prepare smoke 是接口验证，不代表真实 90 个候选已经导出或测量。
