# 学习率配对实验：执行与真实账本审核

`tools/run_learning_campaign.py` 执行十种架构 × 三个学习率 × 三个训练 seed 的 90 个独立训练单元。它使用已冻结的 learner，并把训练、开发评估和账本放在新的输出目录。准备目录的三个 `study/jobs` 保持空且只读；不能直接调用旧的 `frame_study.run_study` 代替这个执行器。

每个单元计划 1200 次完整更新。一次更新为 1024 个环境 × 48 个策略步，即 49,152 个新端点；每单元计划 58,982,400 个新端点，完整网格计划 5,308,416,000 个。以上是预算，不代表已经消费。策略为 100 Hz，物理与 PC 反馈 PD 为 200 Hz。

## 输入与输出

Kaiser 已完成 CPU 初态核验的输入为：

```text
prepared_root:
/home/kaiser/robot-rl-sim60/experiments/frame-learning-20261007-75efe2f-prep-717139f9/prepared

source_root:
/home/kaiser/robot-rl-sim60/experiments/frame-learning-20261007-75efe2f-prep-717139f9/source-75efe2f

prepared manifest canonical SHA256:
70a908932319c82574e2bf22cf8a71fbff32b96a6258447c28c2ffd9547844a8
```

新的 `output_root` 必须与准备目录、learner 源码、旧训练目录和三个前驱 campaign 独立。单元映射为：

```text
output_root/
  manifest.json
  .learning.lock
  .bytecode-cache/
  controller.json
  summary.json
  audit.json
  cells/rate_000/<variant>/seed_1101/
    training/attempt_0000/{request.json,worker.process.json,receipt.json,train/...}
    development/seed_701/attempt_0000/...
    development/seed_1701/attempt_0000/...
    development/seed_2701/attempt_0000/...
    development/seed_3701/attempt_0000/...
```

`rate_000`、`rate_001`、`rate_002` 的学习率取自准备 manifest；当前配方分别为 `1e-5`、`3e-5`、`1e-4`。执行器冻结自身和同目录的三个辅助控制器文件 SHA，冻结 prepared 完整文件库存、learner、配置、快照、初态、前驱定义和既有锁的 device/inode。部署四个工具文件后再执行 `prepare`，随后不能移动或修改它们。

准备时另封存同一 Python executable 的实际目标、文件 SHA、inode，以及 CPU 获取的 Python/Torch/NumPy/CUDA 版本、默认 dtype 和 deterministic flag。后续启动检查与 checkpoint metadata 必须一致；这不是对全部 Torch 底层共享库或硬件等价性的认证。worker 与 CPU 审核使用 `-B` 和独立空 `pycache_prefix`，避免读取冻结包目录中已有的 `.pyc`；旧输入缓存保留。专用前缀一旦出现文件即拒绝继续。

## 命令

用目标 Python 环境运行；`prepare` 会用冻结 learner 启动一个 CPU 子进程验证准备输入，不创建仿真或 PPO worker：

```sh
python tools/run_learning_campaign.py prepare \
  --prepared-root /absolute/prepared \
  --source-root /absolute/frozen-source \
  --prepared-manifest-sha256 PREPARED_CANONICAL_SHA256 \
  --output-root /absolute/new-campaign \
  --resource-lock /absolute/existing-resource/.run.lock \
  --study-lock /absolute/existing-transfer-study/.run.lock \
  --dependencies /absolute/dependencies.json \
  --device cuda:0 \
  --learner-max-seconds 604800 \
  --worker-timeout-seconds 605100

python tools/run_learning_campaign.py validate --manifest /absolute/new-campaign/manifest.json
python tools/run_learning_campaign.py audit --manifest /absolute/new-campaign/manifest.json
python tools/run_learning_campaign.py run --manifest /absolute/new-campaign/manifest.json
```

`validate` 只审核输入与执行定义；`audit` 另行审核真实输出。无训练结果时，`audit` 返回 `not_ready`，仍保留完整 90 个单元及各自四个开发 seed 的空值，不删除失败或缺失单元。

`dependencies.json` 必须包含 `transfer`、`curriculum`、`diagnostics` 三项。每项包含：

```json
{
  "definition": "/absolute/campaign.json",
  "sha256": "the immutable definition's canonical SHA256",
  "summary": "/absolute/summary.json",
  "controller": {"pid": 12345, "start": "verified Linux process start ticks"}
}
```

前两项定义为 `campaign.json`，其摘要绑定 `campaign_sha256`；diagnostics 定义为 `manifest.json`，摘要绑定 `manifest_sha256`。PID/start 来自实际 `/proc` 核验，不能用当前评估 worker 替代 controller。已结束 controller 的原始 handle 可保留，并以真实完成回执及 handle 已消失为证据。

摘要在排队和执行期间会变化，因此不冻结准备时的摘要 SHA。执行器每次读取状态记录当前摘要 SHA，要求三个前驱实际 `completed`、完整结果/样本/检查点/评估文件可核验，且指定 controller 与 owned worker 都不再占用资源，然后按共享 resource 锁、transfer study 锁的顺序获取独占锁并复核。观察等待超时只返回 `waiting`，不会重启、抢占或宣称旧任务停止。

## Fresh 与 resume

首个 attempt 显式传递 `--expected-initial-model-sha256`，要求与 prepared 中对应架构、训练 seed 的完整模型 SHA 一致，包括固定 buffer；`resume`、`initialize_from`、`restore_learning_from` 均为空，课程时钟为零，anchors 为空，retention coefficient 为零。

后续只允许从上一个完整封存端点执行 exact `--resume`，不再次传 fresh guard。request 固定 parent 检查点 SHA、配置、learner、factory、seed、剩余预算与累计课程时钟。审核核对实际 worker 的完整命令、PID/start 与终态，逐字段核对 `run.json`，并由冻结 learner 的 CPU weights-only loader 校验完整模型、Adam 和 RNG；核对 sidecar、metadata、运行 scene group 数量、累计样本，以及 Adam 超参数和逐参数 optimizer step 的连续性。环境 episode/history 在 resume 时重置，学习状态保留。

## 样本与异常计费

逐条优化记录必须具有连续 update、`batch_samples=49152`、`collection.vector_steps=48`、`collection.transitions=49152`。实际 optimizer step 必须为正且不超过 160；PPO 重复端点数为 `optimizer_steps × 1536`，它与新采样端点数分开记录。首步/最终 KL 与 early-stop flag 必须可核验。

默认每次 learner wall-clock cap 为七天，worker deadline 多 300 秒；可在准备时统一设置并冻结，不能在同一网格中临时延长某个候选。时间 cap 无法保证所有学习率与架构都完成样本预算，实际完成程度由账本证明。

SIGTERM 或时间预算可能使原 learner 把非空短 rollout 交给 PPO 并增加 update，因此不能只看 update 编号。短 rollout、optimizer 异常、未封存崩溃或零更新端点均使该单元成为 `incomplete`：整个 attempt reservation 被扣账，不退款、不换 seed、不重新 fresh；可证明的实际消费前缀另列，不能把扣账数当成真实采样数。已完成的前序 attempt 与失败 attempt 的已知前缀分别保留，并给出已知累计消费；未知尾部不声称已测量。

只有成功保存的完整 rollout 停点允许 resume 剩余预算。最终必须同时证明累计 1200 次成功更新、检查点 update 1200、新采样端点 58,982,400。任何一个单元不足，完整网格仍为 `not_ready`。

## 开发评估与后续选择

每个完成的单元无论控制表现如何，都执行全部 50 个固定场景 × 四个开发评估 seed `701/1701/2701/3701`。每场景 8 环境、4001 步，reset 后去掉前 200 步，稳态片段至少 200 个样本；报告与 trace 均绑定该单元 CP1200 的 SHA，评估 request 与实际 worker 记录另行封存 SHA 并审核。评估失败保留原 attempt，不能用有利重试替代它。

这四个 seed 都是开发数据。学习执行器不承担 export、CPU latency、学习率选择或确认评估；`development_complete` 也不等于找到了最优架构。独立 [CPU producer](FRAME_LEARNING_LATENCY.md) 补齐每单元的实际 latency 证据后，[中立 selector](FRAME_LEARNING_RATE_SELECTION.md) 才能封存各架构的开发学习率选择。[确认执行器](FRAME_LEARNING_CONFIRMATION.md) 随后只执行原选模型的 `11701/12701` 留出噪声流，不代表新初态或新扰动域，也不能据确认结果重新选择。这些模块具有执行能力，不代表真实研究已完成；硬件资格仍需独立证据。
