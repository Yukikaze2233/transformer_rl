# Kaiser：暂停后续训

## 状态

**更新：续训队列已于2026-09-18 02:17:58 +08:00全部完成并回收。**
完整收据见 [首轮完成与回收](KAISER_ARCHITECTURE_COMPLETE.md)。以下为恢复启动时的历史记录。

用户于2026-09-17授权继续训练。本批于 **2026-09-17 18:06:11 +08:00**
在独立 tmux `transformer-resume-20260917` 启动。

18:10:07的真实观察：`supervised_attention/seed_1011` 已从update789推进到838，
新增49次更新、802,816条采样，执行392/392个计划优化器步，数值指标均有限。
新update800检查点通过原冻结源码的严格CPU加载；44份Adam参数状态的step均比
update789增加88，与这11次更新一致。环境identity与原检查点相同。

机器可读收据见 [architecture-resume-start.json](evidence/architecture-resume-start.json)。
这些是启动观察，不代表后续队列已经完成。

## 恢复范围与预算

原任务中12项完整训练及84份评估经过再次校验，沿用其结果。
本次剩余9项按原variant-major顺序串行运行，每项训练后完成7场景评估：

| 变体 / seed | 起点 | 本段更新次数 | 最终累计update |
| --- | ---: | ---: | ---: |
| supervised_attention / 1011 | 789 | 188 | 977 |
| supervised_attention / 1022 | 732 | 245 | 977 |
| supervised_attention / 1033 | 新初始化 | 977 | 977 |
| history_mlp / 1011、1022、1033 | 各自新初始化 | 各977 | 各977 |
| history_gru / 1011、1022、1033 | 各自新初始化 | 各977 | 各977 |

原配置、512环境、32步rollout、PPO配方、动作裁剪、任务资产及评估协议保持冻结。
源码仍为 `80a45b40c9afc7e6e3356ccbb6fa15f8933a3970`，
plan SHA为 `7a37cad280f7676b4103c0ec883bf1c8ee16e41adf72a9ae9c843f03ea90ed57`。
prepare阶段核验原plan、全部冻结配置/源码文件、两个停止检查点及已完成评估；
调度器在各job前后再次核验冻结文件。

**这两项是分段续训，不是逐位连续复现。** 恢复模型和Adam状态，但环境、历史窗口
重新初始化，随机数流不恢复。CLI的 `--updates` 表示本段新增更新次数；原始指标和
completion中的采样计数也是分段计数。新result同时保存本段和两段累计采样数。
原1011段还采集过13,824条未进入完整更新的样本，不能将总采样数简单等同于
`977 × 512 × 32`，也不能覆盖旧日志来掩盖这些样本。

## 运行位置与资源

- 原study：`/home/kaiser/robot-rl-sim60/architecture-16m-20260915T0540`
- 新root：`/home/kaiser/robot-rl-sim60/architecture-resume-20260917T1010Z`
- 原源码：`/home/kaiser/robot-rl-sim60/transformer-architecture-20260915`
- 新root包含 `resume-plan.json`、`resume_architecture_study.py`、
  `launch_architecture_resume.sh`、`verify_architecture_resume.py`、`launch.json`、
  `status.json`、`resources.jsonl` 和 `jobs/`。
- 完成或退出时写入 `execution-receipt.json`。遇到失败或训练未达到预算即停止队列，
  不将其标为完成，不自动重试覆盖已有目录。

Kaiser另有Round4任务运行，因此本批并发降为1，保持训练统计配置不变。
启动观察全机GPU利用率89%，显存10,922MiB，可用内存5,761,921,024 bytes。
每job上限10,800秒、训练软上限7,200秒，外层36小时；每10秒记录可用内存，
低于1GiB连续30秒时仅中止本批。子进程清理复用冻结调度器的独立进程组处理。

原实验调度器不支持原地恢复，因此使用新root中的运行脚本调用原CLI的 `--resume`。
原study及其停止收据保持原样；原 `experiment_cli summarize` 不会自动把两个root合并，
最终跨seed汇总需显式读取本次result及原已完成项，并保留分段续训标签。

查看队列：

```bash
ROOT=/home/kaiser/robot-rl-sim60/architecture-resume-20260917T1010Z
python3 -m json.tool "$ROOT/status.json"
tmux attach -t transformer-resume-20260917
```

新输出、模型和原始日志留在远端root；本地只记录本说明与小型启动收据。
