# 首轮七变体：训练完成与完整回收

## 完成状态

续训队列于 **2026-09-18 02:17:58 +08:00** 正常完成，9项续训/新训练及63份场景评估均成功。
与首次完成的12项、84份评估合并后，首轮七变体三个seed共 **21项最终模型、147份评估**。
每项最终累计update为977。检查时该队列及其worker已退出，无需再次发送停止信号。

两个supervised任务从789/732恢复，仍保留“分段续训、仿真及历史重置”的标签。
全部采集训练样本合计336,164,352，其中13,824条属于人工暂停前未完成rollout的采样；
不能抹去这部分后再声称采样数完全相等。评估样本共2,352,000。

本记录证明执行完成与工件完整性，不据此宣称控制任务成功或架构胜出。
旧四Transformer的物理分析、曲线和人工停止收据保持其历史含义；完整七模型比较应
使用本次回收的最终模型及原始评估，并明确两项分段续训的影响。

## 本地回收

目录（相对于仓库）：

`artifacts/recovered/architecture-16m-20260915T0540/recovery-complete-20260917T1820Z/`

该目录包含归档、manifest、逐文件验证收据和解包数据：

- `extracted/original/`：首次study，包括冻结配置、原12项完整结果与两个中断段。
- `extracted/study/`：续训study，包括9项完成结果、lineage、调度器及退出收据。
- `extracted/source/`：实际训练源码的git tracked文件，commit `80a45b4...`。
- `extracted/runtime`、`extracted/recovery_tool`：运行环境脚本与本次快照工具。

研究物理任务与资产另保存在早先的 `20260915T075822Z-8f22/extracted/task/`，
此前暂停后快照也继续保留。远端原始文件保留，回收没有删除模型或日志。

| 核验项 | 结果 |
| --- | --- |
| 回收文件 | 852份，432,114,532 bytes |
| 训练检查点 | 212个，另有1个随机接口探针检查点 |
| 归档大小 | 314,990,837 bytes |
| 归档SHA-256 | `f2bb3d340414f1cf984355252996f640d845d9ee71409ed051f5aa3e4b379963` |
| manifest SHA-256 | `6e001fdef4c1152aa15878ad48bbdc5e7583c3f31c8c696f05a2e34ad54281d8` |
| 路径安全、大小与逐文件SHA | 852/852通过，解包后再次核对 |
| 最终模型严格CPU加载 | 21/21通过，配置、seed、环境identity及累计update匹配 |
| 原始场景评估校验 | 147/147通过，匹配最终模型SHA、环境和协议 |

CPU加载使用本地有限float32频率兼容修复，详见 [CHECKPOINT_PORTABILITY.md](CHECKPOINT_PORTABILITY.md)。
没有改写原检查点。212个训练检查点均做文件哈希校验，严格模型加载验收范围为21个最终模型。

小型核验收据：[architecture-complete-recovery.json](evidence/architecture-complete-recovery.json)。
大归档、权重和原始日志位于被Git忽略的artifacts目录。

## 后续研究状态

首轮已归档。随后按[估计器训练方案](ESTIMATOR_TRAINING_PLAN.md)建立的独立实验仅部分
完成接线，于2026-09-19结束后续安排。最终证据范围和工程选型见
[研究结论与归档](RESEARCH_CONCLUSION.md)。
