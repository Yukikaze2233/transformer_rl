# Changelog

## 0.2.0.dev0 — Unreleased

- 开发单一结构化 Transformer 路径：五帧含当前、100Hz actor、独立81D critic。
- 接入独立 PPO、连续 exposure 课程、完整学习状态 checkpoint 与私有受控会话。
- 保留原有研究入口及其配置、预算与 checkpoint 边界。
- 默认不调度周期评价；策略性能由手动评价确认。
- 源码分为 `formal`、`research`、`common`；正式实现按职责分包，保留既有CLI入口。
- 将日常文档浓缩为README及网络架构、系统架构、使用、开发四份说明，旧研究与详细设计归档。
- 集中正式8192×48采样配置及三seed预算，冻结时展开为完整recipe；Actor支持按有效标签计数归一化的分块梯度累积，周期checkpoint间隔由recipe消费。

这是源码开发版本，不表示机器人正式训练、完整物理验收或实机部署已经完成。

## 0.1.0 — Historical source baseline

- 时间编码策略、packed-frame 对照结构、独立 PPO 与学习状态恢复。
- 原有研究、分析及导出工具保留在历史提交中。

本条目归档已有源码版本，不补造历史 Git release/tag。
