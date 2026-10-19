# transformer_rl

轻量 Transformer 运动控制、独立 PPO 与可恢复训练管理。代码明确分为三部分：

| 目录 | 职责 |
|---|---|
| `src/transformer_rl/formal/` | 单一策略的正式训练、任务、原生仿真与人工评价 |
| `src/transformer_rl/research/` | MLP、历史编码器和 Transformer 的结构比较、旧研究协议与分析 |
| `src/transformer_rl/common/` | 共享数据类型、制品与 RNG、进程控制和 CLI 路由 |

正式路径采用 **100Hz 策略 / 200Hz 物理、5帧含当前（40ms）、35D Actor / 81D Critic、6维动作**。轮支路输出速度目标，课程连续推进，评价由人工调用。网络与 PPO 不依赖 RSL-RL。

原生仿真接入仍在验证；源码开发版本不等于机器人正式训练或实机验收已经完成。旧研究的历史长度、自动评估及预算不作为正式路径的默认配置。

## 阅读入口

- [网络架构](docs/NETWORK.md)：输入输出、Transformer、历史与mask、参数量和部署。
- [系统架构](docs/ARCHITECTURE.md)：源码边界、训练数据流、物理与恢复。
- [使用](docs/USAGE.md)：环境、控制台、启停恢复和研究入口。
- [开发与版本管理](docs/DEVELOPMENT.md)：验证位置、提交、冻结版本与产物管理。
- [变更记录](CHANGELOG.md)。

本机和 CI 只执行静态检查：

```bash
python tools/check_repository.py
```

模型、接口、性能与仿真验证在 Kaiser 执行。训练运行使用冻结源码与配置，不跟随 `main` 热更新。

旧研究报告、详细设计与证据索引保留在 [`docs/archive/`](docs/archive)，不作为当前使用入口。
