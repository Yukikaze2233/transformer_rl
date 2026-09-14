# transformer_rl

这个项目想把一件事做清楚：**在轮腿机器人的控制任务里，什么形式的 Transformer 更合适？**

我们关心的不只是机器人能不能动起来，还包括站立时会不会慢慢漂走、不同高度能不能保持、动作是否抖动，以及这些表现需要多少训练和推理开销。MLP 和 GRU 都是值得认真对待的对照，Transformer 是否有优势，要靠同条件下的结果说明。

代码使用 PyTorch，实现了自己的 PPO、历史管理和实验调度，没有依赖 RSL-RL。仿真通过环境接口接入；目前已在 Kaiser 的 Isaac Lab 环境跑通真实训练与评估。

## 我们在比较什么

所有网络读取相同的本体观测、命令和动作历史。当前默认窗口是16帧，每帧30维，包含传感器数据年龄及实际采样间隔。

| 配置名 | 想回答的问题 |
|---|---|
| `last_token_attention` | 标准因果 Transformer，从最后一个当前帧读出动作，表现如何？ |
| `index_attention` | 在相同位置编码下，独立的命令 query 是否比最后一帧读出更好？ |
| `time_attention` | 用真实时间差编码历史，是否比只用帧位置更合适？ |
| `gated_attention` | 门控残差能否让优化和控制更稳定？ |
| `supervised_attention` | 加入显式状态估计监督，是否能改善时序表示？ |
| `history_mlp` | 同样的历史信息，普通 MLP 能做到什么程度？ |
| `history_gru` | 门控记忆是否已经够用，是否需要 attention？ |

Transformer默认使用64维、2层、4个注意力头。辅助监督单独分组，因为它增加了训练信息，不能把收益全算到结构上。网络尺寸和历史长度都在配置里，命名不绑定机器人型号或实验轮次。

## 怎么判断“更好”

先看任务是否完成，再看完成得是否平稳。

我们复查过已有 MLP 的原始轨迹：有一个高度的波动很小，但机器人每20秒仍会漂移约1米。另一个高度的速度曲线也很平滑，却在持续后退。**“不怎么抖”和“站在正确的位置、保持正确的高度”是两件事。**

因此评估会分开记录：

- 高度、速度和姿态的偏差；
- 静止漂移、异常接触及失败情况；
- 每个环境、每个回合去掉均值后的波动；
- 腿部目标、轮速目标和力矩在采样点之间的变化；
- 多个训练 seed 的差异、实际样本量和计算开销。

稳态统计固定去掉每次 reset 后的前200步，不把 reset 的跳变算作抖动，也不把不同回合的均值差混进去。采样频率为100 Hz时，这些结果只反映相应控制频段，不能当作下位机高频电流环的测量。

## 目前做到哪一步

多架构的训练、检查点恢复、ONNX导出、命名场景评估和批量汇总已经接通。Kaiser上完成了六种网络的短训试验，以及学习率、输出初始化和探索标准差的敏感性研究。

目前采用的工作配方是 `learning_rate=3e-5`、`mean_init_scale=0.1`、`initial_std=0.2`。它让 Transformer 在三个新 seed 上都能充分使用优化预算，但短训的高度和接触表现还不好，不能据此选出赢家。

下一档配置是 **7种网络 × 3个训练 seed**。在512环境、32步 rollout 下，每项977次更新，对应 **16,007,168个样本**。最终模型分别测试低、中、高站立、前进、后退和左右旋转。16M只是学习曲线的第一档，是否继续到64M、128M，要看学习趋势和独立评估。

机械模型本次只做了视觉随动修复。训练继续采用同一研究动力学，视觉组件数量不会自动变成新的物理自由度。模型和驱动仍有研究近似，当前对照也没有覆盖非零通信延迟或推扰，结果会按这个范围解释。

## 快速开始

需要 Python 3.11 或更新版本，建议使用独立环境：

```bash
python -m pip install -e '.[test,export]'
python -m transformer_rl inspect --config configs/control.json
python -m pytest -q
```

`inspect`只显示网络配置和参数量，不创建仿真环境。

先生成实验计划，可以检查将要运行哪些网络、seed和评估场景：

```bash
python -m transformer_rl.experiment_cli plan \
  --spec configs/learning_curves.json --root runs/comparison
```

通用配置中的`environment_factory`默认为`null`，需要先填写实际环境工厂和参数，再生成新计划。配置完成后运行：

```bash
python -m transformer_rl.experiment_cli run \
  --root runs/configured_comparison --max-parallel 2

python -m transformer_rl.experiment_cli summarize \
  --root runs/configured_comparison
```

每项实验有独立进程、优化器、检查点和日志。默认只对最终检查点执行完整场景评估，中间检查点用于恢复和后续分析。已有输出目录不会被覆盖。

## 接入自己的任务

环境实现 `VectorEnv` 接口，提供观测、reward、终止状态以及 reset 前的终态。网络只生成控制量；动作缩放、通信队列和底层控制器仍由环境或控制层负责。

`previous_issued_action`表示上一条已经发出的指令，不意味着电机已经执行。已知的数据年龄与未知延迟也分开编码。这样的区分是为了让训练和真实控制链路有清楚的对应关系。

`examples/isaaclab_task.py`是现有研究任务的接入示例，不是一个自动适配任意机器人资产的工具。Isaac SDK的进程生命周期由专用worker管理，使用方法在[Kaiser记录](docs/KAISER_EXPERIMENTS.md)中。

## 文档

- [训练预算与网络选型计划](docs/TRAINING_EVALUATION_PLAN.md)
- [实验配置、并发和结果汇总](docs/EXPERIMENTS.md)
- [MLP稳态基线](docs/MLP_STEADY_STATE_BASELINE.md) / [稳定性统计口径](docs/STABILITY_METRICS.md)
- [优化敏感性复验](docs/KAISER_SENSITIVITY_RECHECK.md) / [浮点一致性问题](docs/BEHAVIOR_PRECISION.md)
- [检查点与ONNX导出](docs/EXPORT.md)
- [视觉修复审查](docs/CHASSIS_GEOMETRY_REVIEW.md)

维护者：`yukikaze2233 <yingziyuw@gmail.com>`。
