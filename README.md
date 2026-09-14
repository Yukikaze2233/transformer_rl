# transformer_rl

**独立的时间感知 Transformer 控制研究仓库。** Actor 以因果注意力编码本体观测、历史指令与真实时间信息，再通过当前命令 query 输出连续动作。采集、PPO、控制时序和模型导出采用独立接口；不依赖 RSL-RL。

## 优化敏感性与足量训练

已完成学习率、均值输出头初始化和探索std的真实敏感性研究，并在三个新训练seed上复验。候选配方为`learning_rate=3e-5 / mean_init_scale=0.1 / initial_std=0.2`：三个seed均用满8步/update的优化预算，原baseline仅约14–16%。这说明优化节奏改善，**不等于高度/接触表现已经通过**。

同时复现并修复了batch512与4096间的FP32舍入差被log-prob放大、误触发一致性检查的问题；没有放宽原moment容差或改变PPO ratio。详情见[数值证据](docs/BEHAVIOR_PRECISION.md)和[完整复验](docs/KAISER_SENSITIVITY_RECHECK.md)。

[足量训练设计](docs/TRAINING_EVALUATION_PLAN.md)按16M→64M→128M transitions渐进推进，区分调参seed与最终确认seed。[学习曲线规格](configs/learning_curves.json)第一档为6模型×3训练seed，每项在512环境下采16,007,168样本，最终checkpoint评估7个命名场景。仅最终checkpoint自动评估，中间checkpoint用于恢复/后续分析。

## 多架构对照

现支持六种配置变体，统一历史信息、环境与样本预算，分别保存优化器和产物：

| 变体 | 目的 | 默认actor参数量 |
|---|---|---:|
| `time_attention` | 真实时间年龄编码的Transformer基线 | 69,772 |
| `index_attention` | 历史位置编码消融；frame中的时间信息仍保留 | 69,772 |
| `gated_attention` | GRU式深度残差门控，研究优化稳定性 | 168,844 |
| `supervised_attention` | query辅助状态预测，研究显式监督的作用 | 69,967 |
| `history_mlp` | 同窗口、同可见信息的MLP对照 | 74,700 |
| `history_gru` | 每窗从零重算的GRU对照，无跨调用隐藏状态 | 19,230 |

监督变体独立分组；辅助目标列必须由环境解释，不能仅凭列号冒称速度真值。参数量和实际计算量不同，不宣称等参数对照。原始默认网络检查点可通过显式迁移继续加载。

`configs/comparison.json`默认规划6变体×3训练seed，各自用3个独立评估seed验证：

```bash
python -m transformer_rl.experiment_cli plan --spec configs/comparison.json --root runs/comparison
python -m transformer_rl.experiment_cli summarize --root runs/comparison
# 在新规格中配置实际环境工厂后执行，可显式选择并发数：
python -m transformer_rl.experiment_cli run --root runs/configured_comparison --max-parallel 2
```

默认通用规格的工厂为null，只能规划；机器人例子另见[Kaiser接入与实测](docs/KAISER_EXPERIMENTS.md)。调度有独立进程组、超时回收、来源/配置哈希与分层seed统计，详情见[实验指南](docs/EXPERIMENTS.md)。

**Kaiser已完成6/6真实训练＋独立评估pilot**：每项512环境、20次PPO更新、327,680个训练样本；有效并发2，总训练样本1,966,080。全机GPU利用率峰值98%（含已有任务）。3并发启动时WSL内存余量不足，已保留该中止记录。这是单seed短测，不是收敛或架构优胜证明。

## 当前实现

- **时间感知 actor**：默认64维、2层、4 heads、FFN128；16帧历史，末尾附加当前命令 query。无dropout、无可变KV缓存。
- **显式观测时间**：历史时间戳以float64秒保存，先求age再转网络精度。未知sensor age由known标志区分，不等同零延迟。
- **纯PyTorch PPO**：Gaussian raw-action likelihood、clipped policy/value loss、GAE双mask、KL提前停止、更新前完整行为分布校验。
- **采集与存储**：partial reset、跨rollout历史连续、独立tensor快照；time-limit使用reset前终态bootstrap。
- **控制时序组件**：按秒运行的batched延迟指令通道；区分issued、到达、控制侧应用和目标保持；提供独立policy/controller/physics周期约束。
- **检查点与导出**：模型/Adam恢复、严格元数据校验；ONNX导出前通过多种合成历史的CPU ORT一致性检查。

核心保持通用tensor接口；`examples/isaaclab_task.py`已接入Kaiser现有的外部研究物理任务并完成上述pilot。**研究资产、真实通信和下位机PID的硬件一致性仍未验证**，旧部署demo不是硬件依据。当前任务沿用研究合同，额外通信延迟为零；不能由此声称获得延迟鲁棒性或有效实机策略。

## 快速使用

Python 3.11及以上。核心依赖为PyTorch，建议使用独立环境：

```bash
python -m pip install -e '.[test,export]'
python -m transformer_rl inspect --config configs/control.json
python -m pytest -q
```

`inspect`只构造网络并报告参数，不创建环境。配置中的16维本体量、3维命令、6维动作是一个轮腿控制示例；字段含义与缩放由环境明确绑定，仓库名称和模块不绑定型号或实验轮次。

### 接入自己的环境

实现 [`VectorEnv`](src/transformer_rl/types.py)：

- `reset(seed)`返回`VectorObservation`。
- `step(issued_action)`返回`StepResult`，done行已auto-reset。
- `final_critic`必须来自真正的reset前状态，不能用重置后的状态替代。
- 环境工厂签名为`create_env(*, model_config, environment_config, device)`。

已有Gymnasium/Isaac Lab风格tensor环境可用`TensorEnvAdapter`，显式提供观测encoder和终态字段。适配器不猜测关节、命令、传感器时间、动作映射，也不负责创建Simulator；Isaac AppLauncher启动与关闭由环境工厂管理。

未来完成环境接入后，显式训练入口为：

```bash
python -m transformer_rl train \
  --config configs/control.json \
  --env-factory my_task:create_env \
  --device cuda:0 --updates 1000 --rollout-steps 48 \
  --max-seconds 3600 --run-dir runs/experiment
```

`my_task:create_env`是用户提供的接口位置，不是仓库已提供的机器人任务。当前没有默认的环境交互或训练启动。默认不裁剪发出动作；需要时显式配置`--action-clip`，并在环境中实现一致的控制量映射。

run生成`run.json`、`metrics.jsonl`、`checkpoints/`和完成/失败回执。输出目录与检查点拒绝覆盖。`--resume`恢复模型与优化器，指定的`--updates`是本次额外更新数；采集从新episode开始。模型/PPO配置、环境配置、工厂和动作裁剪必须与checkpoint一致。

时间预算是**软预算**：在物理交互和优化边界检查；阻塞的环境调用、初始化或正在进行的一次PPO update不能被此参数强制中断。长任务仍应由外层进程管理器设置硬超时。收到SIGINT/SIGTERM后在边界停止并保存完整优化状态。

### 导出

```bash
python -m transformer_rl export \
  --checkpoint runs/experiment/checkpoints/checkpoint_001000.pt \
  --output runs/experiment/policy.onnx
```

模型只输出确定性raw mean；动作缩放、限幅、通信和PID留在控制层。五个输入为`frames / times / valid / command / now`，静态历史长度、动态batch。详情见[导出约定](docs/EXPORT.md)。

## 默认网络输入

| 字段 | 维数 | 语义 |
|---|---:|---|
| 本体观测 | 16 | 由环境定义并缩放，不自动读取仿真真值 |
| 历史命令 | 3 | 每帧保留当时命令 |
| previous issued action | 6 | 上次发出的控制量，不冒称电机已应用 |
| sensor age | 2 | 秒，协议与时钟确实支持时才已知 |
| age known flags | 2 | 区分未知与已知零延迟 |
| policy interval | 1 | 实际policy事件时间间隔，秒 |
| **单帧合计** | **30** | 历史时间戳与valid mask另行保存 |

默认actor为**69,772参数**，独立critic为**48,897参数**。这些数字只是结构计数；Transformer参数虽小，完整历史attention仍比短窗MLP消耗更多计算与激活显存。

## 为什么尝试Transformer

期待它通过较长的动作—响应历史，适应延迟、异步观测和隐含动力学；当前命令query提供独立读出，使历史命令不被新命令覆盖。时间编码提供物理时间尺度，而不只依赖帧序号。

代价包括更高的训练/推理开销、更多采样需求和可能的控制滞后。query没有增加额外可观测信息，网络也无法恢复完全不可辨识的滑移。应与**同历史、同时间信息**的MLP、复旦式速度估计MLP和GRU比较。当前实现是Transformer在线PPO，不是Decision Transformer；DT/ODT需要独立的轨迹与回报条件化设计。

详细取舍、时间通道、内存算术与硬件边界见[架构评审](docs/ARCHITECTURE_REVIEW.md)，集成契约见[接口说明](docs/INTERFACES.md)。

## 源码结构

| 模块 | 职责 |
|---|---|
| `model.py` / `history.py` | 时序actor、独立critic、帧构造与历史 |
| `storage.py` / `ppo.py` | 完整窗口rollout、GAE、PPO优化 |
| `runner.py` / `adapters.py` | 同步采集与环境契约 |
| `timing.py` | 目标通信延迟、锁存和多频率调度约束 |
| `checkpoint.py` / `export.py` | 模型恢复、ONNX及验证回执 |
| `cli.py` / `config.py` / `types.py` | 入口、参数与共享tensor接口 |

维护者：**yukikaze2233 <yingziyuw@gmail.com>**。
