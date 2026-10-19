# 系统架构

## 代码边界

```text
transformer_rl/
├── formal/       正式运动控制实现
├── research/     网络对照、旧训练与实验协议
├── common/       数据类型、制品/RNG、进程控制和会话路由
└── *_cli.py      少量兼容命令入口
```

`formal` 与 `research` 不互相导入。两者只复用通用机制，不共享课程、模型配置或 checkpoint 格式。旧运行继续使用自己的冻结快照。

正式代码按职责分包：

| 模块 | 职责 |
|---|---|
| `formal/policy.py` | 结构化 Actor、独立 Critic、辅助头 |
| `formal/contracts.py` | 独立持有的观测、标签和动作顺序 |
| `formal/learning/` | 历史与 mask、采样、PPO、预算、checkpoint、更新循环 |
| `formal/control/` | 机械观测、输出轴控制、有效响应、气簧与几何边界 |
| `formal/tasks/` | 请求、命令、连续课程、地形、reward 与复位准备 |
| `formal/simulation/` | native 场景、读写桥、物理步、回合和任务适配 |
| `formal/runtime/` | 对象图组装、SDK worker、受控训练会话 |
| `formal/evaluation/` | 人工调用的评价计划、执行与结果 |

`policy` 不导入任务、控制或 SDK。物理与任务定义不依赖 PPO；学习器消费环境合同，不启动仿真。`runtime` 取得资源后组装对象。包的 `__init__` 不预先导入整套网络或环境。

## 维护与扩展

组装只放在 `runtime/assembly.py`，SDK应用寿命由 `runtime/worker.py`管理，资源和进程寿命由 `runtime/session.py`管理。算法、任务和控制对象通过构造参数取得依赖，不用全局注册表隐藏创建过程。

| 变化 | 主要修改位置 | 保持的边界 |
|---|---|---|
| 换网络或解码器 | `policy.py`、对应配置与检查 | 输入窗口和动作合同；不改仿真实现 |
| 改PPO或辅助损失 | `learning/optimization.py` | 采样、概率、标签与消费预算分别可核对 |
| 增加任务、reward或连续课程 | `tasks/` | 返回明确命令和物理条件；不导入学习器或SDK |
| 换机械资产、电机或传感器 | `control/`及对应native适配 | 通过命名绑定连接输出轴和物理量，不在网络里补传动逻辑 |
| 换仿真后端 | 新后端适配及runtime组装 | 对学习器提供相同的观测、完整转移、reset和控制权语义 |
| 加评价指标 | `evaluation/` | 人工单独调用，不耦入正式PPO循环 |

`contracts.py`是学习器和环境的数据交界：观测、标签、动作顺序与完整转移均显式传递，关键边界持有独立副本。通用类型、原子制品/RNG和文件控制机制留在 `common/`；只有会话路由和CLI允许按需选择正式或研究实现，其他通用模块不反向导入业务包。

这些依赖由 `tools/check_repository.py`静态检查。跨模块复用通过已有合同或通用机制完成；单模块内部逻辑先留在所属模块，不新增无明确职责的utility层。已有冻结训练快照与新源码版本分别维护。

## 网络接口

正式路径使用单一Actor：100Hz、五帧含当前、35D观测和6维动作，独立81D Critic只用于训练。网络层次、输入分组、历史mask、参数量与部署裁剪集中在[网络架构](NETWORK.md)。

## 采样与更新

```text
200Hz物理/反馈 → 100Hz观测与请求 → 完整rollout
 → 冻结旧策略概率与原始窗口 → 当前权重重算 → 独立PPO更新
 → 完整更新边界 checkpoint / 暂停 / 保存退出
```

正式配置集中在 [`configs/formal_training.json`](../configs/formal_training.json)，由 `formal/learning/settings.py` 展开为完整recipe；运行层再绑定资产、源码、初始权重与SDK。修改配置只影响新冻结会话，已有会话的预算和恢复状态不会被改写。

每轮8192环境×48步，共393,216个环境转移，每个环境覆盖0.48秒。PPO为4 epochs×16个逻辑minibatch，每批24,576个端点；Actor按最多3072端点分块累积梯度，再统一裁剪和更新。PPO按整批样本数归一化，各辅助项按整批有效标签数归一化，避免不均匀mask改变权重。Actor与Critic各自最多64次Adam更新，KL回滚包含整个Actor及辅助更新。

采样规模、100Hz和五帧历史对齐复旦[`plane`默认配置](https://github.com/yly-true/fudan_rl_wheel_leg/blob/8204e853dfd2ed06d85a322e1a998c3d20a3be2c/plane/wheel_legged_gym/envs/base/legged_robot_config.py)。其MLP优化参数并非本Transformer的整套配置：

| 设置 | 本正式配置 | 选择依据 |
|---|---|---|
| 环境 / rollout / 历史 | 8192 / 48 / 5帧含当前 | 对齐复旦plane，保持0.48秒rollout |
| epochs / 逻辑minibatches | 4 / 16，前后向分块3072 | 控制激活显存；不增加逻辑Adam步数 |
| 学习率 / clip / 熵系数 | 固定1e-4 / 0.2 / 0.005 | 保留本优化器设置；与v7初始值一致 |
| value系数 / 梯度范数上限 | 4 / 1 | 保留v7基准值，Critic独立优化 |
| gamma / GAE lambda | √0.99 / √0.95 | 保持v7从50Hz换至100Hz后的时间尺度 |
| 周期checkpoint | 每100更新；暂停/停止另存 | 减少常规写盘，启停仍保存完整状态 |

每seed仍为20,000更新，3个seed合计23,592,960,000条fresh转移；这是8192环境下的新增计划，不能沿用原1024环境的总样本数或工期估计。8192环境的原生地形、显存和吞吐尚需Kaiser实测，不因配置已提交而视为可开训。

- 概率比：`r = exp(logπ_new − logπ_old)`。
- 策略损失：`−mean(min(rA, clip(r, 1−ε, 1+ε)A))`。

`A`为有效策略样本的归一化优势，`ε`为clip比例。历史mask、actor/辅助标签mask、GAE传播与value bootstrap分别处理，不把脚本接管的结果归给策略。

课程按连续训练曝光推进，不等待自动评价或通过率晋级。人工评价独立于PPO循环；评价结果不冒充训练重复或真实部署资格。

## 物理与恢复

策略100Hz、物理200Hz，闭链求解固定position12 / velocity6；1kHz只作独立数值参考。请求发出、传输到达、控制应用与物理响应分别计时。电机拟合的有效响应只应用一次，不叠加重复的USB时延。

Actor的传感器模型位于 `control/sensors.py`：200Hz缓存四组具名真读数，100Hz选择过去样本后加入物理单位均匀噪声；原始样本时刻随读数保留，reset行清缓存。观测延迟取0或5/10/15ms，属于工程压力项；每回合以20%概率保留无额外传感器扰动。PD反馈、critic和监督标签使用干净物理状态。噪声幅度及混合比例由 `runtime.tuning.sensors`冻结；回合与噪声各用独立RNG，一行reset不移动其他行的后续噪声序列，冷恢复接续RNG并重新准备物理历史。该路径不表示动力学随机化、通信辨识或实机资格已完成。

checkpoint保存模型、Adam、RNG、课程和消费预算。暂停在完整更新边界保存并留在同进程；保存退出后的冷恢复保留学习状态，但重建物理回合与历史。它不保证断电且无checkpoint时恢复未保存状态。

网络、资产、控制参数、SDK和时序分别绑定来源。代码和静态检查不能代替原生碰撞、闭链、任务可达域或Sim2Real验收。
