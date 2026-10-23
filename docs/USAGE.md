# 使用

## 环境与检查

本机和GitHub CI只做源码静态分析：

```bash
python tools/check_repository.py
git diff --check
```

模型、unit/interface、性能与仿真检查在Kaiser的已配置环境中运行，不在本机建训练venv或下载依赖。开发安装入口为 `python -m pip install -e '.[test,train,export]'`，无需每次运行重复安装。

## 训练控制台

Kaiser使用完整路径 `/home/kaiser/.local/bin/trainctl`。以下 `<session>` 是部署器已登记的会话别名或运行目录：

```bash
/home/kaiser/.local/bin/trainctl status <session>
/home/kaiser/.local/bin/trainctl watch <session>
/home/kaiser/.local/bin/trainctl pause <session>
/home/kaiser/.local/bin/trainctl resume <session>
/home/kaiser/.local/bin/trainctl stop <session>
```

看板快捷键：`p`暂停、`r`继续、`s`保存退出、`q`离开看板。离开看板或SSH断线不会停止训练。`--plain`关闭颜色，`status <session> --json`用于机器读取。

暂停和停止都等待完整更新边界，不发送强杀信号。`resume`从同进程暂停点继续，或从正常保存退出的完整学习状态恢复；预算不会清零。冷恢复会重置物理回合和历史。

正式会话由部署器冻结完整源码、recipe、runtime及资产绑定，再登记控制台。已有 `init`、`adopt`分别服务研究会话与研究队列，不拿旧研究配置冒充正式recipe。`jobs`也只适用于研究队列。

正式采样参数在 [`configs/formal_training.json`](../configs/formal_training.json)：4096环境、24步、50Hz策略/200Hz物理、3个训练seed各35,000更新。完整展开后的参数写入运行目录的 `recipe.json`，checkpoint每100更新保存一次。修改源码配置不会改变已冻结会话。

Lab 正式接入使用经过验收的源码和两个已审核 JSON：runtime 绑定当前 boot、机械资产、电机响应及地形 atlas；resources 保留原 GPU、RAM、磁盘门槛和四个锁的身份。正式入口在一次命令中冻结 35k 配置并创建全新会话。默认只准备、不启动 Isaac：

```bash
/home/kaiser/software/isaaclab-3.0.0/IsaacLab/.venv/bin/python -m transformer_rl.formal.lab_env.entry \
  --settings configs/formal_training.json \
  --runtime-json <reviewed-runtime.json> \
  --resources-json <reviewed-resources.json> \
  --output <new-session-directory> --seed 2102 \
  --python /home/kaiser/software/isaaclab-3.0.0/IsaacLab/.venv/bin/python
```

命令使用与原生验证相同的解释器，并显式用 `--python` 绑定已验证的 worker；省略 `--python` 时入口只会绑定当前解释器，不能证明 worker 是已验证的 Lab 3.0 环境。需要在准备后立即启动时追加 `--start`；仅初始化并暂停用 `--start --paused`。入口默认写入 action transport 的整策略步延迟 `0,1`（50Hz 下 0 或 20ms）。`nominal_fraction=0.5` 是**唯一**的零延迟概率：零延迟先从非零延迟集合中排除，再按该比例覆盖，因此实际为零延迟的 reset 行约为 50%，不是 75%。这是额外传输不确定度的压力项，**不是 USB 单向时延测量，也不是重新加入实测电机响应的标称延迟**。实际响应曲线保持不变；没有证据时不能把这组压力参数称为实测分布。可用 `--action-delay-steps` 指定其他非负整数步数，或用 `--no-action-transport` 保留 runtime 中已有的配置。

新会话可追加 `--control-timing-json configs/control_timing.json`，将完整上/下行预算和回合随机化课程写入冻结 runtime。模板的0.1–5ms总往返按一半分给两向；PD与Actor复用上行，额外Actor观测lag、策略目标lag和辨识电机响应各只经过一次。模板以5000更新为曝光终点：0时延/噪声增量和单位气簧强度逐步展开到原配置范围及0.9–1.1气簧强度。完整曝光时目标队列仍为0/20ms各50%，观测“无额外lag”约60%；上行不包含在这些零额外比例中。没有硬件证据支持的jitter/dropout不默认添加。

新增 `collection.control_timing` 输出当前回合配置与末个物理区间的范围/均值，不冒充整轮池化统计。新冷恢复同时校验并恢复通信、传感器、动作队列和气簧随机流；物理缓存与活动回合重新建立。模板只在新建会话显式启用，不会更新已有preflight或其runtime。

默认冻结 35k 四阶段；完整入口不提供 30k 缩减模式，`--no-jump` 仍只属于独立 prepare 路径，不能用于扩容旧运行。入口不注册别名；命令成功后会打印冻结后的具体会话目录。原生场地、完整更新及控制边界验收齐备后，用 `/home/kaiser/.local/bin/trainctl start <new-session-directory>` 启动，再按上面的 pause/resume/stop 控制。已部署的启动器同时提供 SDK 所需的库路径和缓存配置；仅指定解释器不足以代替这些环境设置，直接使用 `entry --start` 时也必须继承相同环境。已有会话只能走其正常保存端点恢复，不再次执行 prepare；冷恢复不会清零预算。源码必须来自已部署的审核版本，不能混用本机工作区与旧冻结 release。

runtime 可显式添加 `wheel_material`，在首次任务 reset 前随机化两轮的原生碰撞材质。例如以下为包含低摩擦端点的工程先验：

```json
"wheel_material": {
  "static_friction": [0.1, 1.2],
  "dynamic_friction": [0.1, 1.0],
  "restitution": [0.02, 0.2],
  "num_buckets": 64,
  "anchor_lower_endpoint": true
}
```

64 桶由独立 CPU RNG 生成，两轮分别选桶；同一轮的所有碰撞形状共用一组系数，动摩擦不大于静摩擦。只启动写入一次，并核对完整 shape 表与 Critic 缓存；不修改其他机身材质，不在每回合创建新材质。冷启动以同一 seed 重建并比较材质 SHA。缺少该字段或设为 `null` 时保持源轮胎材质。这里是**轮体材料系数**，轮地等效系数还取决于地面材质和 combine mode；不能把这组数值直接称为实测轮地摩擦。

正式入口默认关闭逐物理子步闭链误差统计，不加入闭链误差终止或晋级门槛。独立诊断可使用 `--loop-telemetry`；这不改变求解器、PPO、reset 或奖励，也不等于机械闭链已通过物理资格。

恢复出生由同一策略控制；脚本准备期间不计 PPO 信用或恢复时限。脚本交回策略时，动作延迟队列按行清空；脚本期间未获 PPO 信用的动作不会在 handoff 后重放。新鲜的上一 tick 估计持续稳定后切回普通高度跟踪；独立的物理稳定证据只记录成功统计，不生成部署端缺失的控制信息。恢复超时按截断处理，保留 PRE-reset bootstrap。日志同时保留出生任务计数与实际采样占比，避免把出生比例当作学习比例。

正式 worker 默认启用 TensorBoard，每个 attempt 写入独立目录：

```text
<session>/attempts/<attempt>/train/tensorboard/formal/
```

可查看 reward、return/advantage、PPO loss/KL/entropy、actor/critic 梯度、采样与更新时间、任务族/命令模式占比、奖励分项、课程阶段与消费计数等已发布曲线。TensorBoard 是旁路记录：metrics JSON 与学习 journal 仍是预算和恢复的权威来源，绘图失败不会回滚已提交的 PPO 更新。 `elapsed_s` 包含控制边界等待；暂停后的首条更新可能包含整段暂停时间。比较稳定轮速时应分别查看 `timing/collection_s` 与 `timing/optimization_and_publication_s`，并将计划暂停与真实采样、优化耗时区分。查看示例：

```bash
tensorboard --logdir <session>/attempts --host 0.0.0.0 --port 6006
```

外部技能命令统一通过 `LabTerrainCourse.submit_request(skill, RequestCommand(...))` 提交：`skill` 为 `jump` 或 `stair`，`source` 为 `manual` 或 `tof`。ID、增高米数和可选 cancel/safety-abort 均为按环境行排列的张量，命令提交后由环境持有副本；每个技能每个边界只接受一个未消费的数据包。BUSY 或 NOT_READY 也会消费新 ID，重试须使用更大的 ID。两个新请求同时到达时 jump 优先，已活动的技能保留仲裁权。物理 reset 只清本行待处理命令，冷恢复不重放物理动作；此接口不改变训练的暂停状态。

资源等待、失败与运行状态以看板和本次运行收据为准；不重复启动第二个控制器。跨boot恢复先核验并取得对应授权，旧PID和旧数值游标不再复用。

## 正式训练与人工评价

正式训练只有一个Actor，课程连续曝光，不自动调度周期评价。训练使用冻结release；修改 `main` 不会热改该运行。

### 完整课程与预算

每个训练seed独立初始化，连续训练35,000更新；阶段之间保留同一网络、两个Adam、RNG与历史消费预算。新回合按下表分配任务，配比顺序为高度、运动、转向、地形、请求、恢复。

| 更新区间 | 阶段 | 新回合任务配比 |
|---|---|---|
| 1–5,000 | warmup | 60% / 20% / 10% / 0% / 0% / 10% |
| 5,001–25,000 | rough | 15% / 25% / 15% / 35% / 0% / 10% |
| 25,001–30,000 | hard | 15% / 15% / 15% / 35% / 0% / 20% |
| 30,001–35,000 | jump | 20% / 10% / 10% / 10% / 50% / 0% |

配比按**新回合数**分配，不保证不同持续时长任务的转移数比例相同。阶段按实际消费的rollout推进；没有自动评价或95%通过率门槛。

| 能力 | 完整请求范围与内容 |
|---|---|
| 高度与姿态 | 请求域0.17–0.40m、标称0.32m；起步0.28–0.32m，再连续扩展。请求域不等于105°资产已通过的物理可达行程；含七类命令轨迹 |
| 前后运动 | 当前请求域−2.7–2.7m/s，包含零速、反向与启停；不请求机身横移 |
| 转向 | 请求域±6π rad/s；常规转向桶±2π，独立低速自旋2π–3.25π、高速自旋3.25π–6π；dash桶前后2–2.7m/s并限制yaw包络；当前不含云台坐标系任务 |
| 地形 | 目标为±15°坡面、0–2cm粗糙高度、连续0–200mm迎面台阶、0–150mm下台阶、楼梯、平台和摩擦变化；当前静态atlas仅包含有限几何档位，v2单级/双级真实资产独立标识 |
| 跳跃请求 | 最终请求域为机身相对起跳支撑高度的顶点上升量3–40cm（标称高度下绝对顶点约0.35–0.72m）；课程从较小上限连续扩展；轮底净空单独作为腾空证据；同一Actor持续控制 |
| 恢复 | 16类出生姿态：直立/蹲伏及释放、左右侧倒、前后倾、前后倒地及左右腿非对称、腹部/背部着地且腿在上方 |

每类连续量使用70%域内、15%边界、15%标称邻域抽样，并保留20%标称回合。请求域不等于全部组合物理可达：速度与转向经所选轮半径、轮距及轮速目标包络处理，原请求、准入命令和限幅分别记录。重复ID、取消与安全接管由可部署请求状态机处理；自动训练定时器发出新请求ID，协议异常序列由人工评价显式给出。

正式课程在四个阶段预算内连续过渡：任务族权重在阶段边界平滑插值，普通任务范围在rough结束前扩到完整域，hard保留完整域复习。rough的地形等级上限也逐步放开到完整atlas，逐环境成功/失败仍控制实际等级。坡面每个等级同时保留正负方向；周期命令时长按现有变化率下限展开，日志分别记录请求、轮速包络投影和实际下发范围。恢复先采直立/蹲伏，再引入倾倒和释放，warmup第4000更新起保留全部16姿态。jump按自身曝光扩展：初期先做静止、标称支撑高度跳跃，再加入运动组合；最后1000更新保留完整3–40cm域。四阶段预算、目标任务族权重和jump恢复出生比例0均不变。出生配额、实际env-step占用、mask后可用于PPO/辅助监督的样本数分别记录；后者仍不等于经过优化器拒绝后保留的梯度步数。

地形路由同时受任务请求上界和逐环境成功/失败等级限制，选中的tile不能覆盖技能等级。固定单级/双级必须带真实高度剖面，缺元数据的旧atlas需另建新版本；成熟rough行才可选择满足请求的固定场地。小于最小几何档位的请求明确落到flat，不能向上取整为更高台阶；请求连续不代表有限atlas几何连续。扩大后的yaw／跳升请求域仍需原生物理验证，不能将请求覆盖视为策略能力。

已有有效扫描的个别环境离开物理地形边界时，终止并reset该行，将该区间从Actor、Critic、辅助监督、GAE和bootstrap中排除，单独计为查询拒绝。初始/reset miss、非有限物理状态、mesh内部空洞/索引错误或整批miss仍报告实际查询错误；不会以零高度冒充命中。此行为需随新的冻结来源部署，运行中的会话不会热加载。

正式计划为seed2101/2102/2103各35,000更新。正式PPO以`1e-4`为初始学习率；每个**逻辑minibatch**在actor优化器步进前，复用该批actor loss前向得到的均值/标准差计算detached KL，再按KL调整actor/critic共同学习率：KL>2×desired时除以1.5，KL<0.5×desired时乘以1.5，边界为`1e-5–1e-2`，与核对到的RSL-RL adaptive PPO一致。随后actor和critic在同一轮调度结果下分别`step()`，不再让critic只使用actor最终LR。正式配置`kl_early_stop=false`、`validate_behavior=false`：不再有限KL超限回滚、停步、health门禁终止或更新前mean/std/logp重算allclose门禁；旧logp、PPO概率比、batch有限值和reset/GAE合同保持不变，非有限梯度/KL仍是不可继续的数值故障。当前formal为4096×24、50Hz策略/200Hz物理、5 epochs×4 minibatches（20步/update），与华南虎公开的4096×24、50Hz/200Hz、5×4=20步/update对齐；H5在这种情况下覆盖80ms，仍不是华南虎的单帧MLP。critic反向为`value_coefficient×MSE`（当前value系数4，即4×MSE），与参考的4×MSE一致，但奖励权重和预训练权重仍是独立问题。每次4096×24＝98,304条fresh转移，每seed3,440,640,000条，总计10,321,920,000条；脚本准备等无Actor梯度区间单独mask，不冒充全部都是有效策略样本。周期checkpoint每100更新保存，暂停/停止另存完整学习状态。实际请求域以该运行冻结的 `recipe.json` 为准，修改域需要新的身份，不能改写旧记录。

部署顺序为：冻结名义105°、上界107°的选中机械资产与电机响应来源 → 核验12/6闭链、reset与时序 → 实测4096环境、50Hz策略/200Hz物理下的内存和完整采样/PPO → 冻结正式CLI会话并启动 → 按固定课程训练。人工评价按需加载checkpoint；训练完成后分别比较三个训练seed，再做部署导出与实机差距检查。完整课程配置不表示各能力已经学会，工期以4096环境的实际吞吐测量为准。

人工评价加载指定checkpoint，显式选择任务与案例，使用同一观测、reset和动作映射。检查高度、速度、姿态、静止漂移、接触与饱和，同时记录时间和计算开销。`formal/evaluation/`提供独立执行器，不能用旧研究CLI的格式替代正式格式。

正式仿真工厂完成接入验证后才启动PPO。网络单元检查通过不表示机器人已经在训；生产运行以真实worker和学习日志为准。

### 观测缩放与噪声

Critic在网络入口按物理组裁剪后缩放：力矩±100 N·m后×0.05，角加速度±1000 rad/s²后×0.01，高度±10m后×5。Actor与地形扫描已有编码，不再重复缩放。环境快照和监督标签保留原始物理单位；该变化需要显式source迁移，不能热改正在运行的worker。

Actor噪声在物理量上加一次均匀随机误差，然后编码；最终半幅为：

| 物理量 | 最终噪声半幅 |
|---|---|
| 陀螺 | 0.25 rad/s |
| 重力投影 | 0.05，无量纲 |
| 腿关节位置 | 0.025 rad |
| 腿关节速度 | 0.5 rad/s |
| 轮速 | 1 rad/s |

这些是物理单位下的半幅，不是统一的百分比。默认20%标称回合不加噪，其余回合由现有课程逐步增大幅度；Critic真值、命令、动作历史和辅助标签不加这些噪声。噪声、传感器历史延迟、动作队列延迟及实测电机响应分别配置，不用USB RTT替代单向延迟。

## 研究入口

旧命令入口继续保留，实现在 `research/`：

```bash
python -m transformer_rl --help
python -m transformer_rl frame --help
python -m transformer_rl.experiment_cli --help
```

这些入口有独立的配置、checkpoint和研究预算。旧实验的自动评价、网络选型或回退机制不会加入正式训练循环。详细复现说明与历史结果见[研究归档](archive/research)。

运行产物与原始录包、权重、数组放在忽略的本地运行目录；源码提交和训练发布规则见[开发与版本管理](DEVELOPMENT.md)。
