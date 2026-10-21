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

正式采样参数在 [`configs/formal_training.json`](../configs/formal_training.json)：8192环境、48步、3个训练seed各35,000更新。完整展开后的参数写入运行目录的 `recipe.json`，checkpoint每100更新保存一次。修改源码配置不会改变已冻结会话。

Lab 正式接入使用经过验收的源码和两个已审核 JSON：runtime 绑定当前 boot、机械资产、电机响应及地形 atlas；resources 保留原 GPU、RAM、磁盘门槛和四个锁的身份。以下创建全新会话，不启动 Isaac 或训练；不接收旧诊断 checkpoint，也不覆盖已有运行：

```bash
/home/kaiser/robot-rl-sim60/env/bin/python -m transformer_rl.formal.lab_env.prepare \
  --settings configs/formal_training.json \
  --runtime-json <reviewed-runtime.json> \
  --resources-json <reviewed-resources.json> \
  --output <new-session-directory> --seed 2102
```

默认冻结35k四阶段；`--no-jump`显式创建不同的30k三阶段会话，不能用于扩容旧运行。原生场地、完整更新及控制边界验收齐备后，用 `/home/kaiser/.local/bin/trainctl start <new-session-directory>` 启动，再按上面的 pause/resume/stop 控制。已有会话只能走其正常保存端点恢复，不再次执行 prepare；冷恢复不会清零预算。源码必须来自已部署的审核版本，不能混用本机工作区与旧冻结 release。

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
| 高度与姿态 | 目标为选中105°资产的完整机械行程；当前默认仍为0.20–0.35m、标称0.32m，完整上下界待机械核算；含七类命令轨迹 |
| 前后运动 | 当前请求域−2.7–2.7m/s，包含零速、反向与启停；不请求机身横移 |
| 转向 | 请求域−6π–6π rad/s，原地转向及与前后运动、高度的组合；当前不含云台坐标系任务 |
| 地形 | 目标为±15°坡面、0–2cm粗糙高度、连续0–200mm迎面台阶、0–150mm下台阶、楼梯、平台和摩擦变化；当前静态atlas仅包含有限几何档位，v2单级/双级真实资产独立标识 |
| 跳跃请求 | 最终请求域为机身相对起跳支撑高度的顶点上升量20–40cm（标称高度下绝对顶点约0.52–0.72m）；课程从较小上升量连续扩展；轮底净空单独作为腾空证据；同一Actor持续控制 |
| 恢复 | 16类出生姿态：直立/蹲伏及释放、左右侧倒、前后倾、前后倒地及左右腿非对称、腹部/背部着地且腿在上方 |

每类连续量使用70%域内、15%边界、15%标称邻域抽样，并保留20%标称回合。请求域不等于全部组合物理可达：速度与转向经所选轮半径、轮距及轮速目标包络处理，原请求、准入命令和限幅分别记录。重复ID、取消与安全接管由可部署请求状态机处理；自动训练定时器发出新请求ID，协议异常序列由人工评价显式给出。

完整机械高度与从易到难的阶段内调度尚未完成；当前四段只改变任务比例。扩大后的yaw／跳升请求域已经接入采样和观测，仍需原生物理验证，不能沿用旧范围的测试或将请求覆盖视为策略能力。

正式计划为seed2101/2102/2103各35,000更新。每次8192×48＝393,216条fresh转移，每seed13,762,560,000条，总计41,287,680,000条；脚本准备等无Actor梯度区间单独mask，不冒充全部都是有效策略样本。周期checkpoint每100更新保存，暂停/停止另存完整学习状态。实际请求域以该运行冻结的 `recipe.json` 为准，修改域需要新的身份，不能改写旧记录。

部署顺序为：冻结名义105°、上界107°的选中机械资产与电机响应来源 → 核验12/6闭链、reset与时序 → 实测8192环境内存和完整采样/PPO → 冻结正式CLI会话并启动 → 按固定课程训练。人工评价按需加载checkpoint；训练完成后分别比较三个训练seed，再做部署导出与实机差距检查。完整课程配置不表示各能力已经学会，工期以8192环境的实际吞吐测量为准。

人工评价加载指定checkpoint，显式选择任务与案例，使用同一观测、reset和动作映射。检查高度、速度、姿态、静止漂移、接触与饱和，同时记录时间和计算开销。`formal/evaluation/`提供独立执行器，不能用旧研究CLI的格式替代正式格式。

正式仿真工厂完成接入验证后才启动PPO。网络单元检查通过不表示机器人已经在训；生产运行以真实worker和学习日志为准。

## 研究入口

旧命令入口继续保留，实现在 `research/`：

```bash
python -m transformer_rl --help
python -m transformer_rl frame --help
python -m transformer_rl.experiment_cli --help
```

这些入口有独立的配置、checkpoint和研究预算。旧实验的自动评价、网络选型或回退机制不会加入正式训练循环。详细复现说明与历史结果见[研究归档](archive/research)。

运行产物与原始录包、权重、数组放在忽略的本地运行目录；源码提交和训练发布规则见[开发与版本管理](DEVELOPMENT.md)。
