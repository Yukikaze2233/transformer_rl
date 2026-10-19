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

正式采样参数只修改 [`configs/formal_training.json`](../configs/formal_training.json)：8192环境、48步、3个训练seed各20,000更新。部署器调用 `StructuredTrainingSession.create_from_settings(...)`，同时传入已核验的资产、任务、初始权重、runtime和资源绑定；该调用只冻结会话，启动仍走现有CLI。完整展开后的参数写入运行目录的 `recipe.json`，checkpoint每100更新保存一次。修改源码配置不会改变已冻结会话。

资源等待、失败与运行状态以看板和本次运行收据为准；不重复启动第二个控制器。跨boot恢复先核验并取得对应授权，旧PID和旧数值游标不再复用。

## 正式训练与人工评价

正式训练只有一个Actor，课程连续曝光，不自动调度周期评价。训练使用冻结release；修改 `main` 不会热改该运行。

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
