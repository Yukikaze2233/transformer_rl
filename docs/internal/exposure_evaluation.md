# 固定矩阵的独立物理评估

`exposure_evaluation` 供 `exposure_campaign` 在各训练阶段完成后调用。策略家族和历史长度来自完整协议；物理数据接口采用已冻结的六电机底盘适配器。一次请求对应一个 job、一个阶段、一个 validation seed，并包含该组合声明的全部场景和全部 replica，不能根据结果删减场景或改变分母。

请求绑定协议原文件的 SHA、controller 原文件、实际阶段 endpoint、模型与控制合同、环境快照及所有场景合同。endpoint 校验会检查实际 checkpoint、Adam、随机状态、训练记录及完整阶段链。评估进程使用确定性的原始均值，然后按训练合同限幅；它拥有独立评估 seed，不复用训练中的环境或历史状态。

评估保存完整的 policy-rate PRE-reset 轨迹。轨迹同时记录原始均值、issued action、物理目标和力矩、任务指标、稳定性信号、终止标志、episode 编号，以及**推理之前**的 episode age。不能用已经推进后的物理时间戳替代推理 age。`age >= H - 1` 属于 full-history，较短窗口属于 reset-filled；两种样本的统计和分母分别保存。差分跨 reset、指令变化或两种历史窗口边界时不计入相应历史窗口的速度、目标变化率和力矩变化率。

发布结果时，verifier 从实际 NPZ 中重放聚合指标及各场景指标，并逐项比较 report，不能凭 report 自报指标。底盘指标和稳定性信号名称由适配器接口固定；修改 JSON、补签外层 SHA、删除困难场景或缺失信号，都不能替代完整轨迹。正常完成还要求实际 worker 的终止记录、原命令和进程身份相符。

provider 直接接受 validation 角色。held-out 角色必须携带由 `exposure_selection` 重新核验全部 validation 原始证据后封存的 choice 收据，绑定原协议完整矩阵及确切 checkpoint；不能提前观察 held-out 来挑选网络。campaign 默认先完成全部训练和 validation，再在同一 controller 与两个原始 FD 的授权下封存并运行完整 held-out 比较；`--validation-only` 仅关闭 validation。该选择仍是未通过真实部署 latency 门槛的 provisional 控制质量选择。

`tests/test_exposure_evaluation.py` 使用六电机合成物理环境，执行实际 CPU PPO、checkpoint 加载、`FrameHistory` 推理、异步 reset、磁盘轨迹写入和重放。独立进程测试使用真实 `-m transformer_rl.exposure_evaluation worker` 命令、父进程身份和两个实际继承的 flock；固定在测试 SDK 树中的 `sitecustomize.py` **只替换原队列定义/闭合 provider 与 Isaac factory resolver**。这些替代是合成 fixture，不能证明真实原队列已经闭合或 Isaac 仿真已经运行。测试存储合同使用真实 CPU checkpoint、完整 ControlTrace NPZ、optimizer 日志及实际缓存文件清单四类校准；这些 CPU 样本不代表生产 Isaac 的启动/缓存或大模型存储上限。原始 Counter 与 report 标签是代码固定的 owned-worker attribution，不能称为独立 PhysX 场景行身份验证。实际 launcher 行为由 campaign 测试另行覆盖。

CPU 接口验证不意味着生产 campaign 已启动、held-out 选择已完成、真实 100 Hz 控制期限已满足或硬件已验证。轨迹的 policy-rate 物理力矩与机械功率 proxy，也不能代替传输到达时间、控制器实际应用时刻、高频电流环与真实能耗测量。
