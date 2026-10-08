# 按实际策略历史区分控制表现

`evaluate_frame_policy(..., control_metrics=True, history_control=True)` 增加可选的历史条件统计，默认入口与既有报告保持原行为。帧龄在本次 `actor(current)` **之前**复制，不从已经前进的物理时间反推。终止样本仍使用旧回合的历史；下一次推理才将该行年龄归零。

对长度 `H`，`pre_inference_episode_age >= H - 1` 表示完整历史，否则属于 repeat-first 的 reset 填充期。H1 的填充类自然没有样本，所有不可用指标保留 null。各环境可独立 reset；短回合可能始终没有完整窗口，不删除这些回合或改变分母。

`HistoryControlStatistics` 在两类窗口分别保存速度、角速度和高度的偏差、MAE、RMSE，以及按每个环境、回合与窗口去均值的波动。稳态额外要求 reset 后及恒定指令保持时间均达到 `settle_steps`；不足 `min_steady_samples` 的片段单独计数，不把不同回合均值差算作抖动。

二维世界位移的有限差分给出实际漂移速度。腿部物理目标、轮速目标、实际力矩、actor 原始均值和限幅后 issued action 各自保存差分，只有同一回合、同一历史类别且指令未变化的两个端点才贡献区间。类别转换、reset 和指令跳变不制造跨界差分。实际目标与 issued action 是不同事件，机械功率是 `sum(abs(torque * velocity))` 的采样代理，不能代表电流环能耗或传输到达延迟。

开启完整轨迹时，额外保存推理前年龄、原始均值、issued action，以及该 tick 的 reward、任务成功标志、全部明确命名的物理指标和稳定性信号。独立 `exposure_trace` 验证器读取所有声明行、完整 ZIP/NPY 数据与 CRC，并重放报告统计；不凭有限 JSON 数字或低波动认定控制正确。该路径验证的是策略频段的数据；实际模拟器、部署延迟和硬件控制资格仍需真实运行证据。
