# Kaiser：估计器对照的分阶段启动记录

## 实际启动

队列于 **2026-09-18 03:24:35 +08:00** 启动，单任务串行。
运行源码提交为 `5f38d0db8f273aa7f89ea6560eb361a4ad39d44d`，工作树clean；
该提交的GitHub CPU CI通过。

- Source：`/home/kaiser/robot-rl-sim60/transformer-estimator-20260918`
- Study：`/home/kaiser/robot-rl-sim60/estimator-study-20260918`
- tmux：`estimator-study-20260918`
- Package SHA：`902bd3c7f54dd952f017fdf82599f781965a91350b7e49d746261c64ac2054c0`

配置顺序为6项80-update接线、2项16M特权状态诊断、条件性18项64M主对照。
诊断任一必需场景/seed未通过冻结门槛即进入`needs_review`，不自动延长诊断或放宽标准。
接线阶段的10s评估必然可能全部是censored片段，其任务门槛失败不等于接线失败；
这一阶段只验训练、checkpoint、轨迹、评估接口和导出。

## 首次观察

**03:36:59 +08:00**：direct MLP已完成80 updates、1,310,720条训练采样、8,000条评估
采样及ONNX验证；新的PRE-reset轨迹文件SHA校验通过。direct Transformer训练也已到80，
正在评估。此时其余四项接线、特权诊断和正式主对照仍是后续计划，不算完成。

同一时段可用内存约6.32GB，全机GPU利用率90%、显存9,997MiB。
记录见 [estimator-run-start.json](evidence/estimator-run-start.json)。

后续 **03:41:55 +08:00** 观察：两个直接控制网络均已完成接线；速度估计MLP已到
78/80 updates、1,277,952条采样，最近一次PPO及估计器均执行8个优化器步。
队列无失败/中止记录，仍属于接线阶段。

另外在同一台4090执行了四种估计器的短程合成CUDA验收，**零环境采样**：

- 四项PPO与估计器更新均成功；
- 强制拒绝辅助更新后，模型、Adam、CPU与CUDA RNG逐位恢复；
- 双优化器checkpoint重新加载到CUDA后与保存前一致；
- 该验收进程峰值allocated memory约65MB，不代表仿真总显存。

详见 [estimator-cuda-acceptance.json](evidence/estimator-cuda-acceptance.json)。
这些检查不证明策略已经学会控制，也不是MiniPC时延验证。

## 查看、暂停与回收

在Kaiser上：

```bash
source /home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh
SOURCE=/home/kaiser/robot-rl-sim60/transformer-estimator-20260918
ROOT=/home/kaiser/robot-rl-sim60/estimator-study-20260918
export PYTHONPATH="$SOURCE/src:$SOURCE"
python -m json.tool "$ROOT/status.json"
python "$SOURCE/tools/run_estimator_study.py" stop --root "$ROOT"
```

`stop`核对launcher PID、启动tick、脚本和root，再用pidfd发送SIGTERM。所属worker有
120s清理宽限；最终以`execution-receipt.json`及checkpoint为准。仅有`launch.json`
不能证明后续阶段完成。回收继续使用`tools/recover_study.py`的快照/逐文件SHA流程。

每阶段输出在`STAGE/plan/jobs/VARIANT/seed_SEED/`；完成阶段另有`results.json`、
逐训练seed/场景/checkpoint的`summary.json/.csv`。本编排的多检查点评估目录与旧
`experiment_cli summarize`不同，应读取本阶段汇总。

## 小型MLP的工程优先级

当前7万至9万参数矩阵用于受控结构比较，不是声明工程上必须使用这么大网络。
本轮重新核对旧16M结果，history MLP三个seed的stand_mid平均高度bias约−81.95mm，
Last-token约−86.95mm；前进vx MAE分别约0.5051和0.5024m/s，两者都尚未合格。
不能将此解释为Transformer优越，或MLP已经解决任务。

5帧历史、同样frame30与时间特征的速度估计MLP，加相同控制头，实际参数量为：
encoder `[64,64]`时29,545，`[128,64]`时43,945（均不含探索参数与critic）。
这是值得优先研究的工程基线，但改变历史长度属于新的信息窗口实验，需要独立配置
和预算；不能在正在运行的冻结队列中静默替换。
