# 多架构、多 seed 实验编排

接口以 `EXPERIMENT_INTERFACES.md` 为准。独立入口为
`python -m transformer_rl.experiment_cli`，Python API 位于 `transformer_rl.experiments`。
默认通用规格的环境工厂为null，只能规划。额外提供的Isaac Lab研究任务适配器已在Kaiser完成六变体并行pilot，见[实测记录](KAISER_EXPERIMENTS.md)；这不代表真实硬件合同已验证。调度器的单元测试使用fake subprocess，物理训练收据单独记录。

## 对照规格

`configs/comparison.json` 默认安排 6 个变体 × 3 个训练 seed（11、22、33），
每个训练结果使用独立评估 seed 101、102、103。预算、环境与评估协议在规格中统一指定。

| 变体 | 配置变化 | 分组 |
| --- | --- | --- |
| time_attention | 默认配置 | architecture |
| index_attention | time_encoding=index | architecture |
| gated_attention | residual_type=gated | architecture |
| supervised_attention | auxiliary_indices=[25,26,27]，auxiliary_coef=0.1 | supervision |
| history_mlp | actor_type=mlp | architecture |
| history_gru | actor_type=gru | architecture |

**25、26、27 列的语义必须由实际环境确认，不能仅凭列号称为线速度。**
0.1 只是辅助损失系数候选；增加特权监督与纯架构对照属于不同分组，不能混为单一架构排名。
index 编码只消融历史位置编码；frame 中的 sensor age / policy interval 仍保留。

### 公平性边界

plan 严格限制相对 common base 的配置差异：architecture 组只允许改变
`actor_type / time_encoding / residual_type / baseline_hidden / gru_hidden / d_model /
num_layers / num_heads / ffn_dim`。supervision 组额外允许 `model.auxiliary_indices`
与 `ppo.auxiliary_coef`；其他 PPO 参数仍必须相同。architecture 组不能带辅助监督头。
显式重复相同值允许，但改变观测维度、历史长度、action 维度、初始 std、critic 结构等会在创建 root 前拒绝。
默认 history_length 均为 16；需要不同信息窗口或优化预算时，应使用独立规格/root，不能冒称同信息纯架构比较。
共同的 action clip 与环境配置由规格统一控制；工厂实际遵守这些语义仍需环境侧确认。

## 规划与执行

在安装了本仓库及相关依赖的 Python 环境中运行：

```bash
python -m transformer_rl.experiment_cli plan --spec configs/comparison.json --root /data/experiments/comparison
python -m transformer_rl.experiment_cli summarize --root /data/experiments/comparison
```

`base_config` 相对规格文件解析。plan 校验规格与配置、冻结完整默认值和合并后的训练/评估配置，
保存规范化 spec SHA、配置 SHA、plan SHA 与当前导入包的源码清单 SHA。
JSON 空白/键顺序不影响规范化哈希。重复 root（包括已存在空目录）拒绝覆盖。
配置新增字段必须已经由配置模块支持；不会静默忽略尚未实现的字段。

实际环境工厂经核实后，在新的规格文件中设置 `environment_factory: "package.module:factory"`，
重新规划到新 root，再执行：

```bash
python -m transformer_rl.experiment_cli run --root /data/experiments/robot_comparison
```

默认一张 `cuda:0`，`max_parallel=1`。需要两个并发 job 时用户显式选择：

```bash
python -m transformer_rl.experiment_cli run --root /data/experiments/robot_comparison --max-parallel 2
```

CLI 上限覆盖会记录在 `execution.json`；设备按固定 worker slot 轮转分配，slot 覆盖整个 train→eval 序列。
一张卡上的两个 slot 会共享该卡。这只是资源并发上限，不是显存容量或吞吐提升的测量，不能断言两个仿真同卡更快。
job 的 wall-clock timeout 包含训练和所有评估；training.max_seconds 是主训练 CLI 的内部软预算。
清理时先向所属进程组发送 SIGTERM，给予最多 3 秒 grace，再 SIGKILL 并 wait。
这段 grace **额外于 job timeout**，不会将超时任务变为 completed；正常阶段退出若遗留后代，同样有界清理。
grace 值记录在 `execution.json` 的 `termination_grace_seconds`。

run 使用当前Python解释器，argv直接传入subprocess，不使用shell。execution可选`worker_module`，默认`transformer_rl`；Isaac Lab例子使用`examples._isaaclab_process`，在CLI写完产物后以真实退出码关闭SDK。该例子需要源码checkout在PYTHONPATH中，纯core wheel不包含examples。
每个阶段有独立 process group，每个 job 有独立训练目录、优化器和日志；跨 job 有界并发，job 内严格串行。
超时、失败和 SIGINT/SIGTERM 会终止本编排创建的进程组并 wait 回收直接子进程；不会按进程名清理其他用户进程。
正常退出后遗留在该进程组中的后代也会被清理；脱离 session 的外部守护服务不在此简单执行器的管理范围内。
中断时尚未启动的 job 保持 missing。`execution.json` 是一次性执行标记：失败、中断或完成后都不能在原 root 重跑。

## 本机、远端与来源一致性

本机和远端都需要在**同一执行环境下能够 import 本包、环境工厂及其依赖**；本工具不提供 SSH 调度、依赖安装或源码同步。
可将未执行的完整 plan root 复制到远端再 run；生成配置和 job 路径相对 root，
训练完成后的 checkpoint 元数据由主 CLI 写入绝对路径，因此已执行 root 不支持无损搬迁后重新汇总。
运行开始会校验实际导入包的源码 SHA，训练后及评估结束再次核对；plan 后改动 model/source 必须重新规划。
源码哈希包含包内非缓存文件（包括未提交修改），不依赖 git clean 状态。
这不是完整容器快照：外部环境包、资产和驱动仍需用户在实验记录中固定；不要在执行中修改源码。
**包外 factory 的代码不在 package source snapshot 中；plan SHA 没有冻结全部仿真。**
外部 factory、机器人资产、场景、仿真器及驱动应由用户另外提供可追溯的 commit、包版本、文件 SHA 或镜像 digest，
并在实验记录中关联；仅冻结 `module:callable` 字符串不能证明工厂实现未改变。
哈希用于一致性检查，不是抵御能够重写全部 manifest 的攻击者的签名。

## 评估与产物

只有主 CLI 返回成功，且 `train/completion.json` 明确 completed、更新数达到预算，
最后 checkpoint 更新数和实际文件 SHA 匹配时才进入评估。checkpoint 必须位于该 job 的 checkpoints 目录。
每个 eval seed 调用：

```text
python -m transformer_rl evaluate --checkpoint PATH --config MERGED_EVAL_CONFIG --env-factory MODULE:CALLABLE --steps N --seed N --device DEVICE --output PATH [--action-clip X]
```

评估 config 的 model/PPO 保留训练配置，environment 为 common base environment 与 evaluation.environment 的顶层合并。
每个 seed 写入独立 `evaluation_SEED.json`；检查 checkpoint SHA、seed、steps、环境、action clip、deterministic_mean 协议和有限指标。
completion 的 collected_transitions 必须是正整数；评估终止/截断计数各自不超过 transitions。
指标要求 min≤mean≤max（mean 边界允许相对 1e-7 的舍入误差）、rms≥0、0<count≤transitions；count 必须为整数。
退出码为零但缺报告仍为 missing；不将训练收益或 completion 当成物理评估成功。

```text
ROOT/plan.json
ROOT/configs/VARIANT.json
ROOT/configs/VARIANT.evaluation.json
ROOT/execution.json
ROOT/jobs/VARIANT/seed_SEED/train/{completion.json,checkpoints/,metrics.jsonl,...}
ROOT/jobs/VARIANT/seed_SEED/{train.log,evaluation_EVALSEED.log,evaluation_EVALSEED.json,result.json}
ROOT/summary.json
ROOT/summary.csv
```

summarize 可重复刷新两个派生汇总文件；冻结配置、计划及执行产物不会覆盖。
加载时校验 JSON shape、plan/spec/config 哈希、预期 job 路由和 root 路径包含关系。
每个 variant 在 JSON/CSV 中都有 requested/completed/failed/timedout/missing；这些状态互斥，总数等于 requested。
summarize 会重新核实已完成 job 的 checkpoint 和全部评估报告，缺任一 eval seed 时整个 train seed 不进入聚合。

`fairness_checks` 按 group 检查所有 variant/train seed 的 completion.collected_transitions 是否相同，
以及所有独立 eval 报告的 transitions 是否相同（含不同 eval seeds）。固定 updates/steps 不意味着固定样本数，
因为 factory 的并行环境数量 N 可能变化。JSON 保留逐 train seed 的实际训练与逐 eval seed 的 transitions。
预算不一致不会抹去已完成报告，但会令该 group 的 `comparison_available=false` 并给出原因。
缺任一请求的训练 seed 时，variant 和 group 明确 `partial=true`，整个 group 比较不可用；
无报告时样本预算为 unavailable，而不是推断 N 固定。CSV 重复输出 partial、comparison_available、
training_budget_consistent、evaluation_budget_consistent 和 comparison_reasons。
所有检查仅在组内进行，supervision 不混入 architecture；comparison_available 只表示配置/完整性/样本预算检查通过，
不证明外部环境一致、物理成功或某个变体优胜。剩余完整 seed 的统计始终只是描述性结果。

先对同一训练 seed 的全部 eval seeds 等权取均值，再跨训练 seeds 计算 mean 和样本 std（ddof=1；n<2 时 std=null）。
JSON 保留逐 train seed 的状态和 eval-seed 均值，CSV 每个 metric 一行并重复完整完成率信息。
物理指标的 mean/rms/min/max 均分别按该层次聚合；例如 rms 的汇总是各评估报告 RMS 的均值，不是重新池化帧的 RMS。
training_reward_diagnostic 仅为每个 seed 最后一次更新的收益诊断；不宣布 winner。
缺失 seed 显式展示，统计只描述可用的完整 seed 子集，不代表完整实验结论；不挑最佳 seed，也不把 frames 当独立样本。
没有环境提供的物理指标时 physical_metrics_complete=false；该字段只说明报告齐全，不是 physical success。
物理成功仍需环境定义的阈值与任务语义。不同分组、环境、预算和评估协议不混合排名。

## 定向验证

```bash
python -m pytest tests/test_experiments.py -q
```

测试内部 runner 替换 CLI 执行载荷为短生命周期 fake Python subprocess，保留 argv/产物协议。
覆盖冻结与拒绝覆盖、空 factory、源码漂移、路径/配置/计划校验、并发上限、训练→评估顺序、
SHA/更新数/报告检查、层次均值和样本标准差、失败/超时、SIGTERM 清理及无关进程隔离。
新增覆盖不公平 override 拒绝、训练/评估样本预算不一致、组间隔离、partial 比较阻断、
metric 统计一致性、合作式 SIGTERM 清理与忽略终止后的有界 SIGKILL 升级。
