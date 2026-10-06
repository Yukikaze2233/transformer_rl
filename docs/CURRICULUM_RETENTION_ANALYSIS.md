# Gated 课程保持与遗忘后处理

`tools/analyze_curriculum_retention.py` 只用 Python 标准库读取冻结课程与已封存的训练、评估产物。不导入学习器、仿真器、Torch、NumPy，不反序列化模型，也不启动训练或评测。它与 `analyze_frame_learning.py` 分开：后者检查 PPO 的实际优化工作，不读取课程控制矩阵。

此入口支持既有 Gated H31 pilot：mixed→mixed、stationary→stationary、stationary→mixed 三组，至少三个独立训练 seed，400／1200 更新终点，8701／9701 配对评估。每次训练更新有 49,152 条环境样本；1200 更新累计 58,982,400 条。KL 提前停止不改变该环境样本账本。

它不授予所有网络的选型资格，输出始终为 `formal_architecture_selection:false`。课程的 `task_dose_matched:false` 意味着结果是任务课程与暴露变化的效果，不能宣称隔离了纯顺序效应。

## 冻结独立分析协议

先创建输入目录外的新协议文件，再将协议用于捕获。已有文件不会被覆盖；输出也不能放入冻结实验及其 prepared、source、campaign 子目录。

```bash
python3 -B tools/analyze_curriculum_retention.py prepare \
  --campaign-root /absolute/experiment/curriculum_campaign \
  --output /absolute/new_analysis/protocol.json
python3 -B tools/analyze_curriculum_retention.py analyze \
  --protocol /absolute/new_analysis/protocol.json \
  --output /absolute/new_analysis/retention.json
```

`prepare()`、`analyze()` 本身不写文件；CLI 只创建用户指定的外部新文件。失败时保留真实异常，不降低校验要求。缺少报告时 CLI 正常完成一次 `not_ready` 捕获，这个退出状态不表示科学结论已就绪。

协议 `transformer_rl.curriculum_retention_protocol` 包含：

| 字段 | 约束 |
|---|---|
| `campaign_root`、`campaign`、`manifest`、`parent_study` | 输入根与文件 SHA／字节数 |
| `source_sha256`、`snapshot_sha256` | 实际冻结学习器和动力学快照身份 |
| `training_seeds`、`evaluation` | 训练重复与配对重测分开 |
| `old_cases`、`new_cases` | 驻车十种条件和其他四十个案例，覆盖恰好五十个案例 |
| `gates`、`min_completed_episodes` | 原 sealed `snapshot/parent_study.json` 的物理门槛、稳态资格及至少八个完成回合 |
| `metrics` | 原始单位的高度、前向速度、偏航误差、倾角；驻车另报告漂移峰值 |
| `weighting` | 案例内 eval seed 等权、声明案例集合内等权、训练 seed 等权 |
| `acquisition`、`retention`、`missing` | 掌握、保持和未知的固定定义 |
| `sha256` | 除自身之外的规范 JSON 内容 SHA |

`analyze()` 重新从冻结输入推导同一协议；只修改门槛、分组或权重后重新计算协议 SHA，也会被拒绝。原门槛已在训练研究快照中冻结，新增协议不回写原 manifest。

驻车掌握必须同时满足存活率、高度、速度、偏航、倾角和漂移要求。普通持续任务的 `success_rate` 只表示任务拥有的成功口径，不能单独作为控制达标。`stability.available` 只说明样本充分，也不能单独说明学会。

## 输入路径与身份链

现有真实布局以 `curriculum_campaign` 为根；其 `campaign.json` 指向同实验目录下的 prepared manifest 和冻结 source。相对路径、绝对路径和符号链接都必须留在各自允许的根内。

| 产物 | 读取内容 |
|---|---|
| `campaign.json`、`manifest.json` | campaign 身份、manifest 内容 SHA 与文件 SHA、副本一致性 |
| prepared 的 `snapshot/snapshot.json` | 全部 snapshot 文件、原研究 gates；额外未封存文件拒绝 |
| source 的 `src/transformer_rl`、三个 controller/helper 文件 | 文件 SHA 与 prepared learner identity；源码只读取、不执行 |
| `summary.json` | 必须绑定 `digest(campaign.json)`；文件尚不存在可报告 `not_started` |
| `jobs/{arm}/seed_{trainseed}/{phase}/training/attempt_NNN/` | request、receipt、run、completion、PPO 日志、checkpoint 及其 JSON sidecar 的 SHA／父链／样本账本 |
| `control/{arm}/train_{trainseed}/{phase}/seed_{evalseed}/attempt_NNN/` | receipt、五十个案例报告、control、trace 的 SHA 与一致身份 |

训练必须同 seed 各 arm 初始模型 SHA 相同。阶段二必须 `restore_learning_from` 阶段一父模型，保留 Adam/RNG／累计时钟；`initialize_from` 不合格。完整 rollout 的停止与 resume 可沿已封存的连续账本继续，不能退款或重复更新。run 元数据还必须匹配实际学习器 source、environment factory、rollout48、retention系数0、repeat-first 历史 reset、device 及冻结运行配方。

单案例评估应为4001策略步、8个环境、32,008条物理样本，checkpoint400或1200，统计200／200。案例的 model、control SHA、环境合同和原始样本计数须一致。案例内 control 与 suite 的同名 group 须相同。trace 验证文件 SHA 及 JSON 来源，并要求每套50案例×8环境的全部400个评估环境行；这里的400是环境行数，不是更新数。此工具不解析 NPZ 数组、不由轨迹重新计算指标。大文件 SHA 以固定大小块读取。

所有读取文件的 SHA／字节数写入输出 `input_receipts`，结束前再次核验，输入中途改变会拒绝发布。实际749冻结 source 无需等于后来的 main；必须等于其自身 campaign 和 prepared manifest。

## 差分、资格与空值

每个 `cells` 单元是一个 arm、train seed、phase、eval seed、案例。当前三训练 seed 的完整网格为 **1800个单元**。有效单元保存原始单位指标、门槛值、通过与失败理由、回合数、稳态资格及原报告 SHA。缺报告或所需有限指标缺失时，`status:not_ready`、`passed:null`，保留缺项理由。

每案例先在同 train seed 内配对相同 eval seed，再等权求均值。`paired_changes` 共150个 train-seed／案例对，计算：

- `arm_error_changes`：各 arm 的 error1200−error400；正值表示退步。
- `old_error_increase`：驻车案例的 pretrain 原始变化；即使未掌握也保留描述性变化。
- `new_error_reduction`：新案例的 error400−error1200；正值表示改善。
- `pretrain_minus_stationary_error_change`：pretrain 的变化减 stationary 的变化。
- `new_pretrain_minus_stationary_gain`：新任务收益的同方向对照差分，是上一项的相反数。
- `qualified_forgetting`：只有 pretrain 阶段一两套均通过、且阶段二证据充分时才给有符号旧任务变化；另给 `positive_forgetting=max(0,change)`。
- `qualified_forgetting_difference`：还要求 stationary 阶段一确实掌握同旧案例、阶段二证据充分。

连续量不混合物理单位。移动任务的位移不是驻车漂移，不进入新任务的 drift 分数。新任务改善和旧任务退步分别展示，不能以阻止学 B 换得保留 A 来宣布成功。

`training_seed_results.retained_case_fraction` 的分母是阶段一两套都通过的旧案例数；分子是其中阶段二两套仍通过的数。无已掌握旧案例时率与 qualified forgetting 为 null。阶段一缺报告使分母未知；阶段二缺报告使已掌握案例状态未知，两者都会使率为 null。缺项不当失败、不从分母删除。样本充分但实际失败的评估仍为有效失败结果。

十种驻车扰动是一个旧技能的十个案例，`retained_case_fraction` 不叫十技能容量或技能保持率。`forgetting.available` 只表示存在明确合格的案例／训练 seed 对；必须同时查看范围、null 汇总与未就绪单元。

`aggregate` 按独立训练 seed 保存原始值、等权均值和样本标准差。任一声明训练 seed 的该量不可用时，不给部分 seed 的乐观均值。标准差不是置信区间。8701／9701 重测不新增训练政策；固定确定性案例重复也不是独立初态样本。

## 输出与验证

输出格式为 `transformer_rl.curriculum_retention_analysis`，主要字段为 `status`、`expected_cells`、`ready_cells`、`cells`、`paired_changes`、`training_seed_results`、`aggregate`、`unready`、`forgetting`、`input_receipts`、`protocol_sha256`、`analyzer_sha256`、`interpretation`。

全部报告可读且身份有效时 `status:complete`；它仍可能得到“阶段一未学会、遗忘不可定义”。等待中的课程得到1800个未知单元和 `status:not_ready`，不会创造成绩。SHA、路径、身份或预算不匹配则抛出真实错误。

```bash
python3 -B -m pytest -q tests/test_curriculum_retention_analysis.py
```

测试采用同布局的合成封存数据，覆盖真实门槛退步、未学会、正迁移、缺前／后报告、缺有限指标、错误计数、错误父链、weights-only初始化、不同初始模型、错误 source／配方、协议重新封存作弊、路径／符号链接逃逸、JSON异常和外部不覆盖输出。它们验证后处理接口，不冒充机器人学习或遗忘证据。
