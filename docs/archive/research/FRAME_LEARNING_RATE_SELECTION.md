# 固定90单元研究的中立学习率选择与确认接口

本接口用于已经冻结的10种架构 × 3个学习率 × 3个训练 seed 研究。它为每种架构选择自己的学习率，不在架构之间指定赢家，也不表示硬件部署资格。准备输入、source75学习器、原网络架构及课程不改写。入口是独立的 `tools/select_learning_rate.py`。

## 证据顺序与状态

1. 外部训练控制器完成90个单元的实际训练账本核验。每单元必须有1200次完整更新、58,982,400条新鲜rollout样本，同配方恢复链及完整学习状态。第一次初始化的模型SHA与更新后的checkpoint SHA用途不同；这里的延迟和评估均绑定更新1200的checkpoint。
2. 每单元完整开发评估：50个固定case，各8个环境，4个评估seed `701, 1701, 2701, 3701`，每项4001步。原来分别叫validation/final的两个pool在此全部属于开发数据。
3. 独立CPU producer为90个实际新checkpoint逐个导出并测试ONNX。必须绑定对应checkpoint、模型、控制合同、训练配置、冻结学习器与controller，并在同一台CPU、同一线程及运行协议下提供真实文件和进程回执。旧transfer模型的10项测试不能代替本研究的90项证据。
4. 完整开发证据通过原联合门和CPU延迟门后，每种架构在三个学习率中选择，并独占创建不可覆盖的 `choice.json`。
5. 封存choice后才可准备确认计划。确认只使用该架构原选学习率、原三个训练seed及 `11701, 12701` 两个噪声流seed。确认不能重选学习率、重采训练seed或采用表现更好的重试。

`not_ready` 表示缺少任一规定的单元、开发seed、必要数值观察或独立延迟证据。全局不做choice，不删除缺失单元后比较。`no_eligible_rate` 表示完整证据存在，但该架构三个学习率都未通过规定门。完整的低于门槛成功率、超过误差或延迟门、回合不足、观测到无稳态覆盖属于资格失败。生产者允许的零结束回合且success_rate=null，是已观察的删失：保留null，以回合不足判不合格，分数标记 `score_unavailable: success_rate:no_ended_episodes`；不会为它填0/1或判全局缺失。此时其他必要metric字段依然必须存在。其余缺字段或null的必要数值属于未就绪。结构、SHA、来源或协议被改动会直接 `rejected`。

某架构无合格LR会明确保留 `no_eligible_rate`；其他有合格LR的架构仍可封存各自选择。如果十种架构全部无合格LR，不创建choice。不会强行选一种Transformer。

展示用的指标也保持完整分母：pool中任一必要metric为null时，该pool的mean/std保留null，公开finite `count`、`expected_count` 和 `censored_or_missing_count`；不对剩余199/200项平均，也不据其计算三训练seed的指标均值。删失的原值与原因通过原报告和score_unavailable记录保留。

## 保留父研究的评分和联合门

评分沿用prepared父spec冻结的objectives，完整JSON及其SHA写入selection protocol。没有另换多维评分：

\[
s=-4\,\mathrm{success\_rate}
 +\frac{\overline{|e_h|}}{0.03}
 +\frac{\overline{|e_{v_x}|}}{0.15}
 +0.1\frac{\overline{\mathrm{issued\_action\_rate\_rms}}}{100}.
\]

每个训练seed内，50个case先等权平均，再对4个开发seed等权平均。三个训练seed独立计算分数；最终最小化三seed均值加样本标准差：

\[
S=\frac{1}{3}\sum_{i=1}^3s_i+
\sqrt{\frac{1}{2}\sum_{i=1}^3(s_i-\bar s)^2}.
\]

完全同分时取数值更低的学习率。MLP、history MLP和各种Transformer都使用同一套规则。跨训练seed的样本标准差不是置信区间，不将50case×4评估seed当作200个独立训练seed。

每个case × 开发seed × 训练seed都必须同时满足原门：结束回合不少于8、存在原post-settle稳态统计、success_rate≥0.95、高度平均绝对误差≤0.03m、vx平均绝对误差≤0.15m/s、wz平均绝对误差≤0.25rad/s、倾角均值≤0.25rad；站静case还要求回合内最大漂移≤0.20m。yaw、倾角及漂移虽未增加到原标量评分，仍是逐项联合资格门，并公开跨训练seed的均值与样本标准差。它们不能被其他case的高分抵消。

CPU要求ONNX、单线程、1000次实际测试；P99≤8ms、最大值≤10ms、deadline misses=0。实际benchmark中deadline miss按调用耗时≥10ms记录，不能只看最大值边界通过。范围是“history更新 + mean推理 + target映射”，不包含传感器I/O、传输或力矩环，不作为整机部署验收。

`success_rate` 在该冻结研究源码中是已结束回合的 `done & ~terminated & (diagnostic.success | (truncated & ordinary))` 比例。健康ordinary任务的截断可记成功，boundary/blocked截断保留在原分母；它不是单独的任务完整完成率、联合跟踪合格率或不间断全窗口存活率。control packet的 `success_flags` 只来自物理diagnostic.success，二者允许不同。选择继续使用原success gate，并同时保留其他联合门，不改原raw报告或偷偷剔除截断。

每单元输出完整区间样本数、稳态段总数/eligible/short/failed/partial、physical packet结束/失败/success flags以及各轴响应候选、失败、partial和rise/settling删失数。真实接触、力矩、腿部目标速率、轮目标加速度和执行器包络在经过核验的逐case原始metrics/control文件中保留，开发receipt提供这些文件的路径和SHA。不可把无eligible响应的null rise/settling均值补零，也不能把无稳态或失败片段删除以获得更小波动。覆盖计数用于解释证据，不作为case或seed的性能权重。

## 调用接口

以下命令只做文件准备、CPU账本审计和只读统计。selection root、latency root及confirmation root必须与训练输出、prepared输入和source独立；示例路径应替换为同一主机上的实际路径。

```bash
python tools/select_learning_rate.py prepare \
  --campaign-manifest /absolute/runtime/manifest.json \
  --output-root /absolute/selection \
  --latency-index /absolute/latency/index.json
python tools/select_learning_rate.py assess --manifest /absolute/selection/manifest.json
python tools/select_learning_rate.py seal-choice --manifest /absolute/selection/manifest.json
python tools/select_learning_rate.py prepare-confirmation \
  --manifest /absolute/selection/manifest.json \
  --choice /absolute/selection/choice.json \
  --output-root /absolute/confirmation
python tools/select_learning_rate.py audit-confirmation --manifest /absolute/confirmation/manifest.json
```

`prepare` 冻结selection protocol与全部helper bytes。`assess` 每次调用 `run_learning_campaign.audit()` 重新核验实际训练、恢复/预算账本及四seed开发套件。原 `validate_learning_study` 是prepare-only validator，不用来核验runtime。

`seal-choice` 在现有selection锁下重新审计，原子独占创建assessment与choice，不覆盖旧seal。choice同时绑定assessment文件SHA、canonical SHA、protocol SHA及原campaign SHA。准备或审核确认时重新核验开发/runtime/latency证据与原assessment完全相同；任何变化都会拒绝，而不是更新原choice。

`prepare-confirmation` 只写 `status=prepared_not_queued`、`execution_implemented=false` 的冻结计划，包含准确的evaluate-suite命令、配置和checkpoint SHA。它没有启动训练或模拟器，没有排队。[独立确认执行器](FRAME_LEARNING_CONFIRMATION.md) 按照原resource锁和依赖执行这些请求；当前selector文件不实现该调度，原准备计划的字节保持不变。

`audit-confirmation` 验证原请求列表完全一致，每项只有原 `attempt_0000`，实际worker正常结束、命令相同，50case/control/trace覆盖完整。缺证据为not_ready，完整数值不通过为not_confirmed。它返回原choices及确认结果，不生成候选排序或更新LR。

## 开发回执与延迟精确格式

训练控制器接口为 `audit(manifest_path)`；顶层包含 `manifest_sha256, expected_cells=90, actual_cells=90, completed_training_cells, completed_development_cells, status, cells`。单元key是 `rate_000/mlp/seed_1101`。每单元的training含完成/收费更新数、新鲜样本量和更新1200的checkpoint绝对路径+文件SHA；development含四个字符串seed的status及receipt绝对路径+文件SHA。selector不接受删掉单元后的局部grid。

开发receipt的identity严格为 `{manifest_sha256, cell, evaluation_seed, checkpoint_sha256, checkpoint_update:1200, use:"development_only"}`，`directory` 相对原训练controller output_root。它的 `artifacts` 必须包含50个case以及control/trace，均为相对该root的路径+文件SHA。原诊断helper核验所有case的32008条转移、8个环境、相同checkpoint/seed、环境快照、控制指标和trace；selector额外核验model/control身份和原稳定统计协议。

延迟index及单元receipt采用以下JSON形状。这里所有artifact的path都是**绝对路径**，只有export manifest内的graph文件名相对bundle目录。每个标有sha256的artifact是文件原始字节SHA；顶层seal是 `control.digest(去掉sha256后的整个对象)`，不是文件字节SHA。完整index必须提供全部90个key，不能重复或额外加入旧CP。

```json
{
  "format": "transformer_rl.learning_latency_index",
  "schema_version": 1,
  "campaign_sha256": "原runtime manifest canonical SHA",
  "cells": {
    "rate_000/mlp/seed_1101": {
      "status": "completed",
      "receipt": {"path": "/absolute/latency/rate_000/mlp/seed_1101/receipt.json", "sha256": "文件SHA"}
    }
  },
  "sha256": "index canonical SHA"
}
```

```json
{
  "format": "transformer_rl.learning_latency_cell",
  "schema_version": 1,
  "status": "completed",
  "identity": {
    "campaign_sha256": "原runtime manifest canonical SHA",
    "cell": "rate_000/mlp/seed_1101",
    "checkpoint_sha256": "本单元真实更新1200的CP文件SHA",
    "checkpoint_update": 1200,
    "source_sha256": "runtime inputs.source.sha256",
    "configuration_sha256": "cell.training_config.canonical_sha256",
    "control_sha256": "对应training config.control canonical SHA"
  },
  "controllers": {"原runtime全部controller绝对路径": "对应文件SHA"},
  "parity_reference_max_abs": 10.0,
  "parity": {
    "request": {"path": "/absolute/latency/cell/parity.request.json", "sha256": "请求文件SHA"},
    "process": {"path": "/absolute/latency/cell/parity.process.json", "sha256": "进程回执文件SHA"},
    "output": {"path": "/absolute/latency/cell/parity.json", "sha256": "实际参考范围probe文件SHA"}
  },
  "bundle": {"path": "/absolute/latency/cell/bundle/manifest.json", "sha256": "export manifest文件SHA"},
  "benchmark": {"path": "/absolute/latency/cell/benchmark.json", "sha256": "benchmark文件SHA"},
  "export_process": {"path": "/absolute/latency/cell/export.process.json", "sha256": "进程回执文件SHA"},
  "benchmark_process": {"path": "/absolute/latency/cell/benchmark.process.json", "sha256": "进程回执文件SHA"},
  "sha256": "cell receipt canonical SHA"
}
```

进程回执字段包含 `status:"finished", returncode:0, timed_out:false, command:[...], pid:正整数, start:原进程启动时间字符串`，且该PID/start原句柄已不活。command必须由原 `transfer.command` 对冻结source生成，export参数固定为 `export --checkpoint CP --directory BUNDLE_DIR`，benchmark参数固定为 `benchmark --directory BUNDLE_DIR --output BENCHMARK_JSON --backend onnx --threads 1 --iterations 1000`，benchmark结果也必须恰好1000次。这些是实际执行后由独立producer写出的回执；selector不创建这些证据。CPU producer应使用CUDA不可见、`PYTHONDONTWRITEBYTECODE=1`及独占空的 `PYTHONPYCACHEPREFIX` 环境，避免读取旧源码目录缓存；这些bare命令不能与另有 `-B/-X` 参数的训练命令混用身份。

bundle为原 `frame_export.export_frame_policy` 输出，要求 `transformer_rl.packed_policy/v1`、本CP1200、模型/控制与训练配置一致、oldest_to_newest/repeat_first历史，以及实际 `policy.pt` 和 `policy.onnx` 文件SHA。原export的5组输入、batch sizes `[1,7]` 及TorchScript/ONNX parity统计保留。实际 `runtime_versions` 必须含原四键 `torch, numpy, onnx, onnxruntime` 的非空版本字符串，Torch/NumPy与原campaign.runtime.checkpoint_runtime相等，ONNXRuntime与benchmark.machine.onnxruntime相等；缺失或矛盾版本会拒绝，不能只凭图文件SHA替代运行库一致性。

原export检查TorchScript `rtol=1e-5, atol=1e-6`，ONNX `rtol=1e-4, atol=1e-5`，实际逐元素assert通过后才发布bundle。导出输出是未裁剪的raw mean，不能用action bounds或target offset/scale上界代替其参考范围。独立producer必须通过selector提供的 `parity_request(manifest, cell, checkpoint, output_path)` 生成probe请求，再执行 `parity_command(manifest, request_path)` 返回的固定CPU Python命令；这里不执行该命令。该脚本按原generator9271、zeros/random/repeat-first五组输入及batch1/7，读取本CP1200并实测raw mean最大绝对值。request包含format `transformer_rl.learning_export_parity_request/v1`、完整latency identity、冻结source_root、checkpoint ABS artifact、output ABS路径及 `PARITY_INPUTS`；实际output是 `transformer_rl.learning_export_parity_reference/v1`，含相同identity/protocol、reference_max_abs、cuda_initialized=false。request/process/output三份文件都封存SHA，cell receipt的 `parity_reference_max_abs` 必须等于真实probe值。

selector仅检查manifest最大误差≤ `atol+rtol*reference_max_abs` 的必要数值sanity门；不会冒称独立重算逐元素allclose。原逐元素门仍由绑定冻结source和正确终态的实际export进程证明。明显不相容的大误差或未测量的参考范围会拒绝；允许符合原相对门而超过单独atol的正常误差。

benchmark为原frame benchmark输出：`backend, threads, iterations, mean_ms, p99_ms, max_ms, deadline_misses, scope, manifest_sha256, machine`。`manifest_sha256` 绑定bundle文件SHA。machine必须包含 `node, system, architecture, processor, cpu_count, cpu_affinity, onnxruntime`；全部90个单元的整个machine对象SHA相同。没有数值、文件或进程回执时都不能靠模型同构推测延迟。

## 不可变choice与确认回执

choice包含 `format=transformer_rl.learning_rate_choice/v1, selection_sha256, campaign_sha256, assessment:{path ABS, sha256}, assessment_sha256, protocol_sha256, choices, confirmation, sha256`。每架构choice是 `status, rate_id, learning_rate, rank_score`。`formal_architecture_selection` 与 `hardware_deployment_ready` 始终false。

confirmation manifest包含 `choice:{path ABS,sha256}` 和原choice canonical SHA、selection manifest文件SHA、helpers、固定confirmation protocol、原3trainseed×2noise seed的requests。每request含 `cell, training_seed, evaluation_seed, checkpoint:{path ABS,sha256}, configs:{case:{path ABS,sha256}}, directory:相对confirmation root, command, use:"heldout_noise_stream_only"`。期望套件数为选择到LR的架构数×6；十种均有LR时是60项，而不是对所有90个checkpoint重新择LR。

[独立确认执行器](FRAME_LEARNING_CONFIRMATION.md) 必须把每项结果写到原请求directory的 `receipt.json` 和 `worker.process.json`。receipt identity严格含原 `manifest_sha256, cell, evaluation_seed, checkpoint_sha256, checkpoint_update:1200, use:"heldout_noise_stream_only", choice_sha256, confirmation_sha256`；status completed、directory和50case/control/trace artifacts相对confirmation root。不能用开发套件改seed标签代替确认。任何第二attempt、不同训练seed、不同LR或CP、不同命令及扩大scope都会被拒绝。

确认的50个case中38个固定确定性case仍然重复；只有12个noise/combined case的噪声流随seed变化。因此确认范围只叫 **heldout_noise_stream_only**，不叫新的初态、域分布或任务泛化，也不增加独立训练seed数量。独立调度模块已实现，但这里尚无真实确认结果，不把准备计划或CPU协议测试写成执行完成。
