# 十网络配对学习率研究：输入准备与核验

`python -m transformer_rl.learning_study` 只提供 `prepare` 和 `validate`，不启动训练或模拟器。实际训练、开发评估和运行账本由独立的[学习率执行器](FRAME_LEARNING_CAMPAIGN.md)处理；[中立学习率选择器](FRAME_LEARNING_RATE_SELECTION.md)另行审核开发与独立 CPU 延迟证据。准备成功不能写成实验完成，也不会产生 winner 或 eligible 模型。

默认输入是已物化的 `runs/v6_transfer_preparation_20261003/prepared_quotas`。准备器首先核验原 base、study、transfer design、snapshot 完整文件清单、父任务、资产和递归文件/SHA 依赖，再重建共同训练合同、50 个固定评估合同与原联合门槛。它不修改或覆盖这些输入。

## 冻结布局与公平条件

新目录只复制一次环境 `snapshot/`；`parent/` 保存原配置与 snapshot 身份文件。三个安全编号 `rate_000`、`rate_001`、`rate_002` 分别包含自己的 base/spec 和 packed-study，每个是十网络×三个训练 seed。顶层 `manifest.json` 完整枚举90个单元，绑定子计划、所有 train/eval 配置的原始字节 SHA 与 canonical SHA、冻结 learner source、30 个预期初态、环境身份和预算。

默认 LR 为 `{1e-5, 3e-5, 1e-4}`，训练 seed 为 `{1101,1102,1103}`。每个 job 计划1200次更新、1024环境、每次48个新采样端点，即58,982,400个端点；总计划5,308,416,000个端点。这里全是计划量。准备器允许显式指定三个互异正 LR 和三个互异训练 seed，但冻结后不得替换；更改需要新的输出目录。

每个同架构、同 seed 的三个 LR 单元具有相同 actor、critic、探索 std、mean 初始化、历史窗口、课程、场景配额、控制与评估合同。规范 train/eval 配置去掉 `ppo.learning_rate` 后必须完全相等；所有子计划引用同一路径和 SHA 的公共 snapshot。MLP 保持 H1，历史 MLP 与六个 Transformer 保持 H31；Transformer 保持 `oldest`，不会趁 LR 实验引入 `current` 编码。

V6 环境合同保留的 legacy `learning_rate=3e-5` 是共同的非权威字段。本研究实际 Adam 学习率来自 `FrameTrainConfig.ppo.learning_rate`；不声称环境的 legacy 字段随三个 LR 改变。

## CPU 初态及随机数范围

准备器只在 CPU 构造 actor+critic，不构造环境、Adam、PPO 更新或 checkpoint。它在 `torch.random.fork_rng(devices=[])` 中使用 CPU `default_generator.manual_seed`，在成功与异常路径恢复 CPU 默认 RNG，不调用 CUDA seeding 或初始化。初态 SHA 使用现有 `sorted_named_tensor_contents_v1`；清单同时记录 Torch 版本、默认 dtype 和 CPU 设备。validate 在同一源码/runtime 下重新计算30个 SHA；runtime 或初态不符时拒绝，不通过退款、换 seed 或补位修补。

相同 seed 不是跨 runtime 的一致性证明。独立执行器核对 run 中的初态 SHA、配置、冻结 learner source、factory、首次 fresh parent、完整 resume 账本和实际采样预算；仅通过本准备模块的核验不等于这些运行证据已经存在。三个 LR 对应配对训练，独立训练 seed 数仍是3，不能说成9。

即使 Torch 版本与 dtype 相同，不同 CPU 宿主的数学函数也可能生成不同的固定 buffer。完整 state SHA 包含这些 buffer，不能通过忽略它们来宣称初始化完全一致。跨机器核验不一致时，应在目标执行 runtime 重新准备并保留原准备记录；本机 prepare/validate 成功不等于 Kaiser 的初态已验证。

训练入口提供可选的 `expected_initial_model_sha256`，命令行对应 `--expected-initial-model-sha256`，只适用于无父检查点的 fresh 训练。启用后先在 CPU 构造并核对完整 state；不匹配会在创建运行目录、导入环境 factory、CUDA 播种或构造优化器之前拒绝，并恢复调用者的 CPU RNG。匹配后沿用原训练随机数时序，在 run 与检查点元数据中保存 `initialization_guard`。不传入该参数时保持既有行为。

独立执行器从冻结单元取出预期 SHA，在首次 fresh 命令中显式传入，并审核实际 worker 命令、首次初始化证据与运行账本。精确恢复时核对已封存的首次初始化证据、完整学习状态与恢复链，不把 fresh SHA 用作恢复后权重的预期值。接口或 CPU 测试通过仍不能证明90个单元已实际完成训练。

现有环境模块使用由同一个根 seed 派生的独立 generator，初始化 seed 与环境 seed 仍共用数值，尚无分离两种变异来源的协议。探索采样与 PPO minibatch 洗牌使用全局 Torch 随机流；LR 引起不同 KL 早停时，随机调用次数与后续轨迹会分叉。因此本设计固定采样规则与新采样预算，不承诺各 LR 的1200批 rollout 逐端点相同。未来应报告实际 optimizer steps、sample_count、首步/最终 KL 与早停；不能为达到相同梯度步而额外补采样。

## development 与确认集

子 packed-study 中的 validation seeds `{701,1701}` 和名字为 `evaluation.seeds` 的 `{2701,3701}` **全部用于 development**。既有 generic selector 的输出也只能用于 development；它的 Transformer 偏好及打分不能充当本研究的中立最终 LR 选择。

顶层另冻结默认确认 noise seeds `{11701,12701}`，与训练、anchor 和两个 development pool 均不重叠，且不写入任何子计划。必须先根据 development 为每架构决定 LR、封存 choice 的 SHA，然后由独立测试控制器评估确认集；确认集不得用来重新择 LR。独立执行器的前驱完成检查与资源锁用于调度训练和开发评估；学习率选择、独立 CPU latency 及确认评估需各自的证据，不能把 `development_complete` 当作这些步骤完成。现有子 packed-study 可被外部既有命令手工启动，准备模块不是防启动锁或运行授权。

确认范围严格为 `heldout_noise_stream_only`：50个 case 中38个是固定确定性条件，12个 noise/combined case 才有随 eval seed 改变的传感器噪声流。它不是新的初态或扰动域 holdout；固定条件重复也不增加训练 seed。独立初态、域分布与新任务泛化协议仍未实现。

LR 选择必须保留完整三训练 seed 配对网格，以原联合任务门槛先判资格，再看 success、高度、vx、wz、倾角及站静漂移的多维结果，并公开稳态覆盖、响应删失、接触与执行器诊断。case、eval seed 和 train seed 权重预先等权。缺证据应为 `not_ready`，不能删分母；没有合格 LR 时，独立 selector 明确返回 `no_eligible_rate`，不创建赢家。独立执行器提供真实运行核验，中立 selector 沿用原 objectives，以三个训练 seed 的分数均值加样本标准差择 LR；确认核验只读取原选择，不重新排序。准备器和既有偏好 Transformer 的 generic selector 不代替这些步骤。

## 本机命令

```bash
PYTHONPATH=src python -m transformer_rl.learning_study prepare \
  --parent runs/v6_transfer_preparation_20261003/prepared_quotas \
  --directory artifacts/new-learning-rate-study
PYTHONPATH=src python -m transformer_rl.learning_study validate \
  --root artifacts/new-learning-rate-study
```

应使用已安装 Torch 的 CPU 测试环境；无需安装 simulator。两个命令不运行 worker、不写 remote receipt、不排队，已有输出目录、输入内输出、符号链接或逃逸路径均拒绝。validate 同时拒绝子 jobs 中已有运行输出，避免把运行过的目录报告为尚未开始的 preparation；它不是未来 runtime 结果分析器。源码在准备后改变也会拒绝，应重新准备。

`tests/test_learning_study.py` 使用纯文件合同与 CPU 权重，覆盖真实结构的90单元、LR-only 配对、完整种子划分、CPU RNG 恢复及异常、坏父配方、缺依赖、重复或 bool 参数、重新签名后的逻辑篡改、source/config SHA 和路径安全；不会启动模拟器或训练。
