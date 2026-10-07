# 严格资格下的记忆分支执行器

`python -m transformer_rl.retention_campaign` 将已核验的教师、嵌套锚点准备与独立续训连接起来。原来的资格、准备格式、课程、诊断及学习率队列保持原样；新执行器另建协议、输出和账本，不修改原描述符中的历史状态字段。

当前提供方是 Gated Transformer H31 的 pretrain CP400。十个旧站立扰动案例属于同一个站立技能，不能称为十种已获取的技能。此执行器支持该提供方的记忆保留对照，不代表其他架构/H 的资格提供方、多技能遗忘实验或最终架构选择已完成。

## 输入与冻结

准备协议必须显式选择 K、λ 和独立的采集种子。真实数值由实验计划确定，执行器不提供默认 λ。必须先通过原 `prepare_nested_anchors.py::validate_preparation`：全部三训练种子、十个旧案例、两评估种子的原始物理资格，以及全部容量的实际样本、采集来源、嵌套索引和 manifest 均齐备。`not_ready` 无法冻结执行计划。

输入树必须在执行主机真实可读。协议保存新 learner 的完整源文件摘要、原资格/准备提供方、原物理指标验证函数、原始配置、教师及锚点文件收据。运行前和每条分支前重建协议核对；合法的 JSON 自签名或缓存 analysis 不能代替原始证据。

原课程还必须完整闭合，避免把会继续变化的资格分析输入用作冻结依据。新计划同时绑定原课程、诊断及学习率队列的定义、输入、控制器 PID/start、实际 worker 收据和现有锁 inode；不导入旧 learner 执行训练。闭合使用原 helper 的实际源字节及新鲜学习率全网格 audit，缓存证明每次仍重新核验其实际文件和目录清单。

## 公平分支与预算

每份准备分支扩展为 stationary 与 mixed 两个后续环境，均从同一训练种子的完整 CP400 开始。模型、PPO、控制、训练种子不变，只切换后续环境。每条分支新增800更新，1024环境×48步，即39,321,600条新 rollout；期末CP1200累计58,982,400条。继承actor、critic、Adam、全局学习RNG和累计课程时钟，每条新分支从同一个显式私有 retention seed 开始。

完整分母由原准备 manifest 推导。例如两种K×两种λ×三训练种子对应12份准备分支，扩展为24条实际续训、48套独立评估、2,400个案例评估单元。λ=0也保留K分支身份，采样器不抽样、不推进私有RNG；不能通过合并这些对照缩小分母。

启动前独占发布每条分支的完整预算 reservation。失败、部分优化、未封存状态或验证失败保留已消耗及未核验的预算；不退款、不自动重跑、不继续后续分支。已有 campaign 目录拒绝再次运行。`FrameContinuation` 支持完整状态恢复，但本 campaign 的每条新分支使用 `resume=False`；故障恢复必须另作明确计划与预算核验，不能改目录后冒充原实验的成功重试。

## 资源与进程

等待三个原队列全闭合、原控制器与所有 worker 实际终止后，获取原共享资源锁和 transfer-study 锁。只打开既有路径，不创建替代锁。两把独占flock通过FD继承给新子进程；worker检查父PID/start/argv、协议、实际inode、父子FD及锁状态。父控制器意外退出时，活着的子进程继续持锁，避免下一个队列撞上孤儿模拟器。

控制器仅终止自己创建并核实身份的子进程。子进程身份尚未发布时，通过Linux pidfd绑定其内核身份处理失败。观测超时不代表进程结束，也不触发重启。worker有软停止时间，控制器有较长的硬超时；完整rollout才执行PPO，停止和失败另存真实样本计数。

磁盘检查预留全部剩余评估、额外在途trace与周期checkpoint，以未压缩trace预算估计，不依赖压缩节省。空间不足在启动下一条分支前停止。不得删除原教师、正式评估、TensorBoard或未闭合实验来满足预算。

## 训练证明与物理评估

训练退出成功后，验证实际完整checkpoint与teacher SHA、配置、continuation格式/设备/父节点/segment/时钟、原始模型摘要、实际Adam步数、每条PPO日志的batch及mini-batch样本、私有采样器状态、环境来源、实际请求和sidecar。只看文件名、退出码或JSON中的successful状态不足以获得完整学习状态证明。CUDA标记还要求保存CUDA学习RNG；CPU单元检查不代表GPU恢复已经验证。

每条分支分别用原8701、9701评估seed执行全部50案例，每案例8replica、4001步。保留原物理资格门槛和所有control/stability指标，不替换成生存率。原始NPZ逐数组验证完整shape、dtype、有限数、CRC、实际文件SHA及400行覆盖；核对100Hz时间、episode/reset及终止事件。去reset后200步的稳态指标继续使用原定义，失败窗口、无有效样本或事件保留原有缺失信息。

评估保留高度/速度/姿态偏差、世界平面漂移、回合内波动、腿/轮目标与力矩变化等指标。100Hz末子步采样不能证明下位机电流环波动或总电能；此分支也不能据此宣布sim2real或硬件验证完成。

## 调用

先完成原资格、采集和准备工具的实际核验，再在原始输入树所在主机调用：

```bash
python -B -m transformer_rl.retention_campaign freeze \
  --preparation-protocol /absolute/path/preparation_protocol.json \
  --preparation-directory /absolute/path/prepared_anchors \
  --diagnostic-summary /absolute/path/diagnostic_campaign/summary.json \
  --learning-summary /absolute/path/learning_campaign/summary.json \
  --retention-seed 91001 \
  --device cuda:0 \
  --output-root /absolute/path/new_retention_campaign \
  --protocol-output /absolute/path/retention_campaign_protocol.json

python -B -m transformer_rl.retention_campaign run \
  --protocol /absolute/path/retention_campaign_protocol.json
```

上面的种子只是调用示例，必须与实际训练、资格评估、采集种子独立；路径必须替换成真实输入，协议文件位于新campaign目录外。使用包含PyTorch、NumPy、TensorBoard及真实环境依赖的运行环境。测试只有小型合成CPU环境、真实PPO与OS锁接口，不授予真实教师资格，也未运行真实K/λ分支。
