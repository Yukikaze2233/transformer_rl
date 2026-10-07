# 嵌套锚点准备工具

`tools/prepare_nested_anchors.py` 仅支持当前严格课程分析器提供的 **Gated H31、pretrain 分支、CP400** 教师资格。它不收集数据、不调用模拟器、不启动 K/λ 训练，也不代表其他网络或 H 的对比已经完成。λ、分支预算及训练状态只是未执行的描述符；真实 λ 实验配置尚未选定。测试中的 λ 数值属于合成测试。

## 资格与输入

资格来自 `tools/analyze_curriculum_retention.py::analyze` 对原始源代码、snapshot、训练、checkpoint、完整案例网格和评估收据的重新核验。每个教师/案例必须在8701、9701两次 CP400 评估中分别通过原始物理门槛、episode 数及 steady 条件。同一教师的两个 checkpoint SHA 必须一致。缓存 analysis、`success_rate=1` 或自行填写的 `eligible` 均不能代替资格。

完整准备范围保留原协议的所有训练种子和旧10案例。当前三训练种子对应30个教师/案例；加入两种 K、两种 λ 后分母为120个准备单元及12条未执行分支。缺任一资格、采集来源文件、池或足量样本时，整份准备返回 `not_ready`，保留完整分母，**不创建输出目录、不发布锚点**。原始封存协议与源文件必须可审计；源、收据或身份被修改则拒绝，不能凭缓存绕过。

原始 v1 池必须含有限 float32 `frames[N,31,F]`、`mean[N,A]`、正 `std[N,A]`，且 policy、control、teacher checkpoint 与已审计身份一致。`mean` 保持 clamp 前的 Gaussian mean；不更改原始标签。所有池使用同一个预先固定、独立于训练和资格评估的采集种子；每个教师/案例必须有各自的采集报告。同一池路径、SHA 或报告路径不得跨教师/案例复用，重新序列化后完全相同的 tensor 内容也不能证明独立采样，保守拒绝。采集种子不得与训练或资格评估种子重合。报告必须核验案例环境、模型、snapshot、contract、采集种子及池收据。

首版采集来源仅支持直接单案例 `evaluate_frame_policy`。真实 `environment_provenance.contract_sha256` 是有效 contract 的 canonical 内容 SHA，须从冻结 snapshot 的已核验原始 contract 重建；它与 `environment.contract_sha256` 的原始文件 SHA 含义不同。`evaluate_suite` 的分组报告保留父 merged-contract provenance，当前拒绝此类池；后续需要额外冻结并审计完整父批次布局后才能支持，不能随意接受父 SHA。

v1 池本身没有案例、采集时间或 episode 年龄字段，因此必须通过冻结的采集报告和文件 SHA 补足来源链。池可能包含 reset 与 transient 窗口，工具不会声称只采集 steady 状态，也不会裁剪 H 或跨架构复用教师。

## K 与发布

每个固定池使用版本化 SHA256 排序产生主 permutation，输入只包括池 SHA、采集种子、案例及原始行索引，**不包括 K 或 λ**。小 K 的 indices 必须是大 K 的前缀。收据分别记录 requested K 和 actual N；若 N 小于任一请求 K，则该容量对比不齐，整份准备仍为 `not_ready`，不会把 N17 声称为合格 K256/K512。

全部资格与池齐备后，锚点和索引先在同一文件系统中 staging、fsync，再独占硬链接至新目录；manifest 最后发布。多文件目录不保证崩溃原子性，消费者必须先通过 `validate_preparation`，不能根据残留 `.pt` 文件启动训练。已有目录、文件及 dangling symlink 均不会被覆盖；validator 重新核验输入、源代码、分支、索引、tensor SHA、输出清单与 manifest 原始字节。

ready 分支绑定同一训练种子的已审计完整 CP400 checkpoint 路径、SHA、消耗400次更新及其 transition 时钟。它们仍只是描述符：Adam/RNG/clock 恢复执行、独立 retention RNG、phase-B executor、其他架构/H 资格与采集 provider 尚待完成，不能把锚点准备成功称为公平续训已经实现。

## 使用

先执行无写入检查，协议指向的原始输入树必须在执行主机真实可读：

```bash
python tools/prepare_nested_anchors.py assess \
  --qualification-protocol /absolute/path/qualification_protocol.json
```

`freeze` 必须显式提供 capacities、coefficients、anchor seed 和可选 pool receipts；不默认选择 λ。pool receipts 是 JSON 数组，每项为 `training_seed`、`case`、`pool`、`report`，后两项均含绝对 `path`、`sha256`、`bytes`。随后 `prepare --protocol ... --output-directory ...` 返回完整准备状态，成功发布后使用 `validate --protocol ... --output-directory ...`。

冻结计划和准备输出必须位于原始 experiment 之外。仅复制远程协议或缓存收据到本地不能完成原始输入重验。只读内存 readiness 证明须明确记录实际执行源码 SHA、执行函数及传输方式；它不是已部署 CLI 或真实锚点发布。
