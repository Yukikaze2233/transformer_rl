# 实现验证记录

## 敏感性、数值修复和命名场景

新增初始化因子、可选整rollout优化诊断、schema 1/2迁移、敏感性实验和命名评估场景后，完整本机检查为 **805 passed，0 failed，0 skipped，34.69秒**。CI显式获取Git历史，用真实旧实现验证checkpoint兼容性。

真实Kaiser数值复现表明，batch512与4096的均值差约5e-7、KL64约3.4e-12，能在std=0.1时产生约1.7e-5的logp差。修复将moment接近性与各自Gaussian概率自洽分开，原容差、PPO loss和ratio分母保持不变。旧payload离线GPU验收、伪造数据拒绝测试，以及修复后的7项训练/19项评估均通过。见[数值证据](BEHAVIOR_PRECISION.md)、[公平复验](KAISER_SENSITIVITY_RECHECK.md)。

命名场景逐场景统计，不把不同高度当作不同随机seed；默认仅评估最终checkpoint。首档6模型×3训练seed的学习曲线规格已写入配置，后续真实长训进度单独记录，不能把单元测试或80次更新称为收敛。

## 多架构与Kaiser后续验证

2026-09-14增加六变体、辅助学习、检查点迁移、实验编排和独立评估后，本机全仓检查为**532 passed，0 failed，0 skipped，15.86秒**，包含额外SDK进程所有权及worker_module调度回归。默认网络旧检查点的参数顺序、state keys、策略输出与下一次Adam更新已用原实现精确核对。当前包源码SHA `afc83b7a6346b9dd724cb13217fa6f78bbf64297668a66bcf2cb654ac535adf4` 与Kaiser有效pilot一致。

Kaiser真实完成六项各20次PPO更新：合计1,966,080训练transitions、14,400独立评估transitions。正式评估均为单seed、3秒仿真窗口，包含初始瞬态；没有收敛或架构优胜结论。额外通信延迟为零，不构成延迟鲁棒性验收。完整冻结规格、来源哈希、资源及物理指标见[实测记录](KAISER_EXPERIMENTS.md)。

**当前最有用的优化诊断**：统一learning_rate=1e-4、target_kl=0.01时，time/index/supervised三项每次PPO更新仅执行一次optimizer.step，gated/MLP/GRU分别累计116/120/160步。相同采样预算不等于相同梯度计算预算；下一阶段应单独核查学习率、输出初始化和分布尺度的敏感性，再作多seed收敛比较，不用短测回报选赢家。

以下保留初版实现验证记录。

日期：2026-09-13。范围为网络、优化器、采集接口、时序队列、检查点、导出和CLI编排。没有运行机器人训练、真实仿真或收敛benchmark；测试中的单次合成梯度更新和预定义tensor transition仅用于验证实现。

## 最终结果

**完整测试：394 passed，0 failed，0 skipped，6.76秒。**

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/yukikaze/isaacsim60-venv/bin/python -m pytest -q -ra
```

该解释器仅提供PyTorch/pytest/ONNX等依赖，测试没有启动Isaac应用。CUDA设备为RTX 4060 Laptop 8GB，PyTorch为2.11.0+cu128。包含24项CUDA测试，覆盖网络梯度、history reset、延迟通道和optimizer迁移。此结果不是4090吞吐或硬件实时性测试。

此前独立CPU栈为Python3.12.13、Torch2.11.0+cpu、pytest9.1.1、ONNX1.23.0rc1、ONNX Runtime1.30.0；当时全套为368 passed、24 CUDA skips。后续添加两项预处理身份检查并在完整CUDA可用环境完成上述394项测试；受影响的检查点/导出/CLI子集另有88 passed、1 CUDA skip记录。

## 关键验证

| 方面 | 已验证内容 |
|---|---|
| 时序网络 | 因果隔离、当前query命令不改历史表示、padding内容不影响有效输出、梯度贯通 |
| 时间精度 | float64先求差、绝对时间平移、大uptime下的毫秒差异、未知age不冒充零延迟 |
| 历史 | partial reset、同tick幂等、同时间不同数据拒绝、snapshot不别名可变buffer |
| PPO | 手算GAE、terminal/timeout双mask、value clipping、raw/issued区分、KL与行为分布完整核对 |
| 数值 | nonfinite loss/gradient在optimizer step前拒绝；有效padding与storage约定统一 |
| 采集 | reset前final state bootstrap、跨collect连续历史、in-place环境buffer隔离、停止回调 |
| 延迟通道 | 零/分数周期延迟、各env不同lag、乱序/latest-wins、保持目标、partial reset、原子溢出拒绝 |
| 检查点 | weights-only、配置/shape/dtype/finite校验、Adam恢复、预处理buffer与声明配置一致、无覆盖发布 |
| ONNX | opset17、五输入契约、动态batch、固定history、8类合成历史的CPU ORT对齐 |
| CLI | inspect不创建环境；用惰性fake factory检查编排、信号边界、失败回执与已有run保护 |

## 架构检查与构建

`python -m transformer_rl inspect --config configs/control.json`实际输出：

- frame dimension：30。
- actor parameters：69,772。
- critic parameters：48,897。
- environment_started：false。

`uv build --wheel --out-dir /tmp/opencode`成功生成wheel。首次尝试通过SDK解释器执行`python -m pip wheel`时，该解释器没有pip；随后使用独立uv构建完成，未向SDK环境安装pip或替换依赖。

`actionlint .github/workflows/tests.yml`通过。CI配置是Linux CPU测试，CUDA专项由上述本机检查覆盖。新仓库尚未在远端CI执行。

## 验证边界

- 尚未接入经核实的机器人USD/URDF、真实通信协议和下位机闭环模型。
- `TensorEnvAdapter`是明确的tensor接口桥接，不自动提供机器人任务，也不补造缺失的pre-reset终态。
- 候选policy频率与`TimingProfile`是配置，不是实机周期测量。
- 尚无训练收敛、站立/行驶/抗推、高低切换或sim2real结果。
- 尚无目标部署设备端到端p95/p99时延；参数量和MAC算术不能代替该测量。
- 检查点发布的文件系统行为已在Linux验证；Windows原生行为尚未验证。
