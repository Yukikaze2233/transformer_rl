# 实现验证记录

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
