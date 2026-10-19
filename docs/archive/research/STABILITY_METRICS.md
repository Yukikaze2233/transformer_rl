# Policy-rate 分回合稳定性统计

## 用途与边界

`evaluate_policy` 在原有整段 `metrics`、reward 与 termination/truncation 计数之外，
返回独立的 `report["stability"]`。用于以一致协议比较已完成 MLP 与候选 Transformer
的采样信号偏差和回合内波动。低抖动不能自动推出站稳：常值大偏差、倒下后静止也能得到零抖动。
应联合检查 tracking/bias、全区间物理指标、失败与 reset 计数。

本实现与验证使用 CPU 合成张量；未运行训练或仿真，也未产生新的实机证据。

## 输入协议

新增参数位于原有参数末尾，且仅接受关键字：

```python
evaluate_policy(..., action_clip=None, *, settle_steps=200, min_steady_samples=200)
```

- `settle_steps` 必须为非负 Python `int`，`min_steady_samples` 必须为正 Python `int`；
  `bool` 和浮点数均拒绝。
- `settle_steps` 单位是每个 env、每个 episode 的 **step 样本数**。
  默认值在名义 policy 100 Hz 下对应前 2 s，不代表实机测量，也不根据时间自动换算。
- 每个 `StepResult.info` 可以同时提供：
  - `evaluation_signals: dict[str, Tensor[N]]`：非空字符串名称，浮点、有限的 PRE-reset 信号。
  - `evaluation_signal_time: Tensor[N]`：严格 `float64`、有限，单位秒，表示这些物理信号的采样时间。
    所有信号共享该帧的每 env 时间，张量与环境 `done` 同设备。
- 名称集合从第一次 update 起必须固定；允许字典顺序改变，禁止增加、删除或改名。
  无信号时省略上述两项（或提供空字典和 `None` 时间），返回不可用且 `signals={}`。
  只有时间、没有信号也属于协议错误。
- 时间必须在同一 env/episode 的每次 update 间严格递增，包括暖机期；重复 timestamp 和倒退均拒绝。
  reset 后新 episode 可以从新的时间原点开始。
- 必须在自动 reset **之前** 捕获信号及其时间；禁止直接使用已经 reset 的
  `StepResult.observation.timestamp`。actor/观测可用性事件时间与物理信号时间独立。
  校验可以发现时间倒退/重复，无法证明上游传入的时间真实代表物理采样，语义由环境适配器负责。
- 提供无效值、名称变化、缺时间、错误 shape/dtype/device 时评估抛异常，并通过 `finally` 关闭环境。

## 分段与公式

每个 env 独立维护当前 episode。每次 update：先校验并接收该帧 PRE-reset 信号，
再根据 `terminated | truncated` 结算并清空对应行。未 reset 的其他行连续累计。
每段固定丢弃前 `settle_steps` 个样本，余下连续样本数 `n_e >= min_steady_samples`
才参与可用统计；短段所有值均不参与，计数保留。暖机与第一个保留样本之间不计算导数。

评估结束时，仍有样本的 episode 以 **partial** 段按同样规则结算，不冒充完整回合。
刚 reset 后的空 episode 不计段。完全落在暖机内的非空 episode 计为短段，其 post-settle 样本数为 0。
失败 episode 与正常/截断 episode 使用同样规则，不按成败过滤或挑选低抖动窗口。

对每个可用段 `e`，用 Welford 更新均值 `μ_e` 和中心二阶矩 `M2_e`：

- `mean = Σ(n_e × μ_e) / Σ n_e`：保留样本的 signed mean。
  对误差信号可解释为 bias；对目标、力矩等原始信号不能自动解释为跟踪误差。
- `within_episode_std = sqrt(Σ M2_e / Σ n_e)`：总体标准差口径（不是无偏估计）。
  **先逐 env/episode 去均值再合并**，环境间或回合间均值差异不计为 jitter。
- `max_abs = max |x|`：仅可用段保留样本的最大绝对值，不是去均值峰值。
- `derivative_rms = sqrt(Σ ((x_i - x_{i-1}) / (t_i - t_{i-1}))² / D)`：
  `D = Σ(n_e - 1)`，只计算同一可用段内相邻保留样本；不跨 reset、暖机边界或短段。
  各差分等权，使用真实 `delta_t`，不是持续时间加权，也不假定固定 0.01 s。
  `D=0` 时为 `null`，包括允许 `min_steady_samples=1` 的单样本段。
- `episode_mean_min/max/std`：可用段均值的范围和总体标准差，**每段等权**，独立保留段间漂移。

使用一次 stack 将信号、时间和 done 打包到 CPU，再做在线统计；每个 signal/step 不额外逐项同步 GPU。
空间复杂度随 `num_envs × signal_count` 增长，与评估时长无关。

## 固定返回协议

```text
report['stability'] = {
  'available': bool,
  'protocol': {
    'settle_steps': int,
    'min_steady_samples': int,
    'centering': 'per_environment_episode',
    ...
  },
  'signals': {
    name: {
      'mean': float | None,
      'within_episode_std': float | None,
      'derivative_rms': float | None,
      'max_abs': float | None,
      'count': int,
      'segments': int,
      'short_segments': int,
      ...
    }
  },
  'scope': str
}
```

`available` 表示存在可用段，不表示 policy success。没有任何可用段时，核心四个浮点指标均为
`null`，绝不补零；提供过的信号名称与短段计数仍保留。没有提供信号则 `available=False, signals={}`。

追加字段的精确定义：

| 字段 | 定义 |
| --- | --- |
| `count` / `segments` | 可用段的 post-settle 样本数 / 段数，含符合条件的 partial |
| `short_count` / `short_segments` | 短段的 post-settle 样本数 / 段数 |
| `total_count` | 此信号全部输入样本数，包含暖机、短段、失败段 |
| `settled_count` | 被固定暖机协议丢弃的样本数 |
| `derivative_count` | 可用段内有效相邻样本对数 |
| `completed_segments` / `partial_segments` | 可用段中以 done 结束 / 评估截止结束的段数 |
| `short_completed_segments` / `short_partial_segments` | 短段中以 done 结束 / 评估截止结束的段数 |
| `episode_mean_min/max/std` | 可用段均值的最小值、最大值、段等权标准差，无可用段则 `null` |

满足 `total_count = settled_count + short_count + count`。
所有这些计数均按信号跨环境合计；`segments` 不能当作完成或成功回合数。
`EpisodeSignalStatistics.report()` 会最终结算 partial，可重复调用但不重复计数；结算后禁止继续 update。

## 采样限制与解读

`scope` 明确写入采样限制：100 Hz policy 采样无法测量 MCU **高于 50 Hz** 的波动；
更高频内容可能混叠，低频统计也不等价于完整 MCU 频谱。actor 目标或力矩的 derivative
是每政策采样点间的变化率，不是完整下位机电流环响应、控制器实际应用事件或电机电流纹波。

原 `metrics`、reward、`terminated_count`、`truncated_count`、`done_count` 均保留全区间口径，
包含暖机、倒下及自动 reset。稳定性短段筛选不改变这些统计。比较模型时应固定场景、seed、采样协议，
同时检查样本覆盖、偏差、回合均值漂移和失败统计，不能仅按 `within_episode_std` 排名宣布站稳。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES="" /home/yukikaze/isaacsim60-venv/bin/python -m pytest tests/test_stability.py tests/test_evaluation.py
```

覆盖常值偏差、正负波动、环境间/回合间均值差、reset 跳变、逐回合暖机、变 dt、
不同长度段的 M2/count 合并、有效 partial、短窗口、数值稳定性与错误输入关闭环境。
