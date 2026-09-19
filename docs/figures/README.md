# README 图示与数据

## 架构图

- [control-transformer.svg](control-transformer.svg)：`TimeAwareActor` 的 Last-token / Index / Additive 路径，含 `_CausalBlock` 的 Pre-LN、因果mask及残差连接。
- [estimator-controller.svg](estimator-controller.svg)：`EstimatorActor` 与 `HistoryEstimator`，区分部署计算、PPO冻结、独立辅助更新和训练专用critic。

图示根据 [`model.py`](../../src/transformer_rl/model.py)、
[`estimation.py`](../../src/transformer_rl/estimation.py)、
[`optimized_control.json`](../../configs/optimized_control.json) 与
[`estimator_comparison.json`](../../configs/estimator_comparison.json)绘制。
SVG为可编辑矢量源码，使用中文字体及常用技术术语，无外链图片或脚本。
图中的有限窗口编码器每次重新计算历史，不隐含流式KV cache。

30维帧由16维本体观测、3维指令、6维上一条issued action、2维传感器年龄、
2维年龄已知标志和1维策略间隔组成。估计器另拼接elapsed age和valid得到每帧32维；
velocity控制头输入30+3=33维，context为30+4+16=50维。
Transformer估计器采用slot-index编码；当前帧反馈保持直接路径。

## 结果图口径

| 图片 | 数据 | 统计方式 |
|---|---|---|
| [training-and-height.png](training-and-height.png) | [training.csv](data/training.csv)、[evaluation.csv](data/evaluation.csv) | 训练先逐seed做最多25次更新的后向均值，再取3-seed均值与样本SD；每8次更新取一点，含第1/977次。高度图显示单seed与均值±SD |
| [command-response.png](command-response.png) | [evaluation.csv](data/evaluation.csv) | 独立固定指令场景，3个训练seed等权均值；每次reset后去掉前200步。连线连接不同场景，不是时间轨迹 |
| [wiring-height.png](wiring-height.png) | [wiring.csv](data/wiring.csv) | 单训练seed、8-env实测高度均值，每5步取一点并保留终点；无平滑。灰区是前2秒；10秒片段全部为截尾观测 |

SD使用`ddof=1`，不是置信区间。训练reward不是任务成功率，稳态均值也不能证明无漂移。
首轮每项977次更新对应16,007,168条完整rollout样本；额外中断采样另见
[完整回收记录](../KAISER_ARCHITECTURE_COMPLETE.md)。
Supervised的seed 1011/1022分别从789/732次更新恢复，曲线按累计更新拼接，环境与历史重置。

首轮图包含21个最终模型的147份场景报告，评估seed为301；短训图对应seed 2003、
checkpoint 80、评估seed 3003，仅三个完整评估，不将中断或未启动项补成曲线。
汇总CSV是文档用的轻量数据，不包含训练权重和原始日志。

## 重绘

在仓库根目录运行。需要NumPy、Matplotlib及 **Noto Sans CJK SC** 字体：

```bash
python -m pip install numpy matplotlib
python tools/readme_figures.py render
```

`render`只读已提交CSV，在校验SHA后生成三张PNG，不需要Isaac、GPU或回收归档。
字体或Matplotlib版本变化可能影响像素排版，不改变CSV统计值。

如果本地保存了原始回收目录，可以重新验证源文件并提取CSV：

```bash
python tools/readme_figures.py prepare \
  --round1 artifacts/recovered/architecture-16m-20260915T0540/recovery-complete-20260917T1820Z \
  --wiring artifacts/recovered/estimator-results-20260918T1010Z
python tools/readme_figures.py render
```

[provenance.json](data/provenance.json)记录固定回收manifest的SHA、全部读取文件的SHA、
CSV的SHA及抽样／聚合方式。提取时验证更新序列连续性、最终checkpoint与评估绑定、
轨迹SHA及reset标志，再生成数据。两个命令仅处理文档工件，不执行模型推理、训练或仿真。
