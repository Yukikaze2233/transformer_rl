# 35D网络设计图

这些图对应2026-09-29的代码审阅快照，用于[网络架构](../../POLICY_ARCHITECTURE.md)及项目参数、时延记录。图示区分独立网络核心、PPO训练路径和研究机制；不表示新网络已经完成机器人训练。[通用网络比较](../../NETWORK_COMPARISON.md)使用另行维护的[通用配图](../policy-network-overview/README.md)，不沿用这里的项目参数和实测排名图。

| 目录 | 内容 | 数据性质 |
| --- | --- | --- |
| `network` | 历史编码器、Actor、Critic及训练/部署边界 | 接入设计；动作均值网络参数已核对 |
| `encoder` | 35→96嵌入、两层因果注意力、末token读出 | 已实现的普通Transformer结构 |
| `training` | 在线PPO采样与参数更新 | 接入设计，仿真器不参与反向传播 |
| `reward` | V6高度通道的广域成本与局部精核 | 解析函数切片，不是训练曲线 |
| `curriculum` | 六阶段课程与旧能力验收 | V6课程合同；预算是上限而非完成进度 |
| `families` | MLP、堆帧MLP、GRU/TCN、Transformer | 明确区分当前35D实现和候选路线 |
| `costs` | 参数量与CPU前向P99 | 参数实算和本机短测，不是闭环质量排名 |

## 字体与文件

采用白底、蓝色大标题、细黑框箭头及淡色虚线分组。中文使用SimHei或Noto Sans CJK SC；数学变量使用Times New Roman斜体，数字及运算符使用正体。数学上下标显式排版，特殊数学字形可由STIX补全。

每幅图保留可编辑的`diagram.svg`及用于文档的高清`diagram.png`。飞书使用PNG以保留字体与上下标；不将SVG转换出的画板节点作为最终交付，因为画板解析器会替换字体及重排数学文本。仓库不分发字体文件。

## 重绘

需要本机已安装上述字体、fontconfig和librsvg的`rsvg-convert`。从本目录运行，例如：

```bash
rsvg-convert --zoom 2 --output network/diagram.png network/diagram.svg
```

其余结构图同样以SVG为源，用2倍尺寸导出。请在修改后检查PNG中的上下标、箭头、边界、长中文行和四周留白。不同渲染器的字体回退可能不同，不应用另一渲染器未经检查地替换最终图片。

Reward图使用NumPy和Matplotlib，运行：

```bash
python reward/plot.py
```

脚本只计算解析函数，验证零误差处成本为零，生成PNG和SVG，不启动训练或仿真。参数量、短测条件、Reward及课程的代码证据见上层两篇文档以及[结构与能力保持设计](../../FRAME_POLICY_DESIGN.md)。短测未保存逐调用原始时延样本，图中的P99不能当作最坏时延保证。
