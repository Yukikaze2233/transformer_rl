# 三个家族的训练网络架构图

图示对应 `network_variants()` 中的单帧 MLP、历史 MLP 小瓶颈与默认 Transformer，观测为 35D，历史为 31 帧，策略频率为 100 Hz。图中尺寸来自配置，不表示性能排名或训练结果。

白底、蓝色标题、细框箭头和淡色虚线分组沿用参考图格式。中文使用 Noto Sans CJK SC，英文、数字和公式使用 Times New Roman；数学变量斜体，上下标单独排版。源文件是 SVG，文档嵌入两倍尺寸的 PNG。

需要上述字体及 `rsvg-convert`，重绘命令：

```bash
python docs/figures/frame-training/render.py
```
