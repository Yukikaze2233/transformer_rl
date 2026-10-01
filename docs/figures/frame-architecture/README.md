# 网络架构图

四张图仅介绍网络与部署数据流：三类 Actor 总览、默认 Transformer 编码器、多种读出及门控残差、部署组件。公共观测为 35D，输出为 6D，历史窗口为 31 帧，策略频率为 100 Hz；默认 Transformer 使用 128D、2 层、4 头和 512D FFN。

图示采用白底、蓝色标题、细线箭头及淡色虚线分组。中文使用 Noto Sans CJK SC，英文及数字使用 Times New Roman；公式变量斜体，函数、数字与符号正体，上下标单独排版。SVG 为可编辑源图，PNG 是两倍尺寸的文档插图。

在安装上述字体与 `rsvg-convert` 的环境重绘：

```bash
python docs/figures/frame-architecture/render.py
```

尺寸与数据路径对应 `src/transformer_rl/frame_policy.py`、`frame_training.py`、`frame_runtime.py` 和 `chassis_adapter.py`。图中没有性能排名、训练流程或实验结果。
