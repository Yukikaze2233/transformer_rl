"""Render architecture-only figures with explicit Chinese and math typography."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("diagram_style", ROOT.parent / "policy-network-overview/render.py")
style = module_from_spec(spec)
spec.loader.exec_module(style)
style.ROOT = ROOT
Figure, variable = style.Figure, style.variable
BLUE, PURPLE, GOLD = style.BLUE, "#8254B7", "#D5A724"


def box(fig, x, y, width, lines, height=94, size=27):
    fig.box(x, y, width, height, stroke="#2D6198")
    start = y + height / 2 - 17 * (len(lines) - 1) + 10
    for index, line in enumerate(lines):
        fig.text(x + width / 2, start + index * 34, line, size, anchor="middle")


def symbol_box(fig, x, y, width, label, symbol, dimensions, height=112):
    fig.box(x, y, width, height, stroke="#2D6198")
    center = x + width / 2
    fig.text(center, y + 31, label, 26, anchor="middle")
    fig.math(center, y + 69, symbol, 33)
    fig.text(center, y + 99, dimensions, 26, anchor="middle")


def title(fig, text, subtitle):
    fig.text(55, 83, text, 54, color=BLUE, bold=True)
    fig.text(60, 138, subtitle, 28)


def overview():
    fig = Figure(1800, 1190)
    title(fig, "网络架构：当前状态与历史表示", "35D 公共观测 · 6D 动作输出 · 100 Hz 策略 · 31 帧 / 0.30 s 历史")
    fig.box(55, 174, 1690, 739, fill="#FBFAFD", stroke=PURPLE, dashed=True)
    fig.text(1705, 214, "Actor / 部署网络", 29, anchor="end", color=PURPLE)

    fig.text(85, 267, "单帧 MLP", 30, bold=True)
    fig.text(85, 305, "当前帧映射", 26)
    symbol_box(fig, 280, 236, 220, "当前观测", variable("o", "t"), "35D")
    box(fig, 605, 245, 575, ["Actor：全连接网络", "35 → 256 → 128 → 64 → 6"])
    symbol_box(fig, 1360, 236, 285, "动作均值", variable("μ", "t"), "6D")
    fig.arrow("500,292 605,292")
    fig.arrow("1180,292 1360,292")

    fig.text(85, 477, "历史 MLP", 30, bold=True)
    fig.text(85, 514, "固定窗口编码", 26)
    symbol_box(fig, 280, 435, 220, "历史观测", variable("H", "t"), "31 × 35")
    box(fig, 570, 444, 405, ["展平 → 历史编码器", "1085 → 128 → 64 → 3"], size=26)
    symbol_box(fig, 1045, 435, 155, "隐变量", variable("z", "t"), "3D")
    box(fig, 1270, 444, 410, ["当前帧 + 隐变量 → Actor", "38 → 128 → 64 → 32 → 6"], size=25)
    fig.arrow("500,491 570,491")
    fig.arrow("975,491 1045,491")
    fig.arrow("1200,491 1270,491")
    box(fig, 570, 585, 290, ["当前观测 35D"], height=59)
    fig.arrow("860,614 1228,614 1228,520 1270,520")
    fig.text(905, 599, "当前帧直连", 25)
    fig.text(1270, 607, "扩展方案：16D 隐变量", 25)

    fig.text(85, 726, "我们的", 30, bold=True)
    fig.text(85, 766, "Transformer", 26)
    symbol_box(fig, 280, 692, 220, "历史观测", variable("H", "t"), "31 × 35")
    box(fig, 570, 687, 405, ["逐帧投影 + 位置编码", "2 层因果 Transformer", "128D · 4 头 · FFN 512D"], height=122, size=26)
    symbol_box(fig, 1045, 692, 155, "历史读出", variable("z", "t"), "128D")
    box(fig, 1270, 701, 410, ["当前帧 + 隐变量 → Actor", "163 → 256 → 128 → 6"], size=25)
    fig.arrow("500,748 570,748")
    fig.arrow("975,748 1045,748")
    fig.arrow("1200,748 1270,748")
    box(fig, 570, 838, 290, ["当前观测 35D"], height=59)
    fig.arrow("860,867 1228,867 1228,777 1270,777")
    fig.text(905, 852, "当前帧直连", 25)
    fig.text(1270, 850, "末帧 / Query 读出；普通 / 门控残差", 24)

    fig.box(55, 966, 1690, 154, fill="#FFFCF0", stroke=GOLD, dashed=True)
    fig.text(1704, 951, "独立 Critic", 28, anchor="end", color="#9F7816")
    symbol_box(fig, 145, 991, 285, "特权状态", variable("s", "t") + '<tspan baseline-shift="super" font-size="65%" font-style="normal">priv</tspan>', "81D")
    box(fig, 535, 1000, 670, ["价值网络：全连接网络", "81 → 256 → 128 → 64 → 1"])
    symbol_box(fig, 1340, 991, 265, "价值标量", '<tspan font-style="italic">V</tspan>(' + variable("s", "t") + ')', "1D")
    fig.arrow("430,1047 535,1047")
    fig.arrow("1205,1047 1340,1047")
    fig.text(60, 1170, "Actor 与 Critic 不共享编码器；实机保留观测缓存、完整 Actor 与动作映射。", 27)
    fig.save("overview")


def encoder():
    fig = Figure(1800, 1100)
    title(fig, "Transformer：从历史窗口到动作", "默认结构：31 帧 · 128D 表示 · 2 层 Pre-LN · 4 头注意力 · 末帧读出")
    fig.box(55, 177, 1690, 268, fill="#FBFAFD", stroke=PURPLE, dashed=True)
    symbol_box(fig, 90, 236, 235, "历史输入", variable("H", "t"), "[B, 31, 35]")
    box(fig, 415, 236, 325, ["逐帧 Linear：35 → 128", "+ 固定位置编码"], height=112, size=26)
    box(fig, 835, 236, 365, ["因果 Transformer × 2", "[B, 31, 128]"], height=112)
    box(fig, 1295, 236, 345, ["LayerNorm → 最后 token", "历史表示 128D"], height=112, size=25)
    fig.arrow("325,292 415,292")
    fig.arrow("740,292 835,292")
    fig.arrow("1200,292 1295,292")
    fig.text(105, 402, "一个 token 对应一整帧观测；只关注当前位置及过去位置，不读取未来。", 28)

    fig.box(55, 495, 1690, 326, fill="#EEF4FC", stroke="#A2BAD4", dashed=True)
    fig.text(87, 548, "单层编码器展开", 31, color=BLUE, bold=True)
    boxes = [(230, 180, ["LayerNorm"]), (468, 278, ["因果多头注意力", "4 头 × 32D"]),
             (800, 90, ["+"]), (953, 180, ["LayerNorm"]),
             (1195, 300, ["FFN / GELU", "128 → 512 → 128"]), (1555, 90, ["+"])]
    for x, width, lines in boxes:
        box(fig, x, 615, width, lines, height=96, size=27)
    fig.math(120, 674, variable("X") + '<tspan baseline-shift="super" font-size="70%" font-style="italic">ℓ</tspan>', 39)
    fig.arrow("155,663 230,663")
    for points in ["410,663 468,663", "746,663 800,663", "890,663 953,663",
                   "1133,663 1195,663", "1495,663 1555,663", "1645,663 1705,663"]:
        fig.arrow(points)
    fig.arrow("183,663 183,582 845,582 845,615")
    fig.arrow("919,663 919,764 1600,764 1600,711")
    fig.text(444, 568, "残差直连", 25)
    fig.text(1222, 797, "残差直连", 25)

    fig.box(55, 867, 1690, 177, fill="#FFFCF0", stroke=GOLD, dashed=True)
    symbol_box(fig, 100, 900, 245, "当前观测", variable("o", "t"), "35D")
    symbol_box(fig, 410, 900, 245, "历史表示", variable("z", "t"), "128D")
    box(fig, 730, 909, 295, ["拼接 35 + 128", "163D"])
    box(fig, 1100, 909, 335, ["Actor / ELU", "163 → 256 → 128 → 6"])
    symbol_box(fig, 1510, 900, 175, "均值输出", variable("μ", "t"), "6D")
    fig.arrow("345,956 382,956 382,884 705,884 705,932 730,932")
    fig.arrow("655,956 730,980")
    fig.arrow("1025,956 1100,956")
    fig.arrow("1435,956 1510,956")
    fig.text(60, 1083, "零 Dropout；输出层为线性；完整窗口前向计算；历史缓存由调用层维护。", 27)
    fig.save("encoder")


def variants():
    fig = Figure(1800, 1190)
    title(fig, "Transformer 变体：读出与残差", "共享 31 帧、128D、2 层、4 头编码器；改变历史信息进入动作头的方式")
    fig.box(55, 181, 1690, 233, fill="#FCFAED", stroke="#CABC83", dashed=True)
    fig.text(90, 237, "末帧读出", 33, bold=True)
    box(fig, 90, 273, 350, ["编码后历史 tokens", "[B, 31, 128]"])
    box(fig, 530, 273, 330, ["取最后 token", "128D"])
    box(fig, 955, 273, 310, ["拼接当前帧 35D", "163D"])
    box(fig, 1360, 273, 330, ["Actor", "163 → 256 → 128 → 6"])
    for a in ["440,320 530,320", "860,320 955,320", "1265,320 1360,320"]:
        fig.arrow(a)

    fig.box(55, 458, 1690, 337, fill="#EEF4FC", stroke="#A2BAD4", dashed=True)
    fig.text(90, 514, "当前帧 Query 读出", 33, bold=True)
    box(fig, 90, 552, 350, ["编码后历史 tokens", "Key / Value：128D"])
    box(fig, 90, 693, 350, ["当前帧 35D → Query 128D"], height=62, size=25)
    box(fig, 565, 566, 375, ["4 头 Query 注意力", "+ Query 残差 / LayerNorm"], height=115, size=26)
    box(fig, 1040, 576, 300, ["历史读出 128D", "+ 当前帧 35D"])
    box(fig, 1435, 576, 250, ["Actor", "163D → 6D"])
    fig.arrow("440,599 565,599")
    fig.arrow("440,724 497,724 497,650 565,650")
    fig.arrow("940,623 1040,623")
    fig.arrow("1340,623 1435,623")
    fig.text(565, 750, "当前观测生成 Query，向已观测历史选择相关特征。", 26)

    fig.box(55, 839, 1690, 291, fill="#F5F0FA", stroke="#B49ACA", dashed=True)
    fig.text(90, 895, "门控残差", 33, bold=True)
    box(fig, 90, 931, 370, ["输入特征", "注意力或 FFN 输出"])
    fig.math(370, 969, variable("x"), 31)
    fig.math(418, 1003, variable("y"), 31)
    box(fig, 565, 931, 505, ["深度方向门控融合", "保留输入 + 写入候选特征"])
    box(fig, 1180, 931, 505, ["下一子层 / 下一编码层", "末帧读出 → 163D Actor"])
    fig.arrow("460,978 565,978")
    fig.arrow("1070,978 1180,978")
    fig.math(920, 1090, 'G(' + variable("x") + ', ' + variable("y") + ') = (1 − ' + variable("α") + ') ⊙ ' + variable("x") + ' + ' + variable("α") + ' ⊙ ' + variable("h̃"), 37)
    fig.text(60, 1171, "门控发生在单次前向的编码层内；各变体均使用固定历史窗口，不维护跨调用循环隐状态。", 27)
    fig.save("variants")


def deployment():
    fig = Figure(1800, 930)
    title(fig, "部署结构：历史缓存与确定性 Actor", "策略周期 10 ms；历史窗口保存过去数据；完整 Actor 导出为 TorchScript / ONNX")
    fig.box(55, 190, 1690, 575, fill="#FBFAFD", stroke=PURPLE, dashed=True)
    box(fig, 100, 261, 280, ["观测组装与缩放", "当前 35D"])
    box(fig, 475, 261, 295, ["外部历史缓存", "31 × 35 · 旧 → 新"])
    box(fig, 865, 261, 355, ["Transformer 编码器", "历史表示 128D"])
    box(fig, 1315, 261, 330, ["当前帧 + 历史表示", "Actor：163D → 6D"])
    fig.arrow("380,308 475,308")
    fig.arrow("770,308 865,308")
    fig.arrow("1220,308 1315,308")
    fig.arrow("420,308 420,413 1270,413 1270,333 1315,333")
    fig.text(865, 443, "当前帧直连", 27)
    symbol_box(fig, 1280, 551, 350, "确定性动作均值", variable("μ", "t"), "6D")
    box(fig, 805, 560, 380, ["执行限幅与物理量映射", "四路腿位 · 两路轮速"])
    box(fig, 300, 560, 400, ["目标接口 → 底层控制器", "100 Hz 目标 / 1 kHz PD"])
    fig.arrow("1480,355 1480,551")
    fig.arrow("1280,607 1185,607")
    fig.arrow("805,607 700,607")
    fig.arrow("805,656 763,656 763,725 100,725 100,355", dashed=True)
    fig.text(135, 707, "上一条 issued 动作回填到下一帧；控制器接收与物理响应由各自接口记录。", 25)
    fig.text(90, 497, "缓存初始化：重复首帧；网络前向无可变 KV cache 或跨调用隐状态。", 27)
    fig.box(55, 810, 1690, 75, fill="#FFFCF0", stroke=GOLD, dashed=True)
    fig.text(90, 858, "Critic、探索标准差不进入部署图；输入 float32，历史长度与特征顺序由模型元数据确定。", 27)
    fig.save("deployment")


if __name__ == "__main__":
    overview()
    encoder()
    variants()
    deployment()
