"""Render the three-family packed-policy architecture in the document style."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = spec_from_file_location("diagram_style", ROOT.parent / "policy-network-overview/render.py")
style = module_from_spec(spec)
spec.loader.exec_module(style)
style.ROOT = ROOT
Figure, variable = style.Figure, style.variable


def labeled_box(fig, x, y, width, lines, height=90, stroke="#2D6198"):
    fig.box(x, y, width, height, stroke=stroke)
    first = y + height / 2 - 16 * (len(lines) - 1) + 9
    for index, line in enumerate(lines):
        fig.text(x + width / 2, first + index * 32, line, 26, anchor="middle")


def render():
    fig = Figure(1800, 1150)
    fig.text(55, 82, "网络架构：三个家族的统一对照", 54, color=style.BLUE, bold=True)
    fig.text(60, 137, "公共观测 35D · 策略 100 Hz · 历史 31 帧 / 0.30 s · 独立特权 Critic 81D", 28)
    fig.box(55, 168, 1690, 755, fill="#FBFAFD", stroke="#8254B7", dashed=True)
    fig.text(1705, 207, "Actor / 部署", 29, anchor="end", color="#8254B7")

    fig.text(85, 246, "单帧 MLP", 31, bold=True)
    fig.text(85, 280, "当前帧映射", 25)
    labeled_box(fig, 275, 229, 215, ["当前观测", "35D"])
    labeled_box(fig, 590, 229, 565, ["Actor：全连接网络", "35 → 256 → 128 → 64 → 6"])
    labeled_box(fig, 1365, 229, 270, ["动作均值", "四路腿位 / 两路轮速"])
    fig.arrow("490,274 590,274")
    fig.arrow("1155,274 1365,274")
    fig.math(1240, 260, variable("μ", "t"), 32)

    fig.text(85, 435, "历史 MLP", 31, bold=True)
    fig.text(85, 471, "历史编码", 25)
    labeled_box(fig, 275, 418, 215, ["历史观测", "31 × 35"])
    labeled_box(fig, 575, 418, 420, ["历史 MLP 编码器", "1085 → 128 → 64 → 3"])
    labeled_box(fig, 1090, 418, 145, ["latent", "3D"])
    labeled_box(fig, 1320, 418, 360, ["当前帧 + latent → Actor", "38 → 128 → 64 → 32 → 6"])
    fig.arrow("490,463 575,463")
    fig.arrow("995,463 1090,463")
    fig.arrow("1235,463 1320,463")
    labeled_box(fig, 575, 546, 255, ["当前观测 35D"], height=55)
    fig.arrow("830,573 1270,573 1270,488 1320,488")
    fig.text(865, 560, "当前帧直连", 24)
    fig.text(1090, 550, "PPO 学习隐式表示", 23, color="#555555")

    fig.text(85, 693, "我们的", 31, bold=True)
    fig.text(85, 729, "Transformer", 25)
    labeled_box(fig, 275, 672, 215, ["历史观测", "31 × 35"])
    labeled_box(fig, 575, 660, 420, ["逐帧投影 + 位置编码", "2 层因果 Pre-LN Transformer", "d = 128，4 头，FFN = 512"], height=114)
    labeled_box(fig, 1090, 660, 145, ["历史读出", "128D"], height=114)
    labeled_box(fig, 1320, 660, 360, ["当前帧 + latent → Actor", "163 → 256 → 128 → 6"], height=114)
    fig.arrow("490,717 575,717")
    fig.arrow("995,717 1090,717")
    fig.arrow("1235,717 1320,717")
    labeled_box(fig, 575, 825, 255, ["当前观测 35D"], height=55)
    fig.arrow("830,852 1270,852 1270,749 1320,749")
    fig.text(860, 839, "当前帧直连", 24)
    fig.text(1090, 811, "末帧 / 当前帧 Query", 23)
    fig.text(1320, 863, "普通 / 门控残差；宽度与层数可扩展", 22)

    fig.box(55, 965, 1690, 136, fill="#FFFCF0", stroke="#D9AA27", dashed=True)
    labeled_box(fig, 120, 993, 260, ["当前特权观测 81D"], height=75)
    labeled_box(fig, 460, 993, 545, ["独立 Critic：81 → 256 → 128 → 64 → 1"], height=75)
    labeled_box(fig, 1100, 993, 195, ["价值 / GAE"], height=75)
    labeled_box(fig, 1395, 993, 270, ["统一 PPO 更新"], height=75)
    fig.arrow("380,1030 460,1030")
    fig.arrow("1005,1030 1100,1030")
    fig.arrow("1295,1030 1395,1030")
    fig.arrow("1665,1029 1720,1029 1720,924", dashed=True)
    fig.text(1380, 954, "策略梯度更新 Actor", 24, color="#AC2727")
    fig.text(60, 1135, "固定任务、奖励、Critic 与采样预算；先完成任务，再比较平稳性、遗忘、吞吐与部署延迟。", 25)
    fig.save("architecture")


if __name__ == "__main__":
    render()
