"""Draw the implemented actor with paper-style attention and residual blocks.

Only the standard library and rsvg-convert are used. Dimensions are transcribed
from formal/policy.py; the model and simulator are never imported.
"""
from html import escape
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent
WIDTH, HEIGHT = 2280, 2050
INK, MUTED = "#16181b", "#50555d"
COLORS = {
    "attention": "#ffe0b5", "ffn": "#bfe3ec", "norm": "#f7f2bd",
    "embedding": "#f6dce2", "linear": "#dedff0", "gate": "#d4e8d3",
    "panel": "#f2f3f4",
}
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
    f'viewBox="0 0 {WIDTH} {HEIGHT}">',
    "<title>Structured Transformer Actor: implemented spatial-temporal policy</title>",
    "<desc>Five raw frames enter a shared spatial encoder and two causal temporal "
    "blocks. Four actuator queries and one state query read temporal memory. "
    "Current raw features and estimated state feed shared leg and wheel heads. "
    "Only training uses the independent critic and auxiliary prediction head.</desc>",
    '<rect width="100%" height="100%" fill="white"/>',
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0 0L10 5L0 10Z" fill="{INK}"/></marker></defs>',
]


def text(x, y, value, size=26, *, anchor="middle", bold=False, color=INK,
         rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    parts.append(
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{600 if bold else 400}" fill="{color}"{transform} '
        'font-family="Arial,Noto Sans CJK SC,sans-serif">'
        f'{escape(value)}</text>'
    )


def rect(x, y, width, height, color="white", *, radius=15, panel=False,
         dashed=False):
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    stroke = "none" if panel else INK
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{COLORS.get(color, color)}" '
        f'stroke="{stroke}" stroke-width="2.6"{dash}/>'
    )


def box(x, y, width, height, lines, color="white", size=26):
    rect(x, y, width, height, color)
    leading = size * 1.36
    baseline = y + height / 2 - leading * (len(lines) - 1) / 2 + size * .34
    for index, line in enumerate(lines):
        text(x + width / 2, baseline + index * leading, line, size)


def wire(*points, arrow=True, dashed=False, color=INK):
    coordinates = " ".join(f"{x},{y}" for x, y in points)
    dash = ' stroke-dasharray="7 6"' if dashed else ""
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    parts.append(
        f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
        f'stroke-width="2.6" stroke-linejoin="round"{marker}{dash}/>'
    )


def dot(x, y):
    parts.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{INK}"/>')


def addition(x, y):
    parts.append(
        f'<circle cx="{x}" cy="{y}" r="20" fill="white" '
        f'stroke="{INK}" stroke-width="2.6"/>'
    )
    text(x, y + 11, "+", 38)


def encoder(center, width, heads, hidden, attention_label, count):
    """Expand a Pre-LN block, with gates driven by normalized branch inputs."""
    left = center - 190
    rect(center - 250, 630, 500, 750, "panel", panel=True, radius=28)
    text(center - 225, 662, "Pre-LN block", 23, anchor="start")
    text(center + 215, 655, f"× {count}", 30, bold=True)
    box(left, 1280, 380, 55, ["LayerNorm"], "norm")
    box(left, 1130, 380, 105,
        [attention_label, f"{width}D · {heads} heads"], "attention", 27)
    box(center - 115, 1065, 230, 40, ["Gate ⊙"], "gate", 22)
    addition(center, 1020)
    box(left, 920, 380, 55, ["LayerNorm"], "norm")
    box(left, 800, 380, 90,
        ["Feed Forward · GELU", f"{width} → {hidden} → {width}"], "ffn", 26)
    box(center - 115, 730, 230, 40, ["Gate ⊙"], "gate", 22)
    addition(center, 675)
    box(left, 540, 380, 55, ["Final LayerNorm"], "norm")

    for start, end in ((1390, 1335), (1280, 1235), (1130, 1105),
                       (1065, 1040), (1000, 975), (920, 890),
                       (800, 770), (730, 695), (655, 595), (540, 495)):
        wire((center, start), (center, end))
    wire((center, 1355), (center - 220, 1355),
         (center - 220, 1020), (center - 20, 1020))
    wire((center, 990), (center + 230, 990),
         (center + 230, 675), (center + 20, 675))
    dot(center, 1355)
    dot(center, 990)
    # Dashed paths compute per-feature sigmoid gates from the same LN input.
    wire((center, 1255), (center + 205, 1255),
         (center + 205, 1085), (center + 115, 1085), dashed=True)
    wire((center, 905), (center + 205, 905),
         (center + 205, 750), (center + 115, 750), dashed=True)


text(1140, 57, "Structured Transformer Actor", 40, bold=True)
text(1140, 105, "单一运动控制策略 · 策略100 Hz / 物理200 Hz · 5帧含当前，跨度40 ms", 27)
text(350, 180, "Spatial Encoder", 31, bold=True)
text(350, 220, "逐帧共享 · 6个语义身体 token", 24, color=MUTED)
text(350, 260, "Command · Base · L/R legs · L/R wheels", 22, color=MUTED)
text(1000, 180, "Causal Temporal Encoder", 31, bold=True)
text(1000, 220, "下三角 mask · 仅访问当前与过去", 24, color=MUTED)
text(1740, 180, "Query Readout & Action Heads", 31, bold=True)
encoder(350, 64, 2, 128, "Body-graph MHA", 1)
encoder(1000, 96, 4, 192, "Causal MHA", 2)

box(105, 1630, 490, 100,
    ["History · [B, 5, 35]", "t−40, t−30, t−20, t−10, t ms"], "embedding", 26)
box(135, 1480, 430, 95,
    ["Semantic projections", "6 tokens × 64D / frame"], "linear", 27)
wire((350, 1630), (350, 1575))
wire((350, 1480), (350, 1430))
addition(350, 1410)
box(40, 1380, 165, 60, ["Body IDs"], "gate", 24)
wire((205, 1410), (330, 1410))
box(105, 405, 490, 90, ["Body tokens", "[B, 5, 6, 64]"], "linear", 27)
box(785, 1480, 430, 95,
    ["Concat 6×64 → Linear 96", "LayerNorm"], "linear", 26)
wire((595, 450), (660, 450), (660, 1527), (785, 1527))
dot(625, 450)
wire((1000, 1480), (1000, 1430))
addition(1000, 1410)
box(1150, 1380, 260, 60, ["Fixed sin/cos · −4…0"], "gate", 23)
wire((1150, 1410), (1020, 1410))
box(785, 405, 430, 90, ["Temporal memory", "[B, 5, 96]"], "linear", 27)

# Current-body queries are explicitly connected to the spatial output.
box(1390, 1630, 390, 100,
    ["Current body tokens", "select t: 5 × 64D"], "linear", 25)
wire((625, 450), (625, 1770), (1585, 1770), (1585, 1730))
text(1050, 1750, "Select current frame: legs ×2 · wheels ×2 · base ×1", 23)
box(1830, 1630, 390, 100,
    ["Current obs xₜ · 35D", "同一窗口的当前帧直连"], "embedding", 25)
box(1470, 1480, 540, 95,
    ["Query projection · (64 + 11) → 96",
     "5 queries: legs 2 · wheels 2 · state 1"], "linear", 25)
wire((1585, 1630), (1585, 1600), (1640, 1600), (1640, 1575))
wire((2025, 1630), (2025, 1600), (1870, 1600), (1870, 1575))
text(2045, 1585, "command 11D", 22)
wire((1740, 1480), (1740, 1430))
addition(1740, 1410)
box(1990, 1380, 230, 60, ["Query IDs ×5"], "gate", 23)
wire((1990, 1410), (1760, 1410))

rect(1430, 990, 620, 390, "panel", panel=True, radius=28)
text(1455, 1024, "Cross-attention readout", 23, anchor="start")
box(1550, 1220, 380, 100,
    ["Multi-Head Attention", "96D · 4 heads"], "attention", 27)
addition(1740, 1135)
box(1550, 1040, 380, 55, ["LayerNorm"], "norm")
wire((1740, 1390), (1740, 1320))
wire((1740, 1360), (1465, 1360), (1465, 1135), (1720, 1135))
dot(1740, 1360)
text(1770, 1358, "Q", 25)
wire((1215, 450), (1320, 450), (1320, 1270), (1550, 1270))
text(1438, 1248, "K, V", 25)
wire((1740, 1220), (1740, 1155))
wire((1740, 1115), (1740, 1095))
wire((1740, 1040), (1740, 985))
dot(1740, 985)

box(1830, 855, 390, 110,
    ["State head · 96 → 64 → 8", "velocity μ₃, log σ₃ · contact p₂",
     "clamp / sigmoid"], "gate", 22)
wire((1740, 985), (2025, 985), (2025, 965))
text(2025, 1016, "1 state query", 22)
wire((1740, 985), (1600, 985), (1600, 810))
text(1575, 899, "4 action queries", 23, anchor="end")
text(1575, 931, "96D each", 23, anchor="end", color=MUTED)
wire((2025, 855), (2025, 832), (1910, 832), (1910, 810))
text(2045, 824, "estimate 8D", 22)
box(1450, 680, 570, 130,
    ["Branch input concatenation", "Leg: Q96 + local6 + shared25 = 127",
     "Wheel: Q96 + local2 + shared25 = 123",
     "shared25 = base6 + command11 + estimate8"], "linear", 23)
wire((2220, 1680), (2250, 1680), (2250, 750), (2020, 750))
text(2233, 1180, "Current raw bypass", 23,
     rotate=-90, color=MUTED)

box(1430, 475, 310, 150,
    ["Shared leg head", "127 → 96 → 64 → 2", "×2 legs · GELU"], "ffn", 25)
box(1780, 475, 310, 150,
    ["Shared wheel head", "123 → 64 → 32 → 1", "×2 wheels · GELU"], "ffn", 25)
wire((1590, 680), (1590, 625))
wire((1910, 680), (1935, 680), (1935, 625))
wire((1585, 475), (1585, 450), (1740, 450), arrow=False)
wire((1935, 475), (1935, 450), (1740, 450), arrow=False)
dot(1740, 450)
box(1470, 330, 540, 85,
    ["Action mean μₜ · 6D", "4 leg channels + 2 wheel channels"], "gate", 26)
wire((1740, 450), (1740, 415))
box(1430, 220, 660, 75,
    ["Clip & physical mapping", "腿部同轴输出端位置 ×4  /  轮速目标 ×2"], "gate", 24)
wire((1740, 330), (1740, 295))

text(1140, 1820,
     "Gated residual:  y = x + sigmoid(g(LayerNorm(x))) ⊙ F(LayerNorm(x))", 26)
text(1140, 1860,
     "dropout = 0 · 无 RNN / 跨调用隐藏状态 / KV cache · 回合及控制权边界截断历史", 24)
rect(45, 1890, 2190, 110, "white", dashed=True)
text(70, 1924, "Training only", 25, anchor="start", bold=True)
text(70, 1967, "Independent critic: 81 → 256 → 128 → 64 → 1 · ELU",
     24, anchor="start")
text(1070, 1924, "a_raw ∼ N(μ, diag σ²) → clip → decode; logp uses a_raw",
     23, anchor="start")
text(1070, 1967, "Auxiliary: state query96 + issued action6 → 102 → 64 → 6",
     23, anchor="start")
text(1140, 2030,
     "部署 Actor 339,627 参数 · 训练 Actor + Critic 408,824 参数 · 特权真值不直接输入 Actor", 24)

svg = ROOT / "formal-policy.svg"
svg.write_text("\n".join([*parts, "</svg>"]) + "\n")
subprocess.run(
    ["rsvg-convert", "--width", "2800", "--output",
     str(ROOT / "formal-policy.png"), str(svg)],
    check=True,
)
