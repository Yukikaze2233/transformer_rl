"""Draw the implemented actor with paper-style attention and residual blocks.

Only the standard library and rsvg-convert are used. Dimensions are transcribed
from formal/policy.py; the model and simulator are never imported.
"""
from html import escape
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT.parents[1] / 'configs/formal_training.json').read_text())['policy']
if CONFIG['history_length'] != 5:
    raise ValueError('this figure layout requires the five-frame observation contract')
BODY, TIME = CONFIG['body_width'], CONFIG['temporal_width']
LEG = (TIME + 31, *CONFIG['leg_hidden_dims'], 2)
WHEEL = (TIME + 27, *CONFIG['wheel_hidden_dims'], 1)
STATE = (TIME, CONFIG['state_hidden_dim'], 8)
PREDICTION = (TIME + 6, CONFIG['prediction_hidden_dim'], 6)
CRITIC = (CONFIG['critic_input_dim'], *CONFIG['critic_hidden_dims'], 1)
WIDTH, HEIGHT = 2280, 2160
INK, MUTED, BORDER = '#243746', '#667788', '#526779'
COLORS = {
    'attention': '#fae6bf', 'ffn': '#d9eeed', 'norm': '#f5f1d9',
    'embedding': '#f5e5e9', 'linear': '#e2eaf6', 'gate': '#deeee3',
    'panel': '#f5f7fa',
}


def linear_parameters(inputs, outputs):
    return (inputs + 1) * outputs


def mlp_parameters(widths):
    return sum(linear_parameters(a, b) for a, b in zip(widths, widths[1:]))


def block_parameters(width, hidden):
    return (linear_parameters(width, 3 * width) + 3 * linear_parameters(width, width)
            + mlp_parameters((width, hidden, width)) + 4 * width)


# Static layer arithmetic only; rendering never imports the model or torch.
INPUT_PARAMETERS = (sum(linear_parameters(n, BODY) for n in (11, 6, 6, 2))
                    + 6 * BODY + 2 * BODY + linear_parameters(6 * BODY, TIME) + 2 * TIME)
READOUT_PARAMETERS = (linear_parameters(BODY + 11, TIME) + 5 * TIME
                      + 3 * linear_parameters(TIME, TIME) + 2 * TIME)
DEPLOYED_PARAMETERS = (INPUT_PARAMETERS + block_parameters(BODY, CONFIG['body_ffn'])
                       + CONFIG['temporal_layers'] * block_parameters(TIME, CONFIG['temporal_ffn'])
                       + 2 * TIME + READOUT_PARAMETERS + mlp_parameters(STATE)
                       + mlp_parameters(LEG) + mlp_parameters(WHEEL))
ACTOR_PARAMETERS = DEPLOYED_PARAMETERS + mlp_parameters(PREDICTION) + 6
CRITIC_PARAMETERS = mlp_parameters(CRITIC)


def dimensions(widths):
    return ' → '.join(str(width) for width in widths)


parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
    f'viewBox="0 0 {WIDTH} {HEIGHT}">',
    f'<title>Structured Transformer: spatial {BODY}D, temporal {TIME}D, critic {CRITIC[0]}D</title>',
    "<desc>Five raw frames enter a shared spatial encoder and two causal temporal "
    "blocks. Four actuator queries and one state query read temporal memory. "
    "Current raw features and estimated state feed shared leg and wheel heads. "
    "Only training uses the independent critic and auxiliary prediction head.</desc>",
    '<rect width="100%" height="100%" fill="#ffffff"/>',
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
        'font-family="Inter,Arial,Noto Sans CJK SC,sans-serif">'
        f'{escape(value)}</text>'
    )


def rect(x, y, width, height, color="white", *, radius=10, panel=False,
         dashed=False):
    dash = ' stroke-dasharray="9 7"' if dashed else ""
    stroke = "none" if panel else BORDER
    parts.append(
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" fill="{COLORS.get(color, color)}" '
        f'stroke="{stroke}" stroke-width="1.9"{dash}/>'
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
        f'stroke-width="2.2" stroke-linejoin="round"{marker}{dash}/>'
    )


def dot(x, y):
    parts.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{INK}"/>')


def addition(x, y):
    parts.append(
        f'<circle cx="{x}" cy="{y}" r="20" fill="white" '
        f'stroke="{INK}" stroke-width="2.2"/>'
    )
    text(x, y + 11, "+", 38)


def mask_preview(x, y, allowed, label):
    cell = 17
    for row, columns in enumerate(allowed):
        for column, visible in enumerate(columns):
            parts.append(
                f'<rect x="{x + cell * column}" y="{y + cell * row}" '
                f'width="{cell - 2}" height="{cell - 2}" rx="2" '
                f'fill="{"#83b9b0" if visible else "#ffffff"}" stroke="#d0dce2" stroke-width="0.8"/>'
            )
    text(x + len(allowed) * cell + 20, y + 34, label, 22, anchor='start')
    text(x + len(allowed) * cell + 20, y + 67, '色块可见 · 空白屏蔽', 19, anchor='start', color=MUTED)


def encoder(center, width, heads, hidden, attention_label, count):
    """Expand a Pre-LN block, with gates driven by normalized branch inputs."""
    left = center - 190
    rect(center - 250, 630, 500, 750, "panel", panel=True, radius=18)
    text(center - 225, 662, "Pre-LN block", 23, anchor="start")
    text(center + 215, 655, f"× {count}", 30, bold=True)
    box(left, 1280, 380, 55, ["LayerNorm"], "norm")
    box(left, 1130, 380, 105,
        [attention_label, f"{width}D · {heads} heads"], "attention", 27)
    box(center - 115, 1065, 230, 40, ["Sigmoid gate ⊙"], "gate", 22)
    addition(center, 1020)
    box(left, 920, 380, 55, ["LayerNorm"], "norm")
    box(left, 800, 380, 90,
        ["Feed Forward · GELU", f"{width} → {hidden} → {width}"], "ffn", 26)
    box(center - 115, 730, 230, 40, ["Sigmoid gate ⊙"], "gate", 22)
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


text(1140, 57, "Structured Transformer · Single Control Policy", 39, bold=True)
text(1140, 105, "单一运动控制策略 · 策略100 Hz / 物理200 Hz · 5帧含当前，跨度40 ms", 27)
text(350, 180, "01  Spatial Encoder", 31, bold=True)
text(350, 220, "逐帧共享 · 6个语义身体 token", 24, color=MUTED)
text(350, 260, "Command · Base · L/R legs · L/R wheels", 22, color=MUTED)
text(1000, 180, "02  Temporal Encoder", 31, bold=True)
text(1000, 220, "下三角 mask · 仅访问当前与过去", 24, color=MUTED)
text(1740, 180, "03  Query Readout & Action", 31, bold=True)
body_edges = {(2, 4), (3, 5), (2, 3), (4, 5)}
body_mask = [[i == j or i < 2 or j < 2 or (min(i, j), max(i, j)) in body_edges
              for j in range(6)] for i in range(6)]
mask_preview(190, 290, body_mask, '6×6 semantic mask')
mask_preview(835, 290, [[j <= i for j in range(5)] for i in range(5)], '5×5 causal mask')
encoder(350, BODY, CONFIG['body_heads'], CONFIG['body_ffn'], "Semantic-mask MHA", 1)
encoder(1000, TIME, CONFIG['temporal_heads'], CONFIG['temporal_ffn'], "Causal MHA", CONFIG['temporal_layers'])

box(105, 1630, 490, 100,
    ["History · [B, 5, 35]", "t−40, t−30, t−20, t−10, t ms"], "embedding", 26)
box(135, 1480, 430, 95,
    ["Semantic projections", f"6 tokens × {BODY}D / frame"], "linear", 27)
wire((350, 1630), (350, 1575))
wire((350, 1480), (350, 1430))
addition(350, 1410)
box(40, 1380, 165, 60, ["Body IDs"], "gate", 24)
wire((205, 1410), (330, 1410))
box(105, 405, 490, 90, ["Body tokens", f"[B, 5, 6, {BODY}]"], "linear", 27)
box(785, 1480, 430, 95,
    [f"Concat 6×{BODY} → Linear {TIME}", "LayerNorm"], "linear", 26)
wire((595, 450), (660, 450), (660, 1527), (785, 1527))
dot(625, 450)
wire((1000, 1480), (1000, 1430))
addition(1000, 1410)
box(1150, 1380, 260, 60, ["Fixed sin/cos · −4…0"], "gate", 23)
wire((1150, 1410), (1020, 1410))
box(785, 405, 430, 90, ["Temporal memory", f"[B, 5, {TIME}]"], "linear", 27)

# Current-body queries are explicitly connected to the spatial output.
box(1390, 1630, 390, 100,
    ["Current body tokens", f"select t: 5 × {BODY}D"], "linear", 25)
wire((625, 450), (625, 1770), (1585, 1770), (1585, 1730))
text(1050, 1750, "Select current frame: legs ×2 · wheels ×2 · base ×1", 23)
box(1830, 1630, 390, 100,
    ["Current obs xₜ · 35D", "同一窗口的当前帧直连"], "embedding", 25)
box(1470, 1480, 540, 95,
    [f"Query projection · ({BODY} + 11) → {TIME}",
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
    ["Multi-Head Attention", f"{TIME}D · {CONFIG['temporal_heads']} heads"], "attention", 27)
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
    [f"State head · {dimensions(STATE)}", "velocity μ₃, log σ₃ · contact p₂",
     "clamp / sigmoid"], "gate", 22)
wire((1740, 985), (2025, 985), (2025, 965))
text(2025, 1016, "1 state query", 22)
wire((1740, 985), (1600, 985), (1600, 810))
text(1575, 899, "4 action queries", 23, anchor="end")
text(1575, 931, f"{TIME}D each", 23, anchor="end", color=MUTED)
wire((2025, 855), (2025, 832), (1910, 832), (1910, 810))
text(2045, 824, "estimate 8D", 22)
box(1450, 680, 570, 130,
    ["Branch input concatenation", f"Leg: Q{TIME} + local6 + shared25 = {LEG[0]}",
     f"Wheel: Q{TIME} + local2 + shared25 = {WHEEL[0]}",
     "shared25 = base6 + command11 + estimate8"], "linear", 23)
wire((2220, 1680), (2250, 1680), (2250, 750), (2020, 750))
text(2233, 1180, "Current raw bypass", 23,
     rotate=-90, color=MUTED)

box(1430, 475, 310, 150,
    ["Shared leg head", dimensions(LEG), "×2 legs · GELU"], "ffn", 25)
box(1780, 475, 310, 150,
    ["Shared wheel head", dimensions(WHEEL), "×2 wheels · GELU"], "ffn", 25)
wire((1590, 680), (1590, 625))
wire((1910, 680), (1935, 680), (1935, 625))
wire((1585, 475), (1585, 450), (1740, 450), arrow=False)
wire((1935, 475), (1935, 450), (1740, 450), arrow=False)
dot(1740, 450)
box(1470, 330, 540, 85,
    ["Action mean μₜ · 6D", "4 leg channels + 2 wheel channels"], "gate", 26)
wire((1740, 450), (1740, 415))
box(1430, 220, 660, 75,
    ["Clip & target mapping", "腿部同轴输出端位置 ×4  /  轮速目标 ×2"], "gate", 24)
wire((1740, 330), (1740, 295))

text(1140, 1820,
     "Gated residual:  y = x + sigmoid(g(LayerNorm(x))) ⊙ F(LayerNorm(x))", 26)
text(1140, 1860,
     "dropout = 0 · 无 RNN / 跨调用隐藏状态 / KV cache · 回合及控制权边界截断历史", 24)
rect(45, 1905, 2190, 180, "panel", dashed=True, radius=16)
text(75, 1942, "TRAINING ONLY · INDEPENDENT CRITIC", 24, anchor="start", bold=True)
text(75, 1984, f"Environment {CRITIC[0] - 3}D + estimated velocity 3D (detach)",
     25, anchor="start")
text(75, 2034, f"{dimensions(CRITIC)} · ELU", 28, anchor="start", bold=True)
text(1170, 1942, "TRAINING ONLY · AUXILIARY & EXPLORATION", 24, anchor="start", bold=True)
text(1170, 1984, f"Prediction: state query {TIME} + issued action 6", 24, anchor="start")
text(1170, 2023, f"MLP {dimensions(PREDICTION)} · GELU", 25, anchor="start")
text(1170, 2060, "a_raw ∼ N(μ, diag σ²); logp uses a_raw before clipping", 22, anchor="start")
text(1140, 2134,
     f"部署 Actor {DEPLOYED_PARAMETERS:,} 参数 · 训练 Actor {ACTOR_PARAMETERS:,} + Critic {CRITIC_PARAMETERS:,} = {ACTOR_PARAMETERS + CRITIC_PARAMETERS:,}",
     25)

svg = ROOT / "formal-policy.svg"
svg.write_text("\n".join([*parts, "</svg>"]) + "\n")
subprocess.run(
    ["rsvg-convert", "--width", "2800", "--output",
     str(ROOT / "formal-policy.png"), str(svg)],
    check=True,
)
