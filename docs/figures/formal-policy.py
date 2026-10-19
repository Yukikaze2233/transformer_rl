"""Render the implemented policy using a classic Transformer-paper layout."""
from html import escape
from pathlib import Path
import subprocess


ROOT = Path(__file__).parent
WIDTH, HEIGHT = 1880, 1770
INK = '#202020'
COLORS = {'attention': '#f8cfb6', 'ffn': '#bad8e8', 'norm': '#f3edb9',
          'embedding': '#eac7d3', 'linear': '#cce3cb', 'panel': '#f3f3f3'}
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
    f'viewBox="0 0 {WIDTH} {HEIGHT}">',
    '<title>Single control policy: spatial and causal temporal Transformer</title>',
    '<desc>Bottom-to-top inference, with Pre-LN gated residual encoder blocks, '
    'current-state queries, shared actuator decoders, and an independent training critic.</desc>',
    '<rect width="100%" height="100%" fill="white"/>',
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    '<path d="M0 0L10 5L0 10Z" fill="#202020"/></marker></defs>',
]


def text(x, y, value, size=22, *, anchor='middle', bold=False, color=INK):
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
                 f'font-weight="{700 if bold else 400}" fill="{color}" '
                 'font-family="Times New Roman,Noto Serif CJK SC,serif">'
                 f'{escape(value)}</text>')


def rect(x, y, width, height, fill='white', *, dashed=False, radius=12):
    dash = ' stroke-dasharray="8 6"' if dashed else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
                 f'rx="{radius}" fill="{fill}" stroke="{INK}" stroke-width="1.6"{dash}/>')


def box(x, y, width, height, lines, color='white', size=22):
    rect(x, y, width, height, COLORS.get(color, color))
    step = size * 1.35
    start = y + height / 2 - step * (len(lines) - 1) / 2 + size * 0.34
    for index, line in enumerate(lines):
        text(x + width / 2, start + index * step, line, size)


def arrow(points, *, dashed=False):
    dash = ' stroke-dasharray="7 5"' if dashed else ''
    parts.append(f'<polyline points="{points}" fill="none" stroke="{INK}" '
                 f'stroke-width="1.8" marker-end="url(#arrow)"{dash}/>')


def addition(x, y):
    parts.append(f'<circle cx="{x}" cy="{y}" r="19" fill="#eeeeee" '
                 f'stroke="{INK}" stroke-width="1.6"/>')
    text(x, y + 9, '+', 31)


def encoder(center, *, width, heads, hidden, mask, count):
    left = center - 170
    rect(center - 215, 515, 430, 655, COLORS['panel'], radius=18)
    box(left, 1070, 340, 52, ['Layer Norm'], 'norm')
    box(left, 940, 340, 90, [mask, 'Multi-Head Attention', f'{width}D · {heads} heads'],
        'attention', 21)
    box(left, 850, 340, 60, ['Gated Residual Add'], 'norm')
    box(left, 760, 340, 52, ['Layer Norm'], 'norm')
    box(left, 640, 340, 85, ['Feed Forward · GELU', f'{width} → {hidden} → {width}'], 'ffn')
    box(left, 550, 340, 60, ['Gated Residual Add'], 'norm')
    box(left, 430, 340, 52, ['Layer Norm'], 'norm')
    arrow(f'{center},1211 {center},1122')
    arrow(f'{center},1070 {center},1050 {center-85},1050 {center-85},1030')
    arrow(f'{center},1050 {center},1030')
    arrow(f'{center},1050 {center+85},1050 {center+85},1030')
    arrow(f'{center},940 {center},910')
    arrow(f'{center},850 {center},812')
    arrow(f'{center},760 {center},725')
    arrow(f'{center},640 {center},610')
    arrow(f'{center},550 {center},482')
    arrow(f'{center},1140 {center-200},1140 {center-200},880 {left},880')
    arrow(f'{center},830 {center-190},830 {center-190},580 {left},580')
    text(center + 242, 850, f'× {count}', 30)


text(940, 57, 'Single Control Policy · Spatial–Temporal Transformer', 35, bold=True)
text(940, 101, '100 Hz · H = 5（含当前）· 40 ms · 单一 Actor / 6D 动作', 23)
text(300, 171, 'Spatial Encoder', 29, bold=True)
text(850, 171, 'Temporal Encoder', 29, bold=True)
text(1480, 171, 'Current-State Query Readout', 27, bold=True)

encoder(300, width=64, heads=2, hidden=128, mask='Body-Graph Masked', count=1)
encoder(850, width=96, heads=4, hidden=192, mask='Causal Masked', count=2)

box(100, 1425, 400, 80, ['Observation Window [B, 5, 35]', 't−40, t−30, t−20, t−10, t ms'], size=21)
box(130, 1290, 340, 90, ['Semantic Embedding', '6 body tokens × 64D / frame'], 'embedding', 21)
arrow('300,1425 300,1380')
arrow('300,1290 300,1249')
addition(300, 1230)
text(105, 1238, 'Body ID', 21)
arrow('165,1230 281,1230')
text(300, 394, '6 × 64D / frame', 22)
box(650, 1290, 400, 90, ['Concat 6 × 64 → Linear 96', '+ Layer Norm'], 'embedding', 22)
arrow('300,430 300,350 575,350 575,1335 650,1335')
arrow('850,1290 850,1249')
addition(850, 1230)
text(1110, 1207, 'Temporal Encoding', 20)
text(1110, 1239, 'sin/cos (−4 … 0)', 20)
arrow('1015,1230 869,1230')
box(650, 290, 400, 78, ['Temporal Memory', '[B, 5, 96]'], size=23)
arrow('850,430 850,368')

rect(1190, 790, 580, 410, COLORS['panel'], radius=18)
box(1320, 1040, 320, 90, ['Multi-Head Attention', '96D · 4 heads'], 'attention', 22)
box(1320, 950, 320, 60, ['Add'], 'norm')
box(1320, 850, 320, 52, ['Layer Norm'], 'norm')
box(1280, 1290, 400, 90, ['Query Projection', '5 × 96D'], 'embedding')
box(1240, 1425, 480, 80, ['Current Body Tokens + Command', '当前身体编码（左侧输出）+ 11D 命令'], size=20)
arrow('1480,1425 1480,1380')
arrow('1480,1290 1480,1249')
addition(1480, 1230)
text(1695, 1239, 'Query ID', 21)
arrow('1630,1230 1499,1230')
arrow('1480,1211 1480,1130')
text(1500, 1167, 'Q', 23, anchor='start')
arrow('1050,329 1120,329 1120,1085 1320,1085')
text(1230, 1070, 'K, V', 23)
arrow('1480,1040 1480,1010')
arrow('1480,950 1480,902')
arrow('1480,1165 1240,1165 1240,980 1320,980')

box(1210, 640, 260, 115, ['State Head', '96 → 64 → 8', 'μᵥ (3), log σᵥ (3), pcontact (2)'], 'linear', 19)
box(1540, 640, 260, 115, ['Current Raw Features', 'IMU + Command', 'Local Leg / Wheel State'], 'embedding', 19)
box(1210, 330, 590, 155, ['Shared Actuator Decoders · GELU', 'Leg: 127 → 96 → 64 → 2  (per leg)',
                           'Wheel: 123 → 64 → 32 → 1  (per wheel)',
                           'Query + Local State + IMU + Command + Estimate'], 'ffn', 20)
box(1280, 220, 440, 64, ['Action Means μₜ ∈ ℝ⁶'], 'linear', 25)
arrow('1480,850 1480,805 1340,805 1340,755')
arrow('1480,805 1505,805 1505,485')
text(1530, 576, '4 action queries', 19, anchor='start')
arrow('1340,640 1340,485')
text(1322, 576, '8D', 21, anchor='end')
arrow('1670,640 1670,610 1775,610 1775,485')
text(1765, 520, '直连', 18, anchor='end')
arrow('1505,330 1505,284')
text(1500, 309, '4 leg-position channels + 2 wheel-velocity channels', 17)
text(1510, 788, '1 state query', 18, anchor='start')

text(940, 1551, 'Pre-LN gated residual:  y = x + sigmoid(g(LN(x))) ⊙ F(LN(x))', 23)
text(940, 1588, '无 RNN / 跨调用隐藏状态 / KV cache · dropout = 0 · reset / 控制权边界截断历史', 21)
rect(40, 1620, 1800, 105, '#fafafa', dashed=True)
text(60, 1649, 'Training only', 21, anchor='start', bold=True)
text(60, 1685, 'Critic: 81 → 256 → 128 → 64 → 1  (ELU, 独立参数)', 21, anchor='start')
text(970, 1649, 'Auxiliary prediction: State Query 96 + Issued Action 6 → 102 → 64 → 6', 20, anchor='start')
text(970, 1685, 'Diagonal Gaussian sampling · PPO + auxiliary losses · unified KL protection', 20, anchor='start')
text(940, 1753, '部署 Actor: 339,627 参数（保留 8D 估计头） · 训练 Actor + Critic: 408,824 参数', 21)

svg = ROOT / 'formal-policy.svg'
svg.write_text('\n'.join([*parts, '</svg>']) + '\n')
subprocess.run(['rsvg-convert', '--width', '2200', '--output', str(ROOT / 'formal-policy.png'),
                str(svg)], check=True)
