"""Render general policy-network diagrams with explicit CJK and math fonts."""

from html import escape
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parent
BLUE = '#2E648F'


class Figure:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" '
            'refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="black"/></marker></defs>',
        ]

    def text(self, x, y, value, size=26, anchor='start', color='black', bold=False):
        spans = []
        for run in re.findall(r'[\x20-\x7e]+|[^\x20-\x7e]+', value):
            family = 'Times New Roman' if run.isascii() else 'Noto Sans CJK SC'
            spans.append(f'<tspan font-family="{family}">{escape(run)}</tspan>')
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{700 if bold else 400}">'
            + ''.join(spans) + '</text>'
        )

    def math(self, x, y, expression, size=31, anchor='middle'):
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="Times New Roman" '
            f'font-size="{size}" text-anchor="{anchor}">{expression}</text>'
        )

    def box(self, x, y, w, h, fill='white', stroke='black', dashed=False):
        dash = ' stroke-dasharray="12 8" rx="22"' if dashed else ''
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"{dash}/>'
        )

    def arrow(self, points, dashed=False):
        dash = ' stroke-dasharray="7 5"' if dashed else ''
        self.parts.append(
            f'<polyline points="{points}" fill="none" stroke="black" '
            f'stroke-width="2" marker-end="url(#arrow)"{dash}/>'
        )

    def save(self, name):
        svg = ROOT / f'{name}.svg'
        svg.write_text('\n'.join([*self.parts, '</svg>']) + '\n')
        subprocess.run(
            ['rsvg-convert', '--zoom', '2', '--output', str(ROOT / f'{name}.png'), str(svg)],
            check=True,
        )


def variable(base, sub=None):
    value = f'<tspan font-style="italic">{base}</tspan>'
    if sub:
        parts = ''.join(
            f'<tspan font-style="{"italic" if char.isalpha() else "normal"}">{char}</tspan>'
            for char in sub
        )
        value += f'<tspan baseline-shift="sub" font-size="70%">{parts}</tspan>'
    return value


def families():
    fig = Figure(1800, 1190)
    fig.text(55, 85, '策略网络结构对照', 54, color=BLUE, bold=True)
    fig.text(60, 138, '从当前观测到历史表示：五种时间信息处理方式', 28)
    rows = [
        ('(a) 单帧 MLP', '当前状态映射', '当前观测', variable('o', 't'),
         '全连接网络', '非线性映射 → 动作头', '计算路径简单', '信息来自当前输入', '#FCFAED', '#CABC83'),
        ('(b) 堆帧 MLP', '固定窗口展开', '观测历史', variable('H', 't'),
         '展平 → 全连接网络', '固定时间槽 → 动作头', '短时动态基线', '首层规模随窗口增长', '#FCF0F3', '#D0ACBA'),
        ('(c) GRU / LSTM', '递归状态压缩', '新观测＋旧记忆', variable('o', 't') + ', ' + variable('h', 't−1'),
         '门控递归更新', '新记忆 → 动作头', '可逐步增量更新', '需维护内部状态', '#F5F0FA', '#B49ACA'),
        ('(d) TCN', '时间局部性', '观测历史', variable('H', 't'),
         '多层因果膨胀卷积', '末时刻特征 → 动作头', '共享时间权重', '感受野由卷积决定', '#F0F7F3', '#A3C3AF'),
        ('(e) Transformer', '内容关联', '历史＋时间编码', variable('H', 't'),
         '因果注意力＋前馈网络', '历史表示 → 动作头', '选择相关历史', '成本随窗口增长', '#EEF4FC', '#9DB8D8'),
    ]
    for i, row in enumerate(rows):
        title, idea, inp, symbol, net, sub, note1, note2, fill, stroke = row
        y = 200 + i * 177
        fig.text(60, y + 40, title, 30, bold=True)
        fig.text(65, y + 83, idea, 24)
        fig.box(350, y, 1080, 135, fill, stroke, True)
        fig.box(375, y + 25, 225, 85)
        fig.text(487, y + 56, inp, 23, anchor='middle')
        fig.math(487, y + 92, symbol)
        fig.arrow(f'600,{y+68} 665,{y+68}')
        fig.box(670, y + 25, 470, 85)
        fig.text(905, y + 59, net, 27, anchor='middle')
        fig.text(905, y + 94, sub, 23, anchor='middle')
        fig.arrow(f'1140,{y+68} 1205,{y+68}')
        fig.box(1210, y + 25, 190, 85)
        fig.math(1305, y + 68, variable('μ', 't'), 34)
        fig.text(1305, y + 99, '动作均值', 22, anchor='middle')
        fig.text(1470, y + 56, note1, 25)
        fig.text(1470, y + 99, note2, 24)
    fig.text(65, 1138, '图示以连续控制的动作均值为例；各历史编码器均可与当前观测直连、独立 Critic 或辅助估计组合。', 23)
    fig.save('families')


def timing():
    fig = Figure(1800, 1130)
    fig.text(55, 85, '历史跨度与实时控制成本', 54, color=BLUE, bold=True)
    fig.text(60, 140, '参数容量、每次计算和控制链路时延，需要分别衡量', 28)
    fig.box(55, 195, 820, 440, '#FCFAED', '#CABC83', True)
    fig.text(85, 248, '历史窗口按物理时间比较', 31, bold=True)
    fig.math(465, 313, variable('T', 'span') + ' = (' + variable('L') + ' − 1) / ' + variable('f'), 39)
    for y, left, mid, right in [
        (390, '采样频率', '窗口帧数', '首尾跨度'),
        (453, '50 Hz', '16', '0.30 s'),
        (512, '100 Hz', '16', '0.15 s'),
        (571, '100 Hz', '31', '0.30 s'),
    ]:
        for x, value in [(200, left), (475, mid), (725, right)]:
            fig.text(x, y, value, 29, anchor='middle', bold=y == 390)
    fig.box(925, 195, 820, 440, '#F5F0FA', '#B49ACA', True)
    fig.text(955, 248, '窗口增长如何影响网络', 31, bold=True)
    for y, title, detail in [
        (320, '堆帧 MLP', '输入层参数随 LF 增长'),
        (405, '流式 GRU / LSTM', '递归状态大小固定，逐步更新'),
        (490, '整窗 Transformer', '参数共享；密集注意力含 L² 项'),
        (575, '控制频率提高', '每秒计算量＝每次计算量 × 频率'),
    ]:
        fig.text(960, y, title, 27, bold=True)
        fig.text(960, y + 34, detail, 24)
    fig.box(55, 685, 1690, 310, '#EEF4FC', '#9DB8D8', True)
    fig.text(85, 742, '完整控制链路', 31, bold=True)
    labels = ['采样与时间戳', '预处理与历史', '网络推理', '通信与调度', '执行器生效']
    for i, label in enumerate(labels):
        x = 90 + 332 * i
        fig.box(x, 795, 275, 80)
        fig.text(x + 137, 846, label, 27, anchor='middle')
        if i < 4:
            fig.arrow(f'{x+275},835 {x+326},835')
    fig.text(90, 934, '分别记录：推理耗时、周期释放到动作生效的时延、执行时的数据年龄、全部周期中的超时率。', 25)
    fig.text(65, 1046, '表中数值是窗口换算示例。更高频率的控制收益取决于对象动态、传感器更新和底层控制器。', 24)
    fig.text(65, 1089, '过去帧无需等待未来；缓存可减少重算，但必须与位置编码、窗口截断和状态重置语义一致。', 24)
    fig.save('timing')


if __name__ == '__main__':
    families()
    timing()
