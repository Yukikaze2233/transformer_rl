"""Plot the V6 height tracking cost density, without running a simulator."""
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

for family in ['SimHei', 'Times New Roman', 'Times New Roman:style=Italic',
               'Times New Roman:style=Bold']:
    font_path = subprocess.check_output(
        ['fc-match', '-f', '%{file}', family], text=True
    ).strip()
    font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family': 'SimHei', 'axes.unicode_minus': False,
    'svg.fonttype': 'none', 'font.size': 12, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'normal', 'axes.labelcolor': 'black',
    'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
    'mathtext.fallback': 'stix'})

def costs(error_m):
    z = error_m / .040
    broad = -3.0 * z*z / (np.sqrt(1+z*z)+1)
    fine = 1.5 * (np.exp(-(error_m/.015)**2)-1)
    return broad, fine, broad+fine

fig, axes = plt.subplots(1, 2, figsize=(16, 9), facecolor='white')
for ax, span, title in zip(axes, [.12, .035], ['A  广域误差仍可区分', 'B  局部精度看15 mm尺度']):
    error = np.linspace(-span,span,801)
    for values, color, label in zip(costs(error), ['#356fcb','#229aab','#7952b4'],
            ['广域 pseudo-Huber', '局部精核', r'两项合计 $D_h(e_h)$']):
        ax.plot(error*1000, values, color=color, lw=2.8, label=label)
    ax.set_title(title, loc='left', pad=18, fontsize=20)
    ax.set_xlabel(r'支撑相对高度误差 $e_h$ / mm', fontsize=15)
    ax.set_ylabel('相对峰值的成本密度 / (reward/s)', fontsize=15)
    ax.grid(axis='both', color='#e5e5e5', lw=.7)
    ax.set_axisbelow(True)
    ax.axhline(0,color='black',lw=1)
    ax.set_xlim(-span*1000,span*1000)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_fontsize(14)
axes[1].axvline(15, ls='--', color='#229aab', lw=1)
axes[1].axvline(-15, ls='--', color='#229aab', lw=1)
axes[0].legend(loc='lower center',frameon=False,fontsize=11)
fig.suptitle('Reward设计', x=.055, ha='left', y=.97, fontsize=35,
             fontweight='bold', color='#2e648f')
fig.text(.06,.86,'高度通道：广域尺度40 mm、权重3.0；精核尺度15 mm、权重1.5',fontsize=16)
fig.text(.06,.13,r'$D_h(e_h)=-3\left[\sqrt{1+\left(\frac{e_h}{0.040}\right)^2}-1\right]'
         r'+1.5\left[\exp\!\left(-\left(\frac{e_h}{0.015}\right)^2\right)-1\right]$',fontsize=22)
fig.text(.06,.05,'公式切片，非训练曲线；零误差成本为0。mask与峰值项另计，50 Hz下密度乘0.02 s。',fontsize=13)
fig.subplots_adjust(left=.08,right=.97,top=.75,bottom=.27,wspace=.28)
out=Path(__file__).parent
fig.savefig(out/'diagram.png',dpi=180,facecolor=fig.get_facecolor())
fig.savefig(out/'diagram.svg',facecolor=fig.get_facecolor())
svg_path = out / 'diagram.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
assert all(abs(float(v))<1e-12 for v in costs(np.array(0.0)))
print('Saved analytic reward figure; D(0) = 0')
