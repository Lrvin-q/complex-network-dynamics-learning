# -*- coding: utf-8 -*-
"""plot_gamma_probe.py —— 从 gamma_probe_{NET}.json 绘图：逐度均值 vs 分桶均值 vs Pearson r"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase3b')
NETS = ['ER', 'BA']

for net in NETS:
    r = json.load(open(os.path.join(BASE, f'gamma_probe_{net}.json'), encoding='utf-8'))
    degs = sorted(int(d) for d in r['per_degree'].keys())
    means = [r['per_degree'][str(d)]['mean'] for d in degs]
    counts = [r['per_degree'][str(d)]['count'] for d in degs]

    fig, ax = plt.subplots(figsize=(9, 5))
    # 逐度均值（细粒度）
    ax.plot(degs, means, 'o-', color='#c62828', markersize=5, linewidth=1.2,
            label='Per-degree mean P(I→R)')
    # 总体均值线
    ax.axhline(r['gamma_mean'], color='k', linestyle='--', linewidth=1.2,
               label=f'Overall mean = {r["gamma_mean"]:.4f}')
    # 桶均值（粗粒度）
    for b in r['bucket_stats']:
        lo, hi = b['bucket'].split('-')
        ax.axvspan(int(lo), int(hi) + 1, color='gray', alpha=0.12)
    bxs = []
    for b in r['bucket_stats']:
        lo, hi = b['bucket'].split('-')
        bx = (int(lo) + int(hi)) / 2
        bxs.append((bx, b['mean'], b['bucket']))
    if bxs:
        ax.plot([x[0] for x in bxs], [x[1] for x in bxs], 's--', color='#1565c0',
                markersize=7, linewidth=1.5, label='Bucket mean')
        for bx, by, bl in bxs:
            ax.text(bx, by + (r['gamma_std'] * 0.6), f'{bl}', ha='center', fontsize=8, color='#1565c0')
    # y 轴范围压缩以显示微变化
    lo_y = min(means + [r['gamma_mean']]) - 3 * r['gamma_std']
    hi_y = max(means + [r['gamma_mean']]) + 3 * r['gamma_std']
    ax.set_ylim(lo_y, hi_y)
    ax.set_xlabel('Node degree')
    ax.set_ylabel('P(I→R) (model output)')
    ax.set_title(f'{net}: γ probe — std={r["gamma_std"]:.5f}, Pearson r={r["pearson_degree"]:.3f}, '
                 f'slope={r["slope_all_per_deg"]:.2e}/deg')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(BASE, f'gamma_probe_{net}.png'), dpi=150)
    plt.close(fig)
    print(f'saved gamma_probe_{net}.png')

print('done')
