# -*- coding: utf-8 -*-
"""L1b 三 seed（BA）自回归推演——每 seed 一张独立 PNG（1×3 子图）。
每 seed 选取 G1 组的 1 / 3 / 5 点初始三个代表性测试，每个测试独立成子图：
真实 S/I/R 实线 vs 模型预测 S/I/R 虚线（n_mc=50 平均，不跨测试平均）。
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
SEEDS = [7, 21, 42]
N_INF_LIST = [1, 3, 5]
STATE_COLORS = {'S': 'b', 'I': 'g', 'R': 'r'}


def load(seed):
    with open(os.path.join(RESULT_DIR, f'dgsl_l1b_BA_s{seed}.json'), encoding='utf-8') as f:
        return json.load(f)


def get_test(data, g, n_inf):
    return data['autoreg_5x5']['tests'][g * 5 + n_inf - 1]


def main():
    data = {s: load(s) for s in SEEDS}
    for s in SEEDS:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        for c, n_inf in enumerate(N_INF_LIST):
            ax = axes[c]
            t = get_test(data[s], g=0, n_inf=n_inf)
            tf = np.array(t['true'])
            pf = np.array(t['pred'])
            ts = np.arange(len(tf))
            for st in 'SIR':
                ax.plot(ts, tf[:, 0 if st == 'S' else 1 if st == 'I' else 2],
                        STATE_COLORS[st] + '-', linewidth=1.6, alpha=0.9)
                ax.plot(ts, pf[:, 0 if st == 'S' else 1 if st == 'I' else 2],
                        STATE_COLORS[st] + '--', linewidth=1.5, alpha=0.85)
            ax.set_title(f'{n_inf} 点初始感染\nMAE(I)={t["mae"]["I"]:.4f}', fontsize=10)
            ax.set_xlabel('时间步 t')
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25)
            if c == 0:
                ax.set_ylabel('占比')
        # 图例（第一子图）
        handles = []
        for st in 'SIR':
            handles.append(plt.Line2D([0], [0], color=STATE_COLORS[st], linestyle='-', label=f'真实 {st}'))
            handles.append(plt.Line2D([0], [0], color=STATE_COLORS[st], linestyle='--', label=f'预测 {st}'))
        axes[0].legend(handles=handles, fontsize=8, loc='upper right', ncol=2)
        h = data[s]['recon_hard_05']
        fig.suptitle(f'BA 分层采样 L1b · seed {s}（ACC(hard)={h["acc"]:.3f}, '
                     f'边 {h["n_edges_pred"]}/{h["n_edges_true"]}）：自回归推演（G1 组，'
                     f'n_mc=50 平均；实线=真实，虚线=预测）', y=1.0, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(RESULT_DIR, f'fig_l1b_autoreg_s{s}.png'), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print(f'fig_l1b_autoreg_s{s}.png written', flush=True)


if __name__ == '__main__':
    main()
