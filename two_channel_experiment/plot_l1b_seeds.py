# -*- coding: utf-8 -*-
"""L1b 分层采样三 seed（BA）可视化：邻接矩阵 / 自回归曲线 / per-k 参数辨识。"""
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
COLORS = {7: '#1565c0', 21: '#c62828', 42: '#2e7d32'}
BETA_TRUE = 0.3


def load(seed):
    with open(os.path.join(RESULT_DIR, f'dgsl_l1b_BA_s{seed}.json'), encoding='utf-8') as f:
        return json.load(f)


def fig_adjacency(data):
    true_a = np.array(data[7]['matrices']['true_adj'])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    im0 = axes[0].imshow(true_a, cmap='Greys', vmin=0, vmax=1)
    axes[0].set_title('真实邻接矩阵')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    for ax, s in zip(axes[1:], SEEDS):
        hard = np.array(data[s]['matrices']['hard_adj'])
        h = data[s]['recon_hard_05']
        im = ax.imshow(hard, cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f'seed {s} 硬图\nACC={h["acc"]:.3f}, 边 {h["n_edges_pred"]}/{h["n_edges_true"]}',
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes:
        ax.set_xlabel('节点'); ax.set_ylabel('节点')
    fig.suptitle('BA 分层采样 L1b：三 seed 硬图重构 vs 真实', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_l1b_adjacency.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_autoreg(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SEEDS:
        ar = data[s]['autoreg_hard']
        t = np.arange(len(ar['true_curve']))
        if s == SEEDS[0]:
            ax.plot(t, ar['true_curve'], 'k-', linewidth=2.5, label='真实 SIR 模拟')
        ax.plot(t, ar['pred_curve'], '--', linewidth=1.8, color=COLORS[s],
                label=f'seed {s}（MAE={ar["mae_i"]:.4f}）')
    ax.set_xlabel('时间步 t'); ax.set_ylabel('感染者占比')
    ax.set_title('BA 分层采样 L1b：三 seed 自回归推演（多初始节点平均）')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_l1b_autoreg.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_perk(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = np.arange(1, 11)
    ax.plot(ks, [1 - (1 - BETA_TRUE) ** k for k in ks], 'k-', linewidth=2,
            label=f'真实 $1-(1-\\beta)^k$, β={BETA_TRUE}')
    for s in SEEDS:
        perk = data[s]['perk']
        kk = sorted(int(k) for k in perk['per_k'].keys())
        pp = [perk['per_k'][str(k)][1] for k in kk]
        ax.plot(kk, pp, 'o-', color=COLORS[s], markersize=5,
                label=f'seed {s}（$\\hat\\beta$={perk["beta_fit"]:.3f}）')
    ax.set_xlabel('感染者邻居数 k'); ax.set_ylabel('P(S→I)')
    ax.set_title('BA 分层采样 L1b：per-k 参数辨识（三 seed）')
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_l1b_perk.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    data = {s: load(s) for s in SEEDS}
    fig_adjacency(data)
    fig_autoreg(data)
    fig_perk(data)
    print('L1b figures written to', RESULT_DIR, flush=True)
