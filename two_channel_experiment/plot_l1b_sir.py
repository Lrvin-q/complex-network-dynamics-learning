# -*- coding: utf-8 -*-
"""L1b 三 seed（BA）S/I/R 完整演化图：每 seed 一张子图（1×3 并排）。
25 个 5×5 测试的 true/pred 曲线按 S/I/R 分别平均；真实曲线三 seed 相同（唯一基准），
预测曲线随 seed 不同。实线=真实，虚线=预测；蓝/绿/红 = S/I/R。
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
COLORS = {7: '#1565c0', 21: '#c62828', 42: '#2e7d32'}
STATE_COLORS = {'S': 'b', 'I': 'g', 'R': 'r'}


def load(seed):
    with open(os.path.join(RESULT_DIR, f'dgsl_l1b_BA_s{seed}.json'), encoding='utf-8') as f:
        return json.load(f)


def avg_sir(data):
    """25 测试 true/pred 曲线按状态平均 → (T+1, 3)。"""
    tests = data['autoreg_5x5']['tests']
    t = np.array(tests[0]['true'])
    T = len(t)
    true = np.zeros((T, 3))
    pred = np.zeros((T, 3))
    for te in tests:
        true += np.array(te['true']) / len(tests)
        pred += np.array(te['pred']) / len(tests)
    return true, pred


def main():
    data = {s: load(s) for s in SEEDS}
    true_ref = None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharex=True, sharey=True)
    for ax, s in zip(axes, SEEDS):
        true, pred = avg_sir(data[s])
        if true_ref is None:
            true_ref = true
        t = np.arange(len(true))
        for i, c in enumerate('SIR'):
            ax.plot(t, true[:, i], STATE_COLORS[c] + '-', linewidth=1.8, alpha=0.9,
                    label=f'真实 {c}')
            ax.plot(t, pred[:, i], STATE_COLORS[c] + '--', linewidth=1.6, alpha=0.85,
                    label=f'预测 {c}')
        h = data[s]['recon_hard_05']
        mae = data[s]['autoreg_5x5']['mae_i_mean']
        ax.set_title(f'seed {s}\nACC(hard)={h["acc"]:.3f}, 边 {h["n_edges_pred"]}/{h["n_edges_true"]}, '
                     f'MAE(I)={mae:.4f}', fontsize=9)
        ax.set_xlabel('时间步 t')
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.03, 1.03)
        if s == SEEDS[0]:
            ax.set_ylabel('占比')
            ax.legend(fontsize=8, loc='upper right')
    # 验证三 seed 真实曲线一致性
    for s in SEEDS[1:]:
        true_s, _ = avg_sir(data[s])
        print(f'真实曲线差异(s7 vs s{s}) 最大绝对差: {np.abs(true_ref - true_s).max():.6f}', flush=True)
    fig.suptitle('BA 分层采样 L1b：S/I/R 演化（25 个 5×5 测试平均；实线=真实，虚线=预测）',
                 y=0.98, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(RESULT_DIR, 'fig_l1b_sir.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('fig_l1b_sir.png written', flush=True)


if __name__ == '__main__':
    main()
