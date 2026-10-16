# -*- coding: utf-8 -*-
"""
plot_phase2_figures.py —— 从 phase2 的 JSON 重绘全部图（不重跑计算，DSH.md 原则）
输出：two_channel_experiment/results/phase2/*.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase2')
DATA = os.path.join(BASE, 'twob_three_networks.json')
NETS = ['ER', 'WS', 'BA']
COLORS = {'S': 'b', 'I': 'r', 'R': 'g'}
LINESTYLES = {'true': '-', 'pred': '--'}


def load():
    return json.load(open(DATA, encoding='utf-8'))


# ---------------- 图1：训练损失曲线 ----------------
def fig_training(r, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, net in zip(axes, NETS):
        d = r[net]
        ax.plot(d['train_losses'], label='Train', linewidth=1.8)
        ax.plot(d['val_losses'], label='Val', linewidth=1.8)
        ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy Loss')
        ax.set_title(f'({net}) {net} network, best val = {d["best_val"]:.4f}')
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('Training curves of TwoChannelB on three networks', fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- 图2：自回归 5×5 网格 ----------------
def fig_autoreg_grid(r, net, path):
    d = r[net]
    tests = d['tests']
    fig, axes = plt.subplots(5, 5, figsize=(20, 18))
    for g in range(5):
        for n_inf in range(1, 6):
            ax = axes[g, n_inf - 1]
            t = tests[g * 5 + n_inf - 1]
            t_frac = np.array(t['true']); p_frac = np.array(t['pred'])
            ts = np.arange(len(t_frac))
            for i, c in enumerate('SIR'):
                ax.plot(ts, t_frac[:, i], COLORS[c] + '-', linewidth=1.3, alpha=0.85,
                        label=f'True {c}')
                ax.plot(ts, p_frac[:, i], COLORS[c] + '--', linewidth=1.3, alpha=0.85,
                        label=f'Pred {c}')
            ax.set_title(f'Group {g+1}, {n_inf} init infected\nMAE(I)={t["mae"]["I"]:.3f}',
                         fontsize=9)
            ax.set_xlabel('Time step', fontsize=7); ax.set_ylabel('Fraction', fontsize=7)
            ax.set_ylim([-0.05, 1.05]); ax.grid(alpha=0.25)
            if n_inf == 1:
                ax.legend(fontsize=6, loc='upper right')
    fig.suptitle(f'{net} network: autoregressive tests (TwoChannelB)', fontsize=15)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- 图3：误差统计（MAE/RMSE + I-RMSE 热力图） ----------------
def fig_autoreg_stats(r, net, path):
    d = r[net]
    tests = d['tests']
    mae_mat = np.zeros((5, 5)); rmse_i_mat = np.zeros((5, 5))
    for t in tests:
        g, ni = t['group'], t['n_inf'] - 1
        mae_mat[g, ni] = (t['mae']['S'] + t['mae']['I'] + t['mae']['R']) / 3
        rmse_i_mat[g, ni] = t['rmse']['I']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    # (a) 各状态 MAE 分布
    ax = axes[0]
    maes = {c: [t['mae'][c] for t in tests] for c in 'SIR'}
    ax.boxplot([maes[c] for c in 'SIR'], tick_labels=['S', 'I', 'R'])
    ax.set_ylabel('MAE'); ax.set_title(f'({net}) MAE of three states')
    ax.grid(alpha=0.3, axis='y')
    # (b) 全状态 MAE 热力图
    ax = axes[1]
    im = ax.imshow(mae_mat, cmap='YlOrRd', aspect='auto')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{mae_mat[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xlabel('Number of initial infected'); ax.set_ylabel('Seed group')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'{i+1}' for i in range(5)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f'G{i+1}' for i in range(5)])
    ax.set_title('MAE heatmap (all states)'); fig.colorbar(im, ax=ax)
    # (c) I-RMSE 热力图
    ax = axes[2]
    im = ax.imshow(rmse_i_mat, cmap='YlOrRd', aspect='auto')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{rmse_i_mat[i, j]:.2f}', ha='center', va='center', fontsize=8)
    ax.set_xlabel('Number of initial infected'); ax.set_ylabel('Seed group')
    ax.set_xticks(range(5)); ax.set_xticklabels([f'{i+1}' for i in range(5)])
    ax.set_yticks(range(5)); ax.set_yticklabels([f'G{i+1}' for i in range(5)])
    ax.set_title('RMSE heatmap (Infected)'); fig.colorbar(im, ax=ax)
    fig.suptitle(f'{net} network: autoregressive error statistics', fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- 图4：三网络汇总 MAE ----------------
def fig_summary(r, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    mae_i = [r[net]['summary']['mae_i'] for net in NETS]
    mae_all = [r[net]['summary']['mae_all'] for net in NETS]
    x = np.arange(3); w = 0.32
    ax.bar(x - w / 2, mae_i, w, label='MAE (Infected)', color='tomato', alpha=0.85)
    ax.bar(x + w / 2, mae_all, w, label='MAE (all states)', color='steelblue', alpha=0.85)
    for i, (a, b) in enumerate(zip(mae_i, mae_all)):
        ax.text(i - w / 2, a + 0.001, f'{a:.4f}', ha='center', fontsize=8)
        ax.text(i + w / 2, b + 0.001, f'{b:.4f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(NETS)
    ax.set_ylabel('Autoregressive MAE'); ax.set_title('Trajectory error on three networks (TwoChannelB)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == '__main__':
    r = load()
    fig_training(r, os.path.join(BASE, 'fig_training_curves.png'))
    for net in NETS:
        fig_autoreg_grid(r, net, os.path.join(BASE, f'fig_autoreg_{net}.png'))
        fig_autoreg_stats(r, net, os.path.join(BASE, f'fig_autoreg_stats_{net}.png'))
    fig_summary(r, os.path.join(BASE, 'fig_summary_mae.png'))
    print('phase2 figures saved to', BASE)
    for f in sorted(os.listdir(BASE)):
        if f.endswith('.png'):
            print(' -', f)
