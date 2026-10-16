# -*- coding: utf-8 -*-
"""
phase3_plot.py —— 从 phase3_eval.json 重绘全部图（不重跑计算）
输出（results/phase3/）：
  fig_overview_{NET}.png      每网络 2×3 总览（逐k曲线/分桶β/分桶γ/γ相关/自回归MAE/初始数-误差）
  fig_autoreg_grid_{NET}.png  每网络双通道B的 5×5 自回归网格
  fig_perk_all.png            3×3：网络×模型的逐k曲线
  fig_summary_all.png         5 指标 × 3 网络 × 3 模型的汇总对比
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 中文字体（否则中文标签显示为方框）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase3')
DATA = os.path.join(BASE, 'phase3_eval.json')
NETS = ['ER', 'WS', 'BA']
MODELS = ['gcn_dir', 'gcn_bi', 'twob_bi']
MODEL_LABEL = {'gcn_dir': 'GCN-3L (有向, 原实现)', 'gcn_bi': 'GCN-3L (双向)',
               'twob_bi': '双通道B (双向)'}
COLORS = {'gcn_dir': '#9e9e9e', 'gcn_bi': '#e07b39', 'twob_bi': '#c62828'}
BETA_TRUE, GAMMA_TRUE = 0.3, 0.1


def load():
    return json.load(open(DATA, encoding='utf-8'))


def perk_curve(ax, rec, label, color):
    k = np.array(rec['perk']['k']); p = np.array(rec['perk']['pred'])
    ax.plot(k, p, 'o-', color=color, markersize=5, label=label, alpha=0.85)


# ---------------- 每网络 2×3 总览 ----------------
def fig_overview(r, net, path):
    d = r[net]
    labels = d['bucket_labels']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (a) 逐k曲线
    ax = axes[0, 0]
    ks = np.arange(1, 11)
    ax.plot(ks, [1 - (1 - BETA_TRUE) ** k for k in ks], 'k-', linewidth=2, label='Truth β=0.3')
    for m in MODELS:
        rec = d['models'][m]
        perk_curve(ax, rec, MODEL_LABEL[m], COLORS[m])
        ax.text(0.98, 0.98 - 0.06 * (MODELS.index(m) + 1),
                r'$\hat{\beta}$=' + f'{rec["beta_fit"][0]:.3f}',
                transform=ax.transAxes, ha='right', fontsize=9, color=COLORS[m])
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('P(S→I)')
    ax.set_title(f'({net}) P(S→I|k)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) 分桶β
    ax = axes[0, 1]
    x = np.arange(len(labels)); w = 0.25
    for i, m in enumerate(MODELS):
        rec = d['models'][m]
        vals = [rec['beta_by_bucket'].get(b, np.nan) for b in labels]
        ax.bar(x + (i - 1) * w, vals, w, label=MODEL_LABEL[m], color=COLORS[m], alpha=0.85)
    ax.axhline(BETA_TRUE, color='k', linestyle='--', label='Truth 0.3')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('Degree bucket'); ax.set_ylabel('Fitted β')
    ax.set_title(f'({net}) β by degree bucket'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (c) 分桶γ
    ax = axes[0, 2]
    for m in MODELS:
        rec = d['models'][m]
        gb = rec['gamma_by_bucket']
        vals = [gb.get(b, np.nan) for b in labels]
        ax.plot(labels, vals, 'o-', color=COLORS[m], label=MODEL_LABEL[m], markersize=5)
    ax.axhline(GAMMA_TRUE, color='k', linestyle='--', label='Truth 0.1')
    ax.set_xlabel('Degree bucket'); ax.set_ylabel('P(I→R)')
    ax.set_title(f'({net}) γ by degree bucket'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (d) γ 分桶漂移（效应量，替代 Pearson 相关——见 5.8.3 说明）
    ax = axes[1, 0]
    drifts = []
    for m in MODELS:
        gb = d['models'][m]['gamma_by_bucket']
        vals = [gb[b] for b in labels if b in gb]
        drifts.append(float(max(vals) - min(vals)) if vals else 0.0)
    bars = ax.bar(range(3), drifts, color=[COLORS[m] for m in MODELS], alpha=0.85)
    for i, v in enumerate(drifts):
        ax.text(i, v + 0.0002, f'{v:.4f}', ha='center', fontsize=9)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=8)
    ax.set_ylabel('Δγ = max−min of bucket means')
    ax.set_title(f'({net}) γ by-degree drift (effect size, truth=0)')
    ax.grid(alpha=0.3, axis='y')

    # (e) 自回归 MAE(I) 总体
    ax = axes[1, 1]
    maes = [d['models'][m]['autoreg']['mae_i_mean'] for m in MODELS]
    bars = ax.bar(range(3), maes, color=[COLORS[m] for m in MODELS], alpha=0.85)
    for i, v in enumerate(maes):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], fontsize=8)
    ax.set_ylabel('Autoregressive MAE(I)'); ax.set_title(f'({net}) trajectory error (25 tests)')
    ax.grid(alpha=0.3, axis='y')

    # (f) 自回归 MAE(I) vs 初始感染数
    ax = axes[1, 2]
    for m in MODELS:
        tests = d['models'][m]['autoreg']['tests']
        by_ni = {}
        for t in tests:
            by_ni.setdefault(t['n_inf'], []).append(t['mae']['I'])
        ni = sorted(by_ni.keys())
        vals = [np.mean(by_ni[k]) for k in ni]
        ax.plot(ni, vals, 'o-', color=COLORS[m], label=MODEL_LABEL[m], markersize=5)
    ax.set_xlabel('Number of initial infected'); ax.set_ylabel('MAE(I)')
    ax.set_title(f'({net}) error vs initial conditions'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle(f'{net} network: three-model full evaluation', fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- 双通道B 5×5 自回归网格 ----------------
def fig_autoreg_grid(r, net, path):
    tests = r[net]['models']['twob_bi']['autoreg']['tests']
    fig, axes = plt.subplots(5, 5, figsize=(20, 18))
    for g in range(5):
        for n_inf in range(1, 6):
            ax = axes[g, n_inf - 1]
            t = tests[g * 5 + n_inf - 1]
            tf = np.array(t['true']); pf = np.array(t['pred'])
            ts = np.arange(len(tf))
            for i, c in enumerate('SIR'):
                ax.plot(ts, tf[:, i], 'bgr'[i] + '-', linewidth=1.3, alpha=0.85, label=f'True {c}')
                ax.plot(ts, pf[:, i], 'bgr'[i] + '--', linewidth=1.3, alpha=0.85, label=f'Pred {c}')
            ax.set_title(f'Group {g+1}, {n_inf} init\nMAE(I)={t["mae"]["I"]:.3f}', fontsize=9)
            ax.set_xlabel('Time', fontsize=7); ax.set_ylabel('Fraction', fontsize=7)
            ax.set_ylim([-0.05, 1.05]); ax.grid(alpha=0.25)
            if n_inf == 1:
                ax.legend(fontsize=6, loc='upper right')
    fig.suptitle(f'{net}: TwoChannelB autoregressive tests (5×5)', fontsize=15)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- 3×3 逐k曲线 ----------------
def fig_perk_all(r, path):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, net in enumerate(NETS):
        d = r[net]
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            ks = np.arange(1, 11)
            ax.plot(ks, [1 - (1 - BETA_TRUE) ** k for k in ks], 'k-', linewidth=1.8, label='Truth')
            perk_curve(ax, d['models'][m], '', COLORS[m])
            rec = d['models'][m]
            title = (f'{net} | {MODEL_LABEL[m]}\n' + r'$\hat{\beta}$' +
                     f'={rec["beta_fit"][0]:.3f}±{rec["beta_fit"][1]:.3f}, '
                     f'per-k MAE={rec["perk_mae"]:.4f}')
            ax.set_title(title, fontsize=9)
            ax.set_xlabel('k', fontsize=8); ax.set_ylabel('P(S→I)', fontsize=8)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)
    fig.suptitle('Per-k infection probability: networks × models', fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------- 汇总对比 ----------------
def fig_summary_all(r, path):
    metrics = [(r'beta_err', r'|$\hat{\beta}$-0.3|'), ('perk_mae', 'per-k MAE'),
               (r'gamma_err', r'|$\hat{\gamma}$-0.1|'), ('gamma_drift', 'Δγ (bucket drift)'),
               ('autoreg_mae', 'Autoreg MAE(I)')]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    x = np.arange(3); w = 0.25
    for ax, (key, label) in zip(axes, metrics):
        for i, net in enumerate(NETS):
            for j, m in enumerate(MODELS):
                rec = r[net]['models'][m]
                if key == 'beta_err':
                    v = abs(rec['beta_fit'][0] - BETA_TRUE)
                elif key == 'perk_mae':
                    v = rec['perk_mae']
                elif key == 'gamma_err':
                    v = abs(rec['gamma']['mean'] - GAMMA_TRUE)
                elif key == 'gamma_drift':
                    gb = rec['gamma_by_bucket']
                    v = (max(gb.values()) - min(gb.values())) if gb else 0.0
                else:
                    v = rec['autoreg']['mae_i_mean']
                ax.bar(x[i] + (j - 1) * w, v, w, color=COLORS[m], alpha=0.85,
                       label=MODEL_LABEL[m] if i == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels(NETS)
        ax.set_title(label, fontsize=10); ax.grid(alpha=0.3, axis='y')
    axes[0].legend(fontsize=8)
    fig.suptitle('Summary: three networks × three models', fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == '__main__':
    r = load()
    for net in NETS:
        fig_overview(r, net, os.path.join(BASE, f'fig_overview_{net}.png'))
        fig_autoreg_grid(r, net, os.path.join(BASE, f'fig_autoreg_grid_{net}.png'))
    fig_perk_all(r, os.path.join(BASE, 'fig_perk_all.png'))
    fig_summary_all(r, os.path.join(BASE, 'fig_summary_all.png'))
    print('phase3 figures saved to', BASE)
    for f in sorted(os.listdir(BASE)):
        if f.endswith('.png'):
            print(' -', f)
