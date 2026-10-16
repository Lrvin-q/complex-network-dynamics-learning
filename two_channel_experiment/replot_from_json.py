# -*- coding: utf-8 -*-
"""
replot_from_json.py —— 只读结果 JSON 重绘全部实验图（不重跑计算）
原则（DSH.md）：保存的数据必须自包含可 1:1 重绘；本脚本不依赖任何脚本内硬编码常量，
所有配置（分桶、max_k、超参等）均从 JSON 的 config 字段读取。
输出：two_channel_experiment/results/replot/
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(BASE, 'replot')
os.makedirs(OUT, exist_ok=True)

BETA_TRUE = 0.3


# ============================ 工具 ============================
def plot_training(train_losses, val_losses, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy'); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_k_curve(perk, beta_est, title, path):
    ks = np.array(perk['k']); pred = np.array(perk['pred']); truth = np.array(perk['truth'])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, truth, 'k-', linewidth=2, label=f'True 1-(1-β)^k, β={BETA_TRUE}')
    ax.plot(ks, pred, 'ro-', markersize=6, label=f'Model (β≈{beta_est:.3f})')
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('P(S→I)')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_heatmap(err_matrix, bucket_labels, max_k, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(err_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(1, max_k + 1))
    ax.set_xticklabels([str(k) for k in range(1, max_k + 1)])
    ax.set_yticks(range(len(bucket_labels)))
    ax.set_yticklabels(bucket_labels)
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('Degree bucket')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    for i in range(err_matrix.shape[0]):
        for j in range(err_matrix.shape[1]):
            v = err_matrix[i, j]
            if v >= 0:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='black' if v < 0.35 else 'white')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ============================ Phase 0 ============================
def replot_phase0():
    path = os.path.join(BASE, 'phase0', 'phase0_results.json')
    r0 = json.load(open(path, encoding='utf-8'))
    cfg = r0['config']
    labels = cfg['diagnostics']['bucket_labels']
    max_k = cfg['diagnostics']['max_k']
    for key in ('asym', 'bi'):
        m = r0[key]
        tag = f'phase0_{key}'
        plot_training(m['train_losses'], m['val_losses'], f'{m["name"]} training',
                      os.path.join(OUT, f'training_{tag}.png'))
        beta_est = m['fit_beta_overall'][0] if m['fit_beta_overall'] else float('nan')
        plot_k_curve(m['perk'], beta_est, f'{m["name"]}: P(S→I|k)',
                     os.path.join(OUT, f'kcurve_{tag}.png'))
        plot_heatmap(np.array(m['heatmap']), labels, max_k,
                     f'{m["name"]}: |error| (degree×k)', os.path.join(OUT, f'heatmap_{tag}.png'))
    print('phase0 replot done')


# ============================ Phase 1 ============================
def replot_phase1():
    res_path = os.path.join(BASE, 'phase1', 'phase1_results.json')
    sum_path = os.path.join(BASE, 'phase1', 'phase1_summary.json')
    r1 = json.load(open(res_path, encoding='utf-8'))
    s1 = json.load(open(sum_path, encoding='utf-8'))
    cfg = s1['config']
    labels = cfg['diagnostics']['bucket_labels']
    max_k = cfg['diagnostics']['max_k']
    names = list(cfg['models'].keys())
    colors = {'baseline_3L': 'gray', 'baseline_1L': 'orange', 'two_channel_A': 'red',
              'two_channel_Amean': 'blue', 'two_channel_B': 'purple'}

    # --- comparison_bar（3 子图：β误差 / γ-度数相关 / 轨迹MAE） ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    beta_err = [abs(s1[n]['beta'][0] - BETA_TRUE) for n in names]
    beta_std = [s1[n]['beta'][1] for n in names]
    axes[0].bar(range(len(names)), beta_err, yerr=beta_std, capsize=4, color='steelblue', alpha=0.8)
    axes[0].axhline(0, color='k', linewidth=0.8)
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(names, rotation=20, ha='right')
    axes[0].set_ylabel('|β_est − 0.3|'); axes[0].set_title('β estimation error')

    corrs = [s1[n]['gamma_corr'][0] for n in names]
    axes[1].bar(range(len(names)), corrs, color='tomato', alpha=0.8)
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, rotation=20, ha='right')
    axes[1].set_ylabel('corr(γ, degree)'); axes[1].set_title('γ–degree correlation (true=0)')

    maes = [s1[n]['traj_mae'][0] for n in names]
    axes[2].bar(range(len(names)), maes, color='seagreen', alpha=0.8)
    axes[2].set_xticks(range(len(names))); axes[2].set_xticklabels(names, rotation=20, ha='right')
    axes[2].set_ylabel('MAE of I-fraction'); axes[2].set_title('Autoregressive trajectory error')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'comparison_bar.png'), dpi=150)
    plt.close(fig)

    # --- perk_curves（每个模型第一个 seed，即 seed=21） ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = np.arange(1, max_k + 1)
    ax.plot(ks, [1 - (1 - BETA_TRUE) ** k for k in ks], 'k-', linewidth=2, label='True β=0.3')
    for name in names:
        r0 = r1[name][0]
        ax.plot(r0['perk']['k'], r0['perk']['pred'], 'o-', color=colors[name],
                label=f"{name} (seed {r0['seed']})", markersize=5, alpha=0.8)
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('P(S→I)')
    ax.set_title('Per-k infection probability (bidirectional edges)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'perk_curves.png'), dpi=150)
    plt.close(fig)

    # --- beta_by_bucket（均值±std 分组柱状图） ---
    bucket_names = list(s1[names[0]]['beta_by_bucket'].keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bucket_names)); w = 0.15
    for i, name in enumerate(names):
        vals = [s1[name]['beta_by_bucket'].get(b, (np.nan, 0))[0] for b in bucket_names]
        ax.bar(x + (i - len(names) / 2 + 0.5) * w, vals, w, label=name, alpha=0.85)
    ax.axhline(BETA_TRUE, color='k', linestyle='--', label='True β=0.3')
    ax.set_xticks(x); ax.set_xticklabels(bucket_names)
    ax.set_xlabel('Degree bucket'); ax.set_ylabel('Fitted β')
    ax.set_title('β by degree bucket'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'beta_by_bucket.png'), dpi=150)
    plt.close(fig)
    print('phase1 replot done')


if __name__ == '__main__':
    replot_phase0()
    replot_phase1()
    files = sorted(os.listdir(OUT))
    print(f'\nreplot 完成，共 {len(files)} 张图 -> {OUT}')
    for f in files:
        print(' -', f)
