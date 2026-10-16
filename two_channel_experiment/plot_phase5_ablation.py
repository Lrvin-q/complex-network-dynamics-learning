# -*- coding: utf-8 -*-
"""
Phase 5B 绘图：只读 results/phase5/ablation_*.json。
输出：
  fig_ablation_anneal.png    τ 退火 vs 固定（AUC、ACC(dyn)、损失曲线）
  fig_ablation_alt.png       交替 vs 同步（AUC、ACC(dyn)、损失曲线）
  fig_ablation_tc.png        TwoChannel vs 论文 GNN 学习器（三网络 AUC/ACC/MAE 对比）
"""
import json
import os

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase5')
NETS = ['ER', 'WS', 'BA']
NET_LABEL = {'ER': 'ER 随机网络', 'WS': 'WS 小世界网络', 'BA': 'BA 无标度网络'}
CORE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def load(path):
    with open(os.path.join(RESULT_DIR, path), encoding='utf-8') as f:
        return json.load(f)


def fig_pair_losses(results, names, labels, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (name, label) in zip(axes, zip(names, labels)):
        r = results[name]
        rec = r['losses']
        ax.plot(rec['epochs'], rec['val'], linewidth=2, label='验证损失')
        ax.plot(rec['epochs'], rec['train_learner'], alpha=0.6, label='训练（学习器）')
        ax.plot(rec['epochs'], rec['train_gen'], alpha=0.6, label='训练（生成器）')
        ax.set_title(f'{label}\nAUC(soft)={r["recon"]["auc_soft"]:.4f}, '
                     f'ACC(dyn)={r["dynamics_hard"]["acc_dyn"]:.4f}, '
                     f'best_val={rec["best_val"]:.4f}')
        ax.set_xlabel('Epoch'); ax.set_ylabel('交叉熵损失')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, out), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_ablation_anneal():
    # 退火一侧 = 核心 ER（交替）；固定一侧 = 消融文件（100ep 等长）
    d = {'anneal': load('dgsl_core_ER.json'),
         'fixed': load('ablation_anneal_ER_fixed.json')}
    names = ['anneal', 'fixed']
    labels = ['τ 退火 1.0→0.1', '固定 τ=0.5']
    fig_pair_losses(d, names, labels, '消融 5B1：温度退火 vs 固定温度（ER，100 epoch）', 'fig_ablation_anneal.png')


def fig_ablation_alt():
    d = {'alternating': load('dgsl_core_ER.json'),
         'simultaneous': load('dgsl_core_sim_ER.json')}
    names = ['alternating', 'simultaneous']
    labels = ['交替训练 (10, 20)', '同步训练']
    fig_pair_losses(d, names, labels, '消融 5B2：交替 vs 同步训练（ER，100 epoch）', 'fig_ablation_alt.png')


def fig_strategy_compare():
    """三网络 交替 vs 同步：AUC(soft) 与 MAE(I) 硬图对比。"""
    alt = {net: load(f'dgsl_core_{net}.json') for net in NETS}
    sim = {net: load(f'dgsl_core_sim_{net}.json') for net in NETS}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    x = np.arange(len(NETS)); w = 0.32
    for i, (tag, d, c) in enumerate([('交替训练', alt, CORE[0]), ('同步训练', sim, CORE[1])]):
        v = [d[n]['recon']['auc_soft'] for n in NETS]
        ax.bar(x + (i - 0.5) * w, v, w, label=tag, color=c)
        for j, vv in enumerate(v):
            ax.text(x[j] + (i - 0.5) * w, vv + 0.005, f'{vv:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylabel('AUC(soft)'); ax.set_title('训练策略对比：重构判别')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax = axes[1]
    for i, (tag, d, c) in enumerate([('交替训练', alt, CORE[0]), ('同步训练', sim, CORE[1])]):
        v = [d[n]['autoreg_hard']['mae_i'] for n in NETS]
        ax.bar(x + (i - 0.5) * w, v, w, label=tag, color=c)
        for j, vv in enumerate(v):
            ax.text(x[j] + (i - 0.5) * w, vv + 0.003, f'{vv:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylabel('MAE(I) 硬图'); ax.set_title('训练策略对比：自回归动力学')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('消融 5B2：交替 vs 同步训练（100 epoch，三网络）', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_strategy_compare.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_ablation_tc():
    core = {net: load(f'dgsl_core_{net}.json') for net in NETS}
    tc = {net: load(f'ablation_twochannel_{net}.json') for net in NETS}
    metrics = ['AUC(soft)', 'ACC(dyn)', 'MAE(I)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, metric in zip(axes, metrics):
        x = np.arange(len(NETS))
        w = 0.35
        for i, net in enumerate(NETS):
            c = core[net]
            t = tc[net]
            if metric == 'AUC(soft)':
                vc, vt = c['recon']['auc_soft'], t['recon']['auc_soft']
            elif metric == 'ACC(dyn)':
                vc, vt = c['dynamics_hard']['acc_dyn'], t['dynamics_hard']['acc_dyn']
            else:
                vc, vt = c['autoreg_hard']['mae_i'], t['autoreg_hard']['mae_i']
            ax.bar(x[i] - w / 2, vc, w, label='论文 GNN' if i == 0 else None, color=CORE[0])
            ax.bar(x[i] + w / 2, vt, w, label='TwoChannel' if i == 0 else None, color=CORE[1])
            ax.text(x[i] - w / 2, vc, f'{vc:.3f}', ha='center', va='bottom', fontsize=8)
            ax.text(x[i] + w / 2, vt, f'{vt:.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
        ax.set_title(metric)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('消融 5B3：动力学学习器架构（软邻接矩阵聚合）', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_ablation_tc.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_ablation_anneal()
    fig_ablation_alt()
    fig_strategy_compare()
    fig_ablation_tc()
    print('ablation figures written', flush=True)
