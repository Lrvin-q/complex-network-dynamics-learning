# -*- coding: utf-8 -*-
"""
Phase 5 最终绘图：只读 results/phase5/*.json，不重跑计算（DSH.md 原则）。
主配置 = 同步训练（5B2 实证优于论文交替训练）；正则版作为"校准缓解"对比。

输出：
  fig_losses.png             三网络（同步训练）损失 + τ 退火
  fig_adjacency_{NET}.png    真实 | 软图/硬图（无正则）| 硬图（正则）对比
  fig_autoreg_{NET}.png      自回归：真实 vs 软/硬图（无正则）vs 硬图（正则）
  fig_threshold_{NET}.png    软边强度排序 + 间隙 + 自动阈值（无正则）
  fig_perk_{NET}.png         per-k 感染概率拟合（无正则硬图）
  fig_summary.png            无正则 vs 正则：AUC / MAE / 密度对比
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


def load(net, reg=False, sim=True):
    pre = 'dgsl_core_sim' if sim else 'dgsl_core'
    fn = f'{pre}{("_reg" if reg else "")}_{net}.json'
    with open(os.path.join(RESULT_DIR, fn), encoding='utf-8') as f:
        return json.load(f)


def fig_losses(d_nr):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, net in zip(axes, NETS):
        rec = d_nr[net]['losses']
        ax.plot(rec['epochs'], rec['train_learner'], label='训练（学习器）', alpha=0.7)
        ax.plot(rec['epochs'], rec['train_gen'], label='训练（生成器）', alpha=0.7)
        ax.plot(rec['epochs'], rec['val'], label='验证', linewidth=2, color='k')
        ax2 = ax.twinx()
        ax2.plot(rec['epochs'], rec['tau'], '--', color='gray', label='温度 τ')
        ax2.set_ylabel('τ')
        ax.set_xlabel('Epoch'); ax.set_ylabel('交叉熵损失')
        ax.set_title(f'{NET_LABEL[net]}（AUC={d_nr[net]["recon"]["auc_soft"]:.4f}）')
        ax.legend(loc='upper right', fontsize=8)
        ax2.legend(loc='lower left', fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('GGN 训练过程（同步训练）：损失与温度退火', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_losses.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_adjacency(d_nr, d_rg, net, crop=40):
    m_nr = d_nr[net]['matrices']
    m_rg = d_rg[net]['matrices']
    true_a = np.array(m_nr['true_adj'])
    soft_a = np.array(m_nr['soft_adj'])
    hard_a = np.array(m_nr['hard_adj'])
    hard_rg = np.array(m_rg['hard_adj'])
    titles = ['真实邻接矩阵', '软邻接矩阵（无正则）', '硬邻接矩阵（无正则）',
              f'硬邻接矩阵（正则, 阈值={d_rg[net]["recon"]["auto_threshold"]:.3f}）']
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    p5, p95 = np.percentile(soft_a, 5), np.percentile(soft_a, 95)
    mats = [true_a, soft_a, hard_a, hard_rg]
    for ax, m, t in zip(axes, mats, titles):
        if t.startswith('软'):
            im = ax.imshow(m[:crop, :crop], cmap='gray_r', vmin=p5, vmax=p95)
        else:
            im = ax.imshow(m[:crop, :crop], cmap='Greys', vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel('节点'); ax.set_ylabel('节点')
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'图结构重构：{NET_LABEL[net]}', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f'fig_adjacency_{net}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_autoreg(d_nr, d_rg, net):
    ars = d_nr[net]['autoreg_soft']
    arh = d_nr[net]['autoreg_hard']
    arh_rg = d_rg[net]['autoreg_hard']
    t = np.arange(len(ars['true_curve']))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(t, ars['true_curve'], 'k-', linewidth=2.5, label='真实 SIR 模拟')
    ax.plot(t, ars['pred_curve'], 'b--', linewidth=2, label=f'软图推演（无正则, MAE={ars["mae_i"]:.4f}）')
    ax.plot(t, arh['pred_curve'], 'r-.', linewidth=2, label=f'硬图推演（无正则, MAE={arh["mae_i"]:.4f}）')
    ax.plot(t, arh_rg['pred_curve'], 'g:', linewidth=2, label=f'硬图推演（正则, MAE={arh_rg["mae_i"]:.4f}）')
    ax.set_xlabel('时间步 t'); ax.set_ylabel('感染者占比')
    ax.set_title(f'自回归推演（多初始节点平均）：{NET_LABEL[net]}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f'fig_autoreg_{net}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_threshold(d_nr, net):
    recon = d_nr[net]['recon']
    s = np.array(recon['sorted_triu'])
    gaps = s[:-1] - s[1:]
    thr = recon['auto_threshold']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(np.arange(len(s)), s, s=4, alpha=0.6)
    axes[0].axhline(thr, color='red', linestyle='--', linewidth=1.5, label=f'自动阈值={thr:.3f}')
    axes[0].set_xlabel('边强度排名（降序）'); axes[0].set_ylabel('连边概率')
    axes[0].set_title(f'软边强度排序（{NET_LABEL[net]}）')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(np.arange(len(gaps)), gaps, color='#118AB2')
    axes[1].axvline(np.argmax(gaps), color='red', linestyle='--', linewidth=1.5)
    axes[1].set_xlabel('排名 i'); axes[1].set_ylabel('间隙 δ')
    axes[1].set_title('强度间隙（最大间隙处取阈值）')
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f'fig_threshold_{net}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_perk(d_nr, net):
    perk = d_nr[net]['perk']
    ks = sorted(int(k) for k in perk['per_k'].keys())
    counts = np.array([perk['per_k'][str(k)][0] for k in ks])
    ps = np.array([perk['per_k'][str(k)][1] for k in ks])
    ks = np.array(ks)
    keep = counts >= 20
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if keep.sum() >= 2:
        kk = ks[keep]; pp = ps[keep]
        ax.scatter(kk, pp, s=30, color='#1f77b4', label='模型输出 P(S→I)（分桶均值）')
        kth = np.linspace(1, max(kk), 200)
        if perk['beta_fit'] is not None:
            b = perk['beta_fit']
            ax.plot(kth, 1 - (1 - b) ** kth, 'r-', linewidth=2,
                    label=fr'拟合 $1-(1-\hat\beta)^k$, $\hat\beta$={b:.3f}, R²={perk["r2"]:.3f}')
        ax.plot(kth, 1 - (1 - perk['beta_true']) ** kth, 'k--', linewidth=1.5,
                label=f'真实 $1-(1-\\beta)^k$, β={perk["beta_true"]}')
    else:
        ax.text(0.5, 0.5, 'per-k 样本不足（counts<20）', ha='center', transform=ax.transAxes)
    ax.set_xlabel('感染者邻居数 k'); ax.set_ylabel('P(S→I)')
    ax.set_title(f'动力学学习诊断（无正则硬图）：{NET_LABEL[net]}')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f'fig_perk_{net}.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_summary(d_nr, d_rg):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    # (a) AUC 对比
    ax = axes[0]
    x = np.arange(len(NETS)); w = 0.22
    for i, (tag, d, c) in enumerate([('无正则', d_nr, CORE[0]), ('正则', d_rg, CORE[1])]):
        v_soft = [d[n]['recon']['auc_soft'] for n in NETS]
        v_hard = [d[n]['recon']['hard_auto']['auc_hard'] for n in NETS]
        ax.bar(x + (i * 2 - 1) * w, v_soft, w, label=f'{tag} AUC(soft)', color=c)
        ax.bar(x + (i * 2 - 1) * w + w, v_hard, w, label=f'{tag} AUC(hard)', color=c, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_ylabel('AUC')
    ax.set_title('重构判别：AUC(soft/hard)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    # (b) MAE 对比
    ax = axes[1]
    for i, (tag, d, c) in enumerate([('无正则', d_nr, CORE[0]), ('正则', d_rg, CORE[1])]):
        v = [d[n]['autoreg_hard']['mae_i'] for n in NETS]
        ax.bar(x + (i - 0.5) * w * 2, v, w * 2, label=f'{tag} MAE(I)', color=c)
        for j, vv in enumerate(v):
            ax.text(x[j] + (i - 0.5) * w * 2, vv + 0.002, f'{vv:.3f}', ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylabel('MAE（硬图推演 vs 真实）')
    ax.set_title('自回归动力学误差')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    # (c) 密度对比
    ax = axes[2]
    for i, (tag, d, c) in enumerate([('无正则', d_nr, CORE[0]), ('正则', d_rg, CORE[1])]):
        v = [np.array(d[n]['matrices']['soft_adj']).mean() for n in NETS]
        ax.bar(x + (i - 0.5) * w * 2, v, w * 2, label=f'{tag} 软图密度', color=c)
        for j, vv in enumerate(v):
            ax.text(x[j] + (i - 0.5) * w * 2, vv + 0.01, f'{vv:.3f}', ha='center', fontsize=8)
    v_true = [np.array(d_nr[n]['matrices']['true_adj']).mean() for n in NETS]
    ax.plot(x, v_true, 'k^--', markersize=9, label='真实密度')
    for j, vv in enumerate(v_true):
        ax.text(x[j], vv + 0.01, f'{vv:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylabel('边密度 mean(A)')
    ax.set_title('软图密度 vs 真实密度（补偿效应）')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    fig.suptitle('无正则 vs 正则化：重构与动力学综合对比', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, 'fig_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    d_nr = {net: load(net, reg=False, sim=True) for net in NETS}
    d_rg = {net: load(net, reg=True, sim=True) for net in NETS}
    fig_losses(d_nr)
    for net in NETS:
        fig_adjacency(d_nr, d_rg, net)
        fig_autoreg(d_nr, d_rg, net)
        fig_threshold(d_nr, net)
        fig_perk(d_nr, net)
    fig_summary(d_nr, d_rg)
    print('figures written to', RESULT_DIR, flush=True)


if __name__ == '__main__':
    main()
