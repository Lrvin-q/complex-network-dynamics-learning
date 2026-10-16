# -*- coding: utf-8 -*-
"""
phase4_plot.py —— 五模型 × 三网络 全部报告图（只读 phase4 JSON + phase3 损失）
报告只展示以下图（其余不产出）：
  fig_training_curves.png   三网络训练损失曲线（双向双通道sum-3L）
  fig_overview_{NET}.png    三网络综合对比（5 模型，2×3 面板）
  fig_summary_all.png       5 指标 × 3 网络 × 5 模型 汇总
  fig_autoreg_grid_{NET}.png 三网络自回归 5×5 网格（真实 vs 双向双通道sum-3L）
布局修复：共享图例置于图区外顶部；柱状图数值标注上方留白；γ 桶纵轴固定 0.08~0.12；
          自回归网格主标题与子图标题不重叠。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(BASE, 'phase4')
DATA = os.path.join(OUT, 'phase4_five_models.json')
P3 = os.path.join(BASE, 'phase3', 'phase3_eval.json')
NETS = ['ER', 'WS', 'BA']
MODELS = ['gcn1l_dir', 'gcn1l_bi', 'gcn3l_bi', 'gcn3l_bi_mean', 'twob3l_bi']
MODEL_LABEL = {'gcn1l_dir': '有向 GCNsum-1L', 'gcn1l_bi': '双向 GCNsum-1L',
               'gcn3l_bi': '双向 GCNsum-3L', 'gcn3l_bi_mean': '双向 GCNmean-3L',
               'twob3l_bi': '双向双通道sum-3L'}
COLORS = {'gcn1l_dir': '#9e9e9e', 'gcn1l_bi': '#e07b39', 'gcn3l_bi': '#1565c0',
          'gcn3l_bi_mean': '#2e7d32', 'twob3l_bi': '#c62828'}
BETA_TRUE, GAMMA_TRUE = 0.3, 0.1

r = json.load(open(DATA, encoding='utf-8'))
r3 = json.load(open(P3, encoding='utf-8'))


# ---------------- 图1：训练损失曲线（twob3l_bi，损失取 phase3 同配置训练） ----------------
def fig_training():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, net in zip(axes, NETS):
        d = r3[net]['models']['twob_bi']          # phase3 同配置训练的 twob_bi
        ax.plot(d['train_losses'], label='训练', linewidth=1.8)
        ax.plot(d['val_losses'], label='验证', linewidth=1.8)
        ax.set_xlabel('轮次（Epoch）'); ax.set_ylabel('交叉熵损失')
        ax.set_title(f'({net}) {net}，最优验证损失 = {d["best_val"]:.4f}')
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle('三网络训练损失曲线（双向双通道sum-3L）', y=0.9, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, 'fig_training_curves.png'), dpi=150)
    plt.close(fig)


def perk_curve(ax, rec, color):
    k = np.array(rec['perk']['k']); p = np.array(rec['perk']['pred'])
    ax.plot(k, p, 'o-', color=color, markersize=4.5, alpha=0.85)


# ---------------- 图2：综合对比总览（5 模型） ----------------
def fig_overview(net):
    d = r[net]
    labels = d['bucket_labels']
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    # (a) 逐k曲线
    ax = axes[0, 0]
    ks = np.arange(1, 11)
    ax.plot(ks, [1 - (1 - BETA_TRUE) ** k for k in ks], 'k-', linewidth=2, label='真实 β=0.3')
    for m in MODELS:
        perk_curve(ax, d['models'][m], COLORS[m])
    ax.set_xlabel('感染者邻居数 k'); ax.set_ylabel('P(S→I)')
    ax.set_title(f'(a) ({net}) 感染概率 P(S→I|k)'); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (b) 分桶β（图例移出图区，由整图共享）
    ax = axes[0, 1]
    x = np.arange(len(labels)); w = 0.14
    maxv = 0.0
    for i, m in enumerate(MODELS):
        rec = d['models'][m]
        vals = [rec['beta_by_bucket'].get(b, np.nan) for b in labels]
        ax.bar(x + (i - 2) * w, vals, w, label=MODEL_LABEL[m], color=COLORS[m], alpha=0.85)
        maxv = max(maxv, float(np.nanmax(vals)))
    ax.axhline(BETA_TRUE, color='k', linestyle='--', linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel('度数分桶'); ax.set_ylabel('拟合 β')
    ax.set_title(f'(b) ({net}) 分度数桶的感染率 β')
    ax.set_ylim(0, maxv * 1.15)
    ax.grid(alpha=0.3, axis='y')

    # (c) 分桶γ（纵轴固定 0.08~0.12）
    ax = axes[0, 2]
    for m in MODELS:
        gb = d['models'][m]['gamma_by_bucket']
        vals = [gb.get(b, np.nan) for b in labels]
        ax.plot(labels, vals, 'o-', color=COLORS[m], markersize=4.5)
    ax.axhline(GAMMA_TRUE, color='k', linestyle='--', linewidth=1.2)
    ax.set_xlabel('度数分桶'); ax.set_ylabel('P(I→R)')
    ax.set_title(f'(c) ({net}) 分度数桶的恢复率 γ')
    if net == 'BA':
        ax.set_ylim(0.04, 0.11)
    else:
        ax.set_ylim(0.078, 0.125)
    ax.grid(alpha=0.3)

    # (d) Δγ 效应量（标注上方留白）
    ax = axes[1, 0]
    drifts = []
    for m in MODELS:
        gb = d['models'][m]['gamma_by_bucket']
        vals = [gb[b] for b in labels if b in gb]
        drifts.append(float(max(vals) - min(vals)) if vals else 0.0)
    ax.bar(range(5), drifts, color=[COLORS[m] for m in MODELS], alpha=0.85)
    for i, v in enumerate(drifts):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(range(5)); ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS],
                                                rotation=18, ha='right', fontsize=8)
    ax.set_ylabel('Δγ = 分桶均值最大值−最小值')
    ax.set_title(f'(d) ({net}) 恢复率随度数的漂移（理想为 0）')
    ax.set_ylim(0, max(drifts) * 1.2 + 0.001)
    ax.grid(alpha=0.3, axis='y')

    # (e) 自回归 MAE(I)（标注上方留白）
    ax = axes[1, 1]
    maes = [d['models'][m]['autoreg']['mae_i_mean'] for m in MODELS]
    ax.bar(range(5), maes, color=[COLORS[m] for m in MODELS], alpha=0.85)
    for i, v in enumerate(maes):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(range(5)); ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS],
                                                rotation=18, ha='right', fontsize=8)
    ax.set_ylabel('自回归 MAE(I)')
    ax.set_title(f'(e) ({net}) 自回归误差（25 个测试）')
    ax.set_ylim(0, max(maes) * 1.2 + 0.002)
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
        ax.plot(ni, vals, 'o-', color=COLORS[m], markersize=4.5)
    ax.set_xlabel('初始感染人数'); ax.set_ylabel('MAE(I)')
    ax.set_title(f'(f) ({net}) 误差 vs 初始条件')
    ax.set_ylim(0, 0.08)
    ax.grid(alpha=0.3)

    # 共享图例（图区外顶部）与主标题
    handles = [plt.Line2D([0], [0], color=COLORS[m], marker='o', linestyle='-', label=MODEL_LABEL[m])
               for m in MODELS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f'{net} 网络：五模型评估', y=0.02, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(os.path.join(OUT, f'fig_overview_{net}.png'), dpi=150)
    plt.close(fig)


# ---------------- 图3：汇总对比 ----------------
def fig_summary():
    metrics = [(r'beta_err', r'感染率误差 $|\hat{\beta}-0.3|$'), ('perk_mae', 'per-k 平均误差'),
               (r'gamma_err', r'恢复率误差 $|\hat{\gamma}-0.1|$'), ('gamma_drift', 'Δγ（分桶漂移）'),
               ('autoreg_mae', '自回归 MAE(I)')]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    x = np.arange(3); w = 0.14
    for ax, (key, label) in zip(axes, metrics):
        maxv = 0.0
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
                    v = rec['delta_gamma'] or 0.0
                else:
                    v = rec['autoreg']['mae_i_mean']
                ax.bar(x[i] + (j - 2) * w, v, w, color=COLORS[m], alpha=0.85)
                maxv = max(maxv, float(v))
        ax.set_xticks(x); ax.set_xticklabels(NETS)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, maxv * 1.1 + 0.002)
        ax.grid(alpha=0.3, axis='y')
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[m], label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.9),
               ncol=5, fontsize=9, frameon=False)
    fig.suptitle('汇总：三网络 × 五模型', y=0.04, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(os.path.join(OUT, 'fig_summary_all.png'), dpi=150)
    plt.close(fig)


# ---------------- 图4：自回归 5×5 网格（twob3l_bi） ----------------
def fig_autoreg_grid(net):
    tests = r[net]['models']['twob3l_bi']['autoreg']['tests']
    fig, axes = plt.subplots(5, 5, figsize=(20, 18))
    for g in range(5):
        for n_inf in range(1, 6):
            ax = axes[g, n_inf - 1]
            t = tests[g * 5 + n_inf - 1]
            tf = np.array(t['true']); pf = np.array(t['pred'])
            ts = np.arange(len(tf))
            for i, c in enumerate('SIR'):
                ax.plot(ts, tf[:, i], 'bgr'[i] + '-', linewidth=1.3, alpha=0.85, label=f'真实 {c}')
                ax.plot(ts, pf[:, i], 'bgr'[i] + '--', linewidth=1.3, alpha=0.85, label=f'预测 {c}')
            ax.set_title(f'G{g+1} 组 · {n_inf} 个初始感染者\nMAE(I)={t["mae"]["I"]:.3f}', fontsize=9)
            ax.set_xlabel('时间步', fontsize=7); ax.set_ylabel('占比', fontsize=7)
            ax.set_ylim([-0.05, 1.05]); ax.grid(alpha=0.25)
            if n_inf == 1:
                ax.legend(fontsize=6, loc='upper right')
    fig.suptitle(f'{net} 网络：双向双通道sum-3L 自回归测试（5×5）', y=0.95, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(OUT, f'fig_autoreg_grid_{net}.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_training()
    for net in NETS:
        fig_overview(net)
        fig_autoreg_grid(net)
    fig_summary()
    print('phase4 figures saved to', OUT)
    for f in sorted(os.listdir(OUT)):
        if f.endswith('.png'):
            print(' -', f)
