# -*- coding: utf-8 -*-
"""
Phase 6 最终绘图：只读 results/phase6/dgsl_l1b_{ER,WS,BA}.json（最终方案：分层重采样 + valCE 筛选）。
全部标题/标签为中文，面板带 (a)(b)(c) 编号。

输出：
  fig6_losses.png              三网络训练损失 + 指数降温（图1）
  fig6_summary.png             三网络综合对比 (a)(b)(c)（图2）
  fig6_adjacency_all.png       三网络 × (True/Hard) 3×2，完整 100×100（图3）
  fig6_overview_all.png        参数检验 + 自回归汇总 2×3（图4）
  fig6_autoreg_grid_{NET}.png  自回归 5×5 网格（图5-7）
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
NETS = ['ER', 'WS', 'BA']
NET_LABEL = {'ER': 'ER 随机网络', 'WS': 'WS 小世界网络', 'BA': 'BA 无标度网络'}
NET_COLORS = {'ER': '#1565c0', 'WS': '#e07b39', 'BA': '#2e7d32'}
BETA_TRUE, GAMMA_TRUE = 0.3, 0.1


def load(net):
    with open(os.path.join(RESULT_DIR, f'dgsl_l1b_{net}.json'), encoding='utf-8') as f:
        return json.load(f)


def bucket_labels(buckets):
    return [f'{lo}-{hi - 1}' for lo, hi in buckets]


# ---------------- 图1：训练损失（中文） ----------------
def fig6_losses(data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, net in zip(axes, NETS):
        rec = data[net]['losses']
        ax.plot(rec['epochs'], rec['train_learner'], label='训练（学习器）', alpha=0.7)
        ax.plot(rec['epochs'], rec['train_gen'], label='训练（生成器）', alpha=0.7)
        ax.plot(rec['epochs'], rec['val'], label='验证（硬图）', linewidth=2, color='k')
        ax2 = ax.twinx()
        ax2.plot(rec['epochs'], rec['tau'], '--', color='gray', label='温度 τ')
        ax2.set_ylabel('τ')
        ax.set_xlabel('轮次（Epoch）'); ax.set_ylabel('交叉熵损失')
        h = data[net]['recon_hard_05']
        ax.set_title(f'({net}) {NET_LABEL[net]}\nACC(硬图)={h["acc"]:.3f}, 最优验证损失={rec["best_val"]:.4f}',
                     fontsize=9)
        ax.legend(loc='upper right', fontsize=8)
        ax2.legend(loc='lower left', fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('训练过程（硬图学习器 + 分层重采样）：损失曲线与指数降温（3→0.1）',
                 y=0.95, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(RESULT_DIR, 'fig6_losses.png'), dpi=150)
    plt.close(fig)


# ---------------- 图2：综合对比 (a)(b)(c)（修正数据：只读最终 JSON） ----------------
def fig6_summary(data):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x = np.arange(len(NETS))

    # (a) 硬图重构主指标
    ax = axes[0]
    metrics = [('ACC', 'acc', '#1f77b4'), ('TPR', 'tpr', '#2ca02c'), ('FPR', 'fpr', '#d62728')]
    w = 0.26
    for j, (label, key, color) in enumerate(metrics):
        vals = [data[n]['recon_hard_05'][key] for n in NETS]
        ax.bar(x + (j - 1) * w, vals, w, label=label, color=color)
        for i, v in enumerate(vals):
            ax.text(x[i] + (j - 1) * w, v + 0.01, f'{v:.3f}', ha='center', fontsize=7)
    ax.axhline(1.0, color='k', linestyle=':', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('数值')
    ax.set_title('(a) 硬图重构主指标（0.5 阈值）')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

    # (b) 边数还原度与自回归误差
    ax = axes[1]
    ratio = [data[n]['recon_hard_05']['n_edges_pred'] / max(data[n]['recon_hard_05']['n_edges_true'], 1)
             for n in NETS]
    mae5 = [data[n]['eval_extended']['autoreg_5x5']['mae_i_mean'] for n in NETS]
    ax.bar(x - w / 2, ratio, w, label='预测边数 / 真实边数', color=NET_COLORS[NETS[0]])
    ax.bar(x + w / 2, np.array(mae5) * 10, w, label='5×5 自回归 MAE(I)×10', color=NET_COLORS[NETS[1]])
    for i, n in enumerate(NETS):
        h = data[n]['recon_hard_05']
        ax.text(x[i] - w / 2, ratio[i] + 0.03, f'{h["n_edges_pred"]}/{h["n_edges_true"]}',
                ha='center', fontsize=8)
        ax.text(x[i] + w / 2, mae5[i] * 10 + 0.03, f'{mae5[i]:.4f}', ha='center', fontsize=8)
    ax.axhline(1.0, color='k', linestyle=':', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylabel('边数比 / MAE×10')
    ax.set_title('(b) 边数还原度与自回归动力学误差')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(ratio), max(mae5) * 10) * 1.4 + 0.06)

    # (c) 密度对比
    ax = axes[2]
    for i, n in enumerate(NETS):
        soft_d = np.array(data[n]['matrices']['soft_adj']).mean()
        true_d = np.array(data[n]['matrices']['true_adj']).mean()
        ax.bar(x[i] - w / 2, soft_d, w, label='软图密度' if i == 0 else None, color=NET_COLORS[NETS[0]])
        ax.bar(x[i] + w / 2, true_d, w, label='真实密度' if i == 0 else None, color=NET_COLORS[NETS[2]])
        ax.text(x[i] - w / 2, soft_d + 0.004, f'{soft_d:.3f}', ha='center', fontsize=8)
        ax.text(x[i] + w / 2, true_d + 0.004, f'{true_d:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NET_LABEL[n] for n in NETS], fontsize=8)
    ax.set_ylabel('边密度')
    ax.set_title('(c) 软图密度 vs 真实密度')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max([np.array(data[n]['matrices']['soft_adj']).mean()
                        for n in NETS]) * 1.35 + 0.01)

    fig.suptitle('三网络综合对比（硬图重构 / 边数 / 密度）', y=0.98, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(RESULT_DIR, 'fig6_summary.png'), dpi=150)
    plt.close(fig)


# ---------------- 图3：邻接矩阵 3×2（中文 + (a)-(f) 编号，完整 100×100） ----------------
def fig6_adjacency_all(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    idx = 0
    for r, net in enumerate(NETS):
        m = data[net]['matrices']
        true_a = np.array(m['true_adj'])
        hard_a = np.array(m['hard_adj'])
        h = data[net]['recon_hard_05']
        panels = [(true_a, '真实邻接矩阵'),
                  (hard_a, f'硬图（ACC={h["acc"]:.3f}，{h["n_edges_pred"]}/{h["n_edges_true"]} 条边）')]
        for c, (mat, title) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(mat, cmap='Greys', vmin=0, vmax=1)
            ax.set_title(f'({labels[idx]}) {NET_LABEL[net]} · {title}', fontsize=10)
            ax.set_xlabel('节点'); ax.set_ylabel('节点')
            plt.colorbar(im, ax=ax, fraction=0.04)
            idx += 1
    fig.suptitle('网络结构重构：真实邻接矩阵 vs 硬图（0.5 阈值，完整 100×100 节点）',
                 y=0.995, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(RESULT_DIR, 'fig6_adjacency_all.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------- 图4：参数检验 + 自回归汇总 2×3（中文，(f) 去数值） ----------------
def fig6_overview_all(data):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    all_mae = {net: data[net]['eval_extended']['autoreg_5x5']['tests'] for net in NETS}
    bucket_ranges = {net: bucket_labels(data[net]['eval_extended']['config']['buckets'])
                     for net in NETS}

    # (a) per-k 曲线
    ax = axes[0, 0]
    ks = np.arange(1, 11)
    ax.plot(ks, [1 - (1 - BETA_TRUE) ** k for k in ks], 'k-', linewidth=2, label='真实 β=0.3')
    for net in NETS:
        perk = data[net]['eval_extended']['beta']['perk']
        ax.plot(perk['k'], perk['pred'], 'o-', color=NET_COLORS[net], markersize=4.5, alpha=0.85)
    ax.set_title('(a) 感染概率 P(S→I|k) vs 真实曲线')
    ax.set_xlabel('感染者邻居数 k'); ax.set_ylabel('P(S→I)')
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # (b) 分桶β
    ax = axes[0, 1]
    x = np.arange(4); w = 0.26
    maxv = BETA_TRUE
    for i, net in enumerate(NETS):
        bb = data[net]['eval_extended']['beta']['beta_by_bucket']
        labels = bucket_ranges[net]
        vals = [bb.get(b, np.nan) for b in labels] + [np.nan] * (4 - len(labels))
        ax.bar(x + (i - 1) * w, vals, w, color=NET_COLORS[net], alpha=0.85, label=net)
        for j, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[j] + (i - 1) * w, v, f'{v:.3f}', ha='center', va='bottom', fontsize=7)
                maxv = max(maxv, float(v))
    ax.axhline(BETA_TRUE, color='k', linestyle='--', linewidth=1.2)
    ax.set_title('(b) 分度数桶的感染率 β')
    ax.set_xticks(x); ax.set_xticklabels([f'桶{j+1}' for j in range(4)])
    ax.set_xlabel('度数分桶（ER: 3-7/8-10/11-13/14-16, WS: 4-5/6-7/8-9, BA: 3-5/6-9/10-13/14-33）',
                  fontsize=7)
    ax.set_ylabel('拟合 β')
    ax.set_ylim(0, maxv * 1.18)
    ax.grid(alpha=0.3, axis='y')

    # (c) 分桶γ
    ax = axes[0, 2]
    for net in NETS:
        gb = data[net]['eval_extended']['gamma']['gamma_by_bucket']
        labels = bucket_ranges[net]
        vals = [gb.get(b, np.nan) for b in labels] + [np.nan] * (4 - len(labels))
        ax.plot(range(4), vals, 'o-', color=NET_COLORS[net], markersize=5)
    ax.axhline(GAMMA_TRUE, color='k', linestyle='--', linewidth=1.2, label=f'真实 γ={GAMMA_TRUE}')
    ax.set_title('(c) 分度数桶的恢复率 γ')
    ax.set_xticks(x); ax.set_xticklabels([f'桶{j+1}' for j in range(4)])
    ax.set_xlabel('度数分桶（同 (b)）', fontsize=7)
    ax.set_ylabel('P(I→R)')
    ax.set_ylim(0.08, 0.12)
    ax.grid(alpha=0.3)

    # (d) Δγ
    ax = axes[1, 0]
    drifts = [data[net]['eval_extended']['gamma']['delta_gamma'] or 0.0 for net in NETS]
    ax.bar(range(3), drifts, 0.5, color=[NET_COLORS[n] for n in NETS], alpha=0.85)
    for i, v in enumerate(drifts):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(NETS)
    ax.set_ylabel('Δγ = 分桶均值最大值 − 最小值')
    ax.set_title('(d) 恢复率随度数的漂移（理想为 0）')
    ax.set_ylim(0, max(drifts) * 1.3 + 0.0005)
    ax.grid(alpha=0.3, axis='y')

    # (e) 5×5 自回归 MAE(I)
    ax = axes[1, 1]
    means = [float(np.mean([t['mae']['I'] for t in all_mae[n]])) for n in NETS]
    ax.bar(range(3), means, 0.5, color=[NET_COLORS[n] for n in NETS], alpha=0.85)
    for i, v in enumerate(means):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(NETS)
    ax.set_ylabel('MAE(I)')
    ax.set_title('(e) 5×5 自回归 MAE(I)（25 测试平均）')
    ax.set_ylim(0, max(means) * 1.3 + 0.001)
    ax.grid(alpha=0.3, axis='y')

    # (f) MAE(I) vs 初始感染人数（不标注数值）
    ax = axes[1, 2]
    for net in NETS:
        by_ni = {}
        for t in all_mae[net]:
            by_ni.setdefault(t['n_inf'], []).append(t['mae']['I'])
        ni = sorted(by_ni.keys())
        vals = [float(np.mean(by_ni[k])) for k in ni]
        ax.plot(ni, vals, 'o-', color=NET_COLORS[net], markersize=5)
    ax.set_xlabel('初始感染人数'); ax.set_ylabel('MAE(I)')
    ax.set_title('(f) 自回归误差 vs 初始感染人数（G1-G5 平均）')
    ax.set_ylim(0, max([float(np.mean([t['mae']['I'] for t in all_mae[n]])) for n in NETS]) * 1.5 + 0.001)
    ax.grid(alpha=0.3)

    handles = [plt.Line2D([0], [0], color=NET_COLORS[n], marker='o', linestyle='-', label=NET_LABEL[n])
               for n in NETS] + \
              [plt.Line2D([0], [0], color='k', linestyle='-', linewidth=2, label='真实 β=0.3'),
               plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.2,
                          label=f'真实 γ={GAMMA_TRUE}')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.97),
               ncol=5, fontsize=10, frameon=False)
    fig.suptitle('参数检验与自回归汇总（硬图学习器 + 分层重采样）——三网络',
                 y=0.04, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(os.path.join(RESULT_DIR, 'fig6_overview_all.png'), dpi=150)
    plt.close(fig)


# ---------------- 图5-7：自回归 5×5 网格（中文） ----------------
def fig6_autoreg_grid(net):
    tests = data[net]['eval_extended']['autoreg_5x5']['tests']
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
    fig.suptitle(f'{NET_LABEL[net]}：自回归推演（真实实线 vs 硬图预测虚线，5×5 测试）',
                 y=0.95, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(os.path.join(RESULT_DIR, f'fig6_autoreg_grid_{net}.png'), dpi=150)
    plt.close(fig)


data = {net: load(net) for net in NETS}


def main():
    fig6_losses(data)
    fig6_summary(data)
    fig6_adjacency_all(data)
    fig6_overview_all(data)
    for net in NETS:
        fig6_autoreg_grid(net)
    print('phase6 final figures saved to', RESULT_DIR, flush=True)


if __name__ == '__main__':
    main()
