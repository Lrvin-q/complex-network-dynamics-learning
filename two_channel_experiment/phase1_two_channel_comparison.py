# -*- coding: utf-8 -*-
"""
phase1_two_channel_comparison.py —— Phase 1：双通道模型对比消融（全部使用修正后的双向 edge_index）
模型 × 3 seeds：
  (1) baseline_3L  : 现有 GraphConv x3 (sum)          —— 基线
  (2) baseline_1L  : 现有 GraphConv x1 (sum)          —— 排除层数因素
  (3) two_channel_A: 双通道 1-hop（自身+邻居sum聚合）    —— 主推
  (4) two_channel_Amean: 双通道但邻居用 mean           —— 复验 sum 优势
  (5) two_channel_B: 多层分离版（自身通道不聚合）        —— 泛化形态
指标：逐k P(S→I) 误差、拟合β、按度数分桶β、γ-度数相关、自回归I-MAE
输出：two_channel_experiment/results/phase1/
"""
import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_ba_graph, generate_dataset, build_loaders,
                    GraphConvSIR, TwoChannelSIR, TwoChannelGCN, train_model,
                    collect_si_samples, fit_beta, compute_perk_stats,
                    collect_gamma_samples, trajectory_mae, save_json)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase1')
os.makedirs(OUT, exist_ok=True)

# ---------------- 配置 ----------------
N, M = 100, 3
BETA, GAMMA = 0.3, 0.1
N_SEEDS, N_REPEATS, T_MAX = 30, 5, 60
EPOCHS, HIDDEN, DROPOUT, LR, BATCH = 30, 64, 0.2, 0.01, 32
SEEDS = [21, 42, 123]
DEGREE_BUCKETS = [(3, 6), (6, 10), (10, 14), (14, 20), (20, 34)]
MAX_K = 10

MODELS = {
    'baseline_3L': lambda: GraphConvSIR(3, HIDDEN, 3, num_layers=3, dropout=DROPOUT, aggr='sum'),
    'baseline_1L': lambda: GraphConvSIR(3, HIDDEN, 3, num_layers=1, dropout=DROPOUT, aggr='sum'),
    'two_channel_A': lambda: TwoChannelSIR(3, HIDDEN, 3, dropout=DROPOUT, aggr='sum'),
    'two_channel_Amean': lambda: TwoChannelSIR(3, HIDDEN, 3, dropout=DROPOUT, aggr='mean'),
    'two_channel_B': lambda: TwoChannelGCN(3, HIDDEN, 3, num_layers=2, dropout=DROPOUT),
}

print("=" * 70)
print("Phase 1: 双通道模型对比 (双向edge_index, 5模型 × 3 seeds)")
print(f"β=0.3 γ=0.1, hidden={HIDDEN}, epochs={EPOCHS}")
print("=" * 70)

# ---------------- 图与数据（固定，跨seed共享） ----------------
set_seed(SEEDS[0])
edge_index_bi, edge_index_dir, G = generate_ba_graph(N, M)
adj_list = [[] for _ in range(N)]
for u, v in edge_index_dir.t().tolist():
    adj_list[u].append(v); adj_list[v].append(u)

dataset = generate_dataset(edge_index_dir, BETA, GAMMA, N_SEEDS, N_REPEATS, T_MAX)
train_loader, val_loader = build_loaders(dataset, BATCH, val_frac=0.2, seed=SEEDS[0])
print(f"样本数: {len(dataset)} (train {len(train_loader.dataset)} / val {len(val_loader.dataset)})")

def run_one(model, seed):
    set_seed(seed)
    tr, va = train_model(model, train_loader, val_loader, edge_index_bi,
                         epochs=EPOCHS, lr=LR, verbose_every=15)
    res = {'seed': seed, 'best_val': min(va), 'final_train': tr[-1], 'final_val': va[-1]}
    # 逐k
    samples_by_k, samples_by_bk, counts_by_k, _, _ = collect_si_samples(
        model, edge_index_bi, adj_list, DEGREE_BUCKETS, max_k=MAX_K,
        min_per_k=250, min_per_bk=80, max_iters=6000, verbose=False)
    perk = compute_perk_stats(samples_by_k, BETA)
    res['perk'] = perk
    fit = fit_beta(perk['k'], [samples_by_k[k] for k in perk['k']])
    if fit is None:
        fit = [float('nan'), float('nan'), float('nan')]
    res['beta_fit'] = fit
    res['perk_mae'] = float(np.mean(perk['err'])) if perk['err'] else None
    # 分桶β
    res['beta_by_bucket'] = {}
    for b, (lo, hi) in enumerate(DEGREE_BUCKETS):
        klist = [k for k in range(1, MAX_K + 1) if len(samples_by_bk.get((b, k), [])) >= 20]
        if len(klist) >= 2:
            f2 = fit_beta(klist, [samples_by_bk[(b, k)] for k in klist])
            if f2:
                res['beta_by_bucket'][f'{lo}-{hi-1}'] = f2[0]
    # γ
    probs_i, deg_i, corr, pval = collect_gamma_samples(model, edge_index_bi, adj_list,
                                                       n_states=1200, verbose=False)
    res['gamma'] = {'mean': float(probs_i.mean()), 'corr_degree': float(corr), 'pval': float(pval)}
    # 轨迹
    maes = []
    for n_inf in (1, 2, 3):
        state = np.zeros(N, dtype=int)
        state[np.random.choice(N, n_inf, replace=False)] = 1
        maes.append(trajectory_mae(model, edge_index_bi, state, BETA, GAMMA,
                                   T=50, n_mc=30)['mae_i'])
    res['traj_mae'] = float(np.mean(maes))
    return res

results = {}
for name, factory in MODELS.items():
    results[name] = []
    print(f"\n### 模型 {name} ###")
    for seed in SEEDS:
        model = factory()
        print(f"  seed={seed}: 训练({sum(p.numel() for p in model.parameters())}参数)...")
        r = run_one(model, seed)
        print(f"    β拟合={r['beta_fit'][0]:.4f}±{r['beta_fit'][1]:.4f} | "
              f"perk_MAE={r['perk_mae']:.4f} | γ={r['gamma']['mean']:.4f} "
              f"(相关{r['gamma']['corr_degree']:.4f}) | 轨迹MAE={r['traj_mae']:.4f}")
        results[name].append(r)

# ---------------- 汇总 ----------------
print("\n" + "=" * 100)
print("汇总 (均值±标准差, 3 seeds)")
print("=" * 100)
summary = {}
for name, runs in results.items():
    s = {}
    for key in ('beta_fit', 'perk_mae', 'gamma', 'traj_mae'):
        if key == 'beta_fit':
            vals = [r[key][0] for r in runs]
            s['beta'] = (float(np.mean(vals)), float(np.std(vals)))
        elif key == 'gamma':
            s['gamma_mean'] = (float(np.mean([r[key]['mean'] for r in runs])),
                               float(np.std([r[key]['mean'] for r in runs])))
            s['gamma_corr'] = (float(np.mean([r[key]['corr_degree'] for r in runs])),
                               float(np.std([r[key]['corr_degree'] for r in runs])))
        else:
            vals = [r[key] for r in runs]
            s[key] = (float(np.mean(vals)), float(np.std(vals)))
    # 分桶β (仅seed0展示，或三seed平均)
    bk = {}
    for b, (lo, hi) in enumerate(DEGREE_BUCKETS):
        v = [r['beta_by_bucket'].get(f'{lo}-{hi-1}') for r in runs]
        v = [x for x in v if x is not None]
        if v:
            bk[f'{lo}-{hi-1}'] = (float(np.mean(v)), float(np.std(v)))
    s['beta_by_bucket'] = bk
    summary[name] = s
    print(f"\n{name}:")
    print(f"  β         = {s['beta'][0]:.4f} ± {s['beta'][1]:.4f}  (真值 0.3)")
    print(f"  per-k MAE = {s['perk_mae'][0]:.4f} ± {s['perk_mae'][1]:.4f}")
    print(f"  γ         = {s['gamma_mean'][0]:.4f} ± {s['gamma_mean'][1]:.4f}  (真值 0.1)")
    print(f"  γ-度数相关= {s['gamma_corr'][0]:.4f} ± {s['gamma_corr'][1]:.4f}")
    print(f"  轨迹I-MAE = {s['traj_mae'][0]:.4f} ± {s['traj_mae'][1]:.4f}")
    print(f"  分桶β     = { {k: round(v[0], 3) for k, v in bk.items()} }")

save_json(summary, os.path.join(OUT, 'phase1_summary.json'))
save_json(results, os.path.join(OUT, 'phase1_results.json'))

# ---------------- 对比图 ----------------
names = list(MODELS.keys())
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# β误差
ax = axes[0]
beta_err = [abs(summary[n]['beta'][0] - BETA) for n in names]
beta_std = [summary[n]['beta'][1] for n in names]
ax.bar(range(len(names)), beta_err, yerr=beta_std, capsize=4, color='steelblue', alpha=0.8)
ax.axhline(0, color='k', linewidth=0.8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('|β_est − 0.3|'); ax.set_title('β estimation error')

# γ-度数相关
ax = axes[1]
corrs = [summary[n]['gamma_corr'][0] for n in names]
ax.bar(range(len(names)), corrs, color='tomato', alpha=0.8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('corr(γ, degree)'); ax.set_title('γ–degree correlation (true=0)')

# 轨迹MAE
ax = axes[2]
maes = [summary[n]['traj_mae'][0] for n in names]
ax.bar(range(len(names)), maes, color='seagreen', alpha=0.8)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, ha='right')
ax.set_ylabel('MAE of I-fraction'); ax.set_title('Autoregressive trajectory error')

fig.tight_layout()
fig.savefig(os.path.join(OUT, 'comparison_bar.png'), dpi=150)
plt.close(fig)

# 逐k曲线对比（seed0, 画主推模型）
fig, ax = plt.subplots(figsize=(7, 5))
ks = np.arange(1, MAX_K + 1)
ax.plot(ks, [1 - (1 - BETA) ** k for k in ks], 'k-', linewidth=2, label='True β=0.3')
colors = {'baseline_3L': 'gray', 'baseline_1L': 'orange', 'two_channel_A': 'red',
          'two_channel_Amean': 'blue', 'two_channel_B': 'purple'}
for name in names:
    r0 = results[name][0]
    ax.plot(r0['perk']['k'], r0['perk']['pred'], 'o-', color=colors[name],
            label=f"{name} (seed0)", markersize=5, alpha=0.8)
ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('P(S→I)')
ax.set_title('Per-k infection probability (bidirectional edges)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'perk_curves.png'), dpi=150)
plt.close(fig)

# 分桶β对比柱状图
fig, ax = plt.subplots(figsize=(8, 5))
bucket_names = list(summary[names[0]]['beta_by_bucket'].keys())
x = np.arange(len(bucket_names)); w = 0.15
for i, name in enumerate(names):
    vals = [summary[name]['beta_by_bucket'].get(b, (np.nan, 0))[0] for b in bucket_names]
    ax.bar(x + (i - len(names) / 2 + 0.5) * w, vals, w, label=name, alpha=0.85)
ax.axhline(BETA, color='k', linestyle='--', label='True β=0.3')
ax.set_xticks(x); ax.set_xticklabels(bucket_names)
ax.set_xlabel('Degree bucket'); ax.set_ylabel('Fitted β')
ax.set_title('β by degree bucket'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'beta_by_bucket.png'), dpi=150)
plt.close(fig)

print(f"\n结果已保存: {OUT}")
