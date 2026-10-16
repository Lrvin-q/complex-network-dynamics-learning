# -*- coding: utf-8 -*-
"""
phase0_baseline_diagnostics.py —— Phase 0：复现与量化基线
1. 验证“edge_index 方向性 bug”：同配置下 有向(notebook原样) vs 双向 的 β 估计对比
2. 两个模型的全量诊断：逐k误差、按度数分桶 β、(degree×k) 误差热力图、γ-度数相关、自回归MAE
输出：two_channel_experiment/results/phase0/
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_ba_graph, generate_dataset, build_loaders,
                    GraphConvSIR, train_model, collect_si_samples, fit_beta,
                    compute_perk_stats, collect_gamma_samples, trajectory_mae,
                    save_json, plot_k_curve, plot_heatmap, plot_training)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase0')
os.makedirs(OUT, exist_ok=True)

# ---------------- 配置（与notebook主程序一致） ----------------
N, M = 100, 3
BETA, GAMMA = 0.3, 0.1
N_SEEDS, N_REPEATS, T_MAX = 30, 5, 60
EPOCHS, HIDDEN, NUM_LAYERS, DROPOUT, LR, BATCH = 30, 64, 3, 0.2, 0.01, 32
MAIN_SEED = 21
DEGREE_BUCKETS = [(3, 6), (6, 10), (10, 14), (14, 20), (20, 34)]
MAX_K = 10

print("=" * 70)
print("Phase 0: 基线复现与诊断 (BA网络, N=100, β=0.3, γ=0.1)")
print(f"模型: GraphConv x{NUM_LAYERS}层 hidden={HIDDEN} epochs={EPOCHS}")
print("=" * 70)

# ---------------- 图与数据 ----------------
set_seed(MAIN_SEED)
edge_index_bi, edge_index_dir, G = generate_ba_graph(N, M)
print(f"边数: {edge_index_dir.size(1)} (无向), 双向化后 {edge_index_bi.size(1)}")
adj_list = [[] for _ in range(N)]
for u, v in edge_index_dir.t().tolist():
    adj_list[u].append(v); adj_list[v].append(u)

dataset = generate_dataset(edge_index_dir, BETA, GAMMA, N_SEEDS, N_REPEATS, T_MAX)
train_loader, val_loader = build_loaders(dataset, BATCH, val_frac=0.2, seed=MAIN_SEED)
print(f"样本数: {len(dataset)} (train {len(train_loader.dataset)} / val {len(val_loader.dataset)})")

def run_model(name, edge_index, diag=True):
    """训练一个基线模型并做诊断；返回 (model, results_dict)"""
    set_seed(MAIN_SEED)
    model = GraphConvSIR(in_channels=3, hidden_channels=HIDDEN, out_channels=3,
                         num_layers=NUM_LAYERS, dropout=DROPOUT, aggr='sum')
    print(f"\n[{name}] 训练 (参数量 {sum(p.numel() for p in model.parameters())})...")
    tr, va = train_model(model, train_loader, val_loader, edge_index,
                         epochs=EPOCHS, lr=LR, verbose_every=10)
    res = {'name': name, 'train_losses': tr, 'val_losses': va,
           'final_train': tr[-1], 'final_val': va[-1], 'best_val': min(va)}
    plot_training(tr, va, f'{name} training', os.path.join(OUT, f'training_{name}.png'))

    if not diag:
        return model, res

    # --- 诊断1: 逐k P(S->I) ---
    print(f"[{name}] 收集 S→I 样本 (k=1..{MAX_K})...")
    samples_by_k, samples_by_bk, counts_by_k, bucket_of, degrees = collect_si_samples(
        model, edge_index, adj_list, DEGREE_BUCKETS, max_k=MAX_K,
        min_per_k=300, min_per_bk=100, max_iters=8000)
    perk = compute_perk_stats(samples_by_k, BETA)
    fit = fit_beta(perk['k'], [samples_by_k[k] for k in perk['k']])
    res['perk'] = perk
    res['fit_beta_overall'] = fit
    res['counts_by_k'] = counts_by_k
    if fit:
        print(f"[{name}] 逐k拟合 β = {fit[0]:.4f} ± {fit[1]:.4f} (R²={fit[2]:.3f}), 真值 0.3")
        plot_k_curve(perk, fit[0], f'{name}: P(S→I|k)', os.path.join(OUT, f'kcurve_{name}.png'))

    # --- 诊断2: 按度数分桶拟合 β ---
    res['beta_by_bucket'] = {}
    for b, (lo, hi) in enumerate(DEGREE_BUCKETS):
        klist = [k for k in range(1, MAX_K + 1) if len(samples_by_bk.get((b, k), [])) >= 20]
        if len(klist) >= 2:
            f2 = fit_beta(klist, [samples_by_bk[(b, k)] for k in klist])
            if f2:
                res['beta_by_bucket'][f'{lo}-{hi-1}'] = f2[0]

    # --- 诊断3: (degree×k) 误差热力图 ---
    err_mat = np.full((len(DEGREE_BUCKETS), MAX_K), -1.0)
    for b in range(len(DEGREE_BUCKETS)):
        for k in range(1, MAX_K + 1):
            ps = samples_by_bk.get((b, k), [])
            if len(ps) >= 20:
                err_mat[b, k - 1] = abs(np.mean(ps) - (1 - (1 - BETA) ** k))
    res['heatmap'] = err_mat.tolist()
    plot_heatmap(err_mat, DEGREE_BUCKETS, MAX_K, f'{name}: |error| (degree×k)',
                 os.path.join(OUT, f'heatmap_{name}.png'))

    # --- 诊断4: γ 与度数相关性 ---
    print(f"[{name}] 收集 I→R 样本...")
    probs_i, deg_i, corr, pval = collect_gamma_samples(model, edge_index, adj_list, n_states=1500)
    res['gamma'] = {'mean': float(probs_i.mean()), 'std': float(probs_i.std()),
                    'n': int(len(probs_i)), 'corr_degree': float(corr), 'pval': float(pval)}
    # 按度数分桶 γ
    res['gamma_by_bucket'] = {}
    for b, (lo, hi) in enumerate(DEGREE_BUCKETS):
        m = (deg_i >= lo) & (deg_i < hi)
        if m.sum() >= 50:
            res['gamma_by_bucket'][f'{lo}-{hi-1}'] = float(probs_i[m].mean())

    # --- 诊断5: 自回归轨迹 MAE ---
    print(f"[{name}] 自回归评估 (3个初始条件, n_mc=40, T=50)...")
    maes = []
    for n_inf in (1, 2, 3):
        state = np.zeros(N, dtype=int)
        state[np.random.choice(N, n_inf, replace=False)] = 1
        r = trajectory_mae(model, edge_index, state, BETA, GAMMA, T=50, n_mc=40)
        maes.append(r['mae_i'])
    res['trajectory'] = {'mae_i_per_init': maes, 'mae_i_mean': float(np.mean(maes))}
    print(f"[{name}] 轨迹 I-MAE: {res['trajectory']['mae_i_mean']:.4f}")
    return model, res

# ============ 模型1: notebook原样（有向 edge_index） ============
model_asym, res_asym = run_model('baseline_asym', edge_index_dir)

# ============ 模型2: 修正（双向 edge_index） ============
model_bi, res_bi = run_model('baseline_bi', edge_index_bi)

print("\n" + "=" * 70)
print("Phase 0 关键对比")
print("=" * 70)
print(f"  有向(notebook原样):  β≈{res_asym.get('fit_beta_overall', (None,))[0]}, "
      f"γ-度数相关={res_asym['gamma']['corr_degree']:.4f}, "
      f"γ={res_asym['gamma']['mean']:.4f}, 轨迹MAE={res_asym['trajectory']['mae_i_mean']:.4f}")
print(f"  双向(修正):          β≈{res_bi.get('fit_beta_overall', (None,))[0]}, "
      f"γ-度数相关={res_bi['gamma']['corr_degree']:.4f}, "
      f"γ={res_bi['gamma']['mean']:.4f}, 轨迹MAE={res_bi['trajectory']['mae_i_mean']:.4f}")
print(f"  真值: β=0.3, γ=0.1")
print("=" * 70)

save_json({'asym': res_asym, 'bi': res_bi}, os.path.join(OUT, 'phase0_results.json'))
print(f"结果已保存: {OUT}")
