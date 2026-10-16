# -*- coding: utf-8 -*-
"""
phase3_eval.py —— 三网络 × 三模型完整评估（参数逆向求解 + 自回归）
模型：
  gcn_dir : 3层 GraphConv sum + 有向 edge_index（最原始实现，含方向bug）
  gcn_bi  : 3层 GraphConv sum + 双向 edge_index（仅修方向）
  twob_bi : 双通道B + 双向 edge_index（最终方案）
每模型评估：
  (1) 参数逆向求解：逐k P(S→I|k) 曲线与误差、总体β拟合、按度数分桶β、γ辨识（均值/度数相关/分桶）
  (2) 自回归 5×5 测试（与论文图3-5同协议）
只做计算+保存数据（DSH.md 原则），绘图由 phase3_plot.py 完成。
输出：results/phase3/phase3_eval.json + results/phase3/models/{NET}_{MODEL}.pt
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_network, generate_dataset, build_loaders,
                    GraphConvSIR, TwoChannelGCN, train_model,
                    collect_si_samples, fit_beta, compute_perk_stats,
                    collect_gamma_samples, evaluate_autoregressive_full, save_json)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results', 'phase3')
MODEL_DIR = os.path.join(OUT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- 配置 ----------------
N, BETA, GAMMA = 100, 0.3, 0.1
N_SEEDS, N_REPEATS, T_MAX = 30, 5, 60
EPOCHS, HIDDEN, DROPOUT, LR, BATCH = 30, 64, 0.2, 0.01, 32
SEED = 21
MAX_K = 10
MIN_PER_K, MIN_PER_BK, MAX_ITERS = 250, 80, 6000
GAMMA_N_STATES = 1200
HEATMAP_MIN = 20
N_GROUPS, N_INF_CHOICES = 5, 5
EVAL_T, EVAL_NMC = 50, 50

# 每网络的度数分桶（依据各自度数分布）
NETS = {
    'ER': dict(net_type='ER', p=0.1,
               buckets=[(3, 8), (8, 11), (11, 14), (14, 17)]),
    'WS': dict(net_type='WS', k=6, rewiring_p=0.2,
               buckets=[(4, 6), (6, 8), (8, 10)]),
    'BA': dict(net_type='BA', m=3,
               buckets=[(3, 6), (6, 10), (10, 14), (14, 34)]),
}
MODELS = {
    'gcn_dir': dict(kind='gcn', edge='directed'),
    'gcn_bi': dict(kind='gcn', edge='bidirectional'),
    'twob_bi': dict(kind='twob', edge='bidirectional'),
}

CONFIG = {
    'network': {'N': N, 'net_params': {k: dict(v) for k, v in NETS.items()}},
    'sir': {'beta': BETA, 'gamma': GAMMA, 'n_seeds': N_SEEDS, 'n_repeats': N_REPEATS, 'T_max': T_MAX},
    'training': {'epochs': EPOCHS, 'hidden': HIDDEN, 'dropout': DROPOUT, 'lr': LR,
                 'batch_size': BATCH, 'optimizer': 'Adam', 'loss': 'CrossEntropy',
                 'val_frac': 0.2, 'seed': SEED},
    'models': MODELS,
    'diagnostics': {'max_k': MAX_K, 'min_per_k': MIN_PER_K, 'min_per_bk': MIN_PER_BK,
                    'max_iters': MAX_ITERS, 'gamma_n_states': GAMMA_N_STATES,
                    'heatmap_min_samples': HEATMAP_MIN},
    'autoregressive': {'n_groups': N_GROUPS, 'n_inf_choices': N_INF_CHOICES,
                       'T': EVAL_T, 'n_mc': EVAL_NMC,
                       'group_seed_base': 42, 'group_seed_step': 10},
}

print("=" * 70)
print("Phase 3: 三网络 × 三模型 完整评估（参数逆向求解 + 自回归）")
print("=" * 70)

results = {'config': CONFIG}

for net_type, params in NETS.items():
    print(f"\n########## 网络 {net_type} ##########")
    set_seed(SEED)  # 每个网络独立重置，保证图与 phase0/1/2 一致
    e_bi, e_dir, G = generate_network(N=N, net_type=params['net_type'],
                                      p=params.get('p'), m=params.get('m'),
                                      k=params.get('k'), rewiring_p=params.get('rewiring_p'))
    n_nodes = e_bi.max().item() + 1
    adj_list = [[] for _ in range(n_nodes)]
    for u, v in e_dir.t().tolist():
        adj_list[u].append(v); adj_list[v].append(u)
    deg = np.array([len(a) for a in adj_list])
    print(f"节点={n_nodes}, 边={e_dir.size(1)}, 度数 {deg.min()}-{deg.max()} (均值 {deg.mean():.2f})")

    dataset = generate_dataset(e_dir, BETA, GAMMA, N_SEEDS, N_REPEATS, T_MAX)
    train_loader, val_loader = build_loaders(dataset, BATCH, val_frac=0.2, seed=SEED)

    buckets = params['buckets']
    bucket_labels = [f'{lo}-{hi-1}' for lo, hi in buckets]
    print(f"度数分桶: {bucket_labels}")

    results.setdefault(net_type, {'buckets': buckets, 'bucket_labels': bucket_labels,
                                  'network_stats': {'n_nodes': n_nodes,
                                                    'n_edges': int(e_dir.size(1)),
                                                    'degree_range': [int(deg.min()), int(deg.max())],
                                                    'mean_degree': float(deg.mean())},
                                  'models': {}})

    for model_name, mcfg in MODELS.items():
        print(f"\n--- {net_type} / {model_name} ---")
        set_seed(SEED)
        if mcfg['kind'] == 'gcn':
            model = GraphConvSIR(3, HIDDEN, 3, num_layers=3, dropout=DROPOUT, aggr='sum')
        else:
            model = TwoChannelGCN(3, HIDDEN, 3, num_layers=2, dropout=DROPOUT)
        edge_index = e_dir if mcfg['edge'] == 'directed' else e_bi

        tr, va = train_model(model, train_loader, val_loader, edge_index,
                             epochs=EPOCHS, lr=LR,
                             save_path=os.path.join(MODEL_DIR, f'{net_type}_{model_name}.pt'),
                             verbose_every=15)
        print(f"  训练完成: best_val={min(va):.4f}")

        # ---- 参数逆向求解：S→I ----
        samples_by_k, samples_by_bk, counts_by_k, _, degrees = collect_si_samples(
            model, edge_index, adj_list, buckets, max_k=MAX_K,
            min_per_k=MIN_PER_K, min_per_bk=MIN_PER_BK, max_iters=MAX_ITERS, verbose=False)
        perk = compute_perk_stats(samples_by_k, BETA)
        fit = fit_beta(perk['k'], [samples_by_k[k] for k in perk['k']])
        if fit is None:
            fit = [float('nan'), float('nan'), float('nan')]
        beta_by_bucket, beta_missing = {}, {}
        for b, (lo, hi) in enumerate(buckets):
            klist = [k for k in range(1, MAX_K + 1) if len(samples_by_bk.get((b, k), [])) >= HEATMAP_MIN]
            f2 = fit_beta(klist, [samples_by_bk[(b, k)] for k in klist]) if len(klist) >= 2 else None
            if f2:
                beta_by_bucket[bucket_labels[b]] = f2[0]
            else:
                beta_missing[bucket_labels[b]] = f'no fit (valid k count {len(klist)} < 2)'
        # 误差热力图矩阵（(桶, k) 的 |pred - truth|）
        hm = np.full((len(buckets), MAX_K), -1.0)
        for b in range(len(buckets)):
            for k in range(1, MAX_K + 1):
                ps = samples_by_bk.get((b, k), [])
                if len(ps) >= HEATMAP_MIN:
                    hm[b, k - 1] = abs(float(np.mean(ps)) - (1 - (1 - BETA) ** k))
        # (桶, k) 样本数矩阵（用于绘图标注）
        cnt = np.zeros((len(buckets), MAX_K), dtype=int)
        for b in range(len(buckets)):
            for k in range(1, MAX_K + 1):
                cnt[b, k - 1] = len(samples_by_bk.get((b, k), []))

        # ---- 参数逆向求解：I→R ----
        probs_i, deg_i, corr, pval = collect_gamma_samples(
            model, edge_index, adj_list, n_states=GAMMA_N_STATES, verbose=False)
        gamma_by_bucket = {}
        for b, (lo, hi) in enumerate(buckets):
            m = (deg_i >= lo) & (deg_i < hi)
            if m.sum() >= 50:
                gamma_by_bucket[bucket_labels[b]] = float(probs_i[m].mean())

        # ---- 自回归 5×5 ----
        tests = []
        for g in range(N_GROUPS):
            group_seed = 42 + g * 10
            np.random.seed(group_seed); torch.manual_seed(group_seed)
            for n_inf in range(1, N_INF_CHOICES + 1):
                state = np.zeros(n_nodes, dtype=int)
                state[np.random.choice(n_nodes, n_inf, replace=False)] = 1
                t_frac, p_frac = evaluate_autoregressive_full(
                    model, edge_index, state, BETA, GAMMA, T=EVAL_T, n_mc=EVAL_NMC)
                mae = {c: float(np.mean(np.abs(t_frac[:, i] - p_frac[:, i]))) for i, c in enumerate('SIR')}
                rmse = {c: float(np.sqrt(np.mean((t_frac[:, i] - p_frac[:, i]) ** 2))) for i, c in enumerate('SIR')}
                tests.append({'group': g, 'n_inf': n_inf, 'true': t_frac.tolist(),
                              'pred': p_frac.tolist(), 'mae': mae, 'rmse': rmse})

        rec = {
            'train_losses': tr, 'val_losses': va, 'best_val': float(min(va)),
            'perk': perk, 'beta_fit': fit, 'perk_mae': float(np.mean(perk['err'])) if perk['err'] else None,
            'beta_by_bucket': beta_by_bucket, 'beta_by_bucket_missing': beta_missing,
            'heatmap': hm.tolist(), 'heatmap_counts': cnt.tolist(), 'counts_by_k': counts_by_k,
            'gamma': {'mean': float(probs_i.mean()), 'std': float(probs_i.std()),
                      'corr_degree': float(corr), 'pval': float(pval), 'n': int(len(probs_i))},
            'gamma_by_bucket': gamma_by_bucket,
            'autoreg': {'tests': tests,
                        'mae_i_mean': float(np.mean([t['mae']['I'] for t in tests])),
                        'rmse_i_mean': float(np.mean([t['rmse']['I'] for t in tests])),
                        'mae_all_mean': float(np.mean([(t['mae']['S'] + t['mae']['I'] + t['mae']['R']) / 3 for t in tests]))},
            'model_weights': os.path.join('models', f'{net_type}_{model_name}.pt'),
        }
        results[net_type]['models'][model_name] = rec
        print(f"  β拟合={rec['beta_fit'][0]:.4f}±{rec['beta_fit'][1]:.4f} (per-k MAE={rec['perk_mae']:.4f}) | "
              f"γ={rec['gamma']['mean']:.4f} (相关{rec['gamma']['corr_degree']:.3f}) | "
              f"自回归 MAE(I)={rec['autoreg']['mae_i_mean']:.4f}")

save_json(results, os.path.join(OUT, 'phase3_eval.json'))
print(f"\n全部完成，数据已保存: {OUT}/phase3_eval.json")
