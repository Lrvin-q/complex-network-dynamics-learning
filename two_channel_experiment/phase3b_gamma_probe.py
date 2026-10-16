# -*- coding: utf-8 -*-
"""
phase3b_gamma_probe.py —— γ-度数相关探针：拆解 Pearson r 与分桶均值不一致的原因
用 phase3 保存的 twob_bi 权重（无需重训）在 ER/BA 上重采样 I→R，
保存逐节点 (degree, prob, k_I) 数据，供 plot_gamma_probe.py 分析。
输出：results/phase3b/gamma_probe_{NET}.json
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_network, TwoChannelGCN, predict_probs, save_json)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(BASE, 'phase3b')
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = os.path.join(BASE, 'phase3', 'models')

NETS = {
    'ER': dict(net_type='ER', p=0.1),
    'BA': dict(net_type='BA', m=3),
}
N_STATES = 1200
SEED = 21

for net, kw in NETS.items():
    set_seed(SEED)
    e_bi, e_dir, G = generate_network(N=100, **kw)
    n_nodes = e_bi.max().item() + 1
    adj_list = [[] for _ in range(n_nodes)]
    for u, v in e_dir.t().tolist():
        adj_list[u].append(v); adj_list[v].append(u)
    degrees = np.array([len(a) for a in adj_list])

    model = TwoChannelGCN(3, 64, 3, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{net}_twob_bi.pt'), map_location='cpu'))
    model.eval()

    recs = []
    with torch.no_grad():
        for _ in range(N_STATES):
            ratio = float(np.random.uniform(0.1, 0.6))
            n_inf = max(1, int(n_nodes * ratio))
            state = np.zeros(n_nodes, dtype=int)
            state[np.random.choice(n_nodes, n_inf, replace=False)] = 1
            state[state == 0] = 2
            probs = predict_probs(model, state, e_bi)
            for node in np.where(state == 1)[0]:
                k_i = sum(1 for nb in adj_list[node] if state[nb] == 1)
                recs.append((int(degrees[node]), float(probs[node, 2]), int(k_i)))

    deg = np.array([r[0] for r in recs]); prob = np.array([r[1] for r in recs])
    k_i = np.array([r[2] for r in recs])
    print(f'[{net}] n={len(recs)}, P(I→R) 均值={prob.mean():.4f}, 标准差={prob.std():.4f}')
    print(f'  pearson(degree, prob) = {np.corrcoef(deg, prob)[0,1]:.4f}')
    print(f'  pearson(k_I, prob)    = {np.corrcoef(k_i, prob)[0,1]:.4f}')

    # 逐度均值
    per_degree = {}
    for d in sorted(set(deg.tolist())):
        m = deg == d
        per_degree[str(d)] = {'count': int(m.sum()), 'mean': float(prob[m].mean()),
                              'std': float(prob[m].std())}
    # 桶内回归斜率（每桶内 prob ~ degree 的斜率）
    buckets = [(3, 8), (8, 11), (11, 14), (14, 17)] if net == 'ER' else [(3, 6), (6, 10), (10, 14), (14, 34)]
    bucket_stats = []
    for lo, hi in buckets:
        m = (deg >= lo) & (deg < hi)
        if m.sum() < 50:
            continue
        slope = float(np.polyfit(deg[m], prob[m], 1)[0]) if np.unique(deg[m]).size > 1 else 0.0
        bucket_stats.append({'bucket': f'{lo}-{hi-1}', 'n': int(m.sum()),
                             'mean': float(prob[m].mean()), 'slope_per_deg': slope})
    # 全范围回归斜率
    slope_all = float(np.polyfit(deg, prob, 1)[0])
    # 局部 k_I 回归
    slope_ki = float(np.polyfit(k_i, prob, 1)[0])
    # 多元回归 prob ~ degree + k_I
    A = np.stack([np.ones_like(deg), deg, k_i], axis=1)
    coef, *_ = np.linalg.lstsq(A, prob, rcond=None)

    save_json({'n': len(recs), 'gamma_mean': float(prob.mean()), 'gamma_std': float(prob.std()),
               'pearson_degree': float(np.corrcoef(deg, prob)[0, 1]),
               'pearson_kI': float(np.corrcoef(k_i, prob)[0, 1]),
               'slope_all_per_deg': slope_all, 'slope_kI_per_neighbor': slope_ki,
               'regress_intercept': float(coef[0]), 'regress_degree': float(coef[1]),
               'regress_kI': float(coef[2]),
               'per_degree': per_degree, 'bucket_stats': bucket_stats},
              os.path.join(OUT, f'gamma_probe_{net}.json'))
    print(f'  全范围斜率(每度) = {slope_all:.6f}, 局部k_I斜率 = {slope_ki:.6f}')
    print(f'  多元回归: 截距={coef[0]:.5f}, degree系数={coef[1]:.6f}, k_I系数={coef[2]:.6f}')
    print(f'  桶统计: {[(b["bucket"], round(b["mean"],4), round(b["slope_per_deg"],5)) for b in bucket_stats]}')

print('\n探针数据已保存: results/phase3b/')
