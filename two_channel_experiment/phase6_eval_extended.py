# -*- coding: utf-8 -*-
"""
Phase 6 扩展评估：与第一阶段（已知结构）协议完全一致的参数逆向 + 自回归测试。
加载已训练的 dgsl_basic_{NET}.json 权重，用【重构硬图（0.5 阈值）】作为模型输入图：

1. 参数逆向 S→I：collect_si_samples → per-k 曲线 + 加权拟合 β̂ + 分度数桶 β̂（第一阶段同款）；
2. 参数逆向 I→R：collect_gamma_samples → γ̂ 均值 + 与度数相关 + 分桶 γ̂；
3. 自回归 5×5：5 组初始种子（seed=42+g*10）× 5 种初始感染人数（1~5），
   真实轨迹用真实图+真实参数模拟，模型轨迹用重构硬图自回归（T=50, n_mc=50）。

结果写入 results/phase6/dgsl_basic_{NET}.json 的 eval_extended 字段。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    collect_si_samples, fit_beta, compute_perk_stats, collect_gamma_samples,
    save_json, sir_simulation,
)
from dgsl_common import GGNModel  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
DATA_DIR = RESULT_DIR

# ---- 与 phase4 一致的诊断配置 ----
MAX_K, MIN_PER_K, MIN_PER_BK, MAX_ITERS = 10, 250, 80, 6000
GAMMA_N_STATES = 1200
HEATMAP_MIN = 20
N_GROUPS, N_INF_CHOICES = 5, 5
EVAL_T, EVAL_NMC = 50, 50
BETA, GAMMA = 0.3, 0.1
BUCKETS = {
    'ER': [(3, 8), (8, 11), (11, 14), (14, 17)],
    'WS': [(4, 6), (6, 8), (8, 10)],
    'BA': [(3, 6), (6, 10), (10, 14), (14, 34)],
}


class DGSLAdapter:
    """把 DGSL 模型包装成第一阶段接口：__call__(x_flat(N,3), edge_index) -> logits(N,3)。
    图固定为重构硬图 A_hard（0/1 对称矩阵）。"""

    def __init__(self, model, A_hard, device='cpu'):
        self.model = model
        self.A = torch.tensor(np.asarray(A_hard, dtype=np.float32), device=device)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def __call__(self, x, edge_index=None):
        xt = x.unsqueeze(0) if x.dim() == 2 else x
        with torch.no_grad():
            logits = self.model.learner(xt, self.A)
        return logits.squeeze(0)


def hard_edge_index(A_hard):
    """硬图 → edge_index(2,E)（每条无向边一次）+ adj_list（去重双向）。"""
    n = A_hard.shape[0]
    iu = np.triu_indices(n, 1)
    pairs = [(i, j) for i, j in zip(iu[0], iu[1]) if A_hard[i, j] > 0.5]
    if pairs:
        edges = np.array(pairs).T
    else:
        edges = np.zeros((2, 0), dtype=int)
    edge_index = torch.tensor(edges, dtype=torch.long)
    adj_list = [[] for _ in range(n)]
    for u, v in pairs:
        adj_list[u].append(v)
        adj_list[v].append(u)
    return edge_index, adj_list


def autoreg_5x5(adapter, A_hard, true_adj, true_edge_index, n_nodes):
    """5 组种子 × 5 初始感染人数；真实=真实图模拟，模型=硬图自回归。"""
    tests = []
    for g in range(N_GROUPS):
        group_seed = 42 + g * 10
        np.random.seed(group_seed)
        torch.manual_seed(group_seed)
        for n_inf in range(1, N_INF_CHOICES + 1):
            state = np.zeros(n_nodes, dtype=int)
            state[np.random.choice(n_nodes, n_inf, replace=False)] = 1
            # 真实轨迹（真实图 + 真实参数）
            true_frac = np.zeros((EVAL_T + 1, 3))
            for _ in range(EVAL_NMC):
                st = sir_simulation(true_edge_index, BETA, GAMMA, np.where(state == 1)[0].tolist(), EVAL_T)
                for t in range(EVAL_T + 1):
                    true_frac[t, 0] += np.mean(st[t] == 0) / EVAL_NMC
                    true_frac[t, 1] += np.mean(st[t] == 1) / EVAL_NMC
                    true_frac[t, 2] += np.mean(st[t] == 2) / EVAL_NMC
            # 模型轨迹（重构硬图）
            pred_frac = np.zeros((EVAL_T + 1, 3))
            for _ in range(EVAL_NMC):
                s = state.copy()
                for t in range(EVAL_T + 1):
                    pred_frac[t, 0] += np.mean(s == 0) / EVAL_NMC
                    pred_frac[t, 1] += np.mean(s == 1) / EVAL_NMC
                    pred_frac[t, 2] += np.mean(s == 2) / EVAL_NMC
                    if t < EVAL_T:
                        x = np.eye(3)[s].astype(np.float32)
                        xt = torch.tensor(x).unsqueeze(0)
                        logits = adapter(xt, None)
                        probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
                        s = np.array([np.random.choice(3, p=probs[i]) for i in range(n_nodes)])
            mae = {c: float(np.mean(np.abs(true_frac[:, i] - pred_frac[:, i]))) for i, c in enumerate('SIR')}
            rmse = {c: float(np.sqrt(np.mean((true_frac[:, i] - pred_frac[:, i]) ** 2))) for i, c in enumerate('SIR')}
            tests.append({'group': g, 'n_inf': n_inf, 'true': true_frac.tolist(),
                          'pred': pred_frac.tolist(), 'mae': mae, 'rmse': rmse})
    return {
        'tests': tests,
        'mae_i_mean': float(np.mean([t['mae']['I'] for t in tests])),
        'rmse_i_mean': float(np.mean([t['rmse']['I'] for t in tests])),
        'mae_all_mean': float(np.mean([(t['mae']['S'] + t['mae']['I'] + t['mae']['R']) / 3 for t in tests])),
    }


def eval_net(net, prefix, overwrite=False):
    """对指定网络与文件前缀执行扩展评估（β/γ 逆向 + 5×5 自回归）。"""
    path = os.path.join(RESULT_DIR, f'{prefix}_{net}.json')
    d = json.load(open(path, encoding='utf-8'))
    if 'eval_extended' in d and not overwrite:
        print(f'[{net}] 已有 eval_extended，跳过', flush=True)
        return
    cfg = d['config']
    n_nodes = int(cfg['data_meta']['n_nodes'])
    true_adj = np.array(d['matrices']['true_adj'])
    hard = np.array(d['matrices']['hard_adj'])
    np.fill_diagonal(hard, 0)

    model = GGNModel(n_nodes, learner=cfg['learner'], hidden=cfg['hidden'],
                     tau_init=cfg.get('tau_init', 3.0), tau_min=cfg.get('tau_min', 0.1),
                     eps=cfg.get('eps', 1.0),
                     anneal_mode=cfg.get('anneal_mode', 'exp'))
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, d['weights']['model']),
                                     map_location='cpu'))
    model.eval()

    adapter = DGSLAdapter(model, hard)
    edge_index_hard, adj_list_hard = hard_edge_index(hard)
    # 真实图 edge_index（模拟真实轨迹用）
    iu = np.triu_indices(n_nodes, 1)
    pairs = [(i, j) for i, j in zip(iu[0], iu[1]) if true_adj[i, j] > 0.5]
    true_e = torch.tensor(np.array(pairs).T, dtype=torch.long) if pairs else torch.zeros(2, 0, dtype=torch.long)

    buckets = BUCKETS[net]
    bucket_labels = [f'{lo}-{hi-1}' for lo, hi in buckets]

    # ---- S→I 参数逆向（第一阶段同款）----
    samples_by_k, samples_by_bk, counts_by_k, _, degrees = collect_si_samples(
        adapter, edge_index_hard, adj_list_hard, buckets, max_k=MAX_K,
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
            beta_missing[bucket_labels[b]] = 'no fit'

    # ---- I→R 参数逆向 ----
    probs_i, deg_i, corr, pval = collect_gamma_samples(adapter, edge_index_hard, adj_list_hard,
                                                       n_states=GAMMA_N_STATES, verbose=False)
    gamma_by_bucket = {}
    for b, (lo, hi) in enumerate(buckets):
        m = (deg_i >= lo) & (deg_i < hi)
        if m.sum() >= 50:
            gamma_by_bucket[bucket_labels[b]] = float(probs_i[m].mean())
    delta_gamma = float(max(gamma_by_bucket.values()) - min(gamma_by_bucket.values())) if gamma_by_bucket else None

    # ---- 5×5 自回归 ----
    autoreg = autoreg_5x5(adapter, hard, true_adj, true_e, n_nodes)

    d['eval_extended'] = {
        'config': {'max_k': MAX_K, 'min_per_k': MIN_PER_K, 'min_per_bk': MIN_PER_BK,
                   'max_iters': MAX_ITERS, 'gamma_n_states': GAMMA_N_STATES,
                   'n_groups': N_GROUPS, 'n_inf_choices': N_INF_CHOICES,
                   'T': EVAL_T, 'n_mc': EVAL_NMC, 'buckets': buckets,
                   'note': '模型输入图为重构硬图(0.5阈值)；真实轨迹用真实图+真实参数'},
        'beta': {'fit': fit, 'perk': perk, 'perk_mae': float(np.mean(perk['err'])) if perk['err'] else None,
                 'beta_by_bucket': beta_by_bucket, 'beta_by_bucket_missing': beta_missing,
                 'counts_by_k': counts_by_k},
        'gamma': {'mean': float(probs_i.mean()), 'std': float(probs_i.std()),
                  'corr_degree': float(corr), 'pval': float(pval), 'n': int(len(probs_i)),
                  'gamma_by_bucket': gamma_by_bucket, 'delta_gamma': delta_gamma},
        'autoreg_5x5': autoreg,
        'hard_graph': {'n_edges': int(edge_index_hard.size(1)),
                       'n_edges_true': int(true_e.size(1))},
    }
    save_json(d, path)
    print(f'[{net}] beta_hat={fit[0]:.4f} (per-k MAE={d["eval_extended"]["beta"]["perk_mae"]:.4f}) '
          f'gamma_hat={probs_i.mean():.4f}+-{probs_i.std():.4f} delta_gamma={delta_gamma} '
          f'corr_deg={corr:.3f} | 5x5 autoreg MAE(I)={autoreg["mae_i_mean"]:.4f} '
          f'| hard edges {edge_index_hard.size(1)}/{true_e.size(1)}', flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', default='dgsl_basic', help='结果文件前缀')
    ap.add_argument('--nets', nargs='+', default=['ER', 'WS', 'BA'])
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    for net in args.nets:
        eval_net(net, args.prefix, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
