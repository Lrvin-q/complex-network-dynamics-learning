# -*- coding: utf-8 -*-
"""
phase3c_ws_probe.py —— WS 自回归偏差的网络结构归因探针
目标：回答"为什么双通道B在 WS 上自回归误差最大，而 ER/BA 较小"
方法（利用 phase3 保存的 twob_bi 权重，无需重训）：
  1. 结构指标：聚集系数、平均最短路径、度方差、三角形数
  2. 真实/模型轨迹：均值 I 曲线 + 真实逐时步标准差（固有随机性），模型偏差 vs 固有标准差
  3. 校准探针：随机上下文 vs 真实轨迹上下文的逐节点一步误差（检验"轨迹上下文的校准退化"）
输出：results/phase3c/ws_probe.json
"""
import os
import sys
import json
import numpy as np
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_network, TwoChannelGCN, predict_probs,
                    sir_simulation, save_json)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
OUT = os.path.join(BASE, 'phase3c')
os.makedirs(OUT, exist_ok=True)
MODEL_DIR = os.path.join(BASE, 'phase3', 'models')

BETA, GAMMA = 0.3, 0.1
T, N_MC = 50, 100
N_INIT = 3
INIT_SEED = 42
CALIB_RUNS = 20      # 用于上下文校准探针的真实轨迹条数
CALIB_STEPS = 50     # 每条轨迹采样的时间步数（全部）

NETS = {
    'ER': dict(net_type='ER', p=0.1),
    'WS': dict(net_type='WS', k=6, rewiring_p=0.2),
    'BA': dict(net_type='BA', m=3),
}


def true_conditional_probs(state, adj_list, beta, gamma):
    """给定当前状态 (N,)，返回每个节点下一状态的精确条件分布 (N,3)（S/I/R 概率）"""
    N = len(state)
    probs = np.zeros((N, 3))
    for i in range(N):
        if state[i] == 0:
            k = sum(1 for nb in adj_list[i] if state[nb] == 1)
            p_inf = 1 - (1 - beta) ** k
            probs[i] = [1 - p_inf, p_inf, 0.0]
        elif state[i] == 1:
            probs[i] = [0.0, 1 - gamma, gamma]
        else:
            probs[i] = [0.0, 0.0, 1.0]
    return probs


def tv_error(model, state, adj_list, edge_index):
    """模型预测分布与真实条件分布的逐节点总变差距离（0~1），取平均"""
    p_model = predict_probs(model, state, edge_index)
    p_true = true_conditional_probs(state, adj_list, BETA, GAMMA)
    return float(np.mean(np.abs(p_model - p_true).sum(axis=1) / 2))


results = {'config': {'beta': BETA, 'gamma': GAMMA, 'T': T, 'n_mc': N_MC,
                      'n_init': N_INIT, 'init_seed': INIT_SEED, 'calib_runs': CALIB_RUNS}}

for net, kw in NETS.items():
    print(f'\n########## {net} ##########')
    set_seed(21)
    e_bi, e_dir, G = generate_network(N=100, **kw)
    n_nodes = e_bi.max().item() + 1
    adj_list = [[] for _ in range(n_nodes)]
    for u, v in e_dir.t().tolist():
        adj_list[u].append(v); adj_list[v].append(u)

    model = TwoChannelGCN(3, 64, 3, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{net}_twob_bi.pt'), map_location='cpu'))
    model.eval()

    # ---- 结构指标 ----
    C = nx.average_clustering(G)
    try:
        apl = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        apl = float('nan')
    deg = np.array([d for _, d in G.degree()])
    struct = {'clustering': float(C), 'avg_path': float(apl),
              'deg_mean': float(deg.mean()), 'deg_var': float(deg.var()),
              'triangles_mean': float(sum(nx.triangles(G).values()) / n_nodes)}
    print('  结构:', {k: round(v, 3) for k, v in struct.items()})

    # ---- 初始状态（固定） ----
    np.random.seed(INIT_SEED)
    init_state = np.zeros(n_nodes, dtype=int)
    init_state[np.random.choice(n_nodes, N_INIT, replace=False)] = 1
    init_inf = np.where(init_state == 1)[0].tolist()

    # ---- 真实轨迹（均值 + 逐时步标准差） ----
    true_all = np.zeros((N_MC, T + 1))
    true_states_collect = []   # 用于校准探针的轨迹状态快照
    for r in range(N_MC):
        st = sir_simulation(e_bi, BETA, GAMMA, init_inf, T)
        true_all[r] = np.mean(st == 1, axis=1)
        if r < CALIB_RUNS:
            true_states_collect.extend(st[:CALIB_STEPS + 1])
    true_mean = true_all.mean(axis=0)
    true_std = true_all.std(axis=0)

    # ---- 模型自回归轨迹 ----
    pred_all = np.zeros((N_MC, T + 1))
    for r in range(N_MC):
        state = init_state.copy()
        pred_all[r, 0] = np.mean(state == 1)
        for t in range(T):
            probs = predict_probs(model, state, e_bi)
            state = np.array([np.random.choice(3, p=probs[i]) for i in range(n_nodes)])
            pred_all[r, t + 1] = np.mean(state == 1)
    pred_mean = pred_all.mean(axis=0)

    mae_i = float(np.mean(np.abs(pred_mean - true_mean)))
    bias_over_std = float(np.mean(np.abs(pred_mean - true_mean) / true_std.clip(min=1e-3)))
    peak_true = float(true_mean.max())
    peak_pred = float(pred_mean.max())

    # ---- 校准探针 ----
    tv_rollout = []       # 真实轨迹上下文的一步 TV 误差
    tv_random = []        # 随机上下文（均匀比例）的一步 TV 误差
    tv_rollout_by_k = {k: [] for k in range(0, 11)}
    for st in true_states_collect:
        tv_rollout.append(tv_error(model, np.asarray(st), adj_list, e_bi))
        # 按 k 分解（S 节点的 P(I) 误差）
        p_model = predict_probs(model, np.asarray(st), e_bi)
        for node in np.where(np.asarray(st) == 0)[0]:
            k = sum(1 for nb in adj_list[node] if st[nb] == 1)
            if k <= 10:
                tv_rollout_by_k[k].append(abs(p_model[node, 1] - (1 - (1 - BETA) ** k)))
    for _ in range(300):
        ratio = float(np.random.uniform(0.1, 0.7))
        st = np.zeros(n_nodes, dtype=int)
        st[np.random.choice(n_nodes, max(1, int(n_nodes * ratio)), replace=False)] = 1
        tv_random.append(tv_error(model, st, adj_list, e_bi))

    rec = {
        'structure': struct,
        'true_mean': true_mean.tolist(), 'true_std': true_std.tolist(),
        'pred_mean': pred_mean.tolist(),
        'mae_i': mae_i, 'bias_over_std': bias_over_std,
        'peak_true': peak_true, 'peak_pred': peak_pred,
        'tv_rollout_mean': float(np.mean(tv_rollout)),
        'tv_random_mean': float(np.mean(tv_random)),
        'tv_rollout_by_k_mean': {k: (float(np.mean(v)) if v else None)
                                 for k, v in tv_rollout_by_k.items()},
        'tv_rollout_n': len(tv_rollout), 'tv_random_n': len(tv_random),
    }
    print(f'  MAE(I)={mae_i:.4f} | 真实std均值={true_std.mean():.4f} | 偏差/真实std={bias_over_std:.3f} | '
          f'峰值 真实={peak_true:.3f} vs 预测={peak_pred:.3f}')
    print(f'  一步TV误差: 轨迹上下文={rec["tv_rollout_mean"]:.4f} vs 随机上下文={rec["tv_random_mean"]:.4f}')
    results[net] = rec

save_json(results, os.path.join(OUT, 'ws_probe.json'))
print(f'\n探针数据已保存: {OUT}/ws_probe.json')
