# -*- coding: utf-8 -*-
"""
phase2_twob_three_networks.py —— 三类网络（ER/WS/BA）× 双通道 B 模型的训练与自回归评估
只做【计算 + 保存数据】（DSH.md 原则：绘图由 plot_phase2_figures.py 从 JSON 完成）。
输出：
  results/phase2/twob_{NET}.json   —— 训练损失 + 25 组自回归测试（5种子组 × 5初始感染数）+ 汇总统计
  results/phase2/models/{NET}.pt   —— 模型权重（坑8：训练后保存权重）
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_network, generate_dataset, build_loaders,
                    TwoChannelGCN, train_model, evaluate_autoregressive_full, save_json)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results', 'phase2')
MODEL_DIR = os.path.join(OUT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- 配置 ----------------
N, BETA, GAMMA = 100, 0.3, 0.1
N_SEEDS, N_REPEATS, T_MAX = 30, 5, 60
EPOCHS, HIDDEN, DROPOUT, LR, BATCH = 30, 64, 0.2, 0.01, 32
SEED = 21
NET_PARAMS = {'ER': dict(net_type='ER', p=0.1),
              'WS': dict(net_type='WS', k=6, rewiring_p=0.2),
              'BA': dict(net_type='BA', m=3)}
# 自回归评估（与论文图3-5一致）：5 种子组 × 5 初始感染数
N_GROUPS, N_INF_CHOICES = 5, 5
EVAL_T, EVAL_NMC = 50, 50

CONFIG = {
    'network': {'N': N, 'net_params': {k: dict(v) for k, v in NET_PARAMS.items()}},
    'sir': {'beta': BETA, 'gamma': GAMMA, 'n_seeds': N_SEEDS, 'n_repeats': N_REPEATS, 'T_max': T_MAX},
    'training': {'model': 'TwoChannelGCN', 'hidden': HIDDEN, 'num_layers': 2,
                 'dropout': DROPOUT, 'epochs': EPOCHS, 'lr': LR, 'batch_size': BATCH,
                 'optimizer': 'Adam', 'loss': 'CrossEntropy', 'val_frac': 0.2,
                 'seed': SEED, 'edge_index': 'bidirectional'},
    'autoregressive': {'n_groups': N_GROUPS, 'n_inf_choices': N_INF_CHOICES,
                       'T': EVAL_T, 'n_mc': EVAL_NMC,
                       'group_seed_base': 42, 'group_seed_step': 10},
}

print("=" * 70)
print("Phase 2: 三类网络 × 双通道 B (TwoChannelGCN)")
print(f"β={BETA} γ={GAMMA} N={N}, epochs={EPOCHS}, hidden={HIDDEN}")
print("=" * 70)

results = {'config': CONFIG}

for net_type, params in NET_PARAMS.items():
    print(f"\n########## 网络 {net_type} ##########")
    set_seed(SEED)
    edge_index_bi, edge_index_dir, G = generate_network(N=N, **params)
    n_nodes = edge_index_bi.max().item() + 1
    print(f"节点数={n_nodes}, 边数(无向)={edge_index_dir.size(1)}, 平均度={2*edge_index_dir.size(1)/n_nodes:.2f}")
    adj_list = [[] for _ in range(n_nodes)]
    for u, v in edge_index_dir.t().tolist():
        adj_list[u].append(v); adj_list[v].append(u)

    dataset = generate_dataset(edge_index_dir, BETA, GAMMA, N_SEEDS, N_REPEATS, T_MAX)
    train_loader, val_loader = build_loaders(dataset, BATCH, val_frac=0.2, seed=SEED)
    print(f"样本数: {len(dataset)} (train {len(train_loader.dataset)} / val {len(val_loader.dataset)})")

    model = TwoChannelGCN(in_channels=3, hidden=HIDDEN, out_channels=3, num_layers=2, dropout=DROPOUT)
    print(f"训练 (参数量 {sum(p.numel() for p in model.parameters())})...")
    tr, va = train_model(model, train_loader, val_loader, edge_index_bi,
                         epochs=EPOCHS, lr=LR,
                         save_path=os.path.join(MODEL_DIR, f'{net_type}.pt'),
                         verbose_every=10)
    print(f"训练完成: best_val={min(va):.4f}")

    # ---- 自回归评估：5×5 测试 ----
    print("自回归评估 (5 种子组 × 5 初始感染数, n_mc=50, T=50)...")
    tests = []
    for g in range(N_GROUPS):
        group_seed = 42 + g * 10
        np.random.seed(group_seed); torch.manual_seed(group_seed)
        for n_inf in range(1, N_INF_CHOICES + 1):
            state = np.zeros(n_nodes, dtype=int)
            inf_nodes = np.random.choice(n_nodes, n_inf, replace=False)
            state[inf_nodes] = 1
            t_frac, p_frac = evaluate_autoregressive_full(
                model, edge_index_bi, state, BETA, GAMMA,
                T=EVAL_T, n_mc=EVAL_NMC)
            mae = {c: float(np.mean(np.abs(t_frac[:, i] - p_frac[:, i])))
                   for i, c in enumerate('SIR')}
            rmse = {c: float(np.sqrt(np.mean((t_frac[:, i] - p_frac[:, i]) ** 2)))
                    for i, c in enumerate('SIR')}
            tests.append({'group': g, 'n_inf': n_inf,
                          'true': t_frac.tolist(), 'pred': p_frac.tolist(),
                          'mae': mae, 'rmse': rmse})
        print(f"  组 {g+1}/5 完成")

    # 汇总统计
    mae_i_mean = float(np.mean([t['mae']['I'] for t in tests]))
    rmse_i_mean = float(np.mean([t['rmse']['I'] for t in tests]))
    mae_all_mean = float(np.mean([(t['mae']['S'] + t['mae']['I'] + t['mae']['R']) / 3 for t in tests]))
    print(f"[{net_type}] 轨迹 MAE(I)={mae_i_mean:.4f}, RMSE(I)={rmse_i_mean:.4f}, MAE(全状态)={mae_all_mean:.4f}")

    results[net_type] = {
        'network_stats': {'n_nodes': n_nodes, 'n_edges': int(edge_index_dir.size(1)),
                          'mean_degree': float(2 * edge_index_dir.size(1) / n_nodes)},
        'train_losses': tr, 'val_losses': va,
        'best_val': float(min(va)), 'final_train': tr[-1], 'final_val': va[-1],
        'tests': tests,
        'summary': {'mae_i': mae_i_mean, 'rmse_i': rmse_i_mean, 'mae_all': mae_all_mean},
        'model_weights': os.path.join('models', f'{net_type}.pt'),
    }

save_json(results, os.path.join(OUT, 'twob_three_networks.json'))
print(f"\n全部完成，数据已保存: {OUT}/twob_three_networks.json")
print("（绘图由 plot_phase2_figures.py 从 JSON 完成）")
