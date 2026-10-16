# -*- coding: utf-8 -*-
"""
rerun_autoreg_fixed.py —— 用修复后的 sir_simulation 重算 phase2/phase3 的自回归评估
背景（DSH.md 坑10）：旧版 sir_simulation 对双向 edge_index 重复计数邻居，
导致自回归评估的"真实"曲线被系统性抬高。本脚本加载已保存的模型权重，
用修复后的模拟器重新生成真实轨迹，更新 JSON 中的 autoreg 字段（不重训）。
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, generate_network, GraphConvSIR, TwoChannelGCN,
                    evaluate_autoregressive_full, save_json)

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
BETA, GAMMA = 0.3, 0.1
N_GROUPS, N_INF_CHOICES, EVAL_T, EVAL_NMC = 5, 5, 50, 50

NETS = {
    'ER': dict(net_type='ER', p=0.1),
    'WS': dict(net_type='WS', k=6, rewiring_p=0.2),
    'BA': dict(net_type='BA', m=3),
}


def make_model(kind):
    if kind == 'gcn':
        return GraphConvSIR(3, 64, 3, num_layers=3, dropout=0.2, aggr='sum')
    return TwoChannelGCN(3, 64, 3, num_layers=2, dropout=0.2)


def run_autoreg(model, edge_index, n_nodes, seed):
    """5×5 自回归测试，返回 tests 列表 + 汇总"""
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
    return {
        'tests': tests,
        'mae_i_mean': float(np.mean([t['mae']['I'] for t in tests])),
        'rmse_i_mean': float(np.mean([t['rmse']['I'] for t in tests])),
        'mae_all_mean': float(np.mean([(t['mae']['S'] + t['mae']['I'] + t['mae']['R']) / 3 for t in tests])),
    }


# ---------------- Phase 3：9 个 (网络, 模型) ----------------
p3_path = os.path.join(BASE, 'phase3', 'phase3_eval.json')
r3 = json.load(open(p3_path, encoding='utf-8'))
print('===== Phase 3 重算自回归（模型权重已保存） =====')
for net, kw in NETS.items():
    set_seed(21)
    e_bi, e_dir, G = generate_network(N=100, **kw)
    n_nodes = e_bi.max().item() + 1
    for model_name, mcfg in r3['config']['models'].items():
        kind = mcfg['kind']
        edge = mcfg['edge']
        ei = e_dir if edge == 'directed' else e_bi
        model = make_model(kind)
        wpath = os.path.join(BASE, 'phase3', 'models', f'{net}_{model_name}.pt')
        model.load_state_dict(torch.load(wpath, map_location='cpu'))
        model.eval()
        rec = run_autoreg(model, ei, n_nodes, seed=21)
        old = r3[net]['models'][model_name]['autoreg']
        print(f'  [{net}/{model_name}] MAE(I): {old["mae_i_mean"]:.4f} -> {rec["mae_i_mean"]:.4f} '
              f'(rmse {rec["rmse_i_mean"]:.4f})')
        r3[net]['models'][model_name]['autoreg'] = rec
r3['config']['autoreg'] = {'note': 'recomputed with fixed sir_simulation (DSH.md 坑10), 2026-08-16',
                           'T': EVAL_T, 'n_mc': EVAL_NMC, 'n_groups': N_GROUPS,
                           'n_inf_choices': N_INF_CHOICES}
save_json(r3, p3_path)
print(f'updated {p3_path}')

# ---------------- Phase 2：3 个 twob_bi ----------------
p2_path = os.path.join(BASE, 'phase2', 'twob_three_networks.json')
r2 = json.load(open(p2_path, encoding='utf-8'))
print('\n===== Phase 2 重算自回归（twob_bi 权重已保存） =====')
for net, kw in NETS.items():
    set_seed(21)
    e_bi, e_dir, G = generate_network(N=100, **kw)
    n_nodes = e_bi.max().item() + 1
    model = TwoChannelGCN(3, 64, 3, num_layers=2, dropout=0.2)
    model.load_state_dict(torch.load(os.path.join(BASE, 'phase2', 'models', f'{net}.pt'), map_location='cpu'))
    model.eval()
    rec = run_autoreg(model, e_bi, n_nodes, seed=21)
    old = r2[net]['summary']
    print(f'  [{net}] MAE(I): {old["mae_i"]:.4f} -> {rec["mae_i_mean"]:.4f}')
    r2[net]['tests'] = rec['tests']
    r2[net]['summary'] = {'mae_i': rec['mae_i_mean'], 'rmse_i': rec['rmse_i_mean'],
                          'mae_all': rec['mae_all_mean']}
r2['config']['autoreg'] = {'note': 'recomputed with fixed sir_simulation (DSH.md 坑10), 2026-08-16'}
save_json(r2, p2_path)
print(f'updated {p2_path}')
print('\n完成。')
