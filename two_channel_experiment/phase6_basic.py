# -*- coding: utf-8 -*-
"""
Phase 6 —— 基础版 v4（按用户四点改进 + GGN 论文原版机制）：

1. 硬图学习器：每 epoch 采样一张硬图（0.5 阈值，= 二分类 Gumbel-argmax），
   学习器阶段固定用硬图训练（论文 train_bn.py：adj = sample(hard=True)）；
   生成器阶段用软图（带梯度）训练生成器；评估以 0.5 阈值硬图为主指标。
2. 降温策略（论文参数 + 显式终点）：τ 从 10（论文初始值）按总生成器步数
   指数衰减至 0.1：τ(t) = 10·(0.01)^(t/T)（论文为 10×0.9999/批，同属指数族）。
3. 动力学学习器统一用双通道（TwoChannel，承接任务一，软邻接=硬图 0/1 求和聚合）。
4. 正则：仅熵正则 λ_ent=0.005（促进二值化）；不用稀疏/度等"人为"正则。
   无最终精调阶段（交替训练中学习器已在硬图上训练）。

用法：
    python -u phase6_basic.py --nets ER WS BA --threads 4
    python -u phase6_basic.py --smoke
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgsl_common import (  # noqa: E402
    GGNModel, generate_and_save_data, build_loaders_from_arrays, train_ggn,
    soft_adj_metrics, dynamics_metrics, autoregressive_mae, per_k_beta_fit,
    save_json, DEVICE,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')

CONFIG = {
    'experiment': 'phase6_basic_v4',
    'method': ('GGN 基础版：论文式交替 + 硬图学习器(0.5阈值) + 双通道学习器 '
               '+ 指数降温 3->0.1 + 仅熵正则'),
    'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2,
    'beta_true': 0.3, 'gamma_true': 0.1, 'T': 60, 'n_seeds': 30, 'n_repeats': 5,
    'epochs': 100, 'S_d': 40, 'S_n': 20, 'batch_size': 32,
    'lr_learner': 1e-3, 'lr_gen': 0.05,
    'tau_init': 3.0, 'tau_min': 0.1, 'eps': 1.0, 'anneal_mode': 'exp',
    'hidden': 64, 'learner': 'twochannel', 'seed': 21,
    'strategy': 'alternating', 'hard_learner': True, 'S_d_rounds': 3,
    'lam_sp': 0.0, 'lam_ent': 0.005,
}


def run_one(net_type, cfg, eval_light=False):
    os.makedirs(RESULT_DIR, exist_ok=True)
    true_adj, edge_index, tx, ty, vx, vy, meta = generate_and_save_data(
        {'net_type': net_type, 'N': cfg['N'], 'p': cfg['p'], 'm': cfg['m'],
         'k': cfg['k'], 'rewiring_p': cfg['rewiring_p']},
        RESULT_DIR, beta=cfg['beta_true'], gamma=cfg['gamma_true'], T=cfg['T'],
        n_seeds=cfg['n_seeds'], n_repeats=cfg['n_repeats'], seed=cfg['seed'])
    cfg = {**cfg, 'data_meta': meta}
    n_nodes = int(meta['n_nodes'])

    train_loader, val_loader = build_loaders_from_arrays(
        tx, ty, vx, vy, batch_size=cfg['batch_size'])

    model = GGNModel(n_nodes, learner=cfg['learner'], hidden=cfg['hidden'],
                     tau_init=cfg['tau_init'], tau_min=cfg['tau_min'], eps=cfg['eps'],
                     anneal_mode=cfg['anneal_mode'])
    tag = f'{net_type}_basic'
    rec = train_ggn(model, train_loader, val_loader,
                    epochs=cfg['epochs'], S_d=cfg['S_d'], S_n=cfg['S_n'],
                    lr_learner=cfg['lr_learner'], lr_gen=cfg['lr_gen'],
                    strategy=cfg['strategy'], hard_learner=cfg['hard_learner'],
                    S_d_rounds=cfg['S_d_rounds'],
                    val_every=5, lam_sp=cfg['lam_sp'], lam_ent=cfg['lam_ent'],
                    save_dir=os.path.join(RESULT_DIR, 'models'), tag=tag,
                    verbose_every=10)

    # ---- 重构评估：0.5 阈值硬图为主指标 ----
    soft_adj = model.get_adj().numpy()
    np.fill_diagonal(soft_adj, 0.0)
    hard_adj = (soft_adj > 0.5).astype(np.float32)
    np.fill_diagonal(hard_adj, 0.0)
    recon = soft_adj_metrics(true_adj, soft_adj)          # 含 AUC(soft) 与 0.5/自动阈值明细
    hard_half = recon['hard_at_0.5']                       # 主指标：0.5 阈值
    hard_half['auc_hard'] = float(
        __import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(
            true_adj[np.triu_indices(n_nodes, 1)], hard_adj[np.triu_indices(n_nodes, 1)]))

    # ---- 动力学评估（硬图为主） ----
    A_soft_t = torch.tensor(soft_adj, dtype=torch.float32)
    A_hard_t = torch.tensor(hard_adj, dtype=torch.float32)
    dyn_soft = dynamics_metrics(model, val_loader, A_soft_t)
    dyn_hard = dynamics_metrics(model, val_loader, A_hard_t)

    # ---- 自回归（多初始节点平均） ----
    deg = true_adj.sum(1)
    inits = [np.zeros(n_nodes, dtype=int) for _ in range(4)]
    inits[0][int(np.argmax(deg))] = 1
    inits[1][int(np.argmin(deg))] = 1
    inits[2][0] = 1
    rng = np.random.RandomState(cfg['seed'])
    inits[3][int(rng.randint(0, n_nodes))] = 1
    ar_soft = autoregressive_mae(model, soft_adj, inits, true_adj, cfg['beta_true'],
                                 cfg['gamma_true'], T=10 if eval_light else 50,
                                 n_mc=2 if eval_light else 30, n_mc_true=100, seed=cfg['seed'])
    ar_hard = autoregressive_mae(model, hard_adj, inits, true_adj, cfg['beta_true'],
                                 cfg['gamma_true'], T=10 if eval_light else 50,
                                 n_mc=2 if eval_light else 30, n_mc_true=100, seed=cfg['seed'])
    perk = per_k_beta_fit(model, hard_adj, true_adj, beta_true=cfg['beta_true'],
                          n_states=50 if eval_light else 1500, seed=cfg['seed'])

    result = {
        'config': cfg,
        'losses': rec,
        'recon': recon,
        'recon_hard_05': hard_half,
        'dynamics_soft': dyn_soft,
        'dynamics_hard': dyn_hard,
        'autoreg_soft': ar_soft,
        'autoreg_hard': ar_hard,
        'perk': perk,
        'matrices': {
            'true_adj': true_adj.astype(float).tolist(),
            'soft_adj': soft_adj.astype(float).tolist(),
            'hard_adj': hard_adj.astype(float).tolist(),
        },
        'weights': {'model': f'models/{tag}_model.pt'},
    }
    out = os.path.join(RESULT_DIR, f'dgsl_basic_{net_type}.json')
    save_json(result, out)
    print(f'[done] {net_type}: AUC_s={recon["auc_soft"]:.4f} '
          f'ACC(hard@0.5)={hard_half["acc"]:.4f} TPR={hard_half["tpr"]:.4f} '
          f'FPR={hard_half["fpr"]:.4f} edges={hard_half["n_edges_pred"]}/{hard_half["n_edges_true"]} '
          f'acc_dyn={dyn_hard["acc_dyn"]:.4f} MAE_I(hard)={ar_hard["mae_i"]:.4f} '
          f'beta_fit={perk["beta_fit"]}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['ER', 'WS', 'BA'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.smoke:
        cfg = {**CONFIG, 'epochs': 2, 'S_d': 2, 'S_n': 2, 'batch_size': 16, 'hidden': 16,
               'lr_gen': 0.05}
        run_one('ER', cfg, eval_light=True)
        print('SMOKE OK')
        return
    for net in args.nets:
        run_one(net, {**CONFIG, 'net_type': net}, eval_light=False)


if __name__ == '__main__':
    main()
