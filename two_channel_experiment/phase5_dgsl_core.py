# -*- coding: utf-8 -*-
"""
Phase 5A —— DGSL 核心实验（GGN 式，模型无关）：
三网络（ER/WS/BA, N=100）× 两阶段评估（软图重构指标 + 二值化 + 动力学预测 + 自回归）。

严格遵循 DSH.md：计算→保存数据→绘图。本脚本只负责计算与保存 JSON + 权重；
绘图由 plot_phase5_core.py 只读 JSON 完成。

用法：
    python -u phase5_dgsl_core.py                 # 全量：ER WS BA，100 epochs
    python -u phase5_dgsl_core.py --nets ER       # 单网络
    python -u phase5_dgsl_core.py --smoke         # 冒烟测试（小规模全流程）
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

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase5')


def build_config(net_type, epochs, S_d, S_n, batch_size, lr_learner, lr_gen,
                 tau_init, tau_min, eps, hidden, learner, seed, smoke,
                 lam_sp=0.0, lam_ent=0.0):
    cfg = {
        'experiment': 'phase5_dgsl_core',
        'method': 'GGN (Gumbel-Softmax generator + GNN dynamics learner, model-free)',
        'net_type': net_type, 'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2,
        'beta_true': 0.3, 'gamma_true': 0.1, 'T': 60, 'n_seeds': 30, 'n_repeats': 5,
        'epochs': epochs, 'S_d': S_d, 'S_n': S_n, 'batch_size': batch_size,
        'lr_learner': lr_learner, 'lr_gen': lr_gen,
        'tau_init': tau_init, 'tau_min': tau_min, 'eps': eps,
        'hidden': hidden, 'learner': learner, 'seed': seed, 'smoke': bool(smoke),
        'strategy': 'alternating', 'lam_sp': lam_sp, 'lam_ent': lam_ent,
    }
    return cfg


def run_one(net_type, cfg, eval_light=False):
    """训练并评估单个网络，返回结果 dict（含 config）。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    true_adj, edge_index, tx, ty, vx, vy, meta = generate_and_save_data(
        {'net_type': net_type, 'N': cfg['N'], 'p': cfg['p'], 'm': cfg['m'],
         'k': cfg['k'], 'rewiring_p': cfg['rewiring_p']},
        RESULT_DIR, beta=cfg['beta_true'], gamma=cfg['gamma_true'], T=cfg['T'],
        n_seeds=cfg['n_seeds'], n_repeats=cfg['n_repeats'], seed=cfg['seed'])
    cfg['data_meta'] = meta

    train_loader, val_loader = build_loaders_from_arrays(
        tx, ty, vx, vy, batch_size=cfg['batch_size'])
    n_nodes = int(meta['n_nodes'])

    model = GGNModel(n_nodes, learner=cfg['learner'], hidden=cfg['hidden'],
                     tau_init=cfg['tau_init'], tau_min=cfg['tau_min'], eps=cfg['eps'])
    tag = f'{net_type}_{cfg["learner"]}' + ('_reg' if cfg['lam_sp'] > 0 or cfg['lam_ent'] > 0 else '')
    rec = train_ggn(model, train_loader, val_loader,
                    epochs=cfg['epochs'], S_d=cfg['S_d'], S_n=cfg['S_n'],
                    lr_learner=cfg['lr_learner'], lr_gen=cfg['lr_gen'],
                    strategy='alternating', val_every=5,
                    lam_sp=cfg['lam_sp'], lam_ent=cfg['lam_ent'],
                    save_dir=os.path.join(RESULT_DIR, 'models'), tag=tag,
                    verbose_every=10)

    # ---- 重构评估 ----
    soft_adj = model.get_adj().numpy()
    np.fill_diagonal(soft_adj, 0.0)
    recon = soft_adj_metrics(true_adj, soft_adj)
    thr = recon['auto_threshold']
    hard_adj = (soft_adj >= thr).astype(np.float32)
    np.fill_diagonal(hard_adj, 0.0)

    # ---- 动力学评估（软图 / 硬图） ----
    A_soft_t = torch.tensor(soft_adj, dtype=torch.float32)
    A_hard_t = torch.tensor(hard_adj, dtype=torch.float32)
    dyn_soft = dynamics_metrics(model, val_loader, A_soft_t)
    dyn_hard = dynamics_metrics(model, val_loader, A_hard_t)

    # ---- 自回归评估 ----
    init = np.zeros(n_nodes, dtype=int)
    init[0] = 1
    ar_soft = autoregressive_mae(model, soft_adj, init, true_adj,
                                 cfg['beta_true'], cfg['gamma_true'],
                                 T=10 if eval_light else 50,
                                 n_mc=2 if eval_light else 50, seed=cfg['seed'])
    ar_hard = autoregressive_mae(model, hard_adj, init, true_adj,
                                 cfg['beta_true'], cfg['gamma_true'],
                                 T=10 if eval_light else 50,
                                 n_mc=2 if eval_light else 50, seed=cfg['seed'])

    # ---- per-k 诊断（硬图 + 模型） ----
    perk = per_k_beta_fit(model, hard_adj, true_adj, beta_true=cfg['beta_true'],
                          n_states=50 if eval_light else 1500, seed=cfg['seed'])

    result = {
        'config': cfg,
        'losses': {k: (v if isinstance(v, list) else v) for k, v in rec.items()},
        'recon': recon,
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
        'weights': {
            'model': f'models/{tag}_model.pt',
        },
    }
    out = os.path.join(RESULT_DIR, f'dgsl_core{("_reg" if cfg["lam_sp"] > 0 or cfg["lam_ent"] > 0 else "")}_{net_type}.json')
    save_json(result, out)
    print(f'[done] {net_type}: AUC_soft={recon["auc_soft"]:.4f} '
          f'AUC_hard(auto)={recon["hard_auto"]["auc_hard"]:.4f} '
          f'acc_dyn(hard)={dyn_hard["acc_dyn"]:.4f} '
          f'MAE_I(hard)={ar_hard["mae_i"]:.4f} beta_fit={perk["beta_fit"]} '
          f'-> {out}', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['ER', 'WS', 'BA'])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--threads', type=int, default=0, help='torch 线程数；0=不设置')
    ap.add_argument('--lam-sp', type=float, default=0.0, help='稀疏惩罚系数')
    ap.add_argument('--lam-ent', type=float, default=0.0, help='熵惩罚系数')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.smoke:
        cfg = build_config('ER', epochs=2, S_d=1, S_n=2, batch_size=8,
                           lr_learner=1e-3, lr_gen=5e-3, tau_init=1.0, tau_min=0.1,
                           eps=1.0, hidden=16, learner='gnn', seed=21, smoke=True)
        run_one('ER', cfg, eval_light=True)
        print('SMOKE OK')
        return

    for net in args.nets:
        cfg = build_config(net, epochs=args.epochs, S_d=10, S_n=20, batch_size=32,
                           lr_learner=1e-3, lr_gen=5e-3, tau_init=1.0, tau_min=0.1,
                           eps=1.0, hidden=64, learner='gnn', seed=21, smoke=False,
                           lam_sp=args.lam_sp, lam_ent=args.lam_ent)
        run_one(net, cfg, eval_light=False)


if __name__ == '__main__':
    main()
