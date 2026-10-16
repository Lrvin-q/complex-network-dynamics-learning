# -*- coding: utf-8 -*-
"""
Phase 5B —— GGN 机制消融（单进程单配置，便于并行调度）：
  5B1  anneal : τ 退火 (1.0->0.1) vs 固定 τ=0.5        （ER, 50 epochs）
  5B2  alt    : 交替训练 (10,20) vs 同步训练             （ER, 50 epochs）
  5B3  tc     : TwoChannel 动力学学习器（软邻接聚合）三网络（100 epochs，与 core 同配置）

每个调用只跑一个 variant，输出独立 JSON：
  ablation_anneal_ER_{anneal|fixed}.json
  ablation_alt_ER_{alternating|simultaneous}.json
  ablation_twochannel_{ER|WS|BA}.json

遵循 DSH.md：计算→保存数据→绘图；JSON 带完整 config；保存权重。

用法（并行示例，每进程 --threads 3）：
  python -u phase5_dgsl_ablation.py --task anneal --variant anneal --threads 3
  python -u phase5_dgsl_ablation.py --task anneal --variant fixed  --threads 3
  python -u phase5_dgsl_ablation.py --task alt --variant alternating --threads 3
  python -u phase5_dgsl_ablation.py --task alt --variant simultaneous --threads 3
  python -u phase5_dgsl_ablation.py --task tc --nets ER WS BA --threads 3
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
    save_json,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase5')


def load_data(net_type):
    return generate_and_save_data(
        {'net_type': net_type, 'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2},
        RESULT_DIR, beta=0.3, gamma=0.1, T=60, n_seeds=30, n_repeats=5, seed=21)


def run_and_eval(net_type, cfg, eval_full=False):
    true_adj, edge_index, tx, ty, vx, vy, meta = load_data(net_type)
    n_nodes = int(meta['n_nodes'])
    train_loader, val_loader = build_loaders_from_arrays(tx, ty, vx, vy,
                                                         batch_size=cfg['batch_size'])
    model = GGNModel(n_nodes, learner=cfg['learner'], hidden=cfg['hidden'],
                     tau_init=cfg['tau_init'], tau_min=cfg['tau_min'], eps=cfg['eps'])
    tag = f"ab_{net_type}_{cfg['learner']}_{cfg['strategy']}_t{cfg['tau_init']}"
    rec = train_ggn(model, train_loader, val_loader,
                    epochs=cfg['epochs'], S_d=cfg['S_d'], S_n=cfg['S_n'],
                    lr_learner=cfg['lr_learner'], lr_gen=cfg['lr_gen'],
                    strategy=cfg['strategy'], val_every=5,
                    save_dir=os.path.join(RESULT_DIR, 'models'), tag=tag,
                    verbose_every=25)

    soft_adj = model.get_adj().numpy()
    np.fill_diagonal(soft_adj, 0.0)
    recon = soft_adj_metrics(true_adj, soft_adj)
    thr = recon['auto_threshold']
    hard_adj = (soft_adj >= thr).astype(np.float32)
    np.fill_diagonal(hard_adj, 0.0)
    dyn_soft = dynamics_metrics(model, val_loader, torch.tensor(soft_adj, dtype=torch.float32))
    dyn_hard = dynamics_metrics(model, val_loader, torch.tensor(hard_adj, dtype=torch.float32))

    out = {
        'config': {**cfg, 'data_meta': meta},
        'losses': rec,
        'recon': recon,
        'dynamics_soft': dyn_soft,
        'dynamics_hard': dyn_hard,
        'weights': {'model': f'models/{tag}_model.pt'},
    }
    if eval_full:
        init = np.zeros(n_nodes, dtype=int)
        init[0] = 1
        ar_soft = autoregressive_mae(model, soft_adj, init, true_adj, 0.3, 0.1,
                                     T=50, n_mc=50, seed=21)
        ar_hard = autoregressive_mae(model, hard_adj, init, true_adj, 0.3, 0.1,
                                     T=50, n_mc=50, seed=21)
        perk = per_k_beta_fit(model, hard_adj, true_adj, beta_true=0.3,
                              n_states=1500, seed=21)
        out['autoreg_soft'] = ar_soft
        out['autoreg_hard'] = ar_hard
        out['perk'] = perk
        out['matrices'] = {
            'true_adj': true_adj.astype(float).tolist(),
            'soft_adj': soft_adj.astype(float).tolist(),
            'hard_adj': hard_adj.astype(float).tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['anneal', 'alt', 'tc'], required=True)
    ap.add_argument('--variant', choices=['anneal', 'fixed', 'alternating', 'simultaneous'])
    ap.add_argument('--nets', nargs='+', default=['ER', 'WS', 'BA'])
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=50)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    if args.task == 'anneal':
        assert args.variant in ('anneal', 'fixed')
        ti, tm = (1.0, 0.1) if args.variant == 'anneal' else (0.5, 0.5)
        cfg = {'experiment': 'phase5_dgsl_ablation', 'net_type': 'ER',
               'epochs': args.epochs, 'S_d': 10, 'S_n': 20, 'batch_size': 32,
               'lr_learner': 1e-3, 'lr_gen': 5e-3, 'tau_init': ti, 'tau_min': tm,
               'eps': 1.0, 'hidden': 64, 'learner': 'gnn', 'seed': 21,
               'strategy': 'alternating', 'ablation': '5B1_anneal'}
        r = run_and_eval('ER', cfg)
        out = os.path.join(RESULT_DIR, f'ablation_anneal_ER_{args.variant}.json')
        save_json(r, out)
        print(f'[5B1 {args.variant}] AUC_soft={r["recon"]["auc_soft"]:.4f} '
              f'ACC(dyn)={r["dynamics_hard"]["acc_dyn"]:.4f} '
              f'best_val={r["losses"]["best_val"]:.4f} -> {out}', flush=True)
    elif args.task == 'alt':
        assert args.variant in ('alternating', 'simultaneous')
        cfg = {'experiment': 'phase5_dgsl_ablation', 'net_type': 'ER',
               'epochs': args.epochs, 'S_d': 10, 'S_n': 20, 'batch_size': 32,
               'lr_learner': 1e-3, 'lr_gen': 5e-3, 'tau_init': 1.0, 'tau_min': 0.1,
               'eps': 1.0, 'hidden': 64, 'learner': 'gnn', 'seed': 21,
               'strategy': args.variant, 'ablation': '5B2_alt'}
        r = run_and_eval('ER', cfg)
        out = os.path.join(RESULT_DIR, f'ablation_alt_ER_{args.variant}.json')
        save_json(r, out)
        print(f'[5B2 {args.variant}] AUC_soft={r["recon"]["auc_soft"]:.4f} '
              f'ACC(dyn)={r["dynamics_hard"]["acc_dyn"]:.4f} '
              f'best_val={r["losses"]["best_val"]:.4f} -> {out}', flush=True)
    else:
        for net in args.nets:
            cfg = {'experiment': 'phase5_dgsl_ablation', 'net_type': net,
                   'epochs': 100, 'S_d': 10, 'S_n': 20, 'batch_size': 32,
                   'lr_learner': 1e-3, 'lr_gen': 5e-3, 'tau_init': 1.0, 'tau_min': 0.1,
                   'eps': 1.0, 'hidden': 64, 'learner': 'twochannel', 'seed': 21,
                   'strategy': 'alternating', 'ablation': '5B3_twochannel'}
            r = run_and_eval(net, cfg, eval_full=True)
            out = os.path.join(RESULT_DIR, f'ablation_twochannel_{net}.json')
            save_json(r, out)
            print(f'[5B3 {net}] AUC_soft={r["recon"]["auc_soft"]:.4f} '
                  f'ACC(dyn)={r["dynamics_hard"]["acc_dyn"]:.4f} '
                  f'MAE_I={r["autoreg_hard"]["mae_i"]:.4f} -> {out}', flush=True)


if __name__ == '__main__':
    main()
