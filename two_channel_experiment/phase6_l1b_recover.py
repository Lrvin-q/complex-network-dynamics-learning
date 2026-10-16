# -*- coding: utf-8 -*-
"""从 BA L1b 三 seed 权重补评估，生成 dgsl_l1b_BA_s{seed}.json（修复文件名冲突 bug 后的规范名）。"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json  # noqa: E402
from dgsl_common import (  # noqa: E402
    GGNModel, generate_and_save_data, soft_adj_metrics, autoregressive_mae,
    per_k_beta_fit,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
BETA, GAMMA = 0.3, 0.1


def main():
    torch.set_num_threads(4)
    true_adj, edge_index, tx, ty, vx, vy, meta = generate_and_save_data(
        {'net_type': 'BA', 'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2},
        RESULT_DIR, beta=BETA, gamma=GAMMA, T=60, n_seeds=30, n_repeats=5, seed=21)
    n_nodes = int(meta['n_nodes'])

    for seed in [7, 21, 42]:
        model = GGNModel(n_nodes, learner='twochannel', hidden=64,
                         tau_init=3.0, tau_min=0.1, eps=1.0, anneal_mode='exp')
        model.load_state_dict(torch.load(
            os.path.join(RESULT_DIR, 'models', f'BA_l1b_s{seed}_model.pt'),
            map_location='cpu'))
        model.eval()

        soft = model.get_adj().numpy()
        np.fill_diagonal(soft, 0)
        hard = (soft > 0.5).astype(np.float32)
        np.fill_diagonal(hard, 0)
        recon = soft_adj_metrics(true_adj, soft)
        h05 = recon['hard_at_0.5']

        deg = true_adj.sum(1)
        inits = [np.zeros(n_nodes, dtype=int) for _ in range(4)]
        inits[0][int(np.argmax(deg))] = 1
        inits[1][int(np.argmin(deg))] = 1
        inits[2][0] = 1
        inits[3][int(np.random.RandomState(21).randint(0, n_nodes))] = 1
        ar_hard = autoregressive_mae(model, hard, inits, true_adj, BETA, GAMMA,
                                     T=50, n_mc=30, n_mc_true=100, seed=21)
        ar_soft = autoregressive_mae(model, soft, inits, true_adj, BETA, GAMMA,
                                     T=50, n_mc=30, n_mc_true=100, seed=21)
        perk = per_k_beta_fit(model, hard, true_adj, beta_true=BETA,
                              n_states=1500, seed=21)

        result = {
            'seed': seed,
            'recon': recon,
            'recon_hard_05': h05,
            'autoreg_hard': ar_hard,
            'autoreg_soft': ar_soft,
            'perk': perk,
            'matrices': {'true_adj': true_adj.astype(float).tolist(),
                         'soft_adj': soft.astype(float).tolist(),
                         'hard_adj': hard.astype(float).tolist()},
        }
        out = os.path.join(RESULT_DIR, f'dgsl_l1b_BA_s{seed}.json')
        save_json(result, out)
        print(f'[s{seed}] ACC(0.5)={h05["acc"]:.4f} edges={h05["n_edges_pred"]}/{h05["n_edges_true"]} '
              f'MAE_I(hard)={ar_hard["mae_i"]:.4f} beta_fit={perk["beta_fit"]} -> {out}', flush=True)


if __name__ == '__main__':
    main()
