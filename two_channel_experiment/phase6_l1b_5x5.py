# -*- coding: utf-8 -*-
"""对 BA L1b 三 seed 权重补跑 5×5 自回归评估（真实=真实图+真实参数，模型=硬图推演，
T=50, n_mc=50，5 组种子 × 5 初始人数），写入 dgsl_l1b_BA_s{seed}.json 的 autoreg_5x5 字段。"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json  # noqa: E402
from dgsl_common import GGNModel, generate_and_save_data  # noqa: E402
from phase6_eval_extended import autoreg_5x5, DGSLAdapter  # noqa: E402

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')


def main():
    torch.set_num_threads(4)
    true_adj, edge_index, tx, ty, vx, vy, meta = generate_and_save_data(
        {'net_type': 'BA', 'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2},
        RESULT_DIR, beta=0.3, gamma=0.1, T=60, n_seeds=30, n_repeats=5, seed=21)
    n_nodes = int(meta['n_nodes'])
    iu = np.triu_indices(n_nodes, 1)
    pairs = [(i, j) for i, j in zip(iu[0], iu[1]) if true_adj[i, j] > 0.5]
    true_e = torch.tensor(np.array(pairs).T, dtype=torch.long) if pairs else torch.zeros(2, 0, dtype=torch.long)

    for seed in [7, 21, 42]:
        path = os.path.join(RESULT_DIR, f'dgsl_l1b_BA_s{seed}.json')
        d = json.load(open(path, encoding='utf-8'))
        model = GGNModel(n_nodes, learner='twochannel', hidden=64,
                         tau_init=3.0, tau_min=0.1, eps=1.0, anneal_mode='exp')
        model.load_state_dict(torch.load(
            os.path.join(RESULT_DIR, 'models', f'BA_l1b_s{seed}_model.pt'), map_location='cpu'))
        model.eval()
        hard = np.array(d['matrices']['hard_adj'])
        np.fill_diagonal(hard, 0)
        adapter = DGSLAdapter(model, hard)
        d['autoreg_5x5'] = autoreg_5x5(adapter, hard, true_adj, true_e, n_nodes)
        save_json(d, path)
        print(f'[s{seed}] 5x5 autoreg MAE(I)={d["autoreg_5x5"]["mae_i_mean"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
