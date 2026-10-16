# -*- coding: utf-8 -*-
"""
Phase 5 补充：用已保存权重重算自回归评估（多初始节点平均，消除单一初始节点/枢纽假象），
并重算 per-k 诊断；就地更新 results/phase5/*.json 的 autoreg_* 与 perk 字段。
（不重训，仅重评估——对应 DSH.md 坑8 的"权重是重采样的前提"。）

初始节点集合：最大度（枢纽）、最小度（叶子）、节点 0、一个随机节点，共 4 个。
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgsl_common import (  # noqa: E402
    GGNModel, generate_and_save_data, autoregressive_mae, per_k_beta_fit, save_json,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase5')
FILES = [f'dgsl_core_{n}.json' for n in ['ER', 'WS', 'BA']] + \
        [f'dgsl_core_reg_{n}.json' for n in ['ER', 'WS', 'BA']]
torch.set_num_threads(4)


def main():
    for fn in FILES:
        path = os.path.join(RESULT_DIR, fn)
        d = json.load(open(path, encoding='utf-8'))
        cfg = d['config']
        net = cfg['net_type']
        is_reg = cfg.get('lam_sp', 0) > 0 or cfg.get('lam_ent', 0) > 0
        tag = f'{net}_gnn' + ('_reg' if is_reg else '')
        wpath = os.path.join(RESULT_DIR, 'models', f'{tag}_model.pt')
        model = GGNModel(cfg['data_meta']['n_nodes'], learner='gnn', hidden=cfg['hidden'],
                         tau_init=cfg['tau_init'], tau_min=cfg['tau_min'], eps=cfg['eps'])
        model.load_state_dict(torch.load(wpath, map_location='cpu'))
        model.eval()

        true_adj = np.array(d['matrices']['true_adj'])
        soft = np.array(d['matrices']['soft_adj'])
        hard = np.array(d['matrices']['hard_adj'])
        n = true_adj.shape[0]

        deg = true_adj.sum(1)
        inits = [np.zeros(n, dtype=int) for _ in range(4)]
        inits[0][int(np.argmax(deg))] = 1          # 枢纽
        inits[1][int(np.argmin(deg))] = 1          # 叶子
        inits[2][0] = 1                            # 节点 0
        rng = np.random.RandomState(cfg['seed'])
        inits[3][int(rng.randint(0, n))] = 1       # 随机

        ar_soft = autoregressive_mae(model, soft, inits, true_adj, cfg['beta_true'],
                                     cfg['gamma_true'], T=50, n_mc=30, n_mc_true=100,
                                     seed=cfg['seed'])
        ar_hard = autoregressive_mae(model, hard, inits, true_adj, cfg['beta_true'],
                                     cfg['gamma_true'], T=50, n_mc=30, n_mc_true=100,
                                     seed=cfg['seed'])
        perk = per_k_beta_fit(model, hard, true_adj, beta_true=cfg['beta_true'],
                              n_states=1500, seed=cfg['seed'])

        d['autoreg_soft'] = ar_soft
        d['autoreg_hard'] = ar_hard
        d['perk'] = perk
        d['autoreg_note'] = '多初始节点平均（枢纽/叶子/节点0/随机），n_mc=30，T=50'
        save_json(d, path)
        print(f'{fn}: MAE_I(soft)={ar_soft["mae_i"]:.4f} MAE_I(hard)={ar_hard["mae_i"]:.4f} '
              f'per-init={[round(v,4) for v in ar_hard["mae_i_per_init"]]} '
              f'beta_fit={perk["beta_fit"]}', flush=True)


if __name__ == '__main__':
    main()
