# -*- coding: utf-8 -*-
"""L0 阈值校准实验（免费，不重训）：
对 v4（dgsl_basic）软图按强度排序扫描 K：
  - oracle 指标：真实图参照下每个 K 的 ACC/F1/TPR/FPR（L0 天花板，仅诊断用）
  - 无监督选择：现有学习器在 top-K 硬图上的验证集单步 CE（真实场景可用信号）
对比 0.5 阈值基线，并检查"验证 CE 最小"能否选出接近 oracle 的 K。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json  # noqa: E402
from dgsl_common import (  # noqa: E402
    GGNModel, generate_and_save_data, build_loaders_from_arrays, DEVICE,
)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
KS = list(range(50, 3001, 50))


def eval_hard(true_y, y_score, order, K):
    pred = np.zeros(len(true_y), dtype=int)
    pred[order[:K]] = 1
    tp = (pred & true_y).sum(); fp = (pred & (1 - true_y)).sum()
    fn = ((1 - pred) & true_y).sum()
    acc = float(np.mean(pred == true_y))
    tpr = tp / max(tp + fn, 1)
    fpr = fp / max((1 - true_y).sum(), 1)
    p = tp / max(tp + fp, 1)
    f1 = 2 * p * tpr / max(p + tpr, 1e-9)
    return {'K': K, 'acc': acc, 'f1': float(f1), 'tpr': float(tpr), 'fpr': float(fpr)}


def main():
    torch.set_num_threads(4)
    out = {}
    for net in ['ER', 'WS', 'BA']:
        d = json.load(open(os.path.join(D, f'dgsl_basic_{net}.json'), encoding='utf-8'))
        cfg = d['config']
        n_nodes = int(cfg['data_meta']['n_nodes'])
        soft = np.array(d['matrices']['soft_adj'])
        true = np.array(d['matrices']['true_adj'])
        iu = np.triu_indices(n_nodes, 1)
        y_true = true[iu].astype(int)
        y_score = soft[iu]
        order = np.argsort(y_score)[::-1]

        # 加载模型与验证集（验证 CE 作为无监督选择信号）
        model = GGNModel(n_nodes, learner='twochannel', hidden=cfg['hidden'],
                         tau_init=cfg['tau_init'], tau_min=cfg['tau_min'], eps=cfg['eps'],
                         anneal_mode=cfg['anneal_mode'])
        model.load_state_dict(torch.load(os.path.join(D, d['weights']['model']),
                                         map_location='cpu'))
        model.eval()
        _, _, tx, ty, vx, vy, meta = generate_and_save_data(
            {'net_type': net, 'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2},
            D, beta=0.3, gamma=0.1, T=60, n_seeds=30, n_repeats=5, seed=21)
        train_loader, val_loader = build_loaders_from_arrays(tx, ty, vx, vy, batch_size=64)

        # 0.5 阈值基线
        base = eval_hard(y_true, y_score, order, int((y_score > 0.5).sum()))

        # K 扫描
        rows, ce_by_k = [], {}
        with torch.no_grad():
            for K in KS:
                hard = np.zeros((n_nodes, n_nodes), dtype=np.float32)
                pairs = np.stack([iu[0][order[:K]], iu[1][order[:K]]])
                hard[pairs[0], pairs[1]] = 1.0
                hard[pairs[1], pairs[0]] = 1.0
                A_t = torch.tensor(hard, device=DEVICE)
                ce, nv = 0.0, 0
                for x, y in val_loader:
                    logits = model.learner(x.to(DEVICE), A_t)
                    ce += F.cross_entropy(logits.reshape(-1, 3),
                                          y.argmax(-1).reshape(-1).to(DEVICE),
                                          reduction='sum').item()
                    nv += y.size(0) * y.size(1)
                ce_by_k[K] = ce / max(nv, 1)
                rows.append(eval_hard(y_true, y_score, order, K))

        best_f1 = max(rows, key=lambda r: r['f1'])
        best_acc = max(rows, key=lambda r: r['acc'])
        K_ce = min(ce_by_k, key=ce_by_k.get)
        row_ce = eval_hard(y_true, y_score, order, K_ce)

        # 记录每个 K 的 (f1, val_ce) 曲线供后续可视化
        out[net] = {
            'auc_soft': d['recon']['auc_soft'],
            'n_true': int(y_true.sum()),
            'base_0.5': base,
            'oracle_f1': best_f1,
            'oracle_acc': best_acc,
            'valce_min': {'K': K_ce, 'val_ce': ce_by_k[K_ce], 'metrics': row_ce},
            'curves': {'K': KS, 'f1': [r['f1'] for r in rows],
                       'acc': [r['acc'] for r in rows],
                       'val_ce': [ce_by_k[K] for K in KS]},
        }
        print(f"[{net}] AUC={d['recon']['auc_soft']:.3f} n_true={y_true.sum()} | "
              f"0.5阈值: K={base['K']} ACC={base['acc']:.3f} F1={base['f1']:.3f} | "
              f"oracleF1: K={best_f1['K']} F1={best_f1['f1']:.3f} ACC={best_f1['acc']:.3f} "
              f"TPR={best_f1['tpr']:.3f} FPR={best_f1['fpr']:.3f} | "
              f"valCE最小: K={K_ce} F1={row_ce['f1']:.3f} ACC={row_ce['acc']:.3f} "
              f"TPR={row_ce['tpr']:.3f} FPR={row_ce['fpr']:.3f} valCE={ce_by_k[K_ce]:.4f}", flush=True)
    save_json(out, os.path.join(D, 'thresh_calib_l0.json'))
    print('-> thresh_calib_l0.json', flush=True)


if __name__ == '__main__':
    main()
