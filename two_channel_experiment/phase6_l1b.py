# -*- coding: utf-8 -*-
"""
Phase 6 L1b：训练样本重加权（重采样）——BA 验证。

v4 训练完全不变（论文式交替 + 硬图学习器 + 双通道 + 指数降温 3->0.1 + 熵正则），
仅将 train_loader 换成按"信息量"加权的 WeightedRandomSampler：
  w(sample) = 1 + 0.1 * min(n_I, 30)
其中 n_I = X_t 中感染者数量。无感染活动的样本（n_I=0）权重 1，感染进行中的
样本权重最高（≤4），把训练重心移向"结构可辨识证据"所在的样本。

评估：0.5 阈值 + oracle F1（K 扫描，诊断天花板）+ valCE 无监督校准（L0）+ 动力学指标。
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json, set_seed  # noqa: E402
from dgsl_common import (  # noqa: E402
    GGNModel, generate_and_save_data, build_loaders_from_arrays, train_ggn,
    soft_adj_metrics, autoregressive_mae, per_k_beta_fit, DEVICE,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
BETA, GAMMA = 0.3, 0.1
# 无放回分层采样配额：按 n_I 分层（0 / 1-10 / 11-30 / >30）
#   平凡样本(n_I=0)降权至 5%，高信息(1-10)全量保留，中信息(11-30)提升，饱和段(>30)降权
QUOTAS = {0: 300, 1: 2700, 2: 1800, 3: 1200}
NET_PARAMS = {'ER': dict(net_type='ER', p=0.1), 'WS': dict(net_type='WS', k=6, rewiring_p=0.2),
              'BA': dict(net_type='BA', m=3)}


class StratifiedLoader:
    """无放回分层采样 loader：每轮（__iter__）按配额从各层无放回抽取样本组成一轮。
    层内每轮 shuffle 后取前 q 个（无放回）；跨轮重新 shuffle。方差小、分布聚焦、可复现。"""

    def __init__(self, tx, ty, quotas, batch_size, seed=21):
        self.tx, self.ty = tx, ty
        self.batch_size = batch_size
        self.quotas = quotas
        self.rng = np.random.RandomState(seed)
        n_i = tx[:, :, 1].sum(axis=1)
        self.bins = np.digitize(n_i, [0.5, 10.5, 30.5])
        self.layer_idx = [np.where(self.bins == b)[0] for b in range(4)]

    def __iter__(self):
        idx = []
        for b in range(4):
            pool = self.layer_idx[b].copy()
            self.rng.shuffle(pool)
            q = min(self.quotas[b], len(pool))
            idx.extend(pool[:q])
        self.rng.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            j = idx[i:i + self.batch_size]
            yield torch.tensor(self.tx[j]), torch.tensor(self.ty[j])

    def __len__(self):
        n = sum(min(q, len(li)) for q, li in zip(self.quotas.values(), self.layer_idx))
        return max(1, n // self.batch_size)


def oracle_and_valce(model, val_loader, true_adj, soft, n_nodes):
    """oracle F1（诊断天花板）+ valCE 无监督选 K（L0 校准）。"""
    iu = np.triu_indices(n_nodes, 1)
    y_true = true_adj[iu].astype(int)
    y_score = soft[iu]
    order = np.argsort(y_score)[::-1]
    n_true = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    best_f1, best_k, best_acc = -1, 0, 0.0
    k_candidates = sorted(set(list(range(50, 2501, 50)) + [int((y_score > 0.5).sum())]))
    model.eval()
    best_ce, K_ce = float('inf'), None
    with torch.no_grad():
        for K in k_candidates:
            pred = np.zeros(len(y_true), dtype=int)
            pred[order[:K]] = 1
            tp = (pred & y_true).sum(); fp = (pred & (1 - y_true)).sum()
            tpr = tp / max(n_true, 1); fpr = fp / max(n_neg, 1)
            p = tp / max(tp + fp, 1)
            f1 = 2 * p * tpr / max(p + tpr, 1e-9)
            if f1 > best_f1:
                best_f1, best_k, best_acc = float(f1), K, float(np.mean(pred == y_true))
            # valCE
            hard = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            hard[iu[0][order[:K]], iu[1][order[:K]]] = 1.0
            hard[iu[1][order[:K]], iu[0][order[:K]]] = 1.0
            A_t = torch.tensor(hard, device=DEVICE)
            ce, nv = 0.0, 0
            for x, y in val_loader:
                logits = model.learner(x.to(DEVICE), A_t)
                ce += F.cross_entropy(logits.reshape(-1, 3),
                                      y.argmax(-1).reshape(-1).to(DEVICE),
                                      reduction='sum').item()
                nv += y.size(0) * y.size(1)
            ce = ce / max(nv, 1)
            if ce < best_ce:
                best_ce, K_ce = float(ce), K
    return {'oracle_f1': {'K': best_k, 'f1': best_f1, 'acc': best_acc},
            'valce': {'K': K_ce, 'val_ce': best_ce}}


def run(net='BA', seed=21, smoke=False):
    os.makedirs(RESULT_DIR, exist_ok=True)
    set_seed(seed)   # 固定完整随机状态（含 torch），保证可复现
    np_params = NET_PARAMS[net]
    true_adj, edge_index, tx, ty, vx, vy, meta = generate_and_save_data(
        {**np_params, 'N': 100, 'p': np_params.get('p', 0.1), 'm': np_params.get('m', 3),
         'k': np_params.get('k', 6), 'rewiring_p': np_params.get('rewiring_p', 0.2)},
        RESULT_DIR, beta=BETA, gamma=GAMMA, T=60, n_seeds=30, n_repeats=5, seed=seed)
    n_nodes = int(meta['n_nodes'])
    batch = 16 if smoke else 64
    train_loader = StratifiedLoader(tx, ty, QUOTAS, batch, seed=seed)
    val_loader = build_loaders_from_arrays(tx, ty, vx, vy, batch_size=batch)[1]

    model = GGNModel(n_nodes, learner='twochannel', hidden=64,
                     tau_init=3.0, tau_min=0.1, eps=1.0, anneal_mode='exp')
    tag = f'{net}_l1b_s{seed}'
    rec = train_ggn(model, train_loader, val_loader,
                    epochs=2 if smoke else 100, S_d=40, S_n=20,
                    lr_learner=1e-3, lr_gen=0.05, strategy='alternating',
                    hard_learner=True, S_d_rounds=1 if smoke else 3,
                    val_every=5, lam_sp=0.0, lam_ent=0.005,
                    save_dir=os.path.join(RESULT_DIR, 'models'), tag=tag,
                    verbose_every=10)

    # ---- 评估 ----
    soft = model.get_adj().numpy()
    np.fill_diagonal(soft, 0)
    hard05 = (soft > 0.5).astype(np.float32)
    np.fill_diagonal(hard05, 0)
    recon = soft_adj_metrics(true_adj, soft)
    h05 = recon['hard_at_0.5']
    cal = oracle_and_valce(model, val_loader, true_adj, soft, n_nodes)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        A_t = torch.tensor(hard05, dtype=torch.float32)
        for x, y in val_loader:
            logits = model.learner(x.to(DEVICE), A_t)
            tgt = y.argmax(-1)
            correct += (logits.argmax(-1) == tgt).sum().item()
            total += tgt.numel()
    dyn = {'acc_dyn': correct / max(total, 1)}
    deg = true_adj.sum(1)
    inits = [np.zeros(n_nodes, dtype=int) for _ in range(4)]
    inits[0][int(np.argmax(deg))] = 1
    inits[1][int(np.argmin(deg))] = 1
    inits[2][0] = 1
    inits[3][int(np.random.RandomState(21).randint(0, n_nodes))] = 1
    ar_hard = autoregressive_mae(model, hard05, inits, true_adj, BETA, GAMMA,
                                 T=10 if smoke else 50, n_mc=2 if smoke else 30,
                                 n_mc_true=100, seed=21)
    perk = per_k_beta_fit(model, hard05, true_adj, beta_true=BETA,
                          n_states=50 if smoke else 1500, seed=21)

    tag = f'dgsl_l1b_{net}_s{seed}'
    n_i = tx[:, :, 1].sum(axis=1)
    result = {
        'config': {'experiment': 'phase6_l1b', 'net_type': net,
                   'resample': 'StratifiedLoader (no-replacement, quotas by n_I bins)',
                   'quotas': QUOTAS,
                   'epochs': 100, 'S_d_rounds': 3, 'S_n': 20, 'batch_size': 64,
                   'lr_learner': 1e-3, 'lr_gen': 0.05, 'tau_init': 3.0, 'tau_min': 0.1,
                   'anneal_mode': 'exp', 'learner': 'twochannel', 'hidden': 64,
                   'seed': seed, 'lam_ent': 0.005, 'strategy': 'alternating',
                   'hard_learner': True, 'data_meta': meta},
        'losses': rec,
        'recon': recon,
        'recon_hard_05': h05,
        'calibration': cal,
        'dynamics_hard': dyn,
        'autoreg_hard': ar_hard,
        'perk': perk,
        'matrices': {'true_adj': true_adj.astype(float).tolist(),
                     'soft_adj': soft.astype(float).tolist(),
                     'hard_adj': hard05.astype(float).tolist()},
        'weights': {'model': f'models/{net}_l1b_s{seed}_model.pt',
                    'layer_stats': {'n_I0': int((n_i == 0).sum()),
                                    'n_1_10': int(((n_i >= 1) & (n_i <= 10)).sum()),
                                    'n_11_30': int(((n_i > 10) & (n_i <= 30)).sum()),
                                    'n_gt30': int((n_i > 30).sum())}},
    }
    torch.save(model.state_dict(),
               os.path.join(RESULT_DIR, 'models', f'{net}_l1b_s{seed}_model.pt'))
    out = os.path.join(RESULT_DIR, f'{tag}.json')
    save_json(result, out)
    print(f'[done] {net} s{seed} L1b: ACC(0.5)={h05["acc"]:.4f} '
          f'edges={h05["n_edges_pred"]}/{h05["n_edges_true"]} | '
          f'oracle F1={cal["oracle_f1"]["f1"]:.3f} (K={cal["oracle_f1"]["K"]}) | '
          f'valCE K={cal["valce"]["K"]} | acc_dyn={dyn["acc_dyn"]:.4f} '
          f'MAE_I={ar_hard["mae_i"]:.4f} beta_fit={perk["beta_fit"]}', flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['BA'])
    ap.add_argument('--seed', type=int, default=21)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    for net in args.nets:
        run(net=net, seed=args.seed, smoke=args.smoke)
