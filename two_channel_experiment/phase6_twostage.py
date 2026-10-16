# -*- coding: utf-8 -*-
"""
Phase 6 两阶段训练实验（BA）：
  阶段一（1-50 轮）：同步软图粗学——生成器出 A_soft（带 Gumbel 噪声），学习器在软图上
      训练，损失同时更新 A_soft 与学习器参数（同步）；熵正则 λ=0.005 控密度；τ 3→0.5。
  阶段二（51-100 轮）：v4 式交替硬图精修——每 epoch 采样 0.5 阈值硬图，
      学习器只在硬图上训练（生成器冻结）；生成器阶段软图带梯度、学习器冻结，
      仅更新生成器（+熵正则）；τ 0.5→0.1。

评估与 phase6_basic 同口径（硬图 ACC/TPR/FPR/边数、acc_dyn、多初始自回归、per-k β̂），
对比基线 dgsl_basic_BA.json（ACC=0.818）。

用法：python -u phase6_twostage.py [--smoke]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_json  # noqa: E402
from dgsl_common import (  # noqa: E402
    GGNModel, generate_and_save_data, build_loaders_from_arrays,
    soft_adj_metrics, autoregressive_mae, per_k_beta_fit, DEVICE,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
BETA, GAMMA = 0.3, 0.1
NET_PARAMS = {'ER': dict(net_type='ER', p=0.1), 'WS': dict(net_type='WS', k=6, rewiring_p=0.2),
              'BA': dict(net_type='BA', m=3)}


def train_two_stage(model, train_loader, val_loader,
                    epochs1, epochs2, sync_steps, S_d_rounds, S_n,
                    lr_learner, lr_gen, lam_ent, smoke=False):
    opt_l = torch.optim.Adam(model.learner.parameters(), lr=lr_learner)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_gen)
    criterion = nn.CrossEntropyLoss()
    rec = {'epochs': [], 'stage': [], 'train_learner': [], 'train_gen': [],
           'val': [], 'tau': [], 'a_mean': [], 'a_std': []}
    best_val, best_state = float('inf'), None

    def cycle(loader):
        while True:
            for b in loader:
                yield b

    # ================= 阶段一：同步软图粗学（τ 3 -> 0.5） =================
    model.generator.reset_schedule(3.0, 0.5, epochs1 * sync_steps)
    it = iter(cycle(train_loader))
    for epoch in range(epochs1):
        t0 = time.time()
        ep_l = ep_g = 0.0
        for _ in range(sync_steps):
            x, y = next(it)
            x, y = x.to(DEVICE), y.to(DEVICE)
            A = model.generator.get_adj(training=True)
            logits = model.learner(x, A)
            loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
            A_c = A.clamp(1e-6, 1 - 1e-6)
            loss = loss + lam_ent * (-(A_c * torch.log(A_c) + (1 - A_c) * torch.log(1 - A_c))).mean()
            opt_l.zero_grad(); opt_g.zero_grad(); loss.backward()
            opt_l.step(); opt_g.step()
            model.generator.anneal()
            ep_l += loss.item(); ep_g += loss.item()
        do_val = ((epoch + 1) % 5 == 0) or (epoch + 1 == epochs1) or smoke
        if do_val:
            model.eval()
            vloss, nv = 0.0, 0
            with torch.no_grad():
                A_v = model.generator.get_adj(training=False)
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    vloss += criterion(model.learner(x, A_v).reshape(-1, 3),
                                       y.argmax(-1).reshape(-1)).item() * x.size(0)
                    nv += x.size(0)
            model.train()
            avg_v = vloss / max(nv, 1)
            A_s = model.generator.get_adj(training=False).numpy()
            rec['epochs'].append(epoch + 1); rec['stage'].append(1)
            rec['train_learner'].append(ep_l / sync_steps)
            rec['train_gen'].append(ep_g / sync_steps)
            rec['val'].append(avg_v)
            rec['tau'].append(float(model.generator.tau.item()))
            rec['a_mean'].append(float(A_s.mean()))
            rec['a_std'].append(float(A_s.std()))
            if avg_v < best_val:
                best_val = avg_v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'[S1 ep {epoch+1}/{epochs1}] val={avg_v:.4f} tau={rec["tau"][-1]:.3f} '
                  f'A={rec["a_mean"][-1]:.3f} time={time.time()-t0:.1f}s', flush=True)

    # ================= 阶段二：交替硬图精修（τ 0.5 -> 0.1） =================
    model.generator.reset_schedule(0.5, 0.1, epochs2 * S_n)
    it_g = iter(cycle(train_loader))
    for epoch in range(epochs2):
        t0 = time.time()
        ep_l = ep_g = 0.0
        A_hard = model.generator.get_hard_adj(noise=True)
        # 学习器阶段：硬图，S_d_rounds 轮 × 全 loader（生成器冻结）
        for _ in range(S_d_rounds):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model.learner(x, A_hard)
                loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
                opt_l.zero_grad(); loss.backward(); opt_l.step()
                ep_l += loss.item()
        # 生成器阶段：软图带梯度，学习器冻结，仅更新生成器 + 熵正则
        for _ in range(S_n):
            x, y = next(it_g)
            x, y = x.to(DEVICE), y.to(DEVICE)
            A = model.generator.get_adj(training=True)
            logits = model.learner(x, A)
            loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
            A_c = A.clamp(1e-6, 1 - 1e-6)
            loss = loss + lam_ent * (-(A_c * torch.log(A_c) + (1 - A_c) * torch.log(1 - A_c))).mean()
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            model.generator.anneal()
            ep_g += loss.item()
        do_val = ((epoch + 1) % 5 == 0) or (epoch + 1 == epochs2) or smoke
        if do_val:
            model.eval()
            vloss, nv = 0.0, 0
            with torch.no_grad():
                A_v = model.generator.get_hard_adj(noise=False)
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    vloss += criterion(model.learner(x, A_v).reshape(-1, 3),
                                       y.argmax(-1).reshape(-1)).item() * x.size(0)
                    nv += x.size(0)
            model.train()
            avg_v = vloss / max(nv, 1)
            A_s = model.generator.get_adj(training=False).numpy()
            rec['epochs'].append(epochs1 + epoch + 1); rec['stage'].append(2)
            rec['train_learner'].append(ep_l / max(S_d_rounds * len(train_loader), 1))
            rec['train_gen'].append(ep_g / max(S_n, 1))
            rec['val'].append(avg_v)
            rec['tau'].append(float(model.generator.tau.item()))
            rec['a_mean'].append(float(A_s.mean()))
            rec['a_std'].append(float(A_s.std()))
            if avg_v < best_val:
                best_val = avg_v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'[S2 ep {epochs1+epoch+1}/{epochs1+epochs2}] val={avg_v:.4f} '
                  f'tau={rec["tau"][-1]:.3f} A={rec["a_mean"][-1]:.3f} '
                  f'time={time.time()-t0:.1f}s', flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    rec['best_val'] = float(best_val)
    return rec


def run(net='BA', smoke=False, lam_ent=0.005):
    os.makedirs(RESULT_DIR, exist_ok=True)
    np_params = NET_PARAMS[net]
    true_adj, edge_index, tx, ty, vx, vy, meta = generate_and_save_data(
        {**np_params, 'N': 100, 'p': np_params.get('p', 0.1), 'm': np_params.get('m', 3),
         'k': np_params.get('k', 6), 'rewiring_p': np_params.get('rewiring_p', 0.2)},
        RESULT_DIR, beta=BETA, gamma=GAMMA, T=60, n_seeds=30, n_repeats=5, seed=21)
    n_nodes = int(meta['n_nodes'])
    train_loader, val_loader = build_loaders_from_arrays(
        tx, ty, vx, vy, batch_size=64 if not smoke else 16)

    model = GGNModel(n_nodes, learner='twochannel', hidden=64,
                     tau_init=3.0, tau_min=0.5, eps=1.0, anneal_mode='exp')

    if smoke:
        epochs1, epochs2, sync_steps, S_d_rounds, S_n = 2, 2, 8, 1, 5
    else:
        epochs1, epochs2, sync_steps, S_d_rounds, S_n = 50, 50, 30, 3, 20

    rec = train_two_stage(model, train_loader, val_loader,
                          epochs1, epochs2, sync_steps, S_d_rounds, S_n,
                          lr_learner=1e-3, lr_gen=0.05, lam_ent=lam_ent, smoke=smoke)

    # ---- 评估（与 phase6_basic 同口径） ----
    soft = model.get_adj().numpy()
    np.fill_diagonal(soft, 0)
    hard = (soft > 0.5).astype(np.float32)
    np.fill_diagonal(hard, 0)
    recon = soft_adj_metrics(true_adj, soft)
    h05 = recon['hard_at_0.5']
    model.eval()
    correct = total = 0
    ce_sum = 0.0
    with torch.no_grad():
        A_t = torch.tensor(hard, dtype=torch.float32)
        for x, y in val_loader:
            logits = model.learner(x, A_t)
            tgt = y.argmax(-1)
            ce_sum += F.cross_entropy(logits.reshape(-1, 3), tgt.reshape(-1),
                                      reduction='sum').item()
            correct += (logits.argmax(-1) == tgt).sum().item()
            total += tgt.numel()
    dyn = {'acc_dyn': correct / max(total, 1), 'ce': ce_sum / max(total, 1)}
    deg = true_adj.sum(1)
    inits = [np.zeros(n_nodes, dtype=int) for _ in range(4)]
    inits[0][int(np.argmax(deg))] = 1
    inits[1][int(np.argmin(deg))] = 1
    inits[2][0] = 1
    inits[3][int(np.random.RandomState(21).randint(0, n_nodes))] = 1
    ar_hard = autoregressive_mae(model, hard, inits, true_adj, BETA, GAMMA,
                                 T=10 if smoke else 50, n_mc=2 if smoke else 30,
                                 n_mc_true=100, seed=21)
    perk = per_k_beta_fit(model, hard, true_adj, beta_true=BETA,
                          n_states=50 if smoke else 1500, seed=21)

    tag = f'dgsl_twostage_{net}'
    result = {
        'config': {'experiment': 'phase6_twostage', 'net_type': net,
                   'epochs1': epochs1, 'epochs2': epochs2,
                   'sync_steps': sync_steps, 'S_d_rounds': S_d_rounds, 'S_n': S_n,
                   'batch_size': 64, 'lr_learner': 1e-3, 'lr_gen': 0.05,
                   'tau_stage1': [3.0, 0.5], 'tau_stage2': [0.5, 0.1],
                   'anneal_mode': 'exp', 'learner': 'twochannel', 'hidden': 64,
                   'seed': 21, 'lam_ent': lam_ent,
                   'stage1': 'sync soft (learner on A_soft, both updated)',
                   'stage2': 'alternating hard (learner on A_hard, generator soft frozen-learner)',
                   'data_meta': meta},
        'losses': rec,
        'recon': recon,
        'recon_hard_05': h05,
        'dynamics_hard': dyn,
        'autoreg_hard': ar_hard,
        'perk': perk,
        'matrices': {'true_adj': true_adj.astype(float).tolist(),
                     'soft_adj': soft.astype(float).tolist(),
                     'hard_adj': hard.astype(float).tolist()},
        'weights': {'model': f'models/{tag}_model.pt'},
    }
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'models', f'{tag}_model.pt'))
    out = os.path.join(RESULT_DIR, f'{tag}.json')
    save_json(result, out)
    print(f'[done] {net} two-stage: ACC(hard)={h05["acc"]:.4f} TPR={h05["tpr"]:.4f} '
          f'FPR={h05["fpr"]:.4f} edges={h05["n_edges_pred"]}/{h05["n_edges_true"]} '
          f'acc_dyn={dyn["acc_dyn"]:.4f} MAE_I(hard)={ar_hard["mae_i"]:.4f} '
          f'beta_fit={perk["beta_fit"]} -> {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['BA'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--lam-ent', type=float, default=0.005)
    args = ap.parse_args()
    torch.set_num_threads(4)
    for net in args.nets:
        run(net=net, smoke=args.smoke, lam_ent=args.lam_ent)
