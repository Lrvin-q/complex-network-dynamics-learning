# -*- coding: utf-8 -*-
"""
Phase 6 改进实验：多步闭环预测（P>1）——先在 BA 上快速验证。

与基础版 v4 唯一区别：损失从单步 CE 改为 P 步闭环推演损失——
  预测 X^{t+1} 后喂回模型（straight-through 离散化）再预测 X^{t+2}…，
  每步交叉熵求和。假边导致的微小单步偏差随步数放大并传播，多步损失
  对结构错误的梯度远强于单步损失。

其余结构不变：论文式交替 + 硬图学习器(0.5阈值) + 双通道学习器
  + 指数降温 3->0.1 + 仅熵正则。

用法：
    python -u phase6_multistep.py --smoke        # 冒烟（2 epoch 测速）
    python -u phase6_multistep.py --P 3 --epochs 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import set_seed, sir_simulation, save_json  # noqa: E402
from dgsl_common import (  # noqa: E402
    GGNModel, nx_to_numpy, soft_adj_metrics, dynamics_metrics,
    autoregressive_mae, per_k_beta_fit, _struct_reg, DEVICE,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
NET = 'BA'
BETA, GAMMA, T_MAX = 0.3, 0.1, 60
N_SEEDS, N_REPEATS = 30, 5


class MultiStepDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return torch.tensor(self.xs[i]), torch.tensor(self.ys[i])


def generate_multi_data(P):
    """BA 序列数据：每条轨迹取 (P+1) 步窗口 → (x0, y_seq[P,N,3])。"""
    from common import generate_network
    set_seed(21)
    e_bi, e_dir, G = generate_network('BA', N=100, m=3)
    n_nodes = G.number_of_nodes()
    true_adj = nx_to_numpy(G).astype(np.float32)
    set_seed(21)
    xs, ys = [], []
    for _ in range(N_SEEDS):
        init = int(np.random.randint(0, n_nodes))
        for _ in range(N_REPEATS):
            states = sir_simulation(e_dir, BETA, GAMMA, init, T_MAX)   # (T_MAX+1, n)
            for t in range(T_MAX - P + 1):
                xs.append(np.eye(3)[states[t]].astype(np.float32))
                ys.append(np.eye(3)[states[t + 1:t + P + 1]].astype(np.float32))
    xs = np.stack(xs)
    ys = np.stack(ys)
    rng = np.random.RandomState(21)
    idx = np.arange(len(xs))
    rng.shuffle(idx)
    nv = int(0.2 * len(idx))
    return (true_adj, xs[idx[nv:]], ys[idx[nv:]], xs[idx[:nv]], ys[idx[:nv]],
            n_nodes, {'n_train': len(idx[nv:]), 'n_val': nv, 'n_windows': len(idx)})


def closed_loop_logits(learner, x0, A, P):
    """P 步闭环推演（straight-through 离散化），返回每步 logits 列表。"""
    logs = []
    x = x0
    for _ in range(P):
        logits = learner(x, A)
        probs = F.softmax(logits, dim=-1)
        hard = F.one_hot(probs.argmax(-1), 3).float()
        x = hard + probs - probs.detach()   # ST：前向离散、反向软梯度
        logs.append(logits)
    return logs


def train_multistep(model, train_loader, val_loader, P, epochs, S_d_rounds, S_n,
                    lr_learner, lr_gen, lam_ent, verbose_every=10, val_every=5):
    opt_l = torch.optim.Adam(model.learner.parameters(), lr=lr_learner)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_gen)
    criterion = nn.CrossEntropyLoss()
    model.generator.set_schedule(epochs * S_n)   # 降温仅在生成器阶段

    def cycle(loader):
        while True:
            for b in loader:
                yield b

    it_g = iter(cycle(train_loader))
    rec = {'epochs': [], 'train_learner': [], 'train_gen': [], 'val': [],
           'tau': [], 'a_mean': [], 'a_std': []}
    best_val, best_state = float('inf'), None

    for epoch in range(epochs):
        t0 = time.time()
        ep_l = ep_g = 0.0
        A_hard = model.generator.get_hard_adj(noise=True)

        # ---- 学习器阶段：S_d_rounds 轮 × 全 loader，固定硬图，P 步闭环 ----
        for _ in range(S_d_rounds):
            for x0, ys in train_loader:
                x0, ys = x0.to(DEVICE), ys.to(DEVICE)
                logs = closed_loop_logits(model.learner, x0, A_hard, P)
                loss = sum(criterion(l.reshape(-1, 3), ys[:, p].argmax(-1).reshape(-1))
                           for p, l in enumerate(logs))
                opt_l.zero_grad(); loss.backward(); opt_l.step()
                ep_l += loss.item()

        # ---- 生成器阶段：S_n 批，软图带梯度，P 步闭环 + 熵正则 ----
        for _ in range(S_n):
            x0, ys = next(it_g)
            x0, ys = x0.to(DEVICE), ys.to(DEVICE)
            A_soft = model.generator.get_adj(training=True)
            logs = closed_loop_logits(model.learner, x0, A_soft, P)
            loss = sum(criterion(l.reshape(-1, 3), ys[:, p].argmax(-1).reshape(-1))
                       for p, l in enumerate(logs))
            if lam_ent > 0:
                loss = loss + lam_ent * (-(A_soft * torch.log(A_soft.clamp(1e-6, 1)) +
                                           (1 - A_soft) * torch.log((1 - A_soft).clamp(1e-6, 1)))).mean()
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            model.generator.anneal()
            ep_g += loss.item()

        # ---- 验证（确定性硬图，单步 CE，与 baseline 可比） ----
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            model.eval()
            vloss, nv = 0.0, 0
            with torch.no_grad():
                A_val = model.generator.get_hard_adj(noise=False)
                for x0, ys in val_loader:
                    x0, ys = x0.to(DEVICE), ys.to(DEVICE)
                    logits = model.learner(x0, A_val)
                    vloss += criterion(logits.reshape(-1, 3), ys[:, 0].argmax(-1).reshape(-1)).item() * x0.size(0)
                    nv += x0.size(0)
            model.train()
            avg_v = vloss / max(nv, 1)
            A_s = model.generator.get_adj(training=False).numpy()
            rec['epochs'].append(epoch + 1)
            rec['train_learner'].append(ep_l / max(S_d_rounds * len(train_loader), 1))
            rec['train_gen'].append(ep_g / max(S_n, 1))
            rec['val'].append(avg_v)
            rec['tau'].append(float(model.generator.tau.item()))
            rec['a_mean'].append(float(A_s.mean()))
            rec['a_std'].append(float(A_s.std()))
            if avg_v < best_val:
                best_val = avg_v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % verbose_every == 0:
                print(f'[ep {epoch+1}/{epochs}] val={avg_v:.4f} l={ep_l:.3f} g={ep_g:.3f} '
                      f'tau={rec["tau"][-1]:.3f} A={rec["a_mean"][-1]:.3f} '
                      f'time={time.time()-t0:.1f}s', flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    rec['best_val'] = float(best_val)
    return rec


def run(P, epochs, S_d_rounds, batch_size, smoke=False):
    os.makedirs(RESULT_DIR, exist_ok=True)
    true_adj, tx, ty, vx, vy, n_nodes, meta = generate_multi_data(P)
    train_loader = DataLoader(MultiStepDataset(tx, ty), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MultiStepDataset(vx, vy), batch_size=batch_size, shuffle=False)

    model = GGNModel(n_nodes, learner='twochannel', hidden=64,
                     tau_init=3.0, tau_min=0.1, eps=1.0, anneal_mode='exp')
    rec = train_multistep(model, train_loader, val_loader, P,
                          epochs=epochs, S_d_rounds=S_d_rounds, S_n=20,
                          lr_learner=1e-3, lr_gen=0.05, lam_ent=0.005,
                          verbose_every=max(1, epochs // 10))

    # ---- 评估（与 phase6_basic 同口径） ----
    soft = model.get_adj().numpy()
    np.fill_diagonal(soft, 0)
    hard = (soft > 0.5).astype(np.float32)
    np.fill_diagonal(hard, 0)
    recon = soft_adj_metrics(true_adj, soft)
    h05 = recon['hard_at_0.5']
    # 单步 acc_dyn/CE（硬图，与 baseline 同口径；多步 loader 取 y_seq 第一帧）
    model.eval()
    correct = total = 0
    ce_sum = 0.0
    with torch.no_grad():
        A_t = torch.tensor(hard, dtype=torch.float32)
        for x0, ys in val_loader:
            logits = model.learner(x0, A_t)
            tgt = ys[:, 0].argmax(-1)
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

    tag = f'dgsl_multi_BA_P{P}'
    result = {
        'config': {'experiment': 'phase6_multistep', 'net_type': 'BA',
                   'P': P, 'epochs': epochs, 'S_d_rounds': S_d_rounds,
                   'S_n': 20, 'batch_size': batch_size, 'lr_learner': 1e-3,
                   'lr_gen': 0.05, 'tau_init': 3.0, 'tau_min': 0.1,
                   'anneal_mode': 'exp', 'learner': 'twochannel', 'hidden': 64,
                   'seed': 21, 'lam_ent': 0.005, 'strategy': 'alternating',
                   'hard_learner': True, 'data_meta': meta},
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
    print(f'[done] BA P={P}: ACC(hard)={h05["acc"]:.4f} TPR={h05["tpr"]:.4f} '
          f'FPR={h05["fpr"]:.4f} edges={h05["n_edges_pred"]}/{h05["n_edges_true"]} '
          f'acc_dyn={dyn["acc_dyn"]:.4f} MAE_I(hard)={ar_hard["mae_i"]:.4f} '
          f'beta_fit={perk["beta_fit"]} -> {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--P', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.smoke:
        run(P=2, epochs=2, S_d_rounds=1, batch_size=16, smoke=True)
        print('SMOKE OK')
    else:
        run(P=args.P, epochs=args.epochs, S_d_rounds=args.rounds,
            batch_size=args.batch, smoke=False)
