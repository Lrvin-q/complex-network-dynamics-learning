# -*- coding: utf-8 -*-
"""
Phase 6 L2：Teacher-forcing 多步训练（P>1，真实状态回馈，无闭环漂移）。

与 v4 的唯一区别：损失从单步 CE 改为 P 步 teacher-forcing 求和——
  第 p 步输入为真实状态 X^{t+p-1}（非模型预测），假边在每一步都被惩罚，
  结构梯度信号约为单步的 P 倍。理论依据：L0 诊断表明 BA 是"判别问题"
  （真假边得分混叠），需增强训练端结构信号；闭环多步（坑17）因 ST 采样
  漂移失败，teacher-forcing 无此问题。

其余结构不变：论文式交替 + 硬图学习器(0.5阈值) + 双通道 + 指数降温 3->0.1
  + 仅熵正则。评估含 0.5 阈值与 valCE 无监督校准（L0）。

用法：
    python -u phase6_teacherforcing.py --P 3 --epochs 50 --smoke
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
from common import set_seed, sir_simulation, save_json, generate_network  # noqa: E402
from dgsl_common import (  # noqa: E402
    GGNModel, nx_to_numpy, soft_adj_metrics, autoregressive_mae, per_k_beta_fit,
    DEVICE,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'phase6')
BETA, GAMMA, T_MAX = 0.3, 0.1, 60
N_SEEDS, N_REPEATS = 30, 5
NET_PARAMS = {'ER': ('ER', {'p': 0.1}), 'WS': ('WS', {'k': 6, 'rewiring_p': 0.2}),
              'BA': ('BA', {'m': 3})}


class MultiStepDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return torch.tensor(self.xs[i]), torch.tensor(self.ys[i])


def generate_multi_data(net, P):
    kind, params = NET_PARAMS[net]
    set_seed(21)
    e_bi, e_dir, G = generate_network(kind, N=100, **params)
    n_nodes = G.number_of_nodes()
    true_adj = nx_to_numpy(G).astype(np.float32)
    set_seed(21)
    xs, ys = [], []
    for _ in range(N_SEEDS):
        init = int(np.random.randint(0, n_nodes))
        for _ in range(N_REPEATS):
            states = sir_simulation(e_dir, BETA, GAMMA, init, T_MAX)
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


def train_tf(model, train_loader, val_loader, P, epochs, S_d_rounds, S_n,
             lr_learner, lr_gen, lam_ent, verbose_every=10, val_every=5):
    opt_l = torch.optim.Adam(model.learner.parameters(), lr=lr_learner)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=lr_gen)
    criterion = nn.CrossEntropyLoss()
    model.generator.set_schedule(epochs * S_n)

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

        # ---- 学习器阶段：硬图，P 步 teacher-forcing ----
        for _ in range(S_d_rounds):
            for x0, ys in train_loader:
                x0, ys = x0.to(DEVICE), ys.to(DEVICE)
                loss = torch.tensor(0.0, device=DEVICE)
                x = x0
                for p in range(P):
                    logits = model.learner(x, A_hard)
                    loss = loss + criterion(logits.reshape(-1, 3),
                                            ys[:, p].argmax(-1).reshape(-1))
                    x = ys[:, p]   # teacher-forcing：用真实状态
                opt_l.zero_grad(); loss.backward(); opt_l.step()
                ep_l += loss.item()

        # ---- 生成器阶段：软图带梯度，P 步 teacher-forcing + 熵正则 ----
        for _ in range(S_n):
            x0, ys = next(it_g)
            x0, ys = x0.to(DEVICE), ys.to(DEVICE)
            A_soft = model.generator.get_adj(training=True)
            loss = torch.tensor(0.0, device=DEVICE)
            x = x0
            for p in range(P):
                logits = model.learner(x, A_soft)
                loss = loss + criterion(logits.reshape(-1, 3),
                                        ys[:, p].argmax(-1).reshape(-1))
                x = ys[:, p]
            if lam_ent > 0:
                A_c = A_soft.clamp(1e-6, 1 - 1e-6)
                loss = loss + lam_ent * (-(A_c * torch.log(A_c) +
                                           (1 - A_c) * torch.log(1 - A_c))).mean()
            opt_g.zero_grad(); loss.backward(); opt_g.step()
            model.generator.anneal()
            ep_g += loss.item()

        # ---- 验证（确定性硬图，单步 CE 可比） ----
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            model.eval()
            vloss, nv = 0.0, 0
            with torch.no_grad():
                A_val = model.generator.get_hard_adj(noise=False)
                for x0, ys in val_loader:
                    x0, ys = x0.to(DEVICE), ys.to(DEVICE)
                    logits = model.learner(x0, A_val)
                    vloss += criterion(logits.reshape(-1, 3),
                                       ys[:, 0].argmax(-1).reshape(-1)).item() * x0.size(0)
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


def run(net, P, epochs, S_d_rounds, batch_size, smoke=False):
    os.makedirs(RESULT_DIR, exist_ok=True)
    true_adj, tx, ty, vx, vy, n_nodes, meta = generate_multi_data(net, P)
    train_loader = DataLoader(MultiStepDataset(tx, ty), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MultiStepDataset(vx, vy), batch_size=batch_size, shuffle=False)

    model = GGNModel(n_nodes, learner='twochannel', hidden=64,
                     tau_init=3.0, tau_min=0.1, eps=1.0, anneal_mode='exp')
    rec = train_tf(model, train_loader, val_loader, P,
                   epochs=epochs, S_d_rounds=S_d_rounds, S_n=20,
                   lr_learner=1e-3, lr_gen=0.05, lam_ent=0.005,
                   verbose_every=max(1, epochs // 10))

    # ---- 评估：0.5 阈值 + valCE 无监督校准（L0） ----
    soft = model.get_adj().numpy()
    np.fill_diagonal(soft, 0)
    hard05 = (soft > 0.5).astype(np.float32)
    np.fill_diagonal(hard05, 0)
    recon = soft_adj_metrics(true_adj, soft)
    h05 = recon['hard_at_0.5']

    # valCE 选 K（无监督校准）
    iu = np.triu_indices(n_nodes, 1)
    y_true = true_adj[iu].astype(int)
    y_score = soft[iu]
    order = np.argsort(y_score)[::-1]
    model.eval()
    best_ce, best_K = float('inf'), int((y_score > 0.5).sum())
    with torch.no_grad():
        for K in (list(range(100, 2501, 100)) + [int((y_score > 0.5).sum())]):
            hard = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            hard[iu[0][order[:K]], iu[1][order[:K]]] = 1.0
            hard[iu[1][order[:K]], iu[0][order[:K]]] = 1.0
            A_t = torch.tensor(hard, device=DEVICE)
            ce, nv = 0.0, 0
            for x, y in val_loader:
                logits = model.learner(x.to(DEVICE), A_t)
                ce += F.cross_entropy(logits.reshape(-1, 3),
                                      y[:, 0].argmax(-1).reshape(-1).to(DEVICE),
                                      reduction='sum').item()
                nv += y.size(0) * y.size(1)
            ce = ce / max(nv, 1)
            if ce < best_ce:
                best_ce, best_K = float(ce), K
    hard_cal = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    hard_cal[iu[0][order[:best_K]], iu[1][order[:best_K]]] = 1.0
    hard_cal[iu[1][order[:best_K]], iu[0][order[:best_K]]] = 1.0
    np.fill_diagonal(hard_cal, 0)
    cal_metrics = soft_adj_metrics(true_adj, hard_cal)['hard_at_0.5']
    cal_metrics = {k: (v if not isinstance(v, np.generic) else v.item())
                   for k, v in cal_metrics.items()}
    cal_metrics['K'] = best_K

    # 动力学：acc_dyn（0.5 硬图）、多初始自回归（0.5 与校准硬图）
    correct = total = 0
    with torch.no_grad():
        A_t = torch.tensor(hard05, dtype=torch.float32)
        for x0, ys in val_loader:
            logits = model.learner(x0.to(DEVICE), A_t)
            tgt = ys[:, 0].argmax(-1)
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

    tag = f'dgsl_tf_{net}_P{P}'
    result = {
        'config': {'experiment': 'phase6_teacherforcing', 'net_type': net,
                   'P': P, 'epochs': epochs, 'S_d_rounds': S_d_rounds, 'S_n': 20,
                   'batch_size': batch_size, 'lr_learner': 1e-3, 'lr_gen': 0.05,
                   'tau_init': 3.0, 'tau_min': 0.1, 'anneal_mode': 'exp',
                   'learner': 'twochannel', 'hidden': 64, 'seed': 21,
                   'lam_ent': 0.005, 'strategy': 'alternating',
                   'hard_learner': True, 'loss': f'teacher_forcing_P{P}',
                   'data_meta': meta},
        'losses': rec,
        'recon': recon,
        'recon_hard_05': h05,
        'recon_calibrated': cal_metrics,
        'dynamics_hard': dyn,
        'autoreg_hard': ar_hard,
        'perk': perk,
        'matrices': {'true_adj': true_adj.astype(float).tolist(),
                     'soft_adj': soft.astype(float).tolist(),
                     'hard_adj': hard05.astype(float).tolist(),
                     'hard_calibrated': hard_cal.astype(float).tolist()},
        'weights': {'model': f'models/{tag}_model.pt'},
    }
    torch.save(model.state_dict(), os.path.join(RESULT_DIR, 'models', f'{tag}_model.pt'))
    out = os.path.join(RESULT_DIR, f'{tag}.json')
    save_json(result, out)
    print(f'[done] {net} P={P}: ACC(0.5)={h05["acc"]:.4f} '
          f'edges={h05["n_edges_pred"]}/{h05["n_edges_true"]} | '
          f'valCE校准: K={best_K} ACC={cal_metrics["acc"]:.4f} '
          f'TPR={cal_metrics["tpr"]:.4f} FPR={cal_metrics["fpr"]:.4f} | '
          f'acc_dyn={dyn["acc_dyn"]:.4f} MAE_I={ar_hard["mae_i"]:.4f} '
          f'beta_fit={perk["beta_fit"]} -> {out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', default='BA')
    ap.add_argument('--P', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(4)
    run(args.net, P=2 if args.smoke else args.P,
        epochs=2 if args.smoke else args.epochs,
        S_d_rounds=1 if args.smoke else args.rounds,
        batch_size=16 if args.smoke else args.batch, smoke=args.smoke)
