# -*- coding: utf-8 -*-
"""
DGSL（GGN 式）公共模块：Gumbel 生成器 + GNN 动力学学习器 + 交替训练 + 评估指标。

设计依据：Zhang Zhang et al., "A general deep learning framework for network
reconstruction and dynamics learning"（GGN, 2019）——模型无关、数据驱动：
  - 网络生成器：Gumbel-Softmax 可微采样（论文式 (1)，本实现用 Logistic 噪声的对数几率形式，
    即二分类 Gumbel-Softmax 的精确等价），对称化 + 禁自环 + 温度退火（τ→0）；
  - 动力学学习器：论文四映射 GNN（Node→Edge 成对拼接 → Edge→Edge → Edge→Node 软掩码求和
    → Node→Node），输入独热状态、输出 3 类 logits，交叉熵；可选 TwoChannel 变体（复用
    第一阶段架构，软邻接矩阵乘法聚合，O(N) 内存）；
  - 交替训练：论文 Algorithm 1（每 epoch 先学习器 S_d 轮、再生成器 S_n 轮）。

遵循 DSH.md：计算→保存数据→绘图分离；所有结果 JSON 带完整 config；训练后保存权重；
后台运行一律 python -u；CJK 字体配置。
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    set_seed, generate_network, sir_simulation, SIRDataset, save_json, fit_beta,
)

# ---------------- matplotlib CJK 配置（坑 9） ----------------
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device('cpu')


# =====================================================================
# 一、网络生成器（Gumbel-Softmax，论文式 (1)）
# =====================================================================
def logistic_noise(shape, device):
    """Logistic(0,1) 噪声 = 两个独立 Gumbel 之差（二分类 Gumbel-Softmax 的精确噪声形式）。"""
    u = torch.rand(shape, device=device).clamp_min(1e-8)
    return torch.log(u / (1.0 - u))


class GumbelGenerator(nn.Module):
    """可训练对称邻接矩阵生成器：A_ij = sigmoid((R_sym,ij + eps*G_ij)/tau)。

    - R 为 N×N 可训练对数几率；每次前向对称化 R_sym = (R + R^T)/2；
    - 对角（自环）通过掩码强制为 0；
    - G 为 Logistic 噪声，对无向点对共享同一噪声（上三角采样后镜像），保证 A 严格对称；
    - tau 随全局优化步数线性退火 tau_init -> tau_min（论文 τ→0 思想）。
    """

    def __init__(self, N, tau_init=1.0, tau_min=0.1, eps=1.0, r_std=0.5,
                 anneal_mode='linear'):
        super().__init__()
        self.N = N
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.eps = eps
        self.anneal_mode = anneal_mode  # 'linear' | 'exp'（论文式指数衰减）
        raw = torch.randn(N, N) * r_std
        raw = (raw + raw.t()) / 2.0
        self.R = nn.Parameter(raw)
        self.register_buffer('tau', torch.tensor(tau_init, dtype=torch.float32))
        self.global_step = 0
        self.total_steps = 1

    def set_schedule(self, total_steps):
        self.total_steps = max(int(total_steps), 1)

    def reset_schedule(self, tau_init, tau_min, total_steps):
        """两阶段训练用：阶段切换时重置温度区间与步数规划。"""
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.tau.fill_(tau_init)
        self.global_step = 0
        self.total_steps = max(int(total_steps), 1)

    def anneal(self):
        """按全局优化步数退火温度；每调用一次全局步数 +1。

        linear: τ = τ_init + (τ_min - τ_init)·frac
        exp（论文式）: τ = τ_init · (τ_min/τ_init)^frac，如 10→0.1 的指数衰减。
        """
        frac = min(self.global_step / self.total_steps, 1.0)
        if self.anneal_mode == 'exp':
            t = self.tau_init * (self.tau_min / self.tau_init) ** frac
        else:
            t = self.tau_init + (self.tau_min - self.tau_init) * frac
        self.tau.fill_(t)
        self.global_step += 1

    def get_adj(self, training=True):
        R = (self.R + self.R.t()) / 2.0
        if training:
            g = logistic_noise(R.shape, R.device)
            g = torch.triu(g, 1)          # 上三角噪声
            g = g + g.t()                 # 镜像：同一无向点对同一噪声
            A = torch.sigmoid((R + self.eps * g) / self.tau)
        else:
            with torch.no_grad():
                A = torch.sigmoid(R / self.tau.clamp(min=1e-3))
        mask = 1.0 - torch.eye(self.N, device=A.device)
        return (A * mask).clamp(1e-6, 1.0 - 1e-6)

    def get_hard_adj(self, noise=True):
        """0.5 阈值硬图化（= 二分类 Gumbel-Softmax 的 argmax 采样）。

        noise=True：从带噪软采样（σ((R+εG)/τ)）阈值化（论文 sample(hard=True)）；
        noise=False：确定性软图 σ(R/τ) 阈值化（评估用）。
        返回无梯度的 0/1 对称矩阵。
        """
        A = self.get_adj(training=noise)
        return (A > 0.5).float().detach()

    def soft_entropy(self):
        """软邻接矩阵的逐元素熵（0 表示完全确定，ln2≈0.693 表示完全不确定）。"""
        A = self.get_adj(training=False).detach()
        h = -(A * torch.log(A) + (1 - A) * torch.log(1 - A))
        return h.mean().item()


# =====================================================================
# 二、动力学学习器
# =====================================================================
class GNNDynamics(nn.Module):
    """论文 §6.3 四映射 GNN（NRI 式）：Node→Edge / Edge→Edge / Edge→Node / Node→Node。

    forward(x, A): x (B,N,3) 独热状态, A (N,N) 软/硬邻接矩阵 -> logits (B,N,3)。
    软掩码求和：H_v1[i] = Σ_j A_ij · H_e2[i,j]（有序点对，与论文一致）。
    """

    def __init__(self, in_dim=3, hidden=64, out_dim=3, use_fe=True):
        super().__init__()
        self.f_ne = nn.Sequential(nn.Linear(in_dim * 2, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.f_e = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU()) if use_fe else nn.Identity()
        self.f_nu = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.f_out = nn.Linear(hidden, out_dim)

    def forward(self, x, A):
        B, N, _ = x.shape
        xi = x.unsqueeze(2).expand(B, N, N, -1)   # 行 i
        xj = x.unsqueeze(1).expand(B, N, N, -1)   # 列 j
        he1 = torch.cat([xi, xj], dim=-1)         # (B,N,N,2in) 成对拼接
        he2 = self.f_e(self.f_ne(he1))            # (B,N,N,h)
        hv1 = torch.einsum('bije,ij->bie', he2, A)  # 软掩码求和
        hv2 = self.f_nu(hv1)
        return self.f_out(hv2)


class TwoChannelDynamics(nn.Module):
    """第一阶段"自身/邻居双通道"架构的软邻接矩阵版（5B3 优化改进）。

    自身通道：仅自身变换；邻居通道：Σ_j A_ij · h(j)（软求和，保留感染邻居数量信息）；
    读出前拼接。计算量 O(N) 内存，远小于论文 GNN 的 O(N^2)。
    """

    def __init__(self, in_dim=3, hidden=64, out_dim=3):
        super().__init__()
        self.self_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden), nn.ReLU())
        self.neigh_mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                       nn.Linear(hidden, hidden), nn.ReLU())
        self.readout = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                     nn.Linear(hidden, out_dim))

    def forward(self, x, A):
        hs = self.self_mlp(x)                              # (B,N,h)
        hn = torch.einsum('bje,ij->bie', self.neigh_mlp(x), A)  # (B,N,h)
        return self.readout(torch.cat([hs, hn], dim=-1))


class GGNModel(nn.Module):
    """GGN 端到端模型：Gumbel 生成器 + 动力学学习器。"""

    def __init__(self, N, learner='gnn', hidden=64, tau_init=1.0, tau_min=0.1, eps=1.0,
                 use_fe=True, anneal_mode='linear'):
        super().__init__()
        self.N = N
        self.generator = GumbelGenerator(N, tau_init=tau_init, tau_min=tau_min, eps=eps,
                                         anneal_mode=anneal_mode)
        if learner == 'gnn':
            self.learner = GNNDynamics(3, hidden, 3, use_fe=use_fe)
        elif learner == 'twochannel':
            self.learner = TwoChannelDynamics(3, hidden, 3)
        else:
            raise ValueError(f'unknown learner: {learner}')
        self.learner_name = learner

    def forward(self, x, detach_A=False):
        if detach_A:
            with torch.no_grad():
                A = self.generator.get_adj(training=True)
            A = A.detach()
        else:
            A = self.generator.get_adj(training=True)
        return self.learner(x, A), A

    def get_adj(self):
        return self.generator.get_adj(training=False)

    def predict_logits(self, x, A=None):
        """推理：给定 (B,N,3) 状态与邻接矩阵（默认用生成器当前软图），返回 logits。"""
        if A is None:
            A = self.get_adj()
        with torch.no_grad():
            return self.learner(x, A)


# =====================================================================
# 三、数据生成（复用 common.py；存 npz 供所有阶段共享同一份数据）
# =====================================================================
def get_network_config(net_type):
    return {'net_type': net_type, 'N': 100, 'p': 0.1, 'm': 3, 'k': 6, 'rewiring_p': 0.2}


def generate_and_save_data(cfg, save_dir, beta=0.3, gamma=0.1, T=60,
                           n_seeds=30, n_repeats=5, seed=21):
    """生成真实网络 + SIR 轨迹数据并保存 npz（幂等：已存在则直接加载）。

    返回 (true_adj, edge_index, train_pairs, val_pairs, meta)。
    """
    net_type = cfg['net_type']
    N = cfg['N']
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f'data_{net_type}.npz')
    if os.path.exists(npz_path):
        z = np.load(npz_path, allow_pickle=True)
        return (z['true_adj'], z['edge_index'], z['train_x'], z['train_y'],
                z['val_x'], z['val_y'], dict(z['meta'].item()))

    set_seed(seed)
    if net_type == 'ER':
        edge_index_bi, edge_index_dir, G = generate_network('ER', N=N, p=cfg['p'])
    elif net_type == 'BA':
        edge_index_bi, edge_index_dir, G = generate_network('BA', N=N, m=cfg['m'])
    elif net_type == 'WS':
        edge_index_bi, edge_index_dir, G = generate_network('WS', N=N, k=cfg['k'], rewiring_p=cfg['rewiring_p'])
    else:
        raise ValueError(net_type)
    true_adj = np.asarray(nx_to_numpy(G), dtype=np.float32)
    edge_index = edge_index_dir  # 每条无向边一次；sir_simulation 内部去重构建邻接表

    # 训练数据：n_seeds 初始节点 × n_repeats 重复，每条轨迹 T 步 → (X_t, X_{t+1}) 对
    set_seed(seed)
    pairs_x, pairs_y = [], []
    for _ in range(n_seeds):
        initial_node = int(np.random.randint(0, true_adj.shape[0]))
        for _ in range(n_repeats):
            states = sir_simulation(edge_index, beta, gamma, initial_node, T)
            for t in range(T):
                pairs_x.append(np.eye(3)[states[t]].astype(np.float32))
                pairs_y.append(np.eye(3)[states[t + 1]].astype(np.float32))
    pairs_x = np.stack(pairs_x)
    pairs_y = np.stack(pairs_y)
    n = len(pairs_x)
    n_val = int(0.2 * n)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    meta = {
        'net_type': net_type, 'N': N, 'n_nodes': int(true_adj.shape[0]),
        'p': cfg['p'], 'm': cfg['m'], 'k': cfg['k'],
        'rewiring_p': cfg['rewiring_p'], 'beta': beta, 'gamma': gamma, 'T': T,
        'n_seeds': n_seeds, 'n_repeats': n_repeats, 'seed': seed,
        'n_pairs': n, 'n_train': len(train_idx), 'n_val': n_val,
        'n_edges': int(true_adj.sum() / 2),
    }
    np.savez_compressed(npz_path, true_adj=true_adj, edge_index=edge_index.numpy(),
                        train_x=pairs_x[train_idx], train_y=pairs_y[train_idx],
                        val_x=pairs_x[val_idx], val_y=pairs_y[val_idx], meta=np.array(meta))
    return true_adj, edge_index, pairs_x[train_idx], pairs_y[train_idx], \
        pairs_x[val_idx], pairs_y[val_idx], meta


def nx_to_numpy(G):
    import networkx as nx
    return nx.to_numpy_array(G)


def build_loaders_from_arrays(train_x, train_y, val_x, val_y, batch_size=32):
    train_ds = SIRDataset([(train_x[i], train_y[i]) for i in range(len(train_x))])
    val_ds = SIRDataset([(val_x[i], val_y[i]) for i in range(len(val_x))])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# =====================================================================
# 四、训练（交替训练，论文 Algorithm 1）
# =====================================================================
def _struct_reg(A, lam_sp, lam_ent):
    """结构正则化（退化时才开启）：稀疏惩罚 + 熵惩罚。A 为带噪软邻接矩阵。"""
    A_c = A.clamp(1e-6, 1.0 - 1e-6)
    reg = lam_sp * A_c.mean()
    if lam_ent > 0:
        h = -(A_c * torch.log(A_c) + (1 - A_c) * torch.log(1 - A_c))
        reg = reg + lam_ent * h.mean()
    return reg


def train_ggn(model, train_loader, val_loader, epochs=100, S_d=10, S_n=20,
              lr_learner=1e-3, lr_gen=5e-3, strategy='alternating',
              save_dir=None, tag='', verbose_every=10, val_every=5,
              lam_sp=0.0, lam_ent=0.0, st_generator=False, hard_learner=False,
              S_d_rounds=0):
    """训练 GGN。

    strategy='alternating'：每 epoch 先学习器阶段、再生成器阶段（论文 Algorithm 1）。
    hard_learner=True（论文式）：每 epoch 采样一张 0.5 阈值硬图，学习器阶段固定用硬图。
    S_d_rounds>0：学习器阶段跑 S_d_rounds 轮 × 全 loader（论文 Dyn_Steps 语义），
                  否则跑 S_d 批。
    """
    model.train()
    opt_learner = torch.optim.Adam(model.learner.parameters(), lr=lr_learner)
    opt_gen = torch.optim.Adam(model.generator.parameters(), lr=lr_gen)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = (S_d + S_n) if strategy == 'alternating' else (S_d + S_n)
    # 论文式硬图模式：温度只在生成器阶段衰减（drop_temperature 论文仅在生成器批调用）
    model.generator.set_schedule(epochs * S_n if hard_learner else epochs * steps_per_epoch)

    rec = {'epochs': [], 'train_learner': [], 'train_gen': [], 'val': [],
           'tau': [], 'a_mean': [], 'a_std': [], 'a_entropy': [], 'auc_soft': []}
    best_val = float('inf')
    best_state = None

    def cycle_loader(loader):
        while True:
            for batch in loader:
                yield batch

    it_learner = iter(cycle_loader(train_loader))
    it_gen = iter(cycle_loader(train_loader))
    epoch_times = []

    import time
    for epoch in range(epochs):
        t0 = time.time()
        ep_loss_l, ep_loss_g, n_l, n_g = 0.0, 0.0, 0, 0

        if strategy == 'alternating':
            # 论文式硬图模式：每 epoch 采样一张硬图（0.5 阈值，无梯度），
            # 整个学习器阶段固定使用（论文 train_bn.py：adj = sample(hard=True)）
            A_hard = None
            if hard_learner:
                A_hard = model.generator.get_hard_adj(noise=True)
            # ---- 学习器阶段：论文式 S_d_rounds 轮 × 全 loader；否则 S_d 批 ----
            #   （生成器冻结；硬图模式用固定硬图，否则软图截断梯度）
            if S_d_rounds > 0:
                for _ in range(S_d_rounds):
                    for x, y in train_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        if hard_learner:
                            logits = model.learner(x, A_hard)
                        else:
                            with torch.no_grad():
                                A_f = model.generator.get_adj(training=True).detach()
                            logits = model.learner(x, A_f)
                        loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
                        opt_learner.zero_grad(); loss.backward(); opt_learner.step()
                        if not hard_learner:
                            model.generator.anneal()
                        ep_loss_l += loss.item(); n_l += 1
            else:
                for _ in range(S_d):
                    x, y = next(it_learner)
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    if hard_learner:
                        logits = model.learner(x, A_hard)
                    else:
                        logits, _ = model(x, detach_A=True)
                    loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
                    opt_learner.zero_grad(); loss.backward(); opt_learner.step()
                    if not hard_learner:
                        model.generator.anneal()
                    ep_loss_l += loss.item(); n_l += 1
            # ---- 生成器阶段：S_n 轮（学习器冻结，软图带梯度；论文式：降温仅在此阶段） ----
            for _ in range(S_n):
                x, y = next(it_gen)
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, A = model(x, detach_A=False)
                if st_generator:
                    # 直通估计：前向用硬 0/1 邻接（生成器直接优化离散结构的预测损失），
                    # 反向仍走软邻接梯度
                    A_use = (A > 0.5).float() + A - A.detach()
                    logits = model.learner(x, A_use)
                loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
                if lam_sp > 0 or lam_ent > 0:
                    loss = loss + _struct_reg(A, lam_sp, lam_ent)
                opt_gen.zero_grad(); loss.backward(); opt_gen.step()
                model.generator.anneal()
                ep_loss_g += loss.item(); n_g += 1
        else:
            # ---- 同步训练：每步两组参数同时更新 ----
            for _ in range(S_d + S_n):
                x, y = next(it_learner)
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, A = model(x, detach_A=False)
                loss = criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1))
                if lam_sp > 0 or lam_ent > 0:
                    loss = loss + _struct_reg(A, lam_sp, lam_ent)
                opt_learner.zero_grad(); opt_gen.zero_grad(); loss.backward()
                opt_learner.step(); opt_gen.step()
                model.generator.anneal()
                ep_loss_l += loss.item(); ep_loss_g += loss.item(); n_l += 1; n_g += 1

        # ---- 验证（每 val_every 个 epoch 一次，最后一次必验） ----
        do_val = ((epoch + 1) % val_every == 0) or (epoch + 1 == epochs)
        avg_l = ep_loss_l / max(n_l, 1)
        avg_g = ep_loss_g / max(n_g, 1)
        if do_val:
            model.eval()
            vloss, nv = 0.0, 0
            with torch.no_grad():
                if hard_learner:
                    A_val = model.generator.get_hard_adj(noise=False)  # 确定性硬图
                else:
                    A_val = model.generator.get_adj(training=False)
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = model.learner(x, A_val)
                    vloss += criterion(logits.reshape(-1, 3), y.argmax(-1).reshape(-1)).item() * x.size(0)
                    nv += x.size(0)
            model.train()
            avg_v = vloss / max(nv, 1)
            with torch.no_grad():
                A_soft = model.generator.get_adj(training=False).numpy()
            rec['epochs'].append(epoch + 1)
            rec['train_learner'].append(avg_l)
            rec['train_gen'].append(avg_g)
            rec['val'].append(avg_v)
            rec['tau'].append(float(model.generator.tau.item()))
            rec['a_mean'].append(float(A_soft.mean()))
            rec['a_std'].append(float(A_soft.std()))
            rec['a_entropy'].append(model.generator.soft_entropy())
            if avg_v < best_val:
                best_val = avg_v
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if (epoch + 1) % verbose_every == 0:
                print(f'[epoch {epoch+1}/{epochs}] val={avg_v:.4f} l={avg_l:.4f} g={avg_g:.4f} '
                      f'tau={rec["tau"][-1]:.3f} A(μ,σ)=({rec["a_mean"][-1]:.3f},{rec["a_std"][-1]:.3f}) '
                      f'time={time.time()-t0:.1f}s', flush=True)
        epoch_times.append(time.time() - t0)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f'{tag}_model.pt'))

    rec['best_val'] = float(best_val)
    rec['time_per_epoch'] = float(np.mean(epoch_times))
    rec['strategy'] = strategy
    rec['lam_sp'] = lam_sp
    rec['lam_ent'] = lam_ent
    rec['hard_learner'] = hard_learner
    return rec


# =====================================================================
# 五、评估
# =====================================================================
def triu_mask(N):
    iu = np.triu_indices(N, k=1)
    return iu


def soft_adj_metrics(true_adj, soft_adj):
    """软/硬邻接矩阵重构指标（上三角非对角口径，避免对称重复计数）。

    返回 {auc_soft, auc_hard_at05, acc/TPR/FPR(0.5 阈值), auto_threshold, auc_hard_auto, ...}
    """
    from sklearn.metrics import roc_auc_score
    N = true_adj.shape[0]
    iu = triu_mask(N)
    y_true = true_adj[iu]
    y_score = soft_adj[iu]
    auc_soft = float(roc_auc_score(y_true, y_score))

    # 自动阈值：强度降序 + 最大间隙（在中间 90% 范围内搜索，避免头尾伪间隙）
    order = np.argsort(y_score)[::-1]
    s = y_score[order]
    gaps = s[:-1] - s[1:]
    lo = int(0.05 * len(s)); hi = int(0.95 * len(s))
    if hi <= lo + 1:
        auto_thr = float(np.median(y_score))
    else:
        k = lo + int(np.argmax(gaps[lo:hi - 1]))
        auto_thr = float(s[k])

    def hard_metrics(thr):
        y_pred = (y_score >= thr).astype(int)
        acc = float(np.mean(y_pred == y_true.astype(int)))
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        return {'threshold': thr, 'acc': acc, 'tpr': tpr, 'fpr': fpr,
                'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                'n_edges_pred': int(y_pred.sum()), 'n_edges_true': int(y_true.sum())}

    m05 = hard_metrics(0.5)
    m_auto = hard_metrics(auto_thr)
    m_auto['auc_hard'] = float(roc_auc_score(y_true, (y_score >= auto_thr).astype(int)))
    return {
        'auc_soft': auc_soft,
        'auto_threshold': auto_thr,
        'hard_at_0.5': m05,
        'hard_auto': m_auto,
        'soft_stats': {'mean': float(soft_adj.mean()), 'std': float(soft_adj.std()),
                       'min': float(soft_adj.min()), 'max': float(soft_adj.max()),
                       'weak_edges_005_015': int(((soft_adj > 0.05) & (soft_adj < 0.15)).sum() / 2),
                       'weak_edges_01_09': int(((soft_adj > 0.1) & (soft_adj < 0.9)).sum() / 2)},
        'sorted_triu': [float(v) for v in y_score[np.argsort(y_score)[::-1]]],
    }


def dynamics_metrics(model, val_loader, A):
    """动力学预测指标：ACC(dyn)（总体 + 各类别）、平均交叉熵。A: (N,N) 张量。"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total, correct = 0, 0
    ce_sum = 0.0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model.learner(x, A)
            pred = logits.argmax(-1)
            tgt = y.argmax(-1)
            ce_sum += criterion(logits.reshape(-1, 3), tgt.reshape(-1)).item()
            correct += (pred == tgt).sum().item()
            total += tgt.numel()
            for c in range(3):
                m = (tgt == c)
                class_total[c] += m.sum().item()
                class_correct[c] += (pred[m] == c).sum().item()
    return {
        'acc_dyn': correct / max(total, 1),
        'ce': ce_sum / max(total, 1),
        'acc_class': [class_correct[c] / max(class_total[c], 1) for c in range(3)],
        'class_counts': class_total,
    }


def autoregressive_mae(model, A, initial_state, true_adj, beta, gamma, T=50,
                       n_mc=50, n_mc_true=100, seed=21):
    """自回归评估：模型用给定 A（软/硬）从初始状态蒙特卡洛推演 T 步，
    与真实 SIR 模拟（真实网络）的平均感染者占比曲线比较，返回 MAE(I/S/R)。
    若 initial_state 为 list（多个初始状态），返回各初始状态的平均 MAE 与明细。
    """
    if isinstance(initial_state, (list, tuple)) and len(initial_state) > 0 \
            and not isinstance(initial_state[0], (int, np.integer)):
        per_init = [autoregressive_mae_single(model, A, s, true_adj, beta, gamma,
                                              T, n_mc, n_mc_true, seed)
                    for s in initial_state]
        out = {
            'mae_i': float(np.mean([p['mae_i'] for p in per_init])),
            'mae_i_per_init': [float(p['mae_i']) for p in per_init],
            'true_curve': per_init[0]['true_curve'],
            'pred_curve': [float(np.mean([p['pred_curve'][t] for p in per_init]))
                           for t in range(T + 1)],
        }
        return out
    return autoregressive_mae_single(model, A, initial_state, true_adj, beta, gamma,
                                     T, n_mc, n_mc_true, seed)


def autoregressive_mae_single(model, A, initial_state, true_adj, beta, gamma, T=50,
                              n_mc=50, n_mc_true=100, seed=21):
    N = initial_state.shape[0]
    rng = np.random.RandomState(seed)

    # 真实曲线（修复后的 sir_simulation）
    true_frac_i = np.zeros(T + 1)
    for _ in range(n_mc_true):
        states = sir_simulation_traj(true_adj, beta, gamma, initial_state, T, rng)
        true_frac_i += (states == 1).mean(axis=1)
    true_frac_i /= n_mc_true

    # 模型推演
    A_t = torch.tensor(np.asarray(A, dtype=np.float32), device=DEVICE)
    model.eval()
    pred_frac = np.zeros((n_mc, T + 1))
    with torch.no_grad():
        for m in range(n_mc):
            state = initial_state.copy()
            pred_frac[m, 0] = (state == 1).mean()
            for t in range(T):
                x = np.eye(3)[state].astype(np.float32)
                xt = torch.tensor(x, device=DEVICE).unsqueeze(0)
                logits = model.learner(xt, A_t)
                probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
                state = np.array([rng.choice(3, p=probs[i]) for i in range(N)])
                pred_frac[m, t + 1] = (state == 1).mean()
    pred_mean = pred_frac.mean(axis=0)
    mae_i = float(np.mean(np.abs(pred_mean - true_frac_i)))
    return {'mae_i': mae_i, 'true_curve': [float(v) for v in true_frac_i],
            'pred_curve': [float(v) for v in pred_mean]}


def sir_simulation_traj(true_adj, beta, gamma, initial_state, T, rng):
    """矩阵形式 SIR 模拟（真实网络），返回 (T+1, N) 状态序列。"""
    N = true_adj.shape[0]
    state = initial_state.copy()
    states = [state.copy()]
    for _ in range(T):
        new = state.copy()
        infected = (state == 1)
        k = true_adj @ infected.astype(np.float64)
        s_mask = state == 0
        p_inf = 1.0 - (1.0 - beta) ** k[s_mask]
        infect_now = rng.random(s_mask.sum()) < p_inf
        new[s_mask] = np.where(infect_now, 1, 0)
        i_mask = state == 1
        recover_now = rng.random(i_mask.sum()) < gamma
        new[i_mask] = np.where(recover_now, 2, 1)
        state = new
        states.append(state.copy())
    return np.stack(states)


def per_k_beta_fit(model, A, true_adj, beta_true=0.3, n_states=1500, seed=21):
    """per-k 诊断（可选）：在真实 SIR 采样的状态上，用模型（给定 A）预测 P(S→I)，
    按真实感染者邻居数 k 分桶，拟合 β̂ ~ 1-(1-β)^k，与 β_true 对照。
    返回 {per_k: {k: [count, mean_p]}, beta_fit, r2, beta_true, n_states}。
    """
    N = true_adj.shape[0]
    rng = np.random.RandomState(seed)
    A_t = torch.tensor(np.asarray(A, dtype=np.float32), device=DEVICE)
    buckets = {}
    model.eval()
    with torch.no_grad():
        for _ in range(n_states):
            init = np.zeros(N, dtype=int)
            init[int(rng.randint(0, N))] = 1
            states = sir_simulation_traj(true_adj, 0.3, 0.1, init, 20, rng)
            t = int(rng.randint(0, 21))
            state = states[t]
            s_mask = np.where(state == 0)[0]
            if len(s_mask) == 0:
                continue
            x = np.eye(3)[state].astype(np.float32)
            logits = model.learner(torch.tensor(x, device=DEVICE).unsqueeze(0), A_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            infected = (state == 1).astype(np.float64)
            k_arr = (true_adj @ infected)[s_mask]
            for i, node in enumerate(s_mask):
                k = int(round(k_arr[i]))
                if k <= 0:
                    continue
                p = probs[node, 1]
                b = buckets.setdefault(k, [0, 0.0])
                b[0] += 1
                b[1] += p
    keys = sorted(buckets)
    ks = np.array(keys)
    ps = np.array([buckets[k][1] / buckets[k][0] for k in keys])
    counts = np.array([buckets[k][0] for k in keys])
    beta_fit, r2 = None, None
    if len(ks) >= 2:
        try:
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(lambda k, b: 1 - (1 - b) ** k, ks, ps, p0=[0.3],
                                bounds=(1e-6, 0.999), maxfev=20000)
            beta_fit = float(popt[0])
            pred = 1 - (1 - popt[0]) ** ks
            ss_res = float(((ps - pred) ** 2).sum())
            ss_tot = float(((ps - ps.mean()) ** 2).sum())
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
        except Exception:
            pass
    return {'per_k': {int(k): [int(buckets[k][0]), float(buckets[k][1] / buckets[k][0])]
                      for k in keys},
            'beta_fit': beta_fit, 'r2': r2, 'beta_true': beta_true,
            'n_states': n_states}


# =====================================================================
# 六、基础绘图工具（供绘图脚本复用；绘图脚本只读 JSON）
# =====================================================================
def plot_losses(rec, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(rec['epochs'], rec['train_learner'], label='训练（学习器）', alpha=0.8)
    ax.plot(rec['epochs'], rec['train_gen'], label='训练（生成器）', alpha=0.8)
    ax.plot(rec['epochs'], rec['val'], label='验证', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('交叉熵损失'); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
