# -*- coding: utf-8 -*-
"""
common.py —— 双通道实验共享模块
包含：数据生成、SIR模拟、模型定义（基线/双通道A/双通道B）、训练、诊断指标、绘图、结果保存。
"""
import os
import json
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GraphConv
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 0. 随机种子
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ============================================================
# 1. 网络生成（BA），返回：双向edge_index(正确聚合用)、有向edge_index(复现notebook bug用)、G
# ============================================================
def generate_ba_graph(N=100, m=3):
    G = nx.barabasi_albert_graph(N, m)
    edges = np.array(G.edges()).T                      # (2, E)，每条无向边一次，u<v
    edge_index_dir = torch.tensor(edges, dtype=torch.long)                       # notebook原样（方向残缺）
    edge_index_bi = torch.cat([edge_index_dir, edge_index_dir.flip(0)], dim=1)   # 正确双向
    return edge_index_bi, edge_index_dir, G


def generate_network(net_type='BA', N=100, p=0.1, m=3, k=6, rewiring_p=0.2):
    """生成三种经典网络之一，返回 (edge_index_bi, edge_index_dir, G)。
    ER: 连边概率 p（若不连通取最大连通分量）；BA: 每步新增边 m；WS: 初始邻居 k、重连概率 rewiring_p。
    节点标签重编号为 0..N'-1（N' 为连通分量节点数）。"""
    if net_type == 'ER':
        G = nx.erdos_renyi_graph(N, p)
        if not nx.is_connected(G):
            largest = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest)
    elif net_type == 'BA':
        G = nx.barabasi_albert_graph(N, m)
    elif net_type == 'WS':
        G = nx.watts_strogatz_graph(N, k, rewiring_p)
    else:
        raise ValueError(f"unsupported net_type: {net_type}")
    G = nx.convert_node_labels_to_integers(G)
    edges = np.array(G.edges()).T
    edge_index_dir = torch.tensor(edges, dtype=torch.long)
    edge_index_bi = torch.cat([edge_index_dir, edge_index_dir.flip(0)], dim=1)
    return edge_index_bi, edge_index_dir, G

# ============================================================
# 2. SIR 模拟器（离散时间，与notebook一致）
# ============================================================
def sir_simulation(edge_index, beta, gamma, initial_infected, T=50):
    """
    在固定图上模拟SIR过程（邻接表按无向去重构建，兼容有向/双向两种 edge_index）。
    edge_index: 边列表 (2, E)，可为有向（每条无向边一次）或双向（每条两次）
    beta: 感染概率
    gamma: 恢复概率
    initial_infected: 初始感染节点列表（或单个节点索引）
    T: 模拟步数（包括初始状态，输出T+1个时间点）
    返回: states (T+1, N) 每个元素 0=S,1=I,2=R
    注意：若不去重，传入双向 edge_index 会导致每个邻居被计数两次，
    感染概率被系统性放大为 1-(1-beta)^(2k)（自回归评估曾因此产生假象，见 DSH.md 坑10）。
    """
    N = torch.max(edge_index).item() + 1
    adj_set = [set() for _ in range(N)]
    for u, v in edge_index.t().tolist():
        adj_set[u].add(v)
        adj_set[v].add(u)
    adj_list = [sorted(s) for s in adj_set]
    state = np.zeros(N, dtype=int)
    state[initial_infected] = 1
    states = [state.copy()]
    for _ in range(T):
        new_state = state.copy()
        for i in range(N):
            if state[i] == 0:
                infected_neighbors = sum(1 for nb in adj_list[i] if state[nb] == 1)
                if infected_neighbors > 0:
                    prob_infect = 1.0 - (1.0 - beta) ** infected_neighbors
                    if np.random.random() < prob_infect:
                        new_state[i] = 1
            elif state[i] == 1:
                if np.random.random() < gamma:
                    new_state[i] = 2
        state = new_state
        states.append(state.copy())
    return np.array(states)

# ============================================================
# 3. 训练数据集（轨迹切片为 (x_t, x_{t+1}) one-hot 对）
# ============================================================
def generate_dataset(edge_index, beta, gamma, n_seeds=30, n_repeats=5, T=60):
    N = torch.max(edge_index).item() + 1
    dataset = []
    for seed in range(n_seeds):
        initial_node = np.random.randint(0, N)
        for _ in range(n_repeats):
            states = sir_simulation(edge_index, beta, gamma, initial_node, T)
            for t in range(T):
                x_t = np.eye(3)[states[t]].astype(np.float32)
                y_t = np.eye(3)[states[t + 1]].astype(np.float32)
                dataset.append((x_t, y_t))
    return dataset

class SIRDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def build_loaders(dataset, batch_size=32, val_frac=0.2, seed=21):
    set_seed(seed)
    idx = np.random.permutation(len(dataset))
    split = int((1 - val_frac) * len(dataset))
    train_ds = SIRDataset([dataset[i] for i in idx[:split]])
    val_ds = SIRDataset([dataset[i] for i in idx[split:]])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def batch_edge_index(edge_index, batch_size, N, device='cpu'):
    """将图结构扩展到batch：节点索引偏移（与notebook一致）"""
    e = edge_index.repeat(1, batch_size)
    offset = torch.arange(batch_size, device=device) * N
    offset = offset.repeat_interleave(edge_index.size(1)).unsqueeze(0)
    return e + offset

# ============================================================
# 4. 模型
# ============================================================
# 4.1 基线模型（notebook的 ImprovedGCN_Markov，可配置层数与聚合方式）
class GraphConvSIR(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64, out_channels=3,
                 num_layers=3, dropout=0.2, aggr='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return self.lin(x)

# 4.2 双通道模型 A：1-hop 充分统计量版
#     自身通道：MLP(x)          —— 只编码自己的状态
#     邻居通道：MLP(sum邻居one-hot) —— 只编码邻居状态分布（含 k）
#     最后拼接进 head
class TwoChannelSIR(nn.Module):
    def __init__(self, in_channels=3, hidden=64, out_channels=3, dropout=0.2, aggr='sum'):
        super().__init__()
        self.aggr = aggr
        self.self_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.neigh_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels))

    def forward(self, x, edge_index):
        # 邻居 sum/mean 聚合（仅聚合邻居，不含自身）
        m = torch.zeros_like(x)
        m = m.index_add_(0, edge_index[0], x[edge_index[1]])
        if self.aggr == 'mean':
            deg = torch.zeros(x.size(0), 1)
            deg = deg.index_add_(0, edge_index[0], torch.ones(edge_index[0].size(0), 1, dtype=x.dtype))
            m = m / deg.clamp(min=1.0)
        h_self = self.self_mlp(x)
        h_neigh = self.neigh_mlp(m)
        return self.head(torch.cat([h_self, h_neigh], dim=-1))

# 4.3 双通道模型 B：多层分离版
#     自身通道每层只做自身变换（不聚合）；邻居通道每层聚合邻居的“自身嵌入”（sum）
class TwoChannelGCN(nn.Module):
    def __init__(self, in_channels=3, hidden=64, out_channels=3, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed_self = nn.Linear(in_channels, hidden)
        self.self_blocks = nn.ModuleList()
        self.neigh_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.self_blocks.append(nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()))
            self.neigh_blocks.append(nn.Sequential(
                nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU()))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels))

    def forward(self, x, edge_index):
        h_self = F.relu(self.embed_self(x))
        h_neigh = None
        for l in range(len(self.self_blocks)):
            m = torch.zeros_like(h_self)
            m = m.index_add_(0, edge_index[0], h_self[edge_index[1]])  # 聚合邻居的自身嵌入
            h_neigh = self.neigh_blocks[l](m)
            h_self = self.self_blocks[l](h_self)
        return self.head(torch.cat([h_self, h_neigh], dim=-1))

# ============================================================
# 5. 训练（与notebook一致：batch内图结构扩展，CrossEntropy）
# ============================================================
def train_model(model, train_loader, val_loader, edge_index,
                epochs=30, lr=0.01, save_path=None, verbose_every=10, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            bs, N, C = x.shape
            x_flat = x.view(-1, C)
            y_flat = y.view(-1, C)
            e = batch_edge_index(edge_index, bs, N, device)
            logits = model(x_flat, e)
            labels = torch.argmax(y_flat, dim=1)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * bs
        train_losses.append(total / len(train_loader.dataset))
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                bs, N, C = x.shape
                x_flat = x.view(-1, C)
                y_flat = y.view(-1, C)
                e = batch_edge_index(edge_index, bs, N, device)
                logits = model(x_flat, e)
                labels = torch.argmax(y_flat, dim=1)
                vtotal += criterion(logits, labels).item() * bs
        val_losses.append(vtotal / len(val_loader.dataset))
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % verbose_every == 0:
            print(f"  epoch {epoch+1}/{epochs} train={train_losses[-1]:.4f} val={val_losses[-1]:.4f} best={best_val:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path:
        torch.save(model.state_dict(), save_path)
    return train_losses, val_losses

def predict_probs(model, state, edge_index, device='cpu'):
    """给定状态 (N,) 0/1/2，返回模型预测的 (N,3) 概率"""
    model.eval()
    x = np.eye(3)[state].astype(np.float32)
    xt = torch.tensor(x).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(xt.view(-1, 3), edge_index.to(device))
        return F.softmax(logits, dim=1).cpu().numpy()

# ============================================================
# 6. 诊断：感染率 β 提取（控制 k 采样，按度数分组）
# ============================================================
def collect_si_samples(model, edge_index, adj_list, degree_buckets,
                       max_k=10, min_per_k=300, min_per_bk=100, max_iters=8000,
                       device='cpu', verbose=True):
    """
    自适应采样 S 节点，收集 P(S->I|k) 样本。
    返回:
      samples_by_k : dict k -> list[prob]          （所有度数混合）
      samples_by_bk: dict (bucket_idx, k) -> list[prob]
      counts_by_k  : dict k -> count
    """
    N = edge_index.max().item() + 1
    degrees = np.array([len(adj_list[i]) for i in range(N)])
    bucket_of = np.full(N, -1)
    for b, (lo, hi) in enumerate(degree_buckets):
        bucket_of[(degrees >= lo) & (degrees < hi)] = b

    samples_by_k = {k: [] for k in range(1, max_k + 1)}
    samples_by_bk = {}
    counts_by_k = {k: 0 for k in range(1, max_k + 1)}

    with torch.no_grad():
        for it in range(max_iters):
            # 自适应感染比例：优先补齐欠采样的 k
            under = [k for k in range(1, max_k + 1) if counts_by_k[k] < min_per_k]
            if under:
                target_k = np.mean(under)
                ratio = float(np.clip(target_k / degrees.mean() * 0.9, 0.05, 0.85))
            else:
                ratio = float(np.random.uniform(0.15, 0.7))
            n_inf = max(1, int(N * ratio))
            state = np.zeros(N, dtype=int)
            state[np.random.choice(N, n_inf, replace=False)] = 1
            probs = predict_probs(model, state, edge_index, device)
            s_nodes = np.where(state == 0)[0]
            for node in s_nodes:
                k = sum(1 for nb in adj_list[node] if state[nb] == 1)
                if 1 <= k <= max_k:
                    samples_by_k[k].append(float(probs[node, 1]))
                    counts_by_k[k] += 1
                    b = int(bucket_of[node])
                    if b >= 0:
                        samples_by_bk.setdefault((b, k), []).append(float(probs[node, 1]))
            # 停止条件：所有 k 达标 且 每个 (bucket,k) 达标(或采样数足够多)
            if all(counts_by_k[k] >= min_per_k for k in range(1, max_k + 1)):
                if it > 300 and all(len(samples_by_bk.get((b, k), [])) >= min_per_bk
                                    for b in range(len(degree_buckets)) for k in range(1, max_k + 1)):
                    break
                # bucket 无法全部达标时，至少让主k范围尽量覆盖
                if it > max_iters * 0.6:
                    break
    if verbose:
        print("  k 样本数:", {k: counts_by_k[k] for k in range(1, max_k + 1)})
    return samples_by_k, samples_by_bk, counts_by_k, bucket_of, degrees

def fit_beta(k_values, probs_list):
    """加权拟合 P = 1-(1-beta)^k；返回 (beta, std, r2) 或 None"""
    ks = np.array(k_values, dtype=float)
    means = np.array([np.mean(p) for p in probs_list])
    counts = np.array([len(p) for p in probs_list])
    if len(ks) < 2:
        return None
    def f(k, beta):
        return 1 - (1 - beta) ** k
    try:
        weights = counts / counts.sum()
        popt, pcov = curve_fit(f, ks, means, p0=[0.3], bounds=(0, 1),
                               sigma=1 / (weights + 0.01))
        beta = float(popt[0])
        std = float(np.sqrt(pcov[0, 0])) if pcov[0, 0] > 0 else 0.01
        pred = f(ks, beta)
        ss_res = np.sum((means - pred) ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return beta, std, r2
    except Exception:
        return None

def compute_perk_stats(samples_by_k, beta_true):
    """逐 k 的预测均值/误差 vs 真值 1-(1-beta)^k"""
    ks, preds, errs, counts, truths = [], [], [], [], []
    for k in range(1, len(samples_by_k) + 1):
        p = samples_by_k.get(k, [])
        if len(p) >= 20:
            pred = float(np.mean(p))
            truth = 1 - (1 - beta_true) ** k
            ks.append(k); preds.append(pred); truths.append(truth)
            errs.append(abs(pred - truth)); counts.append(len(p))
    return dict(k=ks, pred=preds, truth=truths, err=errs, count=counts)

# ============================================================
# 7. 诊断：恢复率 γ（I->R 概率）及其与度数的相关性
# ============================================================
def collect_gamma_samples(model, edge_index, adj_list, n_states=1500, device='cpu', verbose=True):
    N = edge_index.max().item() + 1
    degrees = np.array([len(adj_list[i]) for i in range(N)])
    all_probs, all_deg = [], []
    with torch.no_grad():
        for _ in range(n_states):
            ratio = float(np.random.uniform(0.1, 0.6))
            n_inf = max(1, int(N * ratio))
            state = np.zeros(N, dtype=int)
            state[np.random.choice(N, n_inf, replace=False)] = 1
            state[state == 0] = 2   # 其余为 R（与notebook一致：隔离 I 节点）
            probs = predict_probs(model, state, edge_index, device)
            i_nodes = np.where(state == 1)[0]
            for node in i_nodes:
                all_probs.append(float(probs[node, 2]))
                all_deg.append(int(degrees[node]))
    all_probs = np.array(all_probs); all_deg = np.array(all_deg)
    corr, pval = pearsonr(all_deg, all_probs)
    if verbose:
        print(f"  γ: 均值={all_probs.mean():.4f}±{all_probs.std():.4f}, "
              f"与度数相关={corr:.4f} (p={pval:.2e}), n={len(all_probs)}")
    return all_probs, all_deg, corr, pval

# ============================================================
# 8. 诊断：自回归轨迹 MAE（I 占比曲线）
# ============================================================
def trajectory_mae(model, edge_index, initial_state, beta, gamma, T=50, n_mc=40, device='cpu'):
    N = initial_state.shape[0]
    init_inf = np.where(initial_state == 1)[0].tolist()
    if len(init_inf) == 0:
        init_inf = [np.random.randint(0, N)]
    # 真实轨迹
    true_frac = np.zeros((T + 1, 3))
    for _ in range(n_mc):
        st = sir_simulation(edge_index, beta, gamma, init_inf, T)
        for t in range(T + 1):
            true_frac[t, 0] += np.mean(st[t] == 0) / n_mc
            true_frac[t, 1] += np.mean(st[t] == 1) / n_mc
            true_frac[t, 2] += np.mean(st[t] == 2) / n_mc
    # 模型自回归
    pred_frac = np.zeros((T + 1, 3))
    for _ in range(n_mc):
        state = initial_state.copy()
        pred_frac[0, 0] += np.mean(state == 0) / n_mc
        pred_frac[0, 1] += np.mean(state == 1) / n_mc
        pred_frac[0, 2] += np.mean(state == 2) / n_mc
        for step in range(T):
            probs = predict_probs(model, state, edge_index, device)
            state = np.array([np.random.choice(3, p=probs[i]) for i in range(N)])
            pred_frac[step + 1, 0] += np.mean(state == 0) / n_mc
            pred_frac[step + 1, 1] += np.mean(state == 1) / n_mc
            pred_frac[step + 1, 2] += np.mean(state == 2) / n_mc
    mae_i = float(np.mean(np.abs(pred_frac[:, 1] - true_frac[:, 1])))
    mae_all = float(np.mean(np.abs(pred_frac - true_frac)))
    return dict(mae_i=mae_i, mae_all=mae_all, true=true_frac, pred=pred_frac)


def evaluate_autoregressive_full(model, edge_index, initial_state, beta, gamma,
                                 T=50, n_mc=50, device='cpu'):
    """完整自回归评估：返回 (true_frac, pred_frac)，均为 (T+1, 3) 的 S/I/R 占比曲线。
    真实轨迹由 SIR 模拟器产生；模型轨迹由模型概率逐时间步采样产生（自回归）。"""
    N = initial_state.shape[0]
    init_inf = np.where(initial_state == 1)[0].tolist()
    true_frac = np.zeros((T + 1, 3))
    pred_frac = np.zeros((T + 1, 3))
    for _ in range(n_mc):
        st = sir_simulation(edge_index, beta, gamma, init_inf, T)
        for t in range(T + 1):
            true_frac[t, 0] += np.mean(st[t] == 0) / n_mc
            true_frac[t, 1] += np.mean(st[t] == 1) / n_mc
            true_frac[t, 2] += np.mean(st[t] == 2) / n_mc
        state = initial_state.copy()
        for t in range(T + 1):
            pred_frac[t, 0] += np.mean(state == 0) / n_mc
            pred_frac[t, 1] += np.mean(state == 1) / n_mc
            pred_frac[t, 2] += np.mean(state == 2) / n_mc
            if t < T:
                probs = predict_probs(model, state, edge_index, device)
                state = np.array([np.random.choice(3, p=probs[i]) for i in range(N)])
    return true_frac, pred_frac

# ============================================================
# 9. 保存 JSON / 绘图
# ============================================================
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=float)

def plot_k_curve(perk, beta_est, title, path):
    ks = np.array(perk['k']); pred = np.array(perk['pred']); truth = np.array(perk['truth'])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(ks, truth, 'k-', linewidth=2, label='True 1-(1-β)^k, β=0.3')
    ax.plot(ks, pred, 'ro-', markersize=6, label=f'Model (β≈{beta_est:.3f})')
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('P(S→I)')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_heatmap(err_matrix, degree_buckets, max_k, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(err_matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(1, max_k + 1))
    ax.set_xticklabels([str(k) for k in range(1, max_k + 1)])
    ax.set_yticks(range(len(degree_buckets)))
    ax.set_yticklabels([f'{lo}-{hi-1}' for lo, hi in degree_buckets])
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('Degree bucket')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    for i in range(err_matrix.shape[0]):
        for j in range(err_matrix.shape[1]):
            v = err_matrix[i, j]
            if v >= 0:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='black' if v < 0.35 else 'white')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_training(train_losses, val_losses, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label='Train'); ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch'); ax.set_ylabel('CrossEntropy'); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
