# -*- coding: utf-8 -*-
"""
plot_ws_probe.py —— WS 自回归偏差结构性归因图（只读 JSON）
输出（results/phase3c/）：
  fig_ws_trajectories.png  三网络：真实 I 曲线±1σ（阴影） vs 模型曲线
  fig_ws_bias.png          三网络：|偏差| 曲线 vs 真实固有波动带
  fig_ws_perk_signed.png   三网络：逐k有符号误差（pred−truth）
  fig_ws_structure.png     结构指标（聚集/度方差/路径长）与自回归误差的关联
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
P3C = os.path.join(BASE, 'phase3c')
P3 = os.path.join(BASE, 'phase3', 'phase3_eval.json')
NETS = ['ER', 'WS', 'BA']
COLORS = {'ER': '#1565c0', 'WS': '#c62828', 'BA': '#2e7d32'}

probe = json.load(open(os.path.join(P3C, 'ws_probe.json'), encoding='utf-8'))
phase3 = json.load(open(P3, encoding='utf-8'))


# ---------------- 图A：轨迹 ±1σ ----------------
def fig_trajectories():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, net in zip(axes, NETS):
        d = probe[net]
        t = np.arange(len(d['true_mean']))
        tm = np.array(d['true_mean']); ts = np.array(d['true_std']); pm = np.array(d['pred_mean'])
        ax.fill_between(t, tm - ts, tm + ts, color=COLORS[net], alpha=0.18, label='True ± 1σ')
        ax.plot(t, tm, color=COLORS[net], linewidth=2, label='True mean')
        ax.plot(t, pm, '--', color='k', linewidth=1.8, label='Model rollout')
        ax.set_xlabel('Time step'); ax.set_ylabel('Infected fraction')
        ax.set_title(f'({net}) MAE(I)={d["mae_i"]:.4f}, bias/std={d["bias_over_std"]:.2f}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    fig.suptitle('Autoregressive rollout vs true dynamics (±1σ intrinsic band)', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(P3C, 'fig_ws_trajectories.png'), dpi=150)
    plt.close(fig)


# ---------------- 图B：偏差时长 vs 固有波动 ----------------
def fig_bias():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, net in zip(axes, NETS):
        d = probe[net]
        t = np.arange(len(d['true_mean']))
        bias = np.abs(np.array(d['pred_mean']) - np.array(d['true_mean']))
        std = np.array(d['true_std'])
        ax.plot(t, bias, color=COLORS[net], linewidth=2, label='|model − true|')
        ax.plot(t, std, '--', color='gray', linewidth=1.5, label='True σ (intrinsic)')
        n_long = int((bias > 0.02).sum())
        ax.fill_between(t, 0, bias, color=COLORS[net], alpha=0.12)
        ax.set_xlabel('Time step'); ax.set_ylabel('I-fraction deviation')
        ax.set_title(f'({net}) steps with |bias|>0.02: {n_long}')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle('Model deviation vs intrinsic stochasticity', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(P3C, 'fig_ws_bias.png'), dpi=150)
    plt.close(fig)


# ---------------- 图C：逐k有符号误差 ----------------
def fig_perk_signed():
    fig, ax = plt.subplots(figsize=(8, 5))
    for net in NETS:
        pk = phase3[net]['models']['twob_bi']['perk']
        err = np.array(pk['pred']) - np.array(pk['truth'])
        ax.plot(pk['k'], err, 'o-', color=COLORS[net], label=net, markersize=5)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel('Infected neighbors k'); ax.set_ylabel('pred − truth (signed)')
    ax.set_title('Per-k signed prediction error of TwoChannelB (random-state probe)')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(P3C, 'fig_ws_perk_signed.png'), dpi=150)
    plt.close(fig)


# ---------------- 图D：结构指标 vs 自回归误差 ----------------
def fig_structure():
    struct_metrics = {'clustering': 'Clustering coefficient', 'deg_var': 'Degree variance',
                      'avg_path': 'Average shortest path length'}
    mae = [probe[n]['mae_i'] for n in NETS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (key, label) in zip(axes, struct_metrics.items()):
        vals = [probe[n]['structure'][key] for n in NETS]
        for i, net in enumerate(NETS):
            ax.scatter(vals[i], mae[i], s=90, color=COLORS[net], label=net, zorder=3)
            ax.annotate(net, (vals[i], mae[i]), textcoords='offset points', xytext=(6, 4), fontsize=9)
        ax.set_xlabel(label); ax.set_ylabel('Autoregressive MAE(I)')
        ax.set_title(f'{label} vs rollout error')
        ax.grid(alpha=0.3)
        if key == 'clustering':
            ax.legend(fontsize=8)
    fig.suptitle('Network structure vs autoregressive error (TwoChannelB)', fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(P3C, 'fig_ws_structure.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_trajectories()
    fig_bias()
    fig_perk_signed()
    fig_structure()
    print('ws probe figures saved to', P3C)
