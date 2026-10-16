# -*- coding: utf-8 -*-
"""
synthesize_report.py —— 汇总 phase0/phase1 结果，输出最终对比报告（markdown + 控制台）
运行时机：phase0 与 phase1 均完成后
"""
import os
import json

BASE = os.path.dirname(os.path.abspath(__file__))
P0 = os.path.join(BASE, 'results', 'phase0', 'phase0_results.json')
P1 = os.path.join(BASE, 'results', 'phase1', 'phase1_summary.json')

lines = []
lines.append("# 双通道信息分离实验报告\n")
lines.append("实验配置：BA网络 N=100 m=3，β=0.3，γ=0.1，训练30epoch/hidden=64/batch=32，"
             "所有模型使用**双向 edge_index**（已修复方向性bug）。\n")

if os.path.exists(P0):
    r0 = json.load(open(P0, encoding='utf-8'))
    lines.append("## Phase 0：方向性 bug 验证\n")
    lines.append("| 模型 | 拟合β | γ | γ-度数相关 | 轨迹I-MAE |")
    lines.append("|---|---|---|---|---|")
    for key, label in (('asym', '有向(notebook原样)'), ('bi', '双向(修正)')):
        r = r0[key]
        fb = r.get('fit_beta_overall')
        beta = f"{fb[0]:.4f}±{fb[1]:.4f}" if fb else "N/A"
        gam = r['gamma']
        traj = r['trajectory']
        lines.append(f"| {label} | {beta} | {gam['mean']:.4f} | {gam['corr_degree']:.4f} "
                     f"(p={gam['pval']:.1e}) | {traj['mae_i_mean']:.4f} |")
    lines.append("")
    lines.append("真值：β=0.3，γ=0.1，γ-度数相关应为 0。\n")
    # 分桶β
    lines.append("### Phase 0：按度数分桶的拟合 β\n")
    for key, label in (('asym', '有向'), ('bi', '双向')):
        bb = r0[key].get('beta_by_bucket', {})
        if bb:
            lines.append(f"- **{label}**: " + ", ".join(f"deg {k}: {v:.3f}" for k, v in sorted(bb.items())))
    lines.append("")

if os.path.exists(P1):
    s = json.load(open(P1, encoding='utf-8'))
    lines.append("## Phase 1：双通道消融对比（3 seeds 均值±标准差）\n")
    lines.append("| 模型 | 拟合β | per-k MAE | γ | γ-度数相关 | 轨迹I-MAE |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in s.items():
        b = r['beta']; pm = r['perk_mae']; gm = r['gamma_mean']; gc = r['gamma_corr']; tm = r['traj_mae']
        lines.append(f"| {name} | {b[0]:.4f}±{b[1]:.4f} | {pm[0]:.4f}±{pm[1]:.4f} | "
                     f"{gm[0]:.4f}±{gm[1]:.4f} | {gc[0]:.4f}±{gc[1]:.4f} | {tm[0]:.4f}±{tm[1]:.4f} |")
    lines.append("")
    lines.append("### Phase 1：按度数分桶的拟合 β\n")
    lines.append("| 模型 | " + " | ".join(s['baseline_3L']['beta_by_bucket'].keys()) + " |")
    lines.append("|---|" + "---|" * len(s['baseline_3L']['beta_by_bucket']))
    for name, r in s.items():
        cells = []
        for b in s['baseline_3L']['beta_by_bucket'].keys():
            v = r['beta_by_bucket'].get(b)
            cells.append(f"{v[0]:.3f}" if v else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")

report = "\n".join(lines)
out_path = os.path.join(BASE, 'results', 'REPORT.md')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(report)
print(f"\n报告已保存: {out_path}")
