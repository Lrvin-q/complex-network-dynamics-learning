# -*- coding: utf-8 -*-
"""
retrofit_metadata.py —— 合规整改：给已有结果 JSON 补充 config 元数据与缺失标注
原则（DSH.md）：保存的数据必须自包含可重绘。本脚本只新增键，不改动任何已有数值。
"""
import os
import json

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

EMPTY_BUCKET_NOTE = "no nodes in this degree range (BA(100,3) degree gap: 12 -> 23)"

# ================= 阶段配置（与 phase0/phase1 脚本一致） =================
PHASE0_CONFIG = {
    'network': {'type': 'BA', 'N': 100, 'm': 3,
                'degree_range': [3, 33], 'empty_degree_buckets': ['14-19']},
    'sir': {'beta': 0.3, 'gamma': 0.1, 'n_seeds': 30, 'n_repeats': 5, 'T_max': 60},
    'training': {'model': 'GraphConvSIR', 'hidden': 64, 'num_layers': 3, 'dropout': 0.2,
                 'epochs': 30, 'lr': 0.01, 'batch_size': 32, 'aggr': 'sum',
                 'optimizer': 'Adam', 'loss': 'CrossEntropy', 'val_frac': 0.2, 'seed': 21},
    'diagnostics': {'degree_buckets': [[3, 6], [6, 10], [10, 14], [14, 20], [20, 34]],
                    'bucket_labels': ['3-5', '6-9', '10-13', '14-19', '20-33'],
                    'max_k': 10, 'min_per_k': 300, 'min_per_bk': 100, 'max_iters': 8000,
                    'gamma_n_states': 1500,
                    'trajectory': {'n_mc': 40, 'T': 50, 'initial_infected_counts': [1, 2, 3]}},
    'edge_index_mode': {'asym': 'directed (notebook original, buggy: half neighbors invisible)',
                        'bi': 'bidirectional (fixed)'},
}

PHASE1_CONFIG = {
    'network': {'type': 'BA', 'N': 100, 'm': 3,
                'degree_range': [3, 33], 'empty_degree_buckets': ['14-19']},
    'sir': {'beta': 0.3, 'gamma': 0.1, 'n_seeds': 30, 'n_repeats': 5, 'T_max': 60},
    'training': {'epochs': 30, 'hidden': 64, 'lr': 0.01, 'batch_size': 32,
                 'dropout': 0.2, 'optimizer': 'Adam', 'loss': 'CrossEntropy',
                 'val_frac': 0.2, 'seeds': [21, 42, 123], 'edge_index': 'bidirectional'},
    'models': {
        'baseline_3L': {'class': 'GraphConvSIR', 'num_layers': 3, 'aggr': 'sum'},
        'baseline_1L': {'class': 'GraphConvSIR', 'num_layers': 1, 'aggr': 'sum'},
        'two_channel_A': {'class': 'TwoChannelSIR', 'aggr': 'sum'},
        'two_channel_Amean': {'class': 'TwoChannelSIR', 'aggr': 'mean'},
        'two_channel_B': {'class': 'TwoChannelGCN', 'num_layers': 2},
    },
    'diagnostics': {'degree_buckets': [[3, 6], [6, 10], [10, 14], [14, 20], [20, 34]],
                    'bucket_labels': ['3-5', '6-9', '10-13', '14-19', '20-33'],
                    'max_k': 10, 'min_per_k': 250, 'min_per_bk': 80, 'max_iters': 6000,
                    'gamma_n_states': 1200,
                    'trajectory': {'n_mc': 30, 'T': 50, 'initial_infected_counts': [1, 2, 3]}},
}

def annotate_phase0(results, config):
    diag = config['diagnostics']
    labels = diag['bucket_labels']
    hi_per_bucket = [b[1] - 1 for b in diag['degree_buckets']]   # 桶内最大可能度数
    empty_buckets = set(config['network']['empty_degree_buckets'])
    for key, model in results.items():
        # heatmap 缺失单元及原因
        hm = model['heatmap']
        missing = {}
        for i, row in enumerate(hm):
            for j, v in enumerate(row):
                if v is None or v < 0:
                    k = j + 1
                    reason = 'insufficient samples (<20)'
                    if labels[i] in empty_buckets:
                        reason = EMPTY_BUCKET_NOTE
                    elif k > hi_per_bucket[i]:
                        reason = f'impossible: k exceeds max degree {hi_per_bucket[i]} of bucket'
                    missing[f'{labels[i]}/k={k}'] = reason
        model['heatmap_missing_cells'] = missing
        # 分桶 β / γ 缺失桶
        for field in ('beta_by_bucket', 'gamma_by_bucket'):
            present = set(model.get(field, {}).keys())
            absent = {}
            for lab in labels:
                if lab not in present:
                    absent[lab] = EMPTY_BUCKET_NOTE if lab in empty_buckets \
                        else 'no fit (insufficient samples)'
            model[f'{field}_missing'] = absent
        model['edge_index'] = config['edge_index_mode'][key]
    results['config'] = config
    return results

def annotate_phase1(results, summary, config):
    diag = config['diagnostics']
    labels = diag['bucket_labels']
    empty_buckets = set(config['network']['empty_degree_buckets'])
    for name, runs in results.items():
        for r in runs:
            present = set(r['beta_by_bucket'].keys())
            absent = {}
            for lab in labels:
                if lab not in present:
                    absent[lab] = EMPTY_BUCKET_NOTE if lab in empty_buckets \
                        else 'no fit (insufficient samples)'
            r['beta_by_bucket_missing'] = absent
    results['config'] = config
    for name, s in summary.items():
        present = set(s['beta_by_bucket'].keys())
        absent = {}
        for lab in labels:
            if lab not in present:
                absent[lab] = EMPTY_BUCKET_NOTE if lab in empty_buckets \
                    else 'no fit in any seed'
        s['beta_by_bucket_missing'] = absent
    summary['config'] = config
    return results, summary

def main():
    p0 = os.path.join(BASE, 'phase0', 'phase0_results.json')
    p1r = os.path.join(BASE, 'phase1', 'phase1_results.json')
    p1s = os.path.join(BASE, 'phase1', 'phase1_summary.json')

    r0 = json.load(open(p0, encoding='utf-8'))
    r0 = annotate_phase0(r0, PHASE0_CONFIG)
    json.dump(r0, open(p0, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=float)
    print(f'updated: {p0}')

    r1 = json.load(open(p1r, encoding='utf-8'))
    s1 = json.load(open(p1s, encoding='utf-8'))
    r1, s1 = annotate_phase1(r1, s1, PHASE1_CONFIG)
    json.dump(r1, open(p1r, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=float)
    json.dump(s1, open(p1s, 'w', encoding='utf-8'), ensure_ascii=False, indent=2, default=float)
    print(f'updated: {p1r}')
    print(f'updated: {p1s}')
    print('done (only added keys; existing values untouched)')

if __name__ == '__main__':
    main()
