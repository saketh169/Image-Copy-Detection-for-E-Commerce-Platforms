#!/usr/bin/env python3
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/home/saketh/Codebase')
import phase3_all_datasets_graph_benchmark as G
import refine_global_tau_maxq as R

ROOT = Path('/home/saketh/json/final_named_artifacts_20260601')
OUT = ROOT / 'latest_tau_allq_attack_tables'
OUT.mkdir(parents=True, exist_ok=True)
ATTACKS = G.ATTACKS

STATE_SOURCES = {
    'cifar': ROOT / 'multi_split_global_tau_local_diagonal/cifar/cifar_state.json',
    'flickr': ROOT / 'multi_split_global_tau_recovery/flickr/flickr_state.json',
    'ucid': ROOT / 'multi_split_global_tau_local_diagonal/ucid/ucid_state.json',
    'amazon': ROOT / 'multi_split_global_tau_local_diagonal/amazon/amazon_state.json',
}
Q_SIZES = {
    'cifar': [100, 200, 500, 1000],
    'flickr': [100, 200, 500, 1000],
    'ucid': [100, 200, 500, 850],
    'amazon': [100, 200, 500, 1000, 2000],
}


def load_state(ds: str):
    p = STATE_SOURCES[ds]
    state = json.load(open(p, 'r', encoding='utf-8'))
    metrics = state.get('metrics', {})
    tau = metrics.get('best_global_threshold', metrics.get('best_global_tau'))
    if tau is None:
        raise RuntimeError(f'No tau found in {p}')
    ckpt = Path(state['checkpoint'])
    if not ckpt.exists():
        raise RuntimeError(f'Missing checkpoint {ckpt}')
    return p, state, metrics, ckpt, float(tau)


@torch.no_grad()
def embed_bank(model, rows, batch_size=64):
    print(f'[bank] embedding known bank rows={len(rows)} batch={batch_size}', flush=True)
    return G.embed_queries(model, rows, None, batch_size=batch_size).astype(np.float32)


def acc_from_dist(dist, q, tau):
    known = float((dist[:q] <= tau).mean() * 100.0)
    unknown = float((dist[q:] > tau).mean() * 100.0)
    return known, unknown, (known + unknown) / 2.0


def summarize(rows):
    return {
        'direct_total': round(float(np.mean([r['direct_total'] for r in rows])), 6),
        'direct_known': round(float(np.mean([r['direct_known'] for r in rows])), 6),
        'direct_unknown': round(float(np.mean([r['direct_unknown'] for r in rows])), 6),
        'direct_ms': round(float(np.mean([r['direct_ms'] for r in rows])), 6),
        'min_attack_total': round(float(min(r['direct_total'] for r in rows)), 6),
        'min_attack_known': round(float(min(r['direct_known'] for r in rows)), 6),
        'min_attack_unknown': round(float(min(r['direct_unknown'] for r in rows)), 6),
    }


def write_markdown(all_payloads):
    lines = ['# Latest Accepted Tau All-Q Attack Tables\n']
    lines.append('Direct exact nearest-neighbor evaluation using the latest accepted tau/checkpoint state for each dataset.\n')
    for ds, payload in all_payloads.items():
        lines.append(f'## {ds.upper()}\n')
        lines.append(f"Checkpoint: `{payload['checkpoint']}`\n")
        lines.append(f"State: `{payload['state_path']}`\n")
        lines.append(f"Tau: `{payload['tau']}`\n")
        qkeys = [str(q) for q in payload['q_sizes']]
        lines.append('| Attack | Tau | ' + ' | '.join([f'Q{q}' for q in qkeys]) + ' |')
        lines.append('|---|---:|' + '|'.join(['---:'] * len(qkeys)) + '|')
        attacks = [r['attack'] for r in payload['queries'][qkeys[0]]['rows']]
        for attack in attacks:
            vals = []
            for q in qkeys:
                row = next(r for r in payload['queries'][q]['rows'] if r['attack'] == attack)
                vals.append(f"{row['direct_total']:.2f}")
            lines.append(f"| {attack} | {payload['tau']:.10f} | " + ' | '.join(vals) + ' |')
        lines.append('')
        lines.append('| Q | Avg | Known | Unknown | Min Attack | Min Known | Min Unknown |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|')
        for q in qkeys:
            s = payload['queries'][q]['summary']
            lines.append(f"| {q} | {s['direct_total']:.6f} | {s['direct_known']:.6f} | {s['direct_unknown']:.6f} | {s['min_attack_total']:.6f} | {s['min_attack_known']:.6f} | {s['min_attack_unknown']:.6f} |")
        lines.append('')
    out = OUT / 'latest_tau_allq_attack_tables.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def run(ds: str):
    state_path, state, metrics, ckpt, tau = load_state(ds)
    cfg = G.DATASETS[ds]
    known, unknown, _stored_bank = G.load_rows(cfg['store'])
    model = G.load_model(ckpt)
    bank = np.ascontiguousarray(embed_bank(model, known, batch_size=64))
    weight = np.ones(bank.shape[1], dtype=np.float32)
    payload = {
        'dataset': ds,
        'mode': 'latest_accepted_single_global_tau_all_queries_all_attacks_direct_exact',
        'state_path': str(state_path),
        'checkpoint': str(ckpt),
        'source_metrics': {k: v for k, v in metrics.items() if not str(k).startswith('_')},
        'tau': tau,
        'known_count': len(known),
        'unknown_count': len(unknown),
        'q_sizes': Q_SIZES[ds],
        'queries': {},
    }
    for q in Q_SIZES[ds]:
        q = int(q)
        qk, qu = G.pick_rows(known, unknown, q)
        rows = []
        print(f'[{ds}] q={q} tau={tau} attacks={len(ATTACKS)}', flush=True)
        for ai, (label, attack_name) in enumerate(ATTACKS, 1):
            t0 = time.perf_counter()
            print(f'[{ds}] q={q} attack {ai:02d}/{len(ATTACKS)} {label}', flush=True)
            kemb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64).astype(np.float32)
            uemb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64).astype(np.float32)
            query = np.ascontiguousarray(np.concatenate([kemb, uemb], axis=0).astype(np.float32))
            dist = R.exact_nearest_gpu(bank, query, weight, query_chunk=512)
            known_acc, unknown_acc, total_acc = acc_from_dist(dist, q, tau)
            rows.append({
                'attack': label,
                'tau': round(tau, 10),
                'direct_known': round(known_acc, 2),
                'direct_unknown': round(unknown_acc, 2),
                'direct_total': round(total_acc, 2),
                'min_known_distance': round(float(dist[:q].min()), 10),
                'max_known_distance': round(float(dist[:q].max()), 10),
                'min_unknown_distance': round(float(dist[q:].min()), 10),
                'max_unknown_distance': round(float(dist[q:].max()), 10),
                'direct_ms': round((time.perf_counter() - t0) * 1000.0 / (2 * q), 6),
            })
        payload['queries'][str(q)] = {'summary': summarize(rows), 'rows': rows}
        out = OUT / f'{ds}_latest_tau_allq_attack_table_direct.json'
        out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(f'[{ds}] q={q} summary {payload["queries"][str(q)]["summary"]}', flush=True)
    out = OUT / f'{ds}_latest_tau_allq_attack_table_direct.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[saved] {out}', flush=True)
    return payload


def main():
    datasets = sys.argv[1:] or ['cifar', 'flickr', 'ucid', 'amazon']
    all_payloads = {}
    for ds in datasets:
        all_payloads[ds] = run(ds)
        existing = {}
        for p in OUT.glob('*_latest_tau_allq_attack_table_direct.json'):
            try:
                data = json.load(open(p, 'r', encoding='utf-8'))
                existing[data['dataset']] = data
            except Exception:
                pass
        md = write_markdown(existing)
        print(f'[markdown] {md}', flush=True)

if __name__ == '__main__':
    main()
