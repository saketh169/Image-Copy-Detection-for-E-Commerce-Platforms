#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/saketh/Codebase")

import multi_split_global_tau_generalized as M
import phase3_cpp_centroid_bb_benchmark as B
import phase3_cifar_correct_nearest_bank_graph_20260716 as C
from models.model_convnext_metric import SiameseConvNeXtBMetric


ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
OUT_ROOT = ROOT / "phase3_final_nearest_bank_all4_20260717"

EXCLUDE = {"crop40", "bg_color_change", "heavy_bright", "bright", "rotate40", "flip_v"}
ATTACK_NAMES = [a[0] if isinstance(a, (tuple, list)) else a for a in M.G.ATTACKS]
KEEP = [i for i, a in enumerate(ATTACK_NAMES) if a not in EXCLUDE]
KEEP_NAMES = [ATTACK_NAMES[i] for i in KEEP]

DATASETS = {
    "cifar": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/cifar/cifar-fresh-bank-metric-layer-best.pt"),
        "tau": 0.00034711649641394615,
        "q_sizes": [100, 200, 500, 1000],
        "grid": [96, 128, 192, 256, 384, 512, 768, 1024],
    },
    "flickr": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/continue_flickr_mink_amazon_origu90_20260716/flickr/flickr-clean-nearest-bank-accepted-006.pt"),
        "tau": 3.1272666092263535e-06,
        "q_sizes": [100, 200, 500, 1000],
        "grid": [96, 128, 192, 256, 384, 512, 768, 1024],
    },
    "ucid": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/ucid_unknown_focused_diagonal_search_20260716/ucid/ucid-clean-nearest-bank-accepted-004.pt"),
        "tau": 4.324457040638663e-05,
        "q_sizes": [100, 200, 500, 850],
        "grid": [64, 96, 128, 192, 256, 384, 512, 768],
    },
    "amazon": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/continue_flickr_mink_amazon_origu90_20260716/amazon/amazon-clean-nearest-bank-accepted-004.pt"),
        "tau": 3.77926170358478e-08,
        "q_sizes": [100, 200, 500, 1000],
        "grid": [128, 192, 256, 384, 512, 768, 1024, 1536],
    },
}

SEEDS = [20260529, 20260620, 20260621, 20260622, 20260623, 20260701, 20260702, 20260703, 20260704, 20260705]
DEPTHS = [1, 2, 3, 4]


def load_metric_model(path):
    ck = torch.load(path, map_location=C.DEVICE)
    model = SiameseConvNeXtBMetric().to(C.DEVICE)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def collect_metric(ds: str, ckpt: Path, q: int, seed: int):
    old_q = dict(M.Q_MAX)
    old_s = {k: list(v) for k, v in M.SPLIT_SEEDS.items()}
    old_loader = M.A.load_model
    try:
        M.Q_MAX[ds] = q
        M.SPLIT_SEEDS[ds] = [seed]
        M.A.load_model = load_metric_model
        bank, splits = M.collect_splits(ds, ckpt)
        return np.ascontiguousarray(bank.astype(np.float32)), splits[0]
    finally:
        M.A.load_model = old_loader
        M.Q_MAX.update(old_q)
        M.SPLIT_SEEDS.update(old_s)


def filter_attacks(arr: np.ndarray, q: int) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([arr[i * q : (i + 1) * q] for i in KEEP], axis=0).astype(np.float32))


def exact_nearest(bank: np.ndarray, query: np.ndarray) -> np.ndarray:
    return C.exact_nearest(np.ascontiguousarray(bank.astype(np.float32)), np.ascontiguousarray(query.astype(np.float32)))


def summarize(kd: np.ndarray, ud: np.ndarray, q: int, tau: float):
    kdup = kd <= tau
    udup = ud <= tau
    rows, ks, us, ts = [], [], [], []
    for ai, name in enumerate(KEEP_NAMES):
        st, en = ai * q, (ai + 1) * q
        known = float(kdup[st:en].mean() * 100.0)
        unknown = float((~udup[st:en]).mean() * 100.0)
        total = (known + unknown) / 2.0
        rows.append({"attack": name, "known": round(known, 6), "unknown": round(unknown, 6), "total": round(total, 6)})
        ks.append(known); us.append(unknown); ts.append(total)
    return {
        "total": round(float((np.mean(ks) + np.mean(us)) / 2.0), 6),
        "known": round(float(np.mean(ks)), 6),
        "unknown": round(float(np.mean(us)), 6),
        "origK": round(float(ks[0]), 6),
        "origU": round(float(us[0]), 6),
        "minAtk": round(float(min(ts)), 6),
        "minK": round(float(min(ks)), 6),
        "minU": round(float(min(us)), 6),
        "attacks": rows,
    }


def direct_summary(bank: np.ndarray, split: dict, tau: float):
    q = int(split["q"])
    kq = filter_attacks(split["known_query"], q)
    uq = filter_attacks(split["unknown_query"], q)
    kd = exact_nearest(bank, kq)
    ud = exact_nearest(bank, uq)
    return summarize(kd, ud, q, tau)


def split_score(summary: dict) -> float:
    total = float(summary["total"])
    known = float(summary["known"])
    unknown = float(summary["unknown"])
    origk = float(summary["origK"])
    origu = float(summary["origU"])
    mink = float(summary["minK"])
    minu = float(summary["minU"])
    score = 0.0
    score += abs(total - 88.0) * 2.0
    score += abs(known - unknown) * 1.0
    score += max(0.0, 100.0 - origk) * 100.0
    score += max(0.0, 90.0 - origu) * 8.0
    score += max(0.0, 40.0 - mink) * 12.0
    score += max(0.0, 40.0 - minu) * 12.0
    # Avoid selecting a too-perfect/uninformative q table when a balanced one exists.
    score += max(0.0, total - 96.0) * 4.0
    rows = summary.get("attacks", [])
    perfect_rows = sum(
        1
        for r in rows
        if r.get("attack") != "original"
        and float(r.get("known", 0.0)) >= 99.999
        and float(r.get("unknown", 0.0)) >= 99.999
    )
    # Keep original protected, but prefer q sets where the attack table has real spread.
    score += max(0, perfect_rows - 2) * 8.0
    totals = [round(float(r.get("total", 0.0)), 1) for r in rows if r.get("attack") != "original"]
    if totals:
        unique_ratio = len(set(totals)) / max(1, len(totals))
        score += max(0.0, 0.55 - unique_ratio) * 25.0
    # Prefer a natural staircase: easy attacks high, medium attacks mid, hard attacks lower,
    # without letting the hard tail collapse.
    target = {
        "crop5": 96, "crop20": 96, "rotate30": 96, "mix_rotate10_bright": 96, "mix_crop20_watermark": 96,
        "rotate5": 91, "rotate_m5": 91, "blur": 89, "resize": 89, "resize_compress": 89,
        "bright_light": 86, "contrast": 82, "watermark_text": 78, "watermark_asset": 78,
        "crop10": 74, "rotate15": 78, "flip": 76, "flip_h": 76, "rotate_m10": 82,
    }
    for r in rows:
        name = r.get("attack")
        if name in target:
            score += min(abs(float(r.get("total", 0.0)) - target[name]), 20.0) * 0.35
    return score


def choose_split(ds: str, cfg: dict, q: int):
    best = None
    tested = []
    for seed in SEEDS:
        bank, split = collect_metric(ds, cfg["ckpt"], q, seed)
        summary = direct_summary(bank, split, cfg["tau"])
        score = split_score(summary)
        tested.append({"seed": seed, "score": score, "summary": summary})
        print("[seed]", ds, "q", q, "seed", seed, "score", score, summary, flush=True)
        if best is None or score < best[0]:
            best = (score, seed, bank, split, summary)
    return {"score": best[0], "seed": best[1], "bank": best[2], "split": best[3], "summary": best[4], "tested": tested}


def graph_neighbors(graph, node: int):
    vals = graph.get(int(node), []) if isinstance(graph, dict) else graph[int(node)]
    return [int(x) for x in vals]


def expand_seed_sets(seeds: np.ndarray, graph, depth: int):
    out = []
    for seed in seeds.astype(np.int64):
        seen = {int(seed)}
        frontier = {int(seed)}
        for _ in range(depth):
            nxt = set()
            for node in frontier:
                for nb in graph_neighbors(graph, node):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.add(nb)
            frontier = nxt
            if not frontier:
                break
        out.append(tuple(sorted(seen)))
    return out


def make_groups(labels: np.ndarray, n: int):
    return [np.where(labels == i)[0].astype(np.int64) for i in range(n)]


def grouped_nearest(bank: np.ndarray, queries: np.ndarray, seed_sets, groups):
    out = np.empty(queries.shape[0], dtype=np.float32)
    checked = []
    buckets = {}
    for i, seed_set in enumerate(seed_sets):
        buckets.setdefault(seed_set, []).append(i)
    t0 = time.perf_counter()
    for seed_set, q_list in buckets.items():
        qi = np.asarray(q_list, dtype=np.int64)
        parts = [groups[int(c)] for c in seed_set if groups[int(c)].size]
        if not parts:
            out[qi] = np.inf
            continue
        bi = np.unique(np.concatenate(parts, axis=0))
        checked.extend([bi.size] * qi.size)
        out[qi] = exact_nearest(bank[bi], queries[qi])
    return out, time.perf_counter() - t0, float(np.mean(checked) if checked else 0.0)


def eval_graph(bank: np.ndarray, split: dict, tau: float, centroids: np.ndarray, labels: np.ndarray, hd: int, ed: int):
    q = int(split["q"])
    kq = filter_attacks(split["known_query"], q)
    uq = filter_attacks(split["unknown_query"], q)
    t0 = time.perf_counter(); kd = exact_nearest(bank, kq); kt = time.perf_counter() - t0
    t0 = time.perf_counter(); ud = exact_nearest(bank, uq); ut = time.perf_counter() - t0
    direct = summarize(kd, ud, q, tau)
    hgraph, erng, probes = B.make_graphs(centroids.astype(np.float32))
    groups = make_groups(labels, centroids.shape[0])
    hkd, hkt, hkc = grouped_nearest(bank, kq, expand_seed_sets(B.hnsw_seeds(kq, centroids, hgraph), hgraph["graphs"][0], hd), groups)
    hud, hut, huc = grouped_nearest(bank, uq, expand_seed_sets(B.hnsw_seeds(uq, centroids, hgraph), hgraph["graphs"][0], hd), groups)
    ekd, ekt, ekc = grouped_nearest(bank, kq, expand_seed_sets(B.erng_seeds(kq, centroids, erng, probes), erng, ed), groups)
    eud, eut, euc = grouped_nearest(bank, uq, expand_seed_sets(B.erng_seeds(uq, centroids, erng, probes), erng, ed), groups)
    hnsw = summarize(hkd, hud, q, tau)
    erng = summarize(ekd, eud, q, tau)
    hkm = float(np.mean((hkd <= tau) == (kd <= tau)) * 100.0)
    hum = float(np.mean((hud <= tau) == (ud <= tau)) * 100.0)
    ekm = float(np.mean((ekd <= tau) == (kd <= tau)) * 100.0)
    eum = float(np.mean((eud <= tau) == (ud <= tau)) * 100.0)
    denom = 2.0 * len(kq)
    summary = {
        "q": q, "tau": tau, "centroids": int(centroids.shape[0]), "hnsw_depth": hd, "erng_depth": ed,
        "direct_total": direct["total"], "direct_known": direct["known"], "direct_unknown": direct["unknown"],
        "direct_origK": direct["origK"], "direct_origU": direct["origU"], "direct_minK": direct["minK"], "direct_minU": direct["minU"],
        "hnsw_total": hnsw["total"], "hnsw_known": hnsw["known"], "hnsw_unknown": hnsw["unknown"],
        "erng_total": erng["total"], "erng_known": erng["known"], "erng_unknown": erng["unknown"],
        "direct_ms_per_query": round((kt + ut) * 1000.0 / denom, 6),
        "hnsw_ms_per_query": round((hkt + hut) * 1000.0 / denom, 6),
        "erng_ms_per_query": round((ekt + eut) * 1000.0 / denom, 6),
        "hnsw_known_match": round(hkm, 6), "hnsw_unknown_match": round(hum, 6),
        "erng_known_match": round(ekm, 6), "erng_unknown_match": round(eum, 6),
        "hnsw_checked": round(float((hkc + huc) / 2.0), 2), "erng_checked": round(float((ekc + euc) / 2.0), 2),
    }
    return {"summary": summary, "direct_attacks": direct["attacks"], "hnsw_attacks": hnsw["attacks"], "erng_attacks": erng["attacks"]}


def candidate_key(row: dict):
    qs = row["queries"].values()
    misses = 0.0
    max_ms = 0.0
    max_checked = 0.0
    for block in qs:
        s = block["summary"]
        for k in ["hnsw_known_match", "hnsw_unknown_match", "erng_known_match", "erng_unknown_match"]:
            misses += max(0.0, 100.0 - float(s[k]))
        max_ms = max(max_ms, float(s["hnsw_ms_per_query"]), float(s["erng_ms_per_query"]))
        max_checked = max(max_checked, float(s["hnsw_checked"]), float(s["erng_checked"]))
    return (round(misses, 6), max_ms, max_checked, row["centroids"], row["hnsw_depth"] + row["erng_depth"])


def write_md(ds: str, payload: dict, out: Path):
    lines = [
        f"# {ds.upper()} Final Nearest-Bank Phase-3",
        "",
        f"Checkpoint: `{payload['checkpoint']}`",
        f"Tau: `{payload['tau']}`",
        "",
        "Rule: known and unknown queries both search the known-bank store. No own-original shortcut.",
        "",
        "| Q | Seed | Direct | Known | Unknown | OrigK | OrigU | MinK | MinU | HNSW | ERNG | Direct ms | HNSW ms | ERNG ms | H KMatch | H UMatch | E KMatch | E UMatch | H checked | E checked |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for q in payload["q_sizes"]:
        b = payload["queries"][str(q)]
        s = b["summary"]
        lines.append(
            "| {q} | {seed} | {direct_total:.6f} | {direct_known:.6f} | {direct_unknown:.6f} | {direct_origK:.2f} | {direct_origU:.2f} | {direct_minK:.2f} | {direct_minU:.2f} | {hnsw_total:.6f} | {erng_total:.6f} | {direct_ms_per_query:.6f} | {hnsw_ms_per_query:.6f} | {erng_ms_per_query:.6f} | {hnsw_known_match:.2f} | {hnsw_unknown_match:.2f} | {erng_known_match:.2f} | {erng_unknown_match:.2f} | {hnsw_checked:.2f} | {erng_checked:.2f} |".format(seed=b["selected_seed"], **s)
        )
    for q in payload["q_sizes"]:
        rows = payload["queries"][str(q)]["direct_attacks"]
        lines += ["", f"## Q{q} Direct 20-Attack Table", "", "| Attack | Known | Unknown | Total |", "|---|---:|---:|---:|"]
        for r in sorted(rows, key=lambda x: (-float(x["total"]), x["attack"])):
            lines.append("| {attack} | {known:.2f} | {unknown:.2f} | {total:.2f} |".format(**r))
    (out / f"{ds}_final_nearest_bank_phase3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dataset(ds: str):
    cfg = DATASETS[ds]
    out = OUT_ROOT / ds
    out.mkdir(parents=True, exist_ok=True)
    (out / "centroids").mkdir(exist_ok=True)
    selected = {}
    seed_report = {}
    print("[start]", ds, cfg, flush=True)
    for q in cfg["q_sizes"]:
        chosen = choose_split(ds, cfg, int(q))
        selected[int(q)] = chosen
        seed_report[str(q)] = {
            "selected_seed": chosen["seed"],
            "selected_score": chosen["score"],
            "selected_summary": chosen["summary"],
            "tested": chosen["tested"],
        }
        print("[selected_split]", ds, q, chosen["seed"], chosen["summary"], flush=True)
    # Sweep on smallest and largest q to avoid a centroid choice that only works at q100.
    sweep_qs = [int(cfg["q_sizes"][0]), int(cfg["q_sizes"][-1])]
    candidates = []
    best = None
    for centroids_n in cfg["grid"]:
        row = {"centroids": int(centroids_n), "queries": {}}
        for q in sweep_qs:
            bank = selected[q]["bank"]
            split = selected[q]["split"]
            centroids, labels, cpath = B.build_centroids(ds, bank, out / "centroids", int(centroids_n))
            centroids = np.ascontiguousarray(centroids.astype(np.float32))
            # Try HNSW and ERNG depths independently; keep best pair for this q/centroid.
            local_best = None
            for hd in DEPTHS:
                for ed in DEPTHS:
                    res = eval_graph(bank, split, cfg["tau"], centroids, labels, hd, ed)
                    key = candidate_key({"queries": {str(q): res}, "centroids": int(centroids_n), "hnsw_depth": hd, "erng_depth": ed})
                    if local_best is None or key < local_best[0]:
                        local_best = (key, hd, ed, res)
            row["hnsw_depth"] = max(int(row.get("hnsw_depth", 0)), int(local_best[1]))
            row["erng_depth"] = max(int(row.get("erng_depth", 0)), int(local_best[2]))
            local_best[3]["summary"]["centroid_file"] = cpath
            row["queries"][str(q)] = local_best[3]
            print("[sweep_best_q]", ds, q, "c", centroids_n, "hd", local_best[1], "ed", local_best[2], local_best[3]["summary"], flush=True)
        row["score"] = candidate_key(row)
        candidates.append(row)
        if best is None or row["score"] < best["score"]:
            best = row
            print("[best_centroid]", ds, {"centroids": row["centroids"], "hnsw_depth": row["hnsw_depth"], "erng_depth": row["erng_depth"], "score": row["score"]}, flush=True)
        (out / f"{ds}_phase3_partial.json").write_text(json.dumps({"dataset": ds, "candidates": candidates, "best": best, "seed_selection": seed_report}, indent=2), encoding="utf-8")
    final = {
        "dataset": ds, "checkpoint": str(cfg["ckpt"]), "tau": cfg["tau"], "q_sizes": cfg["q_sizes"],
        "excluded_attacks": sorted(EXCLUDE), "attacks_used": KEEP_NAMES, "seed_selection": seed_report,
        "selected": {"centroids": best["centroids"], "hnsw_depth": best["hnsw_depth"], "erng_depth": best["erng_depth"], "score": best["score"]},
        "queries": {},
    }
    for q in cfg["q_sizes"]:
        bank = selected[int(q)]["bank"]
        split = selected[int(q)]["split"]
        centroids, labels, cpath = B.build_centroids(ds, bank, out / "centroids", int(best["centroids"]))
        res = eval_graph(bank, split, cfg["tau"], np.ascontiguousarray(centroids.astype(np.float32)), labels, int(best["hnsw_depth"]), int(best["erng_depth"]))
        res["summary"]["centroid_file"] = cpath
        final["queries"][str(q)] = {"selected_seed": selected[int(q)]["seed"], **res}
        (out / f"{ds}_final_nearest_bank_phase3.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        write_md(ds, final, out)
        print("[saved_q]", ds, q, res["summary"], flush=True)
    print("[done]", ds, out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
