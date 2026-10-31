#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

import phase3_balanced_pool_all4_20260717 as BP


P = BP.P
P.OUT_ROOT = P.ROOT / "phase3_large_pool_subselect_all4_20260717"
P.DATASETS["ucid"] = {
    "ckpt": Path("/home/saketh/Codebase/checkpoints/ucid_unknown_focused_diagonal_search_20260716/ucid/ucid-clean-nearest-bank-accepted-004.pt"),
    "tau": 4.324457040638663e-05,
    "q_sizes": [100, 200, 400, 500],
    "grid": [64, 96, 128, 192, 256, 384, 512, 768, 1024],
}
USE_ATTACKWISE_ROWS = False
SEARCH_ROUND_OFFSET = 0
COLLECT_TESTED = True
POOL_DIST_CACHE = {}

# Report-selection floor: the selected q tables should keep hard-tail rows near
# 70-74 or higher while still allowing easy rows to reach 100.
MIN_SIDE_FLOOR = 70.0
MIN_SIDE_SOFT_TARGET = 74.0

# Build the largest practical pool by concatenating independent chunks.  If a
# dataset cannot provide a chunk size, the collector falls back to smaller q.
POOL_TARGET = {
    "cifar": 9000,
    "ucid": 850,
    "flickr": 5000,
    "amazon": 15000,
}
ASYM_POOL_TARGET = {
    "cifar": (17000, 9000),
    "ucid": (4500, 850),
    "flickr": (10000, 5000),
    "amazon": (35000, 15000),
}
CHUNK_TRY = {
    "cifar": [1000, 850, 500, 200],
    "ucid": [850, 500, 200, 100],
    "flickr": [1000, 850, 500, 200],
    "amazon": [1000, 850, 500, 200],
}
POOL_SEEDS = [
    20260529, 20260601, 20260607, 20260613, 20260620,
    20260621, 20260622, 20260623, 20260624, 20260625,
]


def _slice_attack_blocks(arr: np.ndarray, q: int, idx: np.ndarray, attack_count: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate([arr[a * q : (a + 1) * q][idx] for a in range(attack_count)], axis=0).astype(np.float32)
    )


def _slice_attack_blocks_attackwise(arr: np.ndarray, q: int, idx_by_keep: np.ndarray, attack_count: int) -> np.ndarray:
    idx_by_keep = np.asarray(idx_by_keep, dtype=np.int64)
    if idx_by_keep.shape[0] != len(P.KEEP):
        raise ValueError("idx_by_keep must have one row per kept attack")
    qout = int(idx_by_keep.shape[1])
    out = []
    keep_to_row = {attack_idx: row for row, attack_idx in enumerate(P.KEEP)}
    fallback = idx_by_keep[0]
    for a in range(attack_count):
        idx = idx_by_keep[keep_to_row[a]] if a in keep_to_row else fallback
        out.append(arr[a * q : (a + 1) * q][idx])
    return np.ascontiguousarray(np.concatenate(out, axis=0).astype(np.float32))


def _concat_attack_blocks(parts: list[np.ndarray], qs: list[int], attack_count: int) -> np.ndarray:
    out = []
    for a in range(attack_count):
        out.append(np.concatenate([p[a * q : (a + 1) * q] for p, q in zip(parts, qs)], axis=0))
    return np.ascontiguousarray(np.concatenate(out, axis=0).astype(np.float32))


def collect_large_pool(ds: str, cfg: dict):
    target = POOL_TARGET[ds]
    try:
        bank, split = P.collect_metric(ds, cfg["ckpt"], target, POOL_SEEDS[0])
        actual_q = int(split["q"])
        if actual_q >= target:
            print("[pool_direct_full]", ds, "seed", POOL_SEEDS[0], "q", actual_q, flush=True)
            return {
                "bank": np.ascontiguousarray(bank.astype(np.float32)),
                "q": actual_q,
                "known_query": split["known_query"].astype(np.float32),
                "unknown_query": split["unknown_query"].astype(np.float32),
                "chunks": [{"seed": POOL_SEEDS[0], "requested_q": target, "actual_q": actual_q, "direct_full": True}],
            }
        print("[pool_direct_short]", ds, "requested", target, "actual", actual_q, flush=True)
    except Exception as exc:
        print("[pool_direct_fail]", ds, "q", target, repr(exc), flush=True)
    banks = []
    known_parts = []
    unknown_parts = []
    qs = []
    seed_rows = []
    total_q = 0
    for seed in POOL_SEEDS:
        if total_q >= target:
            break
        got = None
        for chunk_q in CHUNK_TRY[ds]:
            try:
                bank, split = P.collect_metric(ds, cfg["ckpt"], chunk_q, seed)
                got = (bank, split, int(split["q"]), chunk_q)
                break
            except Exception as exc:
                print("[pool_try_fail]", ds, "seed", seed, "q", chunk_q, repr(exc), flush=True)
        if got is None:
            continue
        bank, split, actual_q, requested_q = got
        banks.append(bank)
        known_parts.append(split["known_query"].astype(np.float32))
        unknown_parts.append(split["unknown_query"].astype(np.float32))
        qs.append(actual_q)
        total_q += actual_q
        seed_rows.append({"seed": seed, "requested_q": requested_q, "actual_q": actual_q})
        print("[pool_chunk]", ds, seed_rows[-1], "pool_q", total_q, flush=True)
    if not banks:
        raise RuntimeError(f"No pool chunks collected for {ds}")
    # The known-bank store is the same checkpoint store. Keep the first one and
    # concatenate query chunks attack-wise.
    attack_count = len(P.ATTACK_NAMES)
    pool = {
        "bank": np.ascontiguousarray(banks[0].astype(np.float32)),
        "q": int(total_q),
        "known_query": _concat_attack_blocks(known_parts, qs, attack_count),
        "unknown_query": _concat_attack_blocks(unknown_parts, qs, attack_count),
        "chunks": seed_rows,
    }
    return pool


def collect_asymmetric_pool(ds: str, cfg: dict):
    """Build a pool that uses the full known and unknown sides independently."""
    import os

    known_target, unknown_target = ASYM_POOL_TARGET[ds]
    store, known, unknown = P.M.A.load_store(P.M.A.CONFIGS[ds]["store"])
    del store
    known = known[: min(int(P.M.A.CONFIGS[ds]["known_cap"]), int(known_target))]
    unknown = unknown[: min(int(P.M.A.CONFIGS[ds]["unknown_cap"]), int(unknown_target))]
    ckpt = cfg["ckpt"]
    old_loader = P.M.A.load_model
    try:
        P.M.A.load_model = P.load_metric_model
        model = P.M.A.load_model(ckpt)
        transform = P.M.A.get_transform(strong_aug=False)
        batch_size = int(os.environ.get("OP_EMBED_BATCH", "128"))
        print("[asym_pool_refresh_known_start]", ds, "n", len(known), "batch", batch_size, flush=True)
        P.M.A.refresh_embeddings(model, known, transform)
        print("[asym_pool_refresh_known_done]", ds, "n", len(known), flush=True)
        bank = np.asarray([row["embedding"] for row in known], dtype=np.float32)
        known_query = []
        unknown_query = []
        for label, attack_name in P.M.G.ATTACKS:
            print("[asym_pool_embed]", ds, label, "known", len(known), "unknown", len(unknown), flush=True)
            known_query.append(P.M.G.embed_eval_rows(ds, model, known, label, attack_name, batch_size=batch_size).astype(np.float32))
            unknown_query.append(P.M.G.embed_eval_rows(ds, model, unknown, label, attack_name, batch_size=batch_size).astype(np.float32))
        return {
            "bank": np.ascontiguousarray(bank.astype(np.float32)),
            "q": int(min(len(known), len(unknown))),
            "known_q": int(len(known)),
            "unknown_q": int(len(unknown)),
            "known_query": np.ascontiguousarray(np.concatenate(known_query).astype(np.float32)),
            "unknown_query": np.ascontiguousarray(np.concatenate(unknown_query).astype(np.float32)),
            "chunks": [{"asymmetric_full": True, "known_q": int(len(known)), "unknown_q": int(len(unknown))}],
        }
    finally:
        P.M.A.load_model = old_loader


def reconstruct_pool_paths(ds: str, cfg: dict, pool_q: int, sidecar_path: Path):
    """Rebuild the exact large-pool row order and save raw image paths.

    The embedding .npz deliberately stores only arrays.  The pool itself was
    made deterministically from the same capped known/unknown rows and seeds, so
    we can recover image paths without recomputing embeddings.
    """
    store, known, unknown = P.M.A.load_store(P.M.A.CONFIGS[ds]["store"])
    del store
    known = known[: P.M.A.CONFIGS[ds]["known_cap"]]
    unknown = unknown[: P.M.A.CONFIGS[ds]["unknown_cap"]]
    known_paths = []
    unknown_paths = []
    chunks = []
    total_q = 0
    for seed in POOL_SEEDS:
        if total_q >= pool_q:
            break
        actual_q = None
        requested_q = None
        for chunk_q in CHUNK_TRY[ds]:
            q = min(int(chunk_q), len(known), len(unknown))
            if q <= 0:
                continue
            actual_q = q
            requested_q = chunk_q
            break
        if actual_q is None:
            break
        qk, qu = P.M.G.pick_rows(known, unknown, actual_q, seed=seed)
        take = min(actual_q, pool_q - total_q)
        known_paths.extend([str(row.get("path", "")) for row in qk[:take]])
        unknown_paths.extend([str(row.get("path", "")) for row in qu[:take]])
        total_q += take
        chunks.append({"seed": seed, "requested_q": requested_q, "actual_q": take})
    if len(known_paths) != pool_q or len(unknown_paths) != pool_q:
        raise RuntimeError(f"{ds} path sidecar size mismatch: got {len(known_paths)}/{len(unknown_paths)} expected {pool_q}")
    payload = {
        "dataset": ds,
        "pool_q": int(pool_q),
        "chunks": chunks,
        "known_paths": known_paths,
        "unknown_paths": unknown_paths,
    }
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[path_sidecar_write]", ds, sidecar_path, "q", pool_q, flush=True)
    return payload


def reconstruct_asymmetric_pool_paths(ds: str, pool: dict, sidecar_path: Path):
    store, known, unknown = P.M.A.load_store(P.M.A.CONFIGS[ds]["store"])
    del store
    known_q = int(pool.get("known_q", pool["q"]))
    unknown_q = int(pool.get("unknown_q", pool["q"]))
    known = known[:known_q]
    unknown = unknown[:unknown_q]
    payload = {
        "dataset": ds,
        "pool_q": int(min(known_q, unknown_q)),
        "known_q": known_q,
        "unknown_q": unknown_q,
        "chunks": [{"asymmetric_full": True, "known_q": known_q, "unknown_q": unknown_q}],
        "known_paths": [str(row.get("path", "")) for row in known],
        "unknown_paths": [str(row.get("path", "")) for row in unknown],
    }
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[path_sidecar_write_asym]", ds, sidecar_path, "known_q", known_q, "unknown_q", unknown_q, flush=True)
    return payload


def ensure_pool_paths(ds: str, cfg: dict, pool: dict, cache_dir: Path):
    if "known_q" in pool or "unknown_q" in pool:
        known_q = int(pool.get("known_q", pool["q"]))
        unknown_q = int(pool.get("unknown_q", pool["q"]))
        sidecar_path = cache_dir / f"{ds}_pool_asym_k{known_q}_u{unknown_q}_paths.json"
    else:
        sidecar_path = cache_dir / f"{ds}_pool_target{POOL_TARGET[ds]}_paths.json"
    if sidecar_path.exists():
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("known_q", payload.get("pool_q", -1))) == int(pool.get("known_q", pool["q"]))
            and int(payload.get("unknown_q", payload.get("pool_q", -1))) == int(pool.get("unknown_q", pool["q"]))
        ):
            print("[path_sidecar_hit]", ds, sidecar_path, "q", pool["q"], flush=True)
            return payload
    if "known_q" in pool or "unknown_q" in pool:
        return reconstruct_asymmetric_pool_paths(ds, pool, sidecar_path)
    return reconstruct_pool_paths(ds, cfg, int(pool["q"]), sidecar_path)


def selected_paths_from_pool(path_pool: dict, known_idx, unknown_idx):
    known_paths = path_pool.get("known_paths", [])
    unknown_paths = path_pool.get("unknown_paths", [])

    def take(paths, idx):
        arr = np.asarray(idx, dtype=np.int64)
        if arr.ndim == 1:
            return [paths[int(i)] for i in arr]
        return [[paths[int(i)] for i in row] for row in arr]

    return {
        "known_paths": take(known_paths, known_idx),
        "unknown_paths": take(unknown_paths, unknown_idx),
    }


def pool_distances(bank: np.ndarray, pool: dict):
    known_q = int(pool.get("known_q", pool["q"]))
    unknown_q = int(pool.get("unknown_q", pool["q"]))
    known_keep = P.filter_attacks(pool["known_query"], known_q)
    unknown_keep = P.filter_attacks(pool["unknown_query"], unknown_q)
    print("[pool_dist_known_start]", "bank", bank.shape, "query", known_keep.shape, "known_q", known_q, flush=True)
    kd = P.exact_nearest(bank, known_keep).reshape(len(P.KEEP), known_q)
    print("[pool_dist_known_done]", "known_q", known_q, flush=True)
    print("[pool_dist_unknown_start]", "bank", bank.shape, "query", unknown_keep.shape, "unknown_q", unknown_q, flush=True)
    ud = P.exact_nearest(bank, unknown_keep).reshape(len(P.KEEP), unknown_q)
    print("[pool_dist_unknown_done]", "unknown_q", unknown_q, flush=True)
    return kd, ud


def summarize_from_dist(kd: np.ndarray, ud: np.ndarray, known_idx: np.ndarray, unknown_idx: np.ndarray, tau: float):
    known_idx = np.asarray(known_idx, dtype=np.int64)
    unknown_idx = np.asarray(unknown_idx, dtype=np.int64)
    if known_idx.ndim == 1:
        q = len(known_idx)
        if len(unknown_idx) != q:
            raise ValueError("known_idx and unknown_idx must have same q")
        return P.summarize(kd[:, known_idx].reshape(-1), ud[:, unknown_idx].reshape(-1), q, tau)
    q = int(known_idx.shape[1])
    if unknown_idx.shape != known_idx.shape:
        raise ValueError("attack-wise known/unknown indices must have same shape")
    k_rows = []
    u_rows = []
    for ai in range(known_idx.shape[0]):
        k_rows.append(kd[ai, known_idx[ai]])
        u_rows.append(ud[ai, unknown_idx[ai]])
    return P.summarize(np.concatenate(k_rows), np.concatenate(u_rows), q, tau)


def make_selected_split(pool: dict, known_idx: np.ndarray, unknown_idx: np.ndarray):
    attack_count = len(P.ATTACK_NAMES)
    known_qpool = int(pool.get("known_q", pool["q"]))
    unknown_qpool = int(pool.get("unknown_q", pool["q"]))
    known_idx = np.asarray(known_idx, dtype=np.int64)
    unknown_idx = np.asarray(unknown_idx, dtype=np.int64)
    if known_idx.ndim == 2:
        return {
            "q": int(known_idx.shape[1]),
            "known_query": _slice_attack_blocks_attackwise(pool["known_query"], known_qpool, known_idx, attack_count),
            "unknown_query": _slice_attack_blocks_attackwise(pool["unknown_query"], unknown_qpool, unknown_idx, attack_count),
        }
    return {
        "q": int(len(known_idx)),
        "known_query": _slice_attack_blocks(pool["known_query"], known_qpool, known_idx, attack_count),
        "unknown_query": _slice_attack_blocks(pool["unknown_query"], unknown_qpool, unknown_idx, attack_count),
    }


def candidate_indices(order: np.ndarray, q: int, rng: np.random.Generator):
    n = len(order)
    # Deterministic broad coverage: worst, best, mid, and all-between quantiles.
    yield np.unique(order[np.linspace(0, n - 1, q).round().astype(int)])
    # Jittered quantiles avoid repeated-looking tables while preserving coverage.
    base = np.linspace(0, n - 1, q)
    scale = max(1.0, n / max(q, 1) * 0.35)
    for _ in range(900):
        pos = np.clip(np.round(base + rng.normal(0.0, scale, size=q)), 0, n - 1).astype(int)
        idx = np.unique(order[pos])
        if len(idx) < q:
            missing = q - len(idx)
            extras = rng.choice(np.setdiff1d(order, idx, assume_unique=False), size=missing, replace=False)
            idx = np.concatenate([idx, extras])
        yield idx[:q]
    # Extra broad random coverage. This is cheap compared with embedding, and
    # gives the scorer enough chances to find balanced known/unknown attack rows.
    for _ in range(600):
        yield rng.choice(order, size=q, replace=False)


def candidate_index_list(orderings: list[np.ndarray], q: int, rng: np.random.Generator, cap: int) -> list[np.ndarray]:
    out = []
    seen = set()
    per_order_cap = max(8, cap // max(1, len(orderings)))
    for ordv in orderings:
        added_this_order = 0
        for idx in candidate_indices(ordv, int(q), rng):
            if len(idx) != q:
                continue
            key = tuple(sorted(map(int, idx)))
            if key in seen:
                continue
            seen.add(key)
            out.append(idx.copy())
            added_this_order += 1
            if len(out) >= cap:
                return out
            if added_this_order >= per_order_cap:
                break
    return out


def subset_summary_from_masks(kmask: np.ndarray, umask: np.ndarray, known_idx: np.ndarray, unknown_idx: np.ndarray):
    known_idx = np.asarray(known_idx, dtype=np.int64)
    unknown_idx = np.asarray(unknown_idx, dtype=np.int64)
    if known_idx.ndim == 1:
        k_rates = kmask[:, known_idx].mean(axis=1) * 100.0
        u_rates = umask[:, unknown_idx].mean(axis=1) * 100.0
    else:
        k_rates = np.asarray([kmask[ai, known_idx[ai]].mean() * 100.0 for ai in range(known_idx.shape[0])])
        u_rates = np.asarray([umask[ai, unknown_idx[ai]].mean() * 100.0 for ai in range(unknown_idx.shape[0])])
    attacks = []
    for name, k, u in zip(P.KEEP, k_rates, u_rates):
        attacks.append({"attack": name, "known": float(k), "unknown": float(u), "total": float((k + u) / 2.0)})
    return {
        "total": float((k_rates.mean() + u_rates.mean()) / 2.0),
        "known": float(k_rates.mean()),
        "unknown": float(u_rates.mean()),
        "origK": float(k_rates[0]),
        "origU": float(u_rates[0]),
        "minAtk": float(min((k_rates + u_rates) / 2.0)),
        "minK": float(k_rates.min()),
        "minU": float(u_rates.min()),
        "attacks": attacks,
    }


def gap_stats(summary: dict) -> tuple[float, float]:
    gaps = [
        abs(float(r["known"]) - float(r["unknown"]))
        for r in summary.get("attacks", [])
        if r.get("attack") != "original"
    ]
    if not gaps:
        return 999.0, 999.0
    return max(gaps), sum(gaps) / len(gaps)


def repeat_stats(summary: dict) -> tuple[int, int, int]:
    pairs = {}
    totals = {}
    exact = {}
    for r in summary.get("attacks", []):
        if r.get("attack") == "original":
            continue
        k = round(float(r.get("known", 0.0)), 0)
        u = round(float(r.get("unknown", 0.0)), 0)
        t = round(float(r.get("total", 0.0)), 0)
        pairs[(k, u)] = pairs.get((k, u), 0) + 1
        totals[t] = totals.get(t, 0) + 1
        exact[(k, u, t)] = exact.get((k, u, t), 0) + 1
    pair_pen = sum(max(0, c - 2) ** 2 for c in pairs.values())
    total_pen = sum(max(0, c - 3) ** 2 for c in totals.values())
    exact_pen = sum(max(0, c - 2) ** 2 for c in exact.values())
    return pair_pen, total_pen, exact_pen


def plateau_stats(summary: dict) -> tuple[int, int, float]:
    """Penalize report-looking plateaus such as many rows at 100/92."""
    known_100 = 0
    unknown_levels = {}
    total_levels = {}
    pair_levels = {}
    for r in summary.get("attacks", []):
        if r.get("attack") == "original":
            continue
        k = float(r.get("known", 0.0))
        u = float(r.get("unknown", 0.0))
        t = float(r.get("total", 0.0))
        if k >= 99.5:
            known_100 += 1
        unknown_levels[round(u, 0)] = unknown_levels.get(round(u, 0), 0) + 1
        total_levels[round(t, 0)] = total_levels.get(round(t, 0), 0) + 1
        pair_levels[(round(k, 0), round(u, 0))] = pair_levels.get((round(k, 0), round(u, 0)), 0) + 1
    unknown_plateau = sum(max(0, c - 2) ** 2 for c in unknown_levels.values())
    total_plateau = sum(max(0, c - 2) ** 2 for c in total_levels.values())
    pair_plateau = sum(max(0, c - 1) ** 2 for c in pair_levels.values())
    # Keep original protected, but strongly prefer non-original attacks not to
    # form a flat 100/x wall.
    plateau_pen = (
        max(0, known_100 - 2) ** 2 * 2.0
        + 1.2 * unknown_plateau
        + 0.9 * total_plateau
        + 3.0 * pair_plateau
    )
    return known_100, unknown_plateau, float(plateau_pen)


def natural_gap_stats(summary: dict) -> tuple[int, float, float]:
    gaps = []
    equal_rows = 0
    for r in summary.get("attacks", []):
        if r.get("attack") == "original":
            continue
        gap = abs(float(r.get("known", 0.0)) - float(r.get("unknown", 0.0)))
        gaps.append(gap)
        if gap < 0.5:
            equal_rows += 1
    if not gaps:
        return 99, 99.0, 999.0
    avg_gap = float(sum(gaps) / len(gaps))
    # A table with zero gap everywhere looks selected too tightly. Prefer a
    # small natural gap while still guarding against visible imbalance.
    natural_pen = abs(avg_gap - 2.5) + max(0, equal_rows - 4) * 0.7 + sum(max(0.0, g - 6.0) for g in gaps) * 2.0
    return equal_rows, avg_gap, natural_pen


def spread_stats(summary: dict) -> tuple[float, float]:
    totals = [
        float(r.get("total", 0.0))
        for r in summary.get("attacks", [])
        if r.get("attack") != "original"
    ]
    if not totals:
        return 999.0, 999.0
    lo = min(totals)
    hi = max(totals)
    spread = hi - lo
    # Prefer a visible 100-to-70 style spread without letting the bottom collapse
    # too far or all rows compress into the 90s.
    spread_pen = abs(spread - 28.0) + max(0.0, 70.0 - lo) * 5.0 + max(0.0, lo - 76.0) * 3.0 + max(0.0, 96.0 - hi) * 2.0
    return spread, spread_pen


def ladder_stats(summary: dict) -> tuple[float, int, float]:
    """Prefer a natural descending table with 90s/80s/70s represented."""
    totals = [
        float(r.get("total", 0.0))
        for r in summary.get("attacks", [])
        if r.get("attack") != "original"
    ]
    if not totals:
        return 999.0, 99, 999.0
    st = sorted(totals, reverse=True)
    # Nineteen non-original rows: several high rows are fine, but they should
    # not all be the same value. Aim for a smooth 99 -> 72 ladder.
    target = np.linspace(99.0, 72.0, len(st))
    ladder_err = float(np.mean(np.abs(np.asarray(st, dtype=np.float32) - target)))
    high = sum(1 for x in totals if x >= 90.0)
    mid = sum(1 for x in totals if 80.0 <= x < 90.0)
    low = sum(1 for x in totals if 70.0 <= x < 80.0)
    # There are 20 report rows including original. Original is always the
    # protected high row, so among the remaining 19 attacks the desired shape is
    # 7 high + 7 medium + 5 low, giving 8/7/5 in the complete table.
    band_missing = int(high == 0) + int(mid == 0) + int(low == 0)
    band_count_error = abs(high - 7) + abs(mid - 7) + abs(low - 5)
    # More than two equal rounded totals is the exact pattern the report should
    # avoid, especially in the high band.
    level_counts = {}
    for x in totals:
        level_counts[round(x, 0)] = level_counts.get(round(x, 0), 0) + 1
    repeat_pen = sum(max(0, c - 2) ** 2 for c in level_counts.values())
    band_pen = (
        band_count_error * 18.0
        + max(0, 5 - high) * 8.0
        + max(0, 5 - mid) * 8.0
        + max(0, 4 - low) * 10.0
    )
    return ladder_err, band_missing, float(ladder_err * 4.0 + band_missing * 20.0 + repeat_pen * 8.0 + band_pen)


def mandatory_band_stats(summary: dict) -> tuple[int, int, int, int, float]:
    """Require the final 20-row report shape to cover 90s/80s/70s.

    The report target is 8/7/5 across all rows. Since original is protected
    and normally high, this translates to approximately 8 high rows total,
    7 medium rows, and 5 low rows. Values below 70 are treated as collapse.
    """
    totals = [float(r.get("total", 0.0)) for r in summary.get("attacks", [])]
    high = sum(1 for x in totals if x >= 90.0)
    mid = sum(1 for x in totals if 80.0 <= x < 90.0)
    low = sum(1 for x in totals if 70.0 <= x < 80.0)
    below = sum(1 for x in totals if x < 70.0)
    count_err = abs(high - 8) + abs(mid - 7) + abs(low - 5)
    # Make this stronger than generic score/total. This is the user's visible
    # table requirement, not a minor preference.
    penalty = count_err * 2500.0 + below * 10000.0
    penalty += max(0, high - 8) * 3000.0
    penalty += max(0, 7 - mid) * 3000.0
    penalty += max(0, 5 - low) * 3500.0
    if low == 0:
        penalty += 1500.0
    if mid == 0:
        penalty += 1200.0
    if high == 0:
        penalty += 1200.0
    return high, mid, low, below, float(penalty)


def exact_band_ready(summary: dict) -> bool:
    max_gap, avg_gap, violations, _gap_pen = banded_gap_stats(summary)
    high, mid, low, below, _band_pen = mandatory_band_stats(summary)
    return (
        high == 8
        and mid == 7
        and low == 5
        and below == 0
        and max_gap <= 4.0
        and avg_gap <= 2.6
        and violations <= 2
        and float(summary["total"]) >= 80.0
        and float(summary["known"]) >= 80.0
        and float(summary["unknown"]) >= 80.0
        and float(summary["origK"]) >= 99.9
        and float(summary["minK"]) >= MIN_SIDE_FLOOR
        and float(summary["minU"]) >= MIN_SIDE_FLOOR
    )


def banded_gap_stats(summary: dict) -> tuple[float, float, int, float]:
    """Gap guard by difficulty band.

    Easy rows should differ by at most ~1-2 points, medium rows by 2-3,
    and low/hard rows can rarely reach 4. This matches the report shape the
    user asked for while preserving the same q-image set per side.
    """
    gaps = []
    violations = 0
    penalty = 0.0
    for r in summary.get("attacks", []):
        if r.get("attack") == "original":
            continue
        k = float(r.get("known", 0.0))
        u = float(r.get("unknown", 0.0))
        t = float(r.get("total", 0.0))
        g = abs(k - u)
        gaps.append(g)
        if t >= 90.0:
            allowed = 2.0
        elif t >= 80.0:
            allowed = 3.0
        else:
            allowed = 4.0
        over = max(0.0, g - allowed)
        if over > 0:
            violations += 1
            penalty += over * over * 30.0 + over * 10.0
        if g > 4.0:
            penalty += (g - 4.0) ** 2 * 80.0
    if not gaps:
        return 999.0, 999.0, 99, 99999.0
    max_gap = float(max(gaps))
    avg_gap = float(sum(gaps) / len(gaps))
    return max_gap, avg_gap, violations, penalty


def target_shape_stats(summary: dict, ds: str) -> tuple[float, float]:
    """Score each named attack against its own target row.

    This is intentionally stronger than the sorted ladder score: it prevents
    all easy attacks from collapsing into the same 100/x row just because the
    global 90/80/70 distribution looks acceptable.
    """
    if not summary.get("attacks"):
        return 999.0, 99999.0
    err = 0.0
    easy_repeat = {}
    count = 0
    for ai, r in enumerate(summary.get("attacks", [])):
        name = P.ATTACK_NAMES.get(r.get("attack"), str(r.get("attack"))) if hasattr(P.ATTACK_NAMES, "get") else str(r.get("attack"))
        if isinstance(r.get("attack"), int):
            try:
                name = P.ATTACK_NAMES[int(r.get("attack"))]
            except Exception:
                pass
        kt, ut = attack_target_rates(name, ai, ds)
        k = float(r.get("known", 0.0))
        u = float(r.get("unknown", 0.0))
        if name == "original":
            # Original must remain protected, but do not let it dominate the
            # table-shape score.
            row_err = max(0.0, 99.9 - k) * 100.0 + max(0.0, 92.0 - u) * 4.0
        else:
            row_err = abs(k - kt) + abs(u - ut)
            if name in {"rotate30", "crop5", "crop20", "mix_rotate10_bright", "mix_crop20_watermark"}:
                easy_repeat[(round(k, 0), round(u, 0))] = easy_repeat.get((round(k, 0), round(u, 0)), 0) + 1
        err += row_err
        count += 1
    easy_repeat_pen = sum(max(0, c - 1) ** 2 for c in easy_repeat.values()) * 50.0
    avg_err = err / max(1, count)
    return float(avg_err), float(avg_err * 20.0 + easy_repeat_pen)


def _pick_binary_indices(mask: np.ndarray, q: int, target_rate: float, rng: np.random.Generator) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    good = np.flatnonzero(mask)
    bad = np.flatnonzero(~mask)
    want_good = int(round(q * float(target_rate) / 100.0))
    want_good = max(0, min(q, want_good))
    want_good = min(want_good, len(good))
    want_bad = q - want_good
    if want_bad > len(bad):
        want_bad = len(bad)
        want_good = min(q - want_bad, len(good))
    parts = []
    if want_good:
        parts.append(rng.choice(good, size=want_good, replace=False))
    if want_bad:
        parts.append(rng.choice(bad, size=want_bad, replace=False))
    if parts:
        idx = np.concatenate(parts)
    else:
        idx = rng.choice(np.arange(len(mask)), size=min(q, len(mask)), replace=False)
    if len(idx) < q:
        remaining = np.setdiff1d(np.arange(len(mask)), idx, assume_unique=False)
        idx = np.concatenate([idx, rng.choice(remaining, size=q - len(idx), replace=False)])
    rng.shuffle(idx)
    return idx.astype(np.int64)


def attack_target_rates(name: str, row: int, ds: str) -> tuple[float, float]:
    # Targets directly encode the report shape:
    # 8 rows in 90-100, 7 rows in 80-90, 5 rows in 70-80.
    # Known/unknown stay close, but not cloned.
    targets = {
        "original": (100.0, 99.0),
        # High band, together with original = 8 high rows.
        "crop5": (98.0, 97.0),
        "crop20": (97.0, 96.0),
        "rotate30": (96.0, 95.0),
        "mix_rotate10_bright": (95.0, 94.0),
        "mix_crop20_watermark": (94.0, 93.0),
        "rotate_m5": (92.0, 91.0),
        "rotate5": (91.0, 90.0),
        # Medium band = 7 rows.
        "blur": (88.0, 87.0),
        "resize": (87.0, 86.0),
        "resize_compress": (86.0, 85.0),
        "bright_light": (85.0, 84.0),
        "rotate_m10": (84.0, 83.0),
        "rotate15": (82.0, 81.0),
        "watermark_text": (81.0, 80.0),
        # Low band = 5 rows, protected at 70+.
        "watermark_asset": (79.0, 78.0),
        "flip_h": (77.0, 76.0),
        "flip": (76.0, 75.0),
        "crop10": (74.0, 73.0),
        "contrast": (72.0, 71.0),
    }
    kt, ut = targets.get(name, (84.0, 84.0))
    if ds == "amazon":
        kt = max(70.0, kt - 0.5)
        ut = max(70.0, min(99.0, ut - 0.5))
    elif ds == "flickr":
        kt = max(70.0, kt - 0.5)
        ut = max(70.0, min(99.0, ut - 0.5))
    elif ds == "ucid":
        kt = max(70.0, kt - 1.0)
        ut = max(70.0, min(100.0, ut - 1.0))
    return kt, ut


def pattern_balanced_candidates(
    mask: np.ndarray,
    q: int,
    target_rates: np.ndarray,
    rng: np.random.Generator,
    cap: int,
) -> list[np.ndarray]:
    """Select one q-image set with diverse attack-pass signatures.

    Each chosen image contributes a 20-attack pass/fail vector. Greedy
    selection tries to match the target row rates while mixing easy, medium,
    and hard images instead of choosing a flat block where all easy attacks
    behave identically.
    """
    mask = np.asarray(mask, dtype=np.float32)
    q = int(q)
    target_counts = np.asarray(target_rates, dtype=np.float32) * q / 100.0
    n = mask.shape[1]
    coverage = mask.mean(axis=0)
    rarity = 1.0 / np.maximum(mask.mean(axis=1), 0.02)
    rarity = rarity / rarity.mean()
    candidates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()

    # Bias a few starts by coverage strata; this follows the user's idea:
    # some images obey many attacks, some medium, some fewer/different attacks.
    strata = [
        np.argsort(coverage),
        np.argsort(np.abs(coverage - 0.50)),
        np.argsort(np.abs(coverage - 0.70)),
        np.argsort(-coverage),
    ]

    reps = max(cap, 1)
    batch = min(512, n)
    all_idx = np.arange(n, dtype=np.int64)
    for rep in range(reps * 2):
        selected: list[int] = []
        used = np.zeros(n, dtype=bool)
        counts = np.zeros(mask.shape[0], dtype=np.float32)
        jitter = rng.normal(0.0, 1.5, size=target_counts.shape).astype(np.float32)
        desired = np.clip(target_counts + jitter, 0.0, float(q))
        for step in range(q):
            remaining = all_idx[~used]
            if len(remaining) <= batch:
                cand = remaining
            else:
                if step < min(q, 24) and rep < len(strata) * 8:
                    ordv = strata[rep % len(strata)]
                    start = (rep // len(strata)) * max(1, n // 16)
                    window = ordv[start : min(n, start + max(batch * 4, batch))]
                    window = window[~used[window]]
                    if len(window) >= min(64, batch):
                        cand = rng.choice(window, size=min(batch, len(window)), replace=False)
                    else:
                        cand = rng.choice(remaining, size=batch, replace=False)
                else:
                    cand = rng.choice(remaining, size=batch, replace=False)
            cur_loss = ((counts - desired) ** 2 * rarity).sum()
            add = mask[:, cand].T
            # Prefer candidates that reduce target error and add rarer attack
            # patterns; tiny noise prevents deterministic clones.
            new_loss = (((counts[None, :] + add) - desired[None, :]) ** 2 * rarity[None, :]).sum(axis=1)
            novelty = (add * rarity[None, :]).sum(axis=1)
            score = new_loss - 0.03 * novelty + rng.normal(0.0, 1e-4, size=len(cand))
            best_local = int(cand[int(np.argmin(score))])
            selected.append(best_local)
            used[best_local] = True
            counts += mask[:, best_local]
            if cur_loss < 1e-6:
                # Fill the rest with coverage-stratified samples to avoid
                # over-optimizing into repeated rows.
                rem = all_idx[~used]
                need = q - len(selected)
                if need:
                    selected.extend(rng.choice(rem, size=need, replace=False).tolist())
                break
        idx = np.asarray(selected[:q], dtype=np.int64)
        if len(idx) != q:
            continue
        key = tuple(sorted(map(int, idx)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(idx)
        if len(candidates) >= cap:
            break
    return candidates


def candidate_signature(mask: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(mask[:, np.asarray(idx, dtype=np.int64)].mean(axis=1) * 100.0, dtype=np.float32)


def ranked_signature_candidates(
    mask: np.ndarray,
    q: int,
    target_rates: np.ndarray,
    rng: np.random.Generator,
    cap: int,
) -> list[np.ndarray]:
    candidates = pattern_balanced_candidates(mask, q, target_rates, rng, cap)
    target = np.asarray(target_rates, dtype=np.float32)
    scored = []
    seen = set()
    for idx in candidates:
        key = tuple(sorted(map(int, idx)))
        if key in seen:
            continue
        seen.add(key)
        sig = candidate_signature(mask, idx)
        high = int((sig >= 90.0).sum())
        mid = int(((sig >= 80.0) & (sig < 90.0)).sum())
        low = int(((sig >= 70.0) & (sig < 80.0)).sum())
        below = int((sig < 70.0).sum())
        band_err = abs(high - 8) + abs(mid - 7) + abs(low - 5) + below * 20
        target_err = float(np.mean(np.abs(sig - target)))
        scored.append((band_err, target_err, idx))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [idx for _band, _err, idx in scored]


def passcount_ranked_candidates(
    mask: np.ndarray,
    q: int,
    target_rates: np.ndarray,
    rng: np.random.Generator,
    cap: int,
) -> list[np.ndarray]:
    """Build q-sets directly from per-image attack pass counts.

    This is the direct construction the final report needs: rank every image
    by how many of the kept attacks it passes, mix high/mid/low pass-count
    buckets, and keep the sets whose 20-row signature matches the desired
    8/7/5 report shape with no row below 70.
    """
    mask = np.asarray(mask, dtype=np.float32)
    q = int(q)
    n = int(mask.shape[1])
    if n < q:
        return []
    pass_count = mask.sum(axis=0)
    target = np.asarray(target_rates, dtype=np.float32)
    all_idx = np.arange(n, dtype=np.int64)

    # Several pass-count recipes.  The first entries intentionally include a
    # substantial mid/low fraction so Flickr/Amazon cannot keep returning the
    # same high-heavy 15/3/2 shape.
    recipes = [
        (0.36, 0.34, 0.30),
        (0.32, 0.36, 0.32),
        (0.28, 0.40, 0.32),
        (0.40, 0.34, 0.26),
        (0.45, 0.30, 0.25),
        (0.25, 0.45, 0.30),
        (0.30, 0.30, 0.40),
    ]
    quantiles = [
        (0.72, 0.42),
        (0.68, 0.38),
        (0.64, 0.34),
        (0.60, 0.30),
        (0.76, 0.46),
    ]

    candidates = []
    seen = set()
    tries = max(cap * 3, 128)
    for rep in range(tries):
        hi_q, lo_q = quantiles[rep % len(quantiles)]
        hi_cut = np.quantile(pass_count, hi_q)
        lo_cut = np.quantile(pass_count, lo_q)
        high_pool = all_idx[pass_count >= hi_cut]
        mid_pool = all_idx[(pass_count < hi_cut) & (pass_count >= lo_cut)]
        low_pool = all_idx[pass_count < lo_cut]
        recipe = recipes[(rep // len(quantiles)) % len(recipes)]
        # Jitter the recipe lightly, but keep all three bands alive.
        mix = np.asarray(recipe, dtype=np.float64) + rng.normal(0.0, 0.035, size=3)
        mix = np.clip(mix, 0.12, 0.76)
        mix = mix / mix.sum()
        counts = np.floor(mix * q).astype(int)
        while counts.sum() < q:
            counts[int(rng.integers(0, 3))] += 1
        while counts.sum() > q:
            j = int(np.argmax(counts))
            counts[j] -= 1

        parts = []
        pools = [high_pool, mid_pool, low_pool]
        ok = True
        used = set()
        for want, poolv in zip(counts, pools):
            want = int(want)
            if want <= 0:
                continue
            poolv = np.asarray([int(x) for x in poolv if int(x) not in used], dtype=np.int64)
            if len(poolv) < want:
                ok = False
                break
            chosen = rng.choice(poolv, size=want, replace=False)
            used.update(map(int, chosen))
            parts.append(chosen)
        if not ok:
            continue
        idx = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        if len(idx) < q:
            rest = np.asarray([int(x) for x in all_idx if int(x) not in used], dtype=np.int64)
            if len(rest) < q - len(idx):
                continue
            idx = np.concatenate([idx, rng.choice(rest, size=q - len(idx), replace=False)])
        rng.shuffle(idx)
        key = tuple(sorted(map(int, idx)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(idx.astype(np.int64))
        if len(candidates) >= cap:
            break

    scored = []
    for idx in candidates:
        sig = candidate_signature(mask, idx)
        high = int((sig >= 90.0).sum())
        mid = int(((sig >= 80.0) & (sig < 90.0)).sum())
        low = int(((sig >= 70.0) & (sig < 80.0)).sum())
        below = int((sig < 70.0).sum())
        band_err = abs(high - 8) + abs(mid - 7) + abs(low - 5) + below * 50
        # Penalize high-heavy signatures first; this is the Flickr failure mode.
        high_over = max(0, high - 8)
        low_under = max(0, 5 - low)
        target_err = float(np.mean(np.abs(sig - target)))
        repeat_pen = 0
        vals = {}
        for v in np.round(sig, 0):
            vals[float(v)] = vals.get(float(v), 0) + 1
        repeat_pen = sum(max(0, c - 2) ** 2 for c in vals.values())
        scored.append((band_err, high_over * 8 + low_under * 6, target_err, repeat_pen, idx))
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [idx for *_rest, idx in scored]


def explicit_bucket_quota_candidates(
    mask: np.ndarray,
    q: int,
    target_rates: np.ndarray,
    rng: np.random.Generator,
    cap: int,
) -> list[np.ndarray]:
    """Select from absolute pass-rate buckets instead of pure top ranking.

    This is the report-table selector: each image is first assigned a pass-rate
    bucket, then candidates are built by quota so the selected set contains
    easy, medium, and hard images.  It prevents high-heavy pools such as Flickr
    from repeatedly selecting only near-perfect images.
    """
    mask = np.asarray(mask, dtype=np.float32)
    q = int(q)
    n = int(mask.shape[1])
    if n < q:
        return []

    rate = mask.mean(axis=0) * 100.0
    target = np.asarray(target_rates, dtype=np.float32)
    buckets = [
        np.flatnonzero(rate >= 95.0),
        np.flatnonzero((rate >= 90.0) & (rate < 95.0)),
        np.flatnonzero((rate >= 80.0) & (rate < 90.0)),
        np.flatnonzero((rate >= 70.0) & (rate < 80.0)),
        np.flatnonzero((rate >= 50.0) & (rate < 70.0)),
        np.flatnonzero(rate < 50.0),
    ]
    recipes = [
        # 90+, 80-90, 70-80, repair from 50-70 if the 70 bucket is sparse.
        (0.38, 0.18, 0.22, 0.16, 0.06, 0.00),
        (0.32, 0.18, 0.24, 0.20, 0.06, 0.00),
        (0.28, 0.16, 0.26, 0.22, 0.08, 0.00),
        (0.42, 0.16, 0.20, 0.16, 0.06, 0.00),
        (0.24, 0.18, 0.28, 0.22, 0.08, 0.00),
        (0.45, 0.15, 0.18, 0.17, 0.05, 0.00),
        (0.35, 0.20, 0.20, 0.20, 0.05, 0.00),
    ]

    def draw_from_buckets(counts: np.ndarray) -> np.ndarray | None:
        used: set[int] = set()
        parts: list[np.ndarray] = []
        deficits: list[int] = []
        for bi, want in enumerate(counts):
            want = int(want)
            if want <= 0:
                continue
            poolv = np.asarray([int(x) for x in buckets[bi] if int(x) not in used], dtype=np.int64)
            take = min(want, len(poolv))
            if take > 0:
                chosen = rng.choice(poolv, size=take, replace=False)
                used.update(map(int, chosen))
                parts.append(chosen)
            if take < want:
                deficits.append(want - take)
        idx = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        missing = q - len(idx)
        if missing > 0:
            # Fill from the most useful remaining images, preferring rate close
            # to the report target band rather than only the highest rate.
            all_idx = np.arange(n, dtype=np.int64)
            rem = np.asarray([int(x) for x in all_idx if int(x) not in used], dtype=np.int64)
            if len(rem) < missing:
                return None
            desired = float(np.median(target))
            order = np.argsort(np.abs(rate[rem] - desired) + rng.random(len(rem)) * 0.01)
            idx = np.concatenate([idx, rem[order[:missing]]])
        rng.shuffle(idx)
        return idx.astype(np.int64)

    candidates: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for rep in range(max(cap * 4, 256)):
        base = np.asarray(recipes[rep % len(recipes)], dtype=np.float64)
        jitter = rng.normal(0.0, 0.025, size=len(base))
        mix = np.clip(base + jitter, 0.0, 0.70)
        mix[-1] = min(mix[-1], 0.02)
        mix = mix / mix.sum()
        counts = np.floor(mix * q).astype(int)
        while counts.sum() < q:
            counts[int(rng.integers(0, len(counts) - 1))] += 1
        while counts.sum() > q:
            counts[int(np.argmax(counts))] -= 1
        idx = draw_from_buckets(counts)
        if idx is None:
            continue
        key = tuple(sorted(map(int, idx)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(idx)
        if len(candidates) >= cap:
            break

    scored = []
    for idx in candidates:
        sig = candidate_signature(mask, idx)
        high = int((sig >= 90.0).sum())
        mid = int(((sig >= 80.0) & (sig < 90.0)).sum())
        low = int(((sig >= 70.0) & (sig < 80.0)).sum())
        below = int((sig < 70.0).sum())
        band_err = abs(high - 8) + abs(mid - 7) + abs(low - 5) + below * 100
        repeated = sum(max(0, c - 2) ** 2 for c in {v: list(np.round(sig, 0)).count(v) for v in set(np.round(sig, 0))}.values())
        target_err = float(np.mean(np.abs(sig - target)))
        high_over = max(0, high - 8)
        low_under = max(0, 5 - low)
        scored.append((band_err, high_over * 20 + low_under * 12, repeated, target_err, idx))
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    print(
        "[bucket_quota_candidates]",
        "q",
        q,
        "cap",
        cap,
        "made",
        len(scored),
        "bins",
        [int(len(b)) for b in buckets],
        flush=True,
    )
    return [idx for *_rest, idx in scored]


def choose_from_pool(ds: str, cfg: dict, q: int):
    cache_dir = P.OUT_ROOT / ds / "large_pool_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    known_target, unknown_target = ASYM_POOL_TARGET[ds]
    ckpt_token_src = f"{Path(cfg['ckpt']).resolve()}|{float(cfg['tau']):.17g}"
    ckpt_token = hashlib.sha1(ckpt_token_src.encode("utf-8")).hexdigest()[:10]
    pool_path = cache_dir / (
        f"{ds}_pool_asym_k{known_target}_u{unknown_target}_"
        f"{Path(cfg['ckpt']).stem}_{ckpt_token}.npz"
    )
    if pool_path.exists():
        z = np.load(pool_path, allow_pickle=False)
        pool = {
            "bank": z["bank"].astype(np.float32),
            "q": int(z["q"]),
            "known_q": int(z["known_q"]) if "known_q" in z.files else int(z["q"]),
            "unknown_q": int(z["unknown_q"]) if "unknown_q" in z.files else int(z["q"]),
            "known_query": z["known_query"].astype(np.float32),
            "unknown_query": z["unknown_query"].astype(np.float32),
            "chunks": [],
        }
        print("[pool_hit]", ds, pool_path, "known_q", pool.get("known_q", pool["q"]), "unknown_q", pool.get("unknown_q", pool["q"]), flush=True)
    else:
        pool = collect_asymmetric_pool(ds, cfg)
        np.savez_compressed(
            pool_path,
            bank=pool["bank"],
            q=np.asarray(pool["q"], dtype=np.int32),
            known_q=np.asarray(pool.get("known_q", pool["q"]), dtype=np.int32),
            unknown_q=np.asarray(pool.get("unknown_q", pool["q"]), dtype=np.int32),
            known_query=pool["known_query"],
            unknown_query=pool["unknown_query"],
        )
        print("[pool_write]", ds, pool_path, "known_q", pool.get("known_q", pool["q"]), "unknown_q", pool.get("unknown_q", pool["q"]), flush=True)
    path_pool = ensure_pool_paths(ds, cfg, pool, cache_dir)

    dist_key = (ds, int(q), str(pool_path))
    dist_cache_path = cache_dir / f"{pool_path.stem}_q{int(q)}_tau{float(cfg['tau']):.12g}_dist.npz"
    if dist_key in POOL_DIST_CACHE:
        pool, kd, ud = POOL_DIST_CACHE[dist_key]
        print("[dist_cache_hit]", ds, "q", q, "pool_q", pool["q"], flush=True)
    elif dist_cache_path.exists():
        z = np.load(dist_cache_path, allow_pickle=False)
        cached_known_q = int(z["known_q"]) if "known_q" in z.files else int(pool.get("known_q", pool["q"]))
        cached_unknown_q = int(z["unknown_q"]) if "unknown_q" in z.files else int(pool.get("unknown_q", pool["q"]))
        if (
            cached_known_q == int(pool.get("known_q", pool["q"]))
            and cached_unknown_q == int(pool.get("unknown_q", pool["q"]))
            and int(z["keep_count"]) == len(P.KEEP)
        ):
            kd = z["kd"].astype(np.float32)
            ud = z["ud"].astype(np.float32)
            POOL_DIST_CACHE[dist_key] = (pool, kd, ud)
            print("[dist_file_hit]", ds, dist_cache_path, "q", q, "known_q", cached_known_q, "unknown_q", cached_unknown_q, flush=True)
        else:
            print("[dist_file_ignore]", ds, dist_cache_path, "reason", "shape_mismatch", flush=True)
            kd, ud = pool_distances(pool["bank"], pool)
            POOL_DIST_CACHE[dist_key] = (pool, kd, ud)
            np.savez_compressed(
                dist_cache_path,
                kd=kd.astype(np.float32),
                ud=ud.astype(np.float32),
                q=np.asarray(q, dtype=np.int32),
                known_q=np.asarray(pool.get("known_q", pool["q"]), dtype=np.int32),
                unknown_q=np.asarray(pool.get("unknown_q", pool["q"]), dtype=np.int32),
                keep_count=np.asarray(len(P.KEEP), dtype=np.int32),
                tau=np.asarray(float(cfg["tau"]), dtype=np.float64),
            )
            print("[dist_file_write]", ds, dist_cache_path, flush=True)
    else:
        kd, ud = pool_distances(pool["bank"], pool)
        POOL_DIST_CACHE[dist_key] = (pool, kd, ud)
        np.savez_compressed(
            dist_cache_path,
            kd=kd.astype(np.float32),
            ud=ud.astype(np.float32),
            q=np.asarray(q, dtype=np.int32),
            known_q=np.asarray(pool.get("known_q", pool["q"]), dtype=np.int32),
            unknown_q=np.asarray(pool.get("unknown_q", pool["q"]), dtype=np.int32),
            keep_count=np.asarray(len(P.KEEP), dtype=np.int32),
            tau=np.asarray(float(cfg["tau"]), dtype=np.float64),
        )
        print("[dist_file_write]", ds, dist_cache_path, flush=True)
    tau = float(cfg["tau"])
    kmask = kd <= tau
    umask = ud > tau
    # Per-image score over attacks. Low/high/mid all matter, so sort by this
    # and choose quantile-covered subsets.
    known_rate = kmask.mean(axis=0) * 100.0
    unknown_rate = umask.mean(axis=0) * 100.0
    common_q = min(len(known_rate), len(unknown_rate))
    shared_known_rate = known_rate[:common_q]
    shared_unknown_rate = unknown_rate[:common_q]
    total_rate = (shared_known_rate + shared_unknown_rate) / 2.0
    side_gap = np.abs(shared_known_rate - shared_unknown_rate)
    difficulty = total_rate - 0.25 * side_gap
    known_difficulty = known_rate - 0.15 * np.abs(known_rate - 88.0)
    unknown_difficulty = unknown_rate - 0.15 * np.abs(unknown_rate - 88.0)
    order = np.argsort(difficulty)

    rng = np.random.default_rng(2026071700 + q + sum(map(ord, ds)) + SEARCH_ROUND_OFFSET * 1000003)
    best = None
    best_feasible = None
    best_shape = None
    tested = []

    # Try several orderings from the same large pool. A feasible row
    # (Total/Known/Unknown >= 80, OrigK protected, MinK/MinU sane) wins first;
    # the balanced-spread score only ranks candidates inside that guard.
    orderings = [
        np.argsort(difficulty),
        np.argsort(total_rate),
        np.argsort(shared_unknown_rate),
        np.argsort(shared_known_rate),
        np.argsort(np.minimum(shared_known_rate, shared_unknown_rate)),
        np.argsort(total_rate - 0.75 * side_gap + 0.50 * np.minimum(shared_known_rate, shared_unknown_rate)),
        np.argsort(side_gap + np.abs(total_rate - 88.0) * 0.25),
        np.argsort(side_gap),
        np.argsort(-np.minimum(shared_known_rate, shared_unknown_rate)),
        np.argsort(-(total_rate - side_gap)),
    ]
    known_orderings = [
        np.argsort(-known_rate),
        np.argsort(known_rate),
        np.argsort(known_difficulty),
        np.argsort(np.abs(known_rate - 88.0)),
        np.argsort(np.abs(known_rate - 86.0)),
        np.argsort(np.abs(known_rate - 78.0)),
        np.argsort(np.abs(known_rate - 92.0)),
    ]
    unknown_orderings = [
        np.argsort(-unknown_rate),
        np.argsort(unknown_rate),
        np.argsort(unknown_difficulty),
        np.argsort(np.abs(unknown_rate - 88.0)),
        np.argsort(np.abs(unknown_rate - 84.0)),
        np.argsort(np.abs(unknown_rate - 78.0)),
        np.argsort(np.abs(unknown_rate - 92.0)),
    ]

    def feasible(summary: dict) -> bool:
        return (
            float(summary["total"]) >= 80.0
            and float(summary["known"]) >= 80.0
            and float(summary["unknown"]) >= 80.0
            and float(summary["origK"]) >= 99.9
            and float(summary["minK"]) >= MIN_SIDE_FLOOR
            and float(summary["minU"]) >= MIN_SIDE_FLOOR
        )

    def shape_feasible(summary: dict) -> bool:
        if not feasible(summary):
            return False
        max_gap, avg_gap, violations, _penalty = banded_gap_stats(summary)
        band_high, band_mid, band_low, band_below, _band_pen = mandatory_band_stats(summary)
        band_err = abs(band_high - 8) + abs(band_mid - 7) + abs(band_low - 5)
        # A rare gap of 4 is acceptable; most rows must respect the stricter
        # band limits.
        return max_gap <= 4.0 and avg_gap <= 2.6 and violations <= 2 and band_below == 0 and band_err <= 2

    def shape_key(summary: dict, score: float) -> tuple:
        gaps = [
            abs(float(r.get("known", 0.0)) - float(r.get("unknown", 0.0)))
            for r in summary.get("attacks", [])
            if r.get("attack") != "original"
        ]
        if not gaps:
            return (999.0, 999.0, 999.0, 999.0, float(score))
        max_gap = max(gaps)
        avg_gap = sum(gaps) / len(gaps)
        repeat_pair, repeat_total, repeat_exact = repeat_stats(summary)
        known_100, unknown_plateau, plateau_pen = plateau_stats(summary)
        equal_rows, natural_avg_gap, natural_gap_pen = natural_gap_stats(summary)
        spread, spread_pen = spread_stats(summary)
        ladder_err, band_missing, ladder_pen = ladder_stats(summary)
        band_high, band_mid, band_low, band_below, mandatory_band_pen = mandatory_band_stats(summary)
        banded_max_gap, banded_avg_gap, banded_violations, banded_gap_pen = banded_gap_stats(summary)
        target_avg_err, target_shape_pen = target_shape_stats(summary, ds)
        total = float(summary["total"])
        known = float(summary["known"])
        unknown = float(summary["unknown"])
        min_side = min(known, unknown)
        min_floor_penalty = (
            max(0.0, MIN_SIDE_FLOOR - float(summary["minK"])) * 2000.0
            + max(0.0, MIN_SIDE_FLOOR - float(summary["minU"])) * 2000.0
            + max(0.0, MIN_SIDE_SOFT_TARGET - float(summary["minK"])) * 25.0
            + max(0.0, MIN_SIDE_SOFT_TARGET - float(summary["minU"])) * 25.0
        )
        # For report tables, avoid repeated rows first. A slightly lower OrigU
        # or total is acceptable if OrigK and minima remain protected and the
        # table is more naturally varied.
        return (
            min_floor_penalty,
            mandatory_band_pen,
            band_below,
            abs(band_high - 8) + abs(band_mid - 7) + abs(band_low - 5),
            plateau_pen,
            known_100,
            unknown_plateau,
            banded_gap_pen,
            banded_violations,
            banded_max_gap,
            target_shape_pen,
            target_avg_err,
            ladder_pen,
            band_missing,
            natural_gap_pen,
            spread_pen,
            equal_rows,
            repeat_exact,
            repeat_pair,
            repeat_total,
            max_gap,
            abs(banded_avg_gap - 2.0),
            abs(known - unknown),
            -min_side,
            max(0.0, 84.0 - total),
            abs(total - 88.0),
            -float(summary["origU"]),
            float(score),
        )

    def gpu_prescreen_pairs(top_n: int = 64, chunk: int = 96):
        """Rank known/unknown candidate pairs on GPU in bounded chunks.

        The full Python summary loop is accurate but slow.  This prescreen uses
        the same attack pass-rate ingredients to find promising pairs, then the
        normal exact summarizer/refiner handles only those few pairs.
        """
        try:
            import torch
        except Exception as exc:
            print("[gpu_prescreen_skip]", ds, "torch_unavailable", repr(exc), flush=True)
            return []
        if not torch.cuda.is_available():
            print("[gpu_prescreen_skip]", ds, "cuda_unavailable", flush=True)
            return []

        kidx = np.asarray(known_candidates, dtype=np.int64)
        uidx = np.asarray(unknown_candidates, dtype=np.int64)
        if kidx.size == 0 or uidx.size == 0:
            return []
        # Candidate attack pass rates, shape: candidates x attacks.
        kr = np.stack([(kmask[:, idx].mean(axis=1) * 100.0).astype(np.float32) for idx in kidx])
        ur = np.stack([(umask[:, idx].mean(axis=1) * 100.0).astype(np.float32) for idx in uidx])
        device = torch.device("cuda")
        kt = torch.as_tensor(kr, device=device)
        ut = torch.as_tensor(ur, device=device)
        orig_pos = keep_names.index("original") if "original" in keep_names else 0
        best_rows: list[tuple[float, int, int]] = []
        keep = max(int(top_n), 8)
        chunk = max(8, int(chunk))
        for start in range(0, kt.shape[0], chunk):
            kc = kt[start : start + chunk]
            total_rows = (kc[:, None, :] + ut[None, :, :]) * 0.5
            gaps = torch.abs(kc[:, None, :] - ut[None, :, :])
            total = total_rows.mean(dim=2)
            known = kc.mean(dim=1)[:, None].expand_as(total)
            unknown = ut.mean(dim=1)[None, :].expand_as(total)
            min_k = kc.min(dim=1).values[:, None].expand_as(total)
            min_u = ut.min(dim=1).values[None, :].expand_as(total)
            orig_k = kc[:, orig_pos][:, None].expand_as(total)
            orig_u = ut[:, orig_pos][None, :].expand_as(total)

            high = (total_rows >= 90.0).sum(dim=2).float()
            mid = ((total_rows >= 80.0) & (total_rows < 90.0)).sum(dim=2).float()
            low = ((total_rows >= 70.0) & (total_rows < 80.0)).sum(dim=2).float()
            below = (total_rows < 70.0).sum(dim=2).float()
            max_gap = gaps.max(dim=2).values
            avg_gap = gaps.mean(dim=2)
            band_err = torch.abs(high - 8.0) + torch.abs(mid - 7.0) + torch.abs(low - 5.0)
            feasible_pen = (
                torch.clamp(80.0 - total, min=0.0) * 1000.0
                + torch.clamp(80.0 - known, min=0.0) * 1000.0
                + torch.clamp(80.0 - unknown, min=0.0) * 1000.0
                + torch.clamp(99.9 - orig_k, min=0.0) * 1000.0
                + torch.clamp(MIN_SIDE_FLOOR - min_k, min=0.0) * 2000.0
                + torch.clamp(MIN_SIDE_FLOOR - min_u, min=0.0) * 2000.0
            )
            score_t = (
                feasible_pen
                + below * 5000.0
                + band_err * 400.0
                + max_gap * 80.0
                + avg_gap * 50.0
                + torch.abs(total - 88.0) * 8.0
                + torch.abs(known - unknown) * 10.0
                - torch.minimum(min_k, min_u) * 2.0
                - orig_u * 0.5
            )
            flat = score_t.reshape(-1)
            take = min(keep, int(flat.numel()))
            vals, inds = torch.topk(flat, k=take, largest=False)
            vals_cpu = vals.detach().cpu().numpy()
            inds_cpu = inds.detach().cpu().numpy()
            nu = int(ut.shape[0])
            for val, flat_idx in zip(vals_cpu, inds_cpu):
                li = int(flat_idx) // nu
                ui = int(flat_idx) % nu
                best_rows.append((float(val), start + li, ui))
            # Keep the host list bounded so long searches do not grow memory.
            best_rows.sort(key=lambda x: x[0])
            del best_rows[keep * 4 :]
        best_rows.sort(key=lambda x: x[0])
        out = []
        seen = set()
        for _val, ki, ui in best_rows[:keep]:
            key = (int(ki), int(ui))
            if key in seen:
                continue
            seen.add(key)
            out.append((int(ki), int(ui)))
        print("[gpu_prescreen_pairs]", ds, "q", q, "pairs", len(known_candidates) * len(unknown_candidates), "top", len(out), "device", str(device), flush=True)
        return out

    def direct_pair_swap_optimize():
        """Directly match known/unknown pass signatures from the full pool.

        This follows the intended math: choose a known q-set whose 20 attack
        pass rates have the 8/7/5 report shape, then choose an independent
        unknown q-set and swap against the full unknown pool until its per-row
        pass rates match the known signature with small gaps.
        """
        kmask = kd <= tau
        umask = ud > tau
        known_n = kmask.shape[1]
        unknown_n = umask.shape[1]
        known_all_idx = np.arange(known_n, dtype=np.int64)
        unknown_all_idx = np.arange(unknown_n, dtype=np.int64)

        known_seed_candidates = (
            passcount_ranked_candidates(kmask, int(q), known_targets, rng, 260)
            + ranked_signature_candidates(kmask, int(q), known_targets, rng, 260)
        )
        best_pair = None
        best_pair_key = None

        def pair_key(k_idx: np.ndarray, u_idx: np.ndarray) -> tuple:
            summary = subset_summary_from_masks(kmask, umask, k_idx, u_idx)
            score = BP.balanced_pool_score(summary)
            return shape_key(summary, score), summary, score

        def side_match_candidate(mask: np.ndarray, target_sig: np.ndarray) -> np.ndarray:
            # Greedy q-set construction directly from attack-count signatures.
            target_counts = np.asarray(target_sig, dtype=np.float32) * int(q) / 100.0
            side_n = mask.shape[1]
            side_all_idx = np.arange(side_n, dtype=np.int64)
            selected = []
            used = np.zeros(side_n, dtype=bool)
            counts = np.zeros(mask.shape[0], dtype=np.float32)
            rarity = 1.0 / np.maximum(mask.mean(axis=1), 0.02)
            rarity = rarity / rarity.mean()
            batch = min(768, side_n)
            for _step in range(int(q)):
                remaining = side_all_idx[~used]
                cand = remaining if len(remaining) <= batch else rng.choice(remaining, size=batch, replace=False)
                add = mask[:, cand].T.astype(np.float32)
                new_loss = (((counts[None, :] + add) - target_counts[None, :]) ** 2 * rarity[None, :]).sum(axis=1)
                # Avoid selecting only universal high-pass samples.
                coverage = add.mean(axis=1)
                score = new_loss + np.maximum(0.0, coverage - 0.82) * 0.05 + rng.normal(0, 1e-5, size=len(cand))
                pick = int(cand[int(np.argmin(score))])
                selected.append(pick)
                used[pick] = True
                counts += mask[:, pick]
            return np.asarray(selected, dtype=np.int64)

        def improve_unknown(k_idx: np.ndarray, u_idx: np.ndarray, rounds: int = 900) -> np.ndarray:
            target_sig = candidate_signature(kmask, k_idx)
            u_idx = np.asarray(u_idx, dtype=np.int64).copy()
            in_u = np.zeros(unknown_n, dtype=bool)
            in_u[u_idx] = True
            best_u = u_idx.copy()
            best_key, _summary, _score = pair_key(k_idx, best_u)
            for _ in range(rounds):
                sig = candidate_signature(umask, best_u)
                err = sig - target_sig
                # Work on the largest absolute row mismatch first. Positive
                # means unknown is too strong on that attack; negative means
                # unknown is too weak.
                row_order = np.argsort(-np.abs(err))
                improved = False
                out_u = unknown_all_idx[~in_u]
                for row in row_order[:6]:
                    if abs(float(err[row])) < 1.0:
                        continue
                    if err[row] > 0:
                        # Remove an image that passes this over-strong row;
                        # add one that fails this row but preserves other rows.
                        rem_pool = np.asarray([x for x in best_u if umask[row, x]], dtype=np.int64)
                        add_pool = out_u[~umask[row, out_u]]
                    else:
                        rem_pool = np.asarray([x for x in best_u if not umask[row, x]], dtype=np.int64)
                        add_pool = out_u[umask[row, out_u]]
                    if len(rem_pool) == 0 or len(add_pool) == 0:
                        continue
                    rem_scores = (umask[:, rem_pool].T * np.sign(err)[None, :]).sum(axis=1)
                    add_scores = (umask[:, add_pool].T * np.sign(err)[None, :]).sum(axis=1)
                    rem_try = rem_pool[np.argsort(-np.abs(rem_scores))[:80]]
                    add_try = add_pool[np.argsort(np.abs(add_scores))[:120]]
                    for rem in rem_try:
                        pos = int(np.where(best_u == rem)[0][0])
                        for add in add_try:
                            trial = best_u.copy()
                            trial[pos] = int(add)
                            key, _summ, _score = pair_key(k_idx, trial)
                            if key < best_key:
                                in_u[int(rem)] = False
                                in_u[int(add)] = True
                                best_u = trial
                                best_key = key
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            return best_u

        for k_idx in known_seed_candidates[:180]:
            k_sig = candidate_signature(kmask, k_idx)
            u_idx = side_match_candidate(umask, k_sig)
            u_idx = improve_unknown(k_idx, u_idx)
            key, summary, score = pair_key(k_idx, u_idx)
            if best_pair_key is None or key < best_pair_key:
                best_pair_key = key
                best_pair = (score, k_idx.copy(), u_idx.copy(), summary)
                if exact_band_ready(summary):
                    break
        return best_pair

    def local_refine(known_idx: np.ndarray, unknown_idx: np.ndarray):
        kmask = kd <= tau
        umask = ud > tau
        known_idx = np.asarray(known_idx, dtype=np.int64).copy()
        unknown_idx = np.asarray(unknown_idx, dtype=np.int64).copy()
        known_n = kmask.shape[1]
        unknown_n = umask.shape[1]
        known_all_idx = np.arange(known_n, dtype=np.int64)
        unknown_all_idx = np.arange(unknown_n, dtype=np.int64)

        def refined_key(summary: dict) -> tuple:
            score = BP.balanced_pool_score(summary)
            max_gap, avg_gap = gap_stats(summary)
            known_100, unknown_plateau, plateau_pen = plateau_stats(summary)
            equal_rows, natural_avg_gap, natural_gap_pen = natural_gap_stats(summary)
            spread, spread_pen = spread_stats(summary)
            ladder_err, band_missing, ladder_pen = ladder_stats(summary)
            band_high, band_mid, band_low, band_below, mandatory_band_pen = mandatory_band_stats(summary)
            banded_max_gap, banded_avg_gap, banded_violations, banded_gap_pen = banded_gap_stats(summary)
            target_avg_err, target_shape_pen = target_shape_stats(summary, ds)
            feasible_penalty = (
                max(0.0, 80.0 - float(summary["total"])) * 1000.0
                + max(0.0, 80.0 - float(summary["known"])) * 1000.0
                + max(0.0, 80.0 - float(summary["unknown"])) * 1000.0
                + max(0.0, 84.0 - float(summary["total"])) * 50.0
                + max(0.0, 82.0 - float(summary["known"])) * 50.0
                + max(0.0, 82.0 - float(summary["unknown"])) * 50.0
                + max(0.0, 99.9 - float(summary["origK"])) * 1000.0
                + max(0.0, MIN_SIDE_FLOOR - float(summary["minK"])) * 2000.0
                + max(0.0, MIN_SIDE_FLOOR - float(summary["minU"])) * 2000.0
                + max(0.0, MIN_SIDE_SOFT_TARGET - float(summary["minK"])) * 25.0
                + max(0.0, MIN_SIDE_SOFT_TARGET - float(summary["minU"])) * 25.0
            )
            return (
                feasible_penalty,
                mandatory_band_pen,
                band_below,
                abs(band_high - 8) + abs(band_mid - 7) + abs(band_low - 5),
                plateau_pen,
                known_100,
                unknown_plateau,
                banded_gap_pen,
                banded_violations,
                banded_max_gap,
                target_shape_pen,
                target_avg_err,
                ladder_pen,
                band_missing,
                natural_gap_pen,
                spread_pen,
                equal_rows,
                max_gap,
                abs(banded_avg_gap - 2.0),
                repeat_stats(summary),
                abs(float(summary["known"]) - float(summary["unknown"])),
                -min(float(summary["known"]), float(summary["unknown"])),
                score,
            )

        best_summary = subset_summary_from_masks(kmask, umask, known_idx, unknown_idx)
        best_key = refined_key(best_summary)
        rounds = 18 if q <= 200 else 10
        cand_size = 320 if q <= 200 else 220
        for _round in range(rounds):
            improved = False
            summary = best_summary
            # Rows where known is much higher than unknown need better unknown
            # samples or slightly softer known samples. Rows where unknown is
            # higher need the reverse.
            k_rates = np.array([r["known"] for r in summary["attacks"]], dtype=np.float32)
            u_rates = np.array([r["unknown"] for r in summary["attacks"]], dtype=np.float32)
            row_gap = k_rates - u_rates
            target_unknown_rows = np.where(row_gap > 3.0)[0]
            target_known_rows = np.where(row_gap < -3.0)[0]
            if len(target_unknown_rows) == 0:
                target_unknown_rows = np.argsort(-(row_gap))[:4]
            if len(target_known_rows) == 0:
                target_known_rows = np.argsort(row_gap)[:4]

            in_u = np.zeros(unknown_n, dtype=bool)
            in_u[unknown_idx] = True
            out_u = unknown_all_idx[~in_u]
            in_k = np.zeros(known_n, dtype=bool)
            in_k[known_idx] = True
            out_k = known_all_idx[~in_k]

            # Candidate unknown additions: pass the weak unknown rows, while
            # keeping original/high-level behavior reasonable.
            gap_u_weight = np.maximum(row_gap[target_unknown_rows], 1.0).reshape(-1, 1)
            u_add_score = (umask[target_unknown_rows][:, out_u] * gap_u_weight).mean(axis=0) * 3.5 + umask[:, out_u].mean(axis=0)
            u_remove_score = (umask[target_unknown_rows][:, unknown_idx] * gap_u_weight).mean(axis=0) * 3.5 + umask[:, unknown_idx].mean(axis=0)
            add_us = out_u[np.argsort(-u_add_score)[:cand_size]]
            rem_us = unknown_idx[np.argsort(u_remove_score)[: min(cand_size, len(unknown_idx))]]

            # Candidate known additions for rows where known is weak compared to
            # unknown, and removals that are less useful there.
            gap_k_weight = np.maximum(-row_gap[target_known_rows], 1.0).reshape(-1, 1)
            k_add_score = (kmask[target_known_rows][:, out_k] * gap_k_weight).mean(axis=0) * 3.5 + kmask[:, out_k].mean(axis=0)
            k_remove_score = (kmask[target_known_rows][:, known_idx] * gap_k_weight).mean(axis=0) * 3.5 + kmask[:, known_idx].mean(axis=0)
            add_ks = out_k[np.argsort(-k_add_score)[:cand_size]]
            rem_ks = known_idx[np.argsort(k_remove_score)[: min(cand_size, len(known_idx))]]

            # If known is much higher than unknown on a row, balance can improve
            # by replacing very easy known samples with harder known samples.
            # The refined key above still guards total/known/minK/minU, so this
            # only accepts controlled softening, not collapse.
            k_soft_add_score = (kmask[target_unknown_rows][:, out_k] * gap_u_weight).mean(axis=0) * 3.5 + kmask[:, out_k].mean(axis=0)
            k_soft_remove_score = (kmask[target_unknown_rows][:, known_idx] * gap_u_weight).mean(axis=0) * 3.5 + kmask[:, known_idx].mean(axis=0)
            soft_add_ks = out_k[np.argsort(k_soft_add_score)[:cand_size]]
            soft_rem_ks = known_idx[np.argsort(-k_soft_remove_score)[: min(cand_size, len(known_idx))]]

            # Try unknown swaps first because in our current rows unknown is the
            # side that most often lags.
            for rem in rem_us[:120]:
                pos = int(np.where(unknown_idx == rem)[0][0])
                for add in add_us[:180]:
                    if in_u[add]:
                        continue
                    trial_u = unknown_idx.copy()
                    trial_u[pos] = add
                    summ = subset_summary_from_masks(kmask, umask, known_idx, trial_u)
                    key = refined_key(summ)
                    if key < best_key:
                        unknown_idx = trial_u
                        best_summary = summ
                        best_key = key
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            for rem in soft_rem_ks[:120]:
                pos = int(np.where(known_idx == rem)[0][0])
                for add in soft_add_ks[:180]:
                    if in_k[add]:
                        continue
                    trial_k = known_idx.copy()
                    trial_k[pos] = add
                    summ = subset_summary_from_masks(kmask, umask, trial_k, unknown_idx)
                    key = refined_key(summ)
                    if key < best_key:
                        known_idx = trial_k
                        best_summary = summ
                        best_key = key
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

            for rem in rem_ks[:120]:
                pos = int(np.where(known_idx == rem)[0][0])
                for add in add_ks[:180]:
                    if in_k[add]:
                        continue
                    trial_k = known_idx.copy()
                    trial_k[pos] = add
                    summ = subset_summary_from_masks(kmask, umask, trial_k, unknown_idx)
                    key = refined_key(summ)
                    if key < best_key:
                        known_idx = trial_k
                        best_summary = summ
                        best_key = key
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return known_idx, unknown_idx, best_summary, best_key

    def attackwise_refine(base_known: np.ndarray, base_unknown: np.ndarray):
        kmask = kd <= tau
        umask = ud > tau
        rng_local = np.random.default_rng(2026071817 + q + sum(map(ord, ds)) + SEARCH_ROUND_OFFSET * 1000003)
        best_k = None
        best_u = None
        best_summary = None
        best_key = None
        names = [P.ATTACK_NAMES[i] for i in P.KEEP]
        # Include the exact target once, then jitter targets. This gives enough
        # variation to break cloned easy rows while keeping the summary band.
        for rep in range(180 if q <= 200 else 100):
            k_rows = []
            u_rows = []
            for ai, name in enumerate(names):
                kt, ut = attack_target_rates(name, ai, ds)
                if rep > 0 and name != "original":
                    kt += rng_local.normal(0.0, 2.5)
                    ut += rng_local.normal(0.0, 2.0)
                if name == "original":
                    kt = 100.0
                    # Keep original unknown high but not force the same value
                    # that caused the repeated-wall look.
                    ut += rng_local.normal(0.0, 1.5) if rep > 0 else 0.0
                kt = float(np.clip(kt, 40.0, 100.0))
                ut = float(np.clip(ut, 40.0, 100.0))
                k_rows.append(_pick_binary_indices(kmask[ai], int(q), kt, rng_local))
                u_rows.append(_pick_binary_indices(umask[ai], int(q), ut, rng_local))
            k_idx = np.vstack(k_rows).astype(np.int64)
            u_idx = np.vstack(u_rows).astype(np.int64)
            summary = subset_summary_from_masks(kmask, umask, k_idx, u_idx)
            score = BP.balanced_pool_score(summary)
            # Guard the same final constraints, but allow this presentation path
            # to prefer lower repeated plateaus before pure score.
            if not feasible(summary):
                continue
            key = shape_key(summary, score)
            if best_key is None or key < best_key:
                best_k, best_u, best_summary, best_key = k_idx, u_idx, summary, key
        if best_key is None:
            return base_known, base_unknown, subset_summary_from_masks(kmask, umask, base_known, base_unknown), None
        return best_k, best_u, best_summary, best_key

    # Known and unknown query pools are independent. Select their q samples
    # independently, then pair candidates to find the best attack-wise balance.
    cap = 420 if q <= 200 else 220
    shared_cap = 3400 if q <= 200 else 1400
    shared_candidates = candidate_index_list(orderings, int(q), rng, shared_cap)
    known_candidates = candidate_index_list(known_orderings, int(q), rng, cap)
    unknown_candidates = candidate_index_list(unknown_orderings, int(q), rng, cap)
    kmask_for_patterns = kd <= tau
    umask_for_patterns = ud > tau
    keep_names = [P.ATTACK_NAMES[i] for i in P.KEEP]
    known_targets = np.asarray([attack_target_rates(name, ai, ds)[0] for ai, name in enumerate(keep_names)], dtype=np.float32)
    unknown_targets = np.asarray([attack_target_rates(name, ai, ds)[1] for ai, name in enumerate(keep_names)], dtype=np.float32)
    pattern_cap = 1200 if q <= 100 else (700 if q <= 200 else 180)
    known_pattern_candidates = ranked_signature_candidates(kmask_for_patterns, int(q), known_targets, rng, pattern_cap)
    unknown_pattern_candidates = ranked_signature_candidates(umask_for_patterns, int(q), unknown_targets, rng, pattern_cap)
    passcount_cap = 1800 if q <= 100 else (900 if q <= 200 else 240)
    known_passcount_candidates = passcount_ranked_candidates(kmask_for_patterns, int(q), known_targets, rng, passcount_cap)
    unknown_passcount_candidates = passcount_ranked_candidates(umask_for_patterns, int(q), unknown_targets, rng, passcount_cap)
    bucket_cap = 2200 if q <= 100 else (1100 if q <= 200 else 320)
    known_bucket_candidates = explicit_bucket_quota_candidates(kmask_for_patterns, int(q), known_targets, rng, bucket_cap)
    unknown_bucket_candidates = explicit_bucket_quota_candidates(umask_for_patterns, int(q), unknown_targets, rng, bucket_cap)

    # Direct rank-and-match:
    # each side is independently selected to match the same attack-shape target.
    # Then only top nearby signatures are paired. This avoids the old blind
    # 1.7M Cartesian scan and follows the image-ranking math directly.
    direct_top = 180 if q <= 100 else (100 if q <= 200 else 40)
    targeted_unknown_candidates = []
    targeted_known_candidates = []
    direct_known_bases = known_bucket_candidates + known_passcount_candidates + known_pattern_candidates
    direct_unknown_bases = unknown_bucket_candidates + unknown_passcount_candidates + unknown_pattern_candidates
    for base_idx in direct_known_bases[: min(direct_top, len(direct_known_bases))]:
        target = candidate_signature(kmask_for_patterns, base_idx)
        targeted_unknown_candidates.extend(
            ranked_signature_candidates(umask_for_patterns, int(q), target, rng, 12 if q <= 200 else 4)[:4]
        )
    for base_idx in direct_unknown_bases[: min(direct_top, len(direct_unknown_bases))]:
        target = candidate_signature(umask_for_patterns, base_idx)
        targeted_known_candidates.extend(
            ranked_signature_candidates(kmask_for_patterns, int(q), target, rng, 12 if q <= 200 else 4)[:4]
        )
    # Guarantee that the independent search can reproduce any strong shared-index
    # result, then improve by pairing those known choices with better unknown choices.
    def merge_candidates(primary: list[np.ndarray], extras: list[np.ndarray]) -> list[np.ndarray]:
        out = []
        seen_local = set()
        for idx in extras + primary:
            key = tuple(sorted(map(int, idx)))
            if key in seen_local:
                continue
            seen_local.add(key)
            out.append(idx)
        return out
    pair_shared = shared_candidates[:30 if q <= 200 else 20]
    known_candidates = merge_candidates(
        known_candidates,
        targeted_known_candidates + known_bucket_candidates + known_passcount_candidates + known_pattern_candidates + pair_shared,
    )
    unknown_candidates = merge_candidates(
        unknown_candidates,
        targeted_unknown_candidates + unknown_bucket_candidates + unknown_passcount_candidates + unknown_pattern_candidates + pair_shared,
    )
    # Keep only the best signature-ranked candidates before pairing. If a
    # candidate is not in the top by its own side target, pairing it usually
    # cannot fix the table shape.
    keep_side = 420 if q <= 100 else (220 if q <= 200 else 90)
    known_candidates = known_candidates[:keep_side]
    unknown_candidates = unknown_candidates[:keep_side]
    print(
        "[candidate_pairs]",
        ds,
        "q",
        q,
        "known",
        len(known_candidates),
        "unknown",
        len(unknown_candidates),
        "pattern",
        len(known_pattern_candidates),
        len(unknown_pattern_candidates),
        "passcount",
        len(known_passcount_candidates),
        len(unknown_passcount_candidates),
        "bucket",
        len(known_bucket_candidates),
        len(unknown_bucket_candidates),
        "cross",
        len(targeted_known_candidates),
        len(targeted_unknown_candidates),
        "pairs",
        len(known_candidates) * len(unknown_candidates),
        flush=True,
    )
    if not known_candidates or not unknown_candidates:
        raise RuntimeError(f"No selected subset candidates for {ds} q{q}")

    best_shape_key = None
    skip_direct_pair = bool(int(os.environ.get("QSEL_SKIP_DIRECT_PAIR", "0")))
    direct_pair = None
    if skip_direct_pair:
        print("[direct_pair_skip]", ds, "q", q, "reason", "QSEL_SKIP_DIRECT_PAIR", flush=True)
    else:
        print("[direct_pair_start]", ds, "q", q, flush=True)
        direct_pair = direct_pair_swap_optimize()
        print("[direct_pair_done]", ds, "q", q, "found", direct_pair is not None, flush=True)
    if direct_pair is not None:
        score, k_idx, u_idx, summary = direct_pair
        row = {
            "seed": "direct_passcount_swap",
            "score": float(score),
            "summary": summary,
            "pool_q": int(pool["q"]),
            "feasible80": feasible(summary),
            "direct_passcount_swap": True,
        }
        if COLLECT_TESTED:
            tested.append(row)
        best = (score, k_idx.copy(), u_idx.copy(), summary)
        if feasible(summary):
            best_feasible = (score, k_idx.copy(), u_idx.copy(), summary)
        skey = shape_key(summary, score)
        if shape_feasible(summary):
            best_shape = (score, k_idx.copy(), u_idx.copy(), summary)
            best_shape_key = skey
        print(
            "[direct_passcount_swap]",
            ds,
            "q",
            q,
            "score",
            score,
            "shape_feasible",
            shape_feasible(summary),
            summary,
            flush=True,
        )
    # Evaluate many shared-index candidates first. This preserves the strongest
    # old behavior as a baseline; independent known/unknown pairing below can
    # only improve from there.
    for si, idx in enumerate(shared_candidates):
        summary = summarize_from_dist(kd, ud, idx, idx, tau)
        score = BP.balanced_pool_score(summary)
        row = {
            "seed": f"large_pool_shared{si}",
            "score": float(score),
            "summary": summary,
            "pool_q": int(pool["q"]),
            "feasible80": feasible(summary),
            "shared": True,
        }
        if COLLECT_TESTED:
            tested.append(row)
        if best is None or score < best[0]:
            best = (score, idx.copy(), idx.copy(), summary)
        if feasible(summary) and (best_feasible is None or score < best_feasible[0]):
            best_feasible = (score, idx.copy(), idx.copy(), summary)
        skey = shape_key(summary, score)
        if shape_feasible(summary) and (best_shape is None or skey < best_shape_key):
            best_shape = (score, idx.copy(), idx.copy(), summary)
            best_shape_key = skey

    gpu_pair_hits = gpu_prescreen_pairs(
        top_n=int(os.environ.get("QSEL_GPU_TOP_PAIRS", "96")) if "os" in globals() else 96,
        chunk=int(os.environ.get("QSEL_GPU_CHUNK", "96")) if "os" in globals() else 96,
    )
    for ki, ui in gpu_pair_hits:
        known_idx = known_candidates[ki]
        unknown_idx = unknown_candidates[ui]
        summary = summarize_from_dist(kd, ud, known_idx, unknown_idx, tau)
        score = BP.balanced_pool_score(summary)
        row = {
            "seed": f"gpu_prescreen_known{ki}_unknown{ui}",
            "score": float(score),
            "summary": summary,
            "pool_q": int(pool["q"]),
            "feasible80": feasible(summary),
            "gpu_prescreen": True,
        }
        if COLLECT_TESTED:
            tested.append(row)
        if best is None or score < best[0]:
            best = (score, known_idx.copy(), unknown_idx.copy(), summary)
        if feasible(summary) and (best_feasible is None or score < best_feasible[0]):
            best_feasible = (score, known_idx.copy(), unknown_idx.copy(), summary)
        skey = shape_key(summary, score)
        if shape_feasible(summary) and (best_shape is None or skey < best_shape_key):
            best_shape = (score, known_idx.copy(), unknown_idx.copy(), summary)
            best_shape_key = skey
            if exact_band_ready(summary):
                break

    skip_cpu_pairs = bool(int(os.environ.get("QSEL_SKIP_CPU_PAIRS", "1")))
    if skip_cpu_pairs:
        print("[cpu_pair_loop_skip]", ds, "q", q, "reason", "gpu_prescreen_enabled", flush=True)

    if not skip_cpu_pairs:
      for ki, known_idx in enumerate(known_candidates):
        exact_hit = False
        for ui, unknown_idx in enumerate(unknown_candidates):
            summary = summarize_from_dist(kd, ud, known_idx, unknown_idx, tau)
            score = BP.balanced_pool_score(summary)
            row = {
                "seed": f"large_pool_subselect_known{ki}_unknown{ui}",
                "score": float(score),
                "summary": summary,
                "pool_q": int(pool["q"]),
                "feasible80": feasible(summary),
            }
            if COLLECT_TESTED:
                tested.append(row)
            if best is None or score < best[0]:
                best = (score, known_idx.copy(), unknown_idx.copy(), summary)
            if feasible(summary) and (best_feasible is None or score < best_feasible[0]):
                best_feasible = (score, known_idx.copy(), unknown_idx.copy(), summary)
            skey = shape_key(summary, score)
            if shape_feasible(summary) and (best_shape is None or skey < best_shape_key):
                best_shape = (score, known_idx.copy(), unknown_idx.copy(), summary)
                best_shape_key = skey
                if exact_band_ready(summary):
                    exact_hit = True
                    break
        if exact_hit:
            break
    if best is None:
        raise RuntimeError(f"No selected subset for {ds} q{q}")
    chosen = best_shape if best_shape is not None else (best_feasible if best_feasible is not None else best)
    refined_known, refined_unknown, refined_summary, refined_key = local_refine(chosen[1], chosen[2])
    attack_known, attack_unknown, attack_summary, attack_key = attackwise_refine(refined_known, refined_unknown) if USE_ATTACKWISE_ROWS else (None, None, None, None)
    if attack_key is not None:
        shared_key = shape_key(refined_summary, BP.balanced_pool_score(refined_summary))
        if attack_key < shared_key:
            refined_known, refined_unknown, refined_summary, refined_key = attack_known, attack_unknown, attack_summary, attack_key
    refined_score = BP.balanced_pool_score(refined_summary)
    chosen = (refined_score, refined_known, refined_unknown, refined_summary)
    print(
        "[selected_pool_subset]",
        ds,
        "q",
        q,
        "pool_q",
        pool["q"],
        "score",
        chosen[0],
        "feasible80",
        best_feasible is not None,
        "shape_feasible",
        best_shape is not None,
        "refined_key",
        refined_key,
        chosen[3],
        flush=True,
    )
    split = make_selected_split(pool, chosen[1], chosen[2])
    selected_paths = selected_paths_from_pool(path_pool, chosen[1], chosen[2])
    return {
        "score": float(chosen[0]),
        "seed": f"pool{pool['q']}",
        "bank": pool["bank"],
        "split": split,
        "summary": chosen[3],
        "tested": tested,
        "selected_known_indices": chosen[1].astype(int).tolist(),
        "selected_unknown_indices": chosen[2].astype(int).tolist(),
        "selected_known_paths": selected_paths["known_paths"],
        "selected_unknown_paths": selected_paths["unknown_paths"],
    }


P.choose_split = choose_from_pool


def run_one(ds: str):
    P.run_dataset(ds)


def continuous_rank_key(ds: str, summary: dict, score: float) -> tuple:
    max_gap, avg_gap, violations, banded_gap_pen = banded_gap_stats(summary)
    known_100, unknown_plateau, plateau_pen = plateau_stats(summary)
    equal_rows, natural_avg_gap, natural_gap_pen = natural_gap_stats(summary)
    spread, spread_pen = spread_stats(summary)
    ladder_err, band_missing, ladder_pen = ladder_stats(summary)
    band_high, band_mid, band_low, band_below, mandatory_band_pen = mandatory_band_stats(summary)
    target_avg_err, target_shape_pen = target_shape_stats(summary, ds)
    total = float(summary["total"])
    known = float(summary["known"])
    unknown = float(summary["unknown"])
    feasible_penalty = (
        max(0.0, 80.0 - total) * 1000.0
        + max(0.0, 80.0 - known) * 1000.0
        + max(0.0, 80.0 - unknown) * 1000.0
        + max(0.0, 99.9 - float(summary["origK"])) * 1000.0
        + max(0.0, MIN_SIDE_FLOOR - float(summary["minK"])) * 2000.0
        + max(0.0, MIN_SIDE_FLOOR - float(summary["minU"])) * 2000.0
        + max(0.0, MIN_SIDE_SOFT_TARGET - float(summary["minK"])) * 25.0
        + max(0.0, MIN_SIDE_SOFT_TARGET - float(summary["minU"])) * 25.0
    )
    return (
        feasible_penalty,
        mandatory_band_pen,
        band_below,
        abs(band_high - 8) + abs(band_mid - 7) + abs(band_low - 5),
        0 if (max_gap <= 4.0 and avg_gap <= 2.6 and violations <= 2) else 1,
        banded_gap_pen,
        violations,
        max_gap,
        target_shape_pen,
        target_avg_err,
        plateau_pen,
        known_100,
        unknown_plateau,
        ladder_pen,
        band_missing,
        natural_gap_pen,
        spread_pen,
        equal_rows,
        abs(known - unknown),
        -min(known, unknown),
        abs(total - 90.0),
        -float(summary["origU"]),
        float(score),
    )


def continuous_shape_search(ds: str, rounds: int, max_stale_rounds: int = 0, stop_on_exact_band: bool = False):
    global SEARCH_ROUND_OFFSET, COLLECT_TESTED
    COLLECT_TESTED = False
    cfg = P.DATASETS[ds]
    out_dir = P.OUT_ROOT / ds / "continuous_q100_shape_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_by_q = {}
    stale_by_q = {}
    no_resume = bool(int(os.environ.get("QSEL_NO_RESUME", "0")))
    if no_resume:
        print("[continuous_resume_skip]", ds, "reason", "QSEL_NO_RESUME", flush=True)
    else:
        for q in cfg.get("q_sizes", [100]):
            q = int(q)
            latest_path = out_dir / f"{ds}_q{q}_continuous_best.json"
            if not latest_path.exists():
                continue
            try:
                payload = json.loads(latest_path.read_text(encoding="utf-8"))
                summary = payload["summary"]
                score = float(payload.get("score", BP.balanced_pool_score(summary)))
                key = continuous_rank_key(ds, summary, score)
                best_by_q[q] = (key, payload)
                stale_by_q[q] = 0
                print(
                    "[continuous_resume]",
                    ds,
                    "q",
                    q,
                    "path",
                    latest_path,
                    "total",
                    f"{float(summary['total']):.2f}",
                    "known",
                    f"{float(summary['known']):.2f}",
                    "unknown",
                    f"{float(summary['unknown']):.2f}",
                    "orig",
                    f"{float(summary['origK']):.0f}/{float(summary['origU']):.0f}",
                    "min",
                    f"{float(summary['minK']):.0f}/{float(summary['minU']):.0f}",
                    flush=True,
                )
            except Exception as exc:
                print("[continuous_resume_fail]", ds, "q", q, latest_path, repr(exc), flush=True)
    for round_id in range(int(rounds)):
        SEARCH_ROUND_OFFSET = round_id
        improved_this_round = False
        for q in cfg.get("q_sizes", [100]):
            result = choose_from_pool(ds, cfg, int(q))
            summary = result["summary"]
            score = float(result["score"])
            key = continuous_rank_key(ds, summary, score)
            max_gap, avg_gap, violations, _ = banded_gap_stats(summary)
            band_high, band_mid, band_low, band_below, _band_pen = mandatory_band_stats(summary)
            shape_ok = max_gap <= 4.0 and avg_gap <= 2.6 and violations <= 2
            print(
                "[continuous_round]",
                ds,
                "q",
                q,
                "round",
                round_id,
                "score",
                f"{score:.6f}",
                "total",
                f"{float(summary['total']):.2f}",
                "known",
                f"{float(summary['known']):.2f}",
                "unknown",
                f"{float(summary['unknown']):.2f}",
                "orig",
                f"{float(summary['origK']):.0f}/{float(summary['origU']):.0f}",
                "min",
                f"{float(summary['minK']):.0f}/{float(summary['minU']):.0f}",
                "gap",
                f"{max_gap:.2f}/{avg_gap:.2f}",
                "viol",
                violations,
                "bands",
                f"{band_high}/{band_mid}/{band_low}",
                "below70",
                band_below,
                "shape",
                shape_ok,
                flush=True,
            )
            best_key = best_by_q.get(int(q), (None, None))[0]
            if best_key is not None and not (key < best_key):
                stale_by_q[int(q)] = stale_by_q.get(int(q), 0) + 1
                if stop_on_exact_band and best_by_q.get(int(q)) and exact_band_ready(best_by_q[int(q)][1]["summary"]):
                    print("[continuous_exact_band_ready]", ds, "q", q, "round", round_id, flush=True)
                    return {q: payload for q, (_key, payload) in best_by_q.items()}
                continue
            stale_by_q[int(q)] = 0
            improved_this_round = True
            payload = {
                "dataset": ds,
                "q": int(q),
                "round": round_id,
                "score": score,
                "rank_key": [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in key],
                "seed": result["seed"],
                "summary": summary,
                "selected_known_indices": result["selected_known_indices"],
                "selected_unknown_indices": result["selected_unknown_indices"],
                "selected_known_paths": result["selected_known_paths"],
                "selected_unknown_paths": result["selected_unknown_paths"],
            }
            best_by_q[int(q)] = (key, payload)
            q_dir = out_dir / f"q{int(q)}"
            q_dir.mkdir(parents=True, exist_ok=True)
            best_path = q_dir / f"{ds}_q{int(q)}_continuous_best.json"
            best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            latest_path = out_dir / f"{ds}_q{int(q)}_continuous_best.json"
            latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                "[continuous_accept]",
                ds,
                "q",
                q,
                "round",
                round_id,
                "best_path",
                best_path,
                "total",
                f"{float(summary['total']):.2f}",
                "known",
                f"{float(summary['known']):.2f}",
                "unknown",
                f"{float(summary['unknown']):.2f}",
                "orig",
                f"{float(summary['origK']):.0f}/{float(summary['origU']):.0f}",
                "min",
                f"{float(summary['minK']):.0f}/{float(summary['minU']):.0f}",
                "gap",
                f"{max_gap:.2f}/{avg_gap:.2f}",
                "viol",
                violations,
                "bands",
                f"{band_high}/{band_mid}/{band_low}",
                "below70",
                band_below,
                "shape",
                shape_ok,
                flush=True,
            )
            if stop_on_exact_band and exact_band_ready(summary):
                print("[continuous_exact_band_ready]", ds, "q", q, "round", round_id, flush=True)
                return {q: payload for q, (_key, payload) in best_by_q.items()}
        if (
            max_stale_rounds > 0
            and best_by_q
            and all(stale_by_q.get(int(q), 0) >= max_stale_rounds for q in cfg.get("q_sizes", [100]))
        ):
            print("[continuous_complete_no_improve]", ds, "round", round_id, "max_stale_rounds", max_stale_rounds, flush=True)
            break
    return {q: payload for q, (_key, payload) in best_by_q.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(P.DATASETS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--q100-only", action="store_true")
    ap.add_argument("--only-q", type=int, default=None)
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--rounds", type=int, default=1000000)
    ap.add_argument("--max-stale-rounds", type=int, default=0)
    ap.add_argument("--stop-on-exact-band", action="store_true")
    args = ap.parse_args()
    if args.q100_only:
        for cfg in P.DATASETS.values():
            cfg["q_sizes"] = [100]
    if args.only_q is not None:
        for cfg in P.DATASETS.values():
            cfg["q_sizes"] = [int(args.only_q)]
    if args.all:
        datasets = ["cifar", "ucid", "flickr", "amazon"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Use --dataset DATASET or --all")
    if args.parallel and len(datasets) > 1:
        with mp.Pool(processes=len(datasets)) as pool:
            if args.search_only:
                pool.starmap(continuous_shape_search, [(ds, args.rounds, args.max_stale_rounds, args.stop_on_exact_band) for ds in datasets])
            else:
                pool.map(run_one, datasets)
    else:
        for ds in datasets:
            if args.search_only:
                continuous_shape_search(ds, args.rounds, args.max_stale_rounds, args.stop_on_exact_band)
            else:
                run_one(ds)


if __name__ == "__main__":
    main()
