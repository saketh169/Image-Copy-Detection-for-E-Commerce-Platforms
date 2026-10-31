#!/usr/bin/env python3
"""Compiled-kernel utilities for Phase-3 direct, HNSW, and ERNG search.

This module supplies data preparation, deterministic centroid layouts, C++
kernel loading, and exact direct/graph calls. Timing policy belongs to the
experiment runner and ``docs/timing_protocol.md``; it is deliberately not
hard-coded here so every benchmark can declare its own reproducible protocol.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from erng_flat import ERNGIndex, build_erng, query_starts, save_erng  # noqa: E402
from hnsw_hierarchical import HNSWIndex, build_hnsw, save_hnsw  # noqa: E402
from bucket_bvh import build_bucket_bvh  # noqa: E402
from point_bvh import build_point_bvh  # noqa: E402


ROOT = Path(os.environ.get("ICD_ARTIFACT_ROOT", "/home/saketh/json/final_named_artifacts_20260601"))
OUT = Path(os.environ.get("ICD_GRAPH_V2_OUT", str(ROOT / "phase3_graph_reconstructed_cpp_20260915")))
KERNEL = HERE / "graph_search_kernels.cpp"
LIB = HERE / "libgraph_search_kernels.so"
MANIFEST = Path(os.environ.get("ICD_ATTACK_MANIFEST", "/home/saketh/Codebase/Attacks_Manifest.json"))
SELECTION = ROOT / "phase3_large_pool_subselect_all4_20260717/final_selected_report_variation_current_20260720"

ATTACKS_26 = [
    "original", "rotate_m5", "rotate_m10", "rotate5", "rotate15", "rotate30", "rotate40",
    "crop5", "crop10", "crop20", "crop40", "bright_light", "bright", "heavy_bright",
    "flip_h", "flip_v", "flip", "resize_compress", "resize", "watermark_text",
    "watermark_asset", "mix_rotate10_bright", "mix_crop20_watermark", "bg_color_change", "blur", "contrast",
]

CFG = {
    "cifar": {"pool": ROOT / "phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3.npz", "tau": 0.0002912748896051198, "q": [100, 200, 500, 1000]},
    "ucid": {"pool": ROOT / "phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850.npz", "tau": 3.5645902244141325e-05, "q": [100, 200, 400, 500]},
    "flickr": {"pool": ROOT / "phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2.npz", "tau": 2.0358857000246644e-06, "q": [100, 200, 500, 1000]},
    "amazon": {"pool": ROOT / "phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000.npz", "tau": 3.77926170358478e-08, "q": [100, 200, 500, 1000]},
}

SWEEP = {
    # These are the complete method-specific centroid ranges requested for
    # the reconstruction sweep, not the previously selected candidates.
    "cifar": {
        "hnsw": [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
        "erng": [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072],
    },
    "flickr": {
        "hnsw": [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096],
        "erng": [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144],
    },
    "ucid": {
        "hnsw": [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
        "erng": [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536],
    },
    "amazon": {
        "hnsw": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192],
        "erng": [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288],
    },
}

PARAMETER_RANGES = {
    "overlap": [1, 2, 3, 4],
    "hnsw_degree": [6, 8, 12, 16, 24, 32, 48],
    "erng_degree": [6, 8, 12, 16, 24, 32],
    # 1 is canonical HNSW. Larger values are the explicitly named
    # multi-entry HNSW extension being evaluated.
    "hnsw_entry_points": [1, 2, 3, 4, 6, 8, 12, 16],
    "erng_entry_points": [1, 2, 3, 4, 6, 8, 12],
    "hnsw_ef": [2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
    "erng_ef": [2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
}

GRAPH_CONFIG = {
    "ucid": {"hnsw": {"m": 12, "degree": 24, "ef": 8, "entries": 1}, "erng": {"degree": 6, "probes": 4, "skip": 2}},
    "flickr": {"hnsw": {"m": 3, "degree": 6, "ef": 8, "entries": 1}, "erng": {"degree": 6, "probes": 4, "skip": 2}},
    "cifar": {"hnsw": {"m": 6, "degree": 12, "ef": 8, "entries": 1}, "erng": {"degree": 6, "probes": 8, "skip": 2}},
    "amazon": {"hnsw": {"m": 12, "degree": 24, "ef": 16, "entries": 1}, "erng": {"degree": 12, "probes": 4, "skip": 3}},
}

_POINT_BVH_CACHE: dict[tuple[int, tuple[int, ...]], tuple[np.ndarray, ...]] = {}


def point_bvh_for(bank: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build the exact-certificate tree once per in-memory bank."""
    key = (int(bank.ctypes.data), tuple(bank.shape))
    cached = _POINT_BVH_CACHE.get(key)
    if cached is None:
        cached = build_point_bvh(bank)
        _POINT_BVH_CACHE[key] = cached
    return cached


def compile_library() -> ctypes.CDLL:
    command = ["g++", "-O3", "-march=native", "-fopenmp", "-shared", "-std=c++17", "-fPIC", str(KERNEL), "-o", str(LIB)]
    if not LIB.exists() or LIB.stat().st_mtime < KERNEL.stat().st_mtime:
        subprocess.check_call(command)
    lib = ctypes.CDLL(str(LIB))
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int)
    lib.direct_l2_threshold_omp.argtypes = [fp, ctypes.c_int, fp, ctypes.c_int, ctypes.c_int, ctypes.c_float, fp, ip]
    lib.hnsw_hierarchical_threshold_omp.argtypes = [
        fp, ctypes.c_int, fp, ctypes.c_int, ctypes.c_int, fp, fp, ctypes.c_int,
        fp, fp, ip, ip, ip, ctypes.c_int,
        fp, fp, ip, ip, ip, ip, ip, ctypes.c_int,
        ip, ip, ip, ip, ctypes.c_int, ctypes.c_int,
        ip, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ip, ip, ip, ip,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    lib.erng_flat_threshold_omp.argtypes = [
        fp, ctypes.c_int, fp, ctypes.c_int, ctypes.c_int, fp, fp, ctypes.c_int,
        ip, ip, ip, ip, ctypes.c_int, ip, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ip, ip, ip, ip,
    ]
    return lib


def fp(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ip(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def dp(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def cfg_for(ds: str) -> dict:
    return CFG[ds]


def load_pool(ds: str) -> tuple[dict, np.ndarray]:
    path = cfg_for(ds)["pool"]
    data = np.load(path, allow_pickle=True)
    pool = {key: data[key] for key in data.files}
    return pool, np.ascontiguousarray(pool["bank"].astype(np.float32))


def attack_order(ds: str, q: int) -> list[str]:
    if MANIFEST.exists():
        try:
            payload = json.loads(MANIFEST.read_text())
            values = payload.get("datasets", {}).get(ds, {}).get("q", {}).get(str(q))
            if values:
                return list(values)
        except (OSError, json.JSONDecodeError):
            pass
    return [name for name in ATTACKS_26 if name != "flip"]


def load_selection(ds: str, q: int) -> dict:
    path = SELECTION / ds / f"{ds}_q{q}_report_variation_selected.json"
    if not path.exists():
        raise FileNotFoundError(f"missing q selection: {path}")
    return json.loads(path.read_text())


def selected_indices(selection: dict) -> tuple[np.ndarray, np.ndarray]:
    known_key = "selected_known_indices" if "selected_known_indices" in selection else "known_indices"
    unknown_key = "selected_unknown_indices" if "selected_unknown_indices" in selection else "unknown_indices"
    return np.asarray(selection[known_key], dtype=np.int64), np.asarray(selection[unknown_key], dtype=np.int64)


def query_matrix(pool: dict, selection: dict, attack: str) -> np.ndarray:
    attack_i = ATTACKS_26.index(attack)
    known_q = int(pool.get("known_q", pool.get("q")))
    unknown_q = int(pool.get("unknown_q", pool.get("q")))
    known_i, unknown_i = selected_indices(selection)
    known = pool["known_query"][attack_i * known_q:(attack_i + 1) * known_q][known_i]
    unknown = pool["unknown_query"][attack_i * unknown_q:(attack_i + 1) * unknown_q][unknown_i]
    return np.ascontiguousarray(np.concatenate([known, unknown], axis=0).astype(np.float32))


def query_batch(pool: dict, ds: str, q: int) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    selection = load_selection(ds, q)
    arrays = []
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for attack in attack_order(ds, q):
        matrix = query_matrix(pool, selection, attack)
        matrix = np.ascontiguousarray(matrix.astype(np.float32))
        end = start + matrix.shape[0]
        arrays.append(matrix)
        slices[attack] = (start, end)
        start = end
    return np.ascontiguousarray(np.concatenate(arrays, axis=0)), slices


def robust_elapsed(callable_fn, warmups: int = 8, repeats: int = 41) -> tuple[float, list[float]]:
    for _ in range(max(0, warmups)):
        callable_fn()
    values = []
    for _ in range(max(3, repeats)):
        begin = time.perf_counter()
        callable_fn()
        values.append(time.perf_counter() - begin)
    ordered = sorted(values)
    trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
    return float(np.median(trimmed)), [float(x) for x in values]


def direct_call(lib, queries: np.ndarray, bank: np.ndarray, threshold2: float):
    d2 = np.empty(len(queries), dtype=np.float32)
    dup = np.empty(len(queries), dtype=np.int32)
    lib.direct_l2_threshold_omp(fp(queries), len(queries), fp(bank), len(bank), bank.shape[1], ctypes.c_float(threshold2), fp(d2), ip(dup))
    return d2, dup


def graph_call_hnsw(lib, queries, bank, centroids, radii, cidx, offsets, index: HNSWIndex, ef: int, threshold2: float, bvh=None, point_bvh=None, profile: bool = False, bucket_limit: int = 8, entry_count: int | None = None):
    bvh = bvh if bvh is not None else build_bucket_bvh(centroids, radii)
    bvh_lo, bvh_hi, bvh_left, bvh_right, bvh_bucket = bvh
    point_bvh = point_bvh if point_bvh is not None else point_bvh_for(bank)
    point_lo, point_hi, point_left, point_right, point_start, point_end, point_order = point_bvh
    dup = np.empty(len(queries), dtype=np.int32)
    seen = np.empty(len(queries), dtype=np.int32)
    visited = np.empty(len(queries), dtype=np.int32)
    hops = np.empty(len(queries), dtype=np.int32)
    phase_ns = [np.empty(len(queries), dtype=np.float64) for _ in range(4)] if profile else None
    phase_ptrs = [dp(values) for values in phase_ns] if phase_ns is not None else [ctypes.POINTER(ctypes.c_double)() for _ in range(4)]
    starts = len(index.entry_points) if entry_count is None else max(1, min(int(entry_count), len(index.entry_points)))
    lib.hnsw_hierarchical_threshold_omp(
        fp(queries), len(queries), fp(bank), len(bank), bank.shape[1], fp(centroids), fp(radii), len(centroids),
        fp(bvh_lo), fp(bvh_hi), ip(bvh_left), ip(bvh_right), ip(bvh_bucket), len(bvh_bucket),
        fp(point_lo), fp(point_hi), ip(point_left), ip(point_right), ip(point_start), ip(point_end), ip(point_order), len(point_left),
        ip(cidx), ip(offsets), ip(index.adjacency), ip(index.degrees), index.max_level + 1, index.max_degree,
        ip(index.entry_points), ip(index.entry_levels), starts, int(ef), int(bucket_limit),
        ctypes.c_float(threshold2), ip(dup), ip(seen), ip(visited), ip(hops), *phase_ptrs,
    )
    if profile:
        return dup, seen, visited, hops, {
            "navigation_ms_per_query": phase_ns[0] / 1_000_000.0,
            "bucket_ranking_ms_per_query": phase_ns[1] / 1_000_000.0,
            "candidate_bucket_verification_ms_per_query": phase_ns[2] / 1_000_000.0,
            "exactness_certificate_ms_per_query": phase_ns[3] / 1_000_000.0,
        }
    return dup, seen, visited, hops


def graph_call_erng(
    lib,
    queries,
    bank,
    centroids,
    radii,
    cidx,
    offsets,
    index: ERNGIndex,
    probes: int,
    threshold2: float,
    ef: int = 64,
):
    dup = np.empty(len(queries), dtype=np.int32)
    seen = np.empty(len(queries), dtype=np.int32)
    visited = np.empty(len(queries), dtype=np.int32)
    hops = np.empty(len(queries), dtype=np.int32)
    starts = query_starts(index, len(queries))
    lib.erng_flat_threshold_omp(
        fp(queries), len(queries), fp(bank), len(bank), bank.shape[1], fp(centroids), fp(radii), len(centroids),
        ip(cidx), ip(offsets), ip(index.adjacency), ip(index.degrees), int(index.max_degree), ip(starts), int(probes),
        max(1, int(ef)), max(8, int(ef)), ctypes.c_float(threshold2), ip(dup), ip(seen), ip(visited), ip(hops),
    )
    return dup, seen, visited, hops


def accuracy_rows(ds: str, q: int, slices: dict[str, tuple[int, int]], decisions: np.ndarray, direct_truth: np.ndarray) -> dict:
    rows = {}
    for attack, (a, b) in slices.items():
        values = decisions[a:b].astype(bool)
        truth = direct_truth[a:b].astype(bool)
        known = float(values[:q].mean() * 100.0)
        unknown = float((~values[q:]).mean() * 100.0)
        rows[attack] = {
            "known": round(known, 4), "unknown": round(unknown, 4), "total": round((known + unknown) / 2.0, 4),
            "decision_match": round(float(np.mean(values == truth) * 100.0), 4),
        }
    totals = [row["total"] for row in rows.values()]
    return {"attacks": rows, "mean_total": round(float(np.mean(totals)), 4), "min_match": round(min(row["decision_match"] for row in rows.values()), 4)}


def prepare_direct(
    lib,
    pool: dict,
    ds: str,
    bank: np.ndarray,
    q: int,
    measure_timing: bool = True,
    timing_queries_override: np.ndarray | None = None,
) -> dict:
    queries, slices = query_batch(pool, ds, q)
    selection = load_selection(ds, q)
    tau = float(selection.get("tau", cfg_for(ds)["tau"]))
    threshold2 = tau * tau
    # Truth is always computed over the complete requested q batch. Timing
    # uses a fixed deterministic calibration sample so per-query values do
    # not change merely because q has a different batch length.
    d2, direct_dup = direct_call(lib, queries, bank, threshold2)
    if timing_queries_override is None:
        timing_count = min(5000, len(queries))
        timing_indices = np.linspace(0, len(queries) - 1, timing_count, dtype=np.int64)
        timing_queries = np.ascontiguousarray(queries[timing_indices])
    else:
        timing_queries = np.ascontiguousarray(timing_queries_override)
    holder = [None]
    if measure_timing:
        direct_seconds, direct_values = robust_elapsed(
            lambda: holder.__setitem__(0, direct_call(lib, timing_queries, bank, threshold2))
        )
    else:
        direct_seconds, direct_values = 0.0, []
    return {
        "queries": queries,
        "slices": slices,
        "tau": tau,
        "threshold2": threshold2,
        "direct_truth": direct_dup.astype(bool),
        "direct_ms": (direct_seconds * 1000.0 / len(timing_queries)) if measure_timing else None,
        "timing_query_count": len(timing_queries),
        "direct_timing_seconds": direct_values,
        "direct_d2": d2,
    }


def build_layout(ds: str, bank: np.ndarray, count: int, overlap: int, work: Path):
    # Reuse only deterministic centroid/bucket data.  The graph itself is
    # always rebuilt by this module, never loaded from an old graph npz.
    centroids, labels = load_or_build_centroids(ds, bank, count, work)
    if overlap <= 1:
        cidx, offsets, _ = cluster_flat(labels)
    else:
        cache = work / "centroids" / f"{ds}_overlap_c{len(centroids)}_r{overlap}.npz"
        cidx, offsets = overlapping_cluster_flat(bank, centroids, overlap, cache)
    radii = radii_for(bank, centroids, cidx, offsets, work, ds, len(centroids), overlap)
    return np.ascontiguousarray(centroids), np.ascontiguousarray(radii), np.ascontiguousarray(cidx), np.ascontiguousarray(offsets)


def _centroid_cache_candidates(ds: str, count: int, work: Path) -> list[Path]:
    return [
        work / "centroids" / f"{ds}_centroids_c{count}.npz",
        ROOT / "phase3_full_centroid_sweep_hnsw_erng_20260721" / ds / "centroids" / f"{ds}_centroids_c{count}.npz",
        ROOT / "phase3_fixed_entry_allq_hnsw_erng_20260721" / ds / "centroids" / f"{ds}_cpp_centroids_c{count}.npz",
        ROOT / ds / "phase3_cpp_centroid_branchbound" / f"{ds}_cpp_centroids_c{count}.npz",
        ROOT / ds / "phase3_centroid_branchbound_autotune" / f"{ds}_autotune_centroids_c{count}.npz",
        ROOT / ds / "phase3_centroid_hnsw_erng" / f"{ds}_centroid_graph_kmeans_c{count}.npz",
    ]


def _bank_fingerprint(bank: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(bank, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _nearest_labels(bank: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    labels = np.empty(len(bank), dtype=np.int32)
    for start in range(0, len(bank), 4096):
        stop = min(len(bank), start + 4096)
        diff = bank[start:stop, None, :] - centroids[None, :, :]
        labels[start:stop] = np.argmin(np.sum(diff * diff, axis=2), axis=1).astype(np.int32)
    return labels


def _numpy_centroids(bank: np.ndarray, count: int, seed: int = 20260915) -> tuple[np.ndarray, np.ndarray]:
    """Small dependency-free fallback for hosts without scikit-learn."""
    if count < 1 or count > len(bank):
        raise ValueError(f"centroid count must be in [1, bank size]; got C={count}, bank={len(bank)}")
    rng = np.random.default_rng(seed + count)
    sample_size = min(len(bank), max(count * 8, 32768))
    sample = bank[rng.choice(len(bank), sample_size, replace=False)]
    centroids = sample[rng.choice(sample_size, count, replace=False)].copy()
    for _ in range(32):
        labels = _nearest_labels(sample, centroids)
        sums = np.zeros_like(centroids, dtype=np.float64)
        counts = np.bincount(labels, minlength=count).astype(np.int64)
        np.add.at(sums, labels, sample.astype(np.float64))
        valid = counts > 0
        centroids[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
        if not np.all(valid):
            centroids[~valid] = sample[rng.choice(sample_size, int((~valid).sum()), replace=False)]
    return np.ascontiguousarray(centroids.astype(np.float32)), _nearest_labels(bank, centroids)


def load_or_build_centroids(ds: str, bank: np.ndarray, count: int, work: Path) -> tuple[np.ndarray, np.ndarray]:
    if count < 1 or count > len(bank):
        raise ValueError(f"centroid count must be in [1, bank size]; got C={count}, bank={len(bank)}")
    fingerprint = _bank_fingerprint(bank)
    target = work / "centroids" / f"{ds}_centroids_c{count}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as data:
            metadata = str(data["metadata"].item()) if "metadata" in data.files else ""
            expected = json.dumps({"bank_sha256": fingerprint, "count": count, "algorithm": "numpy-kmeans-sample-v2"}, sort_keys=True)
            if metadata == expected and "centroids" in data.files and "labels" in data.files:
                return np.ascontiguousarray(data["centroids"], dtype=np.float32), np.ascontiguousarray(data["labels"], dtype=np.int32)
        print(f"[{ds}] rejecting centroid cache with missing/mismatched provenance: {target}", flush=True)
    print(f"[{ds}] building bank-matched deterministic centroids C={count}", flush=True)
    centroids, labels = _numpy_centroids(bank, count)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({"bank_sha256": fingerprint, "count": count, "algorithm": "numpy-kmeans-sample-v2"}, sort_keys=True)
    np.savez_compressed(target, centroids=centroids, labels=labels, metadata=np.asarray(metadata))
    return centroids, labels


def cluster_flat(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    count = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable").astype(np.int32)
    counts = np.bincount(labels, minlength=count).astype(np.int32)
    offsets = np.zeros(count + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    return np.ascontiguousarray(order), np.ascontiguousarray(offsets), counts


def overlapping_cluster_flat(bank: np.ndarray, centroids: np.ndarray, overlap: int, cache_path: Path):
    signature = json.dumps({"bank_sha256": _bank_fingerprint(bank), "centroids_sha256": _bank_fingerprint(centroids), "overlap": int(overlap), "algorithm": "nearest-topk-v1"}, sort_keys=True)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            metadata = str(data["metadata"].item()) if "metadata" in data.files else ""
            if metadata == signature:
                return np.ascontiguousarray(data["cluster_indices"], dtype=np.int32), np.ascontiguousarray(data["offsets"], dtype=np.int32)
    overlap = min(int(overlap), len(centroids))
    memberships = [[] for _ in range(len(centroids))]
    for start in range(0, len(bank), 1024):
        stop = min(len(bank), start + 1024)
        diff = bank[start:stop, None, :] - centroids[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        nearest = np.argpartition(d2, kth=overlap - 1, axis=1)[:, :overlap]
        for row, ids in enumerate(nearest):
            for cid in ids:
                memberships[int(cid)].append(start + row)
    counts = np.asarray([len(x) for x in memberships], dtype=np.int32)
    offsets = np.zeros(len(centroids) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts, dtype=np.int32)
    values = np.empty(int(offsets[-1]), dtype=np.int32)
    pos = 0
    for ids in memberships:
        values[pos:pos + len(ids)] = ids
        pos += len(ids)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, cluster_indices=values, offsets=offsets, overlap=np.asarray([overlap], dtype=np.int32), metadata=np.asarray(signature))
    return np.ascontiguousarray(values), np.ascontiguousarray(offsets)


def radii_for(bank: np.ndarray, centroids: np.ndarray, cidx: np.ndarray, offsets: np.ndarray, work: Path, ds: str, count: int, overlap: int):
    path = work / "radii" / f"{ds}_radii_c{count}_o{overlap}.npz"
    signature = json.dumps({"bank_sha256": _bank_fingerprint(bank), "centroids_sha256": _bank_fingerprint(centroids), "count": int(count), "overlap": int(overlap), "algorithm": "bucket-max-radius-v1"}, sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            metadata = str(data["metadata"].item()) if "metadata" in data.files else ""
            if metadata == signature:
                return np.ascontiguousarray(data["radii"], dtype=np.float32)
    radii = np.zeros(len(centroids), dtype=np.float32)
    for cid in range(len(centroids)):
        ids = cidx[offsets[cid]:offsets[cid + 1]]
        if len(ids):
            diff = bank[ids] - centroids[cid]
            radii[cid] = np.sqrt(float(np.max(np.sum(diff * diff, axis=1))))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, radii=radii, metadata=np.asarray(signature))
    return np.ascontiguousarray(radii)


def run_candidate(lib, ds: str, pool: dict, bank: np.ndarray, q: int, count: int, overlap: int, work: Path, method: str, direct_info: dict | None = None, shared_direct_ms: float | None = None, config_override: dict | None = None):
    cfg = dict(GRAPH_CONFIG[ds][method])
    if config_override:
        cfg.update(config_override)
    centroids, radii, cidx, offsets = build_layout(ds, bank, count, overlap, work)
    if method == "hnsw":
        index = build_hnsw(
            centroids, m=cfg["m"], max_degree_base=cfg["degree"],
            ef_construction=max(64, cfg["ef"] * 4), seed=20260915 + count,
            entry_count=int(cfg.get("entries", 1)),
        )
        graph_path = work / "graphs" / f"{ds}_hnsw_hierarchical_c{count}_o{overlap}.npz"
        save_hnsw(index, graph_path)
    else:
        local = max(1, cfg["degree"] - cfg["skip"])
        index = build_erng(centroids, local_degree=local, skip_count=cfg["skip"], probes=cfg["probes"], seed=20260915 + count)
        graph_path = work / "graphs" / f"{ds}_erng_flat_c{count}_o{overlap}.npz"
        save_erng(index, graph_path)

    direct_info = direct_info or prepare_direct(lib, pool, ds, bank, q)
    queries = direct_info["queries"]
    slices = direct_info["slices"]
    tau = direct_info["tau"]
    threshold2 = direct_info["threshold2"]
    direct_truth = direct_info["direct_truth"]

    if method == "hnsw":
        bvh = build_bucket_bvh(centroids, radii)
        graph_fn = lambda: graph_call_hnsw(lib, queries, bank, centroids, radii, cidx, offsets, index, cfg["ef"], threshold2, bvh)
    else:
        graph_fn = lambda: graph_call_erng(lib, queries, bank, centroids, radii, cidx, offsets, index, cfg["probes"], threshold2)

    graph_result = [None]
    graph_seconds, graph_values = robust_elapsed(lambda: graph_result.__setitem__(0, graph_fn()))
    graph_dup, seen, visited, hops = graph_result[0]
    # Direct timing is the saved timing for this exact q size and is shared by
    # HNSW and ERNG. It is never read from an older graph JSON.
    direct_ms = float(shared_direct_ms if shared_direct_ms is not None else direct_info["direct_ms"])
    graph_ms = graph_seconds * 1000.0 / len(queries)
    payload = {
        "dataset": ds, "method": method, "q": q, "centroids": count, "overlap": overlap,
        "graph_parameters": cfg, "graph_file": str(graph_path), "centroid_count": len(centroids),
        "num_queries": len(queries), "tau": tau, "direct_truth_source": "same compiled C++ batch",
        "direct_baseline_source": "same complete q workload split into q100-sized batches",
        "direct_ms_per_query": direct_ms, "graph_ms_per_query": graph_ms, "speedup": direct_ms / max(graph_ms, 1e-12),
        "direct_timing_seconds": direct_info["direct_timing_seconds"], "graph_timing_seconds": graph_values,
        "decision_match": float(np.mean(graph_dup.astype(bool) == direct_truth) * 100.0),
        "avg_seen": float(np.mean(seen)), "avg_visited": float(np.mean(visited)), "avg_hops": float(np.mean(hops)),
        "accuracy": accuracy_rows(ds, q, slices, graph_dup, direct_truth),
    }
    return payload


def run_direct_dataset(ds: str, assigned_gpu: str, only_q: int | None = None) -> None:
    """Measure direct search for every q size before any graph run."""
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / ds / "direct"
    work.mkdir(parents=True, exist_ok=True)
    lib = compile_library()
    pool, bank = load_pool(ds)
    rows = []
    q_values = [only_q] if only_q is not None else list(cfg_for(ds)["q"])
    for q in q_values:
        saved_q = work / f"direct_q{q}.json"
        if only_q is None and saved_q.exists():
            rows.append(json.loads(saved_q.read_text()))
            print(f"[{ds}] direct q={q} already saved; reusing", flush=True)
            continue
        print(f"[{ds}] direct calibration q={q} gpu={assigned_gpu}", flush=True)
        info = prepare_direct(lib, pool, ds, bank, q, measure_timing=False)
        # Time the complete q workload in q100-sized chunks.  This preserves
        # the actual q workload while preventing one very large call from
        # dominating the measurement.
        chunk_size = len(query_batch(pool, ds, 100)[0])
        queries = info["queries"]
        chunk_times = []
        for start in range(0, len(queries), chunk_size):
            chunk = np.ascontiguousarray(queries[start:start + chunk_size])
            holder = [None]
            elapsed, _ = robust_elapsed(
                lambda: holder.__setitem__(0, direct_call(lib, chunk, bank, info["threshold2"]))
            )
            chunk_times.append(elapsed)
        info["direct_ms"] = 1000.0 * sum(chunk_times) / len(queries)
        info["timing_query_count"] = len(queries)
        info["direct_timing_seconds"] = chunk_times
        row = {
            "q": q,
            "num_queries": len(info["queries"]),
            "direct_ms_per_query": info["direct_ms"],
            "timing_query_count": info["timing_query_count"],
            "timing_reference": "complete q workload split into q100-sized batches",
            "timing_batch_size": chunk_size,
            "timing_batch_count": len(info["direct_timing_seconds"]),
            "timing_seconds": info["direct_timing_seconds"],
        }
        rows.append(row)
        saved_q.write_text(json.dumps({"dataset": ds, "assigned_gpu": assigned_gpu, **row}, indent=2))
        (work / "direct_progress.json").write_text(json.dumps({"dataset": ds, "q_rows": rows}, indent=2))
    if only_q is not None:
        print(f"[{ds}] DIRECT_Q_DONE q={only_q}", flush=True)
        return
    # The median across q-specific robust measurements is the stable shared
    # denominator used for every graph candidate in this dataset.
    shared = float(np.median([row["direct_ms_per_query"] for row in rows]))
    payload = {
        "dataset": ds,
        "assigned_gpu": assigned_gpu,
        "protocol": "8 warmups, 41 measured C++ calls, trim min/max, median",
        "q_rows": rows,
        "shared_direct_ms_per_query": shared,
    }
    path = work / "direct_timing.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[{ds}] DIRECT_DONE {path}", flush=True)


def run_direct_validate(ds: str, assigned_gpu: str) -> None:
    """Calibrate q100 once, then compute truth-only passes for other q sizes."""
    OUT.mkdir(parents=True, exist_ok=True)
    work = OUT / ds / "direct"
    work.mkdir(parents=True, exist_ok=True)
    lib = compile_library()
    pool, bank = load_pool(ds)
    rows = []
    for q in cfg_for(ds)["q"]:
        timed = q == 100
        print(f"[{ds}] direct {'timing' if timed else 'truth'} q={q} gpu={assigned_gpu}", flush=True)
        info = prepare_direct(lib, pool, ds, bank, q, measure_timing=timed)
        truth_path = work / f"direct_truth_q{q}.npz"
        np.savez_compressed(truth_path, direct_dup=info["direct_truth"], direct_d2=info["direct_d2"])
        row = {
            "q": q,
            "num_queries": len(info["queries"]),
            "direct_truth_file": str(truth_path),
            "direct_ms_per_query": info["direct_ms"] if timed else None,
            "timing_seconds": info["direct_timing_seconds"] if timed else [],
        }
        rows.append(row)
        (work / "direct_validation_progress.json").write_text(json.dumps({"dataset": ds, "q_rows": rows}, indent=2))
    shared = float(next(row["direct_ms_per_query"] for row in rows if row["q"] == 100))
    payload = {
        "dataset": ds,
        "assigned_gpu": assigned_gpu,
        "protocol": "q100: 8 warmups, 41 measured C++ calls, trim min/max, median; other q sizes: one truth-only pass",
        "q_rows": rows,
        "shared_direct_ms_per_query": shared,
    }
    path = work / "direct_timing.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[{ds}] DIRECT_VALIDATE_DONE {path}", flush=True)


def run_dataset(ds: str, phase: str, assigned_gpu: str, method_filter: str = "both") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Separate method roots are mandatory when HNSW and ERNG run in parallel;
    # otherwise their progress/result JSON files could overwrite one another.
    work = OUT / ds / (method_filter if method_filter != "both" else "combined")
    work.mkdir(parents=True, exist_ok=True)
    lib = compile_library()
    pool, bank = load_pool(ds)
    qs = list(cfg_for(ds)["q"])
    records = {"dataset": ds, "phase": phase, "assigned_gpu": assigned_gpu, "bank_shape": list(bank.shape), "records": []}
    direct_cache: dict[int, dict] = {}
    direct_file = OUT / ds / "direct" / "direct_timing.json"
    saved_direct = json.loads(direct_file.read_text()) if direct_file.exists() else None

    def direct_for(q: int) -> dict:
        if q not in direct_cache:
            if saved_direct is not None:
                print(f"[{ds}] direct truth once for q={q}; using saved direct denominator", flush=True)
                direct_cache[q] = prepare_direct(lib, pool, ds, bank, q, measure_timing=False)
                saved_row = next(row for row in saved_direct["q_rows"] if int(row["q"]) == int(q))
                direct_cache[q]["direct_ms"] = float(saved_row["direct_ms_per_query"])
                direct_cache[q]["direct_timing_seconds"] = saved_row.get("timing_seconds", [])
            elif q == 100:
                print(f"[{ds}] direct timing calibration once for q=100", flush=True)
                direct_cache[q] = prepare_direct(lib, pool, ds, bank, q, measure_timing=True)
            else:
                print(f"[{ds}] direct truth once for q={q}; reusing q100 timing", flush=True)
                direct_cache[q] = prepare_direct(lib, pool, ds, bank, q, measure_timing=False)
        return direct_cache[q]

    direct_reference = direct_for(100)
    shared_direct_ms = None

    if phase == "sweep":
        # q100 is the calibration query set for every centroid candidate;
        # selected configurations then run all available q sizes in phase=final.
        methods = (method_filter,) if method_filter != "both" else ("hnsw", "erng")
        for method in methods:
            for count in SWEEP[ds][method]:
                if count > len(bank):
                    continue
                print(f"[{ds}] sweep {method} C={count} q=100 gpu={assigned_gpu}", flush=True)
                entry_values = PARAMETER_RANGES["hnsw_entry_points"] if method == "hnsw" else PARAMETER_RANGES["erng_entry_points"]
                for entry_count in entry_values:
                    variant = f"c{count}_e{entry_count}"
                    override = {"entries": entry_count} if method == "hnsw" else {"probes": entry_count}
                    print(f"[{ds}] sweep {method} C={count} ent={entry_count} q=100 gpu={assigned_gpu}", flush=True)
                    item = run_candidate(lib, ds, pool, bank, 100, count, 1, work / "sweep" / variant / method, method, direct_for(100), None, override)
                    records["records"].append(item)
                    (work / "sweep_progress.json").write_text(json.dumps(records, indent=2))
    else:
        # Use the planned centroid count and overlap, then validate every q
        # size with the same built index and timing procedure.
        count = {"ucid": 96, "flickr": 384, "cifar": 768, "amazon": 1536}[ds]
        overlap = {"ucid": 2, "flickr": 4, "cifar": 4, "amazon": 4}[ds]
        methods = (method_filter,) if method_filter != "both" else ("hnsw", "erng")
        run_qs = [100] if phase == "q100" else qs
        for method in methods:
            for q in run_qs:
                print(f"[{ds}] final {method} C={count} q={q} gpu={assigned_gpu}", flush=True)
                item = run_candidate(lib, ds, pool, bank, q, count, overlap, work / phase / method, method, direct_for(q), None)
                records["records"].append(item)
                (work / "final_progress.json").write_text(json.dumps(records, indent=2))
    (work / f"{phase}_results.json").write_text(json.dumps(records, indent=2))
    print(f"[{ds}] DONE {work / (phase + '_results.json')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(CFG))
    parser.add_argument("--phase", choices=["direct", "direct_validate", "sweep", "smoke", "q100", "final"], default="sweep")
    parser.add_argument("--method", choices=["both", "hnsw", "erng"], default="both")
    parser.add_argument("--assigned-gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "unassigned"))
    parser.add_argument("--q", type=int, default=None, help="Run one q size only in direct phase")
    args = parser.parse_args()
    if args.phase == "direct":
        run_direct_dataset(args.dataset, args.assigned_gpu, args.q)
    elif args.phase == "direct_validate":
        run_direct_validate(args.dataset, args.assigned_gpu)
    else:
        run_dataset(args.dataset, args.phase, args.assigned_gpu, args.method)


if __name__ == "__main__":
    main()
