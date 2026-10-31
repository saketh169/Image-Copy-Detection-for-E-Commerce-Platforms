#!/usr/bin/env python3
"""Iterable HNSW sweeps with full-Q validation and shared direct baselines."""
from __future__ import annotations

import hashlib
import argparse
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

# A single-query production-style measurement should not spawn a large OpenMP
# team for every individual call. Respect an explicit caller setting.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import graph_benchmark as B
import hnsw_hierarchical as H
from timing import latency_summary, trimmed_median


OUT_ROOT = os.environ.get("HNSW_SWEEP_OUT") or os.environ.get("HNSW_UCID_OUT")
if not OUT_ROOT:
    raise RuntimeError("Set HNSW_SWEEP_OUT to a fresh campaign output directory")
OUT = Path(OUT_ROOT)
DATASET_BASELINES = {
    "ucid": {"overlap": 2, "degree": 8, "entries": 1, "ef": 8},
    "flickr": {"overlap": 3, "degree": 8, "entries": 1, "ef": 8},
    "cifar": {"overlap": 3, "degree": 12, "entries": 2, "ef": 12},
    "amazon": {"overlap": 4, "degree": 16, "entries": 2, "ef": 16},
}
VALUES = {
    "overlap": [1, 2, 3, 4],
    "degree": [6, 8, 12, 16, 24, 32, 48],
    "entries": [1, 2, 3, 4, 6, 8, 12, 16],
    "ef": [2, 4, 6, 8, 12, 16, 24, 32, 48, 64],
}
SCREEN_REPEATS = 3
DIRECT_WARMUPS = 1
DIRECT_REPEATS = 9
FINAL_WARMUPS = 8
FINAL_REPEATS = 41
ENTRY_CAPACITY = 16
TIMING_QUERIES_PER_CELL = 5


def _pairwise_coverage_rows(levels: tuple[list[int], ...], seed_rows: list[tuple[int, ...]]) -> tuple[list[tuple[int, ...]], dict]:
    """Greedily cover every value pair across factors, preserving reference rows."""
    factor_pairs = [
        (left, right)
        for left in range(len(levels))
        for right in range(left + 1, len(levels))
    ]
    universe = {
        (left, left_value, right, right_value)
        for left, right in factor_pairs
        for left_value in levels[left]
        for right_value in levels[right]
    }
    candidates = list(itertools.product(*levels))
    candidates.sort(key=lambda row: hashlib.sha256(json.dumps(row).encode("ascii")).digest())

    def covered(row: tuple[int, ...]) -> set[tuple[int, int, int, int]]:
        return {
            (left, row[left], right, row[right])
            for left, right in factor_pairs
        }

    rows: list[tuple[int, ...]] = []
    for row in seed_rows:
        if row not in candidates:
            raise ValueError(f"Seed setting is outside the factor levels: {row}")
        if row not in rows:
            rows.append(row)
    uncovered = universe.copy()
    for row in rows:
        uncovered.difference_update(covered(row))

    while uncovered:
        best = max(candidates, key=lambda row: len(covered(row) & uncovered))
        newly_covered = covered(best) & uncovered
        if not newly_covered:
            raise RuntimeError("Could not complete pairwise candidate coverage")
        if best not in rows:
            rows.append(best)
        uncovered.difference_update(newly_covered)

    coverage = {
        "factor_pair_count": len(factor_pairs),
        "required_value_pair_count": len(universe),
        "covered_value_pair_count": len(universe - uncovered),
        "all_factor_value_pairs_covered": not uncovered,
        "candidate_count": len(rows),
        "full_cartesian_count": len(candidates),
        "coverage_strength": 2,
    }
    return rows, coverage


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def make_candidate_plan(centroid_counts: list[int], dataset: str) -> tuple[list[tuple[int, list[dict]]], dict]:
    factors = (
        VALUES["overlap"],
        VALUES["entries"],
        VALUES["degree"],
        VALUES["ef"],
    )
    tuples, pairwise = _pairwise_coverage_rows(factors, [])
    configs = [
        {
            "overlap": overlap,
            "entries": entries,
            "degree": degree,
            "ef": ef,
            "ef_construction": max(64, degree * 4),
        }
        for overlap, entries, degree, ef in tuples
    ]
    plan = [(count, [dict(cfg) for cfg in configs]) for count in centroid_counts]
    total = sum(len(candidate_configs) for _, candidate_configs in plan)
    return plan, {
        "method": "deterministic greedy strength-2 covering array over every declared overlap, entry, degree, and ef level; every candidate is evaluated independently at each centroid count",
        "factor_order": ["overlap", "entries", "degree", "ef"],
        "factor_levels": {"overlap": VALUES["overlap"], "entries": VALUES["entries"], "degree": VALUES["degree"], "ef": VALUES["ef"]},
        "seed_candidates": [],
        "candidate_order_tiebreak": "stable hash ordering; no baseline or historical configuration is seeded or weighted",
        "pairwise_coverage": pairwise,
        "centroid_counts": centroid_counts,
        "settings_by_centroid": {str(c): len(candidate_configs) for c, candidate_configs in plan},
        "total_settings": total,
        "settings_per_centroid": len(configs),
        "full_cartesian_grid_tested": False,
        "same_candidate_coverage_at_each_centroid_for_fair_comparison": True,
    }


def sample_indices(slices: dict[str, tuple[int, int]], q: int, seed: int, count_per_cell: int = TIMING_QUERIES_PER_CELL) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for attack, (start, end) in slices.items():
        if end - start != 2 * q:
            raise ValueError(f"Unexpected query slice size for {attack}")
        for offset in (start, start + q):
            take = min(q, count_per_cell)
            chosen.extend((offset + np.sort(rng.choice(q, take, replace=False))).tolist())
    return np.asarray(chosen, dtype=np.int64)


def batch_screen(ds, lib, bank, centroids, radii, cidx, offsets, index, params, prepared, bvh, direct_sample_ms):
    rows = []
    for q in B.CFG[ds]["q"]:
        info = prepared[q]
        sample = info["sample_queries"]
        direct_truth = info["direct_truth"][info["sample_indices"]]
        per_query_medians: list[float] = []
        decisions = np.empty(len(sample), dtype=np.int32)
        image_counts = []
        centroid_counts = []
        for qi, query in enumerate(sample):
            one = np.ascontiguousarray(query[None, :])
            measurements = []
            last = None
            for repeat in range(SCREEN_REPEATS + 1):
                begin = time.perf_counter()
                last = B.graph_call_hnsw(
                    lib, one, bank, centroids, radii, cidx, offsets, index,
                    params["ef"], info["threshold2"], bvh,
                    entry_count=params["entries"],
                )
                elapsed = time.perf_counter() - begin
                if repeat:
                    measurements.append(elapsed)
            decisions[qi] = int(last[0][0])
            image_counts.append(int(last[1][0]))
            centroid_counts.append(int(last[2][0]))
            per_query_medians.append(trimmed_median(measurements))
        match = float(np.mean(decisions.astype(bool) == direct_truth) * 100.0)
        timing = latency_summary(per_query_medians)
        direct_mean = float(np.mean(direct_sample_ms[q]))
        rows.append({
            "q": q,
            "sample_count": len(sample),
            "sample_match_pct": match,
            "direct_sample_mean_ms_per_query": direct_mean,
            "hnsw_query_latency": timing,
            "screen_speedup": direct_mean / max(timing["mean_ms"], 1e-12),
            "avg_images_checked": float(np.mean(image_counts)),
            "avg_centroids_visited": float(np.mean(centroid_counts)),
        })
    q_means = [r["hnsw_query_latency"]["mean_ms"] for r in rows]
    spread = (max(q_means) - min(q_means)) / max(statistics.mean(q_means), 1e-12) * 100.0
    return rows, spread


def full_validate(ds, lib, bank, centroids, radii, cidx, offsets, index, params, prepared, bvh):
    rows = []
    for q in B.CFG[ds]["q"]:
        info = prepared[q]
        dup, seen, visited, hops, phases = B.graph_call_hnsw(
            lib, info["queries"], bank, centroids, radii, cidx, offsets,
            index, params["ef"], info["threshold2"], bvh,
            profile=True, entry_count=params["entries"],
        )
        per_query_ms = sum(phases.values())
        worst = np.argsort(per_query_ms)[-10:][::-1]
        worst_rows = []
        attack_labels = np.full(len(dup), "unknown_attack", dtype="U64")
        class_labels = np.full(len(dup), "unknown_class", dtype="U16")
        for attack, (start, end) in info["slices"].items():
            attack_labels[start:end] = attack
            class_labels[start:start + q] = "known"
            class_labels[start + q:end] = "unknown"
        for qi in worst:
            worst_rows.append({
                "query_index": int(qi),
                "attack": str(attack_labels[qi]),
                "class": str(class_labels[qi]),
                "internal_ms": float(per_query_ms[qi]),
                "images_checked": int(seen[qi]),
                "centroids_visited": int(visited[qi]),
                "hops": int(hops[qi]),
            })
        rows.append({
            "q": q,
            "query_count": len(dup),
            "decision_match_pct": float(np.mean(dup.astype(bool) == info["direct_truth"]) * 100.0),
            "avg_images_checked": float(np.mean(seen)),
            "avg_centroids_visited": float(np.mean(visited)),
            "avg_hops": float(np.mean(hops)),
            "full_workload_internal_query_latency": latency_summary(per_query_ms / 1000.0),
            "full_workload_phase_latency": {
                name: latency_summary(values / 1000.0)
                for name, values in phases.items()
            },
            "slowest_queries": worst_rows,
            "_profile_arrays": {
                "query_internal_ms": per_query_ms,
                **{f"{name}_ms": values for name, values in phases.items()},
                "attack": attack_labels,
                "class": class_labels,
            },
        })
    return rows


def final_graph_timing(ds, lib, bank, centroids, radii, cidx, offsets, index, params, prepared, bvh, direct_sample_ms):
    rows = []
    for q in B.CFG[ds]["q"]:
        info = prepared[q]
        graph_call = lambda one: B.graph_call_hnsw(
            lib, one, bank, centroids, radii, cidx, offsets, index,
            params["ef"], info["threshold2"], bvh,
            entry_count=params["entries"],
        )
        direct_query_medians = [value / 1000.0 for value in direct_sample_ms[q]]
        graph_query_medians: list[float] = []
        for query in info["sample_queries"]:
            one = np.ascontiguousarray(query[None, :])
            for _ in range(FINAL_WARMUPS):
                graph_call(one)
            gvals = []
            for _ in range(FINAL_REPEATS):
                begin = time.perf_counter()
                graph_call(one)
                gvals.append(time.perf_counter() - begin)
            graph_query_medians.append(trimmed_median(gvals))
        dm = statistics.mean(direct_query_medians)
        gm = statistics.mean(graph_query_medians)
        rows.append({
            "q": q,
            "timed_sample_count": len(info["sample_queries"]),
            "direct_mean_ms_per_query": dm * 1000.0,
            "hnsw_mean_ms_per_query": gm * 1000.0,
            "q_mean_speedup": dm / max(gm, 1e-12),
            "direct_query_ms_min_mean_max": {
                "min": min(direct_query_medians) * 1000.0,
                "mean": statistics.mean(direct_query_medians) * 1000.0,
                "max": max(direct_query_medians) * 1000.0,
            },
            "direct_baseline_shared_across_candidates": True,
            "hnsw_query_ms_min_mean_max": {
                "min": min(graph_query_medians) * 1000.0,
                "mean": statistics.mean(graph_query_medians) * 1000.0,
                "max": max(graph_query_medians) * 1000.0,
            },
            "hnsw_query_latency_distribution": latency_summary(graph_query_medians),
        })
    speeds = [r["q_mean_speedup"] for r in rows]
    mean_speed = statistics.mean(speeds)
    speed_spread = (max(speeds) - min(speeds)) / max(mean_speed, 1e-12) * 100.0
    q_latencies = [r["hnsw_mean_ms_per_query"] for r in rows]
    latency_spread = (max(q_latencies) - min(q_latencies)) / max(statistics.mean(q_latencies), 1e-12) * 100.0
    return rows, [min(speeds), mean_speed, max(speeds)], speed_spread, latency_spread


def _metric_rank(final: dict, centroid_reference: float) -> tuple[float, float, float, float, float, float]:
    timings = final["final_timing_rows"]
    return (
        float(max(r["hnsw_mean_ms_per_query"] for r in timings)),
        float(final["cross_q_latency_mean_spread_pct"]),
        float(max(r["hnsw_query_ms_min_mean_max"]["max"] for r in timings)),
        float(max(r["hnsw_query_latency_distribution"]["p95_ms"] for r in timings)),
        float(statistics.mean(r["hnsw_mean_ms_per_query"] for r in timings)),
        abs(float(final["centroids"]) - centroid_reference),
    )


def _balanced_selection(exact_finalists: list[dict], centroid_reference: float) -> dict | None:
    if not exact_finalists:
        return None
    return min(exact_finalists, key=lambda row: _metric_rank(row, centroid_reference))


def run(ds: str, no_selection: bool = False) -> None:
    manifest_env = os.environ.get("ICD_ATTACK_MANIFEST")
    if not manifest_env or not Path(manifest_env).is_file():
        raise RuntimeError("Set ICD_ATTACK_MANIFEST to the verified 19-attack manifest")
    out = OUT / ds
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Refusing to reuse a non-empty run directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    lib = B.compile_library()
    pool, bank = B.load_pool(ds)
    bank_contiguous = np.ascontiguousarray(bank)
    bank_digest = hashlib.sha256()
    bank_digest.update(np.asarray(bank_contiguous.shape, dtype=np.int64).tobytes())
    bank_digest.update(bank_contiguous.view(np.uint8))
    bank_hash = bank_digest.hexdigest()
    manifest_hash = hashlib.sha256(Path(manifest_env).read_bytes()).hexdigest()
    prepared: dict[int, dict] = {}
    for q in B.CFG[ds]["q"]:
        info = B.prepare_direct(lib, pool, ds, bank, q, measure_timing=False)
        attacks = list(info["slices"])
        if len(attacks) != 19 or "original" not in attacks or "flip_h" not in attacks or "flip" in attacks:
            raise RuntimeError(f"Expected the 19 current attacks (including original and flip_h, excluding flip); got {len(attacks)}: {attacks}")
        indices = sample_indices(info["slices"], q, 20260927 + q)
        expected = len(info["slices"]) * 2 * TIMING_QUERIES_PER_CELL
        if len(indices) != expected:
            raise RuntimeError(f"Expected {expected} stratified timing queries; got {len(indices)} at Q={q}")
        info["sample_indices"] = indices
        info["sample_queries"] = np.ascontiguousarray(info["queries"][indices])
        prepared[q] = info

    # Establish a fixed one-query direct baseline for exactly the same panel
    # used by every graph candidate; never batch-time and divide by Q.
    direct_sample_ms: dict[int, list[float]] = {}
    for q, info in prepared.items():
        values = []
        for query in info["sample_queries"]:
            one = np.ascontiguousarray(query[None, :])
            B.direct_call(lib, one, bank, info["threshold2"])
            for _ in range(DIRECT_WARMUPS):
                B.direct_call(lib, one, bank, info["threshold2"])
            samples = []
            for _ in range(DIRECT_REPEATS):
                begin = time.perf_counter()
                B.direct_call(lib, one, bank, info["threshold2"])
                samples.append(time.perf_counter() - begin)
            values.append(trimmed_median(samples) * 1000.0)
        direct_sample_ms[q] = values
    atomic_json(out / "direct_query_panel.json", {
        "protocol": "one query per call; one warmup plus repeated direct timings; fixed panel shared by every candidate",
        "queries_per_attack_class": TIMING_QUERIES_PER_CELL,
        "q_rows": {
            str(q): {
                "query_count": len(values),
                "latency": latency_summary([x / 1000.0 for x in values]),
                "per_query_ms": values,
            }
            for q, values in direct_sample_ms.items()
        },
    })

    centroid_counts = [int(c) for c in B.SWEEP[ds]["hnsw"] if int(c) <= len(bank)]
    candidate_plan, coverage = make_candidate_plan(centroid_counts, ds)
    total = sum(len(configs) for _, configs in candidate_plan)
    screen_path = out / "screen_records.jsonl"
    records: list[dict] = []
    work = out / "structures"
    layout_cache = {}
    bvh_cache = {}
    index_cache = {}

    def structure_for(count: int, params: dict):
        layout_key = (count, int(params["overlap"]))
        if layout_key not in layout_cache:
            layout_cache[layout_key] = B.build_layout(ds, bank, count, layout_key[1], work)
            bvh_cache[layout_key] = B.build_bucket_bvh(layout_cache[layout_key][0], layout_cache[layout_key][1])
            for (cached_count, _), cached_layout in layout_cache.items():
                if cached_count == count and not np.array_equal(cached_layout[0], layout_cache[layout_key][0]):
                    raise RuntimeError(f"Centroid identity changed across overlap layouts at C={count}")
        graph_key = (count, int(params["degree"]))
        if graph_key not in index_cache:
            centroids = layout_cache[layout_key][0]
            degree = graph_key[1]
            index_cache[graph_key] = H.build_hnsw(
                centroids,
                m=max(2, degree // 2),
                max_degree_base=degree,
                ef_construction=max(64, degree * 4),
                seed=20260915 + count,
                entry_count=ENTRY_CAPACITY,
            )
        return (*layout_cache[layout_key], index_cache[graph_key], bvh_cache[layout_key])

    progress = {"dataset": ds, "phase": "iterable_full_validation", "completed": 0, "total": total, "coverage": coverage}
    atomic_json(out / "progress.json", progress)
    number = 0
    for count, configs in candidate_plan:
        for params in configs:
            number += 1
            centroids, radii, cidx, offsets, index, bvh = structure_for(count, params)
            rows, spread = batch_screen(
                ds, lib, bank, centroids, radii, cidx, offsets, index, params,
                prepared, bvh, direct_sample_ms,
            )
            full_rows = full_validate(ds, lib, bank, centroids, radii, cidx, offsets, index, params, prepared, bvh)
            full_exact = all(row["decision_match_pct"] == 100.0 for row in full_rows)
            for row in full_rows:
                row.pop("_profile_arrays")
            speed_values = [r["screen_speedup"] for r in rows]
            mean_latency_values = [r["hnsw_query_latency"]["mean_ms"] for r in rows]
            rec = {
                "centroids": count,
                "parameters": params,
                "graph_cache_key": {"C": count, "degree": params["degree"], "ef_construction": params["ef_construction"]},
                "sample_all_q_exact": all(r["sample_match_pct"] == 100.0 for r in rows),
                "full_q_exact_match": full_exact,
                "full_validation_rows": full_rows,
                "cross_q_mean_latency_spread_pct": spread,
                "screen_q_mean_speedup_min_mean_max": [min(speed_values), statistics.mean(speed_values), max(speed_values)],
                "screen_cross_q_mean_latency_min_mean_max_ms": [min(mean_latency_values), statistics.mean(mean_latency_values), max(mean_latency_values)],
                "avg_within_q_latency_cv_pct": statistics.mean(r["hnsw_query_latency"]["cv_pct"] for r in rows),
                "avg_within_q_p95_ms": statistics.mean(r["hnsw_query_latency"]["p95_ms"] for r in rows),
                "screen_rows": rows,
            }
            records.append(rec)
            with screen_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            progress.update({"completed": number, "full_query_validated": number, "latest_C": count, "latest_parameters": params})
            atomic_json(out / "progress.json", progress)
            print(f"[{ds}] candidate {number}/{total} C={count} sample_exact={rec['sample_all_q_exact']} full_exact={full_exact} mean={statistics.mean(mean_latency_values):.5f}ms/q qspread={spread:.2f}%", flush=True)

    if no_selection:
        exact_count = sum(bool(record["full_q_exact_match"]) for record in records)
        atomic_json(out / "campaign_summary.json", {
            "dataset": ds,
            "known_bank_size": len(bank),
            "bank_sha256": bank_hash,
            "attack_manifest_sha256": manifest_hash,
            "attack_names": list(prepared[B.CFG[ds]["q"][0]]["slices"]),
            "q_sizes": B.CFG[ds]["q"],
            "candidate_design": coverage,
            "screen_candidates": total,
            "full_validated": len(records),
            "exact_full_q_candidates": exact_count,
            "selection_performed": False,
            "timing_protocol": "Each candidate gets the same fixed stratified panel; full-Q decisions are checked for every candidate. No per-C finalist retiming or global selection is performed.",
        })
        atomic_json(out / "progress.json", {
            "dataset": ds,
            "phase": "complete_no_selection",
            "screen_candidates": total,
            "full_validated": len(records),
            "exact_full_q": exact_count,
            "selection_performed": False,
        })
        print(f"[{ds}] COMPLETE candidates={total} full_validated={len(records)} exact_full_q={exact_count} selection=none", flush=True)
        return

    centroid_reference = float(np.sqrt(min(centroid_counts) * max(centroid_counts)))
    eligible = [r for r in records if r["full_q_exact_match"]]
    per_c_winners = []
    for count in centroid_counts:
        choices = [record for record in eligible if record["centroids"] == count]
        if not choices:
            continue

        def screen_score(record):
            q_means = [row["hnsw_query_latency"]["mean_ms"] for row in record["screen_rows"]]
            q_spread = (max(q_means) - min(q_means)) / max(statistics.mean(q_means), 1e-12) * 100.0
            sampled_max = max(row["hnsw_query_latency"]["max_ms"] for row in record["screen_rows"])
            sampled_p95 = max(row["hnsw_query_latency"]["p95_ms"] for row in record["screen_rows"])
            measured_max = max(row["full_workload_internal_query_latency"]["max_ms"] for row in record["full_validation_rows"])
            return (max(q_means), q_spread, sampled_max, sampled_p95, measured_max)

        per_c_winners.append(min(choices, key=screen_score))

    finalists = []
    for rec in per_c_winners:
        count, params = rec["centroids"], rec["parameters"]
        centroids, radii, cidx, offsets, index, bvh = structure_for(count, params)
        final = {"centroids": count, "parameters": params, "full_q_exact_match": True, "full_validation_rows": rec["full_validation_rows"]}
        if final["full_q_exact_match"]:
            q_rows, speed_stats, speed_spread, latency_spread = final_graph_timing(
                ds, lib, bank, centroids, radii, cidx, offsets, index, params, prepared, bvh, direct_sample_ms
            )
            final.update({
                "final_timing_rows": q_rows,
                "cross_q_q_mean_speedup_min_mean_max": speed_stats,
                "cross_q_q_mean_speedup_spread_pct": speed_spread,
                "cross_q_latency_mean_spread_pct": latency_spread,
                "avg_within_q_query_cv_pct": statistics.mean(r["full_workload_internal_query_latency"]["cv_pct"] for r in rec["full_validation_rows"]),
                "worst_q_internal_p95_ms": max(r["full_workload_internal_query_latency"]["p95_ms"] for r in rec["full_validation_rows"]),
                "timing_spread_is_reported_not_a_rejection_gate": True,
            })
        finalists.append(final)
        atomic_json(out / "finalist_records.json", {"coverage": coverage, "screen_candidate_count": total, "finalists": finalists})
        print(f"[{ds}] robust-final-timing C={count} params={params}", flush=True)

    valid = [r for r in finalists if r.get("full_q_exact_match")]
    mid_candidates = [
        r for r in valid
        if centroid_reference / 2.0 <= r["centroids"] <= centroid_reference * 2.0
    ]
    selected_mid = _balanced_selection(mid_candidates, centroid_reference)
    selected_global = _balanced_selection(valid, centroid_reference)
    selected = selected_mid or selected_global
    selected_graph_file = None
    if selected is not None:
        p = selected["parameters"]
        centroids, _, _, _, selected_index, _ = structure_for(selected["centroids"], p)
        selected_graph_file = out / "final_hnsw.npz"
        B.save_hnsw(selected_index, selected_graph_file)
    result = {
        "dataset": ds,
        "known_bank_size": len(bank),
        "bank_sha256": bank_hash,
        "attack_manifest_sha256": manifest_hash,
        "attack_manifest": str(Path(manifest_env).resolve()),
        "attack_names": list(prepared[B.CFG[ds]["q"][0]]["slices"]),
        "q_sizes": B.CFG[ds]["q"],
        "candidate_design": coverage,
        "screen_candidates": total,
        "screening_rule": "Every strength-2 pairwise-covering candidate is individually timed on the fixed stratified panel and checked against direct decisions over every full-Q workload. No global top-N shortlist is used. One per-C winner is then re-timed with the robust final protocol.",
        "final_acceptance": "100% direct agreement on every full-Q workload. Prefer exact candidates within one octave of the geometric midpoint of the dataset centroid range; report the global exact candidate separately.",
        "direct_timing_protocol": "One fixed stratified panel per Q; each direct query is warmed once and timed 9 times, then its trimmed median is reused as the same direct baseline for every candidate. Final graph timing uses 8 warmups and 41 repetitions per panel query.",
        "centroid_reference_geometric_midpoint": centroid_reference,
        "preferred_centroid_band": [centroid_reference / 2.0, centroid_reference * 2.0],
        "selection_objectives_in_order": ["minimize the slowest Q-size mean latency", "minimize cross-Q spread of mean per-query latency", "minimize the sampled per-query maximum", "minimize worst-Q p95", "minimize cross-Q mean latency", "prefer C near the geometric midpoint of the dataset centroid range"],
        "selected": selected,
        "selected_interior_candidate": selected_mid,
        "selected_global_balanced_candidate": selected_global,
        "selected_graph_file": str(selected_graph_file) if selected_graph_file else None,
        "best_per_centroid": finalists,
    }
    atomic_json(out / "selected_iterable.json", result)
    atomic_json(out / "progress.json", {"dataset": ds, "phase": "complete", "screen_candidates": total, "full_validated": len(records), "exact_full_q": len(eligible), "robust_final_timed_per_centroid": len(finalists)})
    print(f"[{ds}] COMPLETE candidates={total} full_validated={len(records)} exact_full_q={len(eligible)} robust_final_timed={len(finalists)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASET_BASELINES), default="ucid")
    parser.add_argument("--no-selection", action="store_true", help="save every individually evaluated candidate without finalist retiming or global selection")
    args = parser.parse_args()
    run(args.dataset, no_selection=args.no_selection)
