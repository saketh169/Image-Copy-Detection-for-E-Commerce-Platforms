#!/usr/bin/env python3
"""Gate Amazon HNSW sweeps on exactness *and* meaningful certificate pruning.

This is intentionally a preflight, not a replacement for the final all-Q sweep.
It re-evaluates representative per-centroid configurations after a certificate
change. A configuration proceeds to full-Q validation only if it is exact on
the fixed panel and checks at most a configured fraction of the known bank.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
from pathlib import Path

import numpy as np

# The reusable sweep module validates this variable when imported. The
# preflight creates its own explicit output directory and never invokes run().
os.environ.setdefault("HNSW_SWEEP_OUT", ".amazon_certificate_preflight_import")


def _load(name: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


B = _load("graph_benchmark") or _load("benchmark_graphs")
S = _load("hnsw_full_q_sweep") or _load("hnsw_ucid_iterable_sweep_no_select")
H = _load("hnsw_hierarchical")
if B is None or S is None or H is None:
    raise RuntimeError("Run beside the Phase-3 engine and HNSW sweep sources.")


REPRESENTATIVE_C = (384, 512, 768, 1024, 1536, 2048)


def _prior_winner(rows: list[dict], centroid_count: int) -> dict:
    choices = [r for r in rows if r["centroids"] == centroid_count and r["full_q_exact_match"]]
    if not choices:
        raise RuntimeError(f"No exact prior candidate at C={centroid_count}")

    def score(record: dict) -> tuple[float, float, float, float]:
        profile_max = max(
            q["full_workload_internal_query_latency"]["max_ms"]
            for q in record["full_validation_rows"]
        )
        panel_max = max(q["hnsw_query_latency"]["max_ms"] for q in record["screen_rows"])
        panel_mean = max(q["hnsw_query_latency"]["mean_ms"] for q in record["screen_rows"])
        return profile_max, panel_max, panel_mean, record["cross_q_mean_latency_spread_pct"]

    return min(choices, key=score)


def _direct_panel(lib, bank: np.ndarray, prepared: dict[int, dict]) -> dict[int, list[float]]:
    values_by_q: dict[int, list[float]] = {}
    for q, info in prepared.items():
        values = []
        for query in info["sample_queries"]:
            one = np.ascontiguousarray(query[None, :])
            B.direct_call(lib, one, bank, info["threshold2"])
            for _ in range(S.DIRECT_WARMUPS):
                B.direct_call(lib, one, bank, info["threshold2"])
            samples = []
            for _ in range(S.DIRECT_REPEATS):
                started = S.time.perf_counter()
                B.direct_call(lib, one, bank, info["threshold2"])
                samples.append(S.time.perf_counter() - started)
            values.append(S.trimmed_median(samples) * 1000.0 if hasattr(S, "trimmed_median") else S.median_trim(samples) * 1000.0)
        values_by_q[q] = values
    return values_by_q


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=None,
        help="Fingerprint-validated centroid, posting, and radius cache directory to reuse.",
    )
    parser.add_argument("--max-check-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if not 0.0 < args.max_check_fraction <= 1.0:
        raise ValueError("--max-check-fraction must be in (0, 1]")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"Output must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    structures_dir = args.structures_dir or (args.output / "structures")

    old_records = [json.loads(line) for line in args.prior_records.read_text().splitlines()]
    lib = B.compile_library()
    pool, bank = B.load_pool("amazon")
    prepared: dict[int, dict] = {}
    for q in B.CFG["amazon"]["q"]:
        info = B.prepare_direct(lib, pool, "amazon", bank, q, measure_timing=False)
        if len(info["slices"]) != 19:
            raise RuntimeError("Expected the current 19-attack manifest")
        indices = S.sample_indices(info["slices"], q, 20260927 + q)
        info["sample_indices"] = indices
        info["sample_queries"] = np.ascontiguousarray(info["queries"][indices])
        prepared[q] = info
    direct_ms = _direct_panel(lib, bank, prepared)
    max_checks = len(bank) * args.max_check_fraction
    results = []

    for count in REPRESENTATIVE_C:
        prior = _prior_winner(old_records, count)
        params = prior["parameters"]
        centroids, radii, cidx, offsets = B.build_layout(
            "amazon", bank, count, params["overlap"], structures_dir
        )
        index = H.build_hnsw(
            centroids,
            m=max(2, params["degree"] // 2),
            max_degree_base=params["degree"],
            ef_construction=params["ef_construction"],
            seed=20260915 + count,
            entry_count=S.ENTRY_CAPACITY,
        )
        bvh = B.build_bucket_bvh(centroids, radii)
        panel_rows, q_spread = S.batch_screen(
            "amazon", lib, bank, centroids, radii, cidx, offsets, index,
            params, prepared, bvh, direct_ms,
        )
        panel_exact = all(row["sample_match_pct"] == 100.0 for row in panel_rows)
        worst_checks = max(row["avg_images_checked"] for row in panel_rows)
        passes_pruning = worst_checks <= max_checks
        result = {
            "centroids": count,
            "parameters": params,
            "panel_exact": panel_exact,
            "worst_panel_average_images_checked": worst_checks,
            "max_allowed_average_images_checked": max_checks,
            "passes_pruning_gate": passes_pruning,
            "cross_q_mean_latency_spread_pct": q_spread,
            "panel_rows": panel_rows,
        }
        if panel_exact and passes_pruning:
            full_rows = S.full_validate(
                "amazon", lib, bank, centroids, radii, cidx, offsets, index,
                params, prepared, bvh,
            )
            result["full_q_exact"] = all(row["decision_match_pct"] == 100.0 for row in full_rows)
            result["full_validation_rows"] = full_rows
        else:
            result["full_q_exact"] = None
            result["full_validation_skipped_reason"] = "panel exactness or pruning gate failed"
        results.append(result)
        (args.output / "preflight.json").write_text(json.dumps({
            "bank_size": len(bank),
            "max_check_fraction": args.max_check_fraction,
            "representative_centroids": list(REPRESENTATIVE_C),
            "results": results,
        }, indent=2))
        print(
            f"C={count} panel_exact={panel_exact} avg_checks={worst_checks:.1f}/"
            f"{len(bank)} full_q_exact={result['full_q_exact']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
