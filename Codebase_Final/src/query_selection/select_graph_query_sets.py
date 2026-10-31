#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/saketh/Codebase")
import phase3_fast_centroid_multiprobe_20260721 as M  # noqa: E402
import phase3_pruned_hnsw_erng_20260721 as P  # noqa: E402
import phase3_query_entry_hnsw_erng_20260721 as Q  # noqa: E402


DEFAULT_ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
SRC = Path(os.environ.get("PHASE3_GRAPH_SRC", str(DEFAULT_ROOT / "phase3_turbo_centroid_sweep_hnsw_erng_20260721")))
OUT = Path(os.environ.get("PHASE3_GRAPH_OUT", str(SRC / "allq_90to100_reselected")))


def final_key(pack: dict) -> tuple:
    min_speed = pack["min_speed"]
    avg_speed = pack["avg_speed"]
    speed_cv = pack.get("speed_cv", 999.0)
    time_cv = pack.get("time_cv", 999.0)
    speed_spread = pack.get("speed_spread", 999.0)
    max_ms = pack.get("max_ms", 999.0)
    uniform = (
        time_cv <= 0.20
        and speed_cv <= 0.20
        and speed_spread <= 1.75
    )
    near_uniform = (
        time_cv <= 0.30
        and speed_cv <= 0.30
        and speed_spread <= 2.00
    )
    stable_speed_score = min_speed / max(speed_spread, 1.0)
    return (
        pack["min_match"] >= 90.0,
        min_speed >= 5.0,
        uniform,
        near_uniform,
        min_speed >= 10.0,
        -speed_spread,
        -speed_cv,
        -time_cv,
        stable_speed_score,
        min_speed,
        -max_ms,
        pack["min_match"],
        avg_speed,
        pack["avg_match"],
        -pack["avg_ms"],
    )


def match_first_key(pack: dict) -> tuple:
    return (
        pack["min_match"],
        pack["avg_match"],
        -pack.get("speed_spread", 999.0),
        -pack.get("time_cv", 999.0),
        -pack.get("speed_cv", 999.0),
        pack["min_speed"],
        pack["avg_speed"],
        -pack["avg_ms"],
    )


def fallback_key(rec: dict) -> tuple:
    return (
        rec.get("min_decision_match", 0.0),
        rec.get("avg_decision_match", 0.0),
        -rec.get("avg_ms_per_query", 999.0),
    )


def run(ds: str, top: int, threshold: float, only_kind: str | None = None) -> None:
    cfg = M.CFG[ds]
    src = SRC / ds / f"{ds}_query_entry_hnsw_erng.json"
    if not src.exists():
        raise SystemExit(f"missing completed sweep json: {src}")

    out = OUT / ds
    out.mkdir(parents=True, exist_ok=True)

    lib = Q.load_lib()
    npz = np.load(cfg["pool"], allow_pickle=True)
    pool = {k: npz[k] for k in npz.files}
    bank = np.ascontiguousarray(pool["bank"].astype(np.float32))

    direct, truth, mats = {}, {}, {}
    for q in cfg["q"]:
        print("[direct]", ds, q, flush=True)
        tau, rows, gt, mm, summary = M.direct_q(M.load_lib(), bank, pool, ds, q)
        direct[str(q)] = {"tau": tau, "attacks": rows, "summary": summary}
        truth[str(q)] = gt
        mats[str(q)] = mm

    data = json.loads(src.read_text())
    selected, results = {}, {}
    candidate_dump = {}

    def per_q_summary(qpack: dict) -> dict:
        return {
            str(q): {
                "speedup": qpack[str(q)]["speedup"],
                "direct_ms": direct[str(q)]["summary"]["avg_ms_per_query"],
                "graph_ms": qpack[str(q)]["summary"]["avg_ms_per_query"],
                "min_match": qpack[str(q)]["summary"].get("min_decision_match"),
                "avg_match": qpack[str(q)]["summary"].get("avg_decision_match"),
            }
            for q in cfg["q"]
        }

    kinds = [only_kind] if only_kind else ["erng", "hnsw"]
    for kind in kinds:
        rows = [r for r in data["sweep"] if r.get("kind") == kind]
        pool_rows = [r for r in rows if r.get("min_decision_match", 0.0) >= threshold]
        if not pool_rows:
            pool_rows = sorted(rows, key=fallback_key, reverse=True)
        pool_rows = sorted(pool_rows, key=lambda r: r.get("avg_ms_per_query", 999.0))[:top]

        best_speed = None
        best_speed_q = None
        best_match = None
        best_match_q = None
        all_packs = []

        for i, rec in enumerate(pool_rows, 1):
            print("[cand]", ds, kind, i, rec, flush=True)
            centroids, cidx, offsets, _ = M.build_centroids(
                ds, bank, SRC / ds, int(rec["centroids"]), int(rec["overlap"])
            )
            radii = P.radii_for(
                bank, centroids, cidx, offsets, SRC / ds, ds, int(rec["centroids"]), int(rec["overlap"])
            )
            adj, _ = P.build_adj(ds, kind, centroids, int(rec["degree"]), SRC / ds)

            qpack = {}
            speeds, matches, avg_matches, times = [], [], [], []
            for q in cfg["q"]:
                rows2, summary = Q.query_entry_eval(
                    lib,
                    bank,
                    q,
                    direct[str(q)]["tau"],
                    truth[str(q)],
                    mats[str(q)],
                    centroids,
                    radii,
                    cidx,
                    offsets,
                    adj,
                    int(rec["entries"]),
                    int(rec["ef"]),
                )
                direct_ms = direct[str(q)]["summary"]["avg_ms_per_query"]
                speedup = direct_ms / max(summary["avg_ms_per_query"], 1e-12)
                speeds.append(speedup)
                matches.append(summary["min_decision_match"])
                avg_matches.append(summary.get("avg_decision_match", summary["min_decision_match"]))
                times.append(summary["avg_ms_per_query"])
                qpack[str(q)] = {"attacks": rows2, "summary": summary, "speedup": speedup}

            pack = {
                "params": rec,
                "q_results": qpack,
                "min_match": min(matches),
                "avg_match": float(np.mean(avg_matches)),
                "min_speed": min(speeds),
                "avg_speed": float(np.mean(speeds)),
                "max_speed": max(speeds),
                "speed_cv": float(statistics.pstdev(speeds) / np.mean(speeds)) if np.mean(speeds) > 0 else 999.0,
                "speed_spread": float(max(speeds) / max(min(speeds), 1e-12)),
                "time_cv": float(statistics.pstdev(times) / np.mean(times)) if np.mean(times) > 0 else 999.0,
                "time_spread": float(max(times) / max(min(times), 1e-12)),
                "avg_ms": float(np.mean(times)),
                "max_ms": float(max(times)),
            }
            all_packs.append(pack)
            print(
                "[allq]",
                ds,
                kind,
                pack["min_match"],
                pack["avg_match"],
                pack["min_speed"],
                pack["avg_speed"],
                pack["speed_cv"],
                pack["time_cv"],
                pack["avg_ms"],
                flush=True,
            )
            if best_speed is None or final_key(pack) > final_key(best_speed):
                best_speed = pack
                best_speed_q = qpack
                perq = per_q_summary(qpack)
                progress = {
                    "dataset": ds,
                    "kind": kind,
                    "selection": "speed_first_90to100",
                    "pack": best_speed,
                    "per_q": perq,
                }
                (out / f"{ds}_{kind}_progress_speed_first.json").write_text(json.dumps(progress, indent=2))
                print(
                    "[best-speed]",
                    ds,
                    kind,
                    json.dumps(
                        {
                            "min_match": pack["min_match"],
                            "avg_match": pack["avg_match"],
                            "min_speed": pack["min_speed"],
                            "avg_speed": pack["avg_speed"],
                            "speed_cv": pack["speed_cv"],
                            "time_cv": pack["time_cv"],
                            "speed_spread": pack["speed_spread"],
                            "time_spread": pack["time_spread"],
                            "per_q": perq,
                        }
                    ),
                    flush=True,
                )
            if best_match is None or match_first_key(pack) > match_first_key(best_match):
                best_match = pack
                best_match_q = qpack
                perq = per_q_summary(qpack)
                progress = {
                    "dataset": ds,
                    "kind": kind,
                    "selection": "match_first",
                    "pack": best_match,
                    "per_q": perq,
                }
                (out / f"{ds}_{kind}_progress_match_first.json").write_text(json.dumps(progress, indent=2))
                print(
                    "[best-match]",
                    ds,
                    kind,
                    json.dumps(
                        {
                            "min_match": pack["min_match"],
                            "avg_match": pack["avg_match"],
                            "min_speed": pack["min_speed"],
                            "avg_speed": pack["avg_speed"],
                            "speed_cv": pack["speed_cv"],
                            "time_cv": pack["time_cv"],
                            "speed_spread": pack["speed_spread"],
                            "time_spread": pack["time_spread"],
                            "per_q": perq,
                        }
                    ),
                    flush=True,
                )

        candidate_dump[kind] = sorted(all_packs, key=final_key, reverse=True)
        selected[kind] = best_speed
        results[kind] = best_speed_q
        selected[f"{kind}_match_first"] = best_match
        results[f"{kind}_match_first"] = best_match_q

        (out / f"{ds}_{kind}_allq_90to100_candidates.json").write_text(
            json.dumps(candidate_dump[kind], indent=2)
        )

    payload = {
        "dataset": ds,
        "source": str(src),
        "selection_rule": "generalized all-q; candidates from q-cal >= threshold; selected by match band + stable q-to-q avg time/speed + speedup; match-first audit saved",
        "stability_rule": "same graph params across all q sizes; require acceptable speed floor first, then uniform q-to-q speed/time, then raise common speed floor and minimize worst q graph time",
        "threshold": threshold,
        "selected": selected,
        "direct": direct,
        "results": results,
    }
    path = out / f"{ds}_allq_90to100_reselected.json"
    path.write_text(json.dumps(payload, indent=2))
    print("[done]", path, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(M.CFG))
    ap.add_argument("--top", type=int, default=999999)
    ap.add_argument("--threshold", type=float, default=90.0)
    ap.add_argument("--kind", choices=["hnsw", "erng"], default=None)
    args = ap.parse_args()
    run(args.dataset, args.top, args.threshold, args.kind)
