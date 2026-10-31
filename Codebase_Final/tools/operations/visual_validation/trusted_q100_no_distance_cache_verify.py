#!/usr/bin/env python3
"""
Verify the trusted q100 selected sets without reading cached distance rows.

This script uses the same selected q JSONs and the same final pool/store
embedding files that produced the trusted distance-cache results. It does NOT
read kd/ud from the distance cache. Instead it:

1. Loads the trusted pool/store embeddings.
2. Uses the selected q indices from the final q JSON.
3. Checks original known query embeddings equal the same bank embeddings
   selected by index. This is the distance-zero/cache-alignment sanity check.
4. Recomputes nearest-bank distances from query embeddings to the complete
   known-bank store embeddings.
5. Reports accuracy with the same tau threshold.

It is intentionally narrow: q100, trusted selected sets, six concern attacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
TRUSTED = ROOT / "q100_six_attacks_trusted_from_dist_cache_20260726/trusted_six_attacks_allq_from_dist_cache.json"
OUT = ROOT / "trusted_q100_no_distance_cache_verify_20260727"

POOL_FILES = {
    "cifar": ROOT / "phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3.npz",
    "flickr": ROOT / "phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2.npz",
    "ucid": ROOT / "phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850.npz",
    "amazon": ROOT / "phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000.npz",
}

ATTACK_ROWS = {
    # Full pool query arrays store all 26 attacks in M.G.ATTACKS order.
    # The trusted distance cache stores only the filtered 20 attacks, so its
    # dist_row numbers are not valid offsets into known_query/unknown_query.
    "flip_h": 14,
    "flip": 16,
    "resize_compress": 17,
    "resize": 18,
    "watermark_text": 19,
    "watermark_asset": 20,
}


def selected_indices(qdoc: dict[str, Any], side: str) -> list[int]:
    keys = {
        "known": ["known_indices", "selected_known_indices"],
        "unknown": ["unknown_indices", "selected_unknown_indices"],
    }[side]
    for key in keys:
        if key in qdoc:
            return [int(x) for x in qdoc[key]]
    raise KeyError(f"missing {side} selected indices in q json")


def nearest_distances(bank: np.ndarray, query: np.ndarray, device: str, qchunk: int = 64, bchunk: int = 4096) -> np.ndarray:
    bank_t = torch.from_numpy(np.ascontiguousarray(bank)).float().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(query), qchunk):
            q = torch.from_numpy(np.ascontiguousarray(query[i : i + qchunk])).float().to(device)
            best = torch.full((q.shape[0],), float("inf"), device=device)
            for j in range(0, len(bank), bchunk):
                b = bank_t[j : j + bchunk]
                d2 = ((q[:, None, :] - b[None, :, :]) ** 2).sum(2).min(1).values
                best = torch.minimum(best, d2)
            out.append(torch.sqrt(torch.clamp(best, min=0)).cpu().numpy())
    return np.concatenate(out).astype("float32")


def accuracy(kd: np.ndarray, ud: np.ndarray, tau: float) -> tuple[float, float, float]:
    known = float((kd <= tau).mean() * 100.0)
    unknown = float((ud > tau).mean() * 100.0)
    return known, unknown, (known + unknown) / 2.0


def eval_dataset(ds: str, device: str) -> dict[str, Any]:
    trusted = json.loads(TRUSTED.read_text())
    q100 = trusted[ds]["q"]["100"]
    q_json = Path(q100["q_json"])
    qdoc = json.loads(q_json.read_text())
    tau = float(q100["tau"])
    known_idx = selected_indices(qdoc, "known")
    unknown_idx = selected_indices(qdoc, "unknown")

    pool_path = POOL_FILES[ds]
    pool = np.load(pool_path, allow_pickle=False)
    bank = pool["bank"].astype("float32", copy=False)
    known_query = pool["known_query"].astype("float32", copy=False)
    unknown_query = pool["unknown_query"].astype("float32", copy=False)
    known_q = int(pool["known_q"]) if "known_q" in pool.files else int(len(known_query) // 26)
    unknown_q = int(pool["unknown_q"]) if "unknown_q" in pool.files else int(pool["q"])

    # Original known attack row is 0. For known originals, selected query
    # embeddings should be identical to the same selected bank rows.
    orig_known_q = known_query[np.array(known_idx, dtype=np.int64)]
    orig_bank = bank[np.array(known_idx, dtype=np.int64)]
    max_abs_query_bank_diff = float(np.max(np.abs(orig_known_q - orig_bank))) if len(known_idx) else 0.0
    original_zero_ok = bool(max_abs_query_bank_diff <= 1e-6)

    rows = []
    for attack, row in ATTACK_ROWS.items():
        k_start = row * known_q
        u_start = row * unknown_q
        kq = known_query[k_start + np.array(known_idx, dtype=np.int64)]
        uq = unknown_query[u_start + np.array(unknown_idx, dtype=np.int64)]
        kd = nearest_distances(bank, kq, device)
        ud = nearest_distances(bank, uq, device)
        known, unknown, total = accuracy(kd, ud, tau)

        trusted_row = next((r for r in q100["rows"] if r["attack"] == attack), None)
        rows.append(
            {
                "attack": attack,
                "known": round(known, 6),
                "unknown": round(unknown, 6),
                "total": round(total, 6),
                "trusted_known": None if trusted_row is None else trusted_row["known"],
                "trusted_unknown": None if trusted_row is None else trusted_row["unknown"],
                "trusted_total": None if trusted_row is None else trusted_row["total"],
                "matches_trusted": bool(
                    trusted_row
                    and abs(known - float(trusted_row["known"])) < 1e-5
                    and abs(unknown - float(trusted_row["unknown"])) < 1e-5
                    and abs(total - float(trusted_row["total"])) < 1e-5
                ),
                "known_min_dist": float(kd.min()),
                "known_max_dist": float(kd.max()),
                "unknown_min_dist": float(ud.min()),
                "unknown_max_dist": float(ud.max()),
            }
        )

    return {
        "dataset": ds,
        "pool_store": str(pool_path),
        "q_json": str(q_json),
        "tau": tau,
        "known_selected": len(known_idx),
        "unknown_selected": len(unknown_idx),
        "bank_size": int(len(bank)),
        "known_pool_size": int(known_q),
        "unknown_pool_size": int(unknown_q),
        "original_known_query_equals_bank_max_abs_diff": max_abs_query_bank_diff,
        "original_zero_alignment_ok": original_zero_ok,
        "distance_cache_used_for_accuracy": False,
        "rows": rows,
        "all_rows_match_trusted": all(r["matches_trusted"] for r in rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "cifar", "ucid", "flickr", "amazon"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    datasets = ["cifar", "ucid", "flickr", "amazon"] if args.dataset == "all" else [args.dataset]
    combined = {}
    for ds in datasets:
        print(f"[trusted-no-dist-cache] {ds}", flush=True)
        res = eval_dataset(ds, args.device)
        combined[ds] = res
        (OUT / f"{ds}_trusted_q100_no_distance_cache_verify.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(
            f"{ds}: alignment={res['original_zero_alignment_ok']} "
            f"max_abs={res['original_known_query_equals_bank_max_abs_diff']:.3g} "
            f"rows_match_trusted={res['all_rows_match_trusted']}",
            flush=True,
        )
        for row in res["rows"]:
            print(
                "  {attack}: {known:.2f}/{unknown:.2f}/{total:.2f} "
                "trusted={trusted_known:.2f}/{trusted_unknown:.2f}/{trusted_total:.2f} "
                "match={matches_trusted}".format(**row),
                flush=True,
            )

    (OUT / "combined_trusted_q100_no_distance_cache_verify.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"saved {OUT / 'combined_trusted_q100_no_distance_cache_verify.json'}", flush=True)


if __name__ == "__main__":
    main()
