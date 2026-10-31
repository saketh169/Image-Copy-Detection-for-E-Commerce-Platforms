#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
TRUSTED = ROOT / "q100_six_attacks_trusted_from_dist_cache_20260726/trusted_six_attacks_allq_from_dist_cache.json"
OUT = ROOT / "trusted_q100_original_only_verify_20260727"

POOL_FILES = {
    "cifar": ROOT / "phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3.npz",
    "flickr": ROOT / "phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2.npz",
    "ucid": ROOT / "phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850.npz",
    "amazon": ROOT / "phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000.npz",
}


def selected_indices(qdoc: dict[str, Any], side: str) -> list[int]:
    keys = {
        "known": ["known_indices", "selected_known_indices"],
        "unknown": ["unknown_indices", "selected_unknown_indices"],
    }[side]
    for key in keys:
        if key in qdoc:
            return [int(x) for x in qdoc[key]]
    raise KeyError(f"missing {side} selected indices")


def nearest_distances(bank: np.ndarray, query: np.ndarray, device: str) -> np.ndarray:
    bank_t = torch.from_numpy(np.ascontiguousarray(bank)).float().to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(query), 64):
            q = torch.from_numpy(np.ascontiguousarray(query[i : i + 64])).float().to(device)
            best = torch.full((q.shape[0],), float("inf"), device=device)
            for j in range(0, len(bank), 4096):
                b = bank_t[j : j + 4096]
                d2 = ((q[:, None, :] - b[None, :, :]) ** 2).sum(2).min(1).values
                best = torch.minimum(best, d2)
            outs.append(torch.sqrt(torch.clamp(best, min=0)).cpu().numpy())
    return np.concatenate(outs).astype("float32")


def eval_dataset(ds: str, device: str) -> dict[str, Any]:
    trusted = json.loads(TRUSTED.read_text())
    q100 = trusted[ds]["q"]["100"]
    qdoc = json.loads(Path(q100["q_json"]).read_text())
    tau = float(q100["tau"])
    known_idx = np.array(selected_indices(qdoc, "known"), dtype=np.int64)
    unknown_idx = np.array(selected_indices(qdoc, "unknown"), dtype=np.int64)

    pool_path = POOL_FILES[ds]
    pool = np.load(pool_path, allow_pickle=False)
    bank = pool["bank"].astype("float32", copy=False)
    known_query = pool["known_query"].astype("float32", copy=False)
    unknown_query = pool["unknown_query"].astype("float32", copy=False)

    # Original attack is row 0 in the pool arrays.
    known_q = known_query[known_idx]
    unknown_q = unknown_query[unknown_idx]

    max_abs_query_bank_diff = float(np.max(np.abs(known_q - bank[known_idx]))) if len(known_idx) else 0.0
    kd = nearest_distances(bank, known_q, device)
    ud = nearest_distances(bank, unknown_q, device)
    known_acc = float((kd <= tau).mean() * 100.0)
    unknown_acc = float((ud > tau).mean() * 100.0)
    total = (known_acc + unknown_acc) / 2.0

    return {
        "dataset": ds,
        "pool_store": str(pool_path),
        "q_json": q100["q_json"],
        "tau": tau,
        "bank_size": int(len(bank)),
        "known_selected": int(len(known_idx)),
        "unknown_selected": int(len(unknown_idx)),
        "original_known_query_equals_bank_max_abs_diff": max_abs_query_bank_diff,
        "original_zero_alignment_ok": bool(max_abs_query_bank_diff <= 1e-6),
        "known_accuracy": round(known_acc, 6),
        "unknown_accuracy": round(unknown_acc, 6),
        "total_accuracy": round(total, 6),
        "known_min_dist": float(kd.min()),
        "known_max_dist": float(kd.max()),
        "unknown_min_dist": float(ud.min()),
        "unknown_max_dist": float(ud.max()),
        "distance_cache_used": False,
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
        res = eval_dataset(ds, args.device)
        combined[ds] = res
        (OUT / f"{ds}_trusted_q100_original_only_verify.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(
            f"{ds}: known={res['known_accuracy']:.2f} unknown={res['unknown_accuracy']:.2f} "
            f"total={res['total_accuracy']:.2f} alignment={res['original_zero_alignment_ok']} "
            f"max_abs={res['original_known_query_equals_bank_max_abs_diff']:.3g}",
            flush=True,
        )
    (OUT / "combined_trusted_q100_original_only_verify.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
