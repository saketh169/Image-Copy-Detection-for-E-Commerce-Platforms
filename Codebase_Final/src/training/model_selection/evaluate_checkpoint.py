import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE  # noqa: E402
from dataloader import get_transform  # noqa: E402
from models.model_convnext import SiameseConvNeXtB  # noqa: E402


def load_model(ckpt_path: str):
    model = SiameseConvNeXtB().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(path: str, transform):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)


def encode_rows(model, rows, transform, batch_size: int):
    tensors = []
    kept = []
    for row in rows:
        t = preprocess(row["path"], transform)
        if t is not None:
            tensors.append(t)
            kept.append(row)
    if not tensors:
        return kept, torch.empty((0, 384), dtype=torch.float32)
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(DEVICE)
            emb, _, _, _, _, _ = model(batch, batch)
            all_embs.append(emb.detach().cpu())
    return kept, torch.cat(all_embs, dim=0)


def min_dists(query_embs: torch.Tensor, bank_embs: torch.Tensor, chunk: int = 256):
    outs = []
    for i in range(0, query_embs.shape[0], chunk):
        q = query_embs[i : i + chunk]
        d = torch.cdist(q, bank_embs)
        outs.append(d.min(dim=1).values)
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--db-pool", required=True)
    ap.add_argument("--db-fraction", type=float, default=1 / 3)
    ap.add_argument("--eval-max-known", type=int, default=1600)
    ap.add_argument("--eval-max-unknown", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260415)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold-mode", choices=["search", "midpoint"], default="search")
    ap.add_argument("--midpoint-buffer-ratio", type=float, default=0.25)
    args = ap.parse_args()

    pool = json.loads(Path(args.db_pool).read_text(encoding="utf-8"))
    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    random.seed(args.seed + len(pool))
    random.shuffle(known)
    random.shuffle(unknown)
    split_idx = int(len(known) * args.db_fraction)
    db_items = known[:split_idx]
    known_queries = known[split_idx:][: args.eval_max_known]
    unknown_queries = unknown[: args.eval_max_unknown]

    model = load_model(args.ckpt)
    transform = get_transform()
    _, bank = encode_rows(model, db_items, transform, args.batch_size)
    known_rows, known_embs = encode_rows(model, known_queries, transform, args.batch_size)
    unknown_rows, unknown_embs = encode_rows(model, unknown_queries, transform, args.batch_size)

    known_d = min_dists(known_embs, bank)
    unknown_d = min_dists(unknown_embs, bank)

    known_avg = known_d.mean().item()
    unknown_avg = unknown_d.mean().item()

    if args.threshold_mode == "midpoint":
        gap = unknown_avg - known_avg
        buffer = max(0.0, gap * args.midpoint_buffer_ratio)
        dup_thr = known_avg + buffer
        not_thr = unknown_avg - buffer
        if not_thr < dup_thr:
            mid = (known_avg + unknown_avg) / 2.0
            dup_thr = mid
            not_thr = mid
        known_correct = int((known_d <= dup_thr).sum().item())
        unknown_correct = int((unknown_d >= not_thr).sum().item())
        print(
            f"MIDPOINT known {known_correct}/{len(known_rows)} | "
            f"unknown {unknown_correct}/{len(unknown_rows)} | "
            f"dup_thr {dup_thr:.2f} | not_thr {not_thr:.2f} | "
            f"known_avg {known_avg:.2f} | unknown_avg {unknown_avg:.2f}"
        )
        return

    candidates = torch.unique(torch.cat([known_d, unknown_d])).tolist()
    candidates = candidates[:: max(1, len(candidates) // 100)] + [min(candidates), max(candidates)]
    candidates = sorted(set(float(x) for x in candidates))

    best_combined = None
    best_known = None
    best_unknown = None
    for dup_thr in candidates:
        known_correct = int((known_d <= dup_thr).sum().item())
        if best_known is None or known_correct > best_known["known_correct"]:
            best_known = {"dup_thr": dup_thr, "known_correct": known_correct}
        for not_thr in candidates:
            unknown_correct = int((unknown_d >= not_thr).sum().item())
            combined = known_correct + unknown_correct
            balanced = min(
                known_correct / max(1, len(known_rows)),
                unknown_correct / max(1, len(unknown_rows)),
            )
            candidate = {
                "dup_thr": dup_thr,
                "not_thr": not_thr,
                "known_correct": known_correct,
                "unknown_correct": unknown_correct,
                "combined": combined,
                "balanced": balanced,
                "known_avg": known_avg,
                "unknown_avg": unknown_avg,
            }
            if best_combined is None or (candidate["combined"], candidate["balanced"]) > (
                best_combined["combined"],
                best_combined["balanced"],
            ):
                best_combined = candidate
            if best_unknown is None or unknown_correct > best_unknown["unknown_correct"]:
                best_unknown = {"dup_thr": dup_thr, "not_thr": not_thr, "unknown_correct": unknown_correct}

    print(
        f"BEST_COMBINED known {best_combined['known_correct']}/{len(known_rows)} | "
        f"unknown {best_combined['unknown_correct']}/{len(unknown_rows)} | "
        f"dup_thr {best_combined['dup_thr']:.2f} | not_thr {best_combined['not_thr']:.2f} | "
        f"known_avg {best_combined['known_avg']:.2f} | unknown_avg {best_combined['unknown_avg']:.2f}"
    )
    print(f"BEST_KNOWN known {best_known['known_correct']}/{len(known_rows)} | dup_thr {best_known['dup_thr']:.2f}")
    print(
        f"BEST_UNKNOWN unknown {best_unknown['unknown_correct']}/{len(unknown_rows)} | "
        f"dup_thr {best_unknown['dup_thr']:.2f} | not_thr {best_unknown['not_thr']:.2f}"
    )


if __name__ == "__main__":
    main()
