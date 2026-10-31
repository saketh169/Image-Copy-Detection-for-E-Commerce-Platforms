import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/home/saketh/Codebase")

from dataloader import get_transform  # noqa: E402
from models.model_convnext import SiameseConvNeXtB  # noqa: E402


def load_model(ckpt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseConvNeXtB().to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
    model.eval()
    return model, device


def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def augment(img, mode):
    h, w = img.shape[:2]
    if mode == "bright":
        return np.clip(img.astype(np.float32) * 1.18 + 8.0, 0, 255).astype(np.uint8)
    if mode == "contrast":
        return np.clip((img.astype(np.float32) - 127.5) * 1.18 + 127.5, 0, 255).astype(np.uint8)
    if mode == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    if mode == "rotate5":
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if mode == "rotate15":
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if mode == "rotate-15":
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), -15, 1.0)
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if mode == "flip":
        return cv2.flip(img, 1)
    if mode == "crop10":
        y1, x1 = int(h * 0.10), int(w * 0.10)
        y2, x2 = int(h * 0.90), int(w * 0.90)
        return cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)
    if mode == "heavy_bright":
        return np.clip(img.astype(np.float32) * 1.35 + 18.0, 0, 255).astype(np.uint8)
    return img


def encode(model, device, rows, transform, batch_size, aug_cycle):
    tensors = []
    kept = []
    for i, row in enumerate(rows):
        img = load_rgb(row["path"])
        if img is None:
            continue
        mode = aug_cycle[i % len(aug_cycle)]
        tensors.append(transform(augment(img, mode)))
        r = dict(row)
        r["augmentation"] = mode
        kept.append(r)
    outs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i + batch_size]).to(device)
            emb, _, _, _, _, _ = model(batch, batch)
            outs.append(emb.detach().cpu().float())
    return kept, torch.cat(outs, dim=0) if outs else torch.empty((0, 384))


def best_threshold(known_scores, unknown_scores, lower_is_duplicate=True):
    vals = sorted(set(float(x) for x in known_scores + unknown_scores))
    candidates = [vals[0] - 1e-8]
    candidates += [(a + b) / 2 for a, b in zip(vals, vals[1:])]
    candidates.append(vals[-1] + 1e-8)
    best = None
    for thr in candidates:
        if lower_is_duplicate:
            kc = sum(x <= thr for x in known_scores)
            uc = sum(x > thr for x in unknown_scores)
        else:
            kc = sum(x >= thr for x in known_scores)
            uc = sum(x < thr for x in unknown_scores)
        total = kc + uc
        bal = min(kc / max(1, len(known_scores)), uc / max(1, len(unknown_scores)))
        item = (total, bal, kc, uc, thr)
        if best is None or item > best:
            best = item
    total, bal, kc, uc, thr = best
    return thr, kc, uc, total


def by_aug(rows, scores, labels, threshold, lower_is_duplicate):
    out = {}
    for row, score, label in zip(rows, scores, labels):
        pred_known = score <= threshold if lower_is_duplicate else score >= threshold
        ok = pred_known if label == 1 else not pred_known
        key = row["augmentation"]
        out.setdefault(key, [0, 0])
        out[key][0] += int(ok)
        out[key][1] += 1
    return {k: {"correct": v[0], "total": v[1], "acc": round(v[0] / max(1, v[1]), 4)} for k, v in sorted(out.items())}


def metric_scores(query, bank):
    raw = torch.cdist(query, bank)
    qn = torch.nn.functional.normalize(query, dim=1)
    bn = torch.nn.functional.normalize(bank, dim=1)
    norm_d = torch.cdist(qn, bn)
    cos_sim = qn @ bn.T
    cos_dist = 1.0 - cos_sim.max(dim=1).values
    topk = min(3, raw.shape[1])
    top3_mean = raw.topk(topk, largest=False, dim=1).values.mean(dim=1)
    return {
        "raw_l2": raw.min(dim=1).values.tolist(),
        "norm_l2": norm_d.min(dim=1).values.tolist(),
        "cosine_dist": cos_dist.tolist(),
        "top3_mean_l2": top3_mean.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/home/saketh/json/equal30k_from15k")
    ap.add_argument("--bank-json", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--size", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260419)
    args = ap.parse_args()

    base = Path(args.base)
    bank_data = json.loads(Path(args.bank_json).read_text(encoding="utf-8"))
    bank_paths = {row["path"] for row in bank_data["bank_known"]}
    bank = torch.tensor([row["embedding"] for row in bank_data["bank_known"]], dtype=torch.float32)

    models = ["convnext", "resnet", "vit"]
    known_sets, unknown_sets, rows = [], [], {}
    for m in models:
        store = json.loads((base / f"{m}_equal30k_from15k" / f"embeddings_store_{m}_15k.json").read_text())
        ks, us = set(), set()
        for row in store.get("accepted_known", []):
            if row.get("path"):
                ks.add(row["path"])
                rows[row["path"]] = {"path": row["path"], "category": row.get("category", ""), "known": True}
        for row in store.get("accepted_unknown", []):
            if row.get("path"):
                us.add(row["path"])
                rows[row["path"]] = {"path": row["path"], "category": row.get("category", ""), "known": False}
        known_sets.append(ks)
        unknown_sets.append(us)
    common_known = sorted(set.intersection(*known_sets) - bank_paths)
    common_unknown = sorted(set.intersection(*unknown_sets))
    rng = random.Random(args.seed)
    rng.shuffle(common_known)
    rng.shuffle(common_unknown)

    per_side = args.size // 2
    known_rows = [rows[p] for p in common_known[:per_side]]
    unknown_rows = [rows[p] for p in common_unknown[:per_side]]
    aug_cycle = ["bright", "contrast", "blur", "rotate5", "rotate15", "rotate-15", "flip", "crop10", "heavy_bright"]

    model, device = load_model(args.ckpt)
    transform = get_transform()
    k_rows, k_emb = encode(model, device, known_rows, transform, args.batch_size, aug_cycle)
    u_rows, u_emb = encode(model, device, unknown_rows, transform, args.batch_size, aug_cycle)

    k_scores = metric_scores(k_emb, bank)
    u_scores = metric_scores(u_emb, bank)
    report = {}
    for name in k_scores:
        thr, kc, uc, total = best_threshold(k_scores[name], u_scores[name], lower_is_duplicate=True)
        labels = [1] * len(k_rows) + [0] * len(u_rows)
        report[name] = {
            "threshold": round(float(thr), 6),
            "known_correct": int(kc),
            "known_total": len(k_rows),
            "unknown_correct": int(uc),
            "unknown_total": len(u_rows),
            "total_correct": int(total),
            "total": len(k_rows) + len(u_rows),
            "accuracy": round(total / max(1, len(k_rows) + len(u_rows)), 4),
            "known_avg": round(float(np.mean(k_scores[name])), 6),
            "unknown_avg": round(float(np.mean(u_scores[name])), 6),
            "by_augmentation": by_aug(k_rows + u_rows, k_scores[name] + u_scores[name], labels, thr, True),
        }

    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("report", args.out_json)
    print("test_size", len(k_rows) + len(u_rows), "known", len(k_rows), "unknown", len(u_rows))
    for name, r in report.items():
        print(
            f"{name:13s} | thr {r['threshold']:.6f} | "
            f"known {r['known_correct']}/{r['known_total']} | "
            f"unknown {r['unknown_correct']}/{r['unknown_total']} | "
            f"total {r['total_correct']}/{r['total']} = {r['accuracy']:.4f} | "
            f"avg known {r['known_avg']:.4f} unknown {r['unknown_avg']:.4f}"
        )
    best_name, best_r = max(report.items(), key=lambda kv: (kv[1]["accuracy"], min(kv[1]["known_correct"], kv[1]["unknown_correct"])))
    print("BEST", best_name, best_r)


if __name__ == "__main__":
    main()
