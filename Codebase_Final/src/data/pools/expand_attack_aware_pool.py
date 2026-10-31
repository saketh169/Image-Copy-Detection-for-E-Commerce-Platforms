import argparse
import csv
import json
import os
import pickle
import random
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

sys.path.insert(0, "/home/saketh/Codebase")
from config import DEVICE  # noqa: E402
from dataloader import get_transform  # noqa: E402
from models.model_convnext import SiameseConvNeXtB  # noqa: E402
from transforms import CopyAttackAugmentor  # noqa: E402

ROOT = Path("/home/saketh")
DATA_ROOT = ROOT / "Data Processed"
CLEAN = Path("/home/saketh/json/clean_best_json_20260528")
FULL = Path("/home/saketh/json/clean_best_json_fullratio_20260528")
CKPT = Path("/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528")
OUT = Path("/home/saketh/json/clean_best_json_attackaware_expanded_20260528")

ATTACKS = [
    "original", "rotate_m5", "rotate_m10", "rotate5", "rotate15", "rotate30", "rotate40",
    "crop5", "crop10", "crop20", "crop40", "bright_light", "bright", "heavy_bright",
    "flip_h", "flip_v", "flip", "resize_compress", "resize", "watermark_text",
    "watermark_asset", "mix_rotate10_bright", "mix_crop20_watermark", "bg_color_change",
    "blur", "contrast",
]

CONFIGS = {
    "amazon50k": {
        "store": CLEAN / "amazon50k/embeddings_store_convnext_amazon50k_unified.json",
        "source": CLEAN / "amazon50k/source_pool_convnext_amazon50k.json",
        "table": CLEAN / "amazon50k/amazon50k_q100_26attacks_table.csv",
        "ckpt": CKPT / "amazon50k/amazon50k-epoch-033-adjustment.pt",
        "target_known": 35000,
        "target_unknown": 15000,
        "source_mode": "amazon_categories",
    },
    "flickr10k": {
        "store": CLEAN / "flickr10k/embeddings_store_flickr10k.json",
        "table": CLEAN / "flickr10k/flickr10k_q100_26attacks_table.csv",
        "ckpt": CKPT / "flickr10k/flickr10k-epoch-024-training.pt",
        "target_known": 10000,
        "target_unknown": 5000,
        "source_mode": "flickr30k",
    },
    "cifar": {
        "store": CLEAN / "cifar/embeddings_store_cifar.json",
        "source": FULL / "cifar/cifar_source_pool_fullratio.json",
        "table": CLEAN / "cifar/cifar_q100_26attacks_table.csv",
        "ckpt": CKPT / "cifar/cifar-epoch-015-training.pt",
        "target_known": 20000,
        "target_unknown": 10000,
        "source_mode": "cifar_remap",
    },
}


def load_thresholds(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["attack"]] = float(row["threshold"])
    return out


def load_model(path):
    model = SiameseConvNeXtB().to(DEVICE)
    ckpt = torch.load(str(path), map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.eval()
    return model


def load_rgb(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_keep(img, keep):
    h, w = img.shape[:2]
    y1 = int(h * (1.0 - keep) / 2.0)
    x1 = int(w * (1.0 - keep) / 2.0)
    y2 = max(y1 + 2, int(h - y1))
    x2 = max(x1 + 2, int(w - x1))
    return cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)


def apply_attack(img, attack, aug):
    bank = aug.fixed_attack_bank()
    if attack == "original":
        return img.copy()
    if attack == "rotate_m5":
        return aug.rotate_affine(img, -5)
    if attack == "rotate_m10":
        return aug.rotate_affine(img, -10)
    if attack == "rotate30":
        return aug.rotate_affine(img, 30)
    if attack == "rotate40":
        return aug.rotate_affine(img, 40)
    if attack == "crop5":
        return crop_keep(img, 0.95)
    if attack == "crop10":
        return crop_keep(img, 0.90)
    if attack == "crop20":
        return crop_keep(img, 0.80)
    if attack == "crop40":
        return crop_keep(img, 0.60)
    if attack == "flip_h" or attack == "flip":
        return cv2.flip(img, 1)
    if attack == "flip_v":
        return cv2.flip(img, 0)
    if attack == "watermark_asset":
        return (bank.get("watermark_asset") or bank.get("watermark_text"))(img)
    if attack == "mix_rotate10_bright":
        return aug.apply_bright_family(aug.rotate_affine(img, 10))
    if attack == "mix_crop20_watermark":
        return aug.apply_watermark_text(crop_keep(img, 0.80))
    fn = bank.get(attack)
    if attack == "resize_compress" and fn is None:
        fn = bank.get("resize")
    if attack == "contrast" and fn is None:
        return np.clip((img.astype(np.float32) - 127.5) * 1.25 + 127.5, 0, 255).astype(np.uint8)
    if fn is None:
        raise ValueError(attack)
    return fn(img)


def encode_rows(model, rows, transform, attack="original", batch_size=96):
    aug = CopyAttackAugmentor()
    kept, batch, outs = [], [], []
    with torch.no_grad():
        for row in rows:
            img = load_rgb(row["path"])
            if img is None:
                continue
            batch.append(transform(apply_attack(img, attack, aug)))
            kept.append(row)
            if len(batch) >= batch_size:
                x = torch.stack(batch).to(DEVICE)
                emb, _, _, _, _, _ = model(x, x)
                outs.append(emb.detach().cpu().float())
                batch = []
        if batch:
            x = torch.stack(batch).to(DEVICE)
            emb, _, _, _, _, _ = model(x, x)
            outs.append(emb.detach().cpu().float())
    arr = torch.cat(outs, dim=0).numpy().astype(np.float32)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return kept, arr / n


def key(row):
    return (str(row.get("split", "")), str(row.get("label_id", "")), str(row.get("image_index", "")), os.path.basename(row.get("path", "")))


def remap_rows(rows, source):
    by_key = {key(r): r for r in source}
    by_base = {os.path.basename(r.get("path", "")): r for r in source}
    out = []
    for row in rows:
        mapped = by_key.get(key(row)) or by_base.get(os.path.basename(row.get("path", "")))
        if mapped:
            out.append(dict(mapped))
    return out


def collect_source(name, cfg, store):
    if cfg["source_mode"] == "flickr30k":
        paths = sorted(p for p in Path("/home/saketh/flickr_dataset/flickr30k_images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        return [{"path": str(p), "known": False, "category": "flickr/full30k", "product": f"flickr/{p.stem}", "image_index": i} for i, p in enumerate(paths)]
    if cfg["source_mode"] == "cifar_remap":
        return json.load(open(cfg["source"]))
    source = json.load(open(cfg["source"]))
    source_by_path = {r["path"]: r for r in source}
    cats = sorted({r["category"] for r in source})
    rows = {r["path"]: dict(r) for r in source}
    for cat in cats:
        cdir = DATA_ROOT / cat
        if not cdir.exists():
            continue
        for p in cdir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and ("_org" in p.stem or p.stem.endswith("org")):
                sp = str(p)
                if sp in rows:
                    continue
                rel = p.relative_to(DATA_ROOT)
                parts = rel.parts
                rows[sp] = {
                    "path": sp,
                    "known": False,
                    "category": "/".join(parts[:2]),
                    "product": "/".join(parts[:-1]),
                    "image_index": -1,
                    "top_level": parts[0],
                    "original_only": True,
                }
    return list(rows.values())


def score_candidates(model, transform, bank_index, candidates, thresholds, mode, max_rows):
    candidates = candidates[:max_rows]
    scores = {r["path"]: {"row": r, "pass": 0, "dist_sum": 0.0, "checked": 0} for r in candidates}
    for attack in ATTACKS:
        kept, emb = encode_rows(model, candidates, transform, attack=attack)
        d, _ = bank_index.search(emb.astype(np.float32), 1)
        dist = np.sqrt(d[:, 0])
        thr = thresholds[attack]
        for row, val in zip(kept, dist.tolist()):
            rec = scores[row["path"]]
            ok = val <= thr if mode == "known" else val > thr
            rec["pass"] += int(ok)
            rec["dist_sum"] += float(val)
            rec["checked"] += 1
        print(f"  {mode} attack {attack} checked {len(kept)}", flush=True)
    vals = [v for v in scores.values() if v["checked"] == len(ATTACKS)]
    if mode == "known":
        vals.sort(key=lambda x: (-x["pass"], x["dist_sum"]))
    else:
        vals.sort(key=lambda x: (-x["pass"], -x["dist_sum"]))
    return vals


def run_dataset(name):
    cfg = CONFIGS[name]
    ddir = OUT / name
    ddir.mkdir(parents=True, exist_ok=True)
    store = json.load(open(cfg["store"]))
    source = collect_source(name, cfg, store)
    if name == "cifar":
        base_known = remap_rows(store["accepted_known"], source)
        base_unknown = remap_rows(store["accepted_unknown"], source)
    else:
        base_known = [r for r in store["accepted_known"] if os.path.exists(r["path"])]
        base_unknown = [r for r in store["accepted_unknown"] if os.path.exists(r["path"])]
    used = {r["path"] for r in base_known + base_unknown}
    source = [r for r in source if os.path.exists(r["path"]) and r["path"] not in used]
    print(f"[{name}] base {len(base_known)}/{len(base_unknown)} source_candidates {len(source)}", flush=True)

    need_known = max(0, cfg["target_known"] - len(base_known))
    need_unknown = max(0, cfg["target_unknown"] - len(base_unknown))
    if need_known == 0 and need_unknown == 0:
        print(f"[{name}] already at target", flush=True)
        return

    model = load_model(cfg["ckpt"])
    transform = get_transform()
    _, bank = encode_rows(model, base_known, transform)
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank.astype(np.float32))
    thresholds = load_thresholds(cfg["table"])

    # Prefilter by original distance so attack-aware pass is spent on likely useful rows.
    pre_kept, pre_emb = encode_rows(model, source, transform, attack="original")
    dist, _ = index.search(pre_emb.astype(np.float32), 1)
    ranked = list(zip(np.sqrt(dist[:, 0]).tolist(), pre_kept))
    close = [r for _, r in sorted(ranked, key=lambda x: x[0])]
    far = [r for _, r in sorted(ranked, key=lambda x: x[0], reverse=True)]

    add_known, add_unknown = [], []
    if need_known:
        max_known = min(len(close), max(need_known * 8, 5000))
        known_scores = score_candidates(model, transform, index, close, thresholds, "known", max_known)
        add_known = [dict(s["row"], known=True, attack_pass_count=s["pass"]) for s in known_scores[:need_known]]
        used.update(r["path"] for r in add_known)
    if need_unknown:
        far = [r for r in far if r["path"] not in used]
        max_unknown = min(len(far), max(need_unknown * 3, 5000))
        unknown_scores = score_candidates(model, transform, index, far, thresholds, "unknown", max_unknown)
        add_unknown = [dict(s["row"], known=False, attack_pass_count=s["pass"]) for s in unknown_scores[:need_unknown]]

    final_known = base_known + add_known
    final_unknown = base_unknown + add_unknown
    final = {
        "accepted_known": final_known,
        "accepted_unknown": final_unknown,
        "processed_paths": [r["path"] for r in final_known + final_unknown],
        "candidate_known": [],
        "candidate_unknown": [],
        "last_epoch": "attackaware_expanded_20260528",
        "meta": {
            "base_store": str(cfg["store"]),
            "checkpoint": str(cfg["ckpt"]),
            "target_known": cfg["target_known"],
            "target_unknown": cfg["target_unknown"],
            "added_known": len(add_known),
            "added_unknown": len(add_unknown),
            "selection": "original prefilter plus 26-attack pass-count ranking against baseline thresholds",
        },
    }
    store_path = ddir / f"{name}_attackaware_expanded_store.json"
    source_path = ddir / f"{name}_attackaware_expanded_source_pool.json"
    summary_path = ddir / f"{name}_attackaware_expanded_summary.json"
    store_path.write_text(json.dumps(final, indent=2))
    source_path.write_text(json.dumps(final_known + final_unknown, indent=2))
    summary = {
        "dataset": name,
        "known": len(final_known),
        "unknown": len(final_unknown),
        "total": len(final_known) + len(final_unknown),
        "added_known": len(add_known),
        "added_unknown": len(add_unknown),
        "store": str(store_path),
        "source_pool": str(source_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CONFIGS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
