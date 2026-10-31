import json
import os
import sys
import argparse
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

sys.path.insert(0, "/home/saketh/Codebase")
from config import DEVICE  # noqa: E402
from dataloader import get_transform  # noqa: E402
from models.model_convnext import SiameseConvNeXtB  # noqa: E402


OUT = Path("/home/saketh/json/clean_best_json_ranked_final_20260528")
CLEAN = Path("/home/saketh/json/clean_best_json_20260528")
FULL = Path("/home/saketh/json/clean_best_json_fullratio_20260528")
CKPT = Path("/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528")


CONFIGS = {
    "flickr10k": {
        "source": None,
        "image_dir": Path("/home/saketh/flickr_dataset/flickr30k_images"),
        "store": CLEAN / "flickr10k/embeddings_store_flickr10k.json",
        "ckpt": CKPT / "flickr10k/flickr10k-epoch-024-training.pt",
        "target_known": 10000,
        "target_unknown": 5000,
        "preserve_existing": False,
        "note": "Built from full Flickr30k image folder; ranked split 10k known / 5k unknown.",
    },
    "cifar": {
        "source": FULL / "cifar/cifar_source_pool_fullratio.json",
        "store": CLEAN / "cifar/embeddings_store_cifar.json",
        "ckpt": CKPT / "cifar/cifar-epoch-015-training.pt",
        "target_known": 20000,
        "target_unknown": 10000,
        "preserve_existing": True,
        "note": "Preserved existing accepted rows, then added remaining best-ranked candidates.",
    },
}


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


def encode_rows(model, rows, transform, batch_size=96):
    outs = []
    kept = []
    batch = []
    with torch.no_grad():
        for row in rows:
            img = load_rgb(row["path"])
            if img is None:
                continue
            batch.append(transform(img))
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
    return (
        str(row.get("split", "")),
        str(row.get("label_id", "")),
        str(row.get("image_index", "")),
        os.path.basename(row.get("path", "")),
    )


def path_key(row):
    return row.get("path", "")


def remap_store_rows_to_source(store_rows, source_rows):
    # CIFAR clean accepted store has old deleted paths. Remap by split/label/image_index when possible.
    by_key = {key(row): row for row in source_rows}
    by_base = {os.path.basename(row.get("path", "")): row for row in source_rows}
    out = []
    for row in store_rows:
        mapped = by_key.get(key(row)) or by_base.get(os.path.basename(row.get("path", "")))
        if mapped:
            new_row = dict(mapped)
            new_row["known"] = bool(row.get("known", new_row.get("known", False)))
            out.append(new_row)
    return out


def build_dataset(name, cfg):
    ddir = OUT / name
    ddir.mkdir(parents=True, exist_ok=True)
    if name == "flickr10k" and cfg.get("image_dir"):
        paths = sorted(
            [
                p
                for p in cfg["image_dir"].iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        source_rows = [
            {
                "path": str(path),
                "known": False,
                "category": "flickr/full30k",
                "product": f"flickr/{path.stem}",
                "image_index": idx,
            }
            for idx, path in enumerate(paths)
        ]
    else:
        source_rows = json.load(open(cfg["source"]))
    store = json.load(open(cfg["store"]))
    accepted_known_raw = [dict(r, known=True) for r in store.get("accepted_known", [])]
    accepted_unknown_raw = [dict(r, known=False) for r in store.get("accepted_unknown", [])]

    if name == "cifar":
        accepted_known = remap_store_rows_to_source(accepted_known_raw, source_rows)
        accepted_unknown = remap_store_rows_to_source(accepted_unknown_raw, source_rows)
    else:
        accepted_known = [r for r in accepted_known_raw if os.path.exists(r["path"])]
        accepted_unknown = [r for r in accepted_unknown_raw if os.path.exists(r["path"])]

    print(f"[{name}] source={len(source_rows)} accepted_known={len(accepted_known)} accepted_unknown={len(accepted_unknown)}", flush=True)

    model = load_model(cfg["ckpt"])
    transform = get_transform()
    bank_rows, bank = encode_rows(model, accepted_known, transform)
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank.astype(np.float32))

    source_rows = [r for r in source_rows if os.path.exists(r["path"])]
    scored = []
    chunk = 2048
    for i in range(0, len(source_rows), chunk):
        part = source_rows[i : i + chunk]
        kept, emb = encode_rows(model, part, transform)
        dists, _ = index.search(emb.astype(np.float32), 1)
        for row, dist in zip(kept, np.sqrt(dists[:, 0]).tolist()):
            scored.append((float(dist), row))
        print(f"[{name}] scored {len(scored)}/{len(source_rows)}", flush=True)

    scored_low = sorted(scored, key=lambda x: x[0])
    scored_high = sorted(scored, key=lambda x: x[0], reverse=True)

    if cfg["preserve_existing"]:
        known_by_path = {r["path"]: dict(r, known=True) for r in accepted_known}
        unknown_by_path = {r["path"]: dict(r, known=False) for r in accepted_unknown}
        for _, row in scored_low:
            if len(known_by_path) >= cfg["target_known"]:
                break
            if row["path"] not in known_by_path and row["path"] not in unknown_by_path:
                known_by_path[row["path"]] = dict(row, known=True)
        for _, row in scored_high:
            if len(unknown_by_path) >= cfg["target_unknown"]:
                break
            if row["path"] not in known_by_path and row["path"] not in unknown_by_path:
                unknown_by_path[row["path"]] = dict(row, known=False)
        known = list(known_by_path.values())[: cfg["target_known"]]
        unknown = list(unknown_by_path.values())[: cfg["target_unknown"]]
    else:
        known = []
        used = set()
        for _, row in scored_low:
            if len(known) >= cfg["target_known"]:
                break
            known.append(dict(row, known=True))
            used.add(row["path"])
        unknown = []
        for _, row in scored_high:
            if len(unknown) >= cfg["target_unknown"]:
                break
            if row["path"] not in used:
                unknown.append(dict(row, known=False))
                used.add(row["path"])

    final_store = {
        "accepted_known": known,
        "accepted_unknown": unknown,
        "processed_paths": [r["path"] for r in known + unknown],
        "candidate_known": [],
        "candidate_unknown": [],
        "last_epoch": "ranked_final_20260528",
        "meta": {
            "dataset": name,
            "target_known": cfg["target_known"],
            "target_unknown": cfg["target_unknown"],
            "actual_known": len(known),
            "actual_unknown": len(unknown),
            "selection": "faiss_distance_to_existing_known_bank",
            "checkpoint": str(cfg["ckpt"]),
            "note": cfg["note"],
        },
    }
    source_final = known + unknown
    (ddir / f"{name}_ranked_final_store.json").write_text(json.dumps(final_store, indent=2))
    (ddir / f"{name}_ranked_final_source_pool.json").write_text(json.dumps(source_final, indent=2))
    summary = {
        "dataset": name,
        "known": len(known),
        "unknown": len(unknown),
        "total": len(known) + len(unknown),
        "store": str(ddir / f"{name}_ranked_final_store.json"),
        "source_pool": str(ddir / f"{name}_ranked_final_source_pool.json"),
        "note": cfg["note"],
    }
    (ddir / f"{name}_ranked_final_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CONFIGS), default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    names = [args.dataset] if args.dataset else list(CONFIGS)
    for name in names:
        cfg = CONFIGS[name]
        summaries.append(build_dataset(name, cfg))
    if args.dataset:
        (OUT / f"{args.dataset}_ranked_final_summary_root.json").write_text(json.dumps(summaries, indent=2))
    else:
        (OUT / "ranked_final_summary.json").write_text(json.dumps(summaries, indent=2))
    print("OUT", OUT)


if __name__ == "__main__":
    main()
