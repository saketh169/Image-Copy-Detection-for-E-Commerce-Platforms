import argparse
import json
import os
import random
from pathlib import Path

import cv2
import torch

from config import DEVICE
from dataloader import get_transform
from models.model_convnext import SiameseConvNeXtB


def load_model(ckpt_path: str):
    model = SiameseConvNeXtB().to(DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def load_image(path: str, transform):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img).unsqueeze(0).to(DEVICE)


def build_bank(model, db_items, transform):
    embs = []
    for row in db_items:
        img = load_image(row["path"], transform)
        if img is None:
            continue
        with torch.no_grad():
            emb, _, _, _, _, _ = model(img, img)
        embs.append(emb.detach().cpu())
    if not embs:
        raise SystemExit("No DB embeddings built.")
    return torch.cat(embs, dim=0)


def nearest_distance(model, bank, path, transform):
    img = load_image(path, transform)
    if img is None:
        return None
    with torch.no_grad():
        emb, _, _, _, _, _ = model(img, img)
    dist = torch.cdist(emb.detach().cpu(), bank).min().item()
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known-ckpt", required=True)
    ap.add_argument("--unknown-ckpt", required=True)
    ap.add_argument("--db-pool", required=True)
    ap.add_argument("--dup-threshold", type=float, default=1.80)
    ap.add_argument("--not-threshold", type=float, default=1.55)
    ap.add_argument("--eval-max-known", type=int, default=1000)
    ap.add_argument("--eval-max-unknown", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260415)
    args = ap.parse_args()

    pool = json.loads(Path(args.db_pool).read_text(encoding="utf-8"))
    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    random.seed(args.seed)
    random.shuffle(known)
    random.shuffle(unknown)

    db_items = known[: max(1, int(len(known) * 0.7))]
    known_queries = known[max(1, int(len(known) * 0.7)) :][: args.eval_max_known]
    unknown_queries = unknown[: args.eval_max_unknown]

    transform = get_transform()
    known_model = load_model(args.known_ckpt)
    unknown_model = load_model(args.unknown_ckpt)
    known_bank = build_bank(known_model, db_items, transform)
    unknown_bank = build_bank(unknown_model, db_items, transform)

    known_correct = 0
    unknown_correct = 0
    unsure_known = 0
    unsure_unknown = 0

    for row in known_queries:
        d_known = nearest_distance(known_model, known_bank, row["path"], transform)
        d_unknown = nearest_distance(unknown_model, unknown_bank, row["path"], transform)
        if d_known is None or d_unknown is None:
            continue
        if d_known <= args.dup_threshold:
            known_correct += 1
        elif d_unknown < args.not_threshold:
            unsure_known += 1

    for row in unknown_queries:
        d_known = nearest_distance(known_model, known_bank, row["path"], transform)
        d_unknown = nearest_distance(unknown_model, unknown_bank, row["path"], transform)
        if d_known is None or d_unknown is None:
            continue
        if d_unknown >= args.not_threshold and d_known > args.dup_threshold:
            unknown_correct += 1
        else:
            unsure_unknown += 1

    print(
        f"KNOWN accuracy: {known_correct}/{len(known_queries)} | "
        f"UNKNOWN accuracy: {unknown_correct}/{len(unknown_queries)} | "
        f"dup_thr {args.dup_threshold:.2f} | not_thr {args.not_threshold:.2f} | "
        f"unsure_known {unsure_known} | unsure_unknown {unsure_unknown}"
    )


if __name__ == "__main__":
    main()
