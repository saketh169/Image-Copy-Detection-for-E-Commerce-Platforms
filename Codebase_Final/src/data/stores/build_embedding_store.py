import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms as T

from config import DEVICE, CHECKPOINT_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from models.model_resnet import SiameseResNet50


def get_transform():
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def iter_images(data_root):
    for root, _, files in os.walk(data_root):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(root, name)


def category_key_for_path(path):
    parts = Path(path).parts
    if "Data Processed" in parts:
        idx = parts.index("Data Processed")
        if idx + 3 <= len(parts):
            cat_parts = parts[idx + 1 : -2]
            if cat_parts:
                return "/".join(cat_parts)
    return None


def load_image(path, transform):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img).unsqueeze(0).to(DEVICE)


def extract_embedding(model, path, transform):
    img = load_image(path, transform)
    if img is None:
        return None
    with torch.no_grad():
        emb, _, _, _, _, _ = model(img, img)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)


def min_distance(emb, db_embeddings, chunk_size=4096):
    if db_embeddings is None or len(db_embeddings) == 0:
        return None
    min_dist = None
    for start in range(0, len(db_embeddings), chunk_size):
        chunk = db_embeddings[start:start + chunk_size]
        diff = chunk - emb
        dists = np.sqrt((diff * diff).sum(axis=1))
        cur = float(dists.min())
        if min_dist is None or cur < min_dist:
            min_dist = cur
    return min_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/saketh/Data Processed")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--dup-threshold", type=float, dest="dup_threshold")
    parser.add_argument("--output-emb", default="/home/saketh/json/unique_embeddings.npy")
    parser.add_argument("--output-meta", default="/home/saketh/json/unique_embeddings.jsonl")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--known-categories-json", default="")
    parser.add_argument("--emb-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.dup_threshold is None and args.threshold is None:
        raise SystemExit("must provide --threshold or --dup-threshold")
    dedup_threshold = float(args.dup_threshold if args.dup_threshold is not None else args.threshold)
    known_set = None
    if args.known_categories_json:
        with open(args.known_categories_json, "r", encoding="utf-8") as f:
            known_set = set(json.load(f))

    transform = get_transform()
    model = SiameseResNet50().to(DEVICE)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "resnet_final.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    db_embeddings = []
    kept = 0
    seen = 0

    meta_f = open(args.output_meta, "w", encoding="utf-8")

    for path in iter_images(args.data_root):
        seen += 1
        if args.max_images and seen > args.max_images:
            break
        if known_set is not None:
            cat = category_key_for_path(path)
            if cat not in known_set:
                continue

        emb = extract_embedding(model, path, transform)
        if emb is None:
            continue
        emb = emb * args.emb_scale

        if db_embeddings:
            db_arr = np.vstack(db_embeddings)
            dist = min_distance(emb, db_arr)
            if dist is not None and dist < dedup_threshold:
                continue

        db_embeddings.append(emb)
        kept += 1
        meta_f.write(json.dumps({"path": path}) + "\n")

        if kept % 1000 == 0:
            print(f"kept {kept} / seen {seen}")

    meta_f.close()

    if db_embeddings:
        np.save(args.output_emb, np.vstack(db_embeddings))
    print(f"done: kept={kept}, seen={seen}")


if __name__ == "__main__":
    main()
