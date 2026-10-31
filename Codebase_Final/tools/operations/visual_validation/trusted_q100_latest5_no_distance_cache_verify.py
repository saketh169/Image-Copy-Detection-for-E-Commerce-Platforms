#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, "/home/saketh/Codebase")
from dataloader import get_transform
from models.model_convnext_metric import SiameseConvNeXtBMetric


ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
ORIGINAL_ANCHOR = ROOT / "trusted_q100_original_only_verify_20260727"
OUT = ROOT / "trusted_q100_latest5_no_distance_cache_verify_20260727"

POOL_FILES = {
    "cifar": ROOT / "phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3.npz",
    "flickr": ROOT / "phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2.npz",
    "ucid": ROOT / "phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850.npz",
    "amazon": ROOT / "phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000.npz",
}

PATH_FILES = {
    "cifar": ROOT / "phase3_large_pool_subselect_all4_20260717/cifar/large_pool_cache/cifar_pool_asym_k17000_u9000_cifar-generalized-gap-accepted-003_425324b4d3_paths.json",
    "flickr": ROOT / "phase3_large_pool_subselect_all4_20260717/flickr/large_pool_cache/flickr_pool_asym_k10000_u5000_flickr-fresh-bank-metric-layer-best_a338926af2_paths.json",
    "ucid": ROOT / "phase3_large_pool_subselect_all4_20260717/ucid/fresh20_unique_4500_850/cache/ucid_fresh20_unique_k4500_u850_paths.json",
    "amazon": ROOT / "phase3_large_pool_subselect_all4_20260717/amazon/large_pool_cache/amazon_pool_asym_k5000_u5000_paths.json",
}

CKPTS = {
    "cifar": Path("/home/saketh/Codebase/checkpoints/generalized_gap_guard_diagonal_search_20260717/cifar/cifar-generalized-gap-accepted-003.pt"),
    "flickr": Path("/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/flickr/flickr-fresh-bank-metric-layer-best.pt"),
    "ucid": Path("/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/ucid/ucid-fresh-bank-metric-layer-best.pt"),
    "amazon": Path("/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/amazon/amazon-fresh-bank-metric-layer-best.pt"),
}

ATTACKS = ["flip_h", "resize_latest", "resize_compress_latest", "watermark_text_latest", "watermark_asset_latest"]
TEXTS = [
    "50% OFF", "SALE", "AUTHENTIC", "VERIFIED", "COPY", "ORIGINAL", "CLEARANCE",
    "FREE SHIPPING", "BRAND NEW", "LIMITED", "MEGA DEAL", "HOT PICK", "GENUINE",
    "BEST PRICE", "NEW ARRIVAL", "OFFICIAL", "CHECKED", "QUALITY", "LIMITED STOCK",
    "TRUSTED",
]
POSITIONS = ["tl", "tr", "bl", "br", "center"]


def selected_indices(qdoc: dict[str, Any], side: str) -> list[int]:
    for key in ([f"{side}_indices", f"selected_{side}_indices"]):
        if key in qdoc:
            return [int(x) for x in qdoc[key]]
    raise KeyError(f"missing {side} selected indices")


def read_rgb(path: str | Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def pos_xy(pos: str, w: int, h: int, bw: int, bh: int) -> tuple[int, int]:
    margin = max(4, int(min(w, h) * 0.04))
    if pos == "tl":
        return margin, margin
    if pos == "tr":
        return max(0, w - bw - margin), margin
    if pos == "bl":
        return margin, max(0, h - bh - margin)
    if pos == "br":
        return max(0, w - bw - margin), max(0, h - bh - margin)
    return max(0, (w - bw) // 2), max(0, (h - bh) // 2)


def asset_files() -> list[Path]:
    roots = [
        Path("/home/saketh/Codebase/assets"),
        Path("/home/saketh/Codebase/watermark_assets"),
        Path("/home/saketh/json/final_named_artifacts_20260601/assets"),
    ]
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files += sorted(root.rglob("*.png"))
    return files


ASSETS = asset_files()


def apply_text_watermark(img: np.ndarray, i: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    text = TEXTS[i % len(TEXTS)]
    pos = POSITIONS[(i // len(TEXTS)) % len(POSITIONS)]
    scale = max(0.45, min(1.25, min(h, w) / 180.0))
    thick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = pos_xy(pos, w, h, tw + 12, th + 14)
    y_text = min(h - 4, y + th + 8)
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (min(w - 1, x + tw + 12), min(h - 1, y + th + 14)), (255, 255, 255), -1)
    out = cv2.addWeighted(overlay, 0.32, out, 0.68, 0)
    cv2.putText(out, text, (x + 6, y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick, cv2.LINE_AA)
    return out


def apply_asset_watermark(img: np.ndarray, i: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    pos = POSITIONS[i % len(POSITIONS)]
    scale = [0.35, 0.45, 0.55, 0.65, 0.75][i % 5]
    alpha = [0.471, 0.549, 0.627, 0.706, 0.784][i % 5]
    target = max(10, int(min(h, w) * scale))
    if ASSETS:
        asset = Image.open(ASSETS[i % len(ASSETS)]).convert("RGBA")
        ratio = asset.width / max(1, asset.height)
        aw = max(8, int(target * ratio))
        ah = target
        asset = asset.resize((aw, ah), Image.LANCZOS)
        arr = np.array(asset)
        rgb = arr[:, :, :3]
        a = (arr[:, :, 3:4].astype("float32") / 255.0) * alpha
    else:
        aw = max(24, int(target * 1.5))
        ah = target
        rgb = np.full((ah, aw, 3), 255, dtype=np.uint8)
        cv2.rectangle(rgb, (1, 1), (aw - 2, ah - 2), (30, 30, 30), 2)
        cv2.putText(rgb, "WM", (max(2, aw // 8), max(12, ah // 2)), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, ah / 45), (30, 30, 30), 2, cv2.LINE_AA)
        a = np.full((ah, aw, 1), alpha, dtype="float32")
    x, y = pos_xy(pos, w, h, aw, ah)
    roi = out[y : y + ah, x : x + aw].astype("float32")
    if roi.shape[:2] == rgb.shape[:2]:
        out[y : y + ah, x : x + aw] = np.clip(roi * (1.0 - a) + rgb.astype("float32") * a, 0, 255).astype("uint8")
    return out


def attack_img(img: np.ndarray, attack: str, i: int) -> np.ndarray:
    h, w = img.shape[:2]
    if attack == "flip_h":
        return cv2.flip(img, 1)
    if attack == "resize_latest":
        small = cv2.resize(img, (max(2, int(w * 0.72)), max(2, int(h * 0.72))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if attack == "resize_compress_latest":
        small = cv2.resize(img, (max(2, int(w * 0.55)), max(2, int(h * 0.55))), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(up, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 38])
        if ok:
            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return up
    if attack == "watermark_text_latest":
        return apply_text_watermark(img, i)
    if attack == "watermark_asset_latest":
        return apply_asset_watermark(img, i)
    raise ValueError(attack)


def load_model(ds: str, device: str) -> SiameseConvNeXtBMetric:
    ck = torch.load(CKPTS[ds], map_location=device)
    model = SiameseConvNeXtBMetric().to(device).eval()
    model.load_state_dict(ck["model_state_dict"], strict=False)
    return model


@torch.no_grad()
def embed_paths(model: SiameseConvNeXtBMetric, paths: list[str], device: str, attack: str | None = None, batch: int = 96) -> np.ndarray:
    tr = get_transform()
    outs = []
    xs = []
    for i, p in enumerate(paths):
        img = read_rgb(p)
        if attack is not None:
            img = attack_img(img, attack, i)
        xs.append(tr(img))
        if len(xs) >= batch:
            x = torch.stack(xs).to(device)
            outs.append(model(x, x)[0].detach().cpu().numpy())
            xs = []
    if xs:
        x = torch.stack(xs).to(device)
        outs.append(model(x, x)[0].detach().cpu().numpy())
    return np.concatenate(outs).astype("float32") if outs else np.empty((0, 384), dtype="float32")


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
    anchor = json.loads((ORIGINAL_ANCHOR / f"{ds}_trusted_q100_original_only_verify.json").read_text())
    qdoc = json.loads(Path(anchor["q_json"]).read_text())
    paths_doc = json.loads(PATH_FILES[ds].read_text())
    known_idx = np.array(selected_indices(qdoc, "known"), dtype=np.int64)
    unknown_idx = np.array(selected_indices(qdoc, "unknown"), dtype=np.int64)
    known_paths = qdoc.get("selected_known_paths") or [paths_doc["known_paths"][int(i)] for i in known_idx]
    unknown_paths = qdoc.get("selected_unknown_paths") or [paths_doc["unknown_paths"][int(i)] for i in unknown_idx]
    tau = float(anchor["tau"])

    pool = np.load(POOL_FILES[ds], allow_pickle=False)
    bank = pool["bank"].astype("float32", copy=False)
    model = load_model(ds, device)

    orig_known = embed_paths(model, known_paths, device, None)
    orig_unknown = embed_paths(model, unknown_paths, device, None)
    orig_kd = nearest_distances(bank, orig_known, device)
    orig_ud = nearest_distances(bank, orig_unknown, device)
    orig_known_acc = float((orig_kd <= tau).mean() * 100.0)
    orig_unknown_acc = float((orig_ud > tau).mean() * 100.0)
    orig_total = (orig_known_acc + orig_unknown_acc) / 2.0
    orig_ok = (
        abs(orig_known_acc - float(anchor["known_accuracy"])) < 1e-5
        and abs(orig_unknown_acc - float(anchor["unknown_accuracy"])) < 1e-5
    )

    rows = []
    for attack in ATTACKS:
        kemb = embed_paths(model, known_paths, device, attack)
        uemb = embed_paths(model, unknown_paths, device, attack)
        kd = nearest_distances(bank, kemb, device)
        ud = nearest_distances(bank, uemb, device)
        known = float((kd <= tau).mean() * 100.0)
        unknown = float((ud > tau).mean() * 100.0)
        total = (known + unknown) / 2.0
        rows.append(
            {
                "attack": attack,
                "known": round(known, 6),
                "unknown": round(unknown, 6),
                "total": round(total, 6),
                "read": "valid" if min(known, unknown) >= 70 else "weak",
                "known_min_dist": float(kd.min()),
                "known_max_dist": float(kd.max()),
                "unknown_min_dist": float(ud.min()),
                "unknown_max_dist": float(ud.max()),
            }
        )

    return {
        "dataset": ds,
        "checkpoint": str(CKPTS[ds]),
        "pool_store": str(POOL_FILES[ds]),
        "q_json": anchor["q_json"],
        "tau": tau,
        "bank_size": int(len(bank)),
        "known_selected": int(len(known_paths)),
        "unknown_selected": int(len(unknown_paths)),
        "distance_cache_used": False,
        "original_anchor_file": str(ORIGINAL_ANCHOR / f"{ds}_trusted_q100_original_only_verify.json"),
        "fresh_original_sanity": {
            "known": round(orig_known_acc, 6),
            "unknown": round(orig_unknown_acc, 6),
            "total": round(orig_total, 6),
            "matches_original_anchor": bool(orig_ok),
            "anchor_known": anchor["known_accuracy"],
            "anchor_unknown": anchor["unknown_accuracy"],
            "anchor_total": anchor["total_accuracy"],
        },
        "asset_files_found": len(ASSETS),
        "latest_attack_definitions": {
            "flip_h": "horizontal mirror flip",
            "resize_latest": "downscale to 72%, then upscale with linear interpolation",
            "resize_compress_latest": "downscale to 55%, upscale nearest, then JPEG quality 38",
            "watermark_text_latest": "rotating text labels and positions with translucent text box",
            "watermark_asset_latest": "real PNG asset if available, else WM badge; scale 0.35-0.75, alpha 0.471-0.784",
        },
        "rows": rows,
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
        print(f"[latest5] {ds}", flush=True)
        res = eval_dataset(ds, args.device)
        combined[ds] = res
        (OUT / f"{ds}_trusted_q100_latest5_no_distance_cache_verify.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        s = res["fresh_original_sanity"]
        print(f"{ds}: original_sanity={s['known']:.2f}/{s['unknown']:.2f}/{s['total']:.2f} match_anchor={s['matches_original_anchor']}", flush=True)
        for r in res["rows"]:
            print(f"  {r['attack']}: {r['known']:.2f}/{r['unknown']:.2f}/{r['total']:.2f} {r['read']}", flush=True)
    (OUT / "combined_trusted_q100_latest5_no_distance_cache_verify.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
