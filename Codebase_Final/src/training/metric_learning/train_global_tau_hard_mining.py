#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/saketh/Codebase")
import global_threshold_geometry_adjust as A
import phase3_all_datasets_graph_benchmark as G
from dataloader import get_transform

ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
RUN_ROOT = ROOT / "global_tau_hard_mining_10epoch_20260715"
CKPT_ROOT = Path("/home/saketh/Codebase/checkpoints/global_tau_hard_mining_10epoch_20260715")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# These are the foldable ConvNeXt checkpoints the user selected as the current base.
CONFIGS = {
    "cifar": {
        "store": ROOT / "cifar/cifar_store.json",
        "ckpt": Path("/home/saketh/Codebase/checkpoints/cifar_orig100_unknown_penalty/cifar-orig100-band-001-global-tau.pt"),
        "known_cap": 17000,
        "unknown_cap": 9000,
        "batch": 6,
        "max_batches": 520,
        "lr": 8.0e-8,
        "stage4_lr": 2.0e-9,
        "target_known": 94.0,
        "target_unknown": 94.0,
    },
    "flickr": {
        "store": ROOT / "flickr/flickr_store.json",
        "ckpt": Path("/home/saketh/Codebase/checkpoints/original_protected_rapid_diagonal/flickr/flickr-original-protected-accepted-930-global-tau.pt"),
        "known_cap": 10000,
        "unknown_cap": 5000,
        "batch": 6,
        "max_batches": 560,
        "lr": 9.0e-8,
        "stage4_lr": 2.5e-9,
        "target_known": 94.0,
        "target_unknown": 94.0,
    },
    "ucid": {
        "store": ROOT / "ucid/ucid_store.json",
        "ckpt": Path("/home/saketh/Codebase/checkpoints/final_global_tau_checkpoints/ucid/ucid-global-tau-q100-target-folded.pt"),
        "known_cap": 4500,
        "unknown_cap": 850,
        "batch": 6,
        "max_batches": 420,
        "lr": 8.0e-8,
        "stage4_lr": 2.0e-9,
        "target_known": 94.0,
        "target_unknown": 94.0,
    },
    "amazon": {
        "store": ROOT / "amazon/amazon_store.json",
        "ckpt": Path("/home/saketh/Codebase/checkpoints/amazon_known_first_high_anchor/amazon-known-first-001-global-tau.pt"),
        "known_cap": 35000,
        "unknown_cap": 15000,
        "batch": 5,
        "max_batches": 680,
        "lr": 7.0e-8,
        "stage4_lr": 1.5e-9,
        "target_known": 90.0,
        "target_unknown": 90.0,
    },
}

ATTACK_LABELS = ["original"] + A.ALL_ATTACKS
ATTACK_NAME_BY_LABEL = dict(A._PERATTACK_EVAL_ATTACKS)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_store(path: Path):
    data = read_json(path)
    known = [r for r in data.get("accepted_known", []) if Path(r.get("path", "")).exists() and "embedding" in r]
    unknown = [r for r in data.get("accepted_unknown", []) if Path(r.get("path", "")).exists() and "embedding" in r]
    return data, known, unknown


def load_model_and_tau(path: Path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = A.SiameseConvNeXtB().to(DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    tau = ckpt.get("global_tau")
    metrics = ckpt.get("metrics_q100_at_fold_time") or ckpt.get("metrics") or {}
    if tau is None:
        tau = metrics.get("best_global_threshold") or metrics.get("tau") or metrics.get("threshold")
    if tau is None:
        raise RuntimeError(f"No tau found in checkpoint {path}")
    return model, float(tau), metrics


def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def build_optimizer(model, lr, stage4_lr):
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = []
    # Final projection heads carry most embedding geometry, so update these first.
    for module in [model.global_branch, model.local_branch]:
        for p in module.parameters():
            p.requires_grad_(True)
            train_params.append(p)
    groups = [{"params": train_params, "lr": lr}]
    try:
        stage4 = list(model.backbone.stages[6].parameters())
    except Exception:
        stage4 = []
    for p in stage4:
        p.requires_grad_(True)
    if stage4 and stage4_lr > 0:
        groups.append({"params": stage4, "lr": stage4_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


class HardMiningDataset(Dataset):
    def __init__(self, known, unknown, transform, epoch, max_rows, ds):
        self.transform = transform
        self.epoch = epoch
        self.ds = ds
        self.rows = []
        rng = random.Random(20260715 + epoch)
        k = list(known)
        u = list(unknown)
        rng.shuffle(k)
        rng.shuffle(u)
        # Oversample known for datasets whose known side is weak.
        known_frac = 0.62 if ds in {"cifar", "flickr", "amazon"} else 0.52
        nk = min(len(k), int(max_rows * known_frac))
        nu = min(len(u), max_rows - nk)
        for r in k[:nk]:
            rr = dict(r)
            rr["known"] = True
            self.rows.append(rr)
        for r in u[:nu]:
            rr = dict(r)
            rr["known"] = False
            self.rows.append(rr)
        rng.shuffle(self.rows)
        self.attack_cycle = [
            "original",
            "crop40", "bg_color_change", "heavy_bright", "bright", "rotate40", "flip_v",
            "crop10", "watermark_text", "watermark_asset", "contrast", "rotate15", "rotate_m10",
            "rotate_m5", "rotate5", "blur", "resize", "resize_compress", "flip_h", "flip",
            "crop5", "crop20", "rotate30", "bright_light", "mix_rotate10_bright", "mix_crop20_watermark",
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        bgr = cv2.imread(row["path"])
        if bgr is None:
            raise RuntimeError(row["path"])
        label = self.attack_cycle[(idx * 17 + self.epoch * 29) % len(self.attack_cycle)]
        if label == "original":
            attacked = bgr.copy()
        else:
            attacked = A._apply_eval_attack(bgr.copy(), ATTACK_NAME_BY_LABEL[label])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        attacked_rgb = cv2.cvtColor(attacked, cv2.COLOR_BGR2RGB)
        return (
            self.transform(rgb),
            self.transform(attacked_rgb),
            torch.tensor(row["embedding"], dtype=torch.float32),
            torch.tensor(1.0 if row.get("known") else 0.0, dtype=torch.float32),
            label,
        )


@torch.no_grad()
def refresh_embeddings(model, rows, transform, batch_size=128):
    paths = [r["path"] for r in rows]
    embs = A._embed_paths_for_eval(model, paths, transform, 1.0, batch_size, apply_attacks=False, epoch=0)
    for row, emb in zip(rows, embs.detach().cpu()):
        row["embedding"] = [float(x) for x in emb.tolist()]


@torch.no_grad()
def eval_nearest_bank(ds, model, known, unknown, transform, tau, out_json, q=100):
    q = min(q, len(known), len(unknown))
    qk, qu = G.pick_rows(known, unknown, q)
    bank = torch.tensor([r["embedding"] for r in known], dtype=torch.float32, device=DEVICE)
    rows = []
    for label in ATTACK_LABELS:
        attack_name = None if label == "original" else ATTACK_NAME_BY_LABEL[label]
        k_emb = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64)
        u_emb = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64)
        if not torch.is_tensor(k_emb):
            k_emb = torch.tensor(k_emb, dtype=torch.float32, device=DEVICE)
        if not torch.is_tensor(u_emb):
            u_emb = torch.tensor(u_emb, dtype=torch.float32, device=DEVICE)
        kd = np.asarray(A._nearest_dists_chunked(k_emb, bank, 128), dtype=np.float32)
        ud = np.asarray(A._nearest_dists_chunked(u_emb, bank, 128), dtype=np.float32)
        known_acc = float((kd <= tau).mean() * 100.0)
        unknown_acc = float((ud > tau).mean() * 100.0)
        rows.append({
            "attack": label,
            "known": round(known_acc, 6),
            "unknown": round(unknown_acc, 6),
            "total": round((known_acc + unknown_acc) / 2.0, 6),
            "known_min_dist": round(float(kd.min()), 10),
            "known_max_dist": round(float(kd.max()), 10),
            "unknown_min_dist": round(float(ud.min()), 10),
            "unknown_max_dist": round(float(ud.max()), 10),
        })
    orig = next(r for r in rows if r["attack"] == "original")
    payload = {
        "dataset": ds,
        "q": q,
        "tau": float(tau),
        "total": round(float(np.mean([r["total"] for r in rows])), 6),
        "known": round(float(np.mean([r["known"] for r in rows])), 6),
        "unknown": round(float(np.mean([r["unknown"] for r in rows])), 6),
        "original": round(float(orig["total"]), 6),
        "origK": float(orig["known"]),
        "origU": float(orig["unknown"]),
        "minAtk": round(float(min(r["total"] for r in rows)), 6),
        "minK": round(float(min(r["known"] for r in rows)), 6),
        "minU": round(float(min(r["unknown"] for r in rows)), 6),
        "rows": rows,
    }
    write_json(out_json, payload)
    return payload


def train_one(ds: str, epochs: int):
    cfg = CONFIGS[ds]
    random.seed(20260715)
    np.random.seed(20260715)
    torch.manual_seed(20260715)
    out_dir = RUN_ROOT / ds
    ckpt_dir = CKPT_ROOT / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    store_data, known, unknown = load_store(cfg["store"])
    known = known[: cfg["known_cap"]]
    unknown = unknown[: cfg["unknown_cap"]]
    model, tau, source_metrics = load_model_and_tau(cfg["ckpt"])
    transform = get_transform(strong_aug=True)
    eval_transform = get_transform(strong_aug=False)
    print(f"[{ds}] start checkpoint {cfg['ckpt']}", flush=True)
    print(f"[{ds}] tau {tau:.10g} source total {source_metrics.get('best_global_total')} known {source_metrics.get('best_global_known')} unknown {source_metrics.get('best_global_unknown')}", flush=True)

    model.eval()
    # Make the bank correspond to the starting checkpoint before first validation/training.
    refresh_embeddings(model, known, eval_transform)
    refresh_embeddings(model, unknown, eval_transform)
    baseline = eval_nearest_bank(ds, model, known, unknown, eval_transform, tau, out_dir / "baseline_nearest_bank_q100.json")
    print(f"[{ds}] baseline nearest-bank total {baseline['total']} known {baseline['known']} unknown {baseline['unknown']} orig {baseline['origK']}/{baseline['origU']} minK {baseline['minK']} minU {baseline['minU']}", flush=True)

    for epoch in range(1, epochs + 1):
        # Refresh a bank sample at every epoch from latest known embeddings.
        rng = np.random.default_rng(20260715 + epoch)
        sample_n = min(len(known), 8192 if ds == "amazon" else 4096)
        sample_idx = rng.choice(len(known), size=sample_n, replace=False)
        bank_sample = torch.tensor([known[int(i)]["embedding"] for i in sample_idx], dtype=torch.float32, device=DEVICE)
        dataset = HardMiningDataset(known, unknown, transform, epoch, cfg["max_batches"] * cfg["batch"], ds)
        loader = DataLoader(dataset, batch_size=cfg["batch"], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        opt = build_optimizer(model, cfg["lr"], cfg["stage4_lr"])
        model.train()
        set_bn_eval(model)
        start = time.time()
        sums = {"loss": 0.0, "known": 0.0, "unknown": 0.0, "sep": 0.0}
        nb = min(len(loader), cfg["max_batches"])
        for b, (orig, atk, anchor, is_known, label) in enumerate(loader, 1):
            if b > cfg["max_batches"]:
                break
            orig = orig.to(DEVICE, non_blocking=True)
            atk = atk.to(DEVICE, non_blocking=True)
            anchor = anchor.to(DEVICE, non_blocking=True)
            is_known = is_known.to(DEVICE, non_blocking=True) > 0.5
            orig_emb, _, _, _, _, _ = model(orig, orig)
            atk_emb, _, _, _, _, _ = model(atk, atk)
            loss = torch.tensor(0.0, device=DEVICE)
            known_loss = torch.tensor(0.0, device=DEVICE)
            unknown_loss = torch.tensor(0.0, device=DEVICE)
            sep_loss = torch.tensor(0.0, device=DEVICE)

            if is_known.any():
                ko = orig_emb[is_known]
                ka = atk_emb[is_known]
                kan = anchor[is_known]
                # Known original + all attacks must sit below the single tau.
                own_dist = torch.norm(ka - ko.detach(), dim=1)
                anchor_dist = torch.norm(ka - kan.detach(), dim=1)
                known_band = F.relu(own_dist - tau * 0.72).pow(2).mean() + F.relu(anchor_dist - tau * 0.92).pow(2).mean()
                known_loss = (
                    18.0 * F.smooth_l1_loss(ka, ko.detach())
                    + 5.0 * F.smooth_l1_loss(ka, kan.detach())
                    + 3.0 * F.smooth_l1_loss(ko, kan.detach())
                    + 6.0 * known_band
                    + 3.0 * (1.0 - F.cosine_similarity(ka, ko.detach(), dim=1)).mean()
                )
                loss = loss + known_loss

            if (~is_known).any():
                uo = orig_emb[~is_known]
                ua = atk_emb[~is_known]
                uan = anchor[~is_known]
                d_ua = torch.cdist(ua, bank_sample).min(dim=1).values
                d_uo = torch.cdist(uo, bank_sample).min(dim=1).values
                # Unknown original and attacks should be outside the global tau.
                margin = tau * (1.18 if ds != "amazon" else 1.08)
                sep_loss = F.relu(margin - d_ua).pow(2).mean() + 0.65 * F.relu(margin - d_uo).pow(2).mean()
                unknown_loss = (
                    6.0 * F.smooth_l1_loss(ua, uo.detach())
                    + 2.0 * F.smooth_l1_loss(ua, uan.detach())
                    + 24.0 * sep_loss
                )
                loss = loss + unknown_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 0.25)
            opt.step()
            sums["loss"] += float(loss.detach().cpu())
            sums["known"] += float(known_loss.detach().cpu())
            sums["unknown"] += float(unknown_loss.detach().cpu())
            sums["sep"] += float(sep_loss.detach().cpu())
            if b == 1 or b % 50 == 0:
                div = float(b)
                print(f"[{ds}] epoch {epoch}/{epochs} batch {b}/{nb} loss {sums['loss']/div:.6f} known {sums['known']/div:.6f} unknown {sums['unknown']/div:.6f} sep {sums['sep']/div:.8f}", flush=True)

        model.eval()
        refresh_embeddings(model, known, eval_transform)
        refresh_embeddings(model, unknown, eval_transform)
        eval_payload = eval_nearest_bank(ds, model, known, unknown, eval_transform, tau, out_dir / f"epoch-{epoch:03d}_nearest_bank_q100.json")
        store_payload = dict(store_data)
        store_payload["accepted_known"] = known
        store_payload["accepted_unknown"] = unknown
        store_payload["meta"] = dict(store_payload.get("meta", {}))
        store_payload["meta"].update({
            "run": "global_tau_hard_mining_10epoch_20260715",
            "epoch": epoch,
            "tau": tau,
            "source_checkpoint": str(cfg["ckpt"]),
            "validation_rule": "nearest known-bank distance for known and unknown queries",
        })
        write_json(out_dir / f"{ds}_store_epoch-{epoch:03d}.json", store_payload)
        ckpt_path = ckpt_dir / f"{ds}-epoch-{epoch:03d}-global-tau-hard-mining.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "dataset": ds,
            "epoch": epoch,
            "global_tau": tau,
            "source_checkpoint": str(cfg["ckpt"]),
            "metrics_q100_nearest_bank": eval_payload,
            "training_objective": "global-threshold-aware hard mining: pull known attacks to originals, push unknown originals/attacks away from known bank",
        }, ckpt_path)
        print(f"[{ds}] epoch {epoch} result total {eval_payload['total']} known {eval_payload['known']} unknown {eval_payload['unknown']} orig {eval_payload['origK']}/{eval_payload['origU']} minK {eval_payload['minK']} minU {eval_payload['minU']} time {time.time()-start:.1f}s ckpt {ckpt_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    train_one(args.dataset, args.epochs)


if __name__ == "__main__":
    main()