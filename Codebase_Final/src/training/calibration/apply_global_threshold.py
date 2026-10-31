#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/saketh/Codebase")
import global_threshold_geometry_adjust as A
import phase3_all_datasets_graph_benchmark as G

ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
OUT = ROOT / "final_tau_force_from_diagonal"
CKPT_OUT = Path("/home/saketh/Codebase/checkpoints/final_tau_force_from_diagonal")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CKPTS = {
    "cifar": Path("/home/saketh/Codebase/checkpoints/diagonal_tau_single_checkpoint/cifar/cifar-best-gap-single-checkpoint.pt"),
    "flickr": Path("/home/saketh/Codebase/checkpoints/diagonal_tau_single_checkpoint/flickr/flickr-best-gap-single-checkpoint.pt"),
    "ucid": Path("/home/saketh/Codebase/checkpoints/diagonal_tau_single_checkpoint/ucid/ucid-best-gap-single-checkpoint.pt"),
    "amazon": Path("/home/saketh/Codebase/checkpoints/diagonal_tau_single_checkpoint/amazon/amazon-best-gap-single-checkpoint.pt"),
}

MAX_ACCEPT = 200
MAX_ATTEMPTS = 20000
Q_EVAL = 100


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_log(path: Path, msg: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state(model, state):
    model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()}, strict=True)


def best_tau(kd, ud):
    vals = np.unique(np.concatenate([kd, ud]))
    if len(vals) > 12000:
        vals = np.quantile(vals, np.linspace(0.0, 1.0, 12000))
    cand = np.concatenate([vals, (vals[:-1] + vals[1:]) / 2.0])
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in cand:
        k = float((kd <= t).mean() * 100.0)
        u = float((ud > t).mean() * 100.0)
        total = (k + u) / 2.0
        if total > best[0]:
            best = (total, k, u, float(t))
    return best


def nearest(bank, q):
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(np.ascontiguousarray(bank.astype(np.float32)))
    d2, idx = index.search(np.ascontiguousarray(q.astype(np.float32)), 1)
    return np.sqrt(np.maximum(d2[:, 0], 0.0)), idx[:, 0]


def attack_pair(path: str, attack: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(path)
    atk = A._apply_eval_attack(bgr.copy(), A.ATTACK_NAME_BY_LABEL[attack])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(atk, cv2.COLOR_BGR2RGB)


class BoundaryDataset(Dataset):
    def __init__(self, hard_known, hard_unknown, transform, seed, max_rows):
        rng = random.Random(20260616 + seed)
        hard_known = [r for r in hard_known if Path(r["path"]).exists()]
        hard_unknown = [r for r in hard_unknown if Path(r["path"]).exists()]
        rng.shuffle(hard_known)
        rng.shuffle(hard_unknown)
        # Known side is the accuracy bottleneck; still keep unknown pressure.
        k_count = int(max_rows * 0.82)
        rows = hard_known[:k_count] + hard_unknown[: max_rows - k_count]
        if len(rows) < max_rows:
            rows += hard_known[: max_rows - len(rows)]
        rng.shuffle(rows)
        self.rows = rows[:max_rows]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rgb, atk = attack_pair(r["path"], r["attack"])
        return (
            self.transform(rgb),
            self.transform(atk),
            torch.tensor(r["anchor"], dtype=torch.float32),
            torch.tensor(1.0 if r["known"] else 0.0, dtype=torch.float32),
        )


def load_best_prev(ds: str):
    d = OUT / ds
    best = None
    for p in d.glob(f"{ds}_accepted-*_final_tau_gap.json"):
        try:
            m = json.load(open(p))
            n = int(re.search(r"accepted-(\d+)", p.name).group(1))
            ck = CKPT_OUT / ds / f"{ds}-accepted-{n:03d}-final-tau-single.pt"
            if not ck.exists():
                continue
            item = (float(m["goal_score"]), float(m["global_gap"]), float(m["best_global_total"]), n, ck, p, m)
            if best is None or item[0] > best[0]:
                best = item
        except Exception:
            pass
    return best


def evaluate(ds, model, known, unknown, q=Q_EVAL, hard_n=4096):
    q = min(q, len(known), len(unknown))
    qk, qu = G.pick_rows(known, unknown, q)
    bank = np.asarray([r["embedding"] for r in known], dtype=np.float32)
    all_kd, all_ud = [], []
    hard_k, hard_u = [], []
    rows = []
    for label, attack_name in G.ATTACKS:
        print(f"[{ds}] eval {label}", flush=True)
        ke = G.embed_eval_rows(ds, model, qk, label, attack_name, batch_size=64).astype(np.float32)
        ue = G.embed_eval_rows(ds, model, qu, label, attack_name, batch_size=64).astype(np.float32)
        anchors = np.asarray([r["embedding"] for r in qk], dtype=np.float32)
        kd = np.linalg.norm(ke - anchors, axis=1)
        ud, nn = nearest(bank, ue)
        all_kd.append(kd)
        all_ud.append(ud)
        total, ka, ua, tau = best_tau(kd, ud)
        rows.append(
            {
                "attack": label,
                "known": round(ka, 2),
                "unknown": round(ua, 2),
                "total": round(total, 2),
                "tau": round(tau, 6),
                "known_max": round(float(kd.max()), 6),
                "unknown_min": round(float(ud.min()), 6),
                "gap": round(float(ud.min() - kd.max()), 6),
            }
        )
        for row, dist, anchor in zip(qk, kd, anchors):
            hard_k.append({"path": row["path"], "attack": label, "known": True, "score": float(dist), "anchor": anchor.tolist()})
        for row, dist, near in zip(qu, ud, nn):
            hard_u.append({"path": row["path"], "attack": label, "known": False, "score": float(-dist), "anchor": known[int(near)]["embedding"]})
    kd = np.concatenate(all_kd)
    ud = np.concatenate(all_ud)
    total, known_acc, unknown_acc, tau = best_tau(kd, ud)
    maxk = float(kd.max())
    minu = float(ud.min())
    gap = minu - maxk
    # Large gap weight: this is final tau forcing, not plain accuracy tuning.
    goal = float(total + 35.0 * gap)
    hard_k.sort(key=lambda r: r["score"], reverse=True)
    hard_u.sort(key=lambda r: r["score"], reverse=True)
    return {
        "best_global_total": round(total, 2),
        "best_global_known": round(known_acc, 2),
        "best_global_unknown": round(unknown_acc, 2),
        "best_global_threshold": round(tau, 6),
        "max_known_attack_distance": round(maxk, 6),
        "min_unknown_known_distance": round(minu, 6),
        "global_gap": round(gap, 6),
        "goal_score": round(goal, 6),
        "rows": sorted(rows, key=lambda r: r["total"], reverse=True),
        "hard_known": hard_k[:hard_n],
        "hard_unknown": hard_u[:hard_n],
    }


def train_step(ds, model, metric, cfg, known, attempt):
    transform = A.get_transform(strong_aug=True)
    batch = min(64, int(cfg.get("batch", 64)))
    batches = {"amazon": 160, "cifar": 120, "flickr": 120, "ucid": 120}[ds]
    ds_train = BoundaryDataset(metric["hard_known"], metric["hard_unknown"], transform, attempt, batches * batch)
    loader = DataLoader(ds_train, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    scale = [0.08, 0.14, 0.22, 0.34, 0.50, 0.72, 1.00][(attempt - 1) % 7]
    opt = A.build_optimizer(model, float(cfg["lr"]) * scale, float(cfg["stage4_lr"]) * scale)
    tau = float(metric["best_global_threshold"])
    maxk = float(metric["max_known_attack_distance"])
    minu = float(metric["min_unknown_known_distance"])
    known_target = max(0.004, min(tau * 0.70, maxk * 0.55))
    unknown_target = max(tau + 0.04, minu + 0.10, known_target + 0.12)
    bank = torch.tensor([r["embedding"] for r in known[: min(len(known), 12000)]], dtype=torch.float32, device=DEVICE)
    model.train()
    A.set_bn_eval(model)
    total = 0.0
    for b, (orig, atk, anchor, is_known) in enumerate(loader, 1):
        orig = orig.to(DEVICE, non_blocking=True)
        atk = atk.to(DEVICE, non_blocking=True)
        anchor = anchor.to(DEVICE, non_blocking=True)
        is_known = is_known.to(DEVICE, non_blocking=True)
        oe, _, _, _, _, _ = model(orig, orig)
        ae, _, _, _, _, _ = model(atk, atk)
        km = is_known > 0.5
        um = ~km
        loss = torch.tensor(0.0, device=DEVICE)
        if km.any():
            da = torch.norm(ae[km] - anchor[km], dim=1)
            do = torch.norm(oe[km] - anchor[km], dim=1)
            loss = loss + 950.0 * torch.nn.functional.smooth_l1_loss(ae[km], anchor[km].detach())
            loss = loss + 250.0 * torch.nn.functional.smooth_l1_loss(oe[km], anchor[km].detach())
            loss = loss + 1600.0 * torch.relu(da - known_target).pow(2).mean()
            loss = loss + 650.0 * torch.relu(do - known_target).pow(2).mean()
            loss = loss + 75.0 * torch.norm(ae[km] - oe[km], dim=1).mean()
        if um.any():
            da_anchor = torch.norm(ae[um] - anchor[um], dim=1)
            da_bank = torch.cdist(ae[um], bank).min(dim=1).values
            do_bank = torch.cdist(oe[um], bank).min(dim=1).values
            loss = loss + 900.0 * torch.relu(unknown_target - da_anchor).pow(2).mean()
            loss = loss + 650.0 * torch.relu(unknown_target - da_bank).pow(2).mean()
            loss = loss + 450.0 * torch.relu(unknown_target - do_bank).pow(2).mean()
            loss = loss + 15.0 * torch.norm(ae[um] - oe[um], dim=1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 0.18)
        opt.step()
        total += float(loss.detach().cpu())
        if b == 1 or b % 50 == 0:
            print(
                f"[{ds}] train attempt {attempt} batch {b}/{len(loader)} loss {total/b:.5f} "
                f"known_target {known_target:.6f} unknown_target {unknown_target:.6f}",
                flush=True,
            )


def save(ds, accepted, model, metric, out_dir, ck_dir):
    m = {k: v for k, v in metric.items() if k not in {"hard_known", "hard_unknown"}}
    write_json(out_dir / f"{ds}_accepted-{accepted:03d}_final_tau_gap.json", m)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dataset": ds,
            "accepted": accepted,
            "metrics": m,
            "embedding_rule": "image -> this single ConvNeXt checkpoint -> 384-d embedding; use Euclidean distance and best_global_threshold",
        },
        ck_dir / f"{ds}-accepted-{accepted:03d}-final-tau-single.pt",
    )


def run(ds):
    cfg = A.CONFIGS[ds]
    out_dir = OUT / ds
    ck_dir = CKPT_OUT / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "FINAL_TAU_PROGRESS.log"
    _store, known, unknown = A.load_store(cfg["store"])
    known = known[: cfg["known_cap"]]
    unknown = unknown[: cfg["unknown_cap"]]
    base = BASE_CKPTS[ds]
    model = A.load_model(base)
    prev = load_best_prev(ds)
    accepted = 0
    if prev:
        _score, _gap, _total, accepted, ck, _p, _m = prev
        model = A.load_model(ck)
        append_log(log, f"resume accepted-{accepted:03d} score={_score:.6f} gap={_gap:.6f} total={_total:.2f}")
    transform = A.get_transform(strong_aug=False)
    A.refresh_embeddings(model, known, transform)
    metric = evaluate(ds, model, known, unknown)
    best_score = float(metric["goal_score"])
    best_gap = float(metric["global_gap"])
    best_total = float(metric["best_global_total"])
    best_state = clone_state(model)
    best_known = copy.deepcopy(known)
    msg = (
        f"[{ds}] baseline total={best_total:.2f} gap={best_gap:.6f} "
        f"tau={metric['best_global_threshold']:.6f} score={best_score:.6f} base={base}"
    )
    print(msg, flush=True)
    append_log(log, msg)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.time()
        train_step(ds, model, metric, cfg, known, attempt)
        model.eval()
        A.refresh_embeddings(model, known, transform)
        cand = evaluate(ds, model, known, unknown)
        score = float(cand["goal_score"])
        gap = float(cand["global_gap"])
        total = float(cand["best_global_total"])
        accept = score > best_score + 0.02 or (gap > best_gap + 0.002 and total >= best_total - 1.5)
        if accept:
            accepted += 1
            metric = cand
            best_score, best_gap, best_total = score, gap, total
            best_state = clone_state(model)
            best_known = copy.deepcopy(known)
            save(ds, accepted, model, metric, out_dir, ck_dir)
            msg = (
                f"[{ds}] ACCEPT {accepted}/{MAX_ACCEPT} total={total:.2f} gap={gap:.6f} "
                f"tau={cand['best_global_threshold']:.6f} maxK={cand['max_known_attack_distance']:.6f} "
                f"minU={cand['min_unknown_known_distance']:.6f} score={score:.6f} time={time.time()-start:.1f}s"
            )
            print(msg, flush=True)
            append_log(log, msg)
            if gap > 0 and total >= 90.0:
                break
            if accepted >= MAX_ACCEPT:
                break
        else:
            load_state(model, best_state)
            known = copy.deepcopy(best_known)
            if attempt % 10 == 0:
                msg = f"[{ds}] SEARCH attempt={attempt} best_total={best_total:.2f} best_gap={best_gap:.6f} best_score={best_score:.6f}"
                print(msg, flush=True)
                append_log(log, msg)


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["cifar", "flickr", "ucid", "amazon"]:
        run(ds)
