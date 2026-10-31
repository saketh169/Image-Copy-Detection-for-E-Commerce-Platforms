#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/saketh/Codebase")

import multi_split_global_tau_generalized as M
from models.model_convnext_metric import SiameseConvNeXtBMetric


ROOT = Path("/home/saketh/json/final_named_artifacts_20260601")
OUT = ROOT / "generalized_gap_guard_diagonal_search_20260717"
CKPT_OUT = Path("/home/saketh/Codebase/checkpoints/generalized_gap_guard_diagonal_search_20260717")
SPLIT_CACHE = OUT / "explicit_eval_splits"

DATASETS = {
    "cifar": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/fresh_bank_metric_layer_search_20attacks_20260715/cifar/cifar-fresh-bank-metric-layer-best.pt"),
        "q_eval": [(100, 20260529), (100, 20260620), (100, 20260702), (200, 20260620)],
        "orig_u_floor": 92.0,
        "total_floor": 88.0,
        "side_floor": 84.0,
    },
    "flickr": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/balanced_attack_shape_diagonal_search_20260717/flickr/flickr-attack-shape-balanced-accepted-002.pt"),
        "q_eval": [(100, 20260529), (100, 20260620), (100, 20260702), (200, 20260529)],
        "orig_u_floor": 88.0,
        "total_floor": 84.0,
        "side_floor": 80.0,
        "disable_resume": True,
    },
    "ucid": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/balanced_attack_shape_diagonal_search_20260717/ucid/ucid-attack-shape-balanced-accepted-003.pt"),
        "q_eval": [(100, 20260702), (100, 20260620), (100, 20260529), (200, 20260702)],
        "orig_u_floor": 98.0,
        "total_floor": 80.0,
        "side_floor": 78.0,
    },
    "amazon": {
        "ckpt": Path("/home/saketh/Codebase/checkpoints/balanced_attack_shape_diagonal_search_20260717/amazon/amazon-attack-shape-balanced-accepted-003.pt"),
        "q_eval": [(100, 20260529), (100, 20260620), (100, 20260702)],
        "orig_u_floor": 82.0,
        "total_floor": 78.0,
        "side_floor": 76.0,
        "disable_resume": True,
    },
}

EXCLUDE = {"crop40", "bg_color_change", "heavy_bright", "bright", "rotate40", "flip_v"}
ATTACK_NAMES = [a[0] if isinstance(a, (tuple, list)) else a for a in M.G.ATTACKS]
KEEP = [i for i, a in enumerate(ATTACK_NAMES) if a not in EXCLUDE]
KEEP_NAMES = [ATTACK_NAMES[i] for i in KEEP]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(path: Path):
    ck = torch.load(path, map_location=DEVICE)
    model = SiameseConvNeXtBMetric().to(DEVICE)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model


def norm_w(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
    w = np.clip(w, 1e-7, 1e7)
    return (w / max(float(w.mean()), 1e-12)).astype(np.float32)


def filter_attacks(arr: np.ndarray, q: int) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([arr[i * q : (i + 1) * q] for i in KEEP], axis=0).astype(np.float32))


def split_cache_path(ds: str, ckpt: Path, q: int, seed: int) -> Path:
    return SPLIT_CACHE / ds / f"{ckpt.stem}_q{q}_seed{seed}_20attacks.npz"


def collect_one(ds: str, ckpt: Path, q: int, seed: int):
    ep = split_cache_path(ds, ckpt, q, seed)
    if ep.exists():
        z = np.load(ep, allow_pickle=False)
        print(f"[{ds}] explicit_eval_hit q={q} seed={seed} {ep}", flush=True)
        return {
            "q": int(z["q"]),
            "seed": seed,
            "bank": z["bank"].astype(np.float32, copy=False),
            "known": z["known"].astype(np.float32, copy=False),
            "unknown": z["unknown"].astype(np.float32, copy=False),
        }
    old_q = dict(M.Q_MAX)
    old_s = {k: list(v) for k, v in M.SPLIT_SEEDS.items()}
    old_loader = M.A.load_model
    try:
        M.Q_MAX[ds] = q
        M.SPLIT_SEEDS[ds] = [seed]
        M.A.load_model = load_model
        bank, splits = M.collect_splits(ds, ckpt)
        split = splits[0]
        aq = int(split["q"])
        item = {
            "q": aq,
            "seed": seed,
            "bank": bank.astype(np.float32),
            "known": filter_attacks(split["known_query"], aq),
            "unknown": filter_attacks(split["unknown_query"], aq),
        }
        ep.parent.mkdir(parents=True, exist_ok=True)
        tmp = ep.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, q=np.asarray(aq, dtype=np.int32), bank=item["bank"], known=item["known"], unknown=item["unknown"])
        tmp.replace(ep)
        print(f"[{ds}] explicit_eval_write q={q} seed={seed} {ep}", flush=True)
        return item
    finally:
        M.A.load_model = old_loader
        M.Q_MAX.update(old_q)
        M.SPLIT_SEEDS.update(old_s)


@torch.no_grad()
def nearest(bank: np.ndarray, query: np.ndarray, weight: np.ndarray, qchunk=96, bchunk=4096):
    wt = torch.from_numpy(norm_w(weight)).float().to(DEVICE).reshape(1, 1, -1)
    bt = torch.from_numpy(np.ascontiguousarray(bank)).float().to(DEVICE)
    outs = []
    for i in range(0, len(query), qchunk):
        qt = torch.from_numpy(np.ascontiguousarray(query[i:i + qchunk])).float().to(DEVICE)
        best = torch.full((qt.shape[0],), float("inf"), device=DEVICE)
        for j in range(0, len(bank), bchunk):
            bb = bt[j:j + bchunk]
            vals = ((qt[:, None, :] - bb[None, :, :]) ** 2 * wt).sum(2).min(1).values
            best = torch.minimum(best, vals)
        outs.append(torch.sqrt(torch.clamp(best, min=0)).cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


def metrics_from_dist(kd: np.ndarray, ud: np.ndarray, q: int, tau: float):
    kdup = kd <= tau
    udup = ud <= tau
    rows, ks, us, ts, gaps = [], [], [], [], []
    for ai, name in enumerate(KEEP_NAMES):
        st, en = ai * q, (ai + 1) * q
        k = float(kdup[st:en].mean() * 100.0)
        u = float((~udup[st:en]).mean() * 100.0)
        t = (k + u) / 2.0
        rows.append({"attack": name, "known": round(k, 6), "unknown": round(u, 6), "total": round(t, 6), "gap": round(k - u, 6)})
        ks.append(k); us.append(u); ts.append(t)
        if name != "original":
            gaps.append(abs(k - u))
    return {
        "total": round(float(np.mean(ts)), 6),
        "known": round(float(np.mean(ks)), 6),
        "unknown": round(float(np.mean(us)), 6),
        "origK": round(float(ks[0]), 6),
        "origU": round(float(us[0]), 6),
        "minAtk": round(float(np.min(ts)), 6),
        "minK": round(float(np.min(ks)), 6),
        "minU": round(float(np.min(us)), 6),
        "maxGapNonOriginal": round(float(max(gaps)), 6),
        "avgGapNonOriginal": round(float(np.mean(gaps)), 6),
        "rows": rows,
    }


def eval_weight(evalsets: list[dict], w: np.ndarray, cfg: dict):
    dsets = []
    all_d = []
    for e in evalsets:
        kd = nearest(e["bank"], e["known"], w)
        ud = nearest(e["bank"], e["unknown"], w)
        dsets.append((e, kd, ud))
        all_d.append(kd); all_d.append(ud)
    vals = np.unique(np.concatenate(all_d))
    if len(vals) > 20000:
        vals = np.unique(np.quantile(vals, np.linspace(0, 1, 20000)))
    best = None
    for tau in vals:
        mets = []
        for e, kd, ud in dsets:
            m = metrics_from_dist(kd, ud, int(e["q"]), float(tau))
            m["q"] = int(e["q"]); m["seed"] = int(e["seed"])
            mets.append(m)
        agg = aggregate(mets, cfg)
        if best is None or agg["score"] > best["aggregate"]["score"]:
            best = {"tau": float(tau), "sets": mets, "aggregate": agg}
    return best


def aggregate(mets: list[dict], cfg: dict):
    worst_max_gap = max(float(m["maxGapNonOriginal"]) for m in mets)
    mean_avg_gap = float(np.mean([m["avgGapNonOriginal"] for m in mets]))
    mean_total = float(np.mean([m["total"] for m in mets]))
    mean_known = float(np.mean([m["known"] for m in mets]))
    mean_unknown = float(np.mean([m["unknown"] for m in mets]))
    min_origk = min(float(m["origK"]) for m in mets)
    min_origu = min(float(m["origU"]) for m in mets)
    min_mink = min(float(m["minK"]) for m in mets)
    min_minu = min(float(m["minU"]) for m in mets)
    score = 0.0
    score -= worst_max_gap * 60.0
    score -= mean_avg_gap * 25.0
    score -= abs(mean_known - mean_unknown) * 6.0
    score += mean_total * 0.35 + min_mink * 0.45 + min_minu * 0.35
    score -= max(0.0, 100.0 - min_origk) * 1000.0
    # OrigU is not a separate rule at query time, but it is the clearest signal
    # that unknown originals are still too close to the known bank. Push it
    # toward 100 while still allowing useful intermediate candidates.
    score -= max(0.0, 100.0 - min_origu) * 180.0
    score -= max(0.0, float(cfg["orig_u_floor"]) - min_origu) * 100.0
    score -= max(0.0, float(cfg["total_floor"]) - mean_total) * 50.0
    score -= max(0.0, float(cfg["side_floor"]) - mean_known) * 50.0
    score -= max(0.0, float(cfg["side_floor"]) - mean_unknown) * 50.0
    return {
        "score": float(score),
        "meanTotal": round(mean_total, 6),
        "meanKnown": round(mean_known, 6),
        "meanUnknown": round(mean_unknown, 6),
        "minOrigK": round(min_origk, 6),
        "minOrigU": round(min_origu, 6),
        "minK": round(min_mink, 6),
        "minU": round(min_minu, 6),
        "worstMaxGap": round(worst_max_gap, 6),
        "meanAvgGap": round(mean_avg_gap, 6),
    }


def nearest_with_idx(bank: np.ndarray, query: np.ndarray, weight: np.ndarray, qchunk=96, bchunk=4096):
    wt = torch.from_numpy(norm_w(weight)).float().to(DEVICE).reshape(1, 1, -1)
    bt = torch.from_numpy(np.ascontiguousarray(bank)).float().to(DEVICE)
    outs_d, outs_i = [], []
    for i in range(0, len(query), qchunk):
        qt = torch.from_numpy(np.ascontiguousarray(query[i:i + qchunk])).float().to(DEVICE)
        best = torch.full((qt.shape[0],), float("inf"), device=DEVICE)
        best_i = torch.full((qt.shape[0],), -1, dtype=torch.long, device=DEVICE)
        for j in range(0, len(bank), bchunk):
            bb = bt[j:j + bchunk]
            vals, idx = ((qt[:, None, :] - bb[None, :, :]) ** 2 * wt).sum(2).min(1)
            mask = vals < best
            best = torch.where(mask, vals, best)
            best_i = torch.where(mask, idx + j, best_i)
        outs_d.append(torch.sqrt(torch.clamp(best, min=0)).cpu().numpy())
        outs_i.append(best_i.cpu().numpy())
    return np.concatenate(outs_d).astype(np.float32), np.concatenate(outs_i).astype(np.int64)


def _mean_diff(query, bank, idx, fallback_slice):
    if len(query) == 0:
        query = fallback_slice
        idx = np.zeros((len(query),), dtype=np.int64)
    ref = bank[idx]
    return np.mean((query - ref) ** 2, axis=0).astype(np.float32) + 1e-9


def dim_stats(e, w, cfg):
    # Build directions from the actual final rule: nearest known-bank for both sides.
    kd, ki = nearest_with_idx(e["bank"], e["known"], w)
    ud, ui = nearest_with_idx(e["bank"], e["unknown"], w)
    one = eval_weight([e], w, cfg)
    tau = float(one["tau"])
    rows = one["sets"][0]["rows"]
    q = int(e["q"])
    n = e["bank"].shape[1]
    sep = np.zeros(n, dtype=np.float32)   # increase these dims to push close unknowns away
    pull = np.zeros(n, dtype=np.float32)  # decrease these dims to pull hard knowns closer
    balanced = np.zeros(n, dtype=np.float32)
    for ai, row in enumerate(rows):
        name = row["attack"]
        st, en = ai * q, (ai + 1) * q
        kq = e["known"][st:en]
        uq = e["unknown"][st:en]
        kidx = ki[st:en]
        uidx = ui[st:en]
        kgood = kd[st:en] <= tau
        ugoed = ud[st:en] > tau
        kfail = ~kgood
        ufail = ~ugoed
        kdiff = _mean_diff(kq[kfail], e["bank"], kidx[kfail], kq)
        udiff = _mean_diff(uq[ufail], e["bank"], uidx[ufail], uq)
        allk = _mean_diff(kq, e["bank"], kidx, kq)
        allu = _mean_diff(uq, e["bank"], uidx, uq)
        gap = float(row["known"] - row["unknown"])
        if name == "original":
            # Protect originals strongly: unknown originals that are too close get separation,
            # known originals that are hard get pull, but do not let originals dominate all attacks.
            sep += 3.0 * max(0.0, 100.0 - float(row["unknown"])) * udiff
            pull += 3.0 * max(0.0, 100.0 - float(row["known"])) * kdiff
            continue
        if gap > 3.0:
            sep += gap * udiff
            pull += 0.15 * gap * allk
        elif gap < -3.0:
            pull += (-gap) * kdiff
            sep += 0.15 * (-gap) * allu
        else:
            balanced += (allu + allk) * 0.2
        if float(row["known"]) < 85.0:
            pull += (85.0 - float(row["known"])) * kdiff
        if float(row["unknown"]) < 85.0:
            sep += (85.0 - float(row["unknown"])) * udiff
    eps = 1e-8
    sep_dir = norm_w((sep + eps) / (pull + balanced + eps))
    pull_dir = norm_w((sep + balanced + eps) / (pull + eps))
    inverse_pull = norm_w(1.0 / (pull + eps))
    inverse_sep = norm_w(1.0 / (sep + eps))
    ratio = norm_w(sep_dir / np.maximum(pull_dir, eps))
    return {
        "sep_dir": sep_dir,
        "pull_dir": pull_dir,
        "inverse_pull": inverse_pull,
        "inverse_sep": inverse_sep,
        "ratio": ratio,
        "tau": tau,
        "rows": rows,
    }


def candidates(base, stats, rng):
    n = len(base)
    sep_dir = stats["sep_dir"]
    inverse_pull = stats["inverse_pull"]
    ratio = stats["ratio"]
    yield "identity", base
    # Deterministic useful moves first: gap reduction without wild collapse.
    for a in [0.03, 0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.50, 0.70]:
        yield f"gap_sep_a{a}", norm_w(base * np.power(sep_dir, a).astype(np.float32))
        yield f"gap_pull_a{a}", norm_w(base * np.power(inverse_pull, a).astype(np.float32))
        yield f"gap_dual_a{a}", norm_w(base * np.power(sep_dir, a).astype(np.float32) * np.power(inverse_pull, a * 0.75).astype(np.float32))
        yield f"gap_ratio_a{a}", norm_w(base * np.power(ratio, a).astype(np.float32))
    # Old-rule family that gave earlier low raw-gap rows: hard-ratio, inverse-hard-known, whiten+clip.
    for a in [0.13, 0.22, 0.35, 0.50, 0.75]:
        for c in [1.25, 2.0, 3.0, 4.0, 8.0, 16.0]:
            hard_ratio = norm_w(np.clip(ratio, 1.0 / c, c))
            inv_hard = norm_w(np.clip(inverse_pull, 1.0 / c, c))
            all_ratio = norm_w(np.clip(sep_dir * inverse_pull, 1.0 / c, c))
            yield f"old_hard_ratio_a{a}_c{c}", norm_w(base * np.power(hard_ratio, a).astype(np.float32))
            yield f"old_inverse_hard_known_a{a}_c{c}", norm_w(base * np.power(inv_hard, a).astype(np.float32))
            yield f"old_all_ratio_a{a}_c{c}", norm_w(base * np.power(all_ratio, a).astype(np.float32))
    whiten = norm_w((ratio - float(np.mean(ratio))) / (float(np.std(ratio)) + 1e-6) + 1.0)
    whiten = norm_w(np.clip(whiten, 0.05, 20.0))
    for a in [0.13, 0.22, 0.35, 0.50]:
        for c in [1.25, 2.0, 4.0, 8.0, 16.0]:
            wr = norm_w(np.clip(whiten, 1.0 / c, c))
            yield f"old_whiten_ratio_a{a}_c{c}", norm_w(base * np.power(wr, a).astype(np.float32))
    order_sep = np.argsort(sep_dir)
    order_pull = np.argsort(inverse_pull)
    for rep in range(1000000):
        w = base.copy()
        mode = rng.choice(["sep", "pull", "dual", "ratio", "trim_sep", "trim_pull"])
        if mode in ["sep", "dual"]:
            a = float(rng.choice([0.025, 0.04, 0.065, 0.10, 0.16, 0.24, 0.34, 0.48]))
            w *= np.power(sep_dir, a).astype(np.float32)
        if mode in ["pull", "dual"]:
            a = float(rng.choice([0.025, 0.04, 0.065, 0.10, 0.16, 0.24, 0.34, 0.48]))
            w *= np.power(inverse_pull, a).astype(np.float32)
        if mode == "ratio":
            a = float(rng.choice([0.03, 0.06, 0.10, 0.18, 0.30, 0.45, 0.65]))
            w *= np.power(ratio, a).astype(np.float32)
        if mode == "trim_sep":
            frac = float(rng.choice([0.02, 0.04, 0.07, 0.12, 0.18, 0.25]))
            k = max(1, int(n * frac))
            w[order_sep[-k:]] *= float(rng.choice([1.05, 1.10, 1.18, 1.30, 1.55]))
            w[order_sep[:k]] *= float(rng.choice([0.65, 0.75, 0.85, 0.93]))
        if mode == "trim_pull":
            frac = float(rng.choice([0.02, 0.04, 0.07, 0.12, 0.18, 0.25]))
            k = max(1, int(n * frac))
            w[order_pull[-k:]] *= float(rng.choice([1.05, 1.10, 1.18, 1.30, 1.55]))
            w[order_pull[:k]] *= float(rng.choice([0.65, 0.75, 0.85, 0.93]))
        if rng.random() < 0.35:
            w *= np.exp(rng.normal(0, float(rng.choice([0.002, 0.005, 0.012, 0.025])), n)).astype(np.float32)
        yield f"gap_balanced_rep{rep}_{mode}", norm_w(w)


def save_ckpt(ds, cfg, w, result, rule, acc):
    ck = torch.load(cfg["ckpt"], map_location="cpu")
    sd = copy.deepcopy(ck["model_state_dict"])
    sd["metric_scale"] = sd["metric_scale"].float() * torch.tensor(np.sqrt(norm_w(w)), dtype=torch.float32)
    path = CKPT_OUT / ds / f"{ds}-generalized-gap-accepted-{acc:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": "SiameseConvNeXtBMetric",
        "dataset": ds,
        "model_state_dict": sd,
        "global_tau": float(result["tau"]),
        "source_checkpoint": str(cfg["ckpt"]),
        "rule": rule,
        "aggregate": result["aggregate"],
        "sets": result["sets"],
        "kept_attacks": KEEP_NAMES,
        "excluded_attacks": sorted(EXCLUDE),
        "distance_rule": "generalized guard: every image nearest known bank; accept only if multi-q/multi-seed gap improves",
    }, path)
    return str(path)


def accept(old, new, cfg):
    o = old["aggregate"]
    n = new["aggregate"]
    if n["minOrigK"] < 99.99 or n["minOrigU"] < cfg["orig_u_floor"]:
        return False
    if n["meanTotal"] < max(float(cfg["total_floor"]), o["meanTotal"] - 3.5):
        return False
    if n["meanKnown"] < max(float(cfg["side_floor"]) - 2.0, o["meanKnown"] - 4.0):
        return False
    if n["meanUnknown"] < max(float(cfg["side_floor"]) - 2.0, o["meanUnknown"] - 4.0):
        return False
    if n["minK"] < max(20.0, o["minK"] - 8.0) or n["minU"] < max(50.0, o["minU"] - 8.0):
        return False
    if n["worstMaxGap"] < o["worstMaxGap"] - 0.5:
        return True
    if n["worstMaxGap"] <= o["worstMaxGap"] and n["meanAvgGap"] < o["meanAvgGap"] - 0.25:
        return True
    if n["minOrigU"] > o["minOrigU"] + 0.5 and n["worstMaxGap"] <= o["worstMaxGap"] + 3.0 and n["score"] > o["score"] - 50.0:
        return True
    if n["worstMaxGap"] <= o["worstMaxGap"] + 0.25 and n["score"] > o["score"] + 5.0:
        return True
    return False


def prefilter_accept(old, new, cfg):
    o = old["aggregate"]
    n = new["aggregate"]
    if n["minOrigK"] < 99.99 or n["minOrigU"] < cfg["orig_u_floor"]:
        return False
    if n["meanTotal"] < max(float(cfg["total_floor"]) - 1.0, o["meanTotal"] - 5.0):
        return False
    if n["meanKnown"] < max(float(cfg["side_floor"]) - 3.0, o["meanKnown"] - 6.0):
        return False
    if n["meanUnknown"] < max(float(cfg["side_floor"]) - 3.0, o["meanUnknown"] - 6.0):
        return False
    if n["minK"] < max(15.0, o["minK"] - 10.0) or n["minU"] < max(45.0, o["minU"] - 10.0):
        return False
    return n["worstMaxGap"] <= o["worstMaxGap"] + 1.0 or n["meanAvgGap"] < o["meanAvgGap"] - 0.10 or n["score"] > o["score"] + 2.0


def load_named_weight_override(ds: str, rule_prefix: str):
    roots = [
        ROOT / "fresh_bank_metric_layer_search_20attacks_20260715" / ds,
        ROOT / "fresh_bank_metric_layer_search_20260715" / ds,
        ROOT / "nearest_bank_diagonal_search" / ds,
    ]
    for root in roots:
        if not root.exists():
            continue
        for pth in root.glob("*.jsonl"):
            try:
                for line in pth.read_text(errors="replace").splitlines():
                    if rule_prefix not in line:
                        continue
                    obj = json.loads(line)
                    w = obj.get("weight") or obj.get("weights") or obj.get("diagonal_weight") or obj.get("w")
                    if w is not None:
                        print("WEIGHT_OVERRIDE", ds, rule_prefix, pth, flush=True)
                        return np.asarray(w, dtype=np.float32)
                    wp = obj.get("weight_path") or obj.get("weights_path") or obj.get("diagonal_weight_path")
                    if wp and Path(wp).exists():
                        print("WEIGHT_OVERRIDE", ds, rule_prefix, wp, flush=True)
                        return np.load(wp).astype(np.float32)
            except Exception as exc:
                print("WEIGHT_OVERRIDE_SKIP", pth, repr(exc), flush=True)
    return None


def latest_relative_weight(ds: str, cfg: dict, dim: int):
    prev = sorted((CKPT_OUT / ds).glob(f"{ds}-generalized-gap-accepted-*.pt"))
    if not prev:
        return np.ones(dim, dtype=np.float32), None
    latest = prev[-1]
    try:
        base_ck = torch.load(cfg["ckpt"], map_location="cpu")
        new_ck = torch.load(latest, map_location="cpu")
        base_scale = base_ck["model_state_dict"].get("metric_scale")
        new_scale = new_ck["model_state_dict"].get("metric_scale")
        if base_scale is None or new_scale is None:
            return np.ones(dim, dtype=np.float32), str(latest)
        base_scale = base_scale.detach().float().cpu().numpy().reshape(-1)
        new_scale = new_scale.detach().float().cpu().numpy().reshape(-1)
        if len(base_scale) != dim or len(new_scale) != dim:
            return np.ones(dim, dtype=np.float32), str(latest)
        rel = (new_scale / np.maximum(base_scale, 1e-12)) ** 2
        return norm_w(rel.astype(np.float32)), str(latest)
    except Exception as exc:
        print("RESUME_WEIGHT_FAIL", ds, latest, repr(exc), flush=True)
        return np.ones(dim, dtype=np.float32), str(latest)


def run(ds: str, max_candidates: int):
    cfg = dict(DATASETS[ds])
    OUT.mkdir(parents=True, exist_ok=True)
    print("START", ds, "ckpt", cfg["ckpt"], "eval", cfg["q_eval"], "device", DEVICE, flush=True)
    evalsets = [collect_one(ds, cfg["ckpt"], q, seed) for q, seed in cfg["q_eval"]]
    prefilter_sets = [evalsets[0]]
    base_dim = evalsets[0]["bank"].shape[1]
    if cfg.get("disable_resume"):
        base, resume_ckpt = np.ones(base_dim, dtype=np.float32), None
    else:
        base, resume_ckpt = latest_relative_weight(ds, cfg, base_dim)
    best_result = eval_weight(evalsets, base, cfg)
    best_prefilter = eval_weight(prefilter_sets, base, cfg)
    best_w = base.copy()
    if resume_ckpt:
        best_ckpt = resume_ckpt
        base_rule = "resume_relative_weight"
    else:
        best_ckpt = save_ckpt(ds, cfg, best_w, best_result, "baseline_guard", 0)
        base_rule = "baseline_guard"
    best = {"rule": base_rule, "checkpoint": best_ckpt, **best_result}
    print("BASE", ds, best["aggregate"], "tau", best["tau"], "checkpoint", best_ckpt, "resume", bool(resume_ckpt), flush=True)
    stats = dim_stats(evalsets[0], best_w, cfg)
    rng = np.random.default_rng(2026071731 + sum(map(ord, ds)))
    accepted = [best]
    acc = 0
    full_checked = 0
    for ci, (rule, w) in enumerate(candidates(best_w, stats, rng), 1):
        if ci > max_candidates:
            break
        pre = eval_weight(prefilter_sets, w, cfg)
        if not prefilter_accept(best_prefilter, pre, cfg):
            if ci % 100 == 0:
                print("PREFILTERED", ds, ci, "full_checked", full_checked, "best", best["aggregate"], "accepted", acc, "rule", best["rule"], flush=True)
            continue
        full_checked += 1
        result = eval_weight(evalsets, w, cfg)
        if accept(best, result, cfg):
            acc += 1
            best_w = norm_w(w)
            stats = dim_stats(evalsets[0], best_w, cfg)
            best_prefilter = eval_weight(prefilter_sets, best_w, cfg)
            ckpt = save_ckpt(ds, cfg, best_w, result, rule, acc)
            best = {"rule": rule, "checkpoint": ckpt, **result}
            accepted.append(best)
            print("ACCEPT", ds, acc, "cand", ci, result["aggregate"], "tau", result["tau"], "rule", rule, "checkpoint", ckpt, flush=True)
        if ci % 10 == 0:
            print("CHECKED", ds, ci, "full_checked", full_checked, "best", best["aggregate"], "accepted", acc, "rule", best["rule"], flush=True)
    out = OUT / f"{ds}_generalized_gap_guard.json"
    out.write_text(json.dumps({"dataset": ds, "best": best, "accepted": accepted}, indent=2), encoding="utf-8")
    print("DONE", ds, out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    ap.add_argument("--max-candidates", type=int, default=250000)
    args = ap.parse_args()
    run(args.dataset, args.max_candidates)


if __name__ == "__main__":
    main()
