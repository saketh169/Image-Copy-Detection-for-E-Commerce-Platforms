"""
train_convnext.py — Training Loop for Siamese ConvNeXt-B
=========================================================

ConvNeXt-B (Convolution-next, medium size) — Modern CNN approach.

Architecture:
  Input: 224×224×3 image
  Backbone: ConvNeXt-B (from scratch, no pretrained weights)
    - Stage 3 (14×14×384): Local feature extraction
    - Stage 4 (7×7×768): Global feature extraction
  Output: 384-dim embedding (256 global + 128 local)

Key Differences vs ResNet-50:
  1. ModernConv blocks: Depthwise 7×7 instead of bottleneck 3×3
  2. Activation: GELU instead of ReLU (smoother gradients)
  3. Normalization: LayerNorm instead of BatchNorm (more stable)
  4. Higher feature dimensions: 768 global (vs 256 in ResNet)
  5. Inverted bottleneck MLP in each block
  6. More efficient parameter use

Mathematical Pipeline Per Batch:
  ─────────────────────────────────

  1. FORWARD PASS:
     emb1, emb2, g1, g2, l1, l2 = model(img1_batch, img2_batch)
     
     ConvNeXt processing:
       x ← PatchEmbed(x)  [4×4 conv, stride 4: 224 → 56, 3 → 96]
       x ← Stage1(x)      [3 blocks, 56×56×96]
       x ← Stage2(x) ↓    [3 blocks, 28×28×192]
       x ← Stage3(x) ↓    [9 blocks, 14×14×384] ← local_feat
       x ← Stage4(x) ↓    [3 blocks, 7×7×768]   ← global_feat

  2. LOSS COMPUTATION (same as ResNet):
     L = L_global + 0.5 × L_local
     
     L_global = mean(contrastive_loss(global1, global2, label))
     L_local = mean(contrastive_loss(local1, local2, label))
     L = L_global + 0.5 × L_local

  3. BACKWARD PASS:
     ∇ = d(L) / dθ
     θ ← θ - α∇  (Adam update)

  4. PRIORITY UPDATE:
     Same as ResNet: priority[i] = distance[i]

Per-Epoch Decay & Pruning:
  ────────────────────────

  Same as ResNet:
    priority ← priority × (0.9 ^ times_trained)
    if times_trained > 0 AND priority < 0.05: REMOVE

Architecture Advantages vs ResNet-50:
  ✓ More modern design (incorporates ViT ideas)
  ✓ Better parameter efficiency (fewer params, same capacity)
  ✓ LayerNorm more stable than BatchNorm for small batches
  ✓ Depthwise convs reduce computation
  ✓ GELU activation smoother gradients
  ✗ Slightly slower inference (complex ops)
  ✗ May need different LR tuning

Why ConvNeXt-B was chosen:
  - Sweet spot: more efficient than ResNet-50
  - Better optimization landscape (LayerNorm, GELU)
  - Scales well to 815K pairs
  - Still CNN, so maintains inductive bias
"""

import sys
import os

# Add parent directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    mem_fraction_env = os.environ.get("BTP_CUDA_MEM_FRACTION", "").strip()
    if mem_fraction_env:
        try:
            torch.cuda.set_per_process_memory_fraction(float(mem_fraction_env), device=0)
        except Exception:
            pass
import torch.optim as optim
import json
import time
import csv
import math
import subprocess
from pathlib import Path
import random
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib

from config import (
    DEVICE, MANIFEST_PATH, CHECKPOINT_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    DECAY_FACTOR, PRUNE_THRESHOLD, MAX_TRAININGS, MIN_REMAINING_PAIRS,
    )
from models.model_convnext import SiameseConvNeXtB
from loss import DualBranchContrastiveLoss
from dataloader import SiameseDataset, get_transform
from priority_sampler import PrioritySampler
from train_logger import TrainingLogger
from torch.utils.data import DataLoader
from transforms import CopyAttackAugmentor


def _percentile(values, q):
    if not values:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _flatten_panel_values(panel_dists):
    vals = []
    for arr in panel_dists.values():
        if not arr:
            continue
        vals.extend(float(x) for x in arr)
    return vals

def _category_for_path(path):
    parts = Path(path).parts
    if "Data Processed" in parts:
        idx = parts.index("Data Processed")
        if idx + 2 < len(parts):
            return "/".join(parts[idx + 1:-2])
    return None


@lru_cache(maxsize=4)
def _load_known_lookup(lookup_path):
    if not lookup_path or not os.path.exists(lookup_path):
        return {}
    with open(lookup_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        return {}
    lookup = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        if not path:
            continue
        lookup[path] = 1.0 if row.get("known") else 0.0
    return lookup


def _known_flag_for_path(path, known_set):
    lookup_path = os.environ.get("BTP_PATH_KNOWN_JSON", "").strip()
    if lookup_path:
        lookup = _load_known_lookup(lookup_path)
        if lookup:
            return lookup.get(path, 0.0)
    category = _category_for_path(path)
    return 1.0 if category in known_set else 0.0

def _load_image_for_eval(path, transform):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img).unsqueeze(0).to(DEVICE)


def _load_tensor_for_attack(path, transform, attack_name=None):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    if attack_name:
        img = _apply_eval_attack(img, attack_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)

def _eval_attack_names():
    preferred = os.environ.get(
        "BTP_EVAL_ATTACKS",
        os.environ.get(
            "BTP_HARD_POS_COVERAGE_ATTACKS",
            "rotate40,crop40,flip_h,flip_v,resize,watermark_asset,bg_color_change,blur,bright_light,bright,heavy_bright",
        ),
    )
    return [name.strip() for name in preferred.split(",") if name.strip()]


def _stable_eval_attack(path, epoch):
    attack_names = _eval_attack_names()
    if not attack_names:
        return None
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % len(attack_names)
    return attack_names[(int(epoch) + offset) % len(attack_names)]


def _stable_panel_attack(path, attack_names, epoch, slot=0):
    if not attack_names:
        return None
    digest = hashlib.md5(f"{path}|{slot}".encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % len(attack_names)
    return attack_names[(int(epoch) + offset + slot) % len(attack_names)]


def _apply_eval_attack(img_bgr, attack_name):
    if not attack_name:
        return img_bgr
    attack_bank = CopyAttackAugmentor().fixed_attack_bank()
    legacy_aliases = {
        "flip": "flip_h",
        "resize_compress": "resize",
        "watermark_text": "watermark_asset",
        "background_swap": "bg_color_change",
        "crop": "crop40",
        "rotate": "rotate40",
    }
    normalized = legacy_aliases.get(attack_name, attack_name)
    fn = attack_bank.get(normalized)
    if fn is not None:
        return fn(img_bgr.copy())
    return img_bgr


def _embed_paths_for_eval(model, paths, transform, emb_scale, batch_size, apply_attacks=False, epoch=0, attack_name=None):
    import cv2
    tensors = []
    embs = []
    model_was_training = model.training
    model.eval()
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        if attack_name:
            img = _apply_eval_attack(img, attack_name)
        elif apply_attacks:
            img = _apply_eval_attack(img, _stable_eval_attack(path, epoch))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensors.append(transform(img))
        if len(tensors) >= batch_size:
            batch = torch.stack(tensors, dim=0).to(DEVICE, non_blocking=True)
            with torch.no_grad():
                emb, _, _, _, _, _ = model(batch, batch)
            embs.append((emb * emb_scale).detach())
            tensors = []
    if tensors:
        batch = torch.stack(tensors, dim=0).to(DEVICE, non_blocking=True)
        with torch.no_grad():
            emb, _, _, _, _, _ = model(batch, batch)
        embs.append((emb * emb_scale).detach())
    if model_was_training:
        model.train()
    if not embs:
        return None
    return torch.cat(embs, dim=0)


def _harvest_attack_panel():
    panel = os.environ.get(
        "BTP_HARVEST_ATTACK_PANEL",
        "rotate40,crop40,flip_h,flip_v,resize,watermark_asset,bg_color_change,blur,bright_light,bright,heavy_bright",
    )
    attacks = [name.strip() for name in panel.split(",") if name.strip()]
    return attacks if attacks else ["original"]

def _nearest_dists_chunked(query_embeddings, db_embeddings, chunk_size):
    outs = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        chunk = query_embeddings[start:start + chunk_size]
        outs.append(torch.cdist(chunk, db_embeddings).min(dim=1).values)
    return torch.cat(outs, dim=0).detach().cpu().tolist()


def _harvest_attack_panel():
    panel = os.environ.get(
        "BTP_HARVEST_ATTACK_PANEL",
        "rotate40,crop40,flip_h,flip_v,resize,watermark_asset,bg_color_change,blur,bright_light,bright,heavy_bright",
    )
    attacks = [x.strip() for x in panel.split(",") if x.strip()]
    return attacks if attacks else ["original"]


def _nearest_self_excluded_dists_chunked(query_embeddings, db_embeddings, chunk_size):
    outs = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        end = min(start + chunk_size, query_embeddings.shape[0])
        chunk = query_embeddings[start:end]
        dists = torch.cdist(chunk, db_embeddings)
        row_idx = torch.arange(end - start, device=dists.device)
        col_idx = torch.arange(start, end, device=dists.device)
        dists[row_idx, col_idx] = float("inf")
        outs.append(dists.min(dim=1).values)
    return torch.cat(outs, dim=0).detach().cpu().tolist()


def _empty_eval_stats(dup_threshold, not_threshold, known_avg=0.0, unknown_avg=0.0):
    return {
        "known_correct": 0,
        "known_total": 0,
        "unknown_correct": 0,
        "unknown_total": 0,
        "dup_threshold": dup_threshold,
        "not_threshold": not_threshold,
        "known_avg": known_avg,
        "unknown_avg": unknown_avg,
        "known_ambig": 0,
        "unknown_ambig": 0,
    }

def _eval_db_retrieval(model, transform, epoch=0):
    pool_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    if not pool_path or not os.path.exists(pool_path):
        print("  [Eval] Skipping DB retrieval eval (no BTP_EVAL_DB_POOL)")
        return None

    with open(pool_path, "r") as f:
        pool = json.load(f)

    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    use_store_bank = os.environ.get("BTP_EVAL_STORE_BANK", "0") == "1"
    fallback_store_known = os.environ.get("BTP_EVAL_FALLBACK_STORE_KNOWN", "1") == "1"
    max_known = int(os.environ.get("BTP_EVAL_MAX_KNOWN", "1000"))
    max_unknown = int(os.environ.get("BTP_EVAL_MAX_UNKNOWN", "1000"))
    eval_batch_size = int(os.environ.get("BTP_EVAL_BATCH_SIZE", "32"))
    dist_chunk_size = int(os.environ.get("BTP_EVAL_DIST_CHUNK", "256"))
    use_all_known = os.environ.get("BTP_EVAL_USE_ALL_KNOWN", "1") == "1"
    apply_eval_attacks = os.environ.get("BTP_EVAL_APPLY_ATTACKS", "1") == "1"
    compare_attacks_to_self = os.environ.get("BTP_HARVEST_COMPARE_ATTACKS_TO_SELF", "0") == "1"
    harvest_unknown_original_only = os.environ.get("BTP_HARVEST_UNKNOWN_ORIGINAL_ONLY", "0") == "1"
    current_epoch = int(os.environ.get("BTP_CURRENT_EPOCH", "0"))

    dynamic_eval_sampling = os.environ.get("BTP_EVAL_DYNAMIC_SAMPLING", "0") == "1"
    if dynamic_eval_sampling:
        random.seed(int(time.time()) + len(pool) + os.getpid())
    else:
        random.seed(20260413 + len(pool))
    random.shuffle(known)
    if use_store_bank:
        store_path = os.environ.get("BTP_EMBED_STORE_JSON", "")
        store = _load_embedding_store(store_path)
        db_items = []
        for row in store.get("accepted_known", []):
            path = row.get("path")
            if not path:
                continue
            db_items.append({
                "path": path,
                "known": True,
                "category": row.get("category", ""),
                "product": row.get("product"),
                "image_index": row.get("image_index"),
            })
        if dynamic_eval_sampling:
            random.seed(int(time.time()) + len(db_items) + os.getpid())
        else:
            random.seed(20260413 + len(db_items))
        random.shuffle(db_items)
        bank_max = int(os.environ.get("BTP_DB_BANK_MAX", "800"))
        db_items = db_items[:bank_max]
        known_queries = known[:max_known]
        if not known_queries and fallback_store_known:
            known_queries = db_items[:min(max_known, len(db_items))]
        bootstrap_on_empty_store = os.environ.get("BTP_HARVEST_BOOTSTRAP_SELF_ON_EMPTY", "1") == "1"
        if bootstrap_on_empty_store and (not db_items) and known:
            # Epoch-1 bootstrap: use active known pool itself as the temporary bank
            # so both known and unknown can be judged against a meaningful bank.
            db_items = known[:max_known]
    elif use_all_known:
        known_queries = known[:max_known]
        db_items = known_queries
    else:
        db_fraction = float(os.environ.get("BTP_EVAL_DB_FRACTION", "0.7"))
        split_idx = int(len(known) * db_fraction)
        db_items = known[:split_idx]
        known_queries = known[split_idx:][:max_known]
    unknown_queries = unknown[:max_unknown]

    emb_scale = float(os.environ.get("BTP_EMB_SCALE", "1.0"))
    db_paths = [r["path"] for r in db_items]
    known_paths = [r["path"] for r in known_queries]
    unknown_paths = [r["path"] for r in unknown_queries]

    db_embeddings = _embed_paths_for_eval(
        model, db_paths, transform, emb_scale, eval_batch_size, apply_attacks=False, epoch=current_epoch
    )
    known_embeddings = (
        _embed_paths_for_eval(
            model, known_paths, transform, emb_scale, eval_batch_size,
            apply_attacks=apply_eval_attacks, epoch=current_epoch
        )
        if known_paths else None
    )
    unknown_embeddings = (
        _embed_paths_for_eval(
            model, unknown_paths, transform, emb_scale, eval_batch_size,
            apply_attacks=apply_eval_attacks, epoch=current_epoch
        )
        if unknown_paths else None
    )

    dup_thr_env = os.environ.get("BTP_DUP_THRESHOLD")
    not_thr_env = os.environ.get("BTP_NOT_THRESHOLD")
    default_dup = float(dup_thr_env) if dup_thr_env is not None else 0.0
    default_not = float(not_thr_env) if not_thr_env is not None else default_dup
    bootstrap_on_empty_store = os.environ.get("BTP_HARVEST_BOOTSTRAP_SELF_ON_EMPTY", "1") == "1"
    bootstrap_mode = bool(use_store_bank and bootstrap_on_empty_store and store_path and len(_load_embedding_store(store_path).get("accepted_known", [])) == 0 and bool(db_items))

    if (db_embeddings is None or db_embeddings.shape[0] == 0) and not bootstrap_mode:
        print(
            f"  [Eval] DB Retrieval: known 0/0 | unknown 0/0 | "
            f"dup_thr {default_dup:.2f} | not_thr {default_not:.2f} | "
            f"known_avg 0.00 | unknown_avg 0.00 | gray known 0 | gray unknown 0"
        )
        return _empty_eval_stats(default_dup, default_not)

    if known_embeddings is not None:
        if db_embeddings is not None and known_embeddings.shape[0] == db_embeddings.shape[0]:
            known_dists = _nearest_self_excluded_dists_chunked(known_embeddings, db_embeddings, dist_chunk_size)
        else:
            known_dists = _nearest_dists_chunked(known_embeddings, db_embeddings, dist_chunk_size) if db_embeddings is not None else []
    else:
        known_dists = []
    unknown_dists = _nearest_dists_chunked(unknown_embeddings, db_embeddings, dist_chunk_size) if (unknown_embeddings is not None and db_embeddings is not None) else []

    if known_dists:
        known_avg = sum(known_dists) / len(known_dists)
    else:
        known_avg = default_dup

    if unknown_dists:
        unknown_avg = sum(unknown_dists) / len(unknown_dists)
    else:
        unknown_avg = default_not

    gap = unknown_avg - known_avg
    buffer_ratio = float(os.environ.get("BTP_DUAL_THRESHOLD_BUFFER", "0.25"))
    buffer = max(0.0, gap * buffer_ratio)
    dup_q = float(os.environ.get("BTP_DUP_PERCENTILE", "0.80"))
    not_q = float(os.environ.get("BTP_NOT_PERCENTILE", "0.20"))
    min_threshold_gap = float(os.environ.get("BTP_MIN_THRESHOLD_GAP", "0.02"))
    if dup_thr_env is not None and not_thr_env is not None:
        dup_threshold = float(dup_thr_env)
        not_threshold = float(not_thr_env)
    else:
        if known_dists and unknown_dists:
            dup_threshold = _percentile(known_dists, dup_q)
            not_threshold = _percentile(unknown_dists, not_q)
            if not_threshold < dup_threshold:
                relaxed_not = max(not_threshold, known_avg + min_threshold_gap)
                if relaxed_not < dup_threshold:
                    mid = (known_avg + unknown_avg) / 2.0
                    dup_threshold = mid
                    not_threshold = mid
                else:
                    not_threshold = relaxed_not
        elif known_dists:
            dup_threshold = _percentile(known_dists, dup_q)
            not_threshold = dup_threshold + min_threshold_gap
        elif unknown_dists:
            not_threshold = _percentile(unknown_dists, not_q)
            dup_threshold = max(0.0, not_threshold - min_threshold_gap)
        else:
            dup_threshold = default_dup
            not_threshold = default_not

    known_correct = sum(1 for d in known_dists if d <= dup_threshold)
    unknown_correct = sum(1 for d in unknown_dists if d >= not_threshold)
    known_ambig = sum(1 for d in known_dists if dup_threshold < d < not_threshold)
    unknown_ambig = sum(1 for d in unknown_dists if dup_threshold < d < not_threshold)

    panel_attacks = []
    known_panel_dists = {}
    unknown_panel_dists = {}
    self_threshold_floor = float(os.environ.get("BTP_HARVEST_SELF_THRESHOLD", "0.34"))
    self_threshold_q = float(os.environ.get("BTP_HARVEST_SELF_PERCENTILE", "0.95"))
    known_self_threshold = self_threshold_floor
    unknown_self_threshold = self_threshold_floor
    if os.environ.get("BTP_HARVEST_ENABLED", "0") == "1":
        panel_attacks = _harvest_attack_panel()
        for attack_name in panel_attacks:
            if compare_attacks_to_self and attack_name == "original":
                continue
            if known_paths:
                attack_embs = _embed_paths_for_eval(
                    model, known_paths, transform, emb_scale, eval_batch_size,
                    apply_attacks=False, epoch=current_epoch, attack_name=attack_name
                )
                if attack_embs is not None:
                    if compare_attacks_to_self and known_embeddings is not None:
                        known_panel_dists[attack_name] = torch.norm(attack_embs - known_embeddings, dim=1).detach().cpu().tolist()
                    else:
                        known_panel_dists[attack_name] = _nearest_dists_chunked(attack_embs, db_embeddings, dist_chunk_size)
                else:
                    known_panel_dists[attack_name] = []
            if unknown_paths:
                if harvest_unknown_original_only:
                    unknown_panel_dists[attack_name] = list(unknown_dists)
                else:
                    attack_embs = _embed_paths_for_eval(
                        model, unknown_paths, transform, emb_scale, eval_batch_size,
                        apply_attacks=False, epoch=current_epoch, attack_name=attack_name
                    )
                    if attack_embs is not None:
                        if compare_attacks_to_self and unknown_embeddings is not None:
                            unknown_panel_dists[attack_name] = torch.norm(attack_embs - unknown_embeddings, dim=1).detach().cpu().tolist()
                        else:
                            unknown_panel_dists[attack_name] = _nearest_dists_chunked(attack_embs, db_embeddings, dist_chunk_size)
                    else:
                        unknown_panel_dists[attack_name] = []

        if compare_attacks_to_self:
            known_self_vals = _flatten_panel_values(known_panel_dists)
            unknown_self_vals = _flatten_panel_values(unknown_panel_dists)
            if known_self_vals:
                known_self_threshold = max(self_threshold_floor, _percentile(known_self_vals, self_threshold_q))
            if unknown_self_vals:
                unknown_self_threshold = max(self_threshold_floor, _percentile(unknown_self_vals, self_threshold_q))
            elif known_self_vals:
                unknown_self_threshold = known_self_threshold

    # When harvest is enabled, report the same image-level gate that harvest uses:
    # original-to-bank plus attacks-to-self/bank consistency. This keeps eval and
    # accepted-count semantics aligned in both bootstrap and normal bank modes.
    if os.environ.get("BTP_HARVEST_ENABLED", "0") == "1" and panel_attacks:
        known_gate = 0
        unknown_gate = 0
        known_gate_gray = 0
        unknown_gate_gray = 0
        known_margin = float(os.environ.get("BTP_HARVEST_KNOWN_MARGIN", "0.03"))
        unknown_margin = float(os.environ.get("BTP_HARVEST_UNKNOWN_MARGIN", "0.03"))
        unknown_original_only = os.environ.get("BTP_HARVEST_UNKNOWN_ORIGINAL_ONLY", "0") == "1"
        panel_min_passes = max(1, min(len(panel_attacks), int(os.environ.get("BTP_HARVEST_PANEL_MIN_PASSES", "4"))))
        strict_attack_names = [a for a in panel_attacks if a != "original"]

        for idx in range(len(known_queries)):
            orig_dist = float(known_dists[idx]) if idx < len(known_dists) else 0.0
            orig_pass = orig_dist <= (dup_threshold - known_margin)
            attack_passes = 0
            for attack_name in strict_attack_names:
                vals = known_panel_dists.get(attack_name, [])
                if idx >= len(vals):
                    continue
                dist = float(vals[idx])
                limit = known_self_threshold if compare_attacks_to_self else (dup_threshold - known_margin)
                if dist <= limit:
                    attack_passes += 1
            total_passes = attack_passes + (1 if orig_pass else 0)
            if orig_pass and total_passes >= panel_min_passes:
                known_gate += 1
            else:
                if orig_pass or attack_passes > 0:
                    known_gate_gray += 1

        for idx in range(len(unknown_queries)):
            orig_dist = float(unknown_dists[idx]) if idx < len(unknown_dists) else 0.0
            orig_pass = orig_dist >= (not_threshold + unknown_margin)
            if unknown_original_only:
                if orig_pass:
                    unknown_gate += 1
                elif orig_dist > 0:
                    unknown_gate_gray += 1
                continue
            attack_passes = 0
            for attack_name in strict_attack_names:
                vals = unknown_panel_dists.get(attack_name, [])
                if idx >= len(vals):
                    continue
                dist = float(vals[idx])
                ok = dist <= unknown_self_threshold if compare_attacks_to_self else dist >= (not_threshold + unknown_margin)
                if ok:
                    attack_passes += 1
            total_passes = attack_passes + (1 if orig_pass else 0)
            if orig_pass and total_passes >= panel_min_passes:
                unknown_gate += 1
            else:
                if orig_pass or attack_passes > 0:
                    unknown_gate_gray += 1

        known_correct = known_gate
        unknown_correct = unknown_gate
        known_ambig = known_gate_gray
        unknown_ambig = unknown_gate_gray

    print(
        f"  [Eval] DB Retrieval: known {known_correct}/{len(known_dists)} | "
        f"unknown {unknown_correct}/{len(unknown_dists)} | "
        f"dup_thr {dup_threshold:.2f} | not_thr {not_threshold:.2f} | "
        f"known_avg {known_avg:.2f} | unknown_avg {unknown_avg:.2f} | "
        f"gray known {known_ambig} | gray unknown {unknown_ambig}"
    )

    eval_stats = {
        "known_correct": known_correct,
        "known_total": len(known_dists),
        "unknown_correct": unknown_correct,
        "unknown_total": len(unknown_dists),
        "dup_threshold": dup_threshold,
        "not_threshold": not_threshold,
        "known_avg": known_avg,
        "unknown_avg": unknown_avg,
        "known_ambig": known_ambig,
        "unknown_ambig": unknown_ambig,
        "bootstrap_mode": bootstrap_mode,
        "db_items": db_items,
        "known_queries": known_queries,
        "unknown_queries": unknown_queries,
        "known_embeddings": known_embeddings,
        "unknown_embeddings": unknown_embeddings,
        "known_base_dists": known_dists,
        "unknown_base_dists": unknown_dists,
        "known_panel_dists": known_panel_dists,
        "unknown_panel_dists": unknown_panel_dists,
        "known_self_threshold": known_self_threshold,
        "unknown_self_threshold": unknown_self_threshold,
        "panel_attacks": panel_attacks,
        "epoch": epoch,
    }
    if os.environ.get("BTP_HARVEST_ENABLED", "0") == "1":
        eval_stats["harvest_stats"] = _apply_harvest_from_eval(epoch, eval_stats=eval_stats)
    else:
        eval_stats["harvest_stats"] = None
    return eval_stats


def _load_db_pool_rows():
    pool_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    if not pool_path or not os.path.exists(pool_path):
        return None, None, None
    with open(pool_path, "r") as f:
        pool = json.load(f)
    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    db_fraction = float(os.environ.get("BTP_EVAL_DB_FRACTION", "0.7"))
    dynamic_eval_sampling = os.environ.get("BTP_EVAL_DYNAMIC_SAMPLING", "0") == "1"
    if dynamic_eval_sampling:
        random.seed(int(time.time()) + len(pool) + os.getpid())
    else:
        random.seed(20260413 + len(pool))
    random.shuffle(known)
    split_idx = int(len(known) * db_fraction)
    db_items = known[:split_idx]
    return pool, known, db_items


def _build_db_bank(model, transform):
    _, _, db_items = _load_db_pool_rows()
    if not db_items:
        return None
    bank_max = int(os.environ.get("BTP_DB_BANK_MAX", "800"))
    emb_scale = float(os.environ.get("BTP_EMB_SCALE", "1.0"))
    db_items = db_items[:bank_max]
    model_was_training = model.training
    model.eval()
    embs = []
    for row in db_items:
        img = _load_image_for_eval(row["path"], transform)
        if img is None:
            continue
        with torch.no_grad():
            emb, _, _, _, _, _ = model(img, img)
        embs.append((emb * emb_scale).detach())
    if model_was_training:
        model.train()
    if not embs:
        return None
    return torch.cat(embs, dim=0).to(DEVICE)


def _load_embedding_store(path):
    if not path or not os.path.exists(path):
        return {
            "accepted_known": [],
            "accepted_unknown": [],
            "processed_paths": [],
            "candidate_known": {},
            "candidate_unknown": {},
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("accepted_known", [])
    data.setdefault("accepted_unknown", [])
    data.setdefault("processed_paths", [])
    data.setdefault("candidate_known", {})
    data.setdefault("candidate_unknown", {})
    if not isinstance(data.get("candidate_known", {}), dict):
        data["candidate_known"] = {}
    if not isinstance(data.get("candidate_unknown", {}), dict):
        data["candidate_unknown"] = {}
    return data


def _save_embedding_store(path, data):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _snapshot_embedding_store(path, data, epoch):
    if not path or os.environ.get("BTP_STORE_SNAPSHOT_EVERY_EPOCH", "1") != "1":
        return ""
    base = Path(path)
    snap = base.with_name(f"{base.stem}_epoch-{int(epoch)}{base.suffix}")
    os.makedirs(snap.parent, exist_ok=True)
    with open(snap, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(snap)


_PERATTACK_EVAL_ATTACKS = [
    ("original", None),
    ("rotate_m5", "rotate-5"),
    ("rotate_m10", "rotate-10"),
    ("rotate5", "rotate5"),
    ("rotate15", "rotate15"),
    ("rotate30", "rotate30"),
    ("rotate40", "rotate40"),
    ("crop5", "crop5"),
    ("crop10", "crop10"),
    ("crop20", "crop20"),
    ("crop40", "crop40"),
    ("bright_light", "bright_light"),
    ("bright", "bright"),
    ("heavy_bright", "heavy_bright"),
    ("flip_h", "flip_h"),
    ("flip_v", "flip_v"),
    ("flip", "flip"),
    ("resize_compress", "resize_compress"),
    ("resize", "resize"),
    ("watermark_text", "watermark_text"),
    ("watermark_asset", "watermark_asset"),
    ("mix_rotate10_bright", "mix_rotate10_bright"),
    ("mix_crop20_watermark", "mix_crop20_watermark"),
    ("bg_color_change", "bg_color_change"),
    ("blur", "blur"),
    ("contrast", "contrast"),
]

_STRICT10_ATTACK_LABELS = [
    "rotate_m10",
    "rotate40",
    "crop40",
    "heavy_bright",
    "flip_h",
    "flip_v",
    "resize_compress",
    "watermark_asset",
    "blur",
    "bg_color_change",
]


def _best_threshold(known_dists, unknown_dists):
    vals = sorted(set(float(x) for x in (known_dists + unknown_dists)))
    if not vals:
        return 0.0
    cands = [vals[0] - 1e-6]
    cands.extend((vals[i] + vals[i + 1]) / 2.0 for i in range(len(vals) - 1))
    cands.append(vals[-1] + 1e-6)
    best_thr = cands[0]
    best_score = -1
    for thr in cands:
        score = sum(1 for d in known_dists if d <= thr) + sum(1 for d in unknown_dists if d > thr)
        if score > best_score:
            best_score = score
            best_thr = thr
    return float(best_thr)


def _run_perattack_epoch_eval(model, transform, ckpt_path, store_path, epoch):
    if os.environ.get("BTP_PERATTACK_EVAL_EVERY", "1") != "1":
        return None
    if not store_path or not os.path.exists(store_path):
        return None
    sample_known = int(os.environ.get("BTP_PERATTACK_EVAL_SAMPLE_KNOWN", "100"))
    sample_unknown = int(os.environ.get("BTP_PERATTACK_EVAL_SAMPLE_UNKNOWN", "100"))
    allow_small_store = os.environ.get("BTP_PERATTACK_EVAL_ALLOW_SMALL_STORE", "0") == "1"
    eval_batch_size = int(os.environ.get("BTP_PERATTACK_EVAL_BATCH_SIZE", os.environ.get("BTP_EVAL_BATCH_SIZE", "64")))
    dist_chunk = int(os.environ.get("BTP_PERATTACK_EVAL_DIST_CHUNK", os.environ.get("BTP_EVAL_DIST_CHUNK", "128")))
    seed = int(os.environ.get("BTP_PERATTACK_EVAL_SEED", "20260513")) + int(epoch)

    store = _load_embedding_store(store_path)
    accepted_known = [r for r in store.get("accepted_known", []) if r.get("path")]
    accepted_unknown = [r for r in store.get("accepted_unknown", []) if r.get("path")]
    if allow_small_store:
        sample_known = min(sample_known, len(accepted_known))
        sample_unknown = min(sample_unknown, len(accepted_unknown))
    if len(accepted_known) < sample_known or len(accepted_unknown) < sample_unknown or sample_known <= 0 or sample_unknown <= 0:
        print(
            f"  [PerAttackEval] skipped epoch {epoch}: store too small "
            f"(known={len(accepted_known)}, unknown={len(accepted_unknown)})"
        )
        return None

    rng = random.Random(seed)
    known_rows = accepted_known[:]
    unknown_rows = accepted_unknown[:]
    rng.shuffle(known_rows)
    rng.shuffle(unknown_rows)
    known_rows = known_rows[:sample_known]
    unknown_rows = unknown_rows[:sample_unknown]

    bank_paths = [r["path"] for r in accepted_known]
    known_paths = [r["path"] for r in known_rows]
    unknown_paths = [r["path"] for r in unknown_rows]

    bank_embs = _embed_paths_for_eval(model, bank_paths, transform, 1.0, eval_batch_size, apply_attacks=False, epoch=0)
    out = {
        "checkpoint": ckpt_path,
        "store": store_path,
        "store_known_total": len(accepted_known),
        "store_unknown_total": len(accepted_unknown),
        "sample_known": len(known_paths),
        "sample_unknown": len(unknown_paths),
        "per_attack_thresholds": {},
        "attacks": {},
    }

    for label, attack_name in _PERATTACK_EVAL_ATTACKS:
        kemb = _embed_paths_for_eval(
            model, known_paths, transform, 1.0, eval_batch_size,
            apply_attacks=False, epoch=0, attack_name=attack_name
        )
        uemb = _embed_paths_for_eval(
            model, unknown_paths, transform, 1.0, eval_batch_size,
            apply_attacks=False, epoch=0, attack_name=attack_name
        )
        kd = _nearest_dists_chunked(kemb, bank_embs, dist_chunk)
        ud = _nearest_dists_chunked(uemb, bank_embs, dist_chunk)
        thr = _best_threshold(kd, ud)
        k_ok = sum(1 for d in kd if d <= thr)
        u_ok = sum(1 for d in ud if d > thr)
        out["per_attack_thresholds"][label] = thr
        out["attacks"][label] = {
            "known": round((k_ok / max(1, len(kd))) * 100.0, 2),
            "unknown": round((u_ok / max(1, len(ud))) * 100.0, 2),
            "total": round(((k_ok + u_ok) / max(1, len(kd) + len(ud))) * 100.0, 2),
            "threshold": round(thr, 6),
        }

    nonorig = [v["total"] for k, v in out["attacks"].items() if k != "original"]
    strict10 = [out["attacks"][k]["total"] for k in _STRICT10_ATTACK_LABELS if k in out["attacks"]]
    out["summary"] = {
        "avg_non_original_total": round(sum(nonorig) / max(1, len(nonorig)), 3),
        "avg_strict10_total": round(sum(strict10) / max(1, len(strict10)), 3),
    }
    out_path = str(Path(store_path).parent / f"epoch-{int(epoch)}_store_q200_26attacks_perattack.json")
    Path(out_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["out_path"] = out_path
    return out


def _fresh_refill_plan(source_rows, target_total, epoch, preserved_count, label):
    force_coverage = os.environ.get("BTP_FORCE_FULL_COVERAGE", "0") == "1"
    total_epochs = int(os.environ.get("BTP_NUM_EPOCHS", "100"))
    epochs_left = max(1, total_epochs - int(epoch))
    dynamic_min = 0
    if force_coverage and source_rows > 0:
        dynamic_min = math.ceil(source_rows / epochs_left)
    configured_min = int(os.environ.get(f"BTP_FORCE_FRESH_{label}_MIN", "0"))
    fresh_min = max(dynamic_min, configured_min)
    available_slots = max(0, target_total - preserved_count)
    return min(available_slots, fresh_min)


def _current_eval_split(pool):
    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    db_fraction = float(os.environ.get("BTP_EVAL_DB_FRACTION", "0.7"))
    dynamic_eval_sampling = os.environ.get("BTP_EVAL_DYNAMIC_SAMPLING", "0") == "1"
    if dynamic_eval_sampling:
        random.seed(int(time.time()) + len(pool) + os.getpid())
    else:
        random.seed(20260413 + len(pool))
    random.shuffle(known)
    split_idx = int(len(known) * db_fraction)
    db_items = known[:split_idx]
    known_queries = known[split_idx:]
    return db_items, known_queries, unknown


def _ensure_active_eval_pool():
    if os.environ.get("BTP_HARVEST_ENABLED", "0") != "1":
        return
    active_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    source_path = os.environ.get("BTP_EMBED_SOURCE_POOL", "")
    store_path = os.environ.get("BTP_EMBED_STORE_JSON", "")
    if not active_path or not source_path or not os.path.exists(source_path):
        return
    reset_active = os.environ.get("BTP_EMBED_RESET_ACTIVE", "0") == "1"
    if os.path.exists(active_path) and not reset_active:
        return

    target_known = int(os.environ.get("BTP_EMBED_ACTIVE_KNOWN", "1500"))
    target_unknown = int(os.environ.get("BTP_EMBED_ACTIVE_UNKNOWN", "500"))
    with open(source_path, "r", encoding="utf-8") as f:
        source_pool = json.load(f)
    store = _load_embedding_store(store_path)
    existing_known_paths = {item.get("path") for item in store.get("accepted_known", []) if item.get("path")}
    existing_unknown_paths = {item.get("path") for item in store.get("accepted_unknown", []) if item.get("path")}
    processed = set(store.get("processed_paths", []))
    processed.update(existing_known_paths)
    processed.update(existing_unknown_paths)
    known_rows = [r for r in source_pool if r.get("known") and r.get("path") not in processed]
    unknown_rows = [r for r in source_pool if (not r.get("known")) and r.get("path") not in processed]
    random.seed(20260416 + len(source_pool))
    random.shuffle(known_rows)
    random.shuffle(unknown_rows)
    active_pool = known_rows[:target_known] + unknown_rows[:target_unknown]
    os.makedirs(os.path.dirname(active_path), exist_ok=True)
    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(active_pool, f, indent=2)


def _apply_harvest_from_eval(epoch, eval_stats=None):
    if os.environ.get("BTP_HARVEST_ENABLED", "0") != "1":
        return None
    active_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    source_path = os.environ.get("BTP_EMBED_SOURCE_POOL", "")
    store_path = os.environ.get("BTP_EMBED_STORE_JSON", "")
    if not active_path or not source_path or not os.path.exists(active_path) or not os.path.exists(source_path):
        return None

    with open(active_path, "r", encoding="utf-8") as f:
        active_pool = json.load(f)
    with open(source_path, "r", encoding="utf-8") as f:
        source_pool = json.load(f)

    panel_attacks = _harvest_attack_panel()
    compare_attacks_to_self = os.environ.get("BTP_HARVEST_COMPARE_ATTACKS_TO_SELF", "0") == "1"
    panel_min_passes = max(1, int(os.environ.get("BTP_HARVEST_PANEL_MIN_PASSES", "4")))
    panel_min_passes = min(panel_min_passes, len(panel_attacks))

    reuse_eval = eval_stats is not None and eval_stats.get("db_items") is not None
    if reuse_eval:
        db_items = eval_stats.get("db_items") or []
        known_queries = eval_stats.get("known_queries") or []
        unknown_queries = eval_stats.get("unknown_queries") or []
        known_embeddings = eval_stats.get("known_embeddings")
        unknown_embeddings = eval_stats.get("unknown_embeddings")
        known_panel_dists = eval_stats.get("known_panel_dists", {}) or {}
        unknown_panel_dists = eval_stats.get("unknown_panel_dists", {}) or {}
        bootstrap_mode = bool(eval_stats.get("bootstrap_mode"))
        if ((not db_items) and (not bootstrap_mode)) or (not known_queries and not unknown_queries):
            return None
    else:
        return None

    store = _load_embedding_store(store_path)
    processed = set(store.get("processed_paths", []))
    candidate_known = store.get("candidate_known", {})
    candidate_unknown = store.get("candidate_unknown", {})
    existing_known_paths = {item.get("path") for item in store.get("accepted_known", []) if item.get("path")}
    existing_unknown_paths = {item.get("path") for item in store.get("accepted_unknown", []) if item.get("path")}
    if eval_stats is not None:
        dup_threshold = float(eval_stats.get("dup_threshold", 0.0))
        not_threshold = float(eval_stats.get("not_threshold", dup_threshold))
    else:
        dup_threshold = float(os.environ.get("BTP_DUP_THRESHOLD", "0.0"))
        not_threshold = float(os.environ.get("BTP_NOT_THRESHOLD", str(dup_threshold)))
    min_streak = max(1, int(os.environ.get("BTP_HARVEST_MIN_STREAK", "2")))
    known_margin = float(os.environ.get("BTP_HARVEST_KNOWN_MARGIN", "0.03"))
    unknown_margin = float(os.environ.get("BTP_HARVEST_UNKNOWN_MARGIN", "0.03"))
    self_threshold_floor = float(os.environ.get("BTP_HARVEST_SELF_THRESHOLD", str(max(0.0, dup_threshold - known_margin))))
    known_self_threshold = float(eval_stats.get("known_self_threshold", self_threshold_floor)) if eval_stats is not None else self_threshold_floor
    unknown_self_threshold = float(eval_stats.get("unknown_self_threshold", self_threshold_floor)) if eval_stats is not None else self_threshold_floor
    pending_known = 0
    pending_unknown = 0

    accepted_known = []
    kept_known = []
    if known_queries and known_embeddings is not None:
        known_base_dists = eval_stats.get("known_base_dists", []) if eval_stats is not None else []
        bootstrap_mode = bool(eval_stats.get("bootstrap_mode")) if eval_stats is not None else False
        for idx, (row, emb) in enumerate(zip(known_queries, known_embeddings)):
            path = row["path"]
            if path in existing_known_paths:
                candidate_known.pop(path, None)
                continue
            pass_dists = []
            fail_dists = []
            orig_dist = float(known_base_dists[idx]) if idx < len(known_base_dists) else 0.0
            orig_pass = (orig_dist <= (dup_threshold - known_margin))
            strict_attack_names = [a for a in panel_attacks if a != "original"]
            for attack_name in strict_attack_names:
                attack_dists = known_panel_dists.get(attack_name, [])
                if idx >= len(attack_dists):
                    continue
                dist = float(attack_dists[idx])
                limit = known_self_threshold if compare_attacks_to_self else (dup_threshold - known_margin)
                if dist <= limit:
                    pass_dists.append((attack_name, dist))
                else:
                    fail_dists.append((attack_name, dist))
            pass_count = len(pass_dists) + (1 if orig_pass else 0)
            best_dist = min(
                [orig_dist] + [dist for _, dist in pass_dists] or [dist for _, dist in fail_dists] or [0.0]
            )
            if orig_pass and pass_count >= panel_min_passes:
                streak = int(candidate_known.get(path, 0)) + 1
                candidate_known[path] = streak
                if streak >= min_streak:
                    accepted_known.append({
                        "epoch": epoch,
                        "path": path,
                        "category": row.get("category", ""),
                        "known": True,
                        "distance": round(float(best_dist), 6),
                        "embedding": [round(float(x), 6) for x in emb.tolist()],
                        "panel_passes": pass_count,
                        "panel_total": len(panel_attacks),
                        "panel_pass_attacks": (["original"] if orig_pass else []) + [name for name, _ in pass_dists],
                    })
                    processed.add(path)
                    candidate_known.pop(path, None)
                else:
                    pending_known += 1
                    kept_known.append(row)
            else:
                candidate_known.pop(path, None)
                kept_known.append(row)

    accepted_unknown = []
    kept_unknown = []
    if unknown_queries and unknown_embeddings is not None:
        bootstrap_mode = bool(eval_stats.get("bootstrap_mode")) if eval_stats is not None else False
        unknown_original_only = os.environ.get("BTP_HARVEST_UNKNOWN_ORIGINAL_ONLY", "0") == "1"
        unknown_base_dists = eval_stats.get("unknown_base_dists", []) if eval_stats is not None else []
        for idx, (row, emb) in enumerate(zip(unknown_queries, unknown_embeddings)):
            path = row["path"]
            if path in existing_unknown_paths:
                candidate_unknown.pop(path, None)
                continue
            pass_dists = []
            fail_dists = []
            orig_dist = float(unknown_base_dists[idx]) if idx < len(unknown_base_dists) else 0.0
            orig_pass = (orig_dist >= (not_threshold + unknown_margin))
            if unknown_original_only:
                if orig_pass:
                    pass_dists.append(("original", orig_dist))
                else:
                    fail_dists.append(("original", orig_dist))
                pass_count = len(pass_dists)
                panel_total = 1
            else:
                strict_attack_names = [a for a in panel_attacks if a != "original"]
                for attack_name in strict_attack_names:
                    attack_dists = unknown_panel_dists.get(attack_name, [])
                    if idx >= len(attack_dists):
                        continue
                    dist = float(attack_dists[idx])
                    if compare_attacks_to_self:
                        near_enough = dist <= unknown_self_threshold
                    else:
                        near_enough = dist >= (not_threshold + unknown_margin)
                    if near_enough:
                        pass_dists.append((attack_name, dist))
                    else:
                        fail_dists.append((attack_name, dist))
                pass_count = len(pass_dists) + (1 if orig_pass else 0)
                panel_total = len(panel_attacks)
            best_dist = max(
                [orig_dist] + [dist for _, dist in pass_dists] or [dist for _, dist in fail_dists] or [0.0]
            )
            required_unknown_passes = 1 if unknown_original_only else panel_min_passes
            if orig_pass and pass_count >= required_unknown_passes:
                streak = int(candidate_unknown.get(path, 0)) + 1
                candidate_unknown[path] = streak
                if streak >= min_streak:
                    accepted_unknown.append({
                        "epoch": epoch,
                        "path": path,
                        "category": row.get("category", ""),
                        "known": False,
                        "distance": round(float(best_dist), 6),
                        "embedding": [round(float(x), 6) for x in emb.tolist()],
                        "panel_passes": pass_count,
                        "panel_total": panel_total,
                        "panel_pass_attacks": (["original"] if orig_pass else []) + [name for name, _ in pass_dists],
                    })
                    processed.add(path)
                    candidate_unknown.pop(path, None)
                else:
                    pending_unknown += 1
                    kept_unknown.append(row)
            else:
                candidate_unknown.pop(path, None)
                kept_unknown.append(row)

    store["accepted_known"].extend(accepted_known)
    store["accepted_unknown"].extend(accepted_unknown)
    store["processed_paths"] = sorted(processed)
    store["candidate_known"] = candidate_known
    store["candidate_unknown"] = candidate_unknown
    store["last_epoch"] = epoch
    store["meta"] = {
        "last_epoch": epoch,
        "accepted_known_total": len(store["accepted_known"]),
        "accepted_unknown_total": len(store["accepted_unknown"]),
        "last_added_known": len(accepted_known),
        "last_added_unknown": len(accepted_unknown),
        "pending_known_total": len(candidate_known),
        "pending_unknown_total": len(candidate_unknown),
        "harvest_min_streak": min_streak,
        "harvest_known_margin": known_margin,
        "harvest_unknown_margin": unknown_margin,
        "harvest_attack_panel": panel_attacks,
        "harvest_panel_min_passes": panel_min_passes,
    }
    _save_embedding_store(store_path, store)
    _snapshot_embedding_store(store_path, store, epoch)

    if os.environ.get("BTP_HARVEST_SKIP_POOL_REBUILD", "0") == "1":
        return {
            "accepted_known": len(accepted_known),
            "accepted_unknown": len(accepted_unknown),
            "known_query_total": len(known_queries),
            "unknown_query_total": len(unknown_queries),
            "accepted_known_total": len(store["accepted_known"]),
            "accepted_unknown_total": len(store["accepted_unknown"]),
            "pending_known": pending_known,
            "pending_unknown": pending_unknown,
            "processed_now_paths": [row["path"] for row in accepted_known] + [row["path"] for row in accepted_unknown],
            "kept_known_wrong": len(known_queries) - len(accepted_known),
            "kept_unknown_wrong": len(unknown_queries) - len(accepted_unknown),
            "added_known": 0,
            "added_unknown": 0,
            "next_active_known": sum(1 for r in active_pool if r.get("known")),
            "next_active_unknown": sum(1 for r in active_pool if not r.get("known")),
            "next_active_total": len(active_pool),
            "harvest_panel": panel_attacks,
            "harvest_panel_min_passes": panel_min_passes,
            "pool_rebuild_skipped": True,
            "unified_eval_harvest": True,
        }

    target_known = int(os.environ.get("BTP_EMBED_ACTIVE_KNOWN", "1500"))
    target_unknown = int(os.environ.get("BTP_EMBED_ACTIVE_UNKNOWN", "500"))
    keep_db_items = os.environ.get("BTP_KEEP_DB_ITEMS_NEXT_POOL", "0") == "1"
    preserved_db_items = list(db_items) if keep_db_items else []
    random.shuffle(kept_known)
    random.shuffle(kept_unknown)
    processed_only_known = [r for r in source_pool if r.get("known") and r.get("path") not in processed]
    processed_only_unknown = [r for r in source_pool if (not r.get("known")) and r.get("path") not in processed]
    fresh_known_min = _fresh_refill_plan(
        source_rows=len(processed_only_known),
        target_total=target_known,
        epoch=epoch,
        preserved_count=len(preserved_db_items),
        label="KNOWN",
    )
    fresh_unknown_min = _fresh_refill_plan(
        source_rows=len(processed_only_unknown),
        target_total=target_unknown,
        epoch=epoch,
        preserved_count=0,
        label="UNKNOWN",
    )
    keep_known_cap = max(0, target_known - len(preserved_db_items) - fresh_known_min)
    keep_unknown_cap = max(0, target_unknown - fresh_unknown_min)
    kept_known = kept_known[:keep_known_cap]
    kept_unknown = kept_unknown[:keep_unknown_cap]
    active_paths = {r["path"] for r in preserved_db_items + kept_known + kept_unknown}
    source_known = [r for r in source_pool if r.get("known") and r.get("path") not in processed and r.get("path") not in active_paths]
    source_unknown = [r for r in source_pool if (not r.get("known")) and r.get("path") not in processed and r.get("path") not in active_paths]
    random.seed(20260416 + epoch + len(source_pool))
    random.shuffle(source_known)
    random.shuffle(source_unknown)

    current_known_total = len(preserved_db_items) + len(kept_known)
    current_unknown_total = len(kept_unknown)
    add_known = source_known[:max(0, target_known - current_known_total)]
    add_unknown = source_unknown[:max(0, target_unknown - current_unknown_total)]
    next_active = preserved_db_items + kept_known + kept_unknown + add_known + add_unknown
    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(next_active, f, indent=2)

    return {
        "accepted_known": len(accepted_known),
        "accepted_unknown": len(accepted_unknown),
        "known_query_total": len(known_queries),
        "unknown_query_total": len(unknown_queries),
        "accepted_known_total": len(store["accepted_known"]),
        "accepted_unknown_total": len(store["accepted_unknown"]),
        "pending_known": pending_known,
        "pending_unknown": pending_unknown,
        "processed_now_paths": [row["path"] for row in accepted_known] + [row["path"] for row in accepted_unknown],
        "kept_known_wrong": len(kept_known),
        "kept_unknown_wrong": len(kept_unknown),
        "added_known": len(add_known),
        "added_unknown": len(add_unknown),
        "next_active_known": sum(1 for r in next_active if r.get("known")),
        "next_active_unknown": sum(1 for r in next_active if not r.get("known")),
        "next_active_total": len(next_active),
        "harvest_panel": panel_attacks,
        "harvest_panel_min_passes": panel_min_passes,
        "pool_rebuild_skipped": False,
        "unified_eval_harvest": True,
    }


def _product_for_path(path):
    parts = Path(path).parts
    if "Data Processed" in parts:
        idx = parts.index("Data Processed")
        if idx + 3 < len(parts):
            return "/".join(parts[idx + 1:-1])
    return None


def _build_pair_sources(manifest):
    product_images = {}
    category_products = defaultdict(set)
    known_products = set()
    unknown_products = set()
    for row in manifest:
        for key in ("path_1", "path_2"):
            path = row.get(key, "")
            category = _category_for_path(path)
            product = _product_for_path(path)
            if not category or not product:
                continue
            product_images.setdefault(product, set()).add(path)
            category_products[category].add(product)
    return {
        "product_images": {k: sorted(v) for k, v in product_images.items()},
        "category_products": {k: sorted(v) for k, v in category_products.items()},
    }


def _refill_manifest(manifest, pair_sources, known_set, target_size, seed):
    if len(manifest) >= target_size:
        return 0
    rng = random.Random(seed + len(manifest) * 17)
    product_images = pair_sources["product_images"]
    category_products = pair_sources["category_products"]
    known_categories = [c for c in category_products if c in known_set]
    unknown_categories = [c for c in category_products if c not in known_set]
    known_products = [p for c in known_categories for p in category_products[c]]
    unknown_products = [p for c in unknown_categories for p in category_products[c]]
    multi_known_products = [p for p in known_products if len(product_images.get(p, [])) >= 2]
    next_idx = len(manifest)
    added = 0

    while len(manifest) < target_size:
        if unknown_products and known_products and rng.random() < 0.75:
            p1 = rng.choice(known_products)
            p2 = rng.choice(unknown_products)
            manifest.append(
                {
                    "row_id": f"refill_neg_{next_idx:06d}",
                    "path_1": rng.choice(product_images[p1]),
                    "path_2": rng.choice(product_images[p2]),
                    "label": 0,
                    "pair_type": "hard_neg",
                    "priority": 0.60,
                    "times_trained": 0,
                    "pair_id": f"refill_neg_{next_idx:06d}",
                }
            )
        elif multi_known_products:
            p = rng.choice(multi_known_products)
            img1, img2 = rng.sample(product_images[p], 2)
            manifest.append(
                {
                    "row_id": f"refill_pos_{next_idx:06d}",
                    "path_1": img1,
                    "path_2": img2,
                    "label": 1,
                    "pair_type": "hard_pos",
                    "priority": 0.80,
                    "times_trained": 0,
                    "pair_id": f"refill_pos_{next_idx:06d}",
                }
            )
        else:
            break
        next_idx += 1
        added += 1
    return added


def _drop_processed_pairs_from_manifest(manifest, processed_paths, known_set=None):
    if not processed_paths:
        return manifest, 0
    processed = set(processed_paths)
    known_set = known_set or set()
    min_trainings = int(os.environ.get("BTP_HARVEST_PRUNE_MIN_TRAININGS", "3"))
    pos_max_dist = float(
        os.environ.get(
            "BTP_HARVEST_PRUNE_POS_MAX_DIST",
            os.environ.get("BTP_PRUNE_KNOWN_MAX_DIST", "8.0"),
        )
    )
    neg_min_dist = float(
        os.environ.get(
            "BTP_HARVEST_PRUNE_NEG_MIN_DIST",
            os.environ.get("BTP_PRUNE_UNKNOWN_MIN_DIST", "14.0"),
        )
    )
    kept = []
    removed = 0
    for row in manifest:
        p1 = row.get("path_1", "")
        p2 = row.get("path_2", "")
        touches_accepted_path = p1 in processed or p2 in processed
        if not touches_accepted_path:
            kept.append(row)
            continue

        times = int(row.get("times_trained", 0))
        if times < min_trainings:
            kept.append(row)
            continue

        try:
            dist = float(row.get("last_distance", row.get("priority", 1e9)))
        except (TypeError, ValueError):
            kept.append(row)
            continue

        label = float(row.get("label", 0.0))
        k1 = bool(_known_flag_for_path(p1, known_set))
        k2 = bool(_known_flag_for_path(p2, known_set))
        is_known_known = k1 and k2
        is_known_unknown = k1 != k2

        remove_pair = False
        if label > 0.5 and is_known_known and dist <= pos_max_dist:
            remove_pair = True
        elif label < 0.5 and is_known_unknown and dist >= neg_min_dist:
            remove_pair = True

        if remove_pair:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def _remaining_pool_counts():
    active_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    source_path = os.environ.get("BTP_EMBED_SOURCE_POOL", "")
    store_path = os.environ.get("BTP_EMBED_STORE_JSON", "")
    if not source_path or not os.path.exists(source_path):
        return 0, 0

    with open(source_path, "r", encoding="utf-8") as f:
        source_pool = json.load(f)
    active_pool = []
    if active_path and os.path.exists(active_path):
        with open(active_path, "r", encoding="utf-8") as f:
            active_pool = json.load(f)
    store = _load_embedding_store(store_path)
    processed = set(store.get("processed_paths", []))
    active_paths = {row.get("path") for row in active_pool}
    unseen = [
        row for row in source_pool
        if row.get("path") not in processed and row.get("path") not in active_paths
    ]
    return len(active_pool), len(unseen)


def _rebuild_manifest_from_remaining_pool(target_size, seed):
    active_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    source_path = os.environ.get("BTP_EMBED_SOURCE_POOL", "")
    store_path = os.environ.get("BTP_EMBED_STORE_JSON", "")
    if not active_path or not os.path.exists(active_path):
        return []
    if not source_path or not os.path.exists(source_path):
        return []

    with open(active_path, "r", encoding="utf-8") as f:
        active_pool = json.load(f)
    if not active_pool:
        return []

    store = _load_embedding_store(store_path)
    processed = set(store.get("processed_paths", []))
    active_paths = {row.get("path") for row in active_pool}
    remaining_rows = [row for row in active_pool if row.get("path") not in processed]
    if not remaining_rows:
        return []

    rng = random.Random(seed)
    product_images = defaultdict(list)
    product_known = {}
    product_category = {}
    category_known_products = defaultdict(list)
    product_has_active = defaultdict(bool)

    for row in remaining_rows:
        path = row.get("path", "")
        product = row.get("product") or _product_for_path(path)
        category = row.get("category") or _category_for_path(path)
        if not path or not product or not category:
            continue
        product_images[product].append(path)
        product_known[product] = bool(row.get("known"))
        product_category[product] = category
        if row.get("known"):
            category_known_products[category].append(product)
        if path in active_paths:
            product_has_active[product] = True

    for product in list(product_images.keys()):
        product_images[product] = sorted(set(product_images[product]))

    known_products = [p for p in product_images if product_known.get(p)]
    unknown_products = [p for p in product_images if not product_known.get(p)]
    all_products = list(product_images)
    multi_known_products = [p for p in known_products if len(product_images.get(p, [])) >= 2]
    same_category_known = [c for c, ps in category_known_products.items() if len(set(ps)) >= 2]
    if not all_products:
        return []

    def choose_product(pool):
        if not pool:
            return None
        active_first = [p for p in pool if product_has_active.get(p)]
        if active_first and (len(active_first) == len(pool) or rng.random() < 0.75):
            pool = active_first
        return rng.choice(pool)

    target_size = max(1, target_size)
    target_hp = int(os.environ.get("BTP_TARGET_HARD_POS", "10000"))
    target_hn = int(os.environ.get("BTP_TARGET_HARD_NEG", "4000"))
    target_sp = int(os.environ.get("BTP_TARGET_SOFT_POS", "4000"))
    target_sn = int(os.environ.get("BTP_TARGET_SOFT_NEG", "2000"))
    raw_targets = {
        "hard_pos": max(0, target_hp),
        "hard_neg": max(0, target_hn),
        "soft_pos": max(0, target_sp),
        "soft_neg": max(0, target_sn),
    }
    total_requested = sum(raw_targets.values())
    if total_requested <= 0:
        raw_targets = {"hard_pos": 1, "hard_neg": 0, "soft_pos": 0, "soft_neg": 0}
        total_requested = 1
    scale = target_size / float(total_requested)
    targets = {k: int(v * scale) for k, v in raw_targets.items()}
    assigned = sum(targets.values())
    if assigned < target_size:
        order = ["hard_pos", "soft_pos", "hard_neg", "soft_neg"]
        idx = 0
        while assigned < target_size:
            targets[order[idx % len(order)]] += 1
            assigned += 1
            idx += 1
    counts = Counter()
    manifest = []
    next_idx = 0

    def add(pair_type, p1, p2, label, priority):
        nonlocal next_idx
        manifest.append(
            {
                "row_id": f"rebuild_{pair_type}_{next_idx:06d}",
                "path_1": p1,
                "path_2": p2,
                "label": label,
                "pair_type": pair_type,
                "priority": priority,
                "times_trained": 0,
                "pair_id": f"rebuild_{pair_type}_{next_idx:06d}",
            }
        )
        counts[pair_type] += 1
        next_idx += 1

    def try_add(pair_type):
        if pair_type in {"hard_pos", "soft_pos"}:
            if not multi_known_products:
                return False
            product = choose_product(multi_known_products)
            imgs = product_images.get(product, [])
            if len(imgs) < 2:
                return False
            if pair_type == "hard_pos":
                img = rng.choice(imgs)
                add(pair_type, img, img, 1, 0.80)
            else:
                img1, img2 = rng.sample(imgs, 2)
                add(pair_type, img1, img2, 1, 0.0)
            return True

        if pair_type == "hard_neg":
            if known_products and unknown_products and rng.random() < 0.80:
                p1 = choose_product(known_products)
                p2 = choose_product(unknown_products)
                add(pair_type, rng.choice(product_images[p1]), rng.choice(product_images[p2]), 0, 0.30)
                return True
            if same_category_known:
                cat = rng.choice(same_category_known)
                prods = list(sorted(set(category_known_products[cat])))
                if len(prods) >= 2:
                    p1, p2 = rng.sample(prods, 2)
                    add(pair_type, rng.choice(product_images[p1]), rng.choice(product_images[p2]), 0, 0.20)
                    return True
            return False

        if pair_type == "soft_neg":
            if known_products and unknown_products and rng.random() < 0.70:
                p1 = choose_product(known_products)
                p2 = choose_product(unknown_products)
                add(pair_type, rng.choice(product_images[p1]), rng.choice(product_images[p2]), 0, 0.0)
                return True
            if len(all_products) < 2:
                return False
            p1 = choose_product(all_products)
            candidates = [
                p for p in all_products
                if p != p1 and product_category.get(p) != product_category.get(p1)
            ]
            if not candidates:
                candidates = [p for p in all_products if p != p1]
            if not candidates:
                return False
            p2 = choose_product(candidates)
            add(pair_type, rng.choice(product_images[p1]), rng.choice(product_images[p2]), 0, 0.0)
            return True

        return False

    while len(manifest) < target_size:
        deficits = [(ptype, targets[ptype] - counts[ptype]) for ptype in targets if targets[ptype] - counts[ptype] > 0]
        if not deficits:
            break
        deficits.sort(key=lambda item: item[1], reverse=True)
        added = False
        for pair_type, _ in deficits:
            if try_add(pair_type):
                added = True
                break
        if not added:
            break

    return manifest


# ═══════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════

def load_manifest(path):
    """Load the manifest JSON file (pair definitions + metadata)."""
    with open(path, 'r') as f:
        return json.load(f)


def save_manifest(manifest, path):
    """Save manifest (with updated priorities and removed pairs)."""
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save model state for resuming training or inference.
    
    Saved items:
      - epoch: which epoch this was
      - model_state_dict: all neural network weights
      - optimizer_state_dict: optimizer state (momentum buffers, etc.)
      - loss: training loss (for reference)
    """
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def save_checkpoint_from_state(state_dict, optimizer, epoch, loss, path, extra=None):
    payload = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)



def cleanup_epoch_checkpoints(checkpoint_dir, model_prefix):
    """Remove intermediate epoch checkpoints after the final checkpoint is saved."""
    removed = 0
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(f"{model_prefix}_epoch_") and filename.endswith(".pt"):
            os.remove(os.path.join(checkpoint_dir, filename))
            removed += 1
    print(f"🧹 Removed {removed} intermediate {model_prefix} epoch checkpoints")

def save_batch_csv(manifest, batch_indices, csv_path, epoch=None, batch_num=None):
    """
    Save actual batch composition to CSV (shows what's training).
    
    - csv_path: Live CSV showing current batch (overwrites each batch)
    - history_csv: Appending CSV tracking ALL batches across epochs
    
    Helps with debugging and monitoring.
    Sorts by type first, then by priority within type.
    """
    batch_idx_list = [idx.item() if hasattr(idx, 'item') else idx for idx in batch_indices]
    batch_pairs = [manifest[i] for i in batch_idx_list]
    
    # Sort by type (hard_pos → hard_neg → soft_pos → soft_neg) then priority
    type_order = {'hard_pos': 0, 'hard_neg': 1, 'soft_pos': 2, 'soft_neg': 3}
    batch_pairs_sorted = sorted(
        batch_pairs,
        key=lambda x: (type_order.get(x.get('pair_type', ''), 99), -x.get('priority', 0))
    )
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # ─── LIVE CSV (current batch only, overwrites) ───
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Rank', 'Row_ID', 'Pair_Type', 'Priority', 'Times_Trained', 'Label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, row in enumerate(batch_pairs_sorted, 1):
            writer.writerow({
                'Rank': i,
                'Row_ID': row.get('row_id', '?'),
                'Pair_Type': row.get('pair_type', '?'),
                'Priority': f"{row.get('priority', 0):.4f}",
                'Times_Trained': row.get('times_trained', 0),
                'Label': row.get('label', '?'),
            })
    
    # ─── HISTORY CSV (all batches, appends) ───
    if epoch is not None and batch_num is not None:
        history_csv_path = csv_path.replace('_live_', '_history_')
        file_exists = os.path.exists(history_csv_path)
        
        with open(history_csv_path, 'a', newline='') as csvfile:
            fieldnames = ['Epoch', 'Batch', 'Rank', 'Pair_Type', 'Priority', 'Times_Trained', 'Label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for i, row in enumerate(batch_pairs_sorted, 1):
                writer.writerow({
                    'Epoch': epoch,
                    'Batch': batch_num,
                    'Rank': i,
                    'Pair_Type': row.get('pair_type', '?'),
                    'Priority': f"{row.get('priority', 0):.4f}",
                    'Times_Trained': row.get('times_trained', 0),
                    'Label': row.get('label', '?'),
                })


# ═══════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════

def train_convnext():
    """
    Main training loop for Siamese ConvNeXt-B.
    
    Algorithm (same as ResNet but with ConvNeXt backbone):
      for epoch in 1..NUM_EPOCHS:
        1. Create PrioritySampler from current manifest
        2. For each batch:
           a. Forward pass → compute loss
           b. Backward pass → update weights
           c. Update pair priorities based on loss
           d. Save batch composition to CSV
        3. Apply decay to all pair priorities
        4. Prune pairs below threshold (only if trained before)
        5. Save checkpoint and updated manifest
    """
    # Allow runtime epoch override for long runs
    run_epochs = int(os.environ.get("BTP_NUM_EPOCHS", str(NUM_EPOCHS)))
    display_epochs = int(
        os.environ.get(
            "BTP_DISPLAY_NUM_EPOCHS",
            "50"
            if os.environ.get("BTP_CKPT_TAG", "") == "convnext_amazon50k_weakfocus_train10_from_e38_20260526"
            else str(run_epochs),
        )
    )
    run_batch_size = int(os.environ.get("BTP_BATCH_SIZE", str(BATCH_SIZE)))

    # Initialize logger
    logger = TrainingLogger(model_name='convnext')
    # NOTE: Appending to existing log file (not clearing) to preserve previous runs
    logger.log_session_start("Siamese ConvNeXt-B")
    
    print("="*60)
    print("  TRAINING: Siamese ConvNeXt-B")
    print("="*60)

    # ─────────────────────────────────────────────────────
    # SETUP: Initialize model, optimizer, loss
    # ─────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Override MANIFEST_PATH to use training dataset (model-specific)
    # This ensures ConvNeXt trains on identical pairs to ResNet/ViT but prunes independently
    manifest_path = os.environ.get(
        "BTP_MANIFEST_PATH",
        os.path.join(os.path.dirname(MANIFEST_PATH), "training_convnext_pairs.json"),
    )
    manifest = load_manifest(manifest_path)
    print(f"📋 Manifest loaded: {len(manifest):,} pairs")
    
    # Count pairs by type
    type_counts = {}
    for p in manifest:
        ptype = p.get('pair_type', 'unknown')
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    print(f"   Breakdown: {type_counts}")

    # Initialize ConvNeXt-B model (FROM SCRATCH, not pretrained)
    model = SiameseConvNeXtB().to(DEVICE)
    use_dp = os.environ.get("BTP_USE_DP", "1") == "1"
    if use_dp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"🧩 DataParallel enabled on {torch.cuda.device_count()} visible GPUs")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: SiameseConvNeXt-B on {DEVICE}")
    print(f"   Total parameters: {total_params:,}")

    ckpt_tag = os.environ.get("BTP_CKPT_TAG", "convnext_2K")

    # Loss function: L = L_global + 0.5 × L_local
    criterion = DualBranchContrastiveLoss().to(DEVICE)

    known_env = os.environ.get("BTP_KNOWN_CATEGORIES", "")
    known_json = os.environ.get("BTP_KNOWN_CATEGORIES_JSON", "")
    if known_json and os.path.exists(known_json):
        with open(known_json, "r", encoding="utf-8") as f:
            known_set = set(json.load(f))
    elif known_env.strip():
        known_set = {c.strip() for c in known_env.split(",") if c.strip()}
    else:
        known_set = set()

    adapt_margin = os.environ.get("BTP_ADAPT_MARGIN", "0") == "1"
    full_coverage = os.environ.get("BTP_FULL_COVERAGE", "1") == "1"
    log_batches_raw = os.environ.get("BTP_LOG_BATCHES", "0").strip()
    if log_batches_raw.lower() in {"", "0", "false", "no"}:
        log_batch_interval = 0
    elif log_batches_raw.lower() in {"true", "yes"}:
        log_batch_interval = 1
    else:
        try:
            log_batch_interval = max(1, int(log_batches_raw))
        except ValueError:
            log_batch_interval = 1
    concise_end_log = os.environ.get("BTP_CONCISE_END_LOG", "0").strip().lower() in {"1", "true", "yes"}
    use_amp = os.environ.get("BTP_USE_AMP", "1") == "1" and str(DEVICE).startswith("cuda")
    margin_step = float(os.environ.get("BTP_MARGIN_STEP", "0.2"))
    margin_min = float(os.environ.get("BTP_MARGIN_MIN", "1.0"))
    margin_max = float(os.environ.get("BTP_MARGIN_MAX", "6.0"))
    target_gap = float(os.environ.get("BTP_TARGET_GAP", "0.6"))
    target_known_max = float(os.environ.get("BTP_TARGET_KNOWN_MAX", "2.0"))
    target_unknown_acc = float(os.environ.get("BTP_TARGET_UNKNOWN_ACC", "0.5"))
    unknown_boost = float(os.environ.get("BTP_UNKNOWN_HARD_BOOST", "1.5"))
    unknown_push_weight = float(os.environ.get("BTP_UNKNOWN_PUSH_WEIGHT", "0.5"))
    db_pair_weight = float(os.environ.get("BTP_PAIR_LOSS_WEIGHT", "0.25"))
    db_known_weight = float(os.environ.get("BTP_DB_KNOWN_WEIGHT", "4.0"))
    db_unknown_weight = float(os.environ.get("BTP_DB_UNKNOWN_WEIGHT", "20.0"))
    db_known_target = float(os.environ.get("BTP_DB_KNOWN_TARGET", "0.8"))
    db_unknown_target = float(os.environ.get("BTP_DB_UNKNOWN_TARGET", "2.0"))
    db_unknown_power = float(os.environ.get("BTP_DB_UNKNOWN_POWER", "4.0"))
    db_known_power = float(os.environ.get("BTP_DB_KNOWN_POWER", "2.0"))
    strict_consistency_enabled = os.environ.get("BTP_STRICT_CONSISTENCY_ENABLED", "1") == "1"
    strict_consistency_weight = float(os.environ.get("BTP_STRICT_CONSISTENCY_WEIGHT", "1.0"))
    strict_bank_known_weight = float(os.environ.get("BTP_STRICT_BANK_KNOWN_WEIGHT", "2.0"))
    strict_bank_unknown_weight = float(os.environ.get("BTP_STRICT_BANK_UNKNOWN_WEIGHT", "2.0"))
    strict_consistency_samples_known = max(0, int(os.environ.get("BTP_STRICT_CONSISTENCY_SAMPLES_KNOWN", "2")))
    strict_consistency_samples_unknown = max(0, int(os.environ.get("BTP_STRICT_CONSISTENCY_SAMPLES_UNKNOWN", "2")))
    strict_consistency_attack_names = [
        name for name in os.environ.get(
            "BTP_STRICT_CONSISTENCY_ATTACKS",
            os.environ.get(
                "BTP_HARVEST_ATTACK_PANEL",
                "original,rotate-10,rotate40,crop40,heavy_bright,flip_h,flip_v,resize,watermark_asset,blur,bg_color_change",
            ),
        ).split(",")
        if name.strip() and name.strip() != "original"
        for name in [name.strip()]
    ]
    strict_consistency_attack_names_known = [
        name for name in os.environ.get("BTP_STRICT_CONSISTENCY_ATTACKS_KNOWN", "").split(",")
        if name.strip() and name.strip() != "original"
        for name in [name.strip()]
    ]
    strict_consistency_attack_names_unknown = [
        name for name in os.environ.get("BTP_STRICT_CONSISTENCY_ATTACKS_UNKNOWN", "").split(",")
        if name.strip() and name.strip() != "original"
        for name in [name.strip()]
    ]
    if not strict_consistency_attack_names_known:
        strict_consistency_attack_names_known = strict_consistency_attack_names
    if not strict_consistency_attack_names_unknown:
        strict_consistency_attack_names_unknown = strict_consistency_attack_names
    unknown_lock = os.environ.get("BTP_UNKNOWN_LOCK", "0") == "1"
    unknown_lock_margin = float(os.environ.get("BTP_UNKNOWN_LOCK_MARGIN", "0.15"))
    unknown_lock_weight = float(os.environ.get("BTP_UNKNOWN_LOCK_WEIGHT", "20.0"))
    unknown_lock_floor = max(
        db_unknown_target,
        float(os.environ.get("BTP_UNKNOWN_LOCK_INIT", str(db_unknown_target))),
    )
    loss_warn_threshold = float(os.environ.get("BTP_LOSS_WARN_THRESHOLD", "200.0"))
    loss_hard_cap = float(os.environ.get("BTP_LOSS_HARD_CAP", "500.0"))
    auto_backoff_decay = float(os.environ.get("BTP_AUTO_BACKOFF_DECAY", "0.7"))
    auto_backoff_min_scale = float(os.environ.get("BTP_AUTO_BACKOFF_MIN_SCALE", "0.1"))
    auto_backoff_streak = max(1, int(os.environ.get("BTP_AUTO_BACKOFF_STREAK", "5")))
    auto_backoff_cooldown_batches = max(0, int(os.environ.get("BTP_AUTO_BACKOFF_COOLDOWN", "10")))
    auto_backoff_improve_delta = float(os.environ.get("BTP_AUTO_BACKOFF_IMPROVE_DELTA", "1.0"))
    grad_clip_norm = float(os.environ.get("BTP_GRAD_CLIP_NORM", "5.0"))
    runtime_db_pair_weight = db_pair_weight
    runtime_unknown_push_weight = unknown_push_weight
    runtime_db_known_weight = db_known_weight
    runtime_db_unknown_weight = db_unknown_weight
    runtime_unknown_lock_weight = unknown_lock_weight
    unknown_only = os.environ.get("BTP_UNKNOWN_ONLY", "0").strip() == "1"
    pos_weight_env = float(os.environ.get("BTP_POS_WEIGHT", "1.0"))
    other_neg_weight_env = float(os.environ.get("BTP_OTHER_NEG_WEIGHT", "1.0"))
    unknown_neg_weight_env = float(os.environ.get("BTP_UNKNOWN_NEG_WEIGHT", "1.0"))
    emb_scale = float(os.environ.get("BTP_EMB_SCALE", "1.0"))
    prune_known_max_dist = float(os.environ.get("BTP_PRUNE_KNOWN_MAX_DIST", "8.0"))
    prune_unknown_min_dist = float(os.environ.get("BTP_PRUNE_UNKNOWN_MIN_DIST", "14.0"))
    prune_hard_pos_required_passes = int(os.environ.get("BTP_PRUNE_HARD_POS_REQUIRED_PASSES", "6"))
    prune_soft_pos_required_passes = int(os.environ.get("BTP_PRUNE_SOFT_POS_REQUIRED_PASSES", "10"))
    min_remaining_floor = int(os.environ.get("BTP_MIN_REMAINING_FLOOR", "0"))
    refill_target_pairs = int(os.environ.get("BTP_REFILL_TARGET_PAIRS", "0"))
    continue_until_pool_done = os.environ.get("BTP_CONTINUE_UNTIL_POOL_DONE", "1") == "1"
    eval_every = max(1, int(os.environ.get("BTP_EVAL_EVERY", "1")))
    stop_known_correct = int(os.environ.get("BTP_STOP_KNOWN_CORRECT", "1400"))
    stop_unknown_correct = int(os.environ.get("BTP_STOP_UNKNOWN_CORRECT", "250"))
    stop_known_total = int(os.environ.get("BTP_STOP_KNOWN_TOTAL", "1600"))
    stop_unknown_total = int(os.environ.get("BTP_STOP_UNKNOWN_TOTAL", "400"))

    runtime_learning_rate = float(os.environ.get('BTP_LEARNING_RATE', str(LEARNING_RATE)))
    runtime_weight_decay = float(os.environ.get('BTP_WEIGHT_DECAY', str(WEIGHT_DECAY)))

    # Optimizer: Adam with weight decay (L2 regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=runtime_learning_rate,
        weight_decay=runtime_weight_decay
    )
    scaler = GradScaler(enabled=use_amp)
    transform = get_transform()
    num_workers = int(os.environ.get("BTP_NUM_WORKERS", "4"))
    prefetch_factor = int(os.environ.get("BTP_PREFETCH_FACTOR", "2"))
    db_bank_refresh_epochs = max(1, int(os.environ.get("BTP_DB_BANK_REFRESH_EPOCHS", "2")))
    write_batch_csv = os.environ.get("BTP_WRITE_BATCH_CSV", "0") == "1"
    max_batches_per_epoch = int(os.environ.get("BTP_MAX_BATCHES_PER_EPOCH", "0"))
    dynamic_cap_loss_threshold = float(os.environ.get("BTP_DYNAMIC_CAP_LOSS_THRESHOLD", "0"))
    db_bank = None
    pair_sources = _build_pair_sources(manifest)
    best_known_correct = -1
    best_unknown_correct = -1
    best_perattack_metric = float(os.environ.get("BTP_PERATTACK_BASELINE_METRIC", "-1"))
    best_perattack_epoch = int(os.environ.get("BTP_PERATTACK_BASELINE_EPOCH", "0"))
    best_perattack_path = os.environ.get("BTP_PERATTACK_BASELINE_LABEL", "")
    perattack_gate_bad_streak = 0
    perattack_gate_best_score = float("-inf")
    perattack_drop_bad_streak = 0
    soup_enabled = False
    soup_state = None

    # Optional resume
    start_epoch = 1
    prev_epoch_avg_total = float("inf")
    resume_ckpt = os.environ.get("BTP_RESUME_CKPT", "")
    if resume_ckpt and os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        reset_optimizer = os.environ.get("BTP_RESET_OPTIMIZER_ON_RESUME", "0") == "1"
        if "optimizer_state_dict" in checkpoint and not reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        elif reset_optimizer:
            print("  [Resume] Optimizer state reset on resume.")
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        prev_epoch_avg_total = float(checkpoint.get("avg_total_loss", prev_epoch_avg_total))
        print(f"🔁 Resuming from {resume_ckpt} at epoch {start_epoch}")

    # ─────────────────────────────────────────────────────
    # MAIN LOOP: iterate over epochs
    # ─────────────────────────────────────────────────────    
    _ensure_active_eval_pool()
    for epoch in range(start_epoch, run_epochs + 1):
        epoch_start = time.time()
        effective_max_batches = max_batches_per_epoch
        if dynamic_cap_loss_threshold > 0 and prev_epoch_avg_total > dynamic_cap_loss_threshold:
            effective_max_batches = 0
        print(f"\n{'─'*60}")
        print(f"EPOCH {epoch}/{display_epochs} | Pairs: {len(manifest):,}")
        print(f"{'─'*60}")

        if dynamic_cap_loss_threshold > 0:
            cap_msg = "full epoch" if effective_max_batches == 0 else f"cap {effective_max_batches}"
            print(
                f"  [Dynamic Cap] prev avg loss {prev_epoch_avg_total:.4f} | "
                f"threshold {dynamic_cap_loss_threshold:.4f} -> {cap_msg}"
            )
        # Log epoch header
        os.environ["BTP_CURRENT_EPOCH"] = str(epoch)
        logger.log_epoch_header(manifest, model, DEVICE, epoch, display_epochs, "Siamese ConvNeXt-B")

        # Create dataset and sampler for this epoch
        dataset = SiameseDataset(manifest_path=manifest_path, transform=transform, return_known_labels=True)
        if db_bank is None or ((epoch - start_epoch) % db_bank_refresh_epochs == 0):
            db_bank = _build_db_bank(model, transform)
        drop_last_batches = os.environ.get("BTP_DROP_LAST", "1") == "1"
        loader_kwargs = {
            "batch_size": run_batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "drop_last": drop_last_batches,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = prefetch_factor
        if full_coverage:
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                **loader_kwargs,
            )
        else:
            sampler = PrioritySampler(manifest)
            dataloader = DataLoader(
                dataset,
                sampler=sampler,
                **loader_kwargs,
            )

        model.train()  # Set to training mode
        
        # Track statistics across epoch
        epoch_total_loss = 0.0
        epoch_global_loss = 0.0
        epoch_local_loss = 0.0
        num_batches = 0
        pairs_trained_this_epoch = 0
        spike_streak = 0
        no_improve_streak = 0
        backoff_cooldown = 0
        best_stability_loss = float("inf")

        # ──────────────────────────────────────────────
        # BATCH LOOP: iterate over mini-batches
        # ──────────────────────────────────────────────
        for batch_idx, (img1, img2, labels, indices, known1, known2) in enumerate(dataloader):
            if effective_max_batches > 0 and batch_idx >= effective_max_batches:
                if not concise_end_log:
                    print(f"  [Epoch Cap] Reached max batches per epoch: {effective_max_batches}")
                break
            # Move to device (GPU/CPU)
            img1   = img1.to(DEVICE, non_blocking=True)
            img2   = img2.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            batch_rows = [manifest[idx.item()] for idx in indices]
            batch_paths_1 = [row.get("path_1", "") for row in batch_rows]
            batch_paths_2 = [row.get("path_2", "") for row in batch_rows]
            known1 = torch.tensor(
                [_known_flag_for_path(path, known_set) for path in batch_paths_1],
                dtype=torch.float32,
                device=DEVICE,
            )
            known2 = torch.tensor(
                [_known_flag_for_path(path, known_set) for path in batch_paths_2],
                dtype=torch.float32,
                device=DEVICE,
            )

            # ── 1. FORWARD PASS ──
            # Both images through shared ConvNeXt backbone
            # Returns: (emb1, emb2, global1, global2, local1, local2)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                emb1, emb2, global1, global2, local1, local2 = model(img1, img2)
                global1_s = global1 * emb_scale
                global2_s = global2 * emb_scale
                local1_s = local1 * emb_scale
                local2_s = local2 * emb_scale

            # ── 2. COMPUTE LOSS ──
            # L_total = L_global + 0.5 × L_local
            unknown_mask = (labels < 0.5) & (known1 != known2)
            if unknown_only:
                pos_weight = 0.0
                other_neg_weight = 0.0
                unknown_neg_weight = 1.0
            else:
                pos_weight = pos_weight_env
                other_neg_weight = other_neg_weight_env
                unknown_neg_weight = unknown_neg_weight_env
            sample_weights = torch.where(
                labels > 0.5,
                torch.tensor(pos_weight, device=DEVICE),
                torch.where(unknown_mask, torch.tensor(unknown_neg_weight, device=DEVICE), torch.tensor(other_neg_weight, device=DEVICE)),
            )
            total_loss, global_loss, local_loss, distances = criterion(
                global1_s, global2_s, local1_s, local2_s, labels, sample_weights=sample_weights
            )
            if unknown_mask.any():
                unknown_push_floor = float(os.environ.get("BTP_UNKNOWN_PUSH_FLOOR", "0.25"))
                unknown_push_cap = float(os.environ.get("BTP_UNKNOWN_PUSH_CAP", "10.0"))
                unknown_dist = distances[unknown_mask].clamp_min(unknown_push_floor)
                unknown_push = (1.0 / unknown_dist).mean().clamp(max=unknown_push_cap)
            else:
                unknown_push = torch.tensor(0.0, device=DEVICE)

            total_loss = (runtime_db_pair_weight * total_loss) + (runtime_unknown_push_weight * unknown_push)

            # Database-style shaping: known should sit near the known bank, unknown should sit outside it.
            if db_bank is not None and db_bank.numel() > 0:
                emb1_s = emb1 * emb_scale
                emb2_s = emb2 * emb_scale
                # torch.cdist on CUDA does not support float16. Keep only this
                # bank-distance computation in float32 when AMP/autocast is on.
                bank_for_batch = db_bank.float()
                min_d1 = torch.cdist(emb1_s.float(), bank_for_batch).min(dim=1).values
                min_d2 = torch.cdist(emb2_s.float(), bank_for_batch).min(dim=1).values

                known_mask_1 = known1 > 0.5
                known_mask_2 = known2 > 0.5
                unknown_mask_1 = known1 < 0.5
                unknown_mask_2 = known2 < 0.5

                db_known_terms = []
                db_unknown_terms = []
                lock_terms = []
                db_term_cap = float(os.environ.get("BTP_DB_TERM_CAP", "8.0"))
                db_known_effective_power = min(
                    db_known_power, float(os.environ.get("BTP_DB_KNOWN_MAX_EFFECTIVE_POWER", "2.0"))
                )
                db_unknown_effective_power = min(
                    db_unknown_power, float(os.environ.get("BTP_DB_UNKNOWN_MAX_EFFECTIVE_POWER", "2.0"))
                )
                lock_effective_power = min(
                    max(2.0, db_unknown_power), float(os.environ.get("BTP_DB_LOCK_MAX_EFFECTIVE_POWER", "2.0"))
                )
                if known_mask_1.any():
                    k1 = torch.relu(min_d1[known_mask_1] - db_known_target).clamp(max=db_term_cap)
                    db_known_terms.append(torch.log1p(k1).pow(db_known_effective_power).mean())
                if known_mask_2.any():
                    k2 = torch.relu(min_d2[known_mask_2] - db_known_target).clamp(max=db_term_cap)
                    db_known_terms.append(torch.log1p(k2).pow(db_known_effective_power).mean())
                if unknown_mask_1.any():
                    u1 = torch.relu(db_unknown_target - min_d1[unknown_mask_1]).clamp(max=db_term_cap)
                    db_unknown_terms.append(torch.log1p(u1).pow(db_unknown_effective_power).mean())
                    if unknown_lock:
                        lock_target = unknown_lock_floor + unknown_lock_margin
                        ul1 = torch.relu(lock_target - min_d1[unknown_mask_1]).clamp(max=db_term_cap)
                        lock_terms.append(torch.log1p(ul1).pow(lock_effective_power).mean())
                if unknown_mask_2.any():
                    u2 = torch.relu(db_unknown_target - min_d2[unknown_mask_2]).clamp(max=db_term_cap)
                    db_unknown_terms.append(torch.log1p(u2).pow(db_unknown_effective_power).mean())
                    if unknown_lock:
                        lock_target = unknown_lock_floor + unknown_lock_margin
                        ul2 = torch.relu(lock_target - min_d2[unknown_mask_2]).clamp(max=db_term_cap)
                        lock_terms.append(torch.log1p(ul2).pow(lock_effective_power).mean())

                if db_known_terms:
                    total_loss = total_loss + (runtime_db_known_weight * torch.stack(db_known_terms).mean())
                if db_unknown_terms:
                    total_loss = total_loss + (runtime_db_unknown_weight * torch.stack(db_unknown_terms).mean())
                if lock_terms:
                    total_loss = total_loss + (runtime_unknown_lock_weight * torch.stack(lock_terms).mean())

                if strict_consistency_enabled and strict_consistency_attack_names:
                    attack_tensors = []
                    attack_anchor_embs = []
                    attack_is_known = []
                    attack_paths = []
                    attack_names_used = []

                    def _queue_attacks(paths, anchors, known_flags, side_tag):
                        for sample_idx, (path, anchor_emb, known_flag) in enumerate(zip(paths, anchors, known_flags)):
                            if not path:
                                continue
                            num_samples = strict_consistency_samples_known if known_flag > 0.5 else strict_consistency_samples_unknown
                            if num_samples <= 0:
                                continue
                            side_attack_names = (
                                strict_consistency_attack_names_known
                                if known_flag > 0.5
                                else strict_consistency_attack_names_unknown
                            )
                            if not side_attack_names:
                                continue
                            for slot in range(num_samples):
                                attack_name = _stable_panel_attack(
                                    f"{side_tag}:{path}", side_attack_names, epoch, slot
                                )
                                if not attack_name:
                                    continue
                                tensor = _load_tensor_for_attack(path, transform, attack_name=attack_name)
                                if tensor is None:
                                    continue
                                attack_tensors.append(tensor)
                                attack_anchor_embs.append(anchor_emb)
                                attack_is_known.append(bool(known_flag > 0.5))
                                attack_paths.append(path)
                                attack_names_used.append(attack_name)

                    _queue_attacks(batch_paths_1, emb1_s, known1.detach().cpu().tolist(), "p1")
                    _queue_attacks(batch_paths_2, emb2_s, known2.detach().cpu().tolist(), "p2")

                    if attack_tensors:
                        anchor_batch = torch.stack(attack_anchor_embs, dim=0)
                        forward_chunk = int(os.environ.get("BTP_STRICT_CONSISTENCY_FORWARD_CHUNK", "0"))
                        if forward_chunk <= 0:
                            forward_chunk = len(attack_tensors)
                        attacked_emb_chunks = []
                        attacked_global_chunks = []
                        attacked_local_chunks = []
                        for start in range(0, len(attack_tensors), forward_chunk):
                            end = min(start + forward_chunk, len(attack_tensors))
                            attacked_batch = torch.stack(attack_tensors[start:end], dim=0).to(DEVICE, non_blocking=True)
                            single_attack_batch = attacked_batch.shape[0] == 1
                            if single_attack_batch:
                                attacked_batch = torch.cat([attacked_batch, attacked_batch], dim=0)
                            with autocast(enabled=use_amp):
                                attacked_emb, _, attacked_global, _, attacked_local, _ = model(attacked_batch, attacked_batch)
                                if single_attack_batch:
                                    attacked_emb = attacked_emb[:1]
                                    attacked_global = attacked_global[:1]
                                    attacked_local = attacked_local[:1]
                                attacked_emb_chunks.append(attacked_emb * emb_scale)
                                attacked_global_chunks.append(attacked_global * emb_scale)
                                attacked_local_chunks.append(attacked_local * emb_scale)
                        attacked_emb_s = torch.cat(attacked_emb_chunks, dim=0)
                        attacked_global_s = torch.cat(attacked_global_chunks, dim=0)
                        attacked_local_s = torch.cat(attacked_local_chunks, dim=0)
                        anchor_global = anchor_batch[:, :attacked_global_s.shape[1]]
                        anchor_local = anchor_batch[:, attacked_global_s.shape[1]:attacked_global_s.shape[1] + attacked_local_s.shape[1]]

                        strict_embed_loss = F.smooth_l1_loss(attacked_emb_s, anchor_batch)
                        strict_global_loss = F.smooth_l1_loss(attacked_global_s, anchor_global)
                        strict_local_loss = F.smooth_l1_loss(attacked_local_s, anchor_local)
                        total_loss = total_loss + strict_consistency_weight * (
                            strict_embed_loss + 0.5 * strict_global_loss + 0.5 * strict_local_loss
                        )

                        if db_bank is not None and db_bank.numel() > 0:
                            bank_for_attacks = db_bank.float()
                            attacked_min = torch.cdist(attacked_emb_s.float(), bank_for_attacks).min(dim=1).values
                            known_attack_terms = []
                            unknown_attack_terms = []
                            for i, is_known_attack in enumerate(attack_is_known):
                                if is_known_attack:
                                    known_attack_terms.append(
                                        torch.relu(attacked_min[i] - db_known_target)
                                    )
                                else:
                                    unknown_attack_terms.append(
                                        torch.relu(db_unknown_target - attacked_min[i])
                                    )
                            if known_attack_terms:
                                total_loss = total_loss + strict_bank_known_weight * torch.stack(known_attack_terms).mean()
                            if unknown_attack_terms:
                                total_loss = total_loss + strict_bank_unknown_weight * torch.stack(unknown_attack_terms).mean()

            raw_total_loss = total_loss.detach()
            raw_loss_value = float(raw_total_loss.item()) if torch.isfinite(raw_total_loss) else float("inf")
            is_spike = (not torch.isfinite(raw_total_loss)) or raw_loss_value > loss_warn_threshold
            if is_spike:
                spike_streak += 1
            else:
                spike_streak = 0

            if torch.isfinite(raw_total_loss):
                if raw_loss_value < (best_stability_loss - auto_backoff_improve_delta):
                    best_stability_loss = raw_loss_value
                    no_improve_streak = 0
                else:
                    no_improve_streak += 1
            else:
                no_improve_streak += 1

            if backoff_cooldown > 0:
                backoff_cooldown -= 1

            should_backoff = (
                backoff_cooldown == 0
                and (
                    (not torch.isfinite(raw_total_loss))
                    or (spike_streak >= auto_backoff_streak)
                    or (raw_loss_value > loss_warn_threshold and no_improve_streak >= auto_backoff_streak)
                )
            )

            if should_backoff:
                min_scale = auto_backoff_min_scale
                runtime_unknown_push_weight = max(unknown_push_weight * min_scale, runtime_unknown_push_weight * auto_backoff_decay)
                runtime_db_unknown_weight = max(db_unknown_weight * min_scale, runtime_db_unknown_weight * auto_backoff_decay)
                runtime_unknown_lock_weight = max(unknown_lock_weight * min_scale, runtime_unknown_lock_weight * auto_backoff_decay)
                runtime_db_pair_weight = max(db_pair_weight * min_scale, runtime_db_pair_weight * auto_backoff_decay)
                unknown_lock_floor = max(db_unknown_target, unknown_lock_floor * 0.95)
                backoff_cooldown = auto_backoff_cooldown_batches
                spike_streak = 0
                no_improve_streak = 0
                if torch.isfinite(raw_total_loss):
                    best_stability_loss = raw_loss_value
                if log_batch_interval > 0:
                    print(
                        "  [Stability] Sustained loss issue "
                        f"({raw_loss_value:.2f}) after streak. Backoff -> "
                        f"pair {runtime_db_pair_weight:.3f}, "
                        f"push {runtime_unknown_push_weight:.3f}, "
                        f"db_unknown {runtime_db_unknown_weight:.3f}, "
                        f"lock {runtime_unknown_lock_weight:.3f}, "
                        f"lock_floor {unknown_lock_floor:.2f} | "
                        f"cooldown {backoff_cooldown}"
                    )

            # Keep training numerically stable even with aggressive unknown push settings.
            total_loss = torch.nan_to_num(
                total_loss, nan=loss_hard_cap, posinf=loss_hard_cap, neginf=0.0
            ).clamp(max=loss_hard_cap)

            # ── 3. BACKWARD PASS ──
            scaler.scale(total_loss).backward()
            if use_amp:
                scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # ── 4. UPDATE MANIFEST ──
            # Store loss value as new priority for each pair
            with torch.no_grad():
                for i, manifest_idx in enumerate(indices):
                    idx = manifest_idx.item()
                    if idx < len(manifest):
                        pair_loss = distances[i].item()
                        manifest[idx]['priority'] = round(pair_loss, 4)
                        manifest[idx]['last_distance'] = round(pair_loss, 4)
                        manifest[idx]['times_trained'] += 1
                        pair_type = manifest[idx].get("pair_type", "")
                        label_val = float(manifest[idx].get("label", 0.0))
                        if label_val > 0.5:
                            if pair_type == "hard_pos":
                                if pair_loss <= prune_known_max_dist:
                                    manifest[idx]["attack_pass_streak"] = int(manifest[idx].get("attack_pass_streak", 0)) + 1
                                else:
                                    manifest[idx]["attack_pass_streak"] = 0
                            elif pair_type == "soft_pos":
                                if pair_loss <= prune_known_max_dist:
                                    manifest[idx]["stable_pass_streak"] = int(manifest[idx].get("stable_pass_streak", 0)) + 1
                                else:
                                    manifest[idx]["stable_pass_streak"] = 0

            # ── 5. TRACK STATISTICS ──
            epoch_total_loss  += total_loss.item()
            epoch_global_loss += global_loss.item()
            epoch_local_loss  += local_loss.item()
            num_batches += 1
            pairs_trained_this_epoch += len(labels)

            avg_loss = epoch_total_loss / num_batches
            should_log_batch = (
                log_batch_interval > 0
                and (
                    batch_idx == 0
                    or ((batch_idx + 1) % log_batch_interval == 0)
                    or (batch_idx + 1) == len(dataloader)
                )
            )
            if should_log_batch:
                print(f"  Batch {batch_idx+1:>5} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Global: {global_loss.item():.4f} | "
                      f"Local: {local_loss.item():.4f}")
                logger.log_batch(batch_idx+1, avg_loss, global_loss.item(), local_loss.item())
            
            # Save actual batch composition to live CSV
            if write_batch_csv:
                csv_path = os.path.join(os.path.dirname(manifest_path), "manifest_live_convnext.csv")
                save_batch_csv(manifest, indices, csv_path, epoch=epoch, batch_num=batch_idx+1)

        # ──────────────────────────────────────────────
        # POST-EPOCH: Decay, prune, checkpoint
        # ──────────────────────────────────────────────
        if num_batches == 0:
            active_left, unseen_left = _remaining_pool_counts()
            if continue_until_pool_done and active_left > 0:
                rebuild_target = max(run_batch_size, active_left)
                rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                if rebuilt:
                    manifest = rebuilt
                    save_manifest(manifest, manifest_path)
                    print(
                        f"  [Continue] Rebuilt manifest from active pool: {len(manifest)} pairs | "
                        f"active left {active_left} | unseen left {unseen_left}"
                    )
                    continue
            print("  ⚠️ No batches processed (empty manifest)")
            break

        avg_total  = epoch_total_loss  / num_batches
        avg_global = epoch_global_loss / num_batches
        avg_local  = epoch_local_loss  / num_batches
        prev_epoch_avg_total = avg_total

        # ── DECAY STEP ──
        # Priority = priority × (0.9 ^ times_trained)
        print(f"\n  Applying decay: priority × 0.9^times_trained")
        for pair in manifest:
            times = pair['times_trained']
            if times > 0:
                decay_penalty = DECAY_FACTOR ** times
                pair['priority'] = round(pair['priority'] * decay_penalty, 4)

        # ── PRUNING STEP ──
        # Remove pairs that are well-learned (soft pairs with low loss)
        # Curriculum: Soft pairs learn faster, remove after 2 trainings if priority < 1.0
        #            Hard pairs need more training, always keep
        #            Extreme cases: Remove any pair with priority < 0.5 (trivial)
        prune_after_epochs = int(os.environ.get("BTP_PRUNE_AFTER_EPOCHS", "5"))
        disable_pruning = os.environ.get("BTP_DISABLE_PRUNING", "0") == "1"
        accepted_only_pruning = os.environ.get("BTP_PRUNE_ACCEPTED_ONLY", "0") == "1"
        if disable_pruning:
            pruning_rule = "DISABLED (harvest-only removal)"
            logger.log_pruning_info(pruning_rule)
            pruned_count = 0
        elif accepted_only_pruning:
            pruning_rule = "ACCEPTED-ONLY (post-harvest strict pair removal)"
            logger.log_pruning_info(pruning_rule)
            pruned_count = 0
        elif epoch < prune_after_epochs:
            pruning_rule = f"DELAYED (epoch<{prune_after_epochs})"
            logger.log_pruning_info(pruning_rule)
            pruned_count = 0
        else:
            pruning_rule = (
                f"ATTACK-AWARE POSITIVE PRUNE "
                f"(hard_pos passes>={prune_hard_pos_required_passes}, soft_pos passes>={prune_soft_pos_required_passes}, "
                f"dist<={prune_known_max_dist:.2f}) "
                f"OR NEGATIVE DISTANCE PRUNE (known_vs_unknown_neg>={prune_unknown_min_dist:.2f}), times>=2"
            )
            print(f"  Pruning pairs: {pruning_rule}")
            logger.log_pruning_info(pruning_rule)
            before_count = len(manifest)
            kept = []
            pruned = []
            for p in manifest:
                times = p.get('times_trained', 0)
                if times < 2:
                    kept.append(p)
                    continue
                dist = float(p.get('last_distance', p.get('priority', 1e9)))
                k1 = bool(_known_flag_for_path(p.get("path_1", ""), known_set))
                k2 = bool(_known_flag_for_path(p.get("path_2", ""), known_set))
                is_known_known = k1 and k2
                is_known_unknown = k1 != k2
                label = float(p.get("label", 0.0))
                prune_it = False
                if label > 0.5 and is_known_known and dist <= prune_known_max_dist:
                    pair_type = p.get("pair_type", "")
                    if pair_type == "hard_pos":
                        passes = int(p.get("attack_pass_streak", 0))
                        if passes >= prune_hard_pos_required_passes:
                            prune_it = True
                    elif pair_type == "soft_pos":
                        passes = int(p.get("stable_pass_streak", 0))
                        if passes >= prune_soft_pos_required_passes:
                            prune_it = True
                if label < 0.5 and is_known_unknown and dist >= prune_unknown_min_dist:
                    prune_it = True
                if not prune_it:
                    kept.append(p)
                else:
                    pruned.append(p)
            if min_remaining_floor and len(kept) < min_remaining_floor:
                pruned.sort(key=lambda p: float(p.get('last_distance', p.get('priority', 0.0))), reverse=True)
                kept.extend(pruned[: max(0, min_remaining_floor - len(kept))])
            manifest = kept
            pruned_count = before_count - len(manifest)
            if refill_target_pairs and len(manifest) < refill_target_pairs:
                added = _refill_manifest(manifest, pair_sources, known_set, refill_target_pairs, epoch)
                if added:
                    print(f"  [Refill] Added {added} hard pairs to bring manifest back to {len(manifest)}")

        # ── SAVE CHECKPOINT ──
        print(f"  [Prune] pairs_pruned {pruned_count} | pairs_remaining {len(manifest)}")
        save_manifest(manifest, manifest_path)  # Save to model-specific file
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_tag}_epoch-{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, avg_total, ckpt_path)

        # ── EPOCH SUMMARY ──
        elapsed = time.time() - epoch_start
        type_counts = {}
        for p in manifest:
            ptype = p.get('pair_type', 'unknown')
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        print(f"\n  📊 Epoch {epoch} Summary:")
        print(f"     Loss: {avg_total:.4f}  (Global: {avg_global:.4f}, Local: {avg_local:.4f})")
        print(f"     Pairs trained: {pairs_trained_this_epoch:,}")
        print(f"     Remaining: {len(manifest):,}")
        print(f"     By type: {type_counts}")
        print(f"     Checkpoint: {ckpt_path}")
        print(f"     Time: {elapsed:.1f}s")

        # Log epoch summary to file
        logger.log_epoch_summary(epoch, avg_total, avg_global, avg_local,
                                pairs_trained_this_epoch, pruned_count, len(manifest),
                                type_counts, elapsed, ckpt_path)
        eval_stats = None
        if epoch % eval_every == 0:
            eval_stats = _eval_db_retrieval(model, transform, epoch)
        if adapt_margin and eval_stats is not None:
            gap = eval_stats["unknown_avg"] - eval_stats["known_avg"]
            current_margin = criterion.global_loss_fn.margin
            new_margin = current_margin
            if gap < target_gap:
                new_margin = min(margin_max, current_margin + margin_step)
            elif eval_stats["known_avg"] > target_known_max:
                new_margin = max(margin_min, current_margin - margin_step)
            if new_margin != current_margin:
                criterion.global_loss_fn.margin = new_margin
                criterion.local_loss_fn.margin = new_margin
                print(f"  [Adapt] Margin {current_margin:.2f} -> {new_margin:.2f} (gap {gap:.2f})")
        if eval_stats is not None:
            if unknown_lock:
                prev_floor = unknown_lock_floor
                candidate_floor = max(prev_floor, eval_stats["unknown_avg"] - unknown_lock_margin)
                unknown_lock_floor = candidate_floor
                prev_target = db_unknown_target
                db_unknown_target = max(db_unknown_target, unknown_lock_floor)
                if db_unknown_target != prev_target:
                    print(
                        f"  [UnknownLock] floor {prev_floor:.2f} -> {unknown_lock_floor:.2f} | "
                        f"target {prev_target:.2f} -> {db_unknown_target:.2f}"
                    )
            if eval_stats["known_correct"] > best_known_correct:
                best_known_correct = eval_stats["known_correct"]
                save_checkpoint(model, optimizer, epoch, avg_total, os.path.join(CHECKPOINT_DIR, f"{ckpt_tag}_best_known.pt"))
            if eval_stats["unknown_correct"] > best_unknown_correct:
                best_unknown_correct = eval_stats["unknown_correct"]
                save_checkpoint(model, optimizer, epoch, avg_total, os.path.join(CHECKPOINT_DIR, f"{ckpt_tag}_best_unknown.pt"))
            unknown_acc = eval_stats["unknown_correct"] / max(1, eval_stats["unknown_total"])
            if unknown_acc < target_unknown_acc:
                boosted = 0
                for pair in manifest:
                    if pair.get("label", 0) != 0:
                        continue
                    k1 = _known_flag_for_path(pair.get("path_1", ""), known_set)
                    k2 = _known_flag_for_path(pair.get("path_2", ""), known_set)
                    if bool(k1) != bool(k2):
                        pair["priority"] = round(pair.get("priority", 0.0) * unknown_boost, 4)
                        boosted += 1
                print(f"  [Boost] Unknown acc {unknown_acc:.3f} < {target_unknown_acc:.3f}. Boosted {boosted} known-vs-unknown negatives.")
            else:
                print(f"  [Boost] Unknown acc {unknown_acc:.3f} >= {target_unknown_acc:.3f}. No boost applied.")
            harvest_stats = eval_stats.get("harvest_stats") if eval_stats is not None else None
            if harvest_stats is not None:
                remove_train_pairs_on_harvest = os.environ.get("BTP_HARVEST_REMOVE_TRAIN_PAIRS", "0") == "1"
                removed_train_pairs = 0
                if remove_train_pairs_on_harvest:
                    manifest, removed_train_pairs = _drop_processed_pairs_from_manifest(
                        manifest, harvest_stats.get("processed_now_paths", []), known_set
                    )
                    if removed_train_pairs:
                        save_manifest(manifest, manifest_path)
                print(f"  [HarvestPrune] pairs_pruned {removed_train_pairs} | pairs_remaining {len(manifest)}")
                print(
                    f"  [Harvest] known_accept {harvest_stats['accepted_known']}/{harvest_stats['known_query_total']} | "
                    f"unknown_accept {harvest_stats['accepted_unknown']}/{harvest_stats['unknown_query_total']} | "
                    f"store_known_total {harvest_stats['accepted_known_total']} | "
                    f"store_unknown_total {harvest_stats['accepted_unknown_total']} | "
                    f"next_active {harvest_stats['next_active_known']}/{harvest_stats['next_active_unknown']}"
                )
                print(
                    f"  [Store] epoch {epoch} updated | "
                    f"added_known {harvest_stats['accepted_known']} | "
                    f"added_unknown {harvest_stats['accepted_unknown']} | "
                    f"store_known_total {harvest_stats['accepted_known_total']} | "
                    f"store_unknown_total {harvest_stats['accepted_unknown_total']}"
                )
                perattack = _run_perattack_epoch_eval(
                    model,
                    transform,
                    ckpt_path,
                    os.environ.get("BTP_EMBED_STORE_JSON", ""),
                    epoch,
                )
                if perattack is not None:
                    strict10_avg = float(perattack["summary"]["avg_strict10_total"])
                    nonorig_avg = float(perattack["summary"]["avg_non_original_total"])
                    original_total = float(perattack["attacks"].get("original", {}).get("total", 0.0))
                    nonorig_totals = [
                        float(v["total"])
                        for k, v in perattack["attacks"].items()
                        if k != "original"
                    ]
                    min_nonorig_total = min(nonorig_totals) if nonorig_totals else 0.0
                    attacks_below_min = sum(1 for v in nonorig_totals if v < float(os.environ.get("BTP_PERATTACK_REQUIRE_MIN_ATTACK", "80.0")))
                    metric_name = os.environ.get("BTP_PERATTACK_STOP_METRIC", "strict10").strip().lower()
                    metric_value = strict10_avg if metric_name == "strict10" else nonorig_avg
                    print(
                        f"  [PerAttackEval] epoch {epoch} | strict10_avg {strict10_avg:.2f} | "
                        f"nonorig_avg {nonorig_avg:.2f} | min_nonorig {min_nonorig_total:.2f} | "
                        f"below_min {attacks_below_min} | original {original_total:.2f} | file {perattack['out_path']}"
                    )
                    if metric_value > best_perattack_metric:
                        best_perattack_metric = metric_value
                        best_perattack_epoch = epoch
                        best_perattack_path = perattack["out_path"]
                    req_original = float(os.environ.get("BTP_PERATTACK_REQUIRE_ORIGINAL", "100.0"))
                    req_min_nonorig = float(os.environ.get("BTP_PERATTACK_REQUIRE_MIN_ATTACK", "80.0"))
                    req_avg_nonorig = float(os.environ.get("BTP_PERATTACK_REQUIRE_AVG_ATTACK", "85.0"))
                    max_below_min = int(os.environ.get("BTP_PERATTACK_MAX_ATTACKS_BELOW_MIN", "0"))
                    gate_patience = max(1, int(os.environ.get("BTP_PERATTACK_GATE_PATIENCE", "1")))
                    gate_improve_delta = float(os.environ.get("BTP_PERATTACK_GATE_IMPROVE_DELTA", "0.25"))
                    gate_log_style = os.environ.get("BTP_PERATTACK_GATE_LOG_STYLE", "full").strip().lower()
                    gate_score = (
                        nonorig_avg
                        + 0.25 * strict10_avg
                        - 2.0 * max(0, attacks_below_min - max_below_min)
                        - 0.5 * max(0.0, req_original - original_total)
                    )
                    if (
                        original_total < req_original
                        or attacks_below_min > max_below_min
                        or nonorig_avg < req_avg_nonorig
                    ):
                        improved_gate = gate_score > (perattack_gate_best_score + gate_improve_delta)
                        if improved_gate:
                            perattack_gate_best_score = gate_score
                            perattack_gate_bad_streak = 1
                        else:
                            perattack_gate_bad_streak += 1
                        if gate_log_style != "concise":
                            print(
                                f"  [PerAttackGate] epoch {epoch} failed: original {original_total:.2f} "
                                f"(need {req_original:.2f}) | attacks_below_{req_min_nonorig:.2f} "
                                f"{attacks_below_min} (allow {max_below_min}) | nonorig_avg "
                                f"{nonorig_avg:.2f} (need {req_avg_nonorig:.2f}) | "
                                f"bad_streak {perattack_gate_bad_streak}/{gate_patience} | "
                                f"gate_score {gate_score:.2f}"
                            )
                        if perattack_gate_bad_streak >= gate_patience:
                            break
                        old_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else runtime_learning_rate
                        lr_mult = float(os.environ.get("BTP_PERATTACK_ADJUST_LR_MULT", "0.85"))
                        lr_floor_mult = float(os.environ.get("BTP_PERATTACK_ADJUST_LR_FLOOR_MULT", "0.35"))
                        strict_mult = float(os.environ.get("BTP_PERATTACK_ADJUST_STRICT_MULT", "1.10"))
                        strict_add_cap = float(os.environ.get("BTP_PERATTACK_ADJUST_STRICT_ADD_CAP", "0.35"))
                        strict_max = float(os.environ.get("BTP_PERATTACK_ADJUST_STRICT_MAX", "999.0"))
                        known_mult = float(os.environ.get("BTP_PERATTACK_ADJUST_KNOWN_MULT", "0.80"))
                        known_min = float(os.environ.get("BTP_PERATTACK_ADJUST_KNOWN_MIN", "0.01"))
                        known_max = float(os.environ.get("BTP_PERATTACK_ADJUST_KNOWN_MAX", "999.0"))
                        unknown_mult = float(os.environ.get("BTP_PERATTACK_ADJUST_UNKNOWN_MULT", "1.12"))
                        unknown_add_cap = float(os.environ.get("BTP_PERATTACK_ADJUST_UNKNOWN_ADD_CAP", "0.08"))
                        unknown_min = float(os.environ.get("BTP_PERATTACK_ADJUST_UNKNOWN_MIN", "0.0"))
                        unknown_max = float(os.environ.get("BTP_PERATTACK_ADJUST_UNKNOWN_MAX", "999.0"))
                        for group in optimizer.param_groups:
                            group["lr"] = max(float(group["lr"]) * lr_mult, runtime_learning_rate * lr_floor_mult)
                        strict_consistency_weight = min(
                            strict_max,
                            min(strict_consistency_weight * strict_mult, strict_consistency_weight + strict_add_cap),
                        )
                        strict_bank_known_weight = min(
                            known_max,
                            max(strict_bank_known_weight * known_mult, known_min),
                        )
                        strict_bank_unknown_weight = min(
                            unknown_max,
                            max(
                                unknown_min,
                                min(strict_bank_unknown_weight * unknown_mult, strict_bank_unknown_weight + unknown_add_cap),
                            ),
                        )
                        if gate_log_style != "concise":
                            print(
                                f"  [PerAttackGate] continuing: patience window active, "
                                f"next epoch will test whether geometry improves."
                            )
                        print(
                            f"  [PerAttackAdjust] lr {old_lr:.2e}->{optimizer.param_groups[0]['lr']:.2e} | "
                            f"strict_cons {strict_consistency_weight:.3f} | "
                            f"known_w {strict_bank_known_weight:.3f} | "
                            f"unknown_w {strict_bank_unknown_weight:.3f}"
                        )
                    else:
                        perattack_gate_bad_streak = 0
                        perattack_gate_best_score = max(perattack_gate_best_score, gate_score)
                    drop_stop = float(os.environ.get("BTP_PERATTACK_DROP_STOP", "3.0"))
                    if (
                        best_perattack_epoch > 0
                        and epoch > best_perattack_epoch
                        and (best_perattack_metric - metric_value) > drop_stop
                    ):
                        perattack_drop_bad_streak += 1
                        print(
                            f"  [PerAttackStop] epoch {epoch} {metric_name}_avg {metric_value:.2f} "
                            f"dropped by {best_perattack_metric - metric_value:.2f} from best epoch "
                            f"{best_perattack_epoch} ({best_perattack_metric:.2f}) | "
                            f"best_eval {best_perattack_path} | "
                            f"drop_streak {perattack_drop_bad_streak}/{gate_patience}"
                        )
                        if perattack_drop_bad_streak >= gate_patience:
                            break
                        print(
                            f"  [PerAttackStop] continuing: patience window active, "
                            f"waiting for recovery trend."
                        )
                    else:
                        perattack_drop_bad_streak = 0
            else:
                print("  [Harvest] No harvest stats returned this epoch.")
                print(f"  [Store] epoch {epoch} updated | added_known 0 | added_unknown 0 | store_known_total 0 | store_unknown_total 0")
                if continue_until_pool_done and len(manifest) < max(1, run_batch_size):
                    active_left, unseen_left = _remaining_pool_counts()
                    if active_left > 0:
                        rebuild_target = max(run_batch_size, active_left)
                        rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                        if len(rebuilt) > len(manifest):
                            manifest = rebuilt
                            save_manifest(manifest, manifest_path)
                            print(
                                f"  [Continue] Rebuilt manifest from active pool: {len(manifest)} pairs | "
                                f"active left {active_left} | unseen left {unseen_left}"
                            )
            if (
                eval_stats["known_total"] >= stop_known_total
                and eval_stats["unknown_total"] >= stop_unknown_total
                and eval_stats["known_correct"] >= stop_known_correct
                and eval_stats["unknown_correct"] >= stop_unknown_correct
            ):
                print(
                    f"  [Target Reached] known {eval_stats['known_correct']}/{eval_stats['known_total']} "
                    f"| unknown {eval_stats['unknown_correct']}/{eval_stats['unknown_total']}"
                )
                break
        active_left, unseen_left = _remaining_pool_counts()
        if MIN_REMAINING_PAIRS and len(manifest) <= MIN_REMAINING_PAIRS and not (
            continue_until_pool_done and active_left > 0
        ):
            print(f"\n✅ Reached target remaining pairs ({len(manifest)} <= {MIN_REMAINING_PAIRS}). Stopping.")
            break
        if len(manifest) == 0:
            if continue_until_pool_done and active_left > 0:
                rebuild_target = max(run_batch_size, active_left)
                rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                if rebuilt:
                    manifest = rebuilt
                    save_manifest(manifest, manifest_path)
                    print(
                        f"\n[Continue] Rebuilt manifest from active pool: {len(manifest)} pairs | "
                        f"active left {active_left} | unseen left {unseen_left}"
                    )
                    continue
            print("\n✅ All pairs mastered! Training complete.")
            break
    # ─────────────────────────────────────────────────────
    # FINAL SAVE
    # ─────────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_tag}_final.pt")
    save_checkpoint(model, optimizer, epoch, avg_total, final_path)
    if soup_enabled and soup_state is not None:
        final_soup_path = os.path.join(CHECKPOINT_DIR, f"{ckpt_tag}_soup_final.pt")
        save_checkpoint_from_state(
            soup_state,
            optimizer,
            epoch,
            avg_total,
            final_soup_path,
            extra={"soup_count": soup_count, "soup_start_epoch": soup_start_epoch},
        )
        print(f"✅ Training finished. Final soup model: {final_soup_path}")
    cleanup_epoch_checkpoints(CHECKPOINT_DIR, "convnext")
    print(f"\n✅ Training finished. Final model: {final_path}")
    logger.log_training_complete(final_path)


if __name__ == "__main__":
    train_convnext()
