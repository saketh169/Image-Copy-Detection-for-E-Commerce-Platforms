"""
train_resnet.py — Training Loop for Siamese ResNet-50
=======================================================

ResNet-50 (Residual Network with 50 layers) — CNN-based approach.

Architecture:
  Input: 224×224×3 image
  Backbone: ResNet-50 (from scratch, no pretrained weights)
    - Layer 3 (14×14×1024): Local feature extraction
    - Layer 4 (7×7×2048): Global feature extraction
  Output: 384-dim embedding (256 global + 128 local)

Mathematical Pipeline Per Batch:
  ─────────────────────────────────

  1. FORWARD PASS:
     emb1, emb2, g1, g2, l1, l2 = model(img1_batch, img2_batch)
     
     where:
       g1, g2 ∈ ℝ^(batch, 256)  — global embeddings
       l1, l2 ∈ ℝ^(batch, 128)  — local embeddings

  2. LOSS COMPUTATION:
     L = L_global + 0.5 × L_local
     
     For each pair:
       D = ||emb1 - emb2||₂  (Euclidean distance)
       
       if label == 1 (same product):
           L_pair = D²
       else (different product):
           L_pair = max(0, margin - D)²
     
     L = mean(L_pair) over all pairs in batch

  3. BACKWARD PASS:
     ∇ = d(L) / dθ  (gradients w.r.t. all parameters)
     θ ← θ - α∇  (Adam optimizer update)

  4. PRIORITY UPDATE:
     For each pair in batch:
       priority[i] = distance[i]  (store loss as priority)
       times_trained[i] += 1

Per-Epoch Decay & Pruning:
  ────────────────────────

  For each pair after epoch:
    decay_penalty = 0.9 ^ times_trained
    priority ← priority × decay_penalty
    
    if times_trained > 0 AND priority < 0.05:
        REMOVE pair from manifest  (assumed learned)

This ensures:
  - Pairs trained at least once before pruning
  - Repeated pairs get exponentially lower priority
  - Easy pairs (low priority) eventually removed
  - Hard pairs (high priority) stay in rotation

Architecture Advantages:
  ✓ Strong baseline, well-understood
  ✓ Efficient convolutional operations
  ✓ Inductive bias for image structure (local→global hierarchy)
  ✓ Fast forward/backward passes
"""

import sys
import os

# Add parent directory to Python path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.backends.cudnn.benchmark = True
import torch.optim as optim
import json
import time
import csv
import math
from pathlib import Path
import random
from collections import Counter, defaultdict
import hashlib

from config import (
    DEVICE, MANIFEST_PATH, CHECKPOINT_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    DECAY_FACTOR, PRUNE_THRESHOLD, MAX_TRAININGS, KNOWN_WEIGHT, CONTRASTIVE_WEIGHT,
    KNOWN_POS_WEIGHT, KNOWN_NEG_WEIGHT, DISABLE_PRUNING, MIN_REMAINING_PAIRS
    , TRAINED_TOP_LEVEL_CATEGORIES
)
from models.model_resnet import SiameseResNet50
from loss import DualBranchContrastiveLoss
from dataloader import SiameseDataset, get_transform
from priority_sampler import PrioritySampler
from train_logger import TrainingLogger
from torch.utils.data import DataLoader

def _category_for_path(path):
    parts = Path(path).parts
    if "Data Processed" in parts:
        idx = parts.index("Data Processed")
        if idx + 2 < len(parts):
            return "/".join(parts[idx + 1:-2])
    return None

def _load_image_for_eval(path, transform):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img).unsqueeze(0).to(DEVICE)

def _eval_attack_names():
    preferred = os.environ.get(
        "BTP_EVAL_ATTACKS",
        os.environ.get(
            "BTP_HARD_POS_COVERAGE_ATTACKS",
            "rotate,crop,bright,flip,resize_compress,watermark_text",
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


def _apply_eval_attack(img_bgr, attack_name):
    import cv2
    import numpy as np

    if not attack_name:
        return img_bgr
    h, w = img_bgr.shape[:2]
    if attack_name == "rotate":
        degrees = random.choice([-40, -35, -30, -25, -20, -15, -10, -5, 5, 10, 15, 20, 25, 30, 35, 40])
        m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        return cv2.warpAffine(img_bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if attack_name == "rotate15":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
        return cv2.warpAffine(img_bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if attack_name == "crop":
        keep_ratio = random.uniform(0.60, 0.90)
        new_h, new_w = max(2, int(h * keep_ratio)), max(2, int(w * keep_ratio))
        top = random.randint(0, max(0, h - new_h))
        left = random.randint(0, max(0, w - new_w))
        return cv2.resize(img_bgr[top:top + new_h, left:left + new_w], (w, h), interpolation=cv2.INTER_LINEAR)
    if attack_name == "crop10":
        y1, x1 = int(h * 0.10), int(w * 0.10)
        y2, x2 = int(h * 0.90), int(w * 0.90)
        return cv2.resize(img_bgr[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)
    if attack_name == "bright":
        alpha = random.uniform(1.12, 1.45)
        beta = random.uniform(6.0, 24.0)
        return np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if attack_name == "flip":
        return cv2.flip(img_bgr, 1)
    if attack_name == "resize_compress":
        small = cv2.resize(img_bgr, (max(32, w // 2), max(32, h // 2)), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        ok, enc = cv2.imencode(".jpg", up, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                return dec
        return up
    if attack_name == "watermark_text":
        out = img_bgr.copy()
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - max(28, h // 8)), (w, h), (255, 255, 255), -1)
        out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
        cv2.putText(
            out,
            "SALE 50% OFF",
            (max(6, w // 20), h - max(10, h // 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, w / 420),
            (20, 20, 20),
            max(1, w // 220),
            cv2.LINE_AA,
        )
        return out
    return img_bgr


def _embed_paths_for_eval(model, paths, transform, emb_scale, batch_size, apply_attacks=False, epoch=0):
    import cv2
    tensors = []
    embs = []
    model_was_training = model.training
    model.eval()
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        if apply_attacks:
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


def _nearest_dists_chunked(query_embeddings, db_embeddings, chunk_size):
    if query_embeddings is None or db_embeddings is None or len(query_embeddings) == 0 or len(db_embeddings) == 0:
        return []
    results = []
    for start in range(0, len(query_embeddings), chunk_size):
        chunk = query_embeddings[start:start + chunk_size]
        dists = torch.cdist(chunk, db_embeddings).min(dim=1).values.detach().cpu().tolist()
        results.extend(dists)
    return results


def _nearest_self_excluded_dists_chunked(query_embeddings, db_embeddings, chunk_size):
    if query_embeddings is None or db_embeddings is None or len(query_embeddings) == 0 or len(db_embeddings) == 0:
        return []
    results = []
    for start in range(0, len(query_embeddings), chunk_size):
        end = min(start + chunk_size, len(query_embeddings))
        chunk = query_embeddings[start:end]
        dists = torch.cdist(chunk, db_embeddings)
        row_idx = torch.arange(end - start, device=dists.device)
        col_idx = torch.arange(start, end, device=dists.device)
        dists[row_idx, col_idx] = float("inf")
        results.extend(dists.min(dim=1).values.detach().cpu().tolist())
    return results


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
    return data


def _save_embedding_store(path, data):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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
    processed = set(store.get("processed_paths", []))
    known_rows = [r for r in source_pool if r.get("known") and r.get("path") not in processed]
    unknown_rows = [r for r in source_pool if (not r.get("known")) and r.get("path") not in processed]
    random.seed(20260416 + len(source_pool))
    random.shuffle(known_rows)
    random.shuffle(unknown_rows)
    active_pool = known_rows[:target_known] + unknown_rows[:target_unknown]
    os.makedirs(os.path.dirname(active_path), exist_ok=True)
    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(active_pool, f, indent=2)


def _harvest_correct_embeddings(model, transform, epoch, eval_stats=None):
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

    db_items, known_queries, unknown_queries = _current_eval_split(active_pool)
    if not db_items or (not known_queries and not unknown_queries):
        return None

    emb_scale = float(os.environ.get("BTP_EMB_SCALE", "1.0"))
    eval_batch_size = int(os.environ.get("BTP_EVAL_BATCH_SIZE", "32"))
    dist_chunk_size = int(os.environ.get("BTP_EVAL_DIST_CHUNK", "256"))
    apply_eval_attacks = os.environ.get("BTP_EVAL_APPLY_ATTACKS", "1") == "1"
    db_paths = [r["path"] for r in db_items]
    known_paths = [r["path"] for r in known_queries]
    unknown_paths = [r["path"] for r in unknown_queries]

    db_embeddings = _embed_paths_for_eval(
        model, db_paths, transform, emb_scale, eval_batch_size, apply_attacks=False, epoch=epoch
    )
    known_embeddings = (
        _embed_paths_for_eval(
            model, known_paths, transform, emb_scale, eval_batch_size,
            apply_attacks=apply_eval_attacks, epoch=epoch
        )
        if known_paths else None
    )
    unknown_embeddings = (
        _embed_paths_for_eval(
            model, unknown_paths, transform, emb_scale, eval_batch_size,
            apply_attacks=apply_eval_attacks, epoch=epoch
        )
        if unknown_paths else None
    )
    if db_embeddings is None:
        return None

    store = _load_embedding_store(store_path)
    processed = set(store.get("processed_paths", []))
    candidate_known = store.get("candidate_known", {})
    candidate_unknown = store.get("candidate_unknown", {})
    if eval_stats is not None:
        dup_threshold = float(eval_stats.get("dup_threshold", 0.0))
        not_threshold = float(eval_stats.get("not_threshold", dup_threshold))
    else:
        dup_threshold = float(os.environ.get("BTP_DUP_THRESHOLD", "0.0"))
        not_threshold = float(os.environ.get("BTP_NOT_THRESHOLD", str(dup_threshold)))
    min_streak = max(1, int(os.environ.get("BTP_HARVEST_MIN_STREAK", "2")))
    known_margin = float(os.environ.get("BTP_HARVEST_KNOWN_MARGIN", "0.03"))
    unknown_margin = float(os.environ.get("BTP_HARVEST_UNKNOWN_MARGIN", "0.03"))
    pending_known = 0
    pending_unknown = 0

    accepted_known = []
    kept_known = []
    if known_queries and known_embeddings is not None:
        known_dists = _nearest_dists_chunked(known_embeddings, db_embeddings, dist_chunk_size)
        for row, emb, dist in zip(known_queries, known_embeddings, known_dists):
            path = row["path"]
            if dist <= (dup_threshold - known_margin):
                streak = int(candidate_known.get(path, 0)) + 1
                candidate_known[path] = streak
                if streak >= min_streak:
                    accepted_known.append({
                        "epoch": epoch,
                        "path": path,
                        "category": row.get("category", ""),
                        "known": True,
                        "distance": round(float(dist), 6),
                        "embedding": [round(float(x), 6) for x in emb.tolist()],
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
        unknown_dists = _nearest_dists_chunked(unknown_embeddings, db_embeddings, dist_chunk_size)
        for row, emb, dist in zip(unknown_queries, unknown_embeddings, unknown_dists):
            path = row["path"]
            if dist >= (not_threshold + unknown_margin):
                streak = int(candidate_unknown.get(path, 0)) + 1
                candidate_unknown[path] = streak
                if streak >= min_streak:
                    accepted_unknown.append({
                        "epoch": epoch,
                        "path": path,
                        "category": row.get("category", ""),
                        "known": False,
                        "distance": round(float(dist), 6),
                        "embedding": [round(float(x), 6) for x in emb.tolist()],
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
    }
    _save_embedding_store(store_path, store)

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
    }


def _drop_processed_pairs_from_manifest(manifest, processed_paths):
    if not processed_paths:
        return manifest, 0
    processed = set(processed_paths)
    kept = []
    removed = 0
    for row in manifest:
        p1 = row.get("path_1", "")
        p2 = row.get("path_2", "")
        if p1 in processed or p2 in processed:
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
    if not source_path or not os.path.exists(source_path):
        return []

    with open(source_path, "r", encoding="utf-8") as f:
        source_pool = json.load(f)
    active_pool = []
    if active_path and os.path.exists(active_path):
        with open(active_path, "r", encoding="utf-8") as f:
            active_pool = json.load(f)
    store = _load_embedding_store(store_path)
    processed = set(store.get("processed_paths", []))
    active_paths = {row.get("path") for row in active_pool}

    active_rows = [row for row in active_pool if row.get("path") not in processed]
    remaining_rows = active_rows if active_rows else [row for row in source_pool if row.get("path") not in processed]
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

    def choose_product(pool):
        if not pool:
            return None
        active_first = [p for p in pool if product_has_active.get(p)]
        if active_first and (len(active_first) == len(pool) or rng.random() < 0.75):
            pool = active_first
        return rng.choice(pool)

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


def _eval_db_retrieval(model, transform):
    pool_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    if not pool_path or not os.path.exists(pool_path):
        print("  [Eval] Skipping DB retrieval eval (no BTP_EVAL_DB_POOL)")
        return None

    with open(pool_path, "r") as f:
        pool = json.load(f)

    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    max_known = int(os.environ.get("BTP_EVAL_MAX_KNOWN", "1000"))
    max_unknown = int(os.environ.get("BTP_EVAL_MAX_UNKNOWN", "1000"))
    eval_batch_size = int(os.environ.get("BTP_EVAL_BATCH_SIZE", "32"))
    use_all_known = os.environ.get("BTP_EVAL_USE_ALL_KNOWN", "1") == "1"
    apply_eval_attacks = os.environ.get("BTP_EVAL_APPLY_ATTACKS", "1") == "1"
    current_epoch = int(os.environ.get("BTP_CURRENT_EPOCH", "0"))

    random.seed(20260413 + len(pool))
    random.shuffle(known)
    if use_all_known:
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

    if db_embeddings is None or len(db_embeddings) == 0:
        print(
            f"  [Eval] DB Retrieval: known 0/0 | unknown 0/0 | "
            f"dup_thr {default_dup:.2f} | not_thr {default_not:.2f} | "
            f"known_avg 0.00 | unknown_avg 0.00 | gray known 0 | gray unknown 0"
        )
        return _empty_eval_stats(default_dup, default_not)

    if use_all_known and known_embeddings is not None and len(known_embeddings) == len(db_embeddings):
        known_dists = _nearest_self_excluded_dists_chunked(
            known_embeddings, db_embeddings, 256 if len(db_embeddings) == 0 else len(db_embeddings)
        )
    else:
        known_dists = _nearest_dists_chunked(
            known_embeddings, db_embeddings, 256 if len(db_embeddings) == 0 else len(db_embeddings)
        )
    unknown_dists = _nearest_dists_chunked(unknown_embeddings, db_embeddings, 256 if len(db_embeddings) == 0 else len(db_embeddings))

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
    if dup_thr_env is not None and not_thr_env is not None:
        dup_threshold = float(dup_thr_env)
        not_threshold = float(not_thr_env)
    else:
        if known_dists and unknown_dists:
            dup_threshold = known_avg + buffer
            not_threshold = unknown_avg - buffer
            if not_threshold < dup_threshold:
                mid = (known_avg + unknown_avg) / 2.0
                dup_threshold = mid
                not_threshold = mid
        elif known_dists:
            dup_threshold = known_avg
            not_threshold = known_avg
        elif unknown_dists:
            dup_threshold = unknown_avg
            not_threshold = unknown_avg
        else:
            dup_threshold = default_dup
            not_threshold = default_not

    known_correct = sum(1 for d in known_dists if d <= dup_threshold)
    unknown_correct = sum(1 for d in unknown_dists if d >= not_threshold)
    known_ambig = sum(1 for d in known_dists if dup_threshold < d < not_threshold)
    unknown_ambig = sum(1 for d in unknown_dists if dup_threshold < d < not_threshold)

    print(
        f"  [Eval] DB Retrieval: known {known_correct}/{len(known_dists)} | "
        f"unknown {unknown_correct}/{len(unknown_dists)} | "
        f"dup_thr {dup_threshold:.2f} | not_thr {not_threshold:.2f} | "
        f"known_avg {known_avg:.2f} | unknown_avg {unknown_avg:.2f} | "
        f"gray known {known_ambig} | gray unknown {unknown_ambig}"
    )
    return {
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
    }


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
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)



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

def train_resnet():
    """
    Main training loop for Siamese ResNet-50.
    
    Algorithm:
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
    # Initialize logger
    logger = TrainingLogger(model_name='resnet')
    # NOTE: Appending to existing log file (not clearing) to preserve previous runs
    logger.log_session_start("Siamese ResNet-50")
    
    print("="*60)
    print("  TRAINING: Siamese ResNet-50")
    print("="*60)

    # ─────────────────────────────────────────────────────
    # SETUP: Initialize model, optimizer, loss
    # ─────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Override MANIFEST_PATH to use training dataset (model-specific)
    # Allow explicit override per run via env to avoid collisions in parallel runs.
    manifest_path = os.environ.get(
        "BTP_MANIFEST_PATH",
        os.path.join(os.path.dirname(MANIFEST_PATH), "training_resnet_pairs.json")
    )
    manifest = load_manifest(manifest_path)
    print(f"📋 Manifest loaded: {len(manifest):,} pairs")
    
    # Count pairs by type
    type_counts = {}
    for p in manifest:
        ptype = p.get('pair_type', 'unknown')
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    print(f"   Breakdown: {type_counts}")

    # Initialize ResNet-50 model (FROM SCRATCH, not pretrained)
    model = SiameseResNet50().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: SiameseResNet50 on {DEVICE}")
    print(f"   Total parameters: {total_params:,}")

    # Loss function: L = L_global + 0.5 × L_local
    criterion = DualBranchContrastiveLoss().to(DEVICE)
    known_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none").to(DEVICE)

    # Known category set (for unknown boosting)
    known_env = os.environ.get("BTP_KNOWN_CATEGORIES", "")
    known_json = os.environ.get("BTP_KNOWN_CATEGORIES_JSON", "")
    if known_json and os.path.exists(known_json):
        with open(known_json, "r", encoding="utf-8") as f:
            known_set = set(json.load(f))
    elif known_env.strip():
        known_set = {c.strip() for c in known_env.split(",") if c.strip()}
    else:
        known_set = set(TRAINED_TOP_LEVEL_CATEGORIES)

    _ensure_active_eval_pool()

    # Adaptive margin settings (optional)
    adapt_margin = os.environ.get("BTP_ADAPT_MARGIN", "0") == "1"
    margin_step = float(os.environ.get("BTP_MARGIN_STEP", "0.2"))
    margin_min = float(os.environ.get("BTP_MARGIN_MIN", "1.0"))
    margin_max = float(os.environ.get("BTP_MARGIN_MAX", "6.0"))
    target_gap = float(os.environ.get("BTP_TARGET_GAP", "0.6"))
    target_known_max = float(os.environ.get("BTP_TARGET_KNOWN_MAX", "2.0"))
    target_unknown_acc = float(os.environ.get("BTP_TARGET_UNKNOWN_ACC", "0.5"))
    unknown_boost = float(os.environ.get("BTP_UNKNOWN_HARD_BOOST", "1.5"))
    unknown_push_weight = float(os.environ.get("BTP_UNKNOWN_PUSH_WEIGHT", "0.5"))
    unknown_only = os.environ.get("BTP_UNKNOWN_ONLY", "0").strip() == "1"
    pos_weight_env = float(os.environ.get("BTP_POS_WEIGHT", "1.0"))
    other_neg_weight_env = float(os.environ.get("BTP_OTHER_NEG_WEIGHT", "1.0"))
    unknown_neg_weight_env = float(os.environ.get("BTP_UNKNOWN_NEG_WEIGHT", "1.0"))
    emb_scale = float(os.environ.get("BTP_EMB_SCALE", "1.0"))
    dynamic_cap_loss_threshold = float(os.environ.get("BTP_DYNAMIC_CAP_LOSS_THRESHOLD", "0"))
    prune_known_max_dist = float(os.environ.get("BTP_PRUNE_KNOWN_MAX_DIST", "8.0"))
    prune_unknown_min_dist = float(os.environ.get("BTP_PRUNE_UNKNOWN_MIN_DIST", "14.0"))

    # Optimizer: Adam with weight decay (L2 regularization)
    # Adam maintains per-parameter learning rates, good for mixed scales
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    run_epochs = int(os.environ.get("BTP_NUM_EPOCHS", str(NUM_EPOCHS)))
    run_batch_size = int(os.environ.get("BTP_BATCH_SIZE", str(BATCH_SIZE)))
    max_batches_per_epoch = int(os.environ.get("BTP_MAX_BATCHES_PER_EPOCH", "0"))
    run_num_workers = int(os.environ.get("BTP_NUM_WORKERS", "2"))
    run_prefetch_factor = int(os.environ.get("BTP_PREFETCH_FACTOR", "2"))
    initial_manifest_target = int(os.environ.get("BTP_MANIFEST_TARGET_PAIRS", str(len(manifest))))
    continue_until_pool_done = os.environ.get("BTP_CONTINUE_UNTIL_POOL_DONE", "1") == "1"
    start_epoch = 1
    prev_epoch_avg_total = float("inf")
    resume_ckpt = os.environ.get("BTP_RESUME_CKPT", "")
    if resume_ckpt and os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        prev_epoch_avg_total = float(checkpoint.get("avg_total_loss", prev_epoch_avg_total))
        print(f"Resuming from {resume_ckpt} at epoch {start_epoch}")

    # ─────────────────────────────────────────────────────
    # MAIN LOOP: iterate over epochs
    # ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, run_epochs + 1):
        epoch_start = time.time()
        effective_max_batches = max_batches_per_epoch
        if dynamic_cap_loss_threshold > 0 and prev_epoch_avg_total > dynamic_cap_loss_threshold:
            effective_max_batches = 0
        print(f"\n{'─'*60}")
        print(f"EPOCH {epoch}/{run_epochs} | Pairs: {len(manifest):,}")
        print(f"{'─'*60}")

        if dynamic_cap_loss_threshold > 0:
            cap_msg = "full epoch" if effective_max_batches == 0 else f"cap {effective_max_batches}"
            print(
                f"  [Dynamic Cap] prev avg loss {prev_epoch_avg_total:.4f} | "
                f"threshold {dynamic_cap_loss_threshold:.4f} -> {cap_msg}"
            )
        # Log epoch header
        os.environ["BTP_CURRENT_EPOCH"] = str(epoch)
        logger.log_epoch_header(manifest, model, DEVICE, epoch, run_epochs, "Siamese ResNet-50")

        # Create dataset and sampler for this epoch
        dataset = SiameseDataset(
            manifest_path=manifest_path,
            transform=get_transform(),
            return_known_labels=True
        )
        sampler = PrioritySampler(manifest)

        dataloader = DataLoader(
            dataset,
            batch_size=run_batch_size,
            sampler=sampler,
            num_workers=run_num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=run_prefetch_factor,
            drop_last=True,
        )

        model.train()  # Set to training mode (enables dropout, etc.)
        
        # Track statistics across epoch
        epoch_total_loss = 0.0
        epoch_global_loss = 0.0
        epoch_local_loss = 0.0
        num_batches = 0
        pairs_trained_this_epoch = 0

        # ──────────────────────────────────────────────
        # BATCH LOOP: iterate over mini-batches
        # ──────────────────────────────────────────────
        for batch_idx, (img1, img2, labels, indices, known1, known2) in enumerate(dataloader):
            if effective_max_batches and batch_idx >= effective_max_batches:
                break
            # Move to device (GPU/CPU)
            img1   = img1.to(DEVICE, non_blocking=True)
            img2   = img2.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            known1 = known1.to(DEVICE, non_blocking=True)
            known2 = known2.to(DEVICE, non_blocking=True)

            # ── 1. FORWARD PASS ──
            # Both images through shared ResNet backbone
            # Returns: (emb1, emb2, global1, global2, local1, local2)
            emb1, emb2, global1, global2, local1, local2 = model(img1, img2)
            global1_s = global1 * emb_scale
            global2_s = global2 * emb_scale
            local1_s = local1 * emb_scale
            local2_s = local2 * emb_scale

            # ── 2. COMPUTE LOSS ──
            # L_total = L_global + 0.5 × L_local
            # Distances are used for priority updates
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
            contrastive_loss, global_loss, local_loss, distances = criterion(
                global1_s, global2_s, local1_s, local2_s, labels, sample_weights=sample_weights
            )
            known_logits1 = model.known_head(emb1).squeeze(1)
            known_logits2 = model.known_head(emb2).squeeze(1)
            known_loss_1 = known_loss_fn(known_logits1, known1)
            known_loss_2 = known_loss_fn(known_logits2, known2)
            weights_1 = torch.where(known1 > 0.5, KNOWN_POS_WEIGHT, KNOWN_NEG_WEIGHT)
            weights_2 = torch.where(known2 > 0.5, KNOWN_POS_WEIGHT, KNOWN_NEG_WEIGHT)
            known_loss = 0.5 * (
                (known_loss_1 * weights_1).mean() + (known_loss_2 * weights_2).mean()
            )
            # Non-saturating push-away loss for known-vs-unknown negatives
            if unknown_mask.any():
                unknown_dist = distances[unknown_mask]
                # Encourage larger distances even after margin
                unknown_push = (1.0 / (unknown_dist + 1e-6)).mean()
            else:
                unknown_push = torch.tensor(0.0, device=DEVICE)

            total_loss = (
                (CONTRASTIVE_WEIGHT * contrastive_loss)
                + (KNOWN_WEIGHT * known_loss)
                + (unknown_push_weight * unknown_push)
            )

            # ── 3. BACKWARD PASS ──
            # Compute gradients and update weights
            optimizer.zero_grad()  # Clear old gradients
            total_loss.backward()  # Backpropagation: compute ∇
            optimizer.step()       # Update: θ ← θ - α∇

            # ── 4. UPDATE MANIFEST ──
            # Store loss value as new priority for each pair
            # Priority reflects how "hard" the pair was to learn
            with torch.no_grad():
                for i, manifest_idx in enumerate(indices):
                    idx = manifest_idx.item()
                    if idx < len(manifest):
                        pair_loss = distances[i].item()
                        manifest[idx]['priority'] = round(pair_loss, 4)
                        manifest[idx]['last_distance'] = round(pair_loss, 4)
                        manifest[idx]['times_trained'] += 1

            # ── 5. TRACK STATISTICS ──
            epoch_total_loss  += total_loss.item()
            epoch_global_loss += global_loss.item()
            epoch_local_loss  += local_loss.item()
            num_batches += 1
            pairs_trained_this_epoch += len(labels)

            avg_loss = epoch_total_loss / num_batches
            print(f"  Batch {batch_idx+1:>5} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Global: {global_loss.item():.4f} | "
                  f"Local: {local_loss.item():.4f} | "
                  f"Known: {known_loss.item():.4f}")
            
            # Log batch to file
            logger.log_batch(batch_idx+1, avg_loss, global_loss.item(), local_loss.item())
            
            # Save actual batch composition to live CSV
            csv_path = os.path.join(os.path.dirname(MANIFEST_PATH), "manifest_live_resnet.csv")
            save_batch_csv(manifest, indices, csv_path, epoch=epoch, batch_num=batch_idx+1)

        # ──────────────────────────────────────────────
        # POST-EPOCH: Decay, prune, checkpoint
        # ──────────────────────────────────────────────
        if num_batches == 0:
            active_left, unseen_left = _remaining_pool_counts()
            if continue_until_pool_done and active_left > 0:
                rebuild_target = active_left if active_left > 0 else unseen_left
                rebuild_target = max(run_batch_size, rebuild_target)
                rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                if rebuilt:
                    manifest = rebuilt
                    save_manifest(manifest, manifest_path)
                    print(
                        f"  [Continue] Rebuilt manifest from remaining pool: {len(manifest)} pairs | "
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
        # More trained pairs get lower priority = less likely sampled
        print(f"\n  Applying decay: priority × 0.9^times_trained")
        for pair in manifest:
            times = pair['times_trained']
            if times > 0:
                decay_penalty = DECAY_FACTOR ** times
                old_priority = pair['priority']
                pair['priority'] = round(pair['priority'] * decay_penalty, 4)
                if pair['priority'] < 0.10:  # Log pairs near removal threshold
                    diff = old_priority - pair['priority']

        # ── PRUNING STEP ──
        # Remove pairs that are well-learned (soft pairs with low loss)
        # Curriculum: Soft pairs learn faster, remove after 2 trainings if priority < 1.0
        #            Hard pairs need more training, always keep
        #            Extreme cases: Remove any pair with priority < 0.5 (trivial)
        pruned_count = 0
        prune_after_epochs = int(os.environ.get("BTP_PRUNE_AFTER_EPOCHS", "5"))
        disable_pruning = DISABLE_PRUNING or os.environ.get("BTP_DISABLE_PRUNING", "0") == "1"
        if disable_pruning:
            pruning_rule = "DISABLED (harvest-only removal)"
            print(f"  Pruning pairs: {pruning_rule}")
            logger.log_pruning_info(pruning_rule)
        elif epoch < prune_after_epochs:
            pruning_rule = f"DELAYED (epoch<{prune_after_epochs})"
            print(f"  Pruning pairs: {pruning_rule}")
            logger.log_pruning_info(pruning_rule)
        else:
            pruning_rule = (
                f"DISTANCE-BASED (known_pos<= {prune_known_max_dist:.2f}) "
                f"OR (known_vs_unknown_neg>= {prune_unknown_min_dist:.2f}), times>=2"
            )
            print(f"  Pruning pairs: {pruning_rule}")
            logger.log_pruning_info(pruning_rule)
            before_count = len(manifest)
            kept = []
            for p in manifest:
                times = p.get('times_trained', 0)
                if times < 2:
                    kept.append(p)
                    continue
                dist = float(p.get('last_distance', p.get('priority', 1e9)))
                c1 = _category_for_path(p.get("path_1", ""))
                c2 = _category_for_path(p.get("path_2", ""))
                is_known_known = (c1 in known_set) and (c2 in known_set)
                is_known_unknown = (c1 in known_set) != (c2 in known_set)
                label = float(p.get("label", 0.0))
                prune_it = False
                if label > 0.5 and is_known_known and dist <= prune_known_max_dist:
                    prune_it = True
                if label < 0.5 and is_known_unknown and dist >= prune_unknown_min_dist:
                    prune_it = True
                if not prune_it:
                    kept.append(p)
            manifest = kept
            pruned_count = before_count - len(manifest)

        # ── SAVE CHECKPOINT ──
        save_manifest(manifest, manifest_path)  # Save to model-specific file
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"resnet_epoch_{epoch}.pt")
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
        print(f"     Pruned: {pruned_count:,}")
        print(f"     Remaining: {len(manifest):,}")
        print(f"     By type: {type_counts}")
        print(f"     Checkpoint: {ckpt_path}")
        print(f"     Time: {elapsed:.1f}s")

        # Log epoch summary to file
        logger.log_epoch_summary(epoch, avg_total, avg_global, avg_local,
                                pairs_trained_this_epoch, pruned_count, len(manifest),
                                type_counts, elapsed, ckpt_path)        # Check if all pairs mastered
        eval_stats = _eval_db_retrieval(model, get_transform())
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

        # Boost known-vs-unknown negatives if unknown accuracy is low
        if eval_stats is not None:
            unknown_acc = eval_stats["unknown_correct"] / max(1, eval_stats["unknown_total"])
            if unknown_acc < target_unknown_acc:
                boosted = 0
                for pair in manifest:
                    if pair.get("label", 0) != 0:
                        continue
                    c1 = _category_for_path(pair.get("path_1", ""))
                    c2 = _category_for_path(pair.get("path_2", ""))
                    if (c1 in known_set) != (c2 in known_set):
                        pair["priority"] = round(pair.get("priority", 0.0) * unknown_boost, 4)
                        boosted += 1
                print(f"  [Boost] Unknown acc {unknown_acc:.3f} < {target_unknown_acc:.3f}. Boosted {boosted} known-vs-unknown negatives.")

        harvest_stats = _harvest_correct_embeddings(model, get_transform(), epoch, eval_stats=eval_stats)
        if harvest_stats is not None:
            manifest, removed_train_pairs = _drop_processed_pairs_from_manifest(
                manifest, harvest_stats.get("processed_now_paths", [])
            )
            if removed_train_pairs:
                save_manifest(manifest, manifest_path)
            print(
                f"  [Harvest] accepted known {harvest_stats['accepted_known']} | "
                f"accepted unknown {harvest_stats['accepted_unknown']} | "
                f"pending known {harvest_stats['pending_known']} | "
                f"pending unknown {harvest_stats['pending_unknown']} | "
                f"kept wrong known {harvest_stats['kept_known_wrong']} | "
                f"kept wrong unknown {harvest_stats['kept_unknown_wrong']} | "
                f"added known {harvest_stats['added_known']} | "
                f"added unknown {harvest_stats['added_unknown']} | "
                f"next active known {harvest_stats['next_active_known']} | "
                f"next active unknown {harvest_stats['next_active_unknown']} | "
                f"next active total {harvest_stats['next_active_total']} | "
                f"train pairs removed {removed_train_pairs} | "
                f"train pairs left {len(manifest)}"
            )
            if continue_until_pool_done and len(manifest) < max(1, run_batch_size):
                active_left, unseen_left = _remaining_pool_counts()
                if active_left > 0:
                    rebuild_target = active_left if active_left > 0 else unseen_left
                    rebuild_target = max(run_batch_size, rebuild_target)
                    rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                    if len(rebuilt) > len(manifest):
                        manifest = rebuilt
                        save_manifest(manifest, manifest_path)
                        print(
                            f"  [Continue] Rebuilt manifest from remaining pool: {len(manifest)} pairs | "
                            f"active left {active_left} | unseen left {unseen_left}"
                        )
        active_left, unseen_left = _remaining_pool_counts()
        if MIN_REMAINING_PAIRS and len(manifest) <= MIN_REMAINING_PAIRS and not (
            continue_until_pool_done and active_left > 0
        ):
            print(f"\n✅ Reached target remaining pairs ({len(manifest)} <= {MIN_REMAINING_PAIRS}). Stopping.")
            break
        if len(manifest) == 0:
            if continue_until_pool_done and active_left > 0:
                rebuild_target = active_left if active_left > 0 else unseen_left
                rebuild_target = max(run_batch_size, rebuild_target)
                rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                if rebuilt:
                    manifest = rebuilt
                    save_manifest(manifest, manifest_path)
                    print(
                        f"\n[Continue] Rebuilt manifest from remaining pool: {len(manifest)} pairs | "
                        f"active left {active_left} | unseen left {unseen_left}"
                    )
                    continue
            print("\n✅ All pairs mastered! Training complete.")
            break

    # ─────────────────────────────────────────────────────
    # FINAL SAVE
    # ─────────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "resnet_final.pt")
    save_checkpoint(model, optimizer, run_epochs, avg_total, final_path)
    print("[KeepCkpt] Skipping cleanup_epoch_checkpoints for resnet")
    print(f"\n✅ Training finished. Final model: {final_path}")
    logger.log_training_complete(final_path)


if __name__ == "__main__":
    train_resnet()
