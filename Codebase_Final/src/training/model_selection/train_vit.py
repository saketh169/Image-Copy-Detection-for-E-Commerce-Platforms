"""
train_vit.py — Training Loop for Siamese Vision Transformer (ViT)
==================================================================

Vision Transformer (ViT-B/16) — Transformer-based approach (no convolutions).

Architecture:
  Input: 224×224×3 image
  Process:
    1. Divide into 16×16 patches → 196 patches (14×14 grid)
    2. Linear projection: each patch → 768-dim
    3. Add learnable positional embeddings
    4. Transformer blocks with Multi-Head Self-Attention
  Backbone: ViT-B with 12 transformer blocks
    - After 6 blocks (intermediate): 196 patches → LocalBranch (128-dim)
    - After 12 blocks (CLS token): Global representation → GlobalBranch (256-dim)
  Output: 384-dim embedding (256 global + 128 local)

Key Differences vs CNNs (ResNet/ConvNeXt):
  1. No convolutions: uses pure self-attention
  2. Patch-based input: 16×16 patches instead of sliding windows
  3. Positional encoding: learned 1D position embeddings (not 2D)
  4. Sequence processing: treats image as sequence of patch embeddings
  5. Global receptive field: every patch sees every other patch from layer 1
  6. No spatial inductive bias: must learn structure from scratch

Mathematical Pipeline Per Batch:
  ─────────────────────────────────

  1. PATCH EMBEDDING:
     For each 16×16 patch:
       patch_vec = flatten(patch)  ∈ ℝ^768  [16×16×3 → 768]
       patch_embed = Linear(patch_vec)  ∈ ℝ^768
     prepend CLS token: [cls_token | patch_embed_1 | ... | patch_embed_196]
     add positional encoding: embed += pos_encoding
     result: (batch, 197, 768)

  2. TRANSFORMER BLOCKS:
     for each block i in 1..12:
       x_normalized = LayerNorm(x)
       
       MultiHeadAttention (12 heads):
         Q, K, V = Linear projections (768 → 768)
         scores = softmax((Q @ K^T) / √64) @ V
         output = Concat[head_1, ..., head_12] @ W_o
       
       x = x + Attention(LayerNorm(x))  [residual]
       x = x + MLP(LayerNorm(x))        [residual]
     
     if i == 6: intermediate = x[:, 1:, :]  [local features, 196 patches]

  3. GLOBAL REPRESENTATION:
     cls_final = x[:, 0, :]  [CLS token after 12 blocks, 768-dim]

  4. LOSS COMPUTATION (same as ResNet/ConvNeXt):
     L = L_global + 0.5 × L_local
     L = mean(contrastive_loss(...))

  5. BACKWARD PASS:
     ∇ = d(L) / dθ
     θ ← θ - α∇  (Adam update)

Per-Epoch Decay & Pruning:
  ────────────────────────

  Same as CNN approaches:
    priority ← priority × (0.9 ^ times_trained)
    if times_trained > 0 AND priority < 0.05: REMOVE

Architecture Advantages:
  ✓ Global receptive field from layer 1 (all patches see all patches)
  ✓ May capture global relationships better than CNNs
  ✓ Better scaling properties (quadratic in sequence length, not image size)
  ✓ Can use knowledge from Vision Transformer pretraining (if available)
  ✗ Higher memory during training (quadratic in patch count)
  ✗ Slower training than CNNs (dense attention computation)
  ✗ Needs more data typically
  ✗ Positional encoding is key — learned from scratch harder

Why ViT was chosen:
  - Pure attention mechanism for image understanding
  - No convolutional inductive bias — learn structure from data
  - Good for large datasets (815K pairs)
  - Interesting comparison baseline
  - Could use pretrained ViT weights if needed

Attention Complexity:
  Quadratic in sequence length: O(n²) where n = num_patches + 1 = 197
  Per-layer computation: 197² × 12heads × batch_size = significant GPU memory
  Mitigation: efficient attention implementations, gradient checkpointing
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
from pathlib import Path
import random
from collections import Counter, defaultdict

from config import (
    DEVICE, MANIFEST_PATH, CHECKPOINT_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    DECAY_FACTOR, PRUNE_THRESHOLD, MAX_TRAININGS, MIN_REMAINING_PAIRS,
)
from models.model_vit import SiameseViT
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

def _embed_paths_for_eval(model, paths, transform, emb_scale, batch_size):
    import cv2
    tensors = []
    embs = []
    model_was_training = model.training
    model.eval()
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
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
    outs = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        chunk = query_embeddings[start:start + chunk_size]
        outs.append(torch.cdist(chunk, db_embeddings).min(dim=1).values)
    return torch.cat(outs, dim=0).detach().cpu().tolist()


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

def _eval_db_retrieval(model, transform):
    pool_path = os.environ.get("BTP_EVAL_DB_POOL", "")
    if not pool_path or not os.path.exists(pool_path):
        print("  [Eval] Skipping DB retrieval eval (no BTP_EVAL_DB_POOL)")
        return None

    with open(pool_path, "r") as f:
        pool = json.load(f)

    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    max_known = int(os.environ.get("BTP_EVAL_MAX_KNOWN", "2000"))
    max_unknown = int(os.environ.get("BTP_EVAL_MAX_UNKNOWN", "2000"))
    eval_batch_size = int(os.environ.get("BTP_EVAL_BATCH_SIZE", "32"))
    dist_chunk_size = int(os.environ.get("BTP_EVAL_DIST_CHUNK", "256"))
    use_all_known = os.environ.get("BTP_EVAL_USE_ALL_KNOWN", "1") == "1"

    dynamic_eval_sampling = os.environ.get("BTP_EVAL_DYNAMIC_SAMPLING", "0") == "1"
    if dynamic_eval_sampling:
        random.seed(int(time.time()) + len(pool) + os.getpid())
    else:
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

    db_embeddings = _embed_paths_for_eval(model, [r["path"] for r in db_items], transform, emb_scale, eval_batch_size)
    known_embeddings = _embed_paths_for_eval(model, [r["path"] for r in known_queries], transform, emb_scale, eval_batch_size) if known_queries else None
    unknown_embeddings = _embed_paths_for_eval(model, [r["path"] for r in unknown_queries], transform, emb_scale, eval_batch_size) if unknown_queries else None

    dup_thr_env = os.environ.get("BTP_DUP_THRESHOLD")
    not_thr_env = os.environ.get("BTP_NOT_THRESHOLD")
    default_dup = float(dup_thr_env) if dup_thr_env is not None else 0.0
    default_not = float(not_thr_env) if not_thr_env is not None else default_dup

    if db_embeddings is None or db_embeddings.shape[0] == 0:
        print(
            f"  [Eval] DB Retrieval: known 0/0 | unknown 0/0 | "
            f"dup_thr {default_dup:.2f} | not_thr {default_not:.2f} | "
            f"known_avg 0.00 | unknown_avg 0.00 | gray known 0 | gray unknown 0"
        )
        return _empty_eval_stats(default_dup, default_not)

    if known_embeddings is not None:
        if use_all_known and known_embeddings.shape[0] == db_embeddings.shape[0]:
            known_dists = _nearest_self_excluded_dists_chunked(known_embeddings, db_embeddings, dist_chunk_size)
        else:
            known_dists = _nearest_dists_chunked(known_embeddings, db_embeddings, dist_chunk_size)
    else:
        known_dists = []
    unknown_dists = _nearest_dists_chunked(unknown_embeddings, db_embeddings, dist_chunk_size) if unknown_embeddings is not None else []

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

def _load_embedding_store(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("accepted_known", [])
        data.setdefault("accepted_unknown", [])
        data.setdefault("processed_paths", [])
        data.setdefault("meta", {})
        return data
    return {
        "accepted_known": [],
        "accepted_unknown": [],
        "processed_paths": [],
        "meta": {},
    }


def _save_embedding_store(path, data):
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _current_eval_split(pool):
    known = [p for p in pool if p.get("known")]
    unknown = [p for p in pool if not p.get("known")]
    db_fraction = float(os.environ.get("BTP_EVAL_DB_FRACTION", "0.7"))
    dynamic_eval_sampling = os.environ.get("BTP_EVAL_DYNAMIC_SAMPLING", "0") == "1"
    if dynamic_eval_sampling:
        random.seed(int(time.time()) + len(pool) + os.getpid())
    else:
        random.seed(20260416 + len(pool))
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
    db_paths = [r["path"] for r in db_items]
    known_paths = [r["path"] for r in known_queries]
    unknown_paths = [r["path"] for r in unknown_queries]

    db_embeddings = _embed_paths_for_eval(model, db_paths, transform, emb_scale, eval_batch_size)
    known_embeddings = _embed_paths_for_eval(model, known_paths, transform, emb_scale, eval_batch_size) if known_paths else None
    unknown_embeddings = _embed_paths_for_eval(model, unknown_paths, transform, emb_scale, eval_batch_size) if unknown_paths else None
    if db_embeddings is None:
        return None

    store = _load_embedding_store(store_path)
    processed = set(store.get("processed_paths", []))
    if eval_stats is not None:
        dup_threshold = float(eval_stats.get("dup_threshold", 0.0))
        not_threshold = float(eval_stats.get("not_threshold", dup_threshold))
    else:
        dup_threshold = float(os.environ.get("BTP_DUP_THRESHOLD", "0.0"))
        not_threshold = float(os.environ.get("BTP_NOT_THRESHOLD", str(dup_threshold)))

    accepted_known = []
    kept_known = []
    if known_queries and known_embeddings is not None:
        known_dists = _nearest_dists_chunked(known_embeddings, db_embeddings, dist_chunk_size)
        for row, emb, dist in zip(known_queries, known_embeddings, known_dists):
            if dist <= dup_threshold:
                accepted_known.append({
                    "epoch": epoch,
                    "path": row["path"],
                    "category": row.get("category", ""),
                    "known": True,
                    "distance": round(float(dist), 6),
                    "embedding": [round(float(x), 6) for x in emb.tolist()],
                })
                processed.add(row["path"])
            else:
                kept_known.append(row)

    accepted_unknown = []
    kept_unknown = []
    if unknown_queries and unknown_embeddings is not None:
        unknown_dists = _nearest_dists_chunked(unknown_embeddings, db_embeddings, dist_chunk_size)
        for row, emb, dist in zip(unknown_queries, unknown_embeddings, unknown_dists):
            if dist >= not_threshold:
                accepted_unknown.append({
                    "epoch": epoch,
                    "path": row["path"],
                    "category": row.get("category", ""),
                    "known": False,
                    "distance": round(float(dist), 6),
                    "embedding": [round(float(x), 6) for x in emb.tolist()],
                })
                processed.add(row["path"])
            else:
                kept_unknown.append(row)

    store["accepted_known"].extend(accepted_known)
    store["accepted_unknown"].extend(accepted_unknown)
    store["processed_paths"] = sorted(processed)
    store["meta"] = {
        "last_epoch": epoch,
        "accepted_known_total": len(store["accepted_known"]),
        "accepted_unknown_total": len(store["accepted_unknown"]),
        "last_added_known": len(accepted_known),
        "last_added_unknown": len(accepted_unknown),
    }
    _save_embedding_store(store_path, store)

    target_known = int(os.environ.get("BTP_EMBED_ACTIVE_KNOWN", "1500"))
    target_unknown = int(os.environ.get("BTP_EMBED_ACTIVE_UNKNOWN", "500"))
    active_paths = {r["path"] for r in db_items + kept_known + kept_unknown}
    source_known = [r for r in source_pool if r.get("known") and r.get("path") not in processed and r.get("path") not in active_paths]
    source_unknown = [r for r in source_pool if (not r.get("known")) and r.get("path") not in processed and r.get("path") not in active_paths]
    random.seed(20260416 + epoch + len(source_pool))
    random.shuffle(source_known)
    random.shuffle(source_unknown)

    current_known_total = len(db_items) + len(kept_known)
    current_unknown_total = len(kept_unknown)
    add_known = source_known[:max(0, target_known - current_known_total)]
    add_unknown = source_unknown[:max(0, target_unknown - current_unknown_total)]
    next_active = db_items + kept_known + kept_unknown + add_known + add_unknown
    with open(active_path, "w", encoding="utf-8") as f:
        json.dump(next_active, f, indent=2)

    return {
        "accepted_known": len(accepted_known),
        "accepted_unknown": len(accepted_unknown),
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

    def product_for_path(path):
        parts = Path(path).parts
        if "Data Processed" in parts:
            idx = parts.index("Data Processed")
            if idx + 3 < len(parts):
                return "/".join(parts[idx + 1:-1])
        return None

    rng = random.Random(seed)
    product_images = defaultdict(list)
    product_known = {}
    product_category = {}
    category_known_products = defaultdict(list)
    product_has_active = defaultdict(bool)

    for row in remaining_rows:
        path = row.get("path", "")
        product = row.get("product") or product_for_path(path)
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
    target_each = target_size // 4
    remainder = target_size - (target_each * 4)
    targets = {
        "hard_pos": target_each,
        "hard_neg": target_each,
        "soft_pos": target_each,
        "soft_neg": target_each + remainder,
    }
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

def train_vit():
    """
    Main training loop for Siamese Vision Transformer (ViT).
    
    Algorithm (same as ResNet/ConvNeXt but with ViT backbone):
      for epoch in 1..NUM_EPOCHS:
        1. Create PrioritySampler from current manifest
        2. For each batch:
           a. Forward pass → patch embedding → transformer blocks → loss
           b. Backward pass → update weights
           c. Update pair priorities based on loss
           d. Save batch composition to CSV
        3. Apply decay to all pair priorities
        4. Prune pairs below threshold (only if trained before)
        5. Save checkpoint and updated manifest
    """
    # Initialize logger
    logger = TrainingLogger(model_name='vit')
    # NOTE: Appending to existing log file (not clearing) to preserve previous runs
    logger.log_session_start("Siamese Vision Transformer (ViT-B/16)")
    
    print("="*60)
    print("  TRAINING: Siamese Vision Transformer (ViT-B/16)")
    print("="*60)

    # ─────────────────────────────────────────────────────
    # SETUP: Initialize model, optimizer, loss
    # ─────────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Use manifest from config/env (BTP_MANIFEST_PATH)
    manifest_path = os.environ.get(
        "BTP_MANIFEST_PATH",
        MANIFEST_PATH,
    )
    manifest = load_manifest(manifest_path)
    print(f"[INFO] Manifest loaded: {len(manifest):,} pairs")
    
    # Count pairs by type
    type_counts = {}
    for p in manifest:
        ptype = p.get('pair_type', 'unknown')
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    print(f"   Breakdown: {type_counts}")

    # Initialize ViT model (FROM SCRATCH, not pretrained)
    model = SiameseViT().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] SiameseViT (ViT-B/16) on {DEVICE}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Patch embedding: 768-dim x 197 tokens (196 patches + 1 CLS)")
    print(f"   Transformer blocks: 12 with 12 attention heads each")

    # Loss function: L = L_global + 0.5 × L_local
    criterion = DualBranchContrastiveLoss().to(DEVICE)
    run_epochs = int(os.environ.get("BTP_NUM_EPOCHS", str(NUM_EPOCHS)))
    run_batch_size = int(os.environ.get("BTP_BATCH_SIZE", str(BATCH_SIZE)))

    known_env = os.environ.get("BTP_KNOWN_CATEGORIES", "")
    known_json = os.environ.get("BTP_KNOWN_CATEGORIES_JSON", "")
    if known_json and os.path.exists(known_json):
        with open(known_json, "r", encoding="utf-8") as f:
            known_set = set(json.load(f))
    elif known_env.strip():
        known_set = {c.strip() for c in known_env.split(",") if c.strip()}
    else:
        known_set = set()

    _ensure_active_eval_pool()

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
    max_batches_per_epoch = int(os.environ.get("BTP_MAX_BATCHES_PER_EPOCH", "0"))
    eval_every = max(1, int(os.environ.get("BTP_EVAL_EVERY", "1")))
    num_workers = int(os.environ.get("BTP_NUM_WORKERS", "2"))
    prefetch_factor = int(os.environ.get("BTP_PREFETCH_FACTOR", "2"))
    prune_known_max_dist = float(os.environ.get("BTP_PRUNE_KNOWN_MAX_DIST", "8.0"))
    prune_unknown_min_dist = float(os.environ.get("BTP_PRUNE_UNKNOWN_MIN_DIST", "14.0"))
    continue_until_pool_done = os.environ.get("BTP_CONTINUE_UNTIL_POOL_DONE", "1") == "1"

    # Optimizer: Adam with weight decay (L2 regularization)
    # Note: ViT may benefit from different learning rate or schedule
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    start_epoch = 1
    resume_ckpt = os.environ.get("BTP_RESUME_CKPT", "")
    if resume_ckpt and os.path.exists(resume_ckpt):
        checkpoint = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[RESUME] {resume_ckpt} -> epoch {start_epoch}")

    # ─────────────────────────────────────────────────────
    # MAIN LOOP: iterate over epochs
    # ─────────────────────────────────────────────────────    
    for epoch in range(start_epoch, run_epochs + 1):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"EPOCH {epoch}/{run_epochs} | Pairs: {len(manifest):,}")
        print(f"{'─'*60}")

        # Log epoch header
        logger.log_epoch_header(manifest, model, DEVICE, epoch, run_epochs, "Siamese ViT (ViT-B/16)")

        # Create dataset and sampler for this epoch
        dataset = SiameseDataset(manifest_path=manifest_path, transform=get_transform(), return_known_labels=True)
        sampler = PrioritySampler(manifest)

        dataloader = DataLoader(
            dataset,
            batch_size=run_batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )

        model.train()  # Set to training mode
        
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
            if max_batches_per_epoch > 0 and batch_idx >= max_batches_per_epoch:
                print(f"  [Epoch Cap] Reached max batches per epoch: {max_batches_per_epoch}")
                break
            # Move to device (GPU/CPU)
            img1   = img1.to(DEVICE, non_blocking=True)
            img2   = img2.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            known1 = known1.to(DEVICE, non_blocking=True)
            known2 = known2.to(DEVICE, non_blocking=True)

            # ── 1. FORWARD PASS ──
            # Both images through shared ViT backbone
            # Process:
            #   1. Patch embedding + positional encoding: 224×224×3 → 197×768
            #   2. Transformer blocks: y = Attention(LayerNorm(x)) + MLP(LayerNorm(x)) + x
            #   3. Extract intermediate (after 6 blocks) for local
            #   4. Extract CLS token (after 12 blocks) for global
            # Returns: (emb1, emb2, global1, global2, local1, local2)
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
                unknown_dist = distances[unknown_mask]
                unknown_push = (1.0 / (unknown_dist + 1e-6)).mean()
            else:
                unknown_push = torch.tensor(0.0, device=DEVICE)

            total_loss = total_loss + (unknown_push_weight * unknown_push)

            # ── 3. BACKWARD PASS ──
            optimizer.zero_grad()  # Clear old gradients
            total_loss.backward()  # Backpropagation: compute ∇
            optimizer.step()       # Update: θ ← θ - α∇

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
                  f"Local: {local_loss.item():.4f}")
            
            # Log batch to file
            logger.log_batch(batch_idx+1, avg_loss, global_loss.item(), local_loss.item())
            
            # Save actual batch composition to live CSV
            csv_path = os.path.join(os.path.dirname(MANIFEST_PATH), "manifest_live_vit.csv")
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
        if disable_pruning:
            pruning_rule = "DISABLED (harvest-only removal)"
            print(f"  Pruning pairs: {pruning_rule}")
            logger.log_pruning_info(pruning_rule)
            pruned_count = 0
        elif epoch < prune_after_epochs:
            pruning_rule = f"DELAYED (epoch<{prune_after_epochs})"
            print(f"  Pruning pairs: {pruning_rule}")
            logger.log_pruning_info(pruning_rule)
            pruned_count = 0
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
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"vit_epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, avg_total, ckpt_path)

        # ── EPOCH SUMMARY ──
        elapsed = time.time() - epoch_start
        type_counts = {}
        for p in manifest:
            ptype = p.get('pair_type', 'unknown')
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        print(f"\n  [SUMMARY] Epoch {epoch} Summary:")
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
                                type_counts, elapsed, ckpt_path)
        eval_stats = None
        if epoch % eval_every == 0:
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
                        rebuild_target = max(run_batch_size, active_left)
                        rebuilt = _rebuild_manifest_from_remaining_pool(rebuild_target, epoch + 20260418)
                        if len(rebuilt) > len(manifest):
                            manifest = rebuilt
                            save_manifest(manifest, manifest_path)
                            print(
                                f"  [Continue] Rebuilt manifest from active pool: {len(manifest)} pairs | "
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
            print("\n[COMPLETE] All pairs mastered! Training complete.")
            break
    # ─────────────────────────────────────────────────────
    # FINAL SAVE
    # ─────────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "vit_final.pt")
    save_checkpoint(model, optimizer, epoch, avg_total, final_path)
    print(f"\n[DONE] Training finished. Final model: {final_path}")
    print("[KeepCkpt] Skipping cleanup_epoch_checkpoints for vit")
    logger.log_training_complete(final_path)


if __name__ == "__main__":
    train_vit()
