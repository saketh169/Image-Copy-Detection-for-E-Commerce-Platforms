"""
train.py — Main Training Loop
================================

From Plan.md Sections 3–4:

Per Epoch:
  1. Load manifest → PrioritySampler forms batches (hardest first)
  2. For each batch:
     a. Forward pass through Siamese ResNet-50
     b. Compute contrastive loss (global + λ × local)
     c. Backpropagation
  3. After each batch:
     a. Record per-pair loss into manifest as new priority
  4. After each epoch:
     a. Apply decay: priority = loss × (0.9 ^ times_trained)
     b. Prune pairs with priority < 0.05 (mastered)
     c. Increment times_trained
     d. Save updated manifest to disk
     e. Save model checkpoint

The manifest SHRINKS over training as easy pairs get pruned.
Hard positives stay longest (always have new random attacks each epoch).
"""

import torch
import torch.optim as optim
import json
import os
import time

from config import (
    DEVICE, MANIFEST_PATH, CHECKPOINT_DIR,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE,
    DECAY_FACTOR, PRUNE_THRESHOLD
)
from model import SiameseResNet50
from loss import DualBranchContrastiveLoss
from dataloader import SiameseDataset, get_transform
from priority_sampler import PrioritySampler
from torch.utils.data import DataLoader


def load_manifest(path):
    """Load the manifest JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_manifest(manifest, path):
    """Save the manifest JSON file (updated priorities, pruned pairs)."""
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model weights + optimizer state for resuming later."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def train():
    print("="*60)
    print("  IMAGE COPY DETECTION — TRAINING")
    print("="*60)

    # ─── 1. Setup ────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load manifest (contains all pair definitions with priorities)
    manifest = load_manifest(MANIFEST_PATH)
    print(f"📋 Manifest loaded: {len(manifest):,} pairs")

    # Initialize model
    model = SiameseResNet50().to(DEVICE)
    print(f"🧠 Model on {DEVICE} | Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function (global + λ × local)
    criterion = DualBranchContrastiveLoss().to(DEVICE)

    # Optimizer (Adam with weight decay for regularization)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ─── 2. Training Loop ────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS} | Pairs remaining: {len(manifest):,}")
        print(f"{'─'*60}")

        # Create dataset and sampler from current manifest
        dataset = SiameseDataset(manifest_path=MANIFEST_PATH, transform=get_transform())
        sampler = PrioritySampler(manifest)

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        model.train()
        epoch_total_loss = 0.0
        epoch_global_loss = 0.0
        epoch_local_loss = 0.0
        num_batches = 0
        pairs_trained_this_epoch = 0

        # ── Iterate over batches ──
        for batch_idx, (img1, img2, labels, indices) in enumerate(dataloader):
            img1   = img1.to(DEVICE)
            img2   = img2.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass — both images through shared ResNet
            emb1, emb2, global1, global2, local1, local2 = model(img1, img2)

            # Compute loss (global + λ × local)
            total_loss, global_loss, local_loss, distances = criterion(
                global1, global2, local1, local2, labels
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # ── Update priorities in manifest for trained pairs ──
            # Each pair's new priority = its loss value
            with torch.no_grad():
                for i, manifest_idx in enumerate(indices):
                    idx = manifest_idx.item()
                    if idx < len(manifest):
                        # Store the per-pair loss as new priority
                        pair_loss = distances[i].item()
                        manifest[idx]['priority'] = round(pair_loss, 4)
                        manifest[idx]['times_trained'] += 1

            # Track stats
            epoch_total_loss  += total_loss.item()
            epoch_global_loss += global_loss.item()
            epoch_local_loss  += local_loss.item()
            num_batches += 1
            pairs_trained_this_epoch += len(labels)

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                avg = epoch_total_loss / num_batches
                print(f"  Batch {batch_idx+1:>5} | Loss: {avg:.4f} | Pairs: {pairs_trained_this_epoch:,}")

        # ─── 3. Post-Epoch: Decay + Prune ────────────────
        if num_batches == 0:
            print("  ⚠️ No batches this epoch (manifest may be empty)")
            continue

        avg_total  = epoch_total_loss  / num_batches
        avg_global = epoch_global_loss / num_batches
        avg_local  = epoch_local_loss  / num_batches

        # Apply decay factor: priority = loss × (0.9 ^ times_trained)
        # This penalizes pairs trained many times (prevents overtraining)
        for pair in manifest:
            times = pair['times_trained']
            if times > 0:
                decay_penalty = DECAY_FACTOR ** times
                pair['priority'] = round(pair['priority'] * decay_penalty, 4)

        # Count pairs before pruning
        before_count = len(manifest)

        # Prune mastered pairs (priority < 0.05)
        # These pairs are too easy now — model has learned them
        manifest = [p for p in manifest if p['priority'] >= PRUNE_THRESHOLD]
        pruned_count = before_count - len(manifest)

        # Save updated manifest
        save_manifest(manifest, MANIFEST_PATH)

        # Save model checkpoint
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, avg_total, ckpt_path)

        # ── Epoch Summary ──
        elapsed = time.time() - epoch_start
        print(f"\n  📊 Epoch {epoch} Summary:")
        print(f"     Total Loss:  {avg_total:.4f}  (Global: {avg_global:.4f}, Local: {avg_local:.4f})")
        print(f"     Pairs trained: {pairs_trained_this_epoch:,}")
        print(f"     Pruned (mastered): {pruned_count:,} pairs")
        print(f"     Remaining: {len(manifest):,} pairs")
        print(f"     Checkpoint: {ckpt_path}")
        print(f"     Time: {elapsed:.1f}s")

        # Breakdown by type
        type_counts = {}
        for p in manifest:
            ptype = p['pair_type']
            type_counts[ptype] = type_counts.get(ptype, 0) + 1
        print(f"     By type: {type_counts}")

        # Check if training should stop (all pairs mastered)
        if len(manifest) == 0:
            print("\n🎉 All pairs mastered! Training complete.")
            break

    # ─── Final Model Save ────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    save_checkpoint(model, optimizer, NUM_EPOCHS, 0, final_path)
    print(f"\n✅ Training finished. Final model saved to {final_path}")


if __name__ == "__main__":
    train()
