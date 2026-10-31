"""
generate_manifest.py — Generate Training Pair Manifest
======================================================

Creates a JSON manifest of training pairs from the universe.
Implements the 4-pillar strategy:
  - Hard Positives (same product, random attacks in RAM)
  - Soft Positives (same product, duplicate files)
  - Hard Negatives (different product, same category)
  - Soft Negatives (different product, different category)

From Plan.md Section 3:
  - soft_pos:  235,500 pairs
  - hard_pos:  329,220 pairs
  - hard_neg:  100,000 pairs
  - soft_neg:  307,720 pairs
  ─────────────────────
  TOTAL:       972,440 pairs

For testing: creates smaller manifest_100.json (100 pairs)
"""

import json
import os
import random
from collections import defaultdict

PROJECT_ROOT = r"E:\BTP"
UNIVERSE_PATH = os.path.join(PROJECT_ROOT, "json", "training_universe.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "json", "test_manifest_100.json")


def load_universe(path):
    """Load universe JSON."""
    if not os.path.exists(path):
        print(f"❌ Universe not found: {path}")
        print(f"   Run: python create_universe.py")
        return None
    
    with open(path, 'r') as f:
        return json.load(f)


def create_test_manifest(universe, pair_count=100):
    """
    Create a small test manifest for quick validation.
    
    Uses 4-pillar strategy:
      25 hard_pos (same product, attacked in RAM)
      25 hard_neg (different product, same category)
      25 soft_pos (same product, duplicate file)
      25 soft_neg (different product, different category)
    """
    
    if not universe:
        return None
    
    manifest = []
    row_id = 0
    
    categories = list(universe.keys())
    
    print(f"\n📋 Creating test manifest ({pair_count} pairs)...")
    print("-" * 60)
    
    # Helper: get random product image
    def random_image(cat, prod=None):
        """Get random image from category/product."""
        if cat not in universe or not universe[cat]:
            return None
        
        if prod is None:
            prod = random.choice(list(universe[cat].keys()))
        
        if prod not in universe[cat] or not universe[cat][prod]:
            return None
        
        return random.choice(universe[cat][prod])
    
    # 25 hard positives (same product, random attacks in RAM)
    print(f"  ✓ Creating hard positives (25 pairs)...")
    for _ in range(25):
        cat = random.choice(categories)
        if not universe[cat]:
            continue
        
        prod = random.choice(list(universe[cat].keys()))
        img = random_image(cat, prod)
        
        if img:
            manifest.append({
                "row_id": row_id,
                "path_1": img,
                "path_2": img,  # Same image (attack in RAM)
                "label": 1,
                "pair_type": "hard_pos",
                "priority": 0.50,
                "times_trained": 0
            })
            row_id += 1
    
    # 25 hard negatives (different product, same category)
    print(f"  ✓ Creating hard negatives (25 pairs)...")
    for _ in range(25):
        cat = random.choice(categories)
        if not universe[cat] or len(universe[cat]) < 2:
            continue
        
        prod1, prod2 = random.sample(list(universe[cat].keys()), 2)
        img1 = random_image(cat, prod1)
        img2 = random_image(cat, prod2)
        
        if img1 and img2:
            manifest.append({
                "row_id": row_id,
                "path_1": img1,
                "path_2": img2,
                "label": 0,
                "pair_type": "hard_neg",
                "priority": 0.25,
                "times_trained": 0
            })
            row_id += 1
    
    # 25 soft positives (same product, duplicate file)
    print(f"  ✓ Creating soft positives (25 pairs)...")
    for _ in range(25):
        cat = random.choice(categories)
        if not universe[cat]:
            continue
        
        prod = random.choice(list(universe[cat].keys()))
        imgs = universe[cat][prod]
        
        if len(imgs) >= 2:
            img1, img2 = random.sample(imgs, 2)
            manifest.append({
                "row_id": row_id,
                "path_1": img1,
                "path_2": img2,
                "label": 1,
                "pair_type": "soft_pos",
                "priority": 0.0,
                "times_trained": 0
            })
            row_id += 1
    
    # 25 soft negatives (different product, different category)
    print(f"  ✓ Creating soft negatives (25 pairs)...")
    for _ in range(25):
        if len(categories) < 2:
            break
        
        cat1, cat2 = random.sample(categories, 2)
        
        img1 = random_image(cat1)
        img2 = random_image(cat2)
        
        if img1 and img2:
            manifest.append({
                "row_id": row_id,
                "path_1": img1,
                "path_2": img2,
                "label": 0,
                "pair_type": "soft_neg",
                "priority": 0.0,
                "times_trained": 0
            })
            row_id += 1
    
    return manifest


def save_manifest(manifest, output_path):
    """Save manifest to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✅ Manifest saved: {output_path}")
    print(f"   Pairs: {len(manifest)}")


def main():
    print("\n" + "="*60)
    print("📋 GENERATING TEST MANIFEST")
    print("="*60)
    
    universe = load_universe(UNIVERSE_PATH)
    
    if universe:
        manifest = create_test_manifest(universe, pair_count=100)
        
        if manifest:
            save_manifest(manifest, OUTPUT_PATH)
            print("\n✅ Done!\n")
        else:
            print("❌ Failed to create manifest\n")
    else:
        print("❌ Failed to load universe\n")


if __name__ == "__main__":
    main()
