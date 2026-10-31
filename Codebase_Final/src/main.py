"""Command entry point for the cleaned Image Copy Detection codebase.

The heavy experiment scripts remain in their stage folders. This file gives a
stable place to discover the final pipeline order without hiding the actual
research scripts behind another abstraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PIPELINE = {
    "manifest": "data/manifests/generate_dataset_manifest.py",
    "embedding-store": "data/stores/build_embedding_store.py",
    "train-convnext": "training/model_selection/train_convnext.py",
    "train-resnet": "training/model_selection/train_resnet.py",
    "train-vit": "training/model_selection/train_vit.py",
    "metric-refinement": "training/metric_learning/train_hard_pair_metric.py",
    "calibrate": "training/calibration/search_diagonal_metric.py",
    "evaluate": "evaluation/attack_tables/build_attack_tables.py",
    "hnsw-sweep": "phase3_graph/experiments/hnsw_full_q_sweep.py",
    "hnsw-core": "phase3_graph/engine/hnsw_hierarchical.py",
    "erng-core": "phase3_graph/engine/erng_flat.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Copy Detection final pipeline helper")
    parser.add_argument("stage", nargs="?", choices=sorted(PIPELINE), help="Pipeline stage to locate")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if not args.stage:
        print("Available final stages:")
        for name, rel in PIPELINE.items():
            print(f"  {name:15s} -> {rel}")
        return

    script = root / PIPELINE[args.stage]
    print(script)


if __name__ == "__main__":
    main()

