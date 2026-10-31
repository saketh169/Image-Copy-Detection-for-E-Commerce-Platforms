"""Discover canonical entry points in the final Image Copy Detection project."""

from __future__ import annotations

import argparse
from pathlib import Path


PIPELINE = {
    "manifest": "src/data/manifests/generate_dataset_manifest.py",
    "train-convnext": "src/training/model_selection/train_convnext.py",
    "metric-refinement": "src/training/metric_learning/train_hard_pair_metric.py",
    "calibrate": "src/training/calibration/search_diagonal_metric.py",
    "evaluate": "src/evaluation/attack_tables/build_attack_tables.py",
    "embedding-store": "src/data/stores/build_embedding_store.py",
    "hnsw-sweep": "src/phase3_graph/experiments/hnsw_full_q_sweep.py",
    "image-export": "tools/operations/visual_validation/create_latest_19attack_image_export.py",
    "validate": "tools/validate_repository.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Image Copy Detection final-project helper")
    parser.add_argument("stage", nargs="?", choices=sorted(PIPELINE))
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.stage:
        print(root / PIPELINE[args.stage])
        return
    for name, relative_path in PIPELINE.items():
        print(f"{name:16s} {relative_path}")


if __name__ == "__main__":
    main()
