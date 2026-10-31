"""Validate that Codebase_Final contains one complete, non-legacy source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "configs/attacks_final19.json",
    "configs/datasets.yaml",
    "configs/final_graph_choices.yaml",
    "src/models/model_convnext_metric.py",
    "src/data/augmentation/attack_augmentor.py",
    "src/data/materialization/build_amazon50k_compact_light3.py",
    "src/data/loaders/training_pair_loader.py",
    "src/data/loaders/image_dataset_loader.py",
    "src/training/model_selection/train_convnext.py",
    "src/training/calibration/search_diagonal_metric.py",
    "src/evaluation/attack_tables/build_attack_tables.py",
    "src/query_selection/select_graph_query_sets.py",
    "src/phase3_graph/engine/hnsw_hierarchical.py",
    "src/phase3_graph/engine/graph_search_kernels.cpp",
    "src/phase3_graph/experiments/hnsw_full_q_sweep.py",
    "tools/operations/visual_validation/create_latest_19attack_image_export.py",
)


def inventory() -> list[dict[str, str]]:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest})
    return rows


def validate() -> list[str]:
    errors = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")
    for forbidden in (ROOT / "legacy", ROOT / "archive", ROOT / "old", ROOT / "src" / "reproduction"):
        if forbidden.exists():
            errors.append(f"forbidden historical directory remains: {forbidden.relative_to(ROOT)}")
    for path in (ROOT / "src").rglob("*.py"):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc.msg}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-inventory", action="store_true")
    args = parser.parse_args()
    errors = validate()
    if args.write_inventory:
        output = ROOT / "docs" / "source_inventory.json"
        output.write_text(json.dumps({"root": ROOT.name, "files": inventory()}, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    if errors:
        print("Repository validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Repository validation passed: {len(inventory())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
