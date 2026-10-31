from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Final_Artifacts" / "images" / "q100_final19_images"
DESTINATION = ROOT / "Final_Artifacts" / "images" / "latest_19attack_50_images_20261022"
DATASETS = ("ucid", "flickr", "cifar", "amazon")
EXPECTED_ATTACKS = (
    "original",
    "rotate_m5",
    "rotate_m10",
    "rotate5",
    "rotate15",
    "rotate30",
    "crop5",
    "crop10",
    "crop20",
    "bright_light",
    "flip_h",
    "resize_compress",
    "resize",
    "watermark_text",
    "watermark_asset",
    "mix_rotate10_bright",
    "mix_crop20_watermark",
    "blur",
    "contrast",
)


def validate_dataset(source_dataset: Path) -> dict[str, object]:
    manifest_path = source_dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folders = sorted(path for path in source_dataset.iterdir() if path.is_dir())
    if len(folders) != 50:
        raise RuntimeError(f"{source_dataset.name}: expected 50 folders, found {len(folders)}")
    if tuple(manifest["attacks"]) != EXPECTED_ATTACKS:
        raise RuntimeError(f"{source_dataset.name}: source attack order does not match the final 19-attack set")

    labels = Counter()
    for folder in folders:
        files = sorted(path.name for path in folder.glob("*.png"))
        expected_files = [f"{index:02d}_{attack}.png" for index, attack in enumerate(EXPECTED_ATTACKS, start=1)]
        if files != expected_files:
            raise RuntimeError(f"{folder}: expected exactly the 19 final attack images")
        labels[folder.name.split("_", 2)[1]] += 1

    return {
        "folders": len(folders),
        "images": len(folders) * len(EXPECTED_ATTACKS),
        "labels": dict(sorted(labels.items())),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
    }


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {DESTINATION}")

    summary: dict[str, object] = {
        "source": str(SOURCE.relative_to(ROOT)),
        "selection": "approved q100 25-known/25-unknown samples from the final verified runs",
        "attack_set": list(EXPECTED_ATTACKS),
        "datasets": {},
    }
    for dataset in DATASETS:
        source_dataset = SOURCE / dataset
        summary["datasets"][dataset] = validate_dataset(source_dataset)

    DESTINATION.mkdir(parents=True)
    for dataset in DATASETS:
        shutil.copytree(SOURCE / dataset, DESTINATION / dataset)

    (DESTINATION / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Created: {DESTINATION}")


if __name__ == "__main__":
    main()
