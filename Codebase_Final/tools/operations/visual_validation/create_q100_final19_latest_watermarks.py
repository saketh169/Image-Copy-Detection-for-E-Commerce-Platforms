from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "Final_Artifacts" / "images" / "q100_final19_base_grouped_20260728.zip"
EXTRACT_DIR = ROOT / "Final_Artifacts" / "images" / "q100_final19_images_latest_watermarks_20260830"
WATERMARK_DIR = ROOT / "Final_Artifacts" / "watermark_samples" / "latest_selected_watermark_50x2_all_datasets_20260829"

ATTACK_TEXT = "14_watermark_text.png"
ATTACK_ASSET = "15_watermark_asset.png"


def only_child_dir(path: Path) -> Path:
    dirs = [p for p in path.iterdir() if p.is_dir()]
    if len(dirs) == 1 and dirs[0].name == "q100_final19_grouped_50x19_images_20260728":
        return dirs[0]
    return path


def sorted_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"))


def main() -> None:
    if not ZIP.exists():
        raise FileNotFoundError(ZIP)
    if not WATERMARK_DIR.exists():
        raise FileNotFoundError(WATERMARK_DIR)
    if EXTRACT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing folder: {EXTRACT_DIR}")

    EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(EXTRACT_DIR)

    qroot = only_child_dir(EXTRACT_DIR)
    replacement_rows = []

    for dataset in ["cifar", "ucid", "flickr", "amazon"]:
        dataset_dir = qroot / dataset
        folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        text_imgs = sorted_pngs(WATERMARK_DIR / dataset / "text")
        asset_imgs = sorted_pngs(WATERMARK_DIR / dataset / "asset")

        if len(folders) != 50:
            raise RuntimeError(f"{dataset}: expected 50 q100 folders, found {len(folders)}")
        if len(text_imgs) != 50:
            raise RuntimeError(f"{dataset}: expected 50 text watermark images, found {len(text_imgs)}")
        if len(asset_imgs) != 50:
            raise RuntimeError(f"{dataset}: expected 50 asset watermark images, found {len(asset_imgs)}")

        for folder, text_img, asset_img in zip(folders, text_imgs, asset_imgs):
            text_dst = folder / ATTACK_TEXT
            asset_dst = folder / ATTACK_ASSET
            if not text_dst.exists() or not asset_dst.exists():
                raise FileNotFoundError(f"Missing watermark attack slot in {folder}")
            shutil.copy2(text_img, text_dst)
            shutil.copy2(asset_img, asset_dst)
            replacement_rows.append({
                "dataset": dataset,
                "folder": str(folder),
                "watermark_text_source": str(text_img),
                "watermark_asset_source": str(asset_img),
                "watermark_text_dest": str(text_dst),
                "watermark_asset_dest": str(asset_dst),
            })

    counts = {}
    for dataset in ["cifar", "ucid", "flickr", "amazon"]:
        dataset_dir = qroot / dataset
        folders = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
        images = list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.jpeg"))
        counts[dataset] = {"folders": len(folders), "images": len(images)}

    summary = {
        "output_dir": str(qroot),
        "source_zip": str(ZIP),
        "watermark_source_dir": str(WATERMARK_DIR),
        "attack_count_per_q_item": 19,
        "datasets": counts,
        "replaced": {
            "watermark_text": ATTACK_TEXT,
            "watermark_asset": ATTACK_ASSET,
            "rows": len(replacement_rows),
        },
        "replacement_rows": replacement_rows,
    }
    (qroot / "latest_watermark_replacement_summary_20260830.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.make_archive(str(qroot), "zip", qroot)
    print(json.dumps(summary["datasets"], indent=2))
    print(str(qroot))
    print(str(qroot) + ".zip")


if __name__ == "__main__":
    main()
