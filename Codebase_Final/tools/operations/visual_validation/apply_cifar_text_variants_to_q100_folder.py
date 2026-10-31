from __future__ import annotations

import json
from pathlib import Path

import cv2

from cifar_text_boundary_proper_20260828 import apply_cifar_text


ROOT = Path(__file__).resolve().parents[1]
Q100_ROOT = (
    ROOT
    / "Final_Artifacts"
    / "images"
    / "q100_all_datasets_final19_20260829"
    / "q100_final19_grouped_50x19_images_20260728"
)
RESULT_JSON = (
    ROOT
    / "Final_Artifacts"
    / "json"
    / "watermark_variants"
    / "cifar"
    / "cifar_text_boundary_proper_results.json"
)


def read_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])


def main() -> None:
    result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    variants = result["valid_variants"][:50]
    cifar_dir = Q100_ROOT / "cifar"
    folders = sorted([p for p in cifar_dir.iterdir() if p.is_dir()])
    if len(folders) != 50:
        raise RuntimeError(f"Expected 50 CIFAR q100 folders, found {len(folders)}")
    if len(variants) < 50:
        raise RuntimeError(f"Expected 50 CIFAR text variants, found {len(variants)}")

    rows = []
    for folder, var in zip(folders, variants):
        src = folder / "01_original.png"
        dst = folder / "14_watermark_text.png"
        img = read_rgb(src)
        attacked = apply_cifar_text(img, var)
        write_rgb(dst, attacked)
        rows.append({
            "folder": str(folder),
            "source": str(src),
            "updated": str(dst),
            "variant_id": var["variant_id"],
            "count": var["count"],
            "scale": var["scale"],
            "alpha": var["alpha"],
            "known": var["known"],
            "unknown": var["unknown"],
            "total": var["total"],
        })

    summary = {
        "q100_root": str(Q100_ROOT),
        "dataset": "cifar",
        "updated_attack": "14_watermark_text.png",
        "source_variants": str(RESULT_JSON),
        "updated_count": len(rows),
        "rows": rows,
    }
    out = Q100_ROOT / "cifar_watermark_text_update_summary_20260829.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"updated": len(rows), "summary": str(out)}, indent=2))


if __name__ == "__main__":
    main()
