import argparse
import json
import random
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect_category_rows(
    data_root: Path,
    categories: list[str],
    known_flag: bool,
    max_images_per_product: int = 0,
):
    rows = []
    for category in categories:
        cat_dir = data_root / category
        if not cat_dir.exists():
            continue
        for product_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            images = [p for p in sorted(product_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            if not images:
                continue
            if max_images_per_product and max_images_per_product > 0:
                chosen_images = images[:max_images_per_product]
            else:
                chosen_images = images
            for image_index, image_path in enumerate(chosen_images):
                rows.append(
                    {
                        "path": str(image_path),
                        "known": known_flag,
                        "category": category,
                        "product": f"{category}/{product_dir.name}",
                        "image_index": image_index,
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/saketh/Data Processed")
    parser.add_argument("--known-json", required=True)
    parser.add_argument("--unknown-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--known-count", type=int, required=True)
    parser.add_argument("--unknown-count", type=int, required=True)
    parser.add_argument(
        "--max-images-per-product",
        type=int,
        default=0,
        help="Use up to this many images per product folder. 0 means all images.",
    )
    parser.add_argument("--seed", type=int, default=20260416)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    known_categories = json.loads(Path(args.known_json).read_text(encoding="utf-8"))
    unknown_categories = json.loads(Path(args.unknown_json).read_text(encoding="utf-8"))

    known_rows = collect_category_rows(
        data_root,
        known_categories,
        True,
        max_images_per_product=args.max_images_per_product,
    )
    unknown_rows = collect_category_rows(
        data_root,
        unknown_categories,
        False,
        max_images_per_product=args.max_images_per_product,
    )

    rng = random.Random(args.seed)
    rng.shuffle(known_rows)
    rng.shuffle(unknown_rows)

    target_known = len(known_rows) if args.known_count <= 0 else args.known_count
    target_unknown = len(unknown_rows) if args.unknown_count <= 0 else args.unknown_count

    if len(known_rows) < target_known:
        raise SystemExit(f"Need {target_known} known rows, found {len(known_rows)}")
    if len(unknown_rows) < target_unknown:
        raise SystemExit(f"Need {target_unknown} unknown rows, found {len(unknown_rows)}")

    pool = known_rows[:target_known] + unknown_rows[:target_unknown]
    rng.shuffle(pool)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"known {target_known}")
    print(f"unknown {target_unknown}")
    print(f"max_images_per_product {args.max_images_per_product if args.max_images_per_product > 0 else 'all'}")


if __name__ == "__main__":
    main()
