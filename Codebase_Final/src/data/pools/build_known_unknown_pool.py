import argparse
import json
import random
from pathlib import Path


def collect_images(data_root, categories):
    images = []
    for top in categories:
        top_dir = data_root / top
        if not top_dir.exists():
            continue
        for subcat in top_dir.iterdir():
            if not subcat.is_dir():
                continue
            for product_dir in subcat.iterdir():
                if not product_dir.is_dir():
                    continue
                for img_path in product_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        images.append(
                            {
                                "path": str(img_path),
                                "category": top,
                                "subcategory": f"{top}/{subcat.name}",
                                "product": f"{top}/{subcat.name}/{product_dir.name}",
                            }
                        )
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/saketh/Data Processed")
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--known-categories", nargs="+", required=True)
    parser.add_argument("--pool-size", type=int, default=2000)
    parser.add_argument("--known-size", type=int, default=None)
    parser.add_argument("--unknown-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260413)
    parser.add_argument("--out", default="/home/saketh/json/db_pool.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    random.seed(args.seed)

    all_images = collect_images(data_root, args.categories)
    if not all_images:
        raise SystemExit("No images found for requested categories.")

    known_set = set(args.known_categories)
    known_images = [r for r in all_images if r["category"] in known_set]
    unknown_images = [r for r in all_images if r["category"] not in known_set]

    random.shuffle(known_images)
    random.shuffle(unknown_images)

    if args.known_size is not None and args.unknown_size is not None:
        pool = known_images[: args.known_size] + unknown_images[: args.unknown_size]
    else:
        # Balance across categories if possible
        by_cat = {}
        for row in all_images:
            by_cat.setdefault(row["category"], []).append(row)

        per_cat = max(1, args.pool_size // max(1, len(by_cat)))
        pool = []
        for cat, rows in by_cat.items():
            random.shuffle(rows)
            pool.extend(rows[:per_cat])

        if len(pool) < args.pool_size:
            remaining = [r for r in all_images if r not in pool]
            random.shuffle(remaining)
            pool.extend(remaining[: args.pool_size - len(pool)])

        pool = pool[: args.pool_size]

    for row in pool:
        row["known"] = row["category"] in known_set

    Path(args.out).write_text(json.dumps(pool, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
