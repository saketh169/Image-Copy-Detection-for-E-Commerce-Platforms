import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from remote_generate_scaled_manifests import (
    DEFAULT_TOP_LEVEL_CATEGORIES,
    balanced_disjoint_split,
    enforce_category_coverage,
    load_products,
    make_manifest,
)


def collect_image_rows(products, selected_categories):
    rows = []
    allowed_categories = set(selected_categories["known"]) | set(selected_categories["unknown"])
    for product in products:
        if product["category"] not in allowed_categories:
            continue
        for image_index, image_path in enumerate(product["images"]):
            row = {
                "path": image_path,
                "known": product["category"] in selected_categories["known"],
                "category": product["category"],
                "product": product["product"],
                "image_index": image_index,
            }
            rows.append(row)
    return rows


def choose_rows(rows, known_total, unknown_total, seed):
    rng = random.Random(seed)
    known_rows = [row for row in rows if row["known"]]
    unknown_rows = [row for row in rows if not row["known"]]
    rng.shuffle(known_rows)
    rng.shuffle(unknown_rows)

    if len(known_rows) < known_total:
        raise SystemExit(f"Need {known_total} known image rows, found {len(known_rows)}")
    if len(unknown_rows) < unknown_total:
        raise SystemExit(f"Need {unknown_total} unknown image rows, found {len(unknown_rows)}")

    chosen = known_rows[:known_total] + unknown_rows[:unknown_total]
    rng.shuffle(chosen)
    return chosen


def rebuild_products_from_rows(products, selected_rows):
    selected_paths = {row["path"] for row in selected_rows}
    product_images = defaultdict(list)
    category_products = defaultdict(list)

    for product in products:
        kept = [img for img in product["images"] if img in selected_paths]
        if not kept:
            continue
        rebuilt = {
            "category": product["category"],
            "product": product["product"],
            "images": kept,
        }
        product_images[product["product"]] = kept
        category_products[product["category"]].append(rebuilt)

    rebuilt_products = [row for rows in category_products.values() for row in rows]
    return rebuilt_products, dict(category_products), dict(product_images)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/saketh/Data Processed")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--models", nargs="+", default=["convnext", "resnet", "vit"])
    parser.add_argument("--pair-count", type=int, default=15000)
    parser.add_argument("--pool-size", type=int, default=15000)
    parser.add_argument("--known-image-count", type=int, default=10000)
    parser.add_argument("--unknown-image-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260418)
    parser.add_argument("--categories", nargs="+", default=sorted(DEFAULT_TOP_LEVEL_CATEGORIES))
    parser.add_argument("--known-json")
    parser.add_argument("--unknown-json")
    parser.add_argument("--balanced-six-types", type=int, default=0)
    parser.add_argument("--known-per-top-level", type=int, default=0)
    parser.add_argument("--unknown-per-top-level", type=int, default=0)
    parser.add_argument("--enforce-category-coverage", type=int, default=1)
    parser.add_argument("--min-pairs-per-category", type=int, default=1)
    args = parser.parse_args()

    if args.known_image_count + args.unknown_image_count != args.pool_size:
        raise SystemExit("known-image-count + unknown-image-count must equal pool-size")

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    products, categories = load_products(data_root, set(args.categories))

    if args.known_json and args.unknown_json:
        known_set = set(json.loads(Path(args.known_json).read_text(encoding="utf-8")))
        unknown_set = set(json.loads(Path(args.unknown_json).read_text(encoding="utf-8")))
    elif args.balanced_six_types:
        if not args.known_per_top_level or not args.unknown_per_top_level:
            raise SystemExit("balanced-six-types requires known-per-top-level and unknown-per-top-level")
        known_set, unknown_set = balanced_disjoint_split(
            categories,
            known_per_top=args.known_per_top_level,
            unknown_per_top=args.unknown_per_top_level,
            seed=args.seed,
        )
    else:
        raise SystemExit("Provide known/unknown category JSON files or use --balanced-six-types")

    selected_categories = {
        "known": known_set,
        "unknown": unknown_set,
    }
    image_rows = collect_image_rows(products, selected_categories)
    chosen_rows = choose_rows(
        image_rows,
        known_total=args.known_image_count,
        unknown_total=args.unknown_image_count,
        seed=args.seed,
    )

    rebuilt_products, rebuilt_categories, product_images = rebuild_products_from_rows(products, chosen_rows)
    known_products = [row for row in rebuilt_products if row["category"] in known_set]
    unknown_products = [row for row in rebuilt_products if row["category"] in unknown_set]
    known_categories = {key: value for key, value in rebuilt_categories.items() if key in known_set}

    manifest = make_manifest(
        args.pair_count,
        rebuilt_products,
        rebuilt_categories,
        args.seed + args.pair_count,
        product_images,
        known_products,
        known_categories,
        unknown_products,
    )
    if args.enforce_category_coverage:
        enforce_category_coverage(
            manifest=manifest,
            known_set=known_set,
            unknown_set=unknown_set,
            categories=rebuilt_categories,
            train_map=product_images,
            seed=args.seed + args.pair_count,
            min_pairs_per_category=max(1, args.min_pairs_per_category),
        )

    missing = [
        row for row in manifest if not Path(row["path_1"]).exists() or not Path(row["path_2"]).exists()
    ]
    if missing:
        raise SystemExit(f"Missing image paths after pool build: {len(missing)}")

    shared_dir = out_root / "shared"
    write_json(shared_dir / "image_pool_15k.json", chosen_rows)
    write_json(shared_dir / "known_categories.json", sorted(known_set))
    write_json(shared_dir / "unknown_categories.json", sorted(unknown_set))
    write_json(
        shared_dir / "metadata.json",
        {
            "pool_size": len(chosen_rows),
            "known_rows": sum(1 for row in chosen_rows if row["known"]),
            "unknown_rows": sum(1 for row in chosen_rows if not row["known"]),
            "pair_count": len(manifest),
            "pair_type_counts": dict(Counter(row["pair_type"] for row in manifest)),
            "models": args.models,
            "independence_note": (
                "Each model should use its own copy of this image pool, own active pool JSON, "
                "own embeddings store JSON, and own checkpoint directory."
            ),
        },
    )

    for model in args.models:
        model_dir = out_root / f"{model}_15k_unified"
        write_json(model_dir / f"source_pool_{model}_15k.json", chosen_rows)
        write_json(model_dir / f"training_{model}_pairs_15k.json", manifest)
        write_json(model_dir / "known_categories.json", sorted(known_set))
        write_json(model_dir / "unknown_categories.json", sorted(unknown_set))
        write_json(
            model_dir / "run_layout.json",
            {
                "model": model,
                "source_pool_json": str(model_dir / f"source_pool_{model}_15k.json"),
                "manifest_json": str(model_dir / f"training_{model}_pairs_15k.json"),
                "active_pool_json_suggestion": str(model_dir / f"active_eval_pool_{model}_15k.json"),
                "embedding_store_json_suggestion": str(model_dir / f"embeddings_store_{model}_15k.json"),
                "checkpoint_tag_suggestion": f"{model}_15k_unified",
                "shared_image_universe_matches_models": args.models,
            },
        )

    print(f"shared pool rows: {len(chosen_rows)}")
    print(f"known rows: {sum(1 for row in chosen_rows if row['known'])}")
    print(f"unknown rows: {sum(1 for row in chosen_rows if not row['known'])}")
    print(f"manifest pairs: {len(manifest)}")
    print(f"pair types: {dict(Counter(row['pair_type'] for row in manifest))}")
    print(f"wrote out root: {out_root}")


if __name__ == "__main__":
    main()
