import json
from pathlib import Path


OUT = Path("/home/saketh/json/target_final_stores_20260528")
OUT.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "cifar": {
        "store": Path("/home/saketh/json/clean_best_json_20260528/cifar/embeddings_store_cifar_pathfixed_reembedded_currentcode.json"),
        "target_known": 17000,
        "target_unknown": 9000,
        "out": OUT / "cifar/cifar_target_17000k_9000u_store.json",
    },
    "ucid": {
        "store": Path("/home/saketh/json/clean_best_json_20260528/ucid/embeddings_store_convnext_ucid_unified.json"),
        "target_known": 3500,
        "target_unknown": 500,
        "out": OUT / "ucid/ucid_target_3500k_500u_store.json",
    },
}


def quality_known(row):
    passes = float(row.get("attack_pass_count", row.get("panel_passes", 0)) or 0)
    total = float(row.get("panel_total", 1) or 1)
    dist = float(row.get("distance", row.get("attack_distance_sum", 1e9)) or 1e9)
    return (passes / max(1.0, total), -dist)


def quality_unknown(row):
    passes = float(row.get("attack_pass_count", row.get("panel_passes", 0)) or 0)
    total = float(row.get("panel_total", 1) or 1)
    dist = float(row.get("distance", row.get("attack_distance_sum", 0)) or 0)
    return (passes / max(1.0, total), dist)


def trim_dataset(name, cfg):
    store = json.load(open(cfg["store"]))
    known = list(store.get("accepted_known", []))
    unknown = list(store.get("accepted_unknown", []))
    known_sorted = sorted(known, key=quality_known, reverse=True)
    unknown_sorted = sorted(unknown, key=quality_unknown, reverse=True)
    keep_known = known_sorted[: cfg["target_known"]]
    keep_unknown = unknown_sorted[: cfg["target_unknown"]]
    out = {
        "accepted_known": keep_known,
        "accepted_unknown": keep_unknown,
        "processed_paths": [r["path"] for r in keep_known + keep_unknown],
        "candidate_known": [],
        "candidate_unknown": [],
        "last_epoch": store.get("last_epoch"),
        "meta": {
            "mode": "target_trim_store_20260528",
            "source_store": str(cfg["store"]),
            "target_known": cfg["target_known"],
            "target_unknown": cfg["target_unknown"],
            "input_known": len(known),
            "input_unknown": len(unknown),
            "removed_known": len(known) - len(keep_known),
            "removed_unknown": len(unknown) - len(keep_unknown),
            "selection": "keep strongest known by panel/pass ratio then lower distance; keep strongest unknown by panel/pass ratio then farther distance",
        },
    }
    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    cfg["out"].write_text(json.dumps(out, indent=2))
    summary = {
        "dataset": name,
        "known": len(keep_known),
        "unknown": len(keep_unknown),
        "total": len(keep_known) + len(keep_unknown),
        "removed_known": len(known) - len(keep_known),
        "removed_unknown": len(unknown) - len(keep_unknown),
        "store": str(cfg["out"]),
    }
    summary_path = cfg["out"].with_name(cfg["out"].stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    for name, cfg in CONFIGS.items():
        trim_dataset(name, cfg)


if __name__ == "__main__":
    main()
