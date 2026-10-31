import json
import statistics
from collections import Counter
from pathlib import Path

base = Path("/home/saketh/json/target_final_stores_20260528")
out = base / "final_tables_report_completed_now_20260530.md"


def load(path):
    return json.load(open(path))


def display_name(name):
    return {
        "Flickr10k": "Flickr",
        "Amazon50k": "Amazon",
    }.get(name, name)


completed = {
    "CIFAR": base / "fixed_threshold_multiq/cifar/cifar_fixed_threshold_multiq.json",
    "Flickr10k": base / "fixed_threshold_multiq/flickr10k/flickr10k_fixed_threshold_multiq.json",
    "UCID": base / "fixed_threshold_multiq/ucid_relaxed_refill/ucid_relaxed_refill_fixed_threshold_multiq.json",
    "Amazon50k": base / "fixed_threshold_multiq/amazon50k/amazon50k_fixed_threshold_multiq.json",
}
threshold_search = {
    "UCID relaxed refill": base / "threshold_search_20260530/ucid_relaxed_refill/ucid_relaxed_refill_multiq_threshold_search.json",
    "Amazon50k": base / "threshold_search_20260530/amazon50k/amazon50k_multiq_threshold_search.json",
}
stores = {
    "CIFAR": base / "cifar/cifar_target_17000k_9000u_store.json",
    "Flickr10k": base / "refill_stream/flickr10k/flickr10k_stream_threshold_store.json",
    "UCID": base / "ucid_relaxed_refill_20260530/ucid/ucid_stream_threshold_store.json",
    "Amazon50k": base / "amazon50k_full50_replace/amazon50k/amazon50k_stream_threshold_store.json",
}
ckpts = {
    "CIFAR": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/cifar/cifar-epoch-015-training.pt",
    "Flickr10k": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/flickr10k/flickr10k-epoch-024-training.pt",
    "UCID": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/ucid/ucid-epoch-036-training.pt",
    "Amazon50k": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/amazon50k/amazon50k-epoch-033-training.pt",
}
clean_folders = {
    "CIFAR": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/cifar",
    "Flickr10k": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/flickr10k",
    "UCID": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/ucid",
    "Amazon50k": "/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/amazon50k",
}
attack_groups = {
    "Original / mandatory gate": ["original"],
    "Soft": [
        "rotate_m5",
        "rotate5",
        "crop5",
        "bright_light",
        "bright",
        "resize",
        "resize_compress",
        "blur",
        "contrast",
    ],
    "Medium": [
        "rotate_m10",
        "rotate15",
        "rotate30",
        "crop10",
        "crop20",
        "flip_h",
        "flip_v",
        "flip",
        "watermark_text",
        "watermark_asset",
    ],
    "Hard": [
        "rotate40",
        "crop40",
        "heavy_bright",
        "mix_rotate10_bright",
        "mix_crop20_watermark",
        "bg_color_change",
    ],
}
attack_order = (
    attack_groups["Original / mandatory gate"]
    + attack_groups["Soft"]
    + attack_groups["Medium"]
    + attack_groups["Hard"]
)


def attack_average(attacks, names, metric="total"):
    vals = [attacks[a][metric] for a in names if a in attacks]
    return sum(vals) / len(vals) if vals else 0.0


def group_stats_from_attacks(attacks):
    return {
        group: {
            "known": attack_average(attacks, names, "known"),
            "unknown": attack_average(attacks, names, "unknown"),
            "total": attack_average(attacks, names, "total"),
            "min_total": min((attacks[a]["total"] for a in names if a in attacks), default=0.0),
            "best_attack": max(
                ((a, attacks[a]["total"]) for a in names if a in attacks),
                key=lambda item: item[1],
                default=("", 0.0),
            ),
            "weakest_attack": min(
                ((a, attacks[a]["total"]) for a in names if a in attacks),
                key=lambda item: item[1],
                default=("", 0.0),
            ),
        }
        for group, names in attack_groups.items()
    }


def best_policy_rows(path):
    if not Path(path).exists():
        return []
    data = load(path)
    rows = []
    for policy, pdata in data.get("policies", {}).items():
        for q, qdata in pdata.get("query_results", {}).items():
            s = qdata["summary"]
            rows.append(
                {
                    "policy": policy,
                    "q": int(q),
                    "all_avg": s["all_avg"],
                    "nonorig_avg": s["nonorig_avg"],
                    "known_avg": s["known_nonorig_avg"],
                    "unknown_avg": s["unknown_nonorig_avg"],
                    "min_nonorig": s["min_nonorig"],
                    "below50": s["known_below50"] + s["unknown_below50"],
                    "original": s["original"],
                }
            )
    return sorted(rows, key=lambda r: (r["all_avg"], r["nonorig_avg"], r["q"]), reverse=True)


def threshold_label(values):
    values = [v for v in values if v is not None]
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if abs(lo - hi) < 1e-9:
        return f"{lo:.6f}"
    return f"{lo:.6f}-{hi:.6f}"


def compact_fixed_rows(path):
    data = load(path)
    queries = list(data["queries"].keys())
    rows = []
    for attack in attack_order:
        vals = []
        thresholds = []
        for q in queries:
            item = data["queries"][q]["attacks"][attack]
            vals.append(item["total"])
            thresholds.append(item.get("threshold"))
        rows.append((attack, threshold_label(thresholds), vals))
    return queries, rows


def compact_threshold_rows(path, policy):
    data = load(path)
    qdata = data["policies"][policy]["query_results"]
    queries = list(qdata.keys())
    rows = []
    for attack in attack_order:
        vals = []
        thresholds = []
        for q in queries:
            item = qdata[q]["attacks"][attack]
            vals.append(item["total"])
            thresholds.append(item.get("threshold"))
        rows.append((attack, threshold_label(thresholds), vals))
    return queries, rows


def summary_rows_for_dataset(name):
    if name in ("CIFAR", "Flickr10k"):
        d = load(completed[name])
        rows = []
        for q, qdata in d["queries"].items():
            s = qdata["summary"]
            rows.append((name, q, s))
        return rows

    if name == "UCID":
        d = load(threshold_search["UCID relaxed refill"])
        qdata = d["policies"]["avg_total"]["query_results"]
    elif name == "Amazon50k":
        d = load(threshold_search["Amazon50k"])
        qdata = d["policies"]["final_total"]["query_results"]
    else:
        return []

    rows = []
    for q, item in qdata.items():
        rows.append((name, q, item["summary"]))
    return rows


def summary_value(summary, key):
    if key in summary:
        return summary[key]
    aliases = {
        "known_nonorig_avg": "known_avg",
        "unknown_nonorig_avg": "unknown_avg",
    }
    return summary[aliases[key]]


def proportional_pair_counts(known, unknown):
    n = known + unknown
    hard_pos = n
    soft_pos = round(0.5 * n)
    hard_neg = round(0.625 * n)
    soft_neg = round(0.375 * n)
    total = hard_pos + soft_pos + hard_neg + soft_neg
    return n, hard_pos, soft_pos, hard_neg, soft_neg, total


def selected_query_attacks(name):
    if name in ("CIFAR", "Flickr10k"):
        return load(completed[name])["queries"], "fixed calibrated thresholds"
    if name == "UCID":
        qdata = load(threshold_search["UCID relaxed refill"])["policies"]["avg_total"]["query_results"]
        return qdata, "latest threshold-search thresholds"
    if name == "Amazon50k":
        qdata = load(threshold_search["Amazon50k"])["policies"]["final_total"]["query_results"]
        return qdata, "latest threshold-search thresholds"
    return {}, ""


def pct(vals, q):
    vals = sorted(vals)
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


lines = []
lines.append("# Final Store / Eval Report - completed outputs as of 2026-05-30")
lines.append("")
lines.append("This report includes completed fixed-threshold multi-query outputs for CIFAR, Flickr, UCID, and Amazon. For each dataset, the same calibrated threshold is used for a given attack across all query sizes.")
lines.append("")
lines.append("## Clean Checkpoint Paths")
lines.append("| Dataset | Clean folder | Final checkpoint | Numbering |")
lines.append("|---|---|---|---|")
lines.append(f"| CIFAR | `{clean_folders['CIFAR']}` | `{ckpts['CIFAR']}` | 1-15 training, final epoch 15 |")
lines.append(f"| Flickr | `{clean_folders['Flickr10k']}` | `{ckpts['Flickr10k']}` | 1-24 training, final epoch 24 |")
lines.append(f"| UCID | `{clean_folders['UCID']}` | `{ckpts['UCID']}` | 1-36 continuous training epochs; final epoch 36 |")
lines.append(f"| Amazon | `{clean_folders['Amazon50k']}` | `{ckpts['Amazon50k']}` | 1-33 continuous training epochs; final epoch 33 |")
lines.append("")
lines.append("Checkpoint manifest: `/home/saketh/Codebase/checkpoints/clean_best_checkpoints_20260528/MANIFEST.tsv`")
lines.append("")
lines.append("## System Architecture")
lines.append("The project is an image-copy detection pipeline built around a Siamese ConvNeXt-B embedding model. Each image is mapped to a compact embedding, and duplicate detection is done by nearest-neighbour distance against the known-image bank.")
lines.append("")
lines.append("| Stage | What happens | Output |")
lines.append("|---|---|---|")
lines.append("| Dataset split | Images are split into known/original-bank images and unknown/non-duplicate images. | Known and unknown source pools |")
lines.append("| Pair manifest | Training pairs are created as hard positives, soft positives, hard negatives, and soft negatives. | Manifest JSON/CSV used by training |")
lines.append("| Attack generation | Soft attacks are cached or sampled; stricter attacks are generated during training/eval to force attack invariance. | Original and attacked image views |")
lines.append("| Siamese training | The two ConvNeXt branches share weights. Positive pairs are pulled together; negative pairs are pushed apart. | Checkpoint per epoch |")
lines.append("| Store harvest/refill | Accepted images are embedded and written to the store as known or unknown rows. | Store JSON with embeddings and paths |")
lines.append("| FAISS retrieval | Query embeddings are compared to the known-bank embeddings using nearest-neighbour search. | Minimum distance to known bank |")
lines.append("| Threshold decision | A query is duplicate if its nearest known-bank distance is within the selected threshold; otherwise it is treated as unknown/non-duplicate. | Duplicate / non-duplicate decision |")
lines.append("")
lines.append("For known images, the intended geometry is: original known image close to the known bank, and all attack variants close to that original neighbourhood. For unknown images, the intended geometry is: original unknown image far from the known bank, while its own attack variants remain close to the unknown original.")
lines.append("")
lines.append("## Dataset Description")
lines.append("The four datasets are not visually identical, so their attack robustness should not be interpreted as a pure model-only comparison. Each dataset has a different image style, object diversity, background behaviour, and source quality.")
lines.append("")
lines.append("| Dataset | Image type | What it mainly contains | Why it matters for copy detection | Final store use |")
lines.append("|---|---|---|---|---|")
lines.append("| CIFAR | Small natural-object/classification images | Low-resolution object categories such as vehicles, animals, ships, trucks, birds, and similar compact objects. | Images are small and class-like, so many transformations preserve the global object shape; embeddings can stay stable for soft/medium attacks, but fine details are limited. | Used as a compact robustness benchmark with `17,000 known / 9,000 unknown` embeddings. |")
lines.append("| Flickr | Real-world photographic images | Outdoor/indoor scenes, people, animals, vehicles, buildings, objects, and mixed natural backgrounds. | High visual diversity makes it useful for testing copy detection beyond product-only images; crops/background changes are harder because scenes contain many distractors. | Used as a broad real-image benchmark with `10,000 known / 5,000 unknown` embeddings. |")
lines.append("| UCID | General image-copy / content-based retrieval style images | Smaller controlled image set with varied natural images and generated attack variants. | Useful for measuring controlled copy-detection geometry, but the available pool is much smaller than Amazon/Flickr, so store expansion is naturally limited. | Used as a focused copy-detection benchmark with the latest relaxed-refill store: `4,500 known / 850 unknown` embeddings. |")
lines.append("| Amazon | E-commerce/product images | Product photos across categories such as electronics, home, sports, cosmetics, baby products, and wearables. | This is the most application-aligned dataset: product images have repeated layouts, white backgrounds, category imbalance, and seller-style variations; known-side hard attacks are more difficult. | Used as the main product-copy dataset with `35,000 known / 15,000 unknown` embeddings. |")
lines.append("")
lines.append("Short reading: CIFAR is compact and class-like, Flickr is broad real-world photography, UCID is a smaller controlled copy-detection set, and Amazon is the product-image target domain.")
lines.append("")
lines.append("## Attack Difficulty Groups")
lines.append("`original` is treated as a mandatory reference gate, not as an attack. The remaining 25 attacks are grouped by practical difficulty for analysis.")
lines.append("")
lines.append("| Group | Attacks | Main reason |")
lines.append("|---|---|---|")
for group, names in attack_groups.items():
    if group == "Original / mandatory gate":
        reason = "Reference image; should remain 100% when the store and checkpoint are aligned."
    elif group == "Soft":
        reason = "Small photometric, resize, blur, or mild geometric changes."
    elif group == "Medium":
        reason = "Moderate geometry, crop, flip, or watermark changes that can shift neighbourhoods."
    else:
        reason = "Severe crop/rotation/background/mixed attacks that most often create the tail failures."
    lines.append(f"| {group} | {', '.join(names)} | {reason} |")
lines.append("")
lines.append("## Architecture Notes Needed To Read The Tables")
lines.append("- `Known` accuracy means attacked versions of known-bank images still retrieve inside the duplicate neighbourhood.")
lines.append("- `Unknown` accuracy means unknown images and their attacks stay outside the known-bank duplicate threshold.")
lines.append("- The reported threshold is attack-specific for analysis. At deployment, the threshold summary gives strict/balanced/loose operating points when the attack type is unknown.")
lines.append("- The store JSON contains image paths plus embeddings. The checkpoint must match the store geometry; otherwise the same paths can produce different retrieval behaviour.")
lines.append("- FAISS is used only to make nearest-neighbour lookup fast. It does not change the model geometry.")
lines.append("")
lines.append("## Mathematical Retrieval Geometry")
lines.append("Let `f_theta(x)` be the ConvNeXt Siamese embedding of image `x`, and let `B_K = {f_theta(k_1), ..., f_theta(k_n)}` be the known-bank embedding set. The retrieval distance used in the tables is the nearest-bank distance:")
lines.append("")
lines.append("```text")
lines.append("D_K(x) = min_{b in B_K} d(f_theta(x), b)")
lines.append("```")
lines.append("")
lines.append("where `d` is the embedding distance used by FAISS/nearest-neighbour search. A query is treated as duplicate when `D_K(x) <= tau`, where `tau` is the selected threshold. When the exact attack type is known during analysis, the report gives one threshold `tau_a` per attack. When attack type is unknown at deployment, the report gives dataset-level strict/balanced/loose threshold ranges.")
lines.append("")
lines.append("For a known image `x` and an attack transform `a_i`, the desired geometry is:")
lines.append("")
lines.append("```text")
lines.append("D_K(x) <= tau_known")
lines.append("d(f_theta(a_i(x)), f_theta(x)) <= epsilon_attack")
lines.append("```")
lines.append("")
lines.append("This gives the triangle-style intuition used in the project:")
lines.append("")
lines.append("```text")
lines.append("D_K(a_i(x)) <= d(f_theta(a_i(x)), f_theta(x)) + D_K(x)")
lines.append("```")
lines.append("")
lines.append("So if the original image is near the known bank, and every attacked variant is near the original embedding, the attacked variants should also remain inside the duplicate neighbourhood.")
lines.append("")
lines.append("For an unknown image `u`, the desired geometry is the opposite with respect to the known bank, while still keeping attack variants stable around their own original:")
lines.append("")
lines.append("```text")
lines.append("D_K(u) > tau_unknown")
lines.append("d(f_theta(a_i(u)), f_theta(u)) <= epsilon_attack")
lines.append("```")
lines.append("")
lines.append("Training uses this idea through positive and negative pair pressure: hard/soft positives reduce embedding distance for duplicate/attack variants, while hard/soft negatives increase separation between known and unknown neighbourhoods.")
lines.append("")
lines.append("## Store Counts")
lines.append("| Dataset | Store JSON | Known | Unknown | Total embeddings |")
lines.append("|---|---|---:|---:|---:|")
for name, path in stores.items():
    store = load(path)
    k = len(store.get("accepted_known", []))
    u = len(store.get("accepted_unknown", []))
    lines.append(f"| {display_name(name)} | `{path}` | {k:,} | {u:,} | {k + u:,} |")
lines.append("")
lines.append("## Proportional Pair Counts From Final Store Ratios")
lines.append("This table uses the same pair-ratio rule for every dataset, scaled from the final store size. The Amazon reference ratio is `hard_pos=N`, `soft_pos=0.5N`, `hard_neg=0.625N`, `soft_neg=0.375N`, so total pairs are `2.5N`.")
lines.append("")
lines.append("| Dataset | Known | Unknown | N | hard_pos | soft_pos | hard_neg | soft_neg | Total pairs |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, path in stores.items():
    store = load(path)
    known = len(store.get("accepted_known", []))
    unknown = len(store.get("accepted_unknown", []))
    vals = proportional_pair_counts(known, unknown)
    lines.append("| {} | {:,} | {:,} | {:,} | {:,} | {:,} | {:,} | {:,} | {:,} |".format(display_name(name), known, unknown, *vals))
lines.append("")
lines.append("Rounded values are used when the final store size is not divisible cleanly by the ratio.")
lines.append("")

amazon_store = load(stores["Amazon50k"])
lines.append("## Amazon Category Coverage")
lines.append("| Side | Images | Categories | Products | Top-level counts |")
lines.append("|---|---:|---:|---:|---|")
for key, label in [("accepted_known", "Known"), ("accepted_unknown", "Unknown")]:
    rows = amazon_store[key]
    cats = Counter(r.get("category", "unknown") for r in rows)
    prods = Counter(r.get("product", "unknown") for r in rows)
    tops = Counter(r.get("top_level") or r.get("category", "unknown").split("/")[0] for r in rows)
    top_str = ", ".join(f"{k} {v}" for k, v in sorted(tops.items()))
    lines.append(f"| {label} | {len(rows):,} | {len(cats):,} | {len(prods):,} | {top_str} |")
lines.append("")

lines.append("## Threshold Summary")
lines.append("| Dataset | Single global threshold | Recommended range | Strict buffer | Balanced buffer | Loose buffer |")
lines.append("|---|---:|---|---:|---:|---:|")
for name, path in completed.items():
    t = load(path)["threshold_summary"]
    lines.append("| {} | {:.6f} | {:.6f} to {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
        display_name(name),
        t["single_global_threshold"],
        t["recommended_range"][0],
        t["recommended_range"][1],
        t["strict_buffer_threshold"],
        t["balanced_buffer_threshold"],
        t["loose_buffer_threshold"],
    ))
lines.append("")

lines.append("## Multi-Query Summaries")
lines.append("CIFAR and Flickr use the fixed calibrated thresholds. UCID and Amazon use the latest threshold-search operating points, so the main summary reflects the best current stores instead of the older fixed-threshold rows.")
lines.append("")
lines.append("| Dataset | Q | All Avg | Non-Orig Avg | Min Non-Orig | Known Avg | Unknown Avg | Overall <70 | Known <70 | Unknown <70 | Known <50 | Unknown <50 | Original |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for dataset in ["CIFAR", "Flickr10k", "UCID", "Amazon50k"]:
    for name, q, s in summary_rows_for_dataset(dataset):
        lines.append(
            "| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {} | {} | {} | {} | {} | {:.2f} |".format(
                display_name(name),
                q,
                s["all_avg"],
                s["nonorig_avg"],
                s["min_nonorig"],
                summary_value(s, "known_nonorig_avg"),
                summary_value(s, "unknown_nonorig_avg"),
                s["overall_below70"],
                s["known_below70"],
                s["unknown_below70"],
                s["known_below50"],
                s["unknown_below50"],
                s["original"],
            )
        )
lines.append("")

lines.append("Interpretation: UCID is reported using its best average-total threshold setting, where q100 crosses the 85 line. Amazon is reported using its final-total threshold setting, where q1000 is the strongest current point and remains close to the 85 target.")
lines.append("")

lines.append("## Soft / Medium / Hard Performance Summary")
lines.append("This table uses the q100 evaluation for each dataset so all four datasets can be compared on the same query size.")
lines.append("")
lines.append("| Dataset | Original | Soft Avg | Medium Avg | Hard Avg | Strongest group | Weakest group | Best attack in weak group | Weakest attack |")
lines.append("|---|---:|---:|---:|---:|---|---|---|---|")
group_rows = {}
for name in ["CIFAR", "Flickr10k", "UCID", "Amazon50k"]:
    q100_attacks = selected_query_attacks(name)[0]["100"]["attacks"]
    group_rows[name] = group_stats_from_attacks(q100_attacks)
for name in ["CIFAR", "Flickr10k", "UCID", "Amazon50k"]:
    stats = group_rows[name]
    non_original = {k: v for k, v in stats.items() if k != "Original / mandatory gate"}
    strongest = max(non_original.items(), key=lambda item: item[1]["total"])[0]
    weakest = min(non_original.items(), key=lambda item: item[1]["total"])[0]
    weakest_attack, weakest_attack_total = stats[weakest]["weakest_attack"]
    best_weak_attack, best_weak_attack_total = stats[weakest]["best_attack"]
    lines.append(
        f"| {display_name(name)} | {stats['Original / mandatory gate']['total']:.2f} | "
        f"{stats['Soft']['total']:.2f} | {stats['Medium']['total']:.2f} | {stats['Hard']['total']:.2f} | "
        f"{strongest} | {weakest} | {best_weak_attack} ({best_weak_attack_total:.2f}) | "
        f"{weakest_attack} ({weakest_attack_total:.2f}) |"
    )
lines.append("")
lines.append("## Dataset Behaviour By Attack Type")
lines.append("- CIFAR performs best overall, especially on soft attacks and most medium attacks; the remaining weakness is mainly severe crop/flip tail behaviour.")
lines.append("- Flickr is very strong on soft attacks such as mild rotation, flips, brightness, resize, blur, and watermark, but harder crop/background attacks remain the main weak area.")
lines.append("- UCID reaches the target overall range after repair, with strong original/resize/blur/mild-rotation behaviour; the difficult side is still severe crop, contrast/brightness, and vertical flip tails.")
lines.append("- Amazon is strongest on original-like, crop5/crop20, rotate30, mixed, resize, and blur behaviour after the final refill; the weak side is known-side performance for heavy visual transformations such as flips, severe crop, brightness, and watermark.")
lines.append("")

for name in ["CIFAR", "Flickr10k", "UCID", "Amazon50k"]:
    queries, threshold_note = selected_query_attacks(name)
    lines.append(f"## {display_name(name)} Per-Attack Tables")
    lines.append(f"Threshold source: {threshold_note}. The threshold shown is the one used for that attack and query setting.")
    for q, qdata in queries.items():
        lines.append("")
        lines.append(f"### {display_name(name)} Q{q}")
        lines.append("| Attack | Known | Unknown | Total | Threshold |")
        lines.append("|---|---:|---:|---:|---:|")
        for attack, v in sorted(qdata["attacks"].items(), key=lambda kv: kv[1]["total"], reverse=True):
            lines.append(f"| {attack} | {v['known']:.2f} | {v['unknown']:.2f} | {v['total']:.2f} | {v['threshold']:.6f} |")
lines.append("")

lines.append("## Compact Combined Accuracy Tables")
lines.append("These tables show only combined total accuracy per attack. They are easier to paste into a paper because `Known` and `Unknown` are not split here.")
lines.append("")
compact_sources = [
    ("CIFAR", "fixed calibrated thresholds", "fixed", completed["CIFAR"], None),
    ("Flickr", "fixed calibrated thresholds", "fixed", completed["Flickr10k"], None),
    ("UCID", "latest threshold-search thresholds", "threshold", threshold_search["UCID relaxed refill"], "avg_total"),
    ("Amazon", "latest threshold-search thresholds", "threshold", threshold_search["Amazon50k"], "final_total"),
]
for name, description, source_type, path, policy in compact_sources:
    if not Path(path).exists():
        continue
    if source_type == "fixed":
        queries, rows = compact_fixed_rows(path)
    else:
        queries, rows = compact_threshold_rows(path, policy)
    lines.append(f"### {name} Combined Accuracy")
    lines.append(f"Threshold source: {description}.")
    lines.append("")
    lines.append("| Attack | Threshold | " + " | ".join(f"Q{q}" for q in queries) + " |")
    lines.append("|---|---:|" + "|".join("---:" for _ in queries) + "|")
    for attack, threshold, vals in rows:
        lines.append("| {} | {} | {} |".format(attack, threshold, " | ".join(f"{v:.2f}" for v in vals)))
    lines.append("")

lines.append("## Key Final Files")
lines.append("| Dataset | Checkpoint | Store JSON | Main eval JSON | Threshold-search JSON |")
lines.append("|---|---|---|---|---|")
for name in ["CIFAR", "Flickr10k", "UCID", "Amazon50k"]:
    if name == "UCID":
        threshold_path = threshold_search["UCID relaxed refill"]
    elif name == "Amazon50k":
        threshold_path = threshold_search["Amazon50k"]
    else:
        threshold_path = ""
    lines.append(
        f"| {display_name(name)} | `{ckpts[name]}` | `{stores[name]}` | `{completed[name]}` | "
        f"{'`' + str(threshold_path) + '`' if threshold_path else ''} |"
    )
lines.append("")

lines.append("## Eval JSON Paths")
for name, path in completed.items():
    lines.append(f"- {display_name(name)} fixed multi-query: `{path}`")
for name, path in threshold_search.items():
    if Path(path).exists():
        clean_name = "UCID" if name == "UCID relaxed refill" else display_name(name)
        lines.append(f"- {clean_name} threshold search: `{path}`")

out.write_text("\n".join(lines))
print(out)
print(len(lines))
