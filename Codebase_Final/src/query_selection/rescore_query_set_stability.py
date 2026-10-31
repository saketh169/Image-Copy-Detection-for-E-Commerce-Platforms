#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from pathlib import Path


LOG_DIR = Path("/home/saketh/json/final_named_artifacts_20260601/phase3_final_nearest_bank_all4_20260717/logs_spread")
OUT = Path("/home/saketh/json/final_named_artifacts_20260601/phase3_final_nearest_bank_all4_20260717/spread_plus_quality_rescore.json")


def spread_quality_score(row: dict) -> float:
    total = float(row.get("total", 0.0))
    known = float(row.get("known", 0.0))
    unknown = float(row.get("unknown", 0.0))
    origk = float(row.get("origK", 0.0))
    origu = float(row.get("origU", 0.0))
    mink = float(row.get("minK", 0.0))
    minu = float(row.get("minU", 0.0))
    attacks = row.get("attacks", [])

    score = 0.0
    # Presentation target: balanced, but not weak.
    score += abs(total - 90.0) * 2.0
    score += abs(known - unknown) * 1.25
    score += max(0.0, 88.0 - total) * 6.0
    score += max(0.0, 88.0 - known) * 6.0
    score += max(0.0, 85.0 - unknown) * 4.0

    # Strong protection for originals and tails.
    score += max(0.0, 100.0 - origk) * 100.0
    score += max(0.0, 95.0 - origu) * 10.0
    score += max(0.0, 40.0 - mink) * 20.0
    score += max(0.0, 40.0 - minu) * 20.0
    # Reward higher OrigU/MinK/MinU when all else is close.
    score -= min(origu, 100.0) * 0.18
    score -= min(mink, 100.0) * 0.12
    score -= min(minu, 100.0) * 0.08

    # Penalize too many identical/perfect rows, but do not overrule quality.
    non_original = [r for r in attacks if r.get("attack") != "original"]
    perfect = sum(
        1
        for r in non_original
        if float(r.get("known", 0.0)) >= 99.999 and float(r.get("unknown", 0.0)) >= 99.999
    )
    score += max(0, perfect - 2) * 4.0
    totals = [round(float(r.get("total", 0.0)), 1) for r in non_original]
    if totals:
        unique_ratio = len(set(totals)) / len(totals)
        score += max(0.0, 0.50 - unique_ratio) * 15.0

    return round(score, 6)


def parse_logs():
    data: dict[str, list[dict]] = {}
    for ds in ["cifar", "flickr", "ucid", "amazon"]:
        logs = sorted(LOG_DIR.glob(f"{ds}_phase3_spread_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        rows = []
        if logs:
            for line in logs[0].read_text(errors="replace").splitlines():
                if not line.startswith("[seed]"):
                    continue
                m = re.search(r"^\[seed\]\s+(\w+)\s+q\s+(\d+)\s+seed\s+(\d+)\s+score\s+([^\s]+)\s+(\{.*\})$", line)
                if not m:
                    continue
                row = ast.literal_eval(m.group(5))
                rows.append({
                    "dataset": ds,
                    "q": int(m.group(2)),
                    "seed": int(m.group(3)),
                    "old_score": float(m.group(4)),
                    "new_score": spread_quality_score(row),
                    **{k: row.get(k) for k in ["total", "known", "unknown", "origK", "origU", "minK", "minU", "minAtk"]},
                    "attacks": row.get("attacks", []),
                })
        data[ds] = rows
    return data


def main():
    data = parse_logs()
    selected = {}
    for ds, rows in data.items():
        by_q = {}
        for row in rows:
            by_q.setdefault(str(row["q"]), []).append(row)
        selected[ds] = {
            q: sorted(qrows, key=lambda r: r["new_score"])[0]
            for q, qrows in by_q.items()
            if qrows
        }
    payload = {"rule": "spread plus higher OrigU/MinK/MinU", "rows": data, "selected": selected}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for ds in ["cifar", "flickr", "ucid", "amazon"]:
        print(f"\n## {ds.upper()}")
        rows = sorted(data.get(ds, []), key=lambda r: (r["q"], r["new_score"]))
        if not rows:
            print("no rows yet")
            continue
        print("| Q | Seed | NewScore | OldScore | Total | Known | Unknown | OrigK | OrigU | MinK | MinU |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows[:20]:
            print("| {q} | {seed} | {new_score:.2f} | {old_score:.2f} | {total:.2f} | {known:.2f} | {unknown:.2f} | {origK:.0f} | {origU:.0f} | {minK:.0f} | {minU:.0f} |".format(**r))


if __name__ == "__main__":
    main()
