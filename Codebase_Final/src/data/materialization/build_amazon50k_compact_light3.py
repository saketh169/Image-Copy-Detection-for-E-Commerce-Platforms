#!/usr/bin/env python3
"""Create a compact, self-contained Amazon-50K light-duplicate image dataset.

Input is the verified local Amazon benchmark with 35K known and 15K unknown
original images. Output uses exactly two direct split roots. Each direct label
folder contains the untouched source `original.jpg` and three deterministic,
random light variants encoded as progressive optimized JPEG quality 92.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


EXPECTED = {"known": 35_000, "unknown": 15_000}
GLOBAL_SEED = 20261030
JPEG_QUALITY = 92
GENERATOR_VERSION = "amazon50k-compact-randomlight3-v1"
FINAL_18 = (
    "crop20", "crop5", "mix_crop20_watermark", "mix_rotate10_bright",
    "rotate30", "watermark_asset", "resize", "resize_compress",
    "watermark_text", "blur", "contrast", "rotate5", "bright_light",
    "rotate_m5", "crop10", "rotate_m10", "flip_h", "rotate15",
)
COMPOUND = {"mix_crop20_watermark", "mix_rotate10_bright"}
COMPANIONS = ("bright_light", "contrast", "blur", "resize", "resize_compress", "watermark_text", "watermark_asset")


@dataclass(frozen=True)
class ParentTask:
    split: str
    category: str
    subcategory: str
    source_path: str
    label: str
    target_folder: str


def source_id(source_path: str) -> str:
    return hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:20]


def seed_for(source_path: str, ordinal: int) -> int:
    payload = f"{GLOBAL_SEED}:{source_path}:{ordinal}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def reflected_rotate(image: Image.Image, degrees: float) -> Image.Image:
    width, height = image.size
    pad = max(1, min(min(width, height) // 2, int(max(width, height) * 0.10)))
    canvas = Image.new("RGB", (width + 2 * pad, height + 2 * pad))
    canvas.paste(image, (pad, pad))
    canvas.paste(image.crop((0, 0, width, pad)).transpose(Image.Transpose.FLIP_TOP_BOTTOM), (pad, 0))
    canvas.paste(image.crop((0, height - pad, width, height)).transpose(Image.Transpose.FLIP_TOP_BOTTOM), (pad, pad + height))
    canvas.paste(image.crop((0, 0, pad, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (0, pad))
    canvas.paste(image.crop((width - pad, 0, width, height)).transpose(Image.Transpose.FLIP_LEFT_RIGHT), (pad + width, pad))
    canvas.paste(image.crop((0, 0, pad, pad)).transpose(Image.Transpose.ROTATE_180), (0, 0))
    canvas.paste(image.crop((width - pad, 0, width, pad)).transpose(Image.Transpose.ROTATE_180), (pad + width, 0))
    canvas.paste(image.crop((0, height - pad, pad, height)).transpose(Image.Transpose.ROTATE_180), (0, pad + height))
    canvas.paste(image.crop((width - pad, height - pad, width, height)).transpose(Image.Transpose.ROTATE_180), (pad + width, pad + height))
    return canvas.rotate(degrees, resample=Image.Resampling.BICUBIC).crop((pad, pad, pad + width, pad + height))


def apply_family(image: Image.Image, family: str, rng: random.Random) -> tuple[Image.Image, list[dict[str, Any]]]:
    width, height = image.size
    if family in {"crop5", "crop10", "crop20"}:
        fraction = rng.uniform(0.03, 0.08)
        cw, ch = max(2, int(width * (1.0 - fraction))), max(2, int(height * (1.0 - fraction)))
        left, top = rng.randint(0, max(0, width - cw)), rng.randint(0, max(0, height - ch))
        edited = image.crop((left, top, left + cw, top + ch)).resize((width, height), Image.Resampling.LANCZOS)
        return edited, [{"family": family, "op": "crop_resize", "fraction": round(fraction, 6), "left": left, "top": top}]
    if family in {"rotate_m5", "rotate_m10", "rotate5", "rotate15", "rotate30"}:
        limits = {"rotate_m5": (-5, -3), "rotate_m10": (-10, -5), "rotate5": (3, 5), "rotate15": (5, 10), "rotate30": (8, 15)}
        low, high = limits[family]
        degrees = rng.uniform(low, high)
        return reflected_rotate(image, degrees), [{"family": family, "op": "reflect_rotate", "degrees": round(degrees, 6)}]
    if family == "bright_light":
        factor = rng.uniform(1.05, 1.15)
        return ImageEnhance.Brightness(image).enhance(factor), [{"family": family, "op": "brightness", "factor": round(factor, 6)}]
    if family == "contrast":
        factor = rng.uniform(0.92, 1.10)
        return ImageEnhance.Contrast(image).enhance(factor), [{"family": family, "op": "contrast", "factor": round(factor, 6)}]
    if family == "blur":
        radius = rng.uniform(0.35, 0.85)
        return image.filter(ImageFilter.GaussianBlur(radius)), [{"family": family, "op": "gaussian_blur", "radius": round(radius, 6)}]
    if family in {"resize", "resize_compress"}:
        scale = rng.uniform(0.80, 0.92)
        resized = image.resize((max(2, int(width * scale)), max(2, int(height * scale))), Image.Resampling.BILINEAR).resize((width, height), Image.Resampling.LANCZOS)
        params: dict[str, Any] = {"family": family, "op": "resample", "scale": round(scale, 6)}
        if family == "resize_compress":
            import io
            intermediate_quality = rng.randint(84, 94)
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=intermediate_quality)
            with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
                resized = decoded.convert("RGB")
            params["intermediate_jpeg_quality"] = intermediate_quality
        return resized, [params]
    if family in {"watermark_text", "watermark_asset"}:
        alpha = rng.randint(35, 75)
        scale = rng.uniform(0.10, 0.17)
        box_width, box_height = max(42, int(width * scale)), max(18, int(width * scale * 0.42))
        left, top = rng.randint(0, max(0, width - box_width)), rng.randint(0, max(0, height - box_height))
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        label = "SHOP" if family == "watermark_asset" else rng.choice(("SALE", "NEW", "OFFER", "DEAL"))
        fill = (255, 255, 255, alpha) if family == "watermark_asset" else (30, 30, 30, alpha)
        outline = (20, 20, 20, alpha) if family == "watermark_asset" else (255, 255, 255, alpha)
        draw.rounded_rectangle((left, top, left + box_width, top + box_height), radius=3, fill=fill, outline=outline, width=1)
        font = ImageFont.load_default()
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.text((left + max(1, (box_width - (bounds[2] - bounds[0])) // 2), top + max(1, (box_height - (bounds[3] - bounds[1])) // 2)), label, fill=outline, font=font)
        edited = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
        return edited, [{"family": family, "op": "small_watermark", "label": label, "alpha": alpha, "left": left, "top": top, "relative_width": round(scale, 6)}]
    if family == "flip_h":
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), [{"family": family, "op": "horizontal_flip"}]
    if family == "mix_rotate10_bright":
        first, a = apply_family(image, "rotate15", rng)
        second, b = apply_family(first, "bright_light", rng)
        return second, a + b
    if family == "mix_crop20_watermark":
        first, a = apply_family(image, "crop20", rng)
        second, b = apply_family(first, "watermark_asset", rng)
        return second, a + b
    raise ValueError(f"Unknown family {family}")


def build_parent(task: ParentTask) -> tuple[str, bool, str | None, list[dict[str, Any]] | None]:
    source, target = Path(task.source_path), Path(task.target_folder)
    if not source.is_file():
        return task.label, False, f"Missing source: {source}", None
    if target.exists():
        expected = [target / "original.jpg", *sorted(target.glob("dup_*.jpg"))]
        if len(expected) == 4 and all(path.is_file() and path.stat().st_size > 0 for path in expected):
            return task.label, False, None, []
        return task.label, False, f"Incomplete target exists: {target}", None
    try:
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        target.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, target / "original.jpg")
        source_seed = seed_for(task.source_path, 0)
        families = random.Random(source_seed).sample(list(FINAL_18), k=3)
        variants: list[dict[str, Any]] = []
        for ordinal, primary in enumerate(families, start=1):
            variant_seed = seed_for(task.source_path, ordinal)
            add_secondary = ordinal == 2 or random.Random(variant_seed ^ 0xA5A5).random() < 0.50
            secondary = None
            if primary not in COMPOUND and add_secondary:
                choices = [name for name in COMPANIONS if name != primary]
                secondary = choices[random.Random(variant_seed ^ 0x5A5A).randrange(len(choices))]
            rng = random.Random(variant_seed)
            edited, operations = apply_family(original, primary, rng)
            if secondary:
                edited, next_operations = apply_family(edited, secondary, rng)
                operations.extend(next_operations)
            name = primary if secondary is None else f"{primary}__{secondary}"
            filename = f"dup_{ordinal:02d}_{name}.jpg"
            temporary = target / f".{filename}.{os.getpid()}.tmp.jpg"
            edited.save(temporary, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
            temporary.replace(target / filename)
            variants.append({"filename": filename, "seed": variant_seed, "primary_family": primary, "secondary_family": secondary, "operations": operations})
        return task.label, True, None, variants
    except Exception as exc:
        return task.label, False, f"{type(exc).__name__}: {exc}", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1 or not args.source.is_dir():
        parser.error("Source must exist and workers must be at least one")
    if args.output.exists():
        parser.error(f"Refusing to overwrite output: {args.output}")
    label_file = args.source / "image_labels.json"
    if not label_file.is_file():
        parser.error(f"Missing source label manifest: {label_file}")
    source_manifest = json.loads(label_file.read_text(encoding="utf-8"))
    labels = source_manifest.get("labels")
    if not isinstance(labels, list):
        raise ValueError("image_labels.json does not contain labels")
    args.output.mkdir(parents=True)
    tasks: list[ParentTask] = []
    seen: dict[str, set[str]] = {"known": set(), "unknown": set()}
    for row in labels:
        split, category, subcategory, filename = row.get("label"), row.get("category"), row.get("subcategory"), row.get("filename")
        relative = row.get("path")
        if split not in EXPECTED or not all(isinstance(value, str) and value for value in (category, subcategory, filename, relative)):
            raise ValueError(f"Malformed source label row: {row!r}")
        source_path = args.source / Path(*Path(relative).parts[1:])
        label = f"{category}__{subcategory}__{Path(filename).stem}"
        if label in seen[split]:
            raise ValueError(f"Duplicate direct label in {split}: {label}")
        seen[split].add(label)
        tasks.append(ParentTask(split, category, subcategory, str(source_path), label, str(args.output / split / label)))
    for split, expected in EXPECTED.items():
        if len(seen[split]) != expected:
            raise ValueError(f"Expected {expected} {split} records; found {len(seen[split])}")
    if seen["known"] & seen["unknown"]:
        raise ValueError("Known and unknown direct labels overlap")

    print(f"Validated {len(seen['known'])} known and {len(seen['unknown'])} unknown source labels.", flush=True)
    metadata_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = copied = existing = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for task, result in zip(tasks, executor.map(build_parent, tasks, chunksize=16)):
            _, created, error, variants = result
            completed += 1
            if error:
                errors.append(error)
            elif created:
                copied += 1
            else:
                existing += 1
            if variants is not None:
                metadata_rows.append({"split": task.split, "label": task.label, "category": task.category, "subcategory": task.subcategory, "source_path": task.source_path, "variants": variants})
            if completed % 1000 == 0 or completed == len(tasks):
                print(f"progress={completed}/{len(tasks)} folders={copied} existing={existing} errors={len(errors)}", flush=True)
    manifests = args.output / "manifests"
    manifests.mkdir(exist_ok=True)
    if errors:
        (manifests / "generation_errors.txt").write_text("\n".join(errors) + "\n", encoding="utf-8")
        raise RuntimeError(f"Generation failed for {len(errors)} folders")
    if len(metadata_rows) != len(tasks):
        raise RuntimeError("Missing manifest rows")
    with (manifests / "folders.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    metadata = {"dataset": "Amazon-50K compact random-light duplicate benchmark", "source": str(args.source), "known_folders": EXPECTED["known"], "unknown_folders": EXPECTED["unknown"], "images_per_folder": 4, "total_images": 200_000, "jpeg_quality": JPEG_QUALITY, "progressive_jpeg": True, "generator_version": GENERATOR_VERSION, "global_seed": GLOBAL_SEED, "created_folders": copied}
    (manifests / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.output / "README.md").write_text("# Amazon-50K Compact Light-Duplicate Dataset\n\nTwo direct roots: `known/` and `unknown/`. Every direct label folder contains the source `original.jpg` plus three deterministic random light variants. Variant JPEGs are optimized progressive quality 92.\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
