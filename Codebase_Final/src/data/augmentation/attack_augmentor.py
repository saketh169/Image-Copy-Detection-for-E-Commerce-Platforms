"""In-memory image attacks for the approved final 19-attack protocol.

The generator operates on OpenCV BGR arrays and never writes variants to disk.
It is deliberately shared by training-pair construction and evaluation helpers
so attack names cannot drift from the final benchmark manifest.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

from config import ASSET_DIR, HARD_POS_MAX_ATTACKS, HARD_POS_MIN_ATTACKS


FINAL_19_ATTACKS = (
    "original", "crop20", "crop5", "mix_crop20_watermark", "mix_rotate10_bright",
    "rotate30", "watermark_asset", "resize", "resize_compress", "watermark_text",
    "blur", "contrast", "rotate5", "bright_light", "rotate_m5", "crop10",
    "rotate_m10", "flip_h", "rotate15",
)
_RANDOM_ATTACKS = tuple(name for name in FINAL_19_ATTACKS if name not in {"original", "mix_crop20_watermark", "mix_rotate10_bright"})


class CopyAttackAugmentor:
    """Generate approved marketplace-like image transformations in memory."""

    def __init__(self, asset_dir: str | Path = ASSET_DIR, seed: int | None = None):
        self.asset_dir = Path(asset_dir)
        self.rng = random.Random(seed)
        self._assets = self._discover_assets()

    def _discover_assets(self) -> list[Path]:
        if not self.asset_dir.is_dir():
            return []
        return sorted(
            path for path in self.asset_dir.rglob("*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

    @staticmethod
    def _restore(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)

    def _rotate(self, image: np.ndarray, degrees: float) -> np.ndarray:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
        return cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _crop(self, image: np.ndarray, fraction: float) -> np.ndarray:
        height, width = image.shape[:2]
        crop_height = max(2, int(height * (1.0 - fraction)))
        crop_width = max(2, int(width * (1.0 - fraction)))
        top = self.rng.randint(0, max(0, height - crop_height))
        left = self.rng.randint(0, max(0, width - crop_width))
        return self._restore(image[top:top + crop_height, left:left + crop_width], image)

    def _brightness(self, image: np.ndarray, beta: int) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)

    def _contrast(self, image: np.ndarray) -> np.ndarray:
        alpha = self.rng.uniform(0.72, 1.35)
        beta = self.rng.randint(-18, 18)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _resize(self, image: np.ndarray, quality: int | None = None) -> np.ndarray:
        height, width = image.shape[:2]
        scale = self.rng.uniform(0.42, 0.75)
        reduced = cv2.resize(image, (max(2, int(width * scale)), max(2, int(height * scale))), interpolation=cv2.INTER_AREA)
        resized = self._restore(reduced, image)
        if quality is None:
            return resized
        ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else resized

    def _watermark_text(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        labels = ("SALE", "50% OFF", "AUTHENTIC", "VERIFIED", "LIMITED")
        label = self.rng.choice(labels)
        height, width = result.shape[:2]
        scale = max(0.45, min(width, height) / 430)
        thickness = max(1, int(scale * 2))
        x = self.rng.randint(5, max(5, width // 2))
        y = self.rng.randint(max(20, height // 6), max(20, height - 8))
        overlay = result.copy()
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 230), thickness + 2, cv2.LINE_AA)
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.72, result, 0.28, 0)

    def _watermark_asset(self, image: np.ndarray) -> np.ndarray:
        if not self._assets:
            return self._watermark_text(image)
        asset = cv2.imread(str(self.rng.choice(self._assets)), cv2.IMREAD_UNCHANGED)
        if asset is None:
            return self._watermark_text(image)
        height, width = image.shape[:2]
        target_width = max(12, int(width * self.rng.uniform(0.12, 0.30)))
        target_height = max(12, int(asset.shape[0] * target_width / max(1, asset.shape[1])))
        asset = cv2.resize(asset, (target_width, target_height), interpolation=cv2.INTER_AREA)
        x = self.rng.randint(0, max(0, width - target_width))
        y = self.rng.randint(0, max(0, height - target_height))
        result = image.copy()
        if asset.ndim == 3 and asset.shape[2] == 4:
            alpha = (asset[:, :, 3:4].astype(np.float32) / 255.0) * self.rng.uniform(0.45, 0.78)
            foreground = asset[:, :, :3].astype(np.float32)
            background = result[y:y + target_height, x:x + target_width].astype(np.float32)
            result[y:y + target_height, x:x + target_width] = (foreground * alpha + background * (1 - alpha)).astype(np.uint8)
        else:
            result[y:y + target_height, x:x + target_width] = cv2.addWeighted(
                result[y:y + target_height, x:x + target_width], 0.45, asset[:, :, :3], 0.55, 0
            )
        return result

    def fixed_attack(self, image: np.ndarray, attack_name: str | None = None) -> np.ndarray:
        """Apply one named final-protocol attack and preserve input resolution."""
        if image is None:
            raise ValueError("image must be a valid OpenCV BGR array")
        name = attack_name or self.rng.choice(_RANDOM_ATTACKS)
        aliases = {"rotate-5": "rotate_m5", "rotate-10": "rotate_m10", "bright": "bright_light", "flip": "flip_h"}
        name = aliases.get(name, name)
        if name == "original":
            return image.copy()
        if name == "crop5":
            return self._crop(image, 0.05)
        if name == "crop10":
            return self._crop(image, 0.10)
        if name == "crop20":
            return self._crop(image, 0.20)
        if name == "rotate_m10":
            return self._rotate(image, -10)
        if name == "rotate_m5":
            return self._rotate(image, -5)
        if name == "rotate5":
            return self._rotate(image, 5)
        if name == "rotate15":
            return self._rotate(image, 15)
        if name == "rotate30":
            return self._rotate(image, 30)
        if name == "flip_h":
            return cv2.flip(image, 1)
        if name == "bright_light":
            return self._brightness(image, self.rng.randint(12, 34))
        if name == "contrast":
            return self._contrast(image)
        if name == "blur":
            return cv2.GaussianBlur(image, (5, 5), self.rng.uniform(0.6, 1.8))
        if name == "resize":
            return self._resize(image)
        if name == "resize_compress":
            return self._resize(image, quality=self.rng.randint(28, 58))
        if name == "watermark_text":
            return self._watermark_text(image)
        if name == "watermark_asset":
            return self._watermark_asset(image)
        if name == "mix_crop20_watermark":
            return self._watermark_asset(self._crop(image, 0.20))
        if name == "mix_rotate10_bright":
            return self._brightness(self._rotate(image, 10), self.rng.randint(12, 34))
        raise ValueError(f"Unsupported final attack: {attack_name}")

    def random_attack_combo(self, image: np.ndarray) -> np.ndarray:
        """Apply a reproducible-sized hard-positive composition in memory."""
        count = self.rng.randint(HARD_POS_MIN_ATTACKS, HARD_POS_MAX_ATTACKS)
        selected = self.rng.sample(_RANDOM_ATTACKS, k=min(count, len(_RANDOM_ATTACKS)))
        result = image.copy()
        for name in selected:
            result = self.fixed_attack(result, name)
        return result
