"""
dataloader.py — Siamese Dataset & DataLoader
==============================================
Reads the master manifest and serves image pairs during training.

Key behavior per pair type:
  - soft_pos:  Load org + duplicate from disk (both exist as files)
  - hard_pos:  Load org twice, apply RANDOM attacks to 2nd copy IN RAM
  - soft_neg:  Load two different products from disk (different categories)
  - hard_neg:  Load two different products from disk (same category)

Returns per sample: (img1_tensor, img2_tensor, label, manifest_index)
  - manifest_index is needed so train.py can update priority back in manifest
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import hashlib
from functools import lru_cache
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms as T

from config import (
    MANIFEST_PATH, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    BATCH_SIZE, DEVICE, TRAINED_TOP_LEVEL_CATEGORIES
)
from data.augmentation.attack_augmentor import CopyAttackAugmentor


@lru_cache(maxsize=4)
def _load_known_lookup(lookup_path):
    if not lookup_path or not os.path.exists(lookup_path):
        return {}
    with open(lookup_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        return {}
    lookup = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        path = row.get("path")
        if not path:
            continue
        lookup[path] = 1.0 if row.get("known") else 0.0
    return lookup


# ─── Standard Transform Pipeline ────────────────────────
# Applied to ALL images before feeding to ResNet-50
def get_transform(strong_aug=False):
    """
    Resize → Convert to tensor → Normalize with ImageNet stats.
    ResNet-50 was pretrained on ImageNet, so we must match its preprocessing.
    """
    if strong_aug:
        return T.Compose([
            T.ToPILImage(),
            T.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04)], p=0.7),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.8))], p=0.35),
            T.RandomApply([T.RandomAffine(degrees=7, translate=(0.03, 0.03), scale=(0.95, 1.05), shear=4)], p=0.35),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),                              # HWC uint8 → CHW float [0,1]
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Match ImageNet distribution
    ])


def _top_level_category(path):
    parts = Path(path).parts
    if "Data Processed" in parts:
        idx = parts.index("Data Processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _coverage_attack_names():
    preferred = os.environ.get(
        "BTP_HARD_POS_COVERAGE_ATTACKS",
        "rotate,crop,bright,flip,resize_compress,watermark_text",
    )
    return [name.strip() for name in preferred.split(",") if name.strip()]


def _soft_pos_attack_names():
    preferred = os.environ.get(
        "BTP_SOFT_POS_ATTACKS",
        "rotate-5,rotate-10,rotate5,rotate15,crop5,bright_light,bright,resize,watermark_asset,blur",
    )
    return [name.strip() for name in preferred.split(",") if name.strip()]


def _stable_attack_offset(path, modulo):
    if modulo <= 0:
        return 0
    digest = hashlib.md5(path.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


class SiameseDataset(Dataset):
    """
    PyTorch Dataset that reads pairs from the manifest.

    The manifest is a list of dicts, each with:
      row_id, path_1, path_2, label, pair_type, priority, times_trained
    """

    def __init__(self, manifest_path=None, transform=None, return_known_labels=False):
        if manifest_path is None:
            manifest_path = MANIFEST_PATH
        
        with open(manifest_path, 'r') as f:
            self.pairs = json.load(f)

        self.augmentor = CopyAttackAugmentor()
        if transform is not None:
            self.transform = transform
        else:
            strong_aug = os.environ.get("BTP_STRONG_AUG", "0").strip() == "1"
            self.transform = get_transform(strong_aug=strong_aug)
        self.return_known_labels = return_known_labels
        lookup_path = os.environ.get("BTP_PATH_KNOWN_JSON", "").strip()
        self.known_lookup = _load_known_lookup(lookup_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        img1_path  = pair['path_1']
        img2_path  = pair['path_2']
        label      = pair['label']
        pair_type  = pair['pair_type']

        # ── Load images (BGR format from cv2) ──
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # Safety: if image file is missing/corrupt, return zeros
        if img1 is None or img2 is None:
            blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            if self.return_known_labels:
                return (
                    blank,
                    blank,
                    torch.tensor(label, dtype=torch.float32),
                    idx,
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                )
            return blank, blank, torch.tensor(label, dtype=torch.float32), idx

        # ── Positive pairs: originals-only runs use self-pairs and attack img2 in RAM ──
        # hard_pos gets stronger multi-attack coverage, soft_pos gets a lighter single attack.
        prebuilt_soft_pos = (
            pair_type == 'soft_pos'
            and bool(pair.get('prebuilt_soft_pos'))
            and img1_path != img2_path
        )
        if pair_type in {'hard_pos', 'soft_pos'} and not prebuilt_soft_pos:
            if os.environ.get("BTP_HARD_POS_COVERAGE", "0") == "1" and hasattr(self.augmentor, "fixed_attack"):
                attack_names = _coverage_attack_names() if pair_type == 'hard_pos' else _soft_pos_attack_names()
                if attack_names:
                    train_step = int(pair.get("times_trained", 0))
                    offset = _stable_attack_offset(img1_path, len(attack_names))
                    primary = attack_names[(train_step + offset) % len(attack_names)]
                    img2 = self.augmentor.fixed_attack(img2, primary)

                    if (
                        pair_type == 'hard_pos'
                        and os.environ.get("BTP_HARD_POS_COVERAGE_COMBO", "1") == "1"
                        and len(attack_names) > 1
                    ):
                        combo_gap = int(os.environ.get("BTP_HARD_POS_COVERAGE_COMBO_GAP", "1"))
                        secondary = attack_names[(train_step + offset + combo_gap) % len(attack_names)]
                        if secondary != primary:
                            img2 = self.augmentor.fixed_attack(img2, secondary)
                        if len(attack_names) > 2 and os.environ.get("BTP_HARD_POS_COVERAGE_TERTIARY", "1") == "1":
                            tertiary_gap = int(os.environ.get("BTP_HARD_POS_COVERAGE_TERTIARY_GAP", "3"))
                            tertiary = attack_names[(train_step + offset + tertiary_gap) % len(attack_names)]
                            if tertiary not in (primary, secondary):
                                img2 = self.augmentor.fixed_attack(img2, tertiary)
                else:
                    img2 = self.augmentor.random_attack_combo(img2)
            else:
                if pair_type == 'hard_pos':
                    img2 = self.augmentor.random_attack_combo(img2)
                else:
                    img2 = self.augmentor.fixed_attack(img2)

        # ── Convert BGR → RGB (PyTorch expects RGB) ──
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # ── Apply standard transforms (resize, tensor, normalize) ──
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        if self.return_known_labels:
            if self.known_lookup:
                known1 = self.known_lookup.get(img1_path, 0.0)
                known2 = self.known_lookup.get(img2_path, 0.0)
            else:
                cat1 = _top_level_category(img1_path)
                cat2 = _top_level_category(img2_path)
                known1 = 1.0 if cat1 in TRAINED_TOP_LEVEL_CATEGORIES else 0.0
                known2 = 1.0 if cat2 in TRAINED_TOP_LEVEL_CATEGORIES else 0.0
            return (
                img1,
                img2,
                torch.tensor(label, dtype=torch.float32),
                idx,
                torch.tensor(known1, dtype=torch.float32),
                torch.tensor(known2, dtype=torch.float32),
            )

        return img1, img2, torch.tensor(label, dtype=torch.float32), idx


def get_dataloader(manifest_path=None, sampler=None, shuffle=False, return_known_labels=False):
    """
    Create a DataLoader from the manifest.
    
    If a custom sampler is provided (priority_sampler), use that.
    Otherwise, optionally shuffle.
    
    Note: When using a sampler, shuffle must be False (PyTorch rule).
    """
    dataset = SiameseDataset(manifest_path=manifest_path, return_known_labels=return_known_labels)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=2,       # Parallel data loading (adjust if issues)
        pin_memory=True,     # Faster GPU transfer
        drop_last=True,      # Drop incomplete last batch
    )

    return loader, dataset
