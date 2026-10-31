"""Runtime configuration shared by canonical training and evaluation scripts.

Environment variables override every machine-specific location.  No datasets,
checkpoints, or generated benchmark outputs are stored in this repository.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
FINAL_ARTIFACTS_ROOT = Path(
    os.environ.get("FINAL_ARTIFACTS_ROOT", PROJECT_ROOT.parent / "Final_Artifacts")
).resolve()
DATASET_ROOT = Path(os.environ.get("DATA_ROOT", FINAL_ARTIFACTS_ROOT / "data")).resolve()
ASSET_DIR = Path(os.environ.get("BTP_ASSET_DIR", FINAL_ARTIFACTS_ROOT / "assets")).resolve()
JSON_ROOT = Path(os.environ.get("BTP_JSON_ROOT", FINAL_ARTIFACTS_ROOT / "json")).resolve()
UNIVERSE_PATH = Path(os.environ.get("BTP_UNIVERSE_PATH", JSON_ROOT / "training_universe.json"))
MANIFEST_PATH = Path(os.environ.get("BTP_MANIFEST_PATH", JSON_ROOT / "training_pairs.json"))
CHECKPOINT_DIR = Path(
    os.environ.get("BTP_CHECKPOINT_DIR", FINAL_ARTIFACTS_ROOT / "checkpoints")
).resolve()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GLOBAL_DIM = 256
LOCAL_DIM = 128
EMBEDDING_DIM = GLOBAL_DIM + LOCAL_DIM

MARGIN = float(os.environ.get("BTP_MARGIN", "1.0"))
LOCAL_WEIGHT = float(os.environ.get("BTP_LOCAL_WEIGHT", "0.5"))
KNOWN_WEIGHT = float(os.environ.get("BTP_KNOWN_WEIGHT", "1.0"))
CONTRASTIVE_WEIGHT = float(os.environ.get("BTP_CONTRASTIVE_WEIGHT", "1.0"))
KNOWN_POS_WEIGHT = float(os.environ.get("BTP_KNOWN_POS_WEIGHT", "1.0"))
KNOWN_NEG_WEIGHT = float(os.environ.get("BTP_KNOWN_NEG_WEIGHT", "3.0"))

TRAINED_TOP_LEVEL_CATEGORIES = {"baby_products", "home"}
NUM_EPOCHS = int(os.environ.get("BTP_NUM_EPOCHS", "50"))
LEARNING_RATE = float(os.environ.get("BTP_LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("BTP_WEIGHT_DECAY", "1e-5"))
BATCH_SIZE = int(os.environ.get("BTP_BATCH_SIZE", "64"))

DECAY_FACTOR = float(os.environ.get("BTP_DECAY_FACTOR", "0.9"))
PRUNE_THRESHOLD = float(os.environ.get("BTP_PRUNE_THRESHOLD", "0.05"))
MAX_TRAININGS = int(os.environ.get("BTP_MAX_TRAININGS", "5"))
DISABLE_PRUNING = os.environ.get("BTP_DISABLE_PRUNING", "0") == "1"
MIN_REMAINING_PAIRS = int(os.environ.get("BTP_MIN_REMAINING_PAIRS", "0"))
INIT_PRIORITY = {"soft_pos": 0.0, "soft_neg": 0.0, "hard_neg": 0.20, "hard_pos": 0.50}
_BATCH_PER_TYPE = max(1, BATCH_SIZE // 4)
BATCH_COMPOSITION = {name: _BATCH_PER_TYPE for name in INIT_PRIORITY}
HARD_POS_MIN_ATTACKS = 3
HARD_POS_MAX_ATTACKS = 4
