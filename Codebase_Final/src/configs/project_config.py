"""
config.py - Central Configuration
===================================
All paths, hyperparameters, and constants in ONE place.
Nothing is hardcoded anywhere else in the codebase.
"""

import torch
import os

# --- Device -------------------------------------------------
# Use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Paths --------------------------------------------------
PROJECT_ROOT    = "/home/saketh"
DATASET_ROOT    = os.path.join(PROJECT_ROOT, "Data Processed")       # Where processed images live
ASSET_DIR       = os.path.join(PROJECT_ROOT, "assets")               # Watermark assets (PNGs)
UNIVERSE_PATH   = os.path.join(PROJECT_ROOT, "json", "training_universe.json")  # Universe mapping (create_universe.py output)
MANIFEST_PATH   = os.path.join(PROJECT_ROOT, "json", "training_resnet_pairs.json")  # Model trainers swap in their own 2k manifest file in the same directory
CHECKPOINT_DIR  = os.environ.get(
    "BTP_CHECKPOINT_DIR",
    os.path.join(PROJECT_ROOT, "Codebase", "checkpoints")
)  # Saved model weights

# --- Image --------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# --- Embedding Dimensions ----------------------------------
GLOBAL_DIM = 256
LOCAL_DIM  = 128
EMBEDDING_DIM = GLOBAL_DIM + LOCAL_DIM

# --- Loss Function -----------------------------------------
MARGIN       = float(os.environ.get("BTP_MARGIN", "1.0"))
LOCAL_WEIGHT = 0.5
KNOWN_WEIGHT = float(os.environ.get("BTP_KNOWN_WEIGHT", "1.0"))
CONTRASTIVE_WEIGHT = float(os.environ.get("BTP_CONTRASTIVE_WEIGHT", "1.0"))
KNOWN_POS_WEIGHT = float(os.environ.get("BTP_KNOWN_POS_WEIGHT", "1.0"))
KNOWN_NEG_WEIGHT = float(os.environ.get("BTP_KNOWN_NEG_WEIGHT", "3.0"))

# --- Known vs Unknown --------------------------------------
# Used by known-category detector training and evaluation
TRAINED_TOP_LEVEL_CATEGORIES = {"baby_products", "home"}

# --- Training ----------------------------------------------
NUM_EPOCHS     = int(os.environ.get("BTP_NUM_EPOCHS", "50"))
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
BATCH_SIZE     = int(os.environ.get("BTP_BATCH_SIZE", "64"))   # Per batch: 16 of each pair type when batch size is 64

# --- Priority Queue ----------------------------------------
DECAY_FACTOR    = 0.9
PRUNE_THRESHOLD = 0.05
MAX_TRAININGS   = 5
DISABLE_PRUNING = os.environ.get("BTP_DISABLE_PRUNING", "0") == "1"
MIN_REMAINING_PAIRS = int(os.environ.get("BTP_MIN_REMAINING_PAIRS", "0"))

# --- Initial Priority Values -------------------------------
INIT_PRIORITY = {
    "soft_pos": 0.0,
    "soft_neg": 0.0,
    "hard_neg": 0.20,
    "hard_pos": 0.50,
}

# --- Batch Composition -------------------------------------
_BATCH_PER_TYPE = max(1, BATCH_SIZE // 4)
BATCH_COMPOSITION = {
    "hard_pos": _BATCH_PER_TYPE,
    "hard_neg": _BATCH_PER_TYPE,
    "soft_pos": _BATCH_PER_TYPE,
    "soft_neg": _BATCH_PER_TYPE,
}

# --- Hard Positive Attack Config ---------------------------
HARD_POS_MIN_ATTACKS = 3
HARD_POS_MAX_ATTACKS = 4
