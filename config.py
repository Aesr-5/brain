"""
Configuration settings for BRATS 2020 Brain Tumor Segmentation
"""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
CHECKPOINTS_DIR = Path("checkpoints")
OUTPUT_DIR = Path("output")
LOGS_DIR = Path("logs")

# Create directories if they don't exist
for directory in [DATA_DIR, CHECKPOINTS_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Training config
TRAIN_CONFIG = {
    "num_epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "num_workers": 4,
    "device": "cuda",  # or "cpu"
}

# Model config
MODEL_CONFIG = {
    "spatial_dims": 3,
    "in_channels": 4,  # T1, T1ce, T2, FLAIR
    "out_channels": 4,  # Background, Necrotic, Edema, Enhancing
    "channels": (32, 64, 128, 256, 512),
    "strides": (2, 2, 2, 2),
    "num_res_units": 2,
}

# Data config
DATA_CONFIG = {
    "roi_x": 128,
    "roi_y": 128,
    "roi_z": 128,
    "train_val_split": 0.8,
    "random_seed": 42,
}

# Augmentation config
AUG_CONFIG = {
    "prob": 0.5,
    "rotate_range": (0.2, 0.2, 0.2),
    "scale_range": (0.9, 1.1),
    "flip_prob": 0.5,
    "noise_std": 0.01,
}

# Class labels
BRATS_LABELS = {
    0: "Background",
    1: "Necrotic/Core",
    2: "Edema",
    3: "Enhancing",
}

# BRATS classes for evaluation
BRATS_CLASSES = {
    "WT": [1, 2, 3],  # Whole tumor
    "TC": [1, 3],     # Tumor core
    "ET": [3],        # Enhancing tumor
}
