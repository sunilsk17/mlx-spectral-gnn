"""
Configuration for Hardware-Aware Spectral Sparsification experiments.
All hyperparameters and paths are centralized here.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Dataset paths
PLANETOID_DATA_DIR = DATA_DIR

# ─── Spectral Pruning ───────────────────────────────────────────────
SPECTRAL_K = 10            # Number of eigenvectors for spectral embedding
PRUNE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]

# ─── GCN Model ───────────────────────────────────────────────────────
HIDDEN_DIM = 64
DROPOUT = 0.5

# ─── Training ────────────────────────────────────────────────────────
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
SEEDS = [42, 123, 456, 789, 1024]

# ─── Datasets ────────────────────────────────────────────────────────
DATASETS = ["cora", "citeseer"]

# ─── Ensure output dirs exist ────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
