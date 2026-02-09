# -*- coding: utf-8 -*-
"""Global configuration for US Market Predictor."""

import os
from pathlib import Path

# ==================== Paths ====================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
SMA_CACHE_DIR = DATA_DIR / "sma_cache"
LOG_DIR = DATA_DIR / "logs"

# ==================== Index Definitions ====================
INDEX_CONFIGS = {
    "NASDAQ": {
        "ticker": "IXIC",
        "fdr_candidates": ["IXIC", "^IXIC"],
        "display_name": "나스닥",
        "emoji": "\U0001f7e2",
    },
}

# ==================== Data Collection ====================
SMA_WINDOWS = [15, 30, 50]
DATA_START_DATE = "2015-01-01"
BOND_CODES = ["DGS2", "DGS10", "DGS30"]
BOND_CHANGE_PERIODS = [5, 20, 60]
VIX_CHANGE_PERIODS = [1, 5, 10]
MA_WINDOWS = [5, 20, 60, 120, 200]

# ==================== Feature Engineering ====================
LAG_PERIODS = [1, 2, 3, 5, 10]
ROLLING_WINDOWS = [5, 10, 20, 60]

# Columns to exclude from features (leakage prevention)
LEAK_COLUMNS = {
    "Target", "TargetDown",
    "after", "after2", "after2_low",
    "suik_rate",
}
BASE_PRICE_COLUMNS = {
    "Open", "High", "Low", "Close", "Adj Close", "Volume", "Change",
}

# ==================== Training Window ====================
TRAIN_WINDOW_YEARS = None  # Use all available data (restored from 3)

# ==================== Target Variable ====================
TARGET_LOOKAHEAD_DAYS = 20   # 20-day lookahead (restored from 5)
TARGET_UP_THRESHOLD = 1.0    # any positive return (price went up)
TARGET_DOWN_THRESHOLD = 1.0  # any negative return (price went down)

# ==================== Calibration ====================
CALIBRATION_ENABLED = True
CALIBRATION_METHOD = "platt"  # "platt" (smooth sigmoid) or "isotonic" (step function)

# ==================== Feature Selection ====================
FEATURE_SELECTION_ENABLED = True
FEATURE_IMPORTANCE_TOP_N = 30

# ==================== Model ====================
LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}

CV_N_SPLITS = 5   # restored from 3 (full data = ~2500+ samples)
CV_GAP = TARGET_LOOKAHEAD_DAYS  # gap between train/test to prevent leakage

# ==================== Meta Learner (Incremental Deep Learning) ====================
META_LEARNER_ENABLED = True
META_LEARNER_HIDDEN = (32, 16)
META_LEARNER_LR = 0.001

# ==================== GitHub Release ====================
MODEL_RELEASE_TAG = "model-latest"
SMA_CACHE_RELEASE_TAG = "sma-cache-latest"

# ==================== Telegram ====================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ==================== Signal Thresholds ====================
# Base rate ~67% (market goes up ~67% of 20-day periods)
# Thresholds set relative to base rate for meaningful signal distribution
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.80,   # significantly above base rate
    "buy": 0.70,          # above base rate
    "neutral": 0.55,      # around/below base rate
    # below neutral = sell
}

# ==================== TQQQ/SPY Allocation ====================
ALLOCATION_TIERS = [
    # (min_prob, max_prob, tqqq_weight, spy_weight, cash_weight, label)
    (0.80, 1.01, 0.60, 0.40, 0.00, "Aggressive"),
    (0.70, 0.80, 0.40, 0.50, 0.10, "Growth"),
    (0.60, 0.70, 0.20, 0.60, 0.20, "Moderate"),
    (0.50, 0.60, 0.00, 0.60, 0.40, "Cautious"),
    (0.00, 0.50, 0.00, 0.30, 0.70, "Defensive"),
]

REBALANCE_HYSTERESIS = 0.05  # 5%p move required to trigger rebalance

# ==================== Probability Clipping ====================
PROB_CLIP_MIN = 0.05  # minimum probability (prevents 0% overconfidence)
PROB_CLIP_MAX = 0.95  # maximum probability (prevents 100% overconfidence)
