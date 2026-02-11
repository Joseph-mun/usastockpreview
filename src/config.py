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
LAG_PERIODS = [1, 5]  # reduced from [1,2,3,5,10] - lag2/3/10 contribute minimally
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

# ==================== Target Variable Mode ====================
TARGET_MODE = "excess_return"  # "raw" (original) or "excess_return" (normalize base rate to ~50%)
TARGET_ROLLING_MEDIAN_WINDOW = 252  # 1-year rolling window for median 20d return

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
    "max_depth": 3,              # Phase 4: Optuna-optimized (trial 81, CV accuracy 60.6%)
    "learning_rate": 0.037332,
    "num_leaves": 48,
    "min_child_samples": 41,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 2.228153,
    "reg_lambda": 5.235061,
    "min_split_gain": 0.003108,
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
# Phase 3: Aligned with calibrated probability distribution (~0.40-0.60 range)
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.60,   # matches Aggressive tier entry
    "buy": 0.55,          # matches Growth tier entry
    "neutral": 0.45,      # Defensive boundary
    # below neutral = sell
}

# ==================== TQQQ/SPY Allocation ====================
# Phase 3: Tier boundaries aligned with calibrated prob distribution (~0.40-0.60)
# TQQQ max 25% (reduced from 40% for decay risk mitigation)
ALLOCATION_TIERS = [
    # (min_prob, max_prob, tqqq_weight, spy_weight, cash_weight, label)
    (0.60, 1.01, 0.25, 0.55, 0.20, "Aggressive"),    # top ~10%
    (0.55, 0.60, 0.15, 0.55, 0.30, "Growth"),         # top ~25%
    (0.50, 0.55, 0.05, 0.55, 0.40, "Moderate"),       # middle
    (0.45, 0.50, 0.00, 0.50, 0.50, "Cautious"),       # bottom ~25%
    (0.00, 0.45, 0.00, 0.25, 0.75, "Defensive"),      # bottom ~10%
]

REBALANCE_HYSTERESIS = 0.05  # 5%p move required to trigger rebalance

# ==================== VIX Volatility Filter ====================
# P0-1: Reduce TQQQ exposure during high-volatility regimes
VIX_FILTER_ENABLED = True
VIX_FILTER_TIERS = [
    # (min_vix, max_vix, tqqq_multiplier, label)
    (0,   15,  1.0,  "Low Vol"),
    (15,  25,  0.7,  "Mid Vol"),
    (25,  35,  0.3,  "High Vol"),
    (35, 999,  0.0,  "Extreme Vol"),
]

# ==================== Market Regime Detection ====================
# P1-2: ADX-based regime detection to penalize TQQQ in range-bound markets
REGIME_DETECTION_ENABLED = True
REGIME_ADX_TREND_THRESHOLD = 25    # ADX > 25 = trending market
REGIME_ADX_RANGE_THRESHOLD = 20    # ADX < 20 = range-bound market
REGIME_TQQQ_RANGE_PENALTY = 0.5    # 50% additional TQQQ reduction in range market

# ==================== Transaction Costs ====================
# P0-2: Realistic backtest with slippage and commission
TRANSACTION_COST_ENABLED = True
SLIPPAGE_PCT = 0.0005      # 0.05% one-way slippage
COMMISSION_PCT = 0.0001    # 0.01% one-way commission

# ==================== Probability Clipping ====================
# Phase 3: Narrowed range for calibrated probabilities (prevents extreme allocation)
PROB_CLIP_MIN = 0.20
PROB_CLIP_MAX = 0.80
