# Design: Accuracy Improvement Phase 3

- Feature: `accuracy-improvement-phase3`
- Created: 2026-02-11
- Plan: `docs/01-plan/features/accuracy-improvement-phase3.plan.md`
- Phase 2 Baseline: WF 57.9%, Return -4.04%, Sharpe -0.25, MDD -18.04%

---

## 1. Changes Overview

| ID | Priority | File | Change | Impact |
|----|:--------:|------|--------|--------|
| C-01 | P0 | `src/config.py` | ALLOCATION_TIERS redesign | Tier boundaries → calibrated prob range |
| C-02 | P0 | `src/config.py` | SIGNAL_THRESHOLDS sync | Signal distribution alignment |
| C-03 | P1 | `src/config.py` | PROB_CLIP adjustment | Prevent overconfidence |
| C-04 | P1 | `backtest.py` | VIX/ADX series forwarding | Activate risk filters in backtest |
| C-05 | P1 | `scripts/optuna_optimize.py` | Optuna HP optimization | Model parameter tuning |

---

## 2. Detailed Changes

### C-01: ALLOCATION_TIERS Redesign (P0)

**File**: `src/config.py`, lines 112-120

**Problem**: Calibrated probabilities concentrate in 0.45-0.55 range (Platt compression). Current tier boundaries (80%/70%/60%/50%) are unreachable. Nearly all predictions fall into Cautious/Defensive → excessive cash, negative returns.

**Current**:
```python
ALLOCATION_TIERS = [
    (0.80, 1.01, 0.40, 0.50, 0.10, "Aggressive"),   # unreachable
    (0.70, 0.80, 0.30, 0.50, 0.20, "Growth"),         # unreachable
    (0.60, 0.70, 0.15, 0.55, 0.30, "Moderate"),
    (0.50, 0.60, 0.00, 0.55, 0.45, "Cautious"),
    (0.00, 0.50, 0.00, 0.30, 0.70, "Defensive"),
]
```

**New**:
```python
ALLOCATION_TIERS = [
    # (min_prob, max_prob, tqqq_weight, spy_weight, cash_weight, label)
    (0.60, 1.01, 0.25, 0.55, 0.20, "Aggressive"),    # top ~10%
    (0.55, 0.60, 0.15, 0.55, 0.30, "Growth"),         # top ~25%
    (0.50, 0.55, 0.05, 0.55, 0.40, "Moderate"),       # middle
    (0.45, 0.50, 0.00, 0.50, 0.50, "Cautious"),       # bottom ~25%
    (0.00, 0.45, 0.00, 0.25, 0.75, "Defensive"),      # bottom ~10%
]
```

**Key differences**:
- TQQQ max: 40% → 25% (37.5% reduction)
- Aggressive threshold: 80% → 60% (reachable by calibrated probs)
- 5%p tier width in the 0.45-0.60 range (where 90%+ of predictions fall)
- Comment update: remove "P1-1" reference, add "Phase 3" reference

**Validation**: After backtest, `tier_distribution` should show all 5 tiers with >5% each.

### C-02: SIGNAL_THRESHOLDS Sync (P0)

**File**: `src/config.py`, lines 101-109

**Current**:
```python
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.70,
    "buy": 0.60,
    "neutral": 0.45,
}
```

**New**:
```python
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.60,
    "buy": 0.55,
    "neutral": 0.45,
}
```

**Rationale**: Align signal thresholds with new ALLOCATION_TIERS boundaries. `strong_buy` at 0.60 matches Aggressive tier entry. `buy` at 0.55 matches Growth tier entry. `neutral` stays at 0.45 (Defensive boundary).

**Comment update**: Line 102-103 comment should reflect new calibrated range.

### C-03: PROB_CLIP Adjustment (P1)

**File**: `src/config.py`, lines 148-150

**Current**:
```python
PROB_CLIP_MIN = 0.05
PROB_CLIP_MAX = 0.95
```

**New**:
```python
PROB_CLIP_MIN = 0.20
PROB_CLIP_MAX = 0.80
```

**Rationale**: With calibrated probabilities already compressed, 0.05/0.95 range is too wide — model will never produce extremes, so these clips are no-ops. Narrowing to 0.20/0.80 prevents any uncalibrated raw probability from causing extreme allocations.

### C-04: VIX/ADX Series Forwarding in Backtest (P1)

**File**: `backtest.py`, line 211

**Problem**: `run_portfolio_backtest()` accepts `vix_series` and `adx_series` parameters, but `backtest.py` calls it without them. VIX filter and ADX regime detection are dead code in backtest.

**Current** (line 211):
```python
port_df = run_portfolio_backtest(prob_series, nasdaq_prices, spy_prices)
```

**Change**:
1. Extract VIX series from `spy` DataFrame (column `vix`, already populated by `DatasetBuilder`)
2. Extract ADX series from `spy` DataFrame (column `adx`, already populated by `DatasetBuilder`)
3. Forward-fill NaN values (non-trading day gaps)
4. Pass to `run_portfolio_backtest()`

**New code** (insert between line 210 and 211):
```python
# Extract VIX/ADX for portfolio simulation risk filters
vix_series = spy["vix"].reindex(prob_series.index).ffill() if "vix" in spy.columns else None
adx_series = spy["adx"].reindex(prob_series.index).ffill() if "adx" in spy.columns else None
```

**Updated call** (line 211):
```python
port_df = run_portfolio_backtest(
    prob_series, nasdaq_prices, spy_prices,
    vix_series=vix_series, adx_series=adx_series,
)
```

**Data source verification**:
- `spy["vix"]`: Populated at `features.py:350` via `get_vix_data()` → `fdr.DataReader("VIX")`, ffill applied at `collectors.py:119`
- `spy["adx"]`: Populated at `features.py:420-423` via `calculate_adx(index_df)`, ffill applied

**Import**: No new imports needed. `get_vix_data` and `get_spy_data` already imported.

### C-05: Optuna Hyperparameter Optimization Script (P1)

**File**: `scripts/optuna_optimize.py` (NEW)

**Purpose**: Replace manually-tuned LGBM_PARAMS with Optuna-optimized parameters using Purged TimeSeriesSplit CV.

**Script structure**:
```python
"""Optuna hyperparameter optimization for LightGBM."""
import optuna
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

# 1. Load data (same as backtest.py steps 1-2)
# 2. Define objective function
def objective(trial):
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_split_gain": trial.suggest_float("min_split_gain", 0.001, 0.1, log=True),
        "verbosity": -1,
        "random_state": 42,
    }
    # Purged TimeSeriesSplit CV (same as trainer.py)
    tscv = TimeSeriesSplit(n_splits=5, gap=20)
    scores = []
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) > 20:
            train_idx = train_idx[:-20]  # purge
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        acc = (model.predict(X.iloc[test_idx]) == y.iloc[test_idx]).mean()
        scores.append(acc)
    return np.mean(scores)

# 3. Run optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# 4. Print best params for manual transfer to config.py
print(f"Best CV accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
```

**Output**: Best parameters printed to stdout. Manually transfer to `src/config.py` LGBM_PARAMS.

**Feature selection**: Use `FEATURE_SELECTION_ENABLED=True` with `FEATURE_IMPORTANCE_TOP_N=30` (same as trainer.py Phase 1 logic).

---

## 3. File Change Matrix

| File | Lines Modified | Lines Added | Type |
|------|:-----------:|:----------:|:----:|
| `src/config.py` | ~15 | 0 | MODIFY |
| `backtest.py` | 1 | 4 | MODIFY |
| `scripts/optuna_optimize.py` | 0 | ~80 | NEW |

---

## 4. Dependency Graph

```
C-01 (ALLOCATION_TIERS) ─┐
                          ├→ Backtest run → Metrics comparison
C-02 (SIGNAL_THRESHOLDS) ┘
C-03 (PROB_CLIP) ─────────→ Config only, affects training pipeline
C-04 (VIX/ADX forward) ──→ Backtest run → VIX/ADX filter metrics
C-05 (Optuna) ───────────→ Separate script → Update config → Retrain
```

**Implementation order**: C-01 + C-02 + C-03 → C-04 → Backtest run → C-05 → Final backtest

---

## 5. Unchanged Components

| Component | Reason |
|-----------|--------|
| `src/data/features.py` | Target fix (Phase 2) preserved. No feature changes. |
| `src/models/trainer.py` | Purged CV (Phase 2) preserved. Optuna results applied via config only. |
| `src/strategy/allocation.py` | Logic unchanged. Behavior changes via config values. |
| `src/strategy/portfolio_backtest.py` | Already accepts vix/adx params. No code change needed. |
| `scripts/verify_no_leakage.py` | Phase 2 leakage checks preserved. |

---

## 6. Test Scenarios

| ID | Test | Expected | Verification Method |
|----|------|----------|:-------------------:|
| T-01 | Tier distribution | All 5 tiers > 5% | `port_metrics["tier_distribution"]` |
| T-02 | TQQQ max weight | ≤ 0.25 | `port_df["tqqq_weight"].max()` |
| T-03 | Strategy return | > 0% (target +5~15%) | `port_metrics["total_return"]` |
| T-04 | Sharpe ratio | > 0.3 | `port_metrics["sharpe_ratio"]` |
| T-05 | VIX filter activations | > 0 | `port_metrics["vix_filter_activations"]` |
| T-06 | ADX regime distribution | 3 regimes present | `port_metrics["regime_distribution"]` |
| T-07 | Signal distribution | 4 signals present | backtest signal_stats |
| T-08 | Leakage check | 4/4 PASS | `verify_no_leakage.py` |
| T-09 | MDD | > -15% | `port_metrics["max_drawdown"]` |
| T-10 | Prob clip range | [0.20, 0.80] in config | config.py inspection |
| T-11 | Optuna best CV | ≥ 55% | optuna_optimize.py output |
| T-12 | VIX series not None | vix column in port_df | `port_df["vix"].notna().sum() > 0` |

---

## 7. Rollback

Phase 3 changes are config-level only (except C-04 and C-05). Rollback:
1. Restore ALLOCATION_TIERS to Phase 2 values (plan.md section 7)
2. Restore SIGNAL_THRESHOLDS to `0.70/0.60/0.45`
3. Restore PROB_CLIP to `0.05/0.95`
4. Remove VIX/ADX lines from backtest.py (revert to single-line call)
5. Delete `scripts/optuna_optimize.py`

Phase 2 leakage fixes (`features.py`, `trainer.py`) are **never** rolled back.

---

## 8. Academic References

| Change | Reference |
|--------|-----------|
| TQQQ weight reduction | Cheng & Madhavan (2009): Leveraged ETF decay risk |
| Allocation boundary optimization | De Miguel, Garlappi & Uppal (2009): Optimal vs Naive Diversification |
| Optuna framework | Akiba et al. (2019): Next-gen Hyperparameter Optimization |
| Purged CV (maintained) | Lopez de Prado (2018): Advances in Financial ML |
