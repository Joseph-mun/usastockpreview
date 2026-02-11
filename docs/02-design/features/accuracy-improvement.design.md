# Design: NASDAQ 20-Day Prediction Accuracy Improvement

- Feature: `accuracy-improvement`
- Phase: Design (Phase 1 only - P0 items)
- Created: 2026-02-11
- Plan ref: `docs/01-plan/features/accuracy-improvement.plan.md`

---

## Scope

Phase 1 (P0) 4개 항목만 구현. Phase 2/3는 Phase 1 결과 확인 후 별도 PDCA.

---

## 1. File Changes

### 1-1. `src/config.py` - Model Parameters + Target Config

#### Change A: LGBM_PARAMS regularization + complexity reduction

**Before** (line 63-76):
```python
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
```

**After**:
```python
LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "max_depth": 5,              # 8 -> 5 (reduce overfitting)
    "learning_rate": 0.05,
    "num_leaves": 31,            # 63 -> 31 (reduce overfitting)
    "min_child_samples": 30,     # 20 -> 30 (more conservative splits)
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,            # NEW: L1 regularization
    "reg_lambda": 2.0,           # NEW: L2 regularization
    "min_split_gain": 0.01,      # NEW: minimum split gain
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}
```

**Rationale**: 2715 samples with max_depth=8/num_leaves=63 is over-complex. reg_alpha/reg_lambda=0 (default) means no regularization at all.

#### Change B: LAG_PERIODS reduction

**Before** (line 33):
```python
LAG_PERIODS = [1, 2, 3, 5, 10]
```

**After**:
```python
LAG_PERIODS = [1, 5]
```

**Rationale**: 5 indicators x 5 lags = 25 lag features -> 5 x 2 = 10. Feature importance shows lag2/3/10 are noise.

#### Change C: Target variable config (NEW constants)

Add after line 52:
```python
# ==================== Target Variable Mode ====================
TARGET_MODE = "excess_return"  # "raw" (original) or "excess_return" (new)
TARGET_ROLLING_MEDIAN_WINDOW = 252  # 1-year rolling window for median 20d return
```

---

### 1-2. `src/data/features.py` - Target Variable Fix

#### Change: `build_target()` function (line 221-249)

**Before**:
```python
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    spy = df.copy()
    close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]
    # ... reverse rolling for after2/after2_low ...
    spy["after"] = close.shift(-TARGET_LOOKAHEAD_DAYS)
    spy["Target"] = np.where(spy["after"] > TARGET_UP_THRESHOLD * close, 1, 0)
    spy["TargetDown"] = np.where(spy["after2_low"] < TARGET_DOWN_THRESHOLD * close, 1, 0)
    spy["suik_rate"] = 100 * (spy["after"] - close) / close
    return spy
```

**After**:
```python
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    spy = df.copy()
    close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]

    # Future max/min over lookahead window (reverse-rolling trick)
    spy_rev = spy.sort_index(ascending=False)
    close_rev = close.sort_index(ascending=False)
    spy_rev["after2"] = close_rev.rolling(window=TARGET_LOOKAHEAD_DAYS).max()
    spy_rev["after2_low"] = close_rev.rolling(window=TARGET_LOOKAHEAD_DAYS).min()
    spy = spy_rev.sort_index(ascending=True)

    # After price (unified to lookahead days)
    spy["after"] = close.shift(-TARGET_LOOKAHEAD_DAYS)

    # Return rate
    spy["suik_rate"] = 100 * (spy["after"] - close) / close

    # Binary targets
    if TARGET_MODE == "excess_return":
        # Excess return: compare 20d return against rolling median
        ret_20d = close.pct_change(TARGET_LOOKAHEAD_DAYS)
        rolling_med = ret_20d.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
        # Shift median by lookahead to prevent leakage
        spy["_rolling_median"] = rolling_med.shift(TARGET_LOOKAHEAD_DAYS)
        spy["Target"] = np.where(
            ret_20d > spy["_rolling_median"], 1, 0
        )
        spy.drop(columns=["_rolling_median"], inplace=True)
    else:
        # Original raw target
        spy["Target"] = np.where(
            spy["after"] > TARGET_UP_THRESHOLD * close, 1, 0
        )

    spy["TargetDown"] = np.where(
        spy["after2_low"] < TARGET_DOWN_THRESHOLD * close, 1, 0
    )

    return spy
```

**Import addition** (line 9): Add `TARGET_MODE, TARGET_ROLLING_MEDIAN_WINDOW` to config imports.

```python
from src.config import (
    DATA_START_DATE, MA_WINDOWS, LAG_PERIODS, ROLLING_WINDOWS,
    TARGET_LOOKAHEAD_DAYS, TARGET_UP_THRESHOLD, TARGET_DOWN_THRESHOLD,
    LEAK_COLUMNS, BASE_PRICE_COLUMNS, TRAIN_WINDOW_YEARS,
    TARGET_MODE, TARGET_ROLLING_MEDIAN_WINDOW,  # NEW
)
```

**Key design decisions**:
- `rolling_med.shift(TARGET_LOOKAHEAD_DAYS)`: prevents future information leakage
- When `TARGET_MODE == "raw"`: preserves backward compatibility
- Base rate normalizes to ~50% (verified: median split by definition)
- `_rolling_median` is dropped immediately (not leaked to features)

---

### 1-3. `backtest.py` - Expanding Walk-Forward

#### Change: Replace single-split with expanding window WF

**Replace `run_backtest()` function** (line 32-186).

Core logic change (line 62-89):

**Before**:
```python
# 3. Walk-forward split
n_days = min(backtest_days, len(X) - 100)
split_idx = len(X) - n_days
X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
# ... single train + single predict block ...
```

**After**:
```python
# 3. Expanding Walk-Forward
eval_days = min(backtest_days, len(X) - 500)  # need 500+ for training
retrain_freq = 60  # retrain every 60 trading days (~3 months)

all_predictions = []
n_windows = 0

for wf_start in range(len(X) - eval_days, len(X), retrain_freq):
    wf_end = min(wf_start + retrain_freq, len(X))
    X_train_wf = X.iloc[:wf_start]
    y_train_wf = y.iloc[:wf_start]
    X_test_wf = X.iloc[wf_start:wf_end]
    y_test_wf = y.iloc[wf_start:wf_end]

    if len(X_train_wf) < 500 or len(X_test_wf) == 0:
        continue

    trainer = ModelTrainer(index_name)
    result = trainer.train(X_train_wf, y_train_wf,
                           status_callback=lambda msg: log(f"   {msg}") if verbose else None)

    feature_cols = trainer.feature_columns
    calibrator = trainer.calibrator
    cal_method = trainer.calibration_method
    model = result["model"]

    X_test_aligned = X_test_wf.reindex(columns=feature_cols).fillna(0)
    raw_probs = model.predict_proba(X_test_aligned.values)[:, 1]

    if calibrator is not None:
        if cal_method == "isotonic":
            cal_probs = calibrator.predict(raw_probs)
        else:
            cal_probs = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
    else:
        cal_probs = raw_probs

    for i, date in enumerate(X_test_wf.index):
        all_predictions.append({
            "date": date,
            "prob": float(cal_probs[i]),
            "actual": int(y_test_wf.iloc[i]),
            "window": n_windows,
        })

    n_windows += 1
    log(f"   Window {n_windows}: train={len(X_train_wf)}, test={len(X_test_wf)}, "
        f"acc={sum(1 for p in all_predictions[-len(X_test_wf):] if (p['prob']>=0.5)==p['actual'])/len(X_test_wf):.1%}")
```

**Downstream changes**:
- `prob_series` built from `all_predictions` list
- `df_results` built from `all_predictions` (same schema as before)
- Portfolio simulation uses full prediction period
- Default `backtest_days` changed from 30 to 252 (1 year)
- `argparse` default: `parser.add_argument("--days", type=int, default=252)`

**HTML report**: No schema change needed. Same table structure, just more rows.

---

## 2. Backward Compatibility

| Item | Compatibility |
|------|---------------|
| `TARGET_MODE = "raw"` | Reverts to original behavior |
| `backtest.py --days 30` | Still works (short WF) |
| `training_pipeline.py` | No changes needed (uses same config) |
| `daily_pipeline.py` | No changes needed |
| `allocation.py` | No changes needed (probability input unchanged) |
| `portfolio_backtest.py` | No changes needed |

---

## 3. Test Scenarios

| # | Test | Expected Result | Verification |
|---|------|-----------------|--------------|
| T-01 | Train with new LGBM_PARAMS | Model trains without error | No crash |
| T-02 | CV accuracy with regularization | Lower than 66.5% (closer to reality) | Print CV scores |
| T-03 | Target base rate with excess_return mode | ~50% (not 67%) | `y.mean() ~ 0.50` |
| T-04 | Target base rate with raw mode | ~67% (unchanged) | `y.mean() ~ 0.67` |
| T-05 | Expanding WF with 252 days | Multiple windows trained | `n_windows >= 4` |
| T-06 | WF accuracy with all Phase 1 changes | 55-60% (180d+ eval) | Print accuracy |
| T-07 | Feature count after LAG_PERIODS reduction | ~60 (down from 96) | `len(X.columns)` |
| T-08 | Rolling median no leakage | `_rolling_median` not in X columns | Check LEAK_COLUMNS |
| T-09 | Portfolio simulation runs on expanded WF | Strategy return computed | No crash |
| T-10 | HTML report generated for expanded WF | File created | File exists |

---

## 4. Data Flow

```
config.py (LGBM_PARAMS, LAG_PERIODS, TARGET_MODE)
    |
    v
features.py::build_target()
    |-- TARGET_MODE == "excess_return" --> rolling median --> Target ~50% base rate
    |-- TARGET_MODE == "raw"           --> original       --> Target ~67% base rate
    |
    v
features.py::DatasetBuilder.build()
    |-- LAG_PERIODS=[1,5] --> ~60 features (was 96)
    |
    v
backtest.py::run_backtest()
    |-- expanding WF, retrain_freq=60
    |-- multiple windows, 252 days eval
    |
    v
trainer.py::ModelTrainer.train()
    |-- max_depth=5, num_leaves=31
    |-- reg_alpha=1.0, reg_lambda=2.0
    |
    v
Results: WF accuracy on 252 days, multiple retrain windows
```

---

## 5. Rollback Strategy

All changes are config-driven. To rollback:

```python
# config.py
LGBM_PARAMS["max_depth"] = 8
LGBM_PARAMS["num_leaves"] = 63
# remove reg_alpha, reg_lambda, min_split_gain

LAG_PERIODS = [1, 2, 3, 5, 10]
TARGET_MODE = "raw"
```

Backtest: `python backtest.py --days 30` reverts to original behavior.

---

## 6. Implementation Order

1. `src/config.py` - LGBM_PARAMS + LAG_PERIODS + TARGET_MODE (all config changes)
2. `src/data/features.py` - build_target() + import update
3. `backtest.py` - expanding WF rewrite
4. Run training: `python -m src.pipelines.training_pipeline`
5. Run backtest: `python backtest.py --days 252`
6. Compare results with baseline
