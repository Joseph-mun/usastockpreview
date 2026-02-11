# accuracy-improvement Analysis Report

> **Analysis Type**: Gap Analysis (Design vs Implementation)
>
> **Project**: us-market-predictor
> **Analyst**: gap-detector
> **Date**: 2026-02-11
> **Design Doc**: [accuracy-improvement.design.md](../02-design/features/accuracy-improvement.design.md)

---

## Match Rate: 97%

```
+---------------------------------------------+
|  Overall Match Rate: 97%                     |
+---------------------------------------------+
|  Total check items:      25                  |
|  Match:                  24 items (96%)      |
|  Minor deviation:         1 item  (4%)       |
|  Not implemented:         0 items (0%)       |
+---------------------------------------------+
```

---

## 1. Analysis Overview

### 1.1 Analysis Purpose

Verify that all Phase 1 (P0) changes specified in the design document for accuracy improvement have been correctly implemented in the codebase.

### 1.2 Analysis Scope

- **Design Document**: `docs/02-design/features/accuracy-improvement.design.md`
- **Implementation Files**:
  - `src/config.py`
  - `src/data/features.py`
  - `backtest.py`
- **Analysis Date**: 2026-02-11

---

## 2. Gap Analysis (Design vs Implementation)

### 2.1 config.py - Change A: LGBM_PARAMS

| Parameter | Design Value | Implementation Value (line) | Status |
|-----------|:------------:|:--------------------------:|:------:|
| max_depth | 5 | 5 (line 71) | Match |
| num_leaves | 31 | 31 (line 73) | Match |
| min_child_samples | 30 | 30 (line 74) | Match |
| reg_alpha | 1.0 | 1.0 (line 77) | Match |
| reg_lambda | 2.0 | 2.0 (line 78) | Match |
| min_split_gain | 0.01 | 0.01 (line 79) | Match |
| objective | "binary" | "binary" (line 68) | Match |
| boosting_type | "gbdt" | "gbdt" (line 69) | Match |
| n_estimators | 500 | 500 (line 70) | Match |
| learning_rate | 0.05 | 0.05 (line 72) | Match |
| subsample | 0.8 | 0.8 (line 75) | Match |
| colsample_bytree | 0.8 | 0.8 (line 76) | Match |
| random_state | 42 | 42 (line 80) | Match |
| verbosity | -1 | -1 (line 81) | Match |
| n_jobs | -1 | -1 (line 82) | Match |

**Result: 15/15 parameters match. PASS.**

### 2.2 config.py - Change B: LAG_PERIODS

| Item | Design | Implementation (line) | Status |
|------|:------:|:--------------------:|:------:|
| LAG_PERIODS | [1, 5] | [1, 5] (line 33) | Match |

**Result: PASS.**

### 2.3 config.py - Change C: Target Variable Config

| Item | Design | Implementation (line) | Status |
|------|:------:|:--------------------:|:------:|
| TARGET_MODE | "excess_return" | "excess_return" (line 55) | Match |
| TARGET_ROLLING_MEDIAN_WINDOW | 252 | 252 (line 56) | Match |
| Section comment | "Target Variable Mode" | "Target Variable Mode" (line 54) | Match |

**Result: PASS.**

### 2.4 features.py - build_target() Function

| Item | Design | Implementation | Status |
|------|--------|----------------|:------:|
| Function signature | `build_target(df: pd.DataFrame) -> pd.DataFrame` | Same (line 222) | Match |
| Reverse-rolling trick for after2/after2_low | Present | Present (lines 234-238) | Match |
| `spy["after"]` = close.shift(-TARGET_LOOKAHEAD_DAYS) | Present | Present (line 241) | Match |
| `spy["suik_rate"]` calculation | Present | Present (line 244) | Match |
| `TARGET_MODE == "excess_return"` branch | Present | Present (line 247) | Match |
| `ret_20d = close.pct_change(TARGET_LOOKAHEAD_DAYS)` | Present | Present (line 249) | Match |
| `rolling_med = ret_20d.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()` | Present | Present (line 250) | Match |
| Shift median by lookahead (leakage prevention) | `rolling_med.shift(TARGET_LOOKAHEAD_DAYS)` | `rolling_med.shift(TARGET_LOOKAHEAD_DAYS)` (line 252) | Match |
| Target assignment: `np.where(ret_20d > shifted_median, 1, 0)` | Present | Present (line 253) | Match |
| `_rolling_median` column dropped | `spy["_rolling_median"] = ...` then `spy.drop(columns=["_rolling_median"])` | Uses local variable `shifted_median` instead of storing in DataFrame | Minor deviation |
| `else` branch (raw mode) | Present | Present (lines 254-258) | Match |
| `TargetDown` preserved | Present | Present (lines 260-262) | Match |

**Minor deviation detail**: The design specifies storing the rolling median in `spy["_rolling_median"]` and then calling `spy.drop(columns=["_rolling_median"], inplace=True)`. The implementation instead uses a local variable `shifted_median = rolling_med.shift(TARGET_LOOKAHEAD_DAYS)` and references it directly in `np.where()`. This is functionally equivalent and arguably cleaner -- the `_rolling_median` column never touches the DataFrame at all, so there is zero leakage risk. No action needed.

**Result: 11/12 exact match, 1 minor deviation (functionally equivalent). PASS.**

### 2.5 features.py - Import Update

| Item | Design | Implementation (line) | Status |
|------|--------|:--------------------:|:------:|
| `TARGET_MODE` in imports | Present | Present (line 13) | Match |
| `TARGET_ROLLING_MEDIAN_WINDOW` in imports | Present | Present (line 13) | Match |

**Result: PASS.**

### 2.6 backtest.py - Expanding Walk-Forward

| Item | Design | Implementation (line) | Status |
|------|--------|:--------------------:|:------:|
| Function signature includes `retrain_freq` param | `run_backtest(backtest_days, retrain_freq, verbose)` | `run_backtest(backtest_days=252, retrain_freq=60, verbose=True)` (line 32) | Match |
| `eval_days = min(backtest_days, len(X) - 500)` | Present | Present (line 70) | Match |
| `retrain_freq = 60` default | Present | Present (line 32) | Match |
| `all_predictions = []` | Present | Present (line 75) | Match |
| `for wf_start in range(...)` expanding loop | `range(len(X) - eval_days, len(X), retrain_freq)` | `range(wf_origin, len(X), retrain_freq)` where `wf_origin = len(X) - eval_days` (lines 71, 80) | Match |
| `X_train_wf = X.iloc[:wf_start]` (expanding) | Present | Present (line 82) | Match |
| `X_test_wf = X.iloc[wf_start:wf_end]` | Present | Present (line 84) | Match |
| Min train size guard: `len(X_train_wf) < 500` | Present | Present (line 87) | Match |
| ModelTrainer usage per window | Present | Present (lines 93-102) | Match |
| Calibration handling (isotonic/platt) | Present | Present (lines 107-113) | Match |
| Prediction accumulation to `all_predictions` list | Present | Present (lines 122-128) | Match |
| Window logging with accuracy | Present | Present (lines 130-131) | Match |
| Default `backtest_days` = 252 | Present | Present (line 32) | Match |
| `argparse --days default=252` | Present | Present (line 331) | Match |
| `argparse --retrain-freq` | Present | Present (line 332) | Match |

**Result: 15/15 items match. PASS.**

---

## 3. Test Scenario Verification (T-01 through T-10)

| # | Test | Design Expectation | Code Support | Status |
|---|------|-------------------|:------------:|:------:|
| T-01 | Train with new LGBM_PARAMS | Model trains without error | LGBM_PARAMS correctly defined in config.py (lines 67-83). ModelTrainer imports from config. | Supported |
| T-02 | CV accuracy with regularization | Lower than 66.5% | reg_alpha=1.0, reg_lambda=2.0, min_split_gain=0.01 all present. More conservative splits (max_depth=5, num_leaves=31, min_child_samples=30). | Supported |
| T-03 | Target base rate with excess_return | ~50% | `TARGET_MODE = "excess_return"` active. Rolling median split by definition produces ~50% base rate. | Supported |
| T-04 | Target base rate with raw mode | ~67% | `else` branch in build_target() preserves original threshold logic (lines 254-258). | Supported |
| T-05 | Expanding WF with 252 days | Multiple windows (n_windows >= 4) | 252 eval days / 60 retrain_freq = 4.2 windows minimum. Loop in lines 80-131. | Supported |
| T-06 | WF accuracy 55-60% (180d+) | Print accuracy | Per-window accuracy logged (line 131). Overall accuracy computed (lines 179-181, 229). | Supported |
| T-07 | Feature count ~60 after LAG reduction | `len(X.columns)` | LAG_PERIODS=[1,5] (line 33 config.py). 5 indicators x 2 lags = 10 lag features. Feature count logged (line 63 backtest.py). | Supported |
| T-08 | `_rolling_median` not in X columns | Check LEAK_COLUMNS | Implementation never adds `_rolling_median` to DataFrame (uses local variable). LEAK_COLUMNS set (config.py lines 37-41) also prevents leakage of Target/after/suik_rate. | Supported |
| T-09 | Portfolio simulation runs on expanded WF | Strategy return computed | Portfolio simulation in backtest.py lines 202-213 uses `prob_series` built from `all_predictions`. | Supported |
| T-10 | HTML report generated | File created | `generate_html_report()` called in main() (line 341). Same table schema, just more rows from expanded WF. | Supported |

**Result: 10/10 test scenarios structurally supported by implementation. PASS.**

---

## 4. Backward Compatibility Verification

| Item | Design Claim | Implementation Check | Status |
|------|-------------|---------------------|:------:|
| `TARGET_MODE = "raw"` reverts to original | `else` branch in build_target() | Lines 254-258: original threshold logic preserved | Match |
| `backtest.py --days 30` still works | Short WF supported | `eval_days = min(30, len(X)-500)` would work; loop still iterates | Match |
| `training_pipeline.py` no changes needed | Uses same config | Not modified in this feature (confirmed by scope) | Match |
| `daily_pipeline.py` no changes needed | No changes | Not modified | Match |
| `allocation.py` no changes needed | Probability input unchanged | `get_allocation(prob)` call preserved (backtest.py line 160) | Match |
| `portfolio_backtest.py` no changes needed | No changes | Not modified | Match |

**Result: 6/6 backward compatibility items confirmed. PASS.**

---

## 5. Data Flow Verification

| Flow Step | Design | Implementation | Status |
|-----------|--------|----------------|:------:|
| config.py provides LGBM_PARAMS, LAG_PERIODS, TARGET_MODE | Correct | config.py exports all three | Match |
| features.py::build_target() uses TARGET_MODE | Correct | Line 247 checks TARGET_MODE | Match |
| excess_return -> rolling median -> Target ~50% | Correct | Lines 249-253 | Match |
| raw -> original threshold -> Target ~67% | Correct | Lines 256-258 | Match |
| LAG_PERIODS=[1,5] -> ~60 features | Correct | Line 33 config.py, used by calculate_lag_features() | Match |
| backtest.py uses expanding WF, retrain_freq=60 | Correct | Lines 70-131 | Match |
| trainer.py uses max_depth=5, num_leaves=31, etc. | Correct | ModelTrainer imports LGBM_PARAMS from config | Match |

**Result: 7/7 data flow steps verified. PASS.**

---

## 6. Overall Scores

| Category | Items Checked | Match | Score | Status |
|----------|:------------:|:-----:|:-----:|:------:|
| config.py Change A (LGBM_PARAMS) | 15 | 15 | 100% | Match |
| config.py Change B (LAG_PERIODS) | 1 | 1 | 100% | Match |
| config.py Change C (TARGET_MODE) | 3 | 3 | 100% | Match |
| features.py build_target() | 12 | 11 | 92% | Minor deviation |
| features.py imports | 2 | 2 | 100% | Match |
| backtest.py expanding WF | 15 | 15 | 100% | Match |
| Test scenarios (T-01 to T-10) | 10 | 10 | 100% | Match |
| Backward compatibility | 6 | 6 | 100% | Match |
| Data flow | 7 | 7 | 100% | Match |
| **Total** | **71** | **70** | **97%** | **Match** |

---

## 7. Differences Found

### Minor Deviations (Design != Implementation, functionally equivalent)

| Item | Design | Implementation | Impact |
|------|--------|----------------|:------:|
| `_rolling_median` handling in build_target() | Store in `spy["_rolling_median"]`, then `spy.drop(columns=["_rolling_median"])` | Uses local variable `shifted_median` (never touches DataFrame) | None (better) |

**Assessment**: The implementation is arguably superior to the design. By using a local variable instead of temporarily adding and removing a DataFrame column, it avoids any risk of the column persisting due to an error mid-execution. No action required.

### Missing Features (Design present, Implementation absent)

None.

### Added Features (Design absent, Implementation present)

None.

---

## 8. Recommended Actions

### No immediate actions required.

Match rate is 97% (above 90% threshold). The single deviation is functionally superior to the design.

### Documentation Update (Optional)

1. **Update design doc line 130-134**: Change the `_rolling_median` pattern to reflect the actual local-variable approach, for documentation accuracy.

---

## 9. Conclusion

All 4 Phase 1 (P0) items from the design document are fully implemented:

1. **LGBM_PARAMS regularization** -- all 6 parameter changes applied correctly
2. **LAG_PERIODS reduction** -- [1, 5] as specified
3. **Target variable mode (excess_return)** -- complete with leakage prevention
4. **Expanding walk-forward backtest** -- with retrain_freq=60, 252-day default, argparse support

All 10 test scenarios (T-01 through T-10) are structurally supported by the implementation. All 6 backward compatibility guarantees hold. The data flow matches the design specification.

**Match Rate: 97%. No action needed. Ready for Report phase.**

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-11 | Initial gap analysis | gap-detector |
