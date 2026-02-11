# Design: Accuracy Improvement Phase 4

- Feature: `accuracy-improvement-phase4`
- Created: 2026-02-11
- Plan: `docs/01-plan/features/accuracy-improvement-phase4.plan.md`
- Phase 3 Baseline: WF 58.3%, Return +2.23%, Sharpe 0.26, Strong Buy actual up 16.7%

---

## 1. Changes Overview

| ID | Priority | File | Change | Impact |
|----|:--------:|------|--------|--------|
| C-01 | P0 | `scripts/diagnose_signal_inversion.py` | Signal inversion diagnosis | Root cause identification |
| C-02 | P0 | `scripts/optuna_optimize.py` | Execute Optuna (existing) | HP optimization |
| C-03 | P0 | `src/config.py` | Apply Optuna results | LGBM_PARAMS update |
| C-04 | P1 | `backtest.py` | Record raw_probs alongside cal_probs | Diagnostic data collection |
| C-05 | P1 | Conditional (based on C-01 results) | Signal inversion fix | Core anomaly resolution |

---

## 2. Detailed Changes

### C-01: Signal Inversion Diagnosis Script (P0)

**File**: `scripts/diagnose_signal_inversion.py` (NEW)

**Purpose**: Identify root cause of inverted signal accuracy (Strong Buy 16.7% vs Sell 42.9%).

**Architecture**: Replicate backtest walk-forward loop, but collect extended diagnostics at each prediction.

**Script structure**:

```python
"""Diagnose signal inversion: why high-confidence predictions are inverted."""
# Step 1: Replicate WF loop from backtest.py (lines 80-131)
# Step 2: For each prediction, record:
#   - date, window, raw_prob, cal_prob, actual, signal
#   - Top 5 feature values at that date
#   - rolling_median value at that date
#   - momentum_20d and zscore_20d at that date
# Step 3: Analyze by signal group

# === CHECK 1: Window-Signal Distribution ===
# For each WF window, show signal breakdown + actual up rate
# Answer: Is inversion concentrated in specific windows?

# === CHECK 2: Raw vs Calibrated Probability ===
# Compare raw_probs for Strong Buy subset vs overall
# Answer: Does calibration create or preserve inversion?

# === CHECK 3: Momentum-Target Correlation at Signal Extremes ===
# For Strong Buy predictions (cal_prob >= 0.60):
#   - Show mean momentum_20d, roc_20d, Change20day
#   - Show mean rolling_median at those dates
#   - Show mean (future_ret - rolling_median) gap
# Answer: Is momentum driving high prob into mean-reversion regime?

# === CHECK 4: Temporal Clustering ===
# Are Strong Buy/Buy dates clustered in specific periods?
# Show date ranges for each signal

# === CHECK 5: Feature Importance at Extremes ===
# For dates where cal_prob >= 0.60 vs cal_prob <= 0.45:
#   - Compare mean feature values
#   - Identify which features differentiate extreme predictions

# === VERDICT ===
# Print which hypothesis (H1-H4) is supported
```

**Data flow**:
1. Load dataset via `DatasetBuilder.build()` (same as backtest.py)
2. Run WF loop with `ModelTrainer.train()` per window
3. At each prediction, access:
   - `model.predict_proba()` → `raw_probs`
   - `trainer.calibrator` → `cal_probs`
   - `spy` DataFrame → `rolling_median`, momentum features
4. Aggregate results into `pd.DataFrame`
5. Run 5 checks, print results

**Key fields to extract from spy DataFrame**:
- `momentum_20d` = `close / close.shift(20) - 1` (from `calculate_momentum`)
- `zscore_20d` = `(close - ma20) / std20` (from `calculate_mean_reversion`)
- `roc_20d` = `100 * (close - close.shift(20)) / close.shift(20)` (from `calculate_roc`)
- `Change20day` = log-cumulative 20d return
- `rolling_median` = computed from target logic in `build_target()`

**Access to rolling_median**: Must recompute from spy data since it's not stored as a column:
```python
past_ret = close.pct_change(TARGET_LOOKAHEAD_DAYS)
rolling_med = past_ret.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
```

**Output format**: Plain text with section headers, one VERDICT line at the end.

**Expected output**:
```
=== CHECK 1: Window-Signal Distribution ===
Window 1: SB=2(0% up), B=5(20% up), N=30(53% up), S=23(43% up)
...

=== CHECK 2: Raw vs Calibrated ===
Strong Buy: raw_mean=0.58, cal_mean=0.61 (calibration amplifies)
Overall: raw_mean=0.50, cal_mean=0.50

=== CHECK 3: Momentum-Target at Extremes ===
Strong Buy dates:
  momentum_20d mean: +8.5% (vs overall +2.1%)
  rolling_median mean: +6.2% (vs overall +3.1%)
  future_ret - rolling_med gap: -2.3%
→ H1 CONFIRMED: High momentum pushes rolling_median up, making excess return harder

=== VERDICT ===
Primary cause: H1 (Momentum-Target Mismatch) [CONFIRMED/REJECTED]
Secondary: H3 (Small Sample) [CONFIRMED/REJECTED]
```

### C-02: Execute Optuna Optimization (P0)

**File**: `scripts/optuna_optimize.py` (EXISTING, created in Phase 3)

**Action**: Execute the script, no code changes needed.

```bash
python scripts/optuna_optimize.py --trials 100
```

**Expected output**: Best parameters with CV accuracy ≥ 55%.

### C-03: Apply Optuna Results to Config (P0)

**File**: `src/config.py`, lines 67-83

**Current**:
```python
LGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "min_split_gain": 0.01,
    "random_state": 42,
    "verbosity": -1,
    "n_jobs": -1,
}
```

**New**: Replace tunable parameters (max_depth, num_leaves, min_child_samples, learning_rate, reg_alpha, reg_lambda, min_split_gain) with Optuna best values. Keep fixed parameters (objective, boosting_type, n_estimators, subsample, colsample_bytree, random_state, verbosity, n_jobs) unchanged.

**Comment update**: Add `# Phase 4: Optuna-optimized (trial N, CV accuracy X.X%)` comment.

### C-04: Record Raw Probabilities in Backtest (P1)

**File**: `backtest.py`, lines 122-128

**Current**: Only `cal_probs` (post-calibration) is recorded.

**Change**: Add `raw_prob` to the all_predictions dict for future diagnostic access.

**Current** (line 122-128):
```python
all_predictions.append({
    "date": date,
    "prob": prob,
    "actual": actual,
    "predicted": predicted,
    "window": n_windows,
})
```

**New**:
```python
all_predictions.append({
    "date": date,
    "prob": prob,
    "raw_prob": float(raw_probs[i]),
    "actual": actual,
    "predicted": predicted,
    "window": n_windows,
})
```

**Additional**: Add `raw_prob` column to `df_results` construction (line 135-141 area) and to signal_stats computation.

### C-05: Conditional Signal Inversion Fix (P1)

**Depends on**: C-01 diagnosis results.

**If H1 confirmed (Momentum-Target Mismatch)** — most likely scenario:

The fix is **NOT** to change the target mode. The `excess_return` target is correct (normalized ~50% base rate, no leakage). Instead, the fix is to help the model learn the **non-linear relationship** between momentum and excess returns.

**Option A: Add momentum-excess interaction features** (preferred):
```python
# In features.py, section 17 or new section:
# Interaction: momentum relative to its own rolling percentile
# High momentum + high percentile → mean-reversion risk
for p in [10, 20]:
    mom_col = f"momentum_{p}d"
    if mom_col in spy.columns:
        pctile = spy[mom_col].rolling(252).rank(pct=True)
        spy[f"mom{p}_pctile"] = pctile
```

This gives the model a **context-aware momentum** feature: not just "momentum is high" but "momentum is high relative to its own history." When `mom20_pctile > 0.9`, the model can learn that this is a mean-reversion zone.

**Option B: Add rolling_median as explicit feature**:
```python
# Already computed in build_target() but discarded
# Expose as feature (known at time t, no leakage)
past_ret = close.pct_change(TARGET_LOOKAHEAD_DAYS)
spy["rolling_med_ret"] = past_ret.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
```

This gives the model direct access to the "bar" it needs to beat, enabling it to learn when the bar is too high.

**Option C: Both A + B combined** (recommended for maximum signal).

**If H3 confirmed (Small Sample)**: Lower SIGNAL_THRESHOLDS to increase sample size. No code change in features.py needed.

**If H4 confirmed (Regime)**: Reduce retrain_freq from 60 to 40 in backtest.py call.

---

## 3. File Change Matrix

| File | Lines Modified | Lines Added | Type |
|------|:-----------:|:----------:|:----:|
| `scripts/diagnose_signal_inversion.py` | 0 | ~150 | NEW |
| `src/config.py` | ~7 | 1 | MODIFY |
| `backtest.py` | ~3 | 2 | MODIFY |
| `src/data/features.py` | 0 | ~15 | MODIFY (conditional) |

---

## 4. Implementation Order

```
C-01 (Diagnosis) ─────→ Results determine C-05 approach
C-02 (Optuna run) ────→ C-03 (Apply to config)
                         ↓
C-04 (raw_prob record) → Backtest
                         ↓
C-05 (Conditional fix) → Final backtest → Verify signal rates
```

**Phase A**: C-01 + C-02 (parallel — independent)
**Phase B**: C-03 + C-04 (apply Optuna + raw_prob recording)
**Phase C**: C-05 (conditional on C-01 verdict)
**Phase D**: Final backtest + verification

---

## 5. Unchanged Components

| Component | Reason |
|-----------|--------|
| `src/models/trainer.py` | No training logic changes. Optuna results via config only. |
| `src/strategy/allocation.py` | Phase 3 allocation preserved. |
| `src/strategy/portfolio_backtest.py` | Phase 3 VIX/ADX forwarding preserved. |
| `scripts/verify_no_leakage.py` | Phase 2 leakage checks preserved. |
| `scripts/optuna_optimize.py` | Execute only, no modifications. |

---

## 6. Test Scenarios

| ID | Test | Expected | Verification Method |
|----|------|----------|:-------------------:|
| T-01 | Diagnosis completes | Verdict printed (H1-H4) | stdout |
| T-02 | Optuna CV accuracy | ≥ 55% | optuna script output |
| T-03 | Strong Buy actual up | > 50% | backtest signal_stats |
| T-04 | Buy actual up | > 50% | backtest signal_stats |
| T-05 | Strategy return | > +5% | port_metrics |
| T-06 | Sharpe ratio | > 0.5 | port_metrics |
| T-07 | MDD | > -15% | port_metrics |
| T-08 | Leakage check | 4/4 PASS | verify_no_leakage.py |
| T-09 | raw_prob in results | present in df_results | backtest inspection |
| T-10 | Signal sample sizes | SB > 10, B > 20 | signal_stats counts |
| T-11 | New features (if C-05) | No leakage in new features | verify_no_leakage.py |
| T-12 | 504-day backtest | Return > 0% | backtest_days=504 |

---

## 7. Rollback

1. Delete `scripts/diagnose_signal_inversion.py`
2. Revert LGBM_PARAMS to Phase 3 values
3. Remove `raw_prob` from backtest.py all_predictions
4. Remove new features from features.py (if C-05 applied)

Phase 2-3 changes are **never** rolled back.

---

## 8. Leakage Guard for New Features

If C-05 adds features, each must pass leakage verification:

| Feature | Data Source | Leakage Risk | Guard |
|---------|-----------|:------------:|-------|
| `mom10_pctile` | `momentum_10d.rolling(252).rank(pct=True)` | NONE — past data only | Uses `.shift(0)` equivalent |
| `mom20_pctile` | `momentum_20d.rolling(252).rank(pct=True)` | NONE — past data only | Rolling window on past returns |
| `rolling_med_ret` | `close.pct_change(20).rolling(252).median()` | NONE — past data only | Same as target's rolling_med, known at time t |

All new features use **only past/current data**. No future information leaks.
