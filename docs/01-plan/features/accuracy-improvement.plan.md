# Plan: NASDAQ 20-Day Prediction Accuracy Improvement

- Feature: `accuracy-improvement`
- Created: 2026-02-11
- Status: Plan

---

## 1. Problem Statement

| Item | Value |
|------|-------|
| CV Accuracy | 0.6650 +/- 0.0634 |
| Walk-Forward 30d Accuracy | **50.0%** (15/30) |
| CV-WF Gap | **16.5%p** |
| Base Rate (Target=1) | 67.1% |
| Top Features | T10Y2Y, DGS2_60up, signal, ret_rollstd60, MA120 |

**Core Issue**: CV accuracy (66.5%) collapses to coin-flip (50%) in walk-forward. This indicates structural overfitting + evaluation methodology weakness.

---

## 2. Root Cause Analysis

### 2.1 Overfitting Sources (Code-Level)

| # | Cause | Location | Severity |
|---|-------|----------|----------|
| 1 | **No regularization** (reg_alpha/reg_lambda missing) | `config.py:63-76` | Critical |
| 2 | **Over-complex model** (max_depth=8, num_leaves=63 for 2715 samples) | `config.py:67-68` | Critical |
| 3 | **Label leakage in CV** (gap=20 insufficient for purging) | `trainer.py:78` | High |
| 4 | **Target variable bias** (base rate 67%, model may parasitize) | `features.py:243` | High |
| 5 | **Redundant features** (96 features with high multicollinearity) | `features.py` | Medium |

### 2.2 Evaluation Weakness

| # | Issue | Impact |
|---|-------|--------|
| 1 | 30-day WF is statistically meaningless (95% CI: [33%, 67%] for true p=0.5) | Cannot distinguish 50% from 60% |
| 2 | Single train-test split depends on specific market regime | Unstable estimate |
| 3 | No retraining during WF period | Model staleness |

---

## 3. Improvement Plan (3 Phases)

### Phase 1: Immediate Fixes (P0) - Expected +8~15%p WF improvement

#### 1-1. Add Regularization + Reduce Complexity

**File**: `src/config.py`

```python
# BEFORE
LGBM_PARAMS = {
    "max_depth": 8,
    "num_leaves": 63,
    # no reg_alpha, reg_lambda
}

# AFTER
LGBM_PARAMS = {
    "max_depth": 5,          # 8 -> 5
    "num_leaves": 31,        # 63 -> 31
    "reg_alpha": 1.0,        # L1 regularization (NEW)
    "reg_lambda": 2.0,       # L2 regularization (NEW)
    "min_split_gain": 0.01,  # minimum split gain (NEW)
    "min_child_samples": 30, # 20 -> 30
}
```

- **Evidence**: Lopez de Prado (2018) - regularization prevents noise fitting
- **Expected**: +3~5%p WF accuracy
- **Difficulty**: Easy

#### 1-2. Fix Target Variable (Excess Return Based)

**File**: `src/data/features.py`, `src/config.py`

Current target: price up after 20 days (base rate ~67%). Model can achieve 67% by always predicting "up".

**New target**: Excess return over rolling median.

```python
# config.py
TARGET_EXCESS_RETURN = True
TARGET_ROLLING_MEDIAN_WINDOW = 252  # 1-year rolling median of 20d returns

# features.py build_target()
rolling_median = close.pct_change(20).rolling(252).median()
spy["Target"] = np.where(
    close.pct_change(20) > rolling_median.shift(20), 1, 0
)
```

This normalizes base rate to ~50%, forcing the model to learn actual patterns.

- **Evidence**: Gu, Kelly & Xiu (2020, RFS) - excess return targets outperform raw direction
- **Expected**: +5~10%p (eliminates base rate parasitism)
- **Difficulty**: Easy

#### 1-3. Extend Walk-Forward Evaluation

**File**: `backtest.py`

```python
# BEFORE: single 30-day window
n_days = min(backtest_days, len(X) - 100)  # default 30

# AFTER: expanding window, multiple periods
def run_expanding_wf(X, y, spy, eval_days=252, retrain_freq=60):
    """Expanding window walk-forward with periodic retraining."""
    results = []
    for start in range(len(X) - eval_days, len(X), retrain_freq):
        X_train = X.iloc[:start]
        X_test = X.iloc[start:start+retrain_freq]
        # train and predict...
```

- **Evidence**: Bailey et al. (2017, JCF) - minimum 180+ days for reliable evaluation
- **Expected**: Reliable accuracy estimate (no accuracy boost, but trustworthy number)
- **Difficulty**: Medium

#### 1-4. Reduce Redundant Lag Features

**File**: `src/config.py`

```python
# BEFORE
LAG_PERIODS = [1, 2, 3, 5, 10]  # 5 lags per indicator

# AFTER
LAG_PERIODS = [1, 5]  # 2 lags per indicator (sufficient)
```

This reduces feature count from 96 to ~60 without losing information.

- **Evidence**: Feature importance shows lag2, lag3, lag10 contribute minimally
- **Expected**: +1~2%p (reduced noise)
- **Difficulty**: Easy

---

### Phase 2: Structural Improvements (P1) - Expected +3~7%p additional

#### 2-1. Purged Cross-Validation

**File**: `src/models/trainer.py`

Replace `TimeSeriesSplit(gap=20)` with Purged+Embargo CV:
- Purge: Remove train samples whose targets overlap with test period
- Embargo: Additional 5-day buffer after test set

```python
def purged_time_series_split(X, y, n_splits=5, purge_days=20, embargo_days=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        test_start = test_idx[0]
        # Purge: remove train samples with target reaching into test
        purge_mask = train_idx < (test_start - purge_days)
        # Embargo: remove train samples right after test
        test_end = test_idx[-1]
        embargo_mask = train_idx > (test_end + embargo_days)
        clean_train = train_idx[purge_mask | embargo_mask]
        yield clean_train, test_idx
```

- **Evidence**: Lopez de Prado (2018), "Advances in Financial ML", Ch. 7
- **Expected**: CV estimate drops to 58-62% (realistic), WF gap closes
- **Difficulty**: Medium

#### 2-2. Sample Recency Weighting

**File**: `src/models/trainer.py`

```python
# Exponential decay weight (half-life ~2 years = ~500 trading days)
days_from_end = (X.index[-1] - X.index).days
sample_weight = np.exp(-0.00138 * days_from_end)  # ln(2)/500
model.fit(X, y, sample_weight=sample_weight)
```

- **Evidence**: Gu et al. (2020) - recency weighting improves OOS performance
- **Expected**: +2~3%p WF
- **Difficulty**: Easy

#### 2-3. New Features (Academic Evidence)

**File**: `src/data/collectors.py`, `src/data/features.py`

| Feature | Source | Evidence |
|---------|--------|----------|
| **Put/Call Ratio** | `FRED:PCERATIO` | Easley, O'Hara & Srinivas (1998, JF) |
| **SKEW Index** | Yahoo `^SKEW` | Bali & Murray (2013, JFQA) |
| **HY OAS** (Credit Spread) | `FRED:BAMLH0A0HYM2` | Gilchrist & Zakrajsek (2012, AER) |
| **DXY** (Dollar Index) | Yahoo `DX-Y.NYB` | Rapach et al. (2013, JF) |

- **Expected**: +2~5%p (orthogonal information to existing features)
- **Difficulty**: Medium

#### 2-4. Adversarial Validation

New utility to detect train/test distribution shift:

```python
def adversarial_validation(X_train, X_test):
    """Identify features that differ between train and test."""
    combined = pd.concat([X_train.assign(is_test=0), X_test.assign(is_test=1)])
    model = LGBMClassifier(max_depth=3, n_estimators=100)
    model.fit(combined.drop('is_test', axis=1), combined['is_test'])
    # AUC > 0.6 = significant distribution shift
    # Top features = most unstable -> removal candidates
```

- **Evidence**: Lopez de Prado (2018), Ch. 9
- **Expected**: Identifies 5-10 unstable features for removal
- **Difficulty**: Easy

#### 2-5. Extend Training Data to 2010

**File**: `src/config.py`

```python
DATA_START_DATE = "2010-01-01"  # was "2015-01-01"
```

Combined with recency weighting to avoid regime contamination.

- **Expected**: +1~3%p (more samples for generalization)
- **Difficulty**: Easy

---

### Phase 3: Advanced (P2) - Expected +2~5%p additional

#### 3-1. 3-Model Stacking Ensemble

```
Level 0: LightGBM + XGBoost + CatBoost (OOF predictions)
Level 1: Logistic Regression meta-learner
```

#### 3-2. Optuna Re-optimization

Expanded search space with regularization params.

#### 3-3. Isotonic Calibration

Switch from Platt to Isotonic for tree models (Niculescu-Mizil & Caruana, 2005).

#### 3-4. Conformal Prediction

Add prediction intervals; skip uncertain predictions.

---

## 4. Realistic Accuracy Targets

| Stage | WF Accuracy (180d+) | Notes |
|-------|---------------------|-------|
| Current | ~50% (30d, unreliable) | Statistically meaningless sample |
| Phase 1 | 55~60% | Regularization + target fix |
| Phase 2 | 58~63% | Purged CV + new features + weighting |
| Phase 3 | 60~65% | Ensemble + optimization |

**Academic benchmark**: Sustained >60% on monthly index prediction is excellent (Gu et al., 2020, RFS).

---

## 5. Files to Modify

| Phase | File | Changes |
|-------|------|---------|
| P1 | `src/config.py` | LGBM_PARAMS (regularization, depth), LAG_PERIODS, TARGET_EXCESS_RETURN |
| P1 | `src/data/features.py` | build_target() excess return logic |
| P1 | `backtest.py` | Expanding window WF, 180d+ evaluation |
| P2 | `src/models/trainer.py` | Purged CV, sample weighting |
| P2 | `src/data/collectors.py` | Put/Call, SKEW, HY OAS, DXY collectors |
| P2 | `src/data/features.py` | New features integration |
| P2 | New: `src/utils/adversarial_validation.py` | Distribution shift detection |

---

## 6. References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Gu, S., Kelly, B., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies*, 33(5).
3. Bailey, D. H. et al. (2017). "The Probability of Backtest Overfitting." *JCF*, 20(4).
4. Gilchrist, S. & Zakrajsek, E. (2012). "Credit Spreads and Business Cycle Fluctuations." *AER*, 102(4).
5. Easley, D., O'Hara, M., & Srinivas, P. S. (1998). "Option Volume and Stock Prices." *JF*, 53(2).
6. Rapach, D. E., Strauss, J. K., & Zhou, G. (2013). "International Stock Return Predictability." *JF*, 68(4).
7. Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML*.
8. Bali, T. G. & Murray, S. (2013). "Does Risk-Neutral Skewness Predict the Cross-Section of Equity Option Portfolio Returns?" *JFQA*, 48(4).
