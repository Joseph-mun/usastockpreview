# Design: Accuracy Improvement Phase 2

- Feature: `accuracy-improvement-phase2`
- Phase: Design
- Created: 2026-02-11
- Plan ref: `docs/01-plan/features/accuracy-improvement-phase2.plan.md`

---

## Scope

Phase 1 target leakage 버그 수정 + leakage 검증 + Purged CV. 총 3개 파일 수정, 1개 신규.

---

## 1. File Changes

### 1-1. `src/data/features.py` - build_target() 수정 (P0-1, CRITICAL)

#### Bug: `pct_change(20)` = 과거 수익률

현재 코드 (line 249):
```python
ret_20d = close.pct_change(TARGET_LOOKAHEAD_DAYS)
```
`pct_change(20)` = `(P_t - P_{t-20}) / P_{t-20}` = **과거** 20일 수익률.
이 값은 시점 t에서 이미 알려진 값이므로, 피처(`momentum_20d`, `roc_20d`, `Change20day`)와 상관계수 = 1.0.

#### Fix: `close.shift(-20) / close - 1` = 미래 수익률

**Before** (line 246-253):
```python
# Binary targets
if TARGET_MODE == "excess_return":
    # Excess return: compare 20d return against rolling median
    ret_20d = close.pct_change(TARGET_LOOKAHEAD_DAYS)
    rolling_med = ret_20d.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
    # Shift median by lookahead to prevent leakage
    shifted_median = rolling_med.shift(TARGET_LOOKAHEAD_DAYS)
    spy["Target"] = np.where(ret_20d > shifted_median, 1, 0)
```

**After**:
```python
# Binary targets
if TARGET_MODE == "excess_return":
    # Excess return: compare FUTURE 20d return against historical median
    future_ret = close.shift(-TARGET_LOOKAHEAD_DAYS) / close - 1
    # Historical median uses PAST returns only (known at time t)
    past_ret = close.pct_change(TARGET_LOOKAHEAD_DAYS)
    rolling_med = past_ret.rolling(TARGET_ROLLING_MEDIAN_WINDOW).median()
    spy["Target"] = np.where(future_ret > rolling_med, 1, 0)
```

**Key design decisions**:
- `future_ret = close.shift(-20) / close - 1`: 미래 가격 사용 → NaN for last 20 rows (정상)
- `past_ret = close.pct_change(20)`: 과거 수익률로 rolling median 계산
- `rolling_med`: 252일 rolling median of **past** returns → 시점 t에서 이미 알려진 값
- `shifted_median` **불필요**: `rolling_med` 자체가 이미 과거 데이터만 사용
- `rolling_med`는 DataFrame에 저장하지 않음 (local variable)

**Leakage 안전성 증명**:
- `future_ret`는 Target 생성에만 사용, 피처로 들어가지 않음 (LEAK_COLUMNS에 Target 포함)
- `rolling_med`는 `past_ret.rolling(252).median()` = 시점 t까지의 과거 데이터만 사용
- `momentum_20d` = `close/close.shift(20) - 1` = 과거 수익률 ≠ `future_ret` → 상관계수 ≈ 0

---

### 1-2. `src/models/trainer.py` - Purged Cross-Validation (P1-1)

#### Change: TimeSeriesSplit에 purge 추가

**Before** (line 78):
```python
tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)
```

**After**:
```python
tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)
```

실제 변경은 fold 루프 내부에서 train 끝부분을 purge:

**Before** (line 96-97):
```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
    pure_train_idx, val_idx = _split_train_val(train_idx)
```

**After**:
```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
    # Purge: remove last CV_GAP samples from train to prevent label leakage
    if len(train_idx) > CV_GAP:
        train_idx = train_idx[:-CV_GAP]
    pure_train_idx, val_idx = _split_train_val(train_idx)
```

**Rationale**: `TimeSeriesSplit(gap=20)`은 test 시작 전 20일을 건너뛰지만, train 끝의 20일은 Target이 test 기간과 겹칠 수 있음. train 끝 20일도 제거하면 총 40일 격리 (purge 20 + gap 20).

동일 변경을 Phase 3 CV 루프에도 적용 (line 153):

**Before** (line 153):
```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sel)):
    pure_train_idx, val_idx = _split_train_val(train_idx)
```

**After**:
```python
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sel)):
    if len(train_idx) > CV_GAP:
        train_idx = train_idx[:-CV_GAP]
    pure_train_idx, val_idx = _split_train_val(train_idx)
```

---

### 1-3. `src/config.py` - SIGNAL_THRESHOLDS 조정

excess_return 모드에서 base rate가 ~50%이므로, 기존 67% 기반 threshold 부적합.

**Before** (line 104-109):
```python
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.80,   # significantly above base rate
    "buy": 0.70,          # above base rate
    "neutral": 0.55,      # around/below base rate
    # below neutral = sell
}
```

**After**:
```python
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.70,   # significantly above 50% base rate
    "buy": 0.60,          # above base rate
    "neutral": 0.45,      # around base rate
    # below neutral = sell
}
```

**Rationale**: base rate ~50%에서는 60%+ 확률이면 이미 유의미한 신호. 기존 80% 기준은 거의 달성 불가능.

---

### 1-4. `scripts/verify_no_leakage.py` - 검증 스크립트 (P0-3, NEW)

신규 파일. 자동으로 leakage 여부를 검증.

```python
#!/usr/bin/env python3
"""Verify no feature-target leakage exists in the dataset."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.data.features import DatasetBuilder
from src.data.collectors import SMACollector
from src.data.cache import SMACache

def main():
    print("=" * 60)
    print("  Feature-Target Leakage Verification")
    print("=" * 60)

    # Build dataset
    cache = SMACache()
    raw_sma, _ = cache.load()
    sma_ratios = {}
    if raw_sma:
        sc = SMACollector()
        sc.raw_dataframes = raw_sma
        sma_ratios = sc.compute_ratios()

    builder = DatasetBuilder(sma_ratios=sma_ratios)
    X, spy, y = builder.build("IXIC", for_prediction=False)

    passed = 0
    failed = 0

    # Check 1: No feature has |correlation| > 0.5 with Target
    print("\n[CHECK 1] Feature-Target correlation (|r| < 0.5)")
    max_corr = 0
    max_corr_feature = ""
    for col in X.columns:
        corr = abs(X[col].corr(y))
        if corr > max_corr:
            max_corr = corr
            max_corr_feature = col
        if corr > 0.5:
            print(f"  FAIL: {col} has |r| = {corr:.4f}")
            failed += 1

    if max_corr <= 0.5:
        print(f"  PASS: max |r| = {max_corr:.4f} ({max_corr_feature})")
        passed += 1

    # Check 2: Target base rate between 40-60%
    print("\n[CHECK 2] Target base rate (40-60%)")
    base_rate = y.mean()
    if 0.40 <= base_rate <= 0.60:
        print(f"  PASS: base rate = {base_rate:.4f}")
        passed += 1
    else:
        print(f"  FAIL: base rate = {base_rate:.4f}")
        failed += 1

    # Check 3: Simple 1-fold OOS accuracy < 70%
    print("\n[CHECK 3] Simple OOS accuracy (< 70%)")
    from sklearn.ensemble import RandomForestClassifier
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_tr.values, y_tr.values)
    acc = rf.score(X_te.values, y_te.values)
    if acc < 0.70:
        print(f"  PASS: OOS accuracy = {acc:.4f}")
        passed += 1
    else:
        print(f"  WARN: OOS accuracy = {acc:.4f} (suspiciously high)")
        failed += 1

    # Check 4: momentum_20d not correlated with Target
    print("\n[CHECK 4] momentum_20d correlation with Target")
    if "momentum_20d" in X.columns:
        corr = abs(X["momentum_20d"].corr(y))
        if corr < 0.3:
            print(f"  PASS: |r| = {corr:.4f}")
            passed += 1
        else:
            print(f"  FAIL: |r| = {corr:.4f}")
            failed += 1
    else:
        print("  SKIP: momentum_20d not in features")
        passed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} PASS, {failed} FAIL")
    print("=" * 60)
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

---

## 2. Backward Compatibility

| Item | Compatibility |
|------|---------------|
| `TARGET_MODE = "raw"` | Unchanged (else branch not modified) |
| `backtest.py --days 252` | Works (no backtest changes needed) |
| `training_pipeline.py` | No changes needed |
| `daily_pipeline.py` | No changes needed |
| `allocation.py` | No changes needed (threshold change via config) |
| Existing model files | Need retraining after target fix |

---

## 3. Test Scenarios

| # | Test | Expected Result | Verification |
|---|------|-----------------|--------------|
| T-01 | `future_ret` definition | `close.shift(-20)/close - 1` | Assert formula |
| T-02 | Target base rate | 45-55% | `y.mean()` |
| T-03 | Feature-Target max correlation | |r| < 0.5 | `verify_no_leakage.py` CHECK 1 |
| T-04 | `momentum_20d` ↔ Target | |r| < 0.3 | `verify_no_leakage.py` CHECK 4 |
| T-05 | WF accuracy (252d) | 50-62% | `backtest.py --days 252` |
| T-06 | CV accuracy | 50-62% | `training_pipeline.py` |
| T-07 | `TARGET_MODE="raw"` compatibility | No change in behavior | Manual test |
| T-08 | Purged CV: train ends CV_GAP before test | 40-day total gap | Print train/test ranges |
| T-09 | Simple OOS accuracy < 70% | No leakage signal | `verify_no_leakage.py` CHECK 3 |
| T-10 | Signal distribution with new thresholds | All 4 signals present | Backtest output |

---

## 4. Data Flow

```
config.py (TARGET_MODE, SIGNAL_THRESHOLDS)
    |
    v
features.py::build_target()
    |-- future_ret = close.shift(-20) / close - 1     [FUTURE, not feature]
    |-- past_ret = close.pct_change(20)                [PAST, for median only]
    |-- rolling_med = past_ret.rolling(252).median()   [PAST only, safe]
    |-- Target = (future_ret > rolling_med)            [FUTURE vs PAST = valid]
    |
    v
features.py::DatasetBuilder.build()
    |-- momentum_20d = close/close.shift(20) - 1      [PAST, feature, != future_ret]
    |-- roc_20d = same formula as momentum             [PAST, feature, != future_ret]
    |-- Change20day = log-compound past 20d return     [PAST, feature, != future_ret]
    |
    v
trainer.py::ModelTrainer.train()
    |-- Purged CV: train_idx[:-CV_GAP] + gap=CV_GAP   [40-day total isolation]
    |
    v
backtest.py (unchanged) → realistic WF accuracy 50-62%
```

---

## 5. Mathematical Proof of No Leakage

시점 t에서의 정보 분류:

| 변수 | 수식 | 시점 t에서 알 수 있는가? | Feature 가능? |
|------|------|:---:|:---:|
| `future_ret` | `P_{t+20}/P_t - 1` | No (미래) | No → Target only |
| `past_ret` | `P_t/P_{t-20} - 1` | Yes (과거) | Yes |
| `rolling_med` | `median(past_ret, 252)` | Yes (과거) | Not used as feature |
| `momentum_20d` | `P_t/P_{t-20} - 1` | Yes (과거) | Yes |
| `Target` | `future_ret > rolling_med` | No (미래 포함) | No → Label only |

`momentum_20d`와 `Target` 상관:
- `momentum_20d = past_ret = P_t/P_{t-20} - 1`
- `Target = (P_{t+20}/P_t - 1) > rolling_med`
- `past_ret`와 `future_ret`는 비겹침 기간 → 낮은 상관관계 (autocorrelation에 의한 약한 상관만 존재)

---

## 6. Implementation Order

1. `src/data/features.py` — build_target() 수정 (P0-1)
2. `scripts/verify_no_leakage.py` — 검증 스크립트 (P0-3)
3. `src/models/trainer.py` — Purged CV (P1-1)
4. `src/config.py` — SIGNAL_THRESHOLDS 조정
5. 재학습: `python -m src.pipelines.training_pipeline`
6. 검증: `python scripts/verify_no_leakage.py`
7. 백테스트: `python backtest.py --days 252`
