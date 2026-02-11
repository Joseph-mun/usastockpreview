# Plan: Accuracy Improvement Phase 2

- Feature: `accuracy-improvement-phase2`
- Created: 2026-02-11
- Previous: `accuracy-improvement` (Phase 1 completed, 97% match rate)
- Priority: **CRITICAL** (Phase 1 target variable has fatal leakage bug)

---

## 0. Phase 1 Post-Mortem: Fatal Target Leakage

### Root Cause

Phase 1의 `excess_return` 타겟은 **과거 수익률을 예측하는 문제**로 정의됨:

```python
# CURRENT (BROKEN)
ret_20d = close.pct_change(TARGET_LOOKAHEAD_DAYS)  # PAST return: (P_t - P_{t-20}) / P_{t-20}
Target = ret_20d > rolling_median(ret_20d)           # "과거 20일이 중앙값 이상인가?"
```

### Evidence

| 검증 항목 | 결과 | 의미 |
|-----------|------|------|
| `momentum_20d` ↔ `ret_20d` 상관계수 | **1.000** | 피처가 타겟 자체 |
| `roc_20d` ↔ `ret_20d` 상관계수 | **1.000** | 피처가 타겟 자체 |
| `Change20day` ↔ `ret_20d` 상관계수 | **1.000** | 피처가 타겟 자체 |
| 과거 Target ↔ 미래 방향 일치율 | **51.4%** | 동전 던지기 수준 |
| WF 96% 정확도 | **가짜** | 이미 알려진 정보 예측 |

### Correct Target Definition

```python
# CORRECT (FUTURE-based)
future_ret_20d = close.shift(-TARGET_LOOKAHEAD_DAYS) / close - 1  # FUTURE return
rolling_med = future_ret_20d.rolling(252).median().shift(TARGET_LOOKAHEAD_DAYS)  # PAST median only
Target = future_ret_20d > rolling_med
```

핵심 차이:
- `pct_change(20)` = 과거 수익률 (시점 t에서 이미 알려진 값)
- `close.shift(-20)/close - 1` = 미래 수익률 (시점 t에서 모르는 값)

---

## 1. Phase 2 Changes (Priority Order)

### P0-1: Target Variable 버그 수정 (CRITICAL)

**파일**: `src/data/features.py` → `build_target()`

변경사항:
1. `ret_20d = close.pct_change(20)` → `future_ret = close.shift(-20) / close - 1`
2. `rolling_med`는 **과거 future_ret 값**만 사용 (leakage 방지를 위해 shift)
3. Target = `future_ret > rolling_med`

### P0-2: Leaky Feature 제거

**파일**: `src/data/features.py` → `DatasetBuilder.build()`

Target이 future 20d return 기반이므로, 동일 기간 과거 수익률 피처는 유지 가능.
단, future return과 직접 상관이 있는 피처는 없으므로 제거 대상 없음.

실제 문제는 P0-1이 해결되면 자동으로 사라짐 (과거 피처 ≠ 미래 타겟).

### P0-3: 검증 스크립트 추가

**파일**: `scripts/verify_no_leakage.py` (신규)

자동 검증 항목:
1. Target과 모든 피처 간 상관계수 |r| < 0.5 확인
2. Target base rate 40-60% 범위 확인
3. 간단한 OOS 테스트: random forest으로 1-fold 정확도 확인 (>70%면 의심)

### P1-1: Purged Cross-Validation

**파일**: `src/models/trainer.py`

현재 `TimeSeriesSplit(gap=20)`에 **embargo 추가**:
- Train 마지막 20일 제거 (purge)
- Test 시작 20일 이전부터 gap (embargo)
- Lopez de Prado (2018) 방법론

### P1-2: Adversarial Validation

**파일**: `scripts/adversarial_validation.py` (신규)

Train/Test 구분 가능 여부 검증:
- Train에 label=0, Test에 label=1 부여
- LightGBM으로 AUC 계산
- AUC > 0.6이면 data shift 경고

### P1-3: Feature Autocorrelation Check

**파일**: `src/data/features.py` → `_build_feature_matrix()`

높은 자기상관 피처 제거:
- Feature와 Feature.shift(20) 간 상관계수 > 0.8인 피처 경고/제거
- 20일 앞뒤로 거의 동일한 값을 가지는 피처는 예측력이 낮음

---

## 2. Expected Results

| Metric | Phase 1 (leaky) | Phase 2 (fixed) | 비고 |
|--------|:---------------:|:---------------:|------|
| Target Base Rate | 47.2% | ~50% | 미래 기반 rolling median |
| WF Accuracy (252d) | 96.0% (가짜) | **52-58%** | 실제 예측 난이도 |
| CV Accuracy | 81.1% | **55-62%** | 정규화 + purged CV |
| Feature-Target max |r|| 1.000 | **< 0.3** | leakage 완전 제거 |
| Sharpe Ratio | 3.46 (가짜) | **0.5-1.5** | 현실적 수준 |

**주의**: 52-58% 정확도가 현실적이며, 금융 시장에서는 55%+ 정확도만으로도 수익 가능.

---

## 3. Implementation Order

1. `src/data/features.py` — build_target() 수정 (P0-1)
2. `scripts/verify_no_leakage.py` — 검증 스크립트 (P0-3)
3. 재학습 + 검증 스크립트 실행
4. `src/models/trainer.py` — Purged CV (P1-1)
5. `scripts/adversarial_validation.py` — Adversarial validation (P1-2)
6. `backtest.py --days 252` 재실행
7. 결과 비교 및 분석

---

## 4. Test Scenarios

| # | Test | Expected | Verification |
|---|------|----------|-------------|
| T-01 | future_ret 정의 검증 | `close.shift(-20)/close - 1` | assert 계산 일치 |
| T-02 | Target base rate | 45-55% | `y.mean()` |
| T-03 | Feature-Target 상관계수 | 모두 |r| < 0.5 | verify_no_leakage.py |
| T-04 | momentum_20d ↔ Target | |r| < 0.3 (not 1.0) | 상관계수 확인 |
| T-05 | WF accuracy | 50-60% | backtest.py --days 252 |
| T-06 | CV accuracy | 52-62% | training_pipeline.py |
| T-07 | raw mode 호환성 | 변경 없음 | TARGET_MODE="raw" 테스트 |
| T-08 | Purged CV gap | train/test 40일 gap | trainer.py 검증 |
| T-09 | Adversarial AUC | < 0.6 | adversarial_validation.py |
| T-10 | Leakage 스크립트 통과 | 모든 체크 PASS | verify_no_leakage.py |

---

## 5. Academic References

- Lopez de Prado (2018): *Advances in Financial Machine Learning* — Purged CV, triple barrier
- Bailey & Lopez de Prado (2014): *The Deflated Sharpe Ratio* — backtest overfitting detection
- Gu, Kelly & Xiu (2020): *Empirical Asset Pricing via ML* — excess return target (correctly defined)

---

## 6. Rollback Strategy

```python
# config.py
TARGET_MODE = "raw"  # excess_return 대신 원래 방식으로 복원
```

Phase 1의 다른 변경사항 (regularization, lag reduction, expanding WF)은 유효하므로 유지.
Target variable만 수정.
