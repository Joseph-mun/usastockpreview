# Plan: Accuracy Improvement Phase 4

- Feature: `accuracy-improvement-phase4`
- Created: 2026-02-11
- Previous: `accuracy-improvement-phase3` (completed, 97% match rate)
- Priority: **CRITICAL** (신호 역전 현상 수정 + Optuna HP 최적화)

---

## 0. Phase 3 Review

### Phase 3 Results

| Metric | Value |
|--------|:-----:|
| WF Accuracy | 58.3% (147/252) |
| Strategy Return | +2.23% |
| Sharpe Ratio | 0.26 |
| MDD | -16.64% |
| Leakage Check | 4/4 PASS |

### Critical Anomaly: Signal Inversion

Phase 3 Report에서 발견된 **치명적 이상 현상**:

| Signal | Count | Avg Prob | Actual Up Rate |
|--------|:-----:|:--------:|:--------------:|
| Strong Buy | 6 | 0.609 | **16.7%** |
| Buy | 19 | 0.579 | **21.1%** |
| Neutral | 136 | 0.495 | 52.9% |
| Sell | 91 | 0.434 | 42.9% |

**문제**: 모델이 "상승 확률이 높다"고 판단할수록 실제로는 **더 많이 하락**한다.
- Strong Buy (prob 0.609) → 실제 상승 16.7% (random 50%의 1/3 수준)
- Sell (prob 0.434) → 실제 상승 42.9% (Sell이 Strong Buy보다 2.5x 정확)

### Root Cause Hypotheses

**H1: Momentum-Target Mismatch (가장 유력)**

Feature 구성이 momentum 중심 (momentum_5d/10d/20d, roc_5d/10d/20d, Change20day) → 최근 강한 상승 = high prob 예측.
그러나 Target은 `excess_return` 모드 → `future_ret > rolling_median(past_ret)`.
장기 상승 후 rolling_median이 높아져 있으므로 "초과 수익" 달성이 어려움.
결과: **momentum features가 높을수록 Target=1 달성이 어려운 역방향 관계**.

**H2: Calibration Distortion**

Platt calibration (sigmoid)이 raw probability와 calibrated probability의 순위를 뒤집을 수 있음. 단, sigmoid는 단조 변환이므로 순위 보존이 보장됨 → H2는 기각 가능.

**H3: Small Sample Size**

Strong Buy 6건, Buy 19건 — 표본이 매우 작아 통계적 유의성이 없을 수 있음. 그러나 **방향이 체계적으로 역전**된다는 점은 noise만으로 설명 어려움.

**H4: Overfitting to Recent Regime**

Walk-forward Window 3의 40% 정확도 (60일 구간)에서 대부분의 Strong Buy가 발생했을 수 있음. 특정 시장 레짐에서 모델이 체계적으로 실패.

---

## 1. Phase 4 Changes (Priority Order)

### P0-1: Signal Inversion 진단 스크립트 (CRITICAL)

**파일**: `scripts/diagnose_signal_inversion.py` (신규)

강건한 근거 수집을 위한 진단:
1. **Window별 Signal-Accuracy 매핑**: 각 WF window에서 signal별 actual up rate
2. **Raw vs Calibrated probability 비교**: calibration 전후의 probability 분포
3. **Feature-Signal 상관**: Strong Buy 시점의 top features 값 분포 vs 평균
4. **Excess return target 검증**: Strong Buy 시점의 `rolling_median` 값 수준
5. **Temporal clustering**: Strong Buy가 특정 기간에 집중되어 있는지

출력: 진단 결과를 stdout에 출력하고, H1-H4 중 어느 것이 원인인지 판정.

### P0-2: Optuna 하이퍼파라미터 최적화 실행

**파일**: `scripts/optuna_optimize.py` (Phase 3에서 생성됨)

Phase 3에서 생성했으나 실행하지 않은 Optuna 스크립트 실행:
```bash
python scripts/optuna_optimize.py --trials 100
```

결과를 `src/config.py` LGBM_PARAMS에 반영.

### P1-1: Signal Inversion 수정 (진단 결과에 따라)

진단 결과에 따른 조건부 수정 방안:

**H1이 원인인 경우 (Momentum-Target Mismatch)**:
- Option A: Target을 `raw` 모드로 변경 (momentum과 일치하는 방향)
- Option B: Momentum features 제거/약화 (mean-reversion features 강화)
- Option C: **Hybrid target** — excess_return에 momentum penalty term 추가

**H3이 원인인 경우 (Small Sample)**:
- Signal threshold를 더 넓게 조정하여 Strong Buy/Buy 표본 확대
- 또는 3-tier signal로 단순화 (Buy/Hold/Sell)

**H4이 원인인 경우 (Regime-dependent)**:
- retrain_freq를 60 → 40으로 단축
- Regime-specific model (VIX high vs low에 따른 별도 모델)

### P1-2: Backtest 기간 확장 검증

현재 252일 (1년) 백테스트를 504일 (2년) 또는 756일 (3년)으로 확장하여 결과의 견고성 확인.

**파일**: `backtest.py`의 `run_backtest(backtest_days=504)` 호출

---

## 2. Expected Results

| Metric | Phase 3 (현재) | Phase 4 (목표) | 근거 |
|--------|:-------------:|:-------------:|------|
| WF Accuracy | 58.3% | 55-60% | Optuna 미세 개선 |
| Strategy Return | +2.23% | **+5~15%** | Signal inversion 수정이 핵심 |
| Sharpe Ratio | 0.26 | **0.5+** | 신호 일관성 개선 |
| MDD | -16.64% | -10~-15% | 역방향 베팅 감소 |
| Strong Buy actual up | 16.7% | **>55%** | Inversion 해소 |
| Buy actual up | 21.1% | **>50%** | Inversion 해소 |

**핵심**: Strong Buy/Buy의 actual up rate를 50% 이상으로 정상화하는 것이 Phase 4의 성공 기준.
현재 이 두 신호가 역방향이므로, 수정만으로 수익률이 대폭 개선될 수 있음.

---

## 3. Implementation Order

1. `scripts/diagnose_signal_inversion.py` — 진단 스크립트 실행 (P0-1)
2. `scripts/optuna_optimize.py --trials 100` — Optuna 실행 (P0-2)
3. `src/config.py` — Optuna 결과 반영
4. 진단 결과에 따른 수정 (P1-1) — 조건부
5. 백테스트 실행 및 신호 검증
6. 백테스트 기간 확장 검증 (P1-2)

---

## 4. Test Scenarios

| # | Test | Expected | Verification |
|---|------|----------|-------------|
| T-01 | Signal inversion 진단 완료 | H1-H4 판정 | diagnose_signal_inversion.py |
| T-02 | Optuna best CV accuracy | ≥ 55% | optuna_optimize.py output |
| T-03 | Strong Buy actual up rate | > 50% | backtest signal_stats |
| T-04 | Buy actual up rate | > 50% | backtest signal_stats |
| T-05 | Strategy return | > +5% | backtest metrics |
| T-06 | Sharpe ratio | > 0.5 | backtest metrics |
| T-07 | MDD | > -15% | portfolio metrics |
| T-08 | Leakage check | 4/4 PASS | verify_no_leakage.py |
| T-09 | 2년 백테스트 수익 | 양수 | backtest_days=504 |
| T-10 | 모든 신호 count | > 20 | sufficient sample size |

---

## 5. Academic References

- Moskowitz, Ooi & Pedersen (2012): *Time Series Momentum* — momentum factor decomposition
- Campbell & Thompson (2008): *Predicting Excess Stock Returns* — excess return prediction pitfalls
- Lopez de Prado (2018): *Advances in Financial ML* — triple barrier, meta-labeling
- Akiba et al. (2019): *Optuna: Next-gen Hyperparameter Optimization*

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Signal inversion이 표본 크기 문제 | Medium | High | 2년+ 백테스트로 표본 확대 |
| Optuna 과적합 | Medium | Medium | Purged CV + OOS validation |
| Target 모드 변경이 전체 파이프라인에 영향 | High | High | 기존 config 백업, 단계적 변경 |
| Feature 제거 시 정확도 하락 | Medium | Medium | AB 비교 (제거 전후) |

---

## 7. Phase 1-3 Journey Summary

| Phase | Focus | Return | Key Finding |
|:-----:|-------|:------:|-------------|
| 1 | Initial model | N/A | Fatal: data leakage (fake 96%) |
| 2 | Leakage fix | -4.04% | Honest 58% accuracy |
| 3 | Allocation optimization | +2.23% | Position sizing > accuracy |
| **4** | **Signal inversion fix + Optuna** | **target +5~15%** | **High-confidence signals are inverted** |
