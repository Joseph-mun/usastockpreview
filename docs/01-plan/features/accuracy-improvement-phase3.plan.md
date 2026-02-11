# Plan: Accuracy Improvement Phase 3

- Feature: `accuracy-improvement-phase3`
- Created: 2026-02-11
- Previous: `accuracy-improvement-phase2` (completed, 97% match rate)
- Priority: **HIGH** (전략 수익률 마이너스 → 수익 전환 필요)

---

## 0. Phase 2 Review

### Phase 2 Results (leakage-free, honest metrics)

| Metric | Value |
|--------|:-----:|
| CV Accuracy | 57.9% ± 7.6% |
| WF Accuracy (252d) | 57.9% (146/252) |
| Strategy Return | **-4.04%** |
| Sharpe Ratio | **-0.25** |
| MDD | -18.04% |
| Leakage Check | 4/4 PASS |

### Root Cause Analysis: 왜 57.9% 정확도에서 수익이 마이너스인가?

1. **ALLOCATION_TIERS 불일치**: Aggressive tier 기준이 80% — calibrated 확률 분포에서 80%+ 도달 거의 불가. 대부분 Cautious/Defensive tier로 분류되어 현금 비중 과다.

2. **Calibration 압축**: Platt calibration이 극단값을 중앙으로 압축. 실제 예측 확률 분포가 0.45-0.55 범위에 집중 → 모든 예측이 Cautious/Neutral로 분류됨.

3. **Window별 편차**: W2(85%) vs W3(38.3%) — 모델이 특정 시장 레짐에서 실패. 전체 평균은 57.9%이나 W3의 대규모 손실이 전체 수익 잠식.

4. **TQQQ 과대 가중**: 오답일 때의 TQQQ 3x 레버리지 손실이 정답일 때의 이익을 상쇄.

---

## 1. Phase 3 Changes (Priority Order)

### P0-1: ALLOCATION_TIERS 재설계 (CRITICAL)

현재 calibrated 확률 분포에 맞게 tier 경계 재설계.

**현재**:
```python
ALLOCATION_TIERS = [
    (0.80, 1.01, 0.40, 0.50, 0.10, "Aggressive"),  # 도달 불가
    (0.70, 0.80, 0.30, 0.50, 0.20, "Growth"),       # 거의 도달 불가
    (0.60, 0.70, 0.15, 0.55, 0.30, "Moderate"),
    (0.50, 0.60, 0.00, 0.55, 0.45, "Cautious"),
    (0.00, 0.50, 0.00, 0.30, 0.70, "Defensive"),
]
```

**변경**: Calibrated 확률 분포(~0.40-0.60 range)에 맞춘 세분화:
```python
ALLOCATION_TIERS = [
    (0.60, 1.01, 0.25, 0.55, 0.20, "Aggressive"),   # 상위 ~10%
    (0.55, 0.60, 0.15, 0.55, 0.30, "Growth"),        # 상위 ~25%
    (0.50, 0.55, 0.05, 0.55, 0.40, "Moderate"),      # 중간
    (0.45, 0.50, 0.00, 0.50, 0.50, "Cautious"),      # 하위 ~25%
    (0.00, 0.45, 0.00, 0.25, 0.75, "Defensive"),     # 하위 ~10%
]
```

핵심: TQQQ max weight 40% → 25% 축소, 확률 경계를 실제 분포 범위로 이동.

### P0-2: SIGNAL_THRESHOLDS 동기화

Allocation tier 변경에 맞춰 SIGNAL_THRESHOLDS도 조정:

```python
SIGNAL_THRESHOLDS = {
    "strong_buy": 0.60,
    "buy": 0.55,
    "neutral": 0.45,
}
```

### P1-1: Optuna 하이퍼파라미터 최적화

**파일**: `scripts/optuna_optimize.py` (신규)

현재 수동 설정된 LGBM_PARAMS를 Optuna로 최적화:
- Search space: max_depth(3-7), num_leaves(15-63), min_child_samples(20-50), learning_rate(0.01-0.1), reg_alpha(0.1-5.0), reg_lambda(0.5-10.0)
- Objective: TimeSeriesSplit CV accuracy (with purge)
- Trials: 100회
- 결과를 config.py에 반영

### P1-2: Probability Clipping 조정

현재 PROB_CLIP_MIN=0.05, PROB_CLIP_MAX=0.95 → calibrated 확률 분포 고려시 불필요하게 넓음.
더 좁은 범위로 조정하여 과도한 확신 방지:

```python
PROB_CLIP_MIN = 0.20
PROB_CLIP_MAX = 0.80
```

### P1-3: Backtest에서 VIX/ADX 전달 활성화

현재 `backtest.py`에서 `run_portfolio_backtest()`에 vix_series, adx_series를 전달하지 않음.
VIX filter와 regime detection이 backtest에서 비활성 → allocation이 순수 확률 기반으로만 동작.

**파일**: `backtest.py`의 portfolio simulation 부분에 VIX/ADX series 전달 추가.

### P2-1: Walk-Forward Window 안정화

Window 3의 38.3% (실패 구간) 분석:
- 해당 기간의 시장 레짐 파악
- 모델 예측 확률 분포 확인
- 필요시 retrain_freq를 60 → 40으로 단축하여 시장 변화에 더 빠르게 대응

---

## 2. Expected Results

| Metric | Phase 2 (현재) | Phase 3 (목표) | 근거 |
|--------|:-------------:|:-------------:|------|
| WF Accuracy | 57.9% | 55-60% | Optuna 미세 개선 |
| Strategy Return | -4.04% | **+5~15%** | Allocation 수정 핵심 |
| Sharpe Ratio | -0.25 | **0.5-1.5** | 현실적 목표 |
| MDD | -18.04% | -10~-15% | TQQQ 축소 + VIX filter |

**핵심**: 정확도 개선보다 **allocation 전략 최적화**로 수익 전환이 목표.
57.9% 정확도 자체는 충분한 edge. 이를 수익으로 전환하는 것이 Phase 3의 핵심.

---

## 3. Implementation Order

1. `src/config.py` — ALLOCATION_TIERS, SIGNAL_THRESHOLDS, PROB_CLIP 수정 (P0-1, P0-2, P1-2)
2. `backtest.py` — VIX/ADX series 전달 활성화 (P1-3)
3. 백테스트 실행 및 결과 확인 (allocation 변경 효과 측정)
4. `scripts/optuna_optimize.py` — Optuna 최적화 스크립트 (P1-1)
5. Optuna 실행 → config.py 업데이트 → 재학습
6. 최종 백테스트 및 비교

---

## 4. Test Scenarios

| # | Test | Expected | Verification |
|---|------|----------|-------------|
| T-01 | Allocation tier 분포 | 5개 tier 골고루 분포 | Backtest tier_distribution |
| T-02 | TQQQ max weight | 25% 이하 | Allocation tier 확인 |
| T-03 | Strategy return | 양수 (+5%+) | backtest.py --days 252 |
| T-04 | Sharpe ratio | > 0.3 | backtest metrics |
| T-05 | VIX filter 활성화 | 고변동 구간에서 TQQQ 감소 | portfolio backtest logs |
| T-06 | Optuna best params | CV accuracy ≥ 55% | optuna_optimize.py 출력 |
| T-07 | Signal 분포 | 4개 signal 모두 존재 | backtest report |
| T-08 | Leakage 검증 유지 | 4/4 PASS | verify_no_leakage.py |
| T-09 | MDD | > -15% | portfolio metrics |
| T-10 | Prob clip 범위 | [0.20, 0.80] | config.py 확인 |

---

## 5. Academic References

- De Miguel, Garlappi & Uppal (2009): *Optimal Versus Naive Diversification* — allocation boundary optimization
- Lopez de Prado (2018): *Advances in Financial Machine Learning* — purged WF, probability calibration
- Akiba et al. (2019): *Optuna: A Next-generation Hyperparameter Optimization Framework*
- Cheng & Madhavan (2009): *The Role of Leveraged ETFs in Portfolios* — TQQQ weight limits

---

## 6. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-----------:|:------:|------------|
| Optuna 과적합 | Medium | High | OOS validation, keep regularization |
| Allocation 과최적화 | Medium | Medium | 다른 기간 백테스트 (2년, 3년) |
| VIX filter 과도 | Low | Medium | Low Vol 비활성화 유지 |
| Prob clip 너무 좁음 | Low | Low | 0.15-0.85 대안 |

---

## 7. Rollback Strategy

```python
# config.py 원복
ALLOCATION_TIERS = [
    (0.80, 1.01, 0.40, 0.50, 0.10, "Aggressive"),
    (0.70, 0.80, 0.30, 0.50, 0.20, "Growth"),
    (0.60, 0.70, 0.15, 0.55, 0.30, "Moderate"),
    (0.50, 0.60, 0.00, 0.55, 0.45, "Cautious"),
    (0.00, 0.50, 0.00, 0.30, 0.70, "Defensive"),
]
```

Phase 2의 leakage fix (features.py, trainer.py)는 무조건 유지.
