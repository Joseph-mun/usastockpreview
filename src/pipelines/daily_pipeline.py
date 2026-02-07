# -*- coding: utf-8 -*-
"""Daily prediction pipeline: load models, fetch data, predict, notify."""

import sys
import time
from datetime import datetime

from src.config import INDEX_CONFIGS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, META_LEARNER_ENABLED
from src.data.collectors import get_index_data, SMACollector
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.models.predictor import IndexPredictor
from src.models.meta_learner import MetaLearner
from src.strategy.allocation import get_allocation, check_rebalance
from src.notification.telegram_bot import TelegramNotifier
from src.notification.formatters import (
    format_daily_summary,
    format_5day_table,
    create_probability_chart,
)


def run_daily_prediction(verbose: bool = True):
    """
    Daily prediction pipeline:
    1. Load pre-trained models
    2. Load SMA cache (from weekly training)
    3. Fetch today's data (index prices, VIX, bonds), build features, predict
    4. Portfolio allocation + Meta Learner
    5. Generate Telegram messages
    6. Send via Telegram
    """
    start_time = time.time()

    def log(msg):
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ── Step 1: Load models ──
    log("Step 1/6: 모델 로드")
    predictor = IndexPredictor()
    loaded = predictor.load_models()
    if not loaded:
        log("ERROR: 로드된 모델이 없습니다. 먼저 학습을 실행하세요.")
        sys.exit(1)
    log(f"  로드 완료: {loaded}")

    # ── Step 2: Load SMA cache ──
    log("Step 2/6: SMA 캐시 로드")
    cache = SMACache()
    raw_sma, sma_meta = cache.load()

    if raw_sma:
        sma_collector = SMACollector()
        sma_collector.raw_dataframes = raw_sma
        sma_ratios = sma_collector.compute_ratios()
        log(f"  SMA 캐시 로드 완료 (keys: {list(raw_sma.keys())})")
    else:
        sma_ratios = {}
        log("  WARNING: SMA 캐시 없음 - SMA 비율은 기본값(0.5) 사용")

    # ── Step 3-4: Build features and predict for each index ──
    log("Step 3/6: 데이터 수집 + 피처 생성 + 예측")
    builder = DatasetBuilder(sma_ratios=sma_ratios)

    predictions = {}
    prev_predictions = {}
    current_prices = {}
    current_indicators = {}
    prob_histories = {}
    price_histories = {}
    all_X = {}

    for index_name, cfg in INDEX_CONFIGS.items():
        if index_name not in loaded:
            log(f"  [{index_name}] 모델 없음 - 건너뜀")
            continue

        try:
            ticker = cfg["ticker"]
            log(f"  [{index_name}] 데이터 수집 중...")

            X, spy, _ = builder.build(ticker, for_prediction=True)
            all_X[index_name] = X

            # Current prediction
            current_features = X.iloc[-1]
            prob = predictor.predict_current(index_name, current_features)
            predictions[index_name] = prob

            # Previous day prediction (for delta)
            if len(X) >= 2:
                prev_features = X.iloc[-2]
                prev_prob = predictor.predict_current(index_name, prev_features)
                prev_predictions[index_name] = prev_prob

            # Current price
            close_col = "Adj Close" if "Adj Close" in spy.columns else "Close"
            current_prices[index_name] = float(spy[close_col].iloc[-1])

            # Indicators (from last row)
            if "vix" in spy.columns and index_name == "NASDAQ":
                current_indicators["vix"] = float(spy["vix"].iloc[-1])
            if "rsi" in spy.columns and index_name == "NASDAQ":
                current_indicators["rsi"] = float(spy["rsi"].iloc[-1])
            if "ratio_sma50" in spy.columns and index_name == "NASDAQ":
                current_indicators["sma50_ratio"] = float(spy["ratio_sma50"].iloc[-1])

            # Probability history (for chart + table)
            prob_hist = predictor.predict_history(index_name, X, days=500)
            prob_histories[index_name] = prob_hist

            # Price history (for chart)
            price_histories[index_name] = spy[close_col].copy()

            signal_text, emoji = IndexPredictor.get_signal(prob)
            log(f"  [{index_name}] 확률: {prob*100:.1f}% → {signal_text}")

        except Exception as e:
            log(f"  [{index_name}] ERROR: {e}")
            continue

    if not predictions:
        log("ERROR: 예측 결과가 없습니다.")
        sys.exit(1)

    # ── Step 4: Allocation + Meta Learner ──
    log("Step 4/6: 포트폴리오 배분 계산")

    # Use NASDAQ probability as primary signal for allocation
    primary_index = "NASDAQ"
    primary_prob = predictions.get(primary_index, 0.5)

    # Meta Learner: adjust probability with incremental learning
    meta_learner = None
    if META_LEARNER_ENABLED:
        try:
            meta_learner = MetaLearner()
            meta_learner.load()

            # Build meta features from recent history
            recent_probs = []
            recent_actuals = []
            nasdaq_hist = prob_histories.get(primary_index)
            if nasdaq_hist is not None and len(nasdaq_hist) >= 6:
                recent_probs = nasdaq_hist["Probability"].iloc[-6:-1].tolist()

            rsi_val = current_indicators.get("rsi", 50.0)
            meta_features = meta_learner.build_features(
                lgbm_prob=primary_prob,
                recent_probs=recent_probs,
                recent_actuals=recent_actuals,
                rsi=rsi_val,
            )
            adjusted_prob = meta_learner.predict(meta_features)
            log(f"  Meta Learner: {primary_prob*100:.1f}% -> {adjusted_prob*100:.1f}%")
            primary_prob = adjusted_prob
        except Exception as e:
            log(f"  Meta Learner 오류 (기본 확률 사용): {e}")

    # Compute allocation
    allocation = get_allocation(primary_prob)

    # Check rebalance (compare with previous day)
    rebalance_info = None
    prev_prob = prev_predictions.get(primary_index)
    if prev_prob is not None:
        prev_alloc = get_allocation(prev_prob)
        rebalance_info = check_rebalance(primary_prob, prev_prob, prev_alloc.tier_label)

    log(f"  Tier: {allocation.tier_label} | TQQQ: {allocation.tqqq_weight*100:.0f}% | SPY: {allocation.spy_weight*100:.0f}% | Cash: {allocation.cash_weight*100:.0f}%")
    if rebalance_info:
        rebal, reason = rebalance_info
        log(f"  Rebalance: {'YES' if rebal else 'NO'} ({reason})")

    # ── Step 5: Send via Telegram ──
    log("Step 5/6: Telegram 메시지 생성")

    # Summary text
    summary = format_daily_summary(
        predictions=predictions,
        prices=current_prices,
        prev_predictions=prev_predictions,
        indicators=current_indicators,
        allocation=allocation,
        rebalance_info=rebalance_info,
    )

    # Chart image
    chart_bytes = None
    try:
        chart_bytes = create_probability_chart(
            history=prob_histories,
            prices=price_histories,
            days=60,
        )
        log("  차트 이미지 생성 완료")
    except Exception as e:
        log(f"  차트 생성 실패: {e}")

    # 5-day table
    table_text = None
    try:
        table_text = format_5day_table(history=prob_histories)
    except Exception as e:
        log(f"  5일 테이블 생성 실패: {e}")

    log("Step 6/6: Telegram 발송")
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        notifier = TelegramNotifier()
        notifier.send_prediction_report(
            summary_text=summary,
            chart_bytes=chart_bytes,
            table_text=table_text,
        )
        log("  Telegram 발송 완료")
    else:
        log("  WARNING: Telegram 설정 없음 - 화면 출력만")
        print("\n" + summary)
        if table_text:
            # Strip HTML tags for console output
            import re
            clean = re.sub(r"<[^>]+>", "", table_text)
            print("\n" + clean)

    elapsed = time.time() - start_time
    log(f"\n완료 (소요 시간: {elapsed:.1f}초)")


def main():
    run_daily_prediction()


if __name__ == "__main__":
    main()
