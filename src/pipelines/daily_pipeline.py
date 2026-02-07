# -*- coding: utf-8 -*-
"""Daily prediction pipeline: load models, fetch data, predict, notify."""

import sys
import time
from datetime import datetime

from src.config import INDEX_CONFIGS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
from src.data.collectors import get_index_data, SMACollector
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.models.predictor import IndexPredictor
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
    3. Fetch today's data (index prices, VIX, bonds)
    4. Build features and predict
    5. Send results via Telegram
    """
    start_time = time.time()

    def log(msg):
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    # ── Step 1: Load models ──
    log("Step 1/5: 모델 로드")
    predictor = IndexPredictor()
    loaded = predictor.load_models()
    if not loaded:
        log("ERROR: 로드된 모델이 없습니다. 먼저 학습을 실행하세요.")
        sys.exit(1)
    log(f"  로드 완료: {loaded}")

    # ── Step 2: Load SMA cache ──
    log("Step 2/5: SMA 캐시 로드")
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
    log("Step 3/5: 데이터 수집 + 피처 생성 + 예측")
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

    # ── Step 5: Send via Telegram ──
    log("Step 4/5: Telegram 메시지 생성")

    # Summary text
    summary = format_daily_summary(
        predictions=predictions,
        prices=current_prices,
        prev_predictions=prev_predictions,
        indicators=current_indicators,
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

    log("Step 5/5: Telegram 발송")
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
