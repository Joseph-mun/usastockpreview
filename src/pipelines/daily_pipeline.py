# -*- coding: utf-8 -*-
"""Daily prediction pipeline: load models, fetch data, predict, notify."""

import json
import sys
import time
from datetime import datetime, timezone, timedelta

from src.config import (
    INDEX_CONFIGS, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, META_LEARNER_ENABLED,
    DATA_DIR, DEFAULT_PROBABILITY, ENSEMBLE_ENABLED, get_logger,
)
from src.data import get_close_col
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


logger = get_logger(__name__)


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
        logger.info(msg)

    # ── Step 1: Load models ──
    log("Step 1/6: 모델 로드")
    ensemble_active = False
    if ENSEMBLE_ENABLED:
        try:
            from src.models.predictor import EnsemblePredictor
            predictor = EnsemblePredictor()
            loaded = predictor.load_models()
            if loaded:
                ensemble_active = True
                log(f"  앙상블 모델 로드 완료: {loaded}")
            else:
                log("  앙상블 모델 없음 → 단일 모델로 폴백")
                predictor = IndexPredictor()
                loaded = predictor.load_models()
        except Exception as e:
            log(f"  앙상블 로드 실패 ({e}) → 단일 모델로 폴백")
            predictor = IndexPredictor()
            loaded = predictor.load_models()
    else:
        predictor = IndexPredictor()
        loaded = predictor.load_models()

    if not loaded:
        log("ERROR: 로드된 모델이 없습니다. 먼저 학습을 실행하세요.")
        sys.exit(1)
    log(f"  로드 완료: {loaded} (앙상블: {'ON' if ensemble_active else 'OFF'})")

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
            close_col = get_close_col(spy)
            current_prices[index_name] = float(spy[close_col].iloc[-1])

            # Indicators (from last row)
            if "vix" in spy.columns and index_name == "NASDAQ":
                current_indicators["vix"] = float(spy["vix"].iloc[-1])
            if "rsi" in spy.columns and index_name == "NASDAQ":
                current_indicators["rsi"] = float(spy["rsi"].iloc[-1])
            if "ratio_sma50" in spy.columns and index_name == "NASDAQ":
                current_indicators["sma50_ratio"] = float(spy["ratio_sma50"].iloc[-1])
            if "adx" in spy.columns and index_name == "NASDAQ":
                current_indicators["adx"] = float(spy["adx"].iloc[-1])

            # Probability history (for chart + table)
            prob_hist = predictor.predict_history(index_name, X, days=500)
            prob_histories[index_name] = prob_hist

            # Price history (for chart)
            price_histories[index_name] = spy[close_col].copy()

            signal_text, emoji = IndexPredictor.get_signal(prob)
            log(f"  [{index_name}] 확률: {prob*100:.1f}% → {signal_text}")

        except Exception as e:
            logger.exception("[%s] 예측 중 오류 발생", index_name)
            continue

    if not predictions:
        log("ERROR: 예측 결과가 없습니다.")
        sys.exit(1)

    # ── Step 4: Allocation + Meta Learner ──
    log("Step 4/6: 포트폴리오 배분 계산")

    # Use NASDAQ probability as primary signal for allocation
    primary_index = "NASDAQ"
    primary_prob = predictions.get(primary_index, DEFAULT_PROBABILITY)

    # Meta Learner: adjust probability with incremental learning
    meta_learner = None
    if META_LEARNER_ENABLED:
        try:
            meta_learner = MetaLearner()
            meta_learner.load()

            # Build meta features from recent history
            recent_probs = []
            nasdaq_hist = prob_histories.get(primary_index)
            if nasdaq_hist is not None and len(nasdaq_hist) >= 6:
                recent_probs = nasdaq_hist["Probability"].iloc[-6:-1].tolist()

            recent_actuals = meta_learner.get_recent_actuals(n=5)

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

    # Compute allocation (with VIX filter + regime detection)
    current_vix = current_indicators.get("vix")
    current_adx = current_indicators.get("adx")
    allocation = get_allocation(primary_prob, vix=current_vix, adx=current_adx)

    # Check rebalance (compare with previous day)
    rebalance_info = None
    prev_prob = prev_predictions.get(primary_index)
    if prev_prob is not None:
        prev_alloc = get_allocation(prev_prob, vix=current_vix, adx=current_adx)
        rebalance_info = check_rebalance(
            primary_prob, prev_prob, prev_alloc.tier_label,
            vix=current_vix, adx=current_adx,
        )

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
    except (ValueError, KeyError, RuntimeError) as e:
        log(f"  차트 생성 실패: {e}")

    # 5-day table
    table_text = None
    try:
        table_text = format_5day_table(history=prob_histories)
    except (ValueError, KeyError) as e:
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
        logger.info("\n%s", summary)
        if table_text:
            import re
            clean = re.sub(r"<[^>]+>", "", table_text)
            logger.info("\n%s", clean)

    # ── Step 6.5: Save prediction signal as JSON ──
    # 자동매매 시스템이 이 파일을 읽어서 매매 신호로 활용
    try:
        kst = timezone(timedelta(hours=9))
        now_kst = datetime.now(kst)

        signal_text, _ = IndexPredictor.get_signal(primary_prob)
        prev_prob_value = prev_predictions.get(primary_index)

        signal_data: dict = {
            "date": now_kst.strftime("%Y-%m-%d"),
            "timestamp": now_kst.isoformat(),
            "probability": round(primary_prob, 4),
            "signal": signal_text,
            "tier": allocation.tier_label,
            "allocation": {
                "tqqq": round(allocation.tqqq_weight, 2),
                "splg": round(allocation.spy_weight, 2),
                "cash": round(allocation.cash_weight, 2),
            },
            "indicators": {
                "rsi": round(current_indicators.get("rsi", 0.0), 2),
                "vix": round(current_indicators.get("vix", 0.0), 2),
                "adx": round(current_indicators.get("adx", 0.0), 2),
                "sma50_ratio": round(current_indicators.get("sma50_ratio", 0.0), 4),
            },
            "vix_filter": allocation.vix_filter_label,
            "regime": allocation.regime_label,
            "prev_probability": round(prev_prob_value, 4) if prev_prob_value is not None else None,
            "meta_learner_adjusted": META_LEARNER_ENABLED and meta_learner is not None,
            "ensemble_active": ensemble_active,
        }

        signal_path = DATA_DIR / "prediction_signal.json"
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.write_text(json.dumps(signal_data, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"  Signal JSON 저장 완료: {signal_path}")
    except (OSError, TypeError, ValueError) as e:
        log(f"  Signal JSON 저장 실패: {e}")

    # ── Step 6.6: Persist probability history CSV (for dashboard) ──
    try:
        import pandas as pd

        prob_history_path = DATA_DIR / "probability_history.csv"
        today_str = signal_data["date"]
        new_row = {
            "date": today_str,
            "probability": signal_data["probability"],
            "signal": signal_data["signal"],
            "tier": signal_data["tier"],
            "tqqq_weight": signal_data["allocation"]["tqqq"],
            "spy_weight": signal_data["allocation"]["splg"],
            "cash_weight": signal_data["allocation"]["cash"],
            "vix": signal_data["indicators"]["vix"],
            "rsi": signal_data["indicators"]["rsi"],
            "adx": signal_data["indicators"]["adx"],
        }

        if prob_history_path.exists():
            hist_df = pd.read_csv(prob_history_path)
            hist_df = hist_df[hist_df["date"] != today_str]  # remove duplicate
        else:
            hist_df = pd.DataFrame()

        hist_df = pd.concat([hist_df, pd.DataFrame([new_row])], ignore_index=True)
        hist_df = hist_df.tail(500)  # keep last 500 days
        hist_df.to_csv(prob_history_path, index=False)
        log(f"  확률 히스토리 저장: {prob_history_path} ({len(hist_df)}일)")
    except Exception as e:
        log(f"  확률 히스토리 저장 실패: {e}")

    # ── Step 6.7: Append predictions JSONL log ──
    try:
        predictions_log_path = DATA_DIR / "predictions_log.jsonl"
        predictions_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(signal_data, ensure_ascii=False) + "\n")
        log(f"  예측 로그 append: {predictions_log_path}")
    except Exception as e:
        log(f"  예측 로그 저장 실패: {e}")

    elapsed = time.time() - start_time
    log(f"\n완료 (소요 시간: {elapsed:.1f}초)")


def main():
    run_daily_prediction()


if __name__ == "__main__":
    main()
