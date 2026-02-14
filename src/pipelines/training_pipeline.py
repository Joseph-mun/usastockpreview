# -*- coding: utf-8 -*-
"""Weekly training pipeline: collect data, train models, save artifacts."""

import argparse
import sys
import time
from datetime import datetime

from src.config import INDEX_CONFIGS, SMA_WINDOWS, ENSEMBLE_ENABLED, get_logger
from src.data.collectors import get_sp500_tickers, SMACollector
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.models.trainer import ModelTrainer, EnsembleTrainer


logger = get_logger(__name__)


def run_training(max_tickers: int = 500, verbose: bool = True, ensemble: bool = False):
    """
    Full training pipeline:
    1. Collect S&P 500 SMA data
    2. Build datasets for each index
    3. Train LightGBM models with TimeSeriesSplit
    4. Save models + SMA cache
    """
    start_time = time.time()

    def log(msg):
        logger.info(msg)

    # ── Step 1: S&P 500 tickers ──
    log("Step 1/5: S&P 500 티커 리스트 수집")
    tickers = get_sp500_tickers()
    log(f"  {len(tickers)}개 티커 확보")

    # ── Step 2: SMA data collection ──
    log(f"Step 2/5: SMA 데이터 수집 ({max_tickers}개 종목)")
    sma_collector = SMACollector()
    sma_collector.collect(
        tickers,
        max_tickers=max_tickers,
        status_callback=lambda msg: log(f"  {msg}") if verbose else None,
    )

    # Save SMA cache
    cache = SMACache()
    cache_path = cache.save(
        sma_collector.raw_dataframes,
        meta={"max_tickers": max_tickers, "windows": SMA_WINDOWS},
    )
    log(f"  SMA 캐시 저장: {cache_path}")

    # Compute ratios
    sma_ratios = sma_collector.compute_ratios()
    log(f"  SMA 비율 계산 완료 (keys: {list(sma_ratios.keys())})")

    # ── Step 3-4: Train models for each index ──
    all_metrics = {}
    builder = DatasetBuilder(sma_ratios=sma_ratios)

    for idx, (index_name, cfg) in enumerate(INDEX_CONFIGS.items()):
        step_num = idx + 3
        log(f"Step {step_num}/5: [{index_name}] 데이터 준비 + 모델 학습")

        use_ensemble = ensemble or ENSEMBLE_ENABLED

        try:
            # Build dataset
            ticker = cfg["ticker"]
            X, spy, y = builder.build(ticker, for_prediction=False)
            log(f"  데이터셋: {len(X)} samples, {len(X.columns)} features, Target=1 비율: {y.mean():.3f}")

            # Train single model (always, for backward compatibility)
            trainer = ModelTrainer(index_name)
            result = trainer.train(
                X, y,
                status_callback=lambda msg: log(f"  {msg}") if verbose else None,
            )
            model_path, meta_path = trainer.save()
            log(f"  단일 모델 저장: {model_path}")

            metrics = result["metrics"]
            all_metrics[index_name] = {
                "cv_accuracy_mean": metrics["cv_accuracy_mean"],
                "cv_accuracy_std": metrics["cv_accuracy_std"],
                "n_samples": metrics["n_samples"],
                "n_features": metrics["n_features"],
                "top_features": list(result["feature_importance"].keys())[:10],
            }

            # Train ensemble if enabled
            if use_ensemble:
                log(f"  [{index_name}] 앙상블 모델 학습 시작")
                ens_trainer = EnsembleTrainer(index_name)
                ens_result = ens_trainer.train(
                    X, y,
                    status_callback=lambda msg: log(f"  {msg}") if verbose else None,
                )
                ens_model_path, ens_meta_path = ens_trainer.save()
                log(f"  앙상블 모델 저장: {ens_model_path}")
                all_metrics[index_name]["ensemble_cv_score"] = ens_result["ensemble_cv_score"]
                all_metrics[index_name]["ensemble_models"] = list(ens_result["models"].keys())

        except Exception as e:
            log(f"  ERROR [{index_name}]: {e}")
            all_metrics[index_name] = {"error": str(e)}

    # ── Summary ──
    elapsed = time.time() - start_time
    log(f"\n{'='*50}")
    log(f"학습 완료 (소요 시간: {elapsed/60:.1f}분)")
    log(f"{'='*50}")
    for name, m in all_metrics.items():
        if "error" in m:
            log(f"  {name}: ERROR - {m['error']}")
        else:
            log(f"  {name}: CV accuracy = {m['cv_accuracy_mean']:.4f} +/- {m['cv_accuracy_std']:.4f}")
            log(f"    Top features: {m['top_features'][:5]}")

    return all_metrics, cache_path


def main():
    parser = argparse.ArgumentParser(description="Weekly model training pipeline")
    parser.add_argument("--max-tickers", type=int, default=500, help="Max S&P 500 stocks to analyze")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--ensemble", action="store_true", help="Train ensemble (LightGBM + XGBoost + CatBoost)")
    args = parser.parse_args()

    metrics, cache_path = run_training(
        max_tickers=args.max_tickers,
        verbose=not args.quiet,
        ensemble=args.ensemble,
    )

    # Send Telegram notification if configured
    try:
        from src.config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
            from src.notification.telegram_bot import TelegramNotifier
            notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            summary_lines = ["<b>Weekly Training Complete</b>\n"]
            for name, m in metrics.items():
                if "error" in m:
                    summary_lines.append(f"  {name}: ERROR")
                else:
                    summary_lines.append(f"  {name}: {m['cv_accuracy_mean']:.4f}")
            notifier.send_text("\n".join(summary_lines))
    except Exception as e:
        logger.error("Telegram notification failed: %s", e)


if __name__ == "__main__":
    main()
