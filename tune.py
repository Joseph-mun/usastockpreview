# -*- coding: utf-8 -*-
"""Hyperparameter tuning with Optuna + target lookahead experiment.

Usage:
  python tune.py                    # default: 80 trials
  python tune.py --n-trials 120     # custom trial count
  python tune.py --lookahead-only   # only test different lookahead values
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import lightgbm as lgb
import optuna

from src.config import get_logger

logger = get_logger(__name__)
from sklearn.model_selection import TimeSeriesSplit

import src.config as config
from src.data.collectors import get_sp500_tickers, SMACollector
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.models.trainer import _split_train_val


optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_dataset(sma_ratios, lookahead_days=None):
    """Build feature matrix. Optionally override target lookahead."""
    original = config.TARGET_LOOKAHEAD_DAYS
    if lookahead_days is not None:
        config.TARGET_LOOKAHEAD_DAYS = lookahead_days
    try:
        builder = DatasetBuilder(sma_ratios=sma_ratios)
        X, spy, y = builder.build("IXIC", for_prediction=False)
        return X, y
    finally:
        config.TARGET_LOOKAHEAD_DAYS = original


def cv_score(X_arr, y_arr, params, n_splits=5, gap=20, top_n=30):
    """Run TimeSeriesSplit CV and return mean accuracy."""
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []

    # Feature selection: quick model on all data
    if top_n and X_arr.shape[1] > top_n:
        quick = lgb.LGBMClassifier(**params)
        quick.fit(X_arr, y_arr)
        top_idx = np.argsort(quick.feature_importances_)[::-1][:top_n]
        X_arr = X_arr[:, top_idx]

    for train_idx, test_idx in tscv.split(X_arr):
        pure_idx, val_idx = _split_train_val(train_idx)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_arr[pure_idx], y_arr[pure_idx],
            eval_set=[(X_arr[val_idx], y_arr[val_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        scores.append(model.score(X_arr[test_idx], y_arr[test_idx]))

    return float(np.mean(scores)), float(np.std(scores))


def make_objective(X_arr, y_arr, gap):
    """Create Optuna objective function."""

    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 80, step=10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        top_n = trial.suggest_int("top_n_features", 20, 40, step=5)
        mean_acc, _ = cv_score(X_arr, y_arr, params, gap=gap, top_n=top_n)
        return mean_acc

    return objective


def run_lookahead_experiment(sma_ratios, params, candidates=(5, 10, 15, 20)):
    """Test different target lookahead values."""
    logger.info("\n" + "=" * 60)
    logger.info("  Target Lookahead Experiment")
    logger.info("=" * 60)

    results = []
    for days in candidates:
        X, y = build_dataset(sma_ratios, lookahead_days=days)
        X_arr = X.values.astype(np.float64)
        y_arr = y.values.astype(int)
        base_rate = float(y_arr.mean())
        mean_acc, std_acc = cv_score(X_arr, y_arr, params, gap=days, top_n=30)
        edge = mean_acc - base_rate
        results.append((days, mean_acc, std_acc, base_rate, edge, len(y)))
        logger.info(
            "  Lookahead=%2dd | CV=%.4f±%.4f | base=%.3f | edge=%+.4f | samples=%d",
            days, mean_acc, std_acc, base_rate, edge, len(y),
        )

    # Best by edge over base rate
    best = max(results, key=lambda r: r[4])
    logger.info("  Best: %dd (edge=%+.4f)", best[0], best[4])
    return results


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--n-trials", type=int, default=80)
    parser.add_argument("--lookahead-only", action="store_true")
    args = parser.parse_args()

    start = time.time()

    # Load SMA data
    logger.info("1. SMA 캐시 로드...")
    cache = SMACache()
    raw_sma, _ = cache.load()
    sma_ratios = {}
    if raw_sma:
        collector = SMACollector()
        collector.raw_dataframes = raw_sma
        sma_ratios = collector.compute_ratios()
        logger.info("   SMA 캐시 로드 완료")
    else:
        logger.warning("   SMA 캐시 없음 - SMA 비율 기본값 사용")

    # Build dataset (default lookahead)
    logger.info("2. 데이터셋 구축...")
    X, y = build_dataset(sma_ratios)
    X_arr = X.values.astype(np.float64)
    y_arr = y.values.astype(int)
    logger.info("   %d samples, %d features, base_rate=%.3f", len(X), X.shape[1], y_arr.mean())

    # Current params baseline
    logger.info("3. 현재 파라미터 기준선...")
    current_params = config.LGBM_PARAMS.copy()
    gap = config.TARGET_LOOKAHEAD_DAYS
    base_mean, base_std = cv_score(X_arr, y_arr, current_params, gap=gap, top_n=30)
    logger.info("   Current CV: %.4f ± %.4f", base_mean, base_std)

    if not args.lookahead_only:
        # Optuna tuning
        logger.info("4. Optuna 하이퍼파라미터 탐색 (%d trials)...", args.n_trials)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            make_objective(X_arr, y_arr, gap),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

        best = study.best_trial
        logger.info("   Best CV accuracy: %.4f", best.value)
        logger.info("   Best params:")
        for k, v in sorted(best.params.items()):
            logger.info("     %s: %s", k, v)

        # Compare
        delta = best.value - base_mean
        logger.info("   Improvement over current: %+.4f (%+.2f%%p)", delta, delta * 100)

        # Top 5 trials
        logger.info("   Top 5 trials:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
        for i, t in enumerate(sorted_trials[:5]):
            logger.info(
                "     #%d CV=%.4f | depth=%s leaves=%s lr=%.4f top_n=%s",
                i + 1, t.value, t.params.get("max_depth"), t.params.get("num_leaves"),
                t.params.get("learning_rate", 0), t.params.get("top_n_features"),
            )

        # Build best params dict for config.py
        bp = best.params.copy()
        top_n = bp.pop("top_n_features")
        best_params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": -1,
        }
        best_params.update(bp)

        logger.info("   Suggested config.py update:")
        logger.info("   LGBM_PARAMS = %s", best_params)
        logger.info("   FEATURE_IMPORTANCE_TOP_N = %d", top_n)
    else:
        best_params = current_params

    # Lookahead experiment
    logger.info("5. Target Lookahead 실험...")
    run_lookahead_experiment(sma_ratios, best_params)

    elapsed = time.time() - start
    logger.info("완료 (소요: %.1f분)", elapsed / 60)


if __name__ == "__main__":
    main()
