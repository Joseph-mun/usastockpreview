# -*- coding: utf-8 -*-
"""Optuna hyperparameter optimization for LightGBM.

Usage:
    python scripts/optuna_optimize.py [--trials 100]

Output:
    Best parameters printed to stdout. Manually update src/config.py LGBM_PARAMS.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    INDEX_CONFIGS, CV_N_SPLITS, CV_GAP,
    FEATURE_SELECTION_ENABLED, FEATURE_IMPORTANCE_TOP_N,
)
from src.data.features import DatasetBuilder
from src.data.cache import SMACache
from src.data.collectors import SMACollector


def load_data():
    """Load dataset (same as backtest.py steps 1-2)."""
    cache = SMACache()
    raw_sma, _ = cache.load()
    sma_ratios = {}
    if raw_sma:
        sma_collector = SMACollector()
        sma_collector.raw_dataframes = raw_sma
        sma_ratios = sma_collector.compute_ratios()

    builder = DatasetBuilder(sma_ratios=sma_ratios)
    X, spy, y = builder.build("IXIC", for_prediction=False)
    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Period: {spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')}")
    print(f"Target=1 ratio: {y.mean():.3f}")
    return X, y


def select_features(X, y):
    """Select top features using baseline LightGBM (same as trainer.py Phase 1)."""
    if not FEATURE_SELECTION_ENABLED:
        return X

    baseline = lgb.LGBMClassifier(
        objective="binary", n_estimators=200, max_depth=5,
        num_leaves=31, verbosity=-1, random_state=42,
    )
    baseline.fit(X, y)
    importances = baseline.feature_importances_
    top_idx = np.argsort(importances)[::-1][:FEATURE_IMPORTANCE_TOP_N]
    selected_cols = X.columns[top_idx].tolist()
    print(f"Selected {len(selected_cols)} features: {selected_cols[:5]}...")
    return X[selected_cols]


def create_objective(X, y):
    """Create Optuna objective function with Purged TimeSeriesSplit CV."""
    tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)

    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 500,
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_split_gain": trial.suggest_float("min_split_gain", 0.001, 0.1, log=True),
            "verbosity": -1,
            "random_state": 42,
            "n_jobs": -1,
        }

        scores = []
        for train_idx, test_idx in tscv.split(X):
            # Purge: remove last CV_GAP samples from train
            if len(train_idx) > CV_GAP:
                train_idx = train_idx[:-CV_GAP]

            model = lgb.LGBMClassifier(**params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            acc = (preds == y.iloc[test_idx]).mean()
            scores.append(acc)

        return np.mean(scores)

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna LightGBM optimization")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials")
    args = parser.parse_args()

    print("=" * 60)
    print("Optuna Hyperparameter Optimization")
    print("=" * 60)

    X, y = load_data()
    X = select_features(X, y)

    objective = create_objective(X, y)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best CV accuracy: {study.best_value:.4f} ({study.best_value*100:.1f}%)")
    print(f"\nBest parameters:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n# Copy to src/config.py LGBM_PARAMS:")
    print("LGBM_PARAMS = {")
    print('    "objective": "binary",')
    print('    "boosting_type": "gbdt",')
    print('    "n_estimators": 500,')
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f'    "{k}": {v:.6f},')
        else:
            print(f'    "{k}": {v},')
    print('    "subsample": 0.8,')
    print('    "colsample_bytree": 0.8,')
    print('    "random_state": 42,')
    print('    "verbosity": -1,')
    print('    "n_jobs": -1,')
    print("}")


if __name__ == "__main__":
    main()
