# -*- coding: utf-8 -*-
"""Model training pipeline with LightGBM and TimeSeriesSplit."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import LGBM_PARAMS, CV_N_SPLITS, CV_GAP, MODEL_DIR


class ModelTrainer:
    """Train LightGBM classifier with time-series cross-validation."""

    def __init__(self, index_name: str, params: dict = None):
        self.index_name = index_name
        self.params = params or LGBM_PARAMS.copy()
        self.model = None
        self.feature_columns: list[str] = []
        self.cv_scores: list[float] = []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback=None,
        status_callback=None,
    ) -> dict:
        """
        Train model with TimeSeriesSplit cross-validation.

        Returns dict with model, metrics, and feature importance.
        """
        if status_callback:
            status_callback(f"[{self.index_name}] 학습 시작 (데이터: {len(X)} rows, {len(X.columns)} features)")

        self.feature_columns = X.columns.tolist()
        X_arr = X.values.astype(np.float64)
        y_arr = y.values.astype(int)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)
        self.cv_scores = []
        fold_details = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            fold_model = lgb.LGBMClassifier(**self.params)
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            score = fold_model.score(X_test, y_test)
            self.cv_scores.append(score)

            fold_details.append({
                "fold": fold + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": round(score, 4),
                "train_end_idx": int(train_idx[-1]),
                "test_start_idx": int(test_idx[0]),
            })

            if status_callback:
                status_callback(f"  Fold {fold + 1}/{CV_N_SPLITS}: accuracy={score:.4f}")
            if progress_callback:
                progress_callback((fold + 1) / (CV_N_SPLITS + 1))

        # Final model on full data
        if status_callback:
            status_callback(f"[{self.index_name}] 전체 데이터로 최종 모델 학습 중...")

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_arr, y_arr)

        if progress_callback:
            progress_callback(1.0)

        # Feature importance
        importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_.tolist(),
        ))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        metrics = {
            "index_name": self.index_name,
            "cv_accuracy_mean": round(float(np.mean(self.cv_scores)), 4),
            "cv_accuracy_std": round(float(np.std(self.cv_scores)), 4),
            "cv_scores": [round(s, 4) for s in self.cv_scores],
            "fold_details": fold_details,
            "n_samples": len(X),
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "trained_at": datetime.now().isoformat(),
            "params": self.params,
        }

        if status_callback:
            mean_acc = metrics["cv_accuracy_mean"]
            std_acc = metrics["cv_accuracy_std"]
            status_callback(f"[{self.index_name}] 완료: CV accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")

        return {
            "model": self.model,
            "metrics": metrics,
            "feature_importance": importance_sorted,
        }

    def save(self, model_dir: str = None) -> tuple[str, str]:
        """
        Save model (joblib) and metadata (JSON) to model_dir.
        Returns (model_path, meta_path).
        """
        model_dir = Path(model_dir) if model_dir else MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        name_lower = self.index_name.lower()
        model_path = model_dir / f"{name_lower}_model.joblib"
        meta_path = model_dir / f"{name_lower}_meta.json"

        if self.model is None:
            raise ValueError("No model to save. Train first.")

        # Save model
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
        }, model_path)

        # Save metadata
        meta = {
            "index_name": self.index_name,
            "feature_columns": self.feature_columns,
            "cv_accuracy_mean": round(float(np.mean(self.cv_scores)), 4) if self.cv_scores else None,
            "cv_accuracy_std": round(float(np.std(self.cv_scores)), 4) if self.cv_scores else None,
            "n_features": len(self.feature_columns),
            "saved_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return str(model_path), str(meta_path)
