# -*- coding: utf-8 -*-
"""Model training pipeline with LightGBM, calibration, and feature selection."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config import (
    LGBM_PARAMS, CV_N_SPLITS, CV_GAP, MODEL_DIR,
    CALIBRATION_ENABLED, CALIBRATION_METHOD,
    FEATURE_SELECTION_ENABLED, FEATURE_IMPORTANCE_TOP_N,
)


class ModelTrainer:
    """Train LightGBM classifier with calibration and feature selection."""

    def __init__(self, index_name: str, params: dict = None):
        self.index_name = index_name
        self.params = params or LGBM_PARAMS.copy()
        self.model = None
        self.feature_columns: list[str] = []
        self.cv_scores: list[float] = []
        self.calibrator = None
        self.calibration_method = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        progress_callback=None,
        status_callback=None,
    ) -> dict:
        """Train model with TimeSeriesSplit CV, calibration, and feature selection."""
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
            })

            if status_callback:
                status_callback(f"  Fold {fold + 1}/{CV_N_SPLITS}: accuracy={score:.4f}")
            if progress_callback:
                progress_callback((fold + 1) / (CV_N_SPLITS + 2))

        # Calibration: fit on last fold's held-out predictions
        last_train_idx, last_test_idx = list(tscv.split(X_arr))[-1]
        if CALIBRATION_ENABLED:
            if status_callback:
                status_callback(f"[{self.index_name}] 확률 캘리브레이션 학습 중...")

            X_cal_train = X_arr[last_train_idx]
            X_cal_test = X_arr[last_test_idx]
            y_cal_train = y_arr[last_train_idx]
            y_cal_test = y_arr[last_test_idx]

            cal_model = lgb.LGBMClassifier(**self.params)
            cal_model.fit(
                X_cal_train, y_cal_train,
                eval_set=[(X_cal_test, y_cal_test)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
            )
            raw_probs = cal_model.predict_proba(X_cal_test)[:, 1]

            if CALIBRATION_METHOD == "isotonic":
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(raw_probs, y_cal_test)
            else:
                self.calibrator = LogisticRegression()
                self.calibrator.fit(raw_probs.reshape(-1, 1), y_cal_test)
            self.calibration_method = CALIBRATION_METHOD

            if status_callback:
                status_callback(f"  캘리브레이션 완료 (method={CALIBRATION_METHOD}, samples={len(y_cal_test)})")

        # Final model on full data
        if status_callback:
            status_callback(f"[{self.index_name}] 전체 데이터로 최종 모델 학습 중...")

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_arr, y_arr)

        # Feature selection: retrain with top features only
        if FEATURE_SELECTION_ENABLED and len(self.feature_columns) > FEATURE_IMPORTANCE_TOP_N:
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:FEATURE_IMPORTANCE_TOP_N]
            selected = [self.feature_columns[i] for i in top_indices]

            if status_callback:
                status_callback(f"[{self.index_name}] 피처 선택: {len(self.feature_columns)} -> {len(selected)}")

            X_selected = X_arr[:, top_indices]
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(X_selected, y_arr)
            self.feature_columns = selected

            # Re-calibrate with selected features
            if CALIBRATION_ENABLED and self.calibrator is not None:
                y_cal_train = y_arr[last_train_idx]
                y_cal_test = y_arr[last_test_idx]
                cal_model2 = lgb.LGBMClassifier(**self.params)
                X_cal_train_sel = X_arr[last_train_idx][:, top_indices]
                X_cal_test_sel = X_arr[last_test_idx][:, top_indices]
                cal_model2.fit(
                    X_cal_train_sel, y_cal_train,
                    eval_set=[(X_cal_test_sel, y_cal_test)],
                    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
                )
                raw_probs2 = cal_model2.predict_proba(X_cal_test_sel)[:, 1]
                if CALIBRATION_METHOD == "isotonic":
                    self.calibrator = IsotonicRegression(out_of_bounds="clip")
                    self.calibrator.fit(raw_probs2, y_cal_test)
                else:
                    self.calibrator = LogisticRegression()
                    self.calibrator.fit(raw_probs2.reshape(-1, 1), y_cal_test)

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
            "calibration_enabled": CALIBRATION_ENABLED,
            "feature_selection": FEATURE_SELECTION_ENABLED,
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
        """Save model, calibrator, and metadata."""
        model_dir = Path(model_dir) if model_dir else MODEL_DIR
        model_dir.mkdir(parents=True, exist_ok=True)

        name_lower = self.index_name.lower()
        model_path = model_dir / f"{name_lower}_model.joblib"
        meta_path = model_dir / f"{name_lower}_meta.json"

        if self.model is None:
            raise ValueError("No model to save. Train first.")

        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "calibrator": self.calibrator,
            "calibration_method": self.calibration_method,
        }, model_path)

        meta = {
            "index_name": self.index_name,
            "feature_columns": self.feature_columns,
            "cv_accuracy_mean": round(float(np.mean(self.cv_scores)), 4) if self.cv_scores else None,
            "cv_accuracy_std": round(float(np.std(self.cv_scores)), 4) if self.cv_scores else None,
            "n_features": len(self.feature_columns),
            "calibration_method": self.calibration_method,
            "saved_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return str(model_path), str(meta_path)
