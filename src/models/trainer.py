# -*- coding: utf-8 -*-
"""Model training pipeline with LightGBM, calibration, and feature selection.

Phase 1 improvements (overfitting reduction):
  - Calibration: OOF (out-of-fold) predictions from ALL folds instead of last fold only
  - Early stopping: separate validation split from train data (not test set)
  - Feature selection: averaged CV importances across all folds
"""

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

# Early stopping validation split ratio (from end of training set)
_VAL_RATIO = 0.1
_VAL_MIN_SAMPLES = 20


def _split_train_val(train_idx):
    """Split train indices into pure-train and validation for early stopping."""
    val_size = max(int(len(train_idx) * _VAL_RATIO), _VAL_MIN_SAMPLES)
    return train_idx[:-val_size], train_idx[-val_size:]


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
        """Train model with TimeSeriesSplit CV, calibration, and feature selection.

        Pipeline:
          1. CV on all features → accuracy scores + averaged feature importances
          2. Feature selection from averaged importances
          3. CV on selected features → OOF predictions for calibration
          4. Fit calibrator on OOF predictions
          5. Train final model on all data with selected features
        """
        if status_callback:
            status_callback(
                f"[{self.index_name}] 학습 시작 "
                f"(데이터: {len(X)} rows, {len(X.columns)} features)"
            )

        all_feature_columns = X.columns.tolist()
        self.feature_columns = all_feature_columns
        X_arr = X.values.astype(np.float64)
        y_arr = y.values.astype(int)

        tscv = TimeSeriesSplit(n_splits=CV_N_SPLITS, gap=CV_GAP)
        self.cv_scores = []
        fold_details = []

        need_fs = (
            FEATURE_SELECTION_ENABLED
            and len(all_feature_columns) > FEATURE_IMPORTANCE_TOP_N
        )

        # Total progress steps: phase1 CV + (phase3 CV if fs) + final train
        total_steps = CV_N_SPLITS + (CV_N_SPLITS if need_fs else 0) + 1
        step = 0

        # ── Phase 1: CV → accuracy + feature importances ──
        accumulated_importances = np.zeros(X_arr.shape[1])
        # If no feature selection, collect OOF here directly
        oof_probs = None if need_fs else np.full(len(y_arr), np.nan)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_arr)):
            pure_train_idx, val_idx = _split_train_val(train_idx)

            fold_model = lgb.LGBMClassifier(**self.params)
            fold_model.fit(
                X_arr[pure_train_idx], y_arr[pure_train_idx],
                eval_set=[(X_arr[val_idx], y_arr[val_idx])],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )

            score = fold_model.score(X_arr[test_idx], y_arr[test_idx])
            self.cv_scores.append(score)
            accumulated_importances += fold_model.feature_importances_

            if oof_probs is not None:
                oof_probs[test_idx] = fold_model.predict_proba(
                    X_arr[test_idx]
                )[:, 1]

            fold_details.append({
                "fold": fold + 1,
                "train_size": len(pure_train_idx),
                "val_size": len(val_idx),
                "test_size": len(test_idx),
                "accuracy": round(score, 4),
            })

            step += 1
            if status_callback:
                status_callback(
                    f"  Fold {fold + 1}/{CV_N_SPLITS}: accuracy={score:.4f}"
                )
            if progress_callback:
                progress_callback(step / total_steps)

        # ── Phase 2: Feature selection (averaged CV importances) ──
        if need_fs:
            avg_importances = accumulated_importances / CV_N_SPLITS
            top_indices = np.argsort(avg_importances)[::-1][
                :FEATURE_IMPORTANCE_TOP_N
            ]
            selected_cols = [all_feature_columns[i] for i in top_indices]
            X_sel = X_arr[:, top_indices]

            if status_callback:
                status_callback(
                    f"[{self.index_name}] 피처 선택: "
                    f"{len(all_feature_columns)} -> {len(selected_cols)}"
                )
            self.feature_columns = selected_cols

            # ── Phase 3: Second CV on selected features → OOF for calibration ──
            oof_probs = np.full(len(y_arr), np.nan)

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sel)):
                pure_train_idx, val_idx = _split_train_val(train_idx)

                m = lgb.LGBMClassifier(**self.params)
                m.fit(
                    X_sel[pure_train_idx], y_arr[pure_train_idx],
                    eval_set=[(X_sel[val_idx], y_arr[val_idx])],
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        lgb.log_evaluation(0),
                    ],
                )
                oof_probs[test_idx] = m.predict_proba(X_sel[test_idx])[:, 1]

                step += 1
                if progress_callback:
                    progress_callback(step / total_steps)
        else:
            X_sel = X_arr

        # ── Phase 4: Calibration from OOF predictions ──
        if CALIBRATION_ENABLED:
            if status_callback:
                status_callback(
                    f"[{self.index_name}] 확률 캘리브레이션 학습 중..."
                )

            valid = ~np.isnan(oof_probs)
            n_cal = int(valid.sum())

            if CALIBRATION_METHOD == "isotonic":
                self.calibrator = IsotonicRegression(out_of_bounds="clip")
                self.calibrator.fit(oof_probs[valid], y_arr[valid])
            else:
                self.calibrator = LogisticRegression()
                self.calibrator.fit(
                    oof_probs[valid].reshape(-1, 1), y_arr[valid]
                )
            self.calibration_method = CALIBRATION_METHOD

            if status_callback:
                status_callback(
                    f"  캘리브레이션 완료 "
                    f"(method={CALIBRATION_METHOD}, OOF samples={n_cal})"
                )

        # ── Phase 5: Final model on all data (selected features) ──
        if status_callback:
            status_callback(
                f"[{self.index_name}] 전체 데이터로 최종 모델 학습 중..."
            )

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X_sel, y_arr)

        step += 1
        if progress_callback:
            progress_callback(1.0)

        # Feature importance (from final model)
        importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_.tolist(),
        ))
        importance_sorted = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

        mean_acc = round(float(np.mean(self.cv_scores)), 4)
        std_acc = round(float(np.std(self.cv_scores)), 4)

        metrics = {
            "index_name": self.index_name,
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": std_acc,
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
            status_callback(
                f"[{self.index_name}] 완료: "
                f"CV accuracy = {mean_acc:.4f} +/- {std_acc:.4f}"
            )

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
