# -*- coding: utf-8 -*-
"""Model prediction: load models, apply calibration, compute probabilities."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, INDEX_CONFIGS, SIGNAL_THRESHOLDS, PROB_CLIP_MIN, PROB_CLIP_MAX


class IndexPredictor:
    """Load trained models and predict calibrated probabilities."""

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.models: dict[str, object] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.metadata: dict[str, dict] = {}
        self.calibrators: dict[str, object] = {}
        self.calibration_methods: dict[str, str] = {}

    def load_models(self) -> list[str]:
        """Load all available index models from model_dir."""
        loaded = []
        for index_name in INDEX_CONFIGS:
            name_lower = index_name.lower()
            model_path = self.model_dir / f"{name_lower}_model.joblib"
            meta_path = self.model_dir / f"{name_lower}_meta.json"

            if not model_path.exists():
                continue

            data = joblib.load(model_path)
            self.models[index_name] = data["model"]
            self.feature_columns[index_name] = data.get("feature_columns", [])
            self.calibrators[index_name] = data.get("calibrator", None)
            self.calibration_methods[index_name] = data.get("calibration_method", None)

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata[index_name] = json.load(f)

            loaded.append(index_name)

        return loaded

    def _calibrate(self, index_name: str, raw_prob: float) -> float:
        """Apply calibration to raw probability, then clip to safe range."""
        calibrator = self.calibrators.get(index_name)
        if calibrator is None:
            return np.clip(raw_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)

        method = self.calibration_methods.get(index_name)
        if method == "isotonic":
            cal = float(calibrator.predict([raw_prob])[0])
        else:
            cal = float(calibrator.predict_proba([[raw_prob]])[0][1])

        return np.clip(cal, PROB_CLIP_MIN, PROB_CLIP_MAX)

    def _calibrate_array(self, index_name: str, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to array of probabilities, then clip to safe range."""
        calibrator = self.calibrators.get(index_name)
        if calibrator is None:
            return np.clip(raw_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)

        method = self.calibration_methods.get(index_name)
        if method == "isotonic":
            cal = calibrator.predict(raw_probs)
        else:
            cal = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

        return np.clip(cal, PROB_CLIP_MIN, PROB_CLIP_MAX)

    def predict_current(self, index_name: str, features: pd.Series) -> float | None:
        """Predict calibrated probability for a single index."""
        if index_name not in self.models:
            return None

        model = self.models[index_name]
        cols = self.feature_columns.get(index_name, [])

        if cols:
            aligned = features.reindex(cols).fillna(0)
            row = aligned.values.reshape(1, -1)
        else:
            row = features.values.reshape(1, -1)

        raw_prob = float(model.predict_proba(row)[0][1])
        return self._calibrate(index_name, raw_prob)

    def predict_history(
        self,
        index_name: str,
        X: pd.DataFrame,
        days: int = 500,
    ) -> pd.DataFrame | None:
        """Compute calibrated probability history for the last N days."""
        if index_name not in self.models:
            return None

        model = self.models[index_name]
        cols = self.feature_columns.get(index_name, [])

        n = min(days, len(X))
        if n <= 0:
            return None

        X_slice = X.iloc[-n:].copy()
        if cols:
            X_slice = X_slice.reindex(columns=cols).fillna(0)

        raw_probs = model.predict_proba(X_slice.values)[:, 1]
        cal_probs = self._calibrate_array(index_name, raw_probs)

        out = pd.DataFrame({"Probability": cal_probs}, index=X_slice.index)
        out.index.name = "Date"
        return out.sort_index()

    def predict_all(self, feature_data: dict[str, pd.Series]) -> dict[str, float]:
        """Predict calibrated probability for all loaded indices."""
        results = {}
        for index_name, features in feature_data.items():
            prob = self.predict_current(index_name, features)
            if prob is not None:
                results[index_name] = prob
        return results

    @staticmethod
    def get_signal(probability: float) -> tuple[str, str]:
        """Convert probability to signal text and emoji."""
        if probability >= SIGNAL_THRESHOLDS["strong_buy"]:
            return "Strong Buy", "\U0001f7e2"
        elif probability >= SIGNAL_THRESHOLDS["buy"]:
            return "Buy", "\U0001f535"
        elif probability >= SIGNAL_THRESHOLDS["neutral"]:
            return "Neutral", "\U0001f7e1"
        else:
            return "Sell", "\U0001f534"
