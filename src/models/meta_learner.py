# -*- coding: utf-8 -*-
"""Incremental MLP meta-learner: adjusts LightGBM probabilities using recent outcomes.

Uses sklearn's MLPClassifier with partial_fit() for daily incremental updates.
No additional dependencies required (sklearn already in requirements).

Design:
  - Input: [lgbm_prob, prob_lag1..5, actual_lag1..5, rsi, vix] (~12 features)
  - Architecture: 12 -> 32 -> 16 -> 1 (sigmoid)
  - Daily update: when 5-day outcome is confirmed, call partial_fit() with 1 sample
  - State: model saved as joblib, training history as CSV
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from src.config import MODEL_DIR, META_LEARNER_HIDDEN, META_LEARNER_LR


class MetaLearner:
    """Incremental MLP that adjusts LightGBM probabilities."""

    FEATURE_NAMES = [
        "lgbm_prob",
        "prob_lag1", "prob_lag2", "prob_lag3", "prob_lag4", "prob_lag5",
        "actual_lag1", "actual_lag2", "actual_lag3", "actual_lag4", "actual_lag5",
        "rsi",
    ]

    def __init__(self, index_name: str = "NASDAQ"):
        self.index_name = index_name
        self.model = MLPClassifier(
            hidden_layer_sizes=META_LEARNER_HIDDEN,
            activation="relu",
            solver="adam",
            learning_rate_init=META_LEARNER_LR,
            max_iter=100,
            warm_start=True,
            random_state=42,
        )
        self.is_fitted = False
        self.history: list[dict] = []
        self._model_dir = MODEL_DIR

    @property
    def model_path(self) -> Path:
        return self._model_dir / f"{self.index_name.lower()}_meta_learner.joblib"

    @property
    def history_path(self) -> Path:
        return self._model_dir / f"{self.index_name.lower()}_meta_history.json"

    def build_features(
        self,
        lgbm_prob: float,
        recent_probs: list[float],
        recent_actuals: list[float],
        rsi: float = 50.0,
    ) -> np.ndarray:
        """Build feature vector for meta-learner prediction."""
        probs = (recent_probs + [0.5] * 5)[:5]
        actuals = (recent_actuals + [0.5] * 5)[:5]
        return np.array([lgbm_prob] + probs + actuals + [rsi / 100.0])

    def predict(self, features: np.ndarray) -> float:
        """Predict adjusted probability. Falls back to raw lgbm_prob if not fitted."""
        if not self.is_fitted:
            return float(features[0])  # fallback to raw lgbm probability

        try:
            prob = self.model.predict_proba(features.reshape(1, -1))[0]
            return float(prob[1]) if len(prob) > 1 else float(prob[0])
        except Exception:
            return float(features[0])

    def update(self, features: np.ndarray, actual: int):
        """Incrementally update with one confirmed outcome."""
        X = features.reshape(1, -1)
        y = np.array([actual])

        if not self.is_fitted:
            # First fit needs both classes
            self.model.partial_fit(X, y, classes=[0, 1])
            self.is_fitted = True
        else:
            self.model.partial_fit(X, y)

        self.history.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "features": features.tolist(),
            "actual": actual,
        })

    def save(self):
        """Save meta-learner model and history."""
        self._model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "model": self.model,
            "is_fitted": self.is_fitted,
        }, self.model_path)

        # Keep only last 500 history entries
        recent = self.history[-500:]
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(recent, f)

    def load(self) -> bool:
        """Load meta-learner model. Returns True if loaded successfully."""
        if not self.model_path.exists():
            return False

        try:
            data = joblib.load(self.model_path)
            self.model = data["model"]
            self.is_fitted = data.get("is_fitted", False)

            if self.history_path.exists():
                with open(self.history_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)

            return True
        except Exception:
            return False

    def get_recent_actuals(self, n: int = 5) -> list[float]:
        """Get recent actual outcomes from history."""
        if not self.history:
            return []
        recent = self.history[-n:]
        return [h["actual"] for h in recent]

    def get_recent_probs(self, n: int = 5) -> list[float]:
        """Get recent lgbm probabilities from history."""
        if not self.history:
            return []
        recent = self.history[-n:]
        return [h["features"][0] for h in recent]
