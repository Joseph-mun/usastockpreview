# -*- coding: utf-8 -*-
"""Model prediction: load models and compute probabilities."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_DIR, INDEX_CONFIGS, SIGNAL_THRESHOLDS


class IndexPredictor:
    """Load trained models and predict probabilities for all indices."""

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.models: dict[str, object] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.metadata: dict[str, dict] = {}

    def load_models(self) -> list[str]:
        """
        Load all available index models from model_dir.
        Returns list of loaded index names.
        """
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

            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata[index_name] = json.load(f)

            loaded.append(index_name)

        return loaded

    def predict_current(self, index_name: str, features: pd.Series) -> float | None:
        """
        Predict current probability for a single index.
        Returns probability of class 1 (rise), or None if model not loaded.
        """
        if index_name not in self.models:
            return None

        model = self.models[index_name]
        cols = self.feature_columns.get(index_name, [])

        if cols:
            aligned = features.reindex(cols).fillna(0)
            row = aligned.values.reshape(1, -1)
        else:
            row = features.values.reshape(1, -1)

        proba = model.predict_proba(row)[0]
        return float(proba[1])

    def predict_history(
        self,
        index_name: str,
        X: pd.DataFrame,
        days: int = 500,
    ) -> pd.DataFrame | None:
        """
        Compute probability history (vectorized) for the last N days.
        Returns DataFrame with 'Probability' column and DatetimeIndex.
        """
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

        proba = model.predict_proba(X_slice.values)[:, 1]
        out = pd.DataFrame({"Probability": proba}, index=X_slice.index)
        out.index.name = "Date"
        return out.sort_index()

    def predict_all(self, feature_data: dict[str, pd.Series]) -> dict[str, float]:
        """
        Predict current probability for all loaded indices.
        feature_data: {"NASDAQ": Series, "SP500": Series, "DOW": Series}
        Returns: {"NASDAQ": 0.72, "SP500": 0.65, "DOW": 0.58}
        """
        results = {}
        for index_name, features in feature_data.items():
            prob = self.predict_current(index_name, features)
            if prob is not None:
                results[index_name] = prob
        return results

    @staticmethod
    def get_signal(probability: float) -> tuple[str, str]:
        """
        Convert probability to signal text and emoji.
        Returns (signal_text, emoji).
        """
        if probability >= SIGNAL_THRESHOLDS["strong_buy"]:
            return "Strong Buy", "\U0001f7e2"
        elif probability >= SIGNAL_THRESHOLDS["buy"]:
            return "Buy", "\U0001f535"
        elif probability >= SIGNAL_THRESHOLDS["neutral"]:
            return "Neutral", "\U0001f7e1"
        else:
            return "Sell", "\U0001f534"
