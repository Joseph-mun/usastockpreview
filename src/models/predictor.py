# -*- coding: utf-8 -*-
"""Model prediction: load models, apply calibration, compute probabilities."""

import hashlib
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    MODEL_DIR, INDEX_CONFIGS, SIGNAL_THRESHOLDS,
    PROB_CLIP_MIN, PROB_CLIP_MAX, ENSEMBLE_WEIGHTS,
)

logger = logging.getLogger(__name__)


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
        """Load all available index models from model_dir.

        Verifies SHA-256 hash from metadata before deserializing joblib files.
        """
        loaded = []
        for index_name in INDEX_CONFIGS:
            name_lower = index_name.lower()
            model_path = self.model_dir / f"{name_lower}_model.joblib"
            meta_path = self.model_dir / f"{name_lower}_meta.json"

            if not model_path.exists():
                continue

            # Load metadata first for hash verification
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            # Verify model file integrity before deserializing
            expected_hash = meta.get("model_sha256")
            if expected_hash:
                actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    logger.error(
                        "[%s] Model integrity check FAILED: expected=%s, actual=%s",
                        index_name, expected_hash[:16], actual_hash[:16],
                    )
                    continue
            else:
                logger.warning(
                    "[%s] No model_sha256 in metadata, skipping integrity check",
                    index_name,
                )

            data = joblib.load(model_path)
            self.models[index_name] = data["model"]
            self.feature_columns[index_name] = data.get("feature_columns", [])
            self.calibrators[index_name] = data.get("calibrator", None)
            self.calibration_methods[index_name] = data.get("calibration_method", None)
            self.metadata[index_name] = meta

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


class EnsemblePredictor:
    """Load and predict with LightGBM + XGBoost + CatBoost ensemble.

    Same interface as IndexPredictor for drop-in replacement.
    """

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.ensembles: dict[str, dict] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.metadata: dict[str, dict] = {}

    def load_models(self) -> list[str]:
        """Load all available ensemble models."""
        loaded = []
        for index_name in INDEX_CONFIGS:
            name_lower = index_name.lower()
            model_path = self.model_dir / f"{name_lower}_ensemble.joblib"
            meta_path = self.model_dir / f"{name_lower}_ensemble_meta.json"

            if not model_path.exists():
                continue

            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            expected_hash = meta.get("model_sha256")
            if expected_hash:
                actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    logger.error(
                        "[%s] Ensemble integrity check FAILED", index_name,
                    )
                    continue

            data = joblib.load(model_path)
            self.ensembles[index_name] = {
                "models": data["models"],
                "calibrators": data.get("calibrators", {}),
                "calibration_methods": data.get("calibration_methods", {}),
            }
            self.feature_columns[index_name] = data.get("feature_columns", [])
            self.metadata[index_name] = meta
            loaded.append(index_name)

        return loaded

    def _calibrate_single(self, calibrator, method: str | None, raw_prob: float) -> float:
        """Apply calibration to a single probability value."""
        if calibrator is None:
            return np.clip(raw_prob, PROB_CLIP_MIN, PROB_CLIP_MAX)
        if method == "isotonic":
            cal = float(calibrator.predict([raw_prob])[0])
        else:
            cal = float(calibrator.predict_proba([[raw_prob]])[0][1])
        return np.clip(cal, PROB_CLIP_MIN, PROB_CLIP_MAX)

    def _calibrate_array(self, calibrator, method: str | None, raw_probs: np.ndarray) -> np.ndarray:
        """Apply calibration to array of probabilities."""
        if calibrator is None:
            return np.clip(raw_probs, PROB_CLIP_MIN, PROB_CLIP_MAX)
        if method == "isotonic":
            cal = calibrator.predict(raw_probs)
        else:
            cal = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        return np.clip(cal, PROB_CLIP_MIN, PROB_CLIP_MAX)

    def predict_current(self, index_name: str, features: pd.Series) -> float | None:
        """Predict with weighted ensemble of calibrated probabilities."""
        if index_name not in self.ensembles:
            return None

        ens = self.ensembles[index_name]
        cols = self.feature_columns.get(index_name, [])

        if cols:
            aligned = features.reindex(cols).fillna(0)
            row = aligned.values.reshape(1, -1)
        else:
            row = features.values.reshape(1, -1)

        weighted_prob = 0.0
        total_weight = 0.0

        for model_name, model in ens["models"].items():
            try:
                raw_prob = float(model.predict_proba(row)[0][1])
                calibrator = ens["calibrators"].get(model_name)
                method = ens["calibration_methods"].get(model_name)
                cal_prob = self._calibrate_single(calibrator, method, raw_prob)

                weight = ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(ens["models"]))
                weighted_prob += cal_prob * weight
                total_weight += weight
            except (ValueError, IndexError) as e:
                logger.warning("Ensemble %s predict error: %s", model_name, e)
                continue

        if total_weight <= 0:
            return None

        return float(np.clip(weighted_prob / total_weight, PROB_CLIP_MIN, PROB_CLIP_MAX))

    def predict_history(
        self,
        index_name: str,
        X: pd.DataFrame,
        days: int = 500,
    ) -> pd.DataFrame | None:
        """Compute ensemble probability history for the last N days."""
        if index_name not in self.ensembles:
            return None

        ens = self.ensembles[index_name]
        cols = self.feature_columns.get(index_name, [])

        n = min(days, len(X))
        if n <= 0:
            return None

        X_slice = X.iloc[-n:].copy()
        if cols:
            X_slice = X_slice.reindex(columns=cols).fillna(0)

        arr = X_slice.values
        weighted_probs = np.zeros(len(arr))
        total_weight = 0.0

        for model_name, model in ens["models"].items():
            try:
                raw_probs = model.predict_proba(arr)[:, 1]
                calibrator = ens["calibrators"].get(model_name)
                method = ens["calibration_methods"].get(model_name)
                cal_probs = self._calibrate_array(calibrator, method, raw_probs)

                weight = ENSEMBLE_WEIGHTS.get(model_name, 1.0 / len(ens["models"]))
                weighted_probs += cal_probs * weight
                total_weight += weight
            except (ValueError, IndexError) as e:
                logger.warning("Ensemble %s history error: %s", model_name, e)
                continue

        if total_weight <= 0:
            return None

        final_probs = np.clip(weighted_probs / total_weight, PROB_CLIP_MIN, PROB_CLIP_MAX)
        out = pd.DataFrame({"Probability": final_probs}, index=X_slice.index)
        out.index.name = "Date"
        return out.sort_index()

    @staticmethod
    def get_signal(probability: float) -> tuple[str, str]:
        """Convert probability to signal text and emoji."""
        return IndexPredictor.get_signal(probability)
