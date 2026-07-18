"""Podium probability models and serializable inference bundle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from f1_podium.features import FEATURE_COLUMNS, feature_matrix


@dataclass(frozen=True)
class PodiumPrediction:
    driver_name: str
    race_name: str
    grid: int
    podium_probability: float
    model_version: str = "gradient-boosting-v1"

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "driver_name": self.driver_name,
            "race_name": self.race_name,
            "grid": self.grid,
            "podium_probability": round(self.podium_probability, 4),
            "model_version": self.model_version,
        }


@dataclass
class PodiumModel:
    estimator: object
    model_name: str

    @classmethod
    def fit_gradient_boosting(
        cls,
        train: pd.DataFrame,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 15,
        l2_regularization: float = 0.2,
    ) -> "PodiumModel":
        estimator = HistGradientBoostingClassifier(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            max_iter=180,
            random_state=42,
        ).fit(feature_matrix(train), train["podium"])
        return cls(estimator=estimator, model_name="gradient-boosting-v1")

    @classmethod
    def fit_logistic(cls, train: pd.DataFrame) -> "PodiumModel":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ]
        ).fit(feature_matrix(train), train["podium"])
        return cls(estimator=estimator, model_name="logistic-v1")

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = self.estimator.predict_proba(feature_matrix(frame))[:, 1]
        return np.clip(probabilities, 0.001, 0.999)

    def predict_one(self, driver_name: str, race_name: str, features: dict[str, float]) -> PodiumPrediction:
        missing = set(FEATURE_COLUMNS).difference(features)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        frame = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        probability = float(self.predict_proba(frame)[0])
        return PodiumPrediction(
            driver_name=driver_name,
            race_name=race_name,
            grid=int(features["grid"]),
            podium_probability=probability,
            model_version=self.model_name,
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PodiumModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("Artifact is not a PodiumModel")
        return model


def grid_baseline_probabilities(frame: pd.DataFrame) -> np.ndarray:
    """A strong transparent prior based only on starting position."""
    grid = frame["grid"].to_numpy(dtype=float)
    return np.clip(0.83 * np.exp(-0.32 * (grid - 1)), 0.02, 0.95)

