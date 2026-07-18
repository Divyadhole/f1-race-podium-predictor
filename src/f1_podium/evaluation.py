"""Probability quality, discrimination, and calibration metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score


def calibration_error(labels: pd.Series, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    actual = labels.to_numpy(dtype=float)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities > lower) & (probabilities <= upper)
        if mask.any():
            error += mask.mean() * abs(actual[mask].mean() - probabilities[mask].mean())
    return float(error)


def evaluate(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    if len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities have different lengths")
    if ((probabilities <= 0) | (probabilities >= 1)).any():
        raise ValueError("Probabilities must be strictly between zero and one")
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "calibration_error": calibration_error(labels, probabilities),
    }

