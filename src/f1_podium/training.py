"""Training, temporal tuning, evaluation, and artifact persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from f1_podium.config import SETTINGS, Settings
from f1_podium.data import load_tables
from f1_podium.evaluation import evaluate
from f1_podium.features import build_feature_table, chronological_split
from f1_podium.model import PodiumModel, grid_baseline_probabilities


def tune_gradient(train):
    validation_year = int(train["year"].max())
    fit = train.loc[train["year"] < validation_year]
    validation = train.loc[train["year"] == validation_year]
    candidates = [
        {"learning_rate": 0.04, "max_leaf_nodes": 7, "l2_regularization": 0.5},
        {"learning_rate": 0.06, "max_leaf_nodes": 15, "l2_regularization": 0.5},
        {"learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    ]
    scores = []
    for params in candidates:
        model = PodiumModel.fit_gradient_boosting(fit, **params)
        score = evaluate(validation["podium"], model.predict_proba(validation))["log_loss"]
        scores.append({"parameters": params, "validation_log_loss": score})
    best = min(scores, key=lambda item: item["validation_log_loss"])
    return best["parameters"], scores


def train_and_evaluate(settings: Settings = SETTINGS) -> dict[str, object]:
    tables = load_tables(settings.raw_dir)
    frame = build_feature_table(tables)
    train, test = chronological_split(frame, settings.test_start_year)
    best_parameters, tuning = tune_gradient(train)
    gradient = PodiumModel.fit_gradient_boosting(train, **best_parameters)
    logistic = PodiumModel.fit_logistic(train)

    metrics = {
        "gradient_boosting": evaluate(test["podium"], gradient.predict_proba(test)),
        "logistic_regression": evaluate(test["podium"], logistic.predict_proba(test)),
        "grid_baseline": evaluate(test["podium"], grid_baseline_probabilities(test)),
    }
    settings.processed_path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(date=frame["date"].dt.date).to_csv(settings.processed_path, index=False)
    gradient.save(settings.model_path)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(),
        "seed": settings.seed,
        "rows": len(frame),
        "races": int(frame["race_id"].nunique()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "selected_parameters": best_parameters,
        "tuning": tuning,
        "metrics": metrics,
    }
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata

