"""Publication-quality EDA and evaluation report generation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from f1_podium.config import ROOT, SETTINGS
from f1_podium.data import load_tables
from f1_podium.features import build_feature_table, chronological_split
from f1_podium.model import PodiumModel


GREEN = "#12624f"
GOLD = "#d6a84b"
BLUE = "#3f6f8f"


def generate_reports(output_dir: Path = ROOT / "reports" / "figures") -> list[Path]:
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_feature_table(load_tables(SETTINGS.raw_dir))
    _, test = chronological_split(frame, SETTINGS.test_start_year)
    model = PodiumModel.load(SETTINGS.model_path)
    probabilities = model.predict_proba(test)
    return [
        _overview(frame, output_dir / "data-overview.png"),
        _grid_performance(frame, output_dir / "grid-vs-podium.png"),
        _calibration(test["podium"], probabilities, output_dir / "calibration.png"),
        _latest_forecast(frame, model, output_dir / "latest-race-forecast.png"),
    ]


def _overview(frame: pd.DataFrame, destination: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    races = frame.drop_duplicates("race_id").groupby("year").size()
    axes[0].bar(races.index.astype(str), races.values, color=GREEN)
    axes[0].set(title="Race coverage by season", xlabel="Season", ylabel="Races")
    points = frame.groupby("constructor_name")["points"].sum().sort_values()
    axes[1].barh(points.index, points.values, color=BLUE)
    axes[1].set(title="Constructor points in sample", xlabel="Points", ylabel="")
    fig.suptitle("Formula 1 relational sample | 2017-2025", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def _grid_performance(frame: pd.DataFrame, destination: Path) -> Path:
    grid = frame.groupby("grid")["podium"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(grid.index, grid.values, color=GREEN)
    ax.set(title="Observed podium rate by starting grid", xlabel="Grid position", ylabel="Podium rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def _calibration(labels: pd.Series, probabilities: np.ndarray, destination: Path) -> Path:
    edges = np.linspace(0, 1, 8)
    points = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities > lower) & (probabilities <= upper)
        if mask.any():
            points.append((probabilities[mask].mean(), labels.to_numpy()[mask].mean(), int(mask.sum())))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#87938f", label="Perfect calibration")
    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", color=GREEN, linewidth=2, label="Production model")
    for predicted, observed, count in points:
        ax.annotate(f"n={count}", (predicted, observed), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlim=(0, 1), ylim=(0, 1), title="Held-out podium probability calibration", xlabel="Predicted probability", ylabel="Observed frequency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def _latest_forecast(frame: pd.DataFrame, model: PodiumModel, destination: Path) -> Path:
    latest = frame.loc[frame["race_id"] == frame["race_id"].max()].copy()
    latest["probability"] = model.predict_proba(latest)
    latest = latest.sort_values("probability")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(latest["driver_name"], latest["probability"], color=GREEN)
    ax.set(title=f"{latest['name'].iloc[0]} podium forecast", xlabel="Podium probability", ylabel="")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def write_metrics(destination: Path = ROOT / "reports" / "metrics-summary.json") -> Path:
    metadata = json.loads(SETTINGS.metadata_path.read_text())
    summary = {
        "dataset": {key: metadata[key] for key in ("rows", "races", "train_rows", "test_rows", "train_end", "test_start")},
        "selected_model": metadata["selected_model"],
        "selected_parameters": metadata["selected_parameters"],
        "metrics": metadata["metrics"],
    }
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    return destination
