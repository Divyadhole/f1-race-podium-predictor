"""Command-line workflows for training and podium inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from f1_podium.config import SETTINGS
from f1_podium.features import FEATURE_COLUMNS
from f1_podium.model import PodiumModel
from f1_podium.training import train_and_evaluate


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="f1-podium", description="F1 podium probability prediction")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="Train, tune, evaluate, and save the model")
    predict = commands.add_parser("predict", help="Predict from a JSON feature record")
    predict.add_argument("features", type=Path)
    predict.add_argument("--model", type=Path, default=SETTINGS.model_path)
    latest = commands.add_parser("rank-latest", help="Rank drivers in the latest processed race")
    latest.add_argument("--data", type=Path, default=SETTINGS.processed_path)
    latest.add_argument("--model", type=Path, default=SETTINGS.model_path)
    latest.add_argument("--output", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "train":
        print(json.dumps(train_and_evaluate(), indent=2))
        return 0
    model = PodiumModel.load(args.model)
    if args.command == "predict":
        payload = json.loads(args.features.read_text())
        features = {name: payload[name] for name in FEATURE_COLUMNS}
        result = model.predict_one(payload["driver_name"], payload["race_name"], features)
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    frame = pd.read_csv(args.data)
    latest_race = frame.loc[frame["race_id"] == frame["race_id"].max()].copy()
    latest_race["podium_probability"] = model.predict_proba(latest_race)
    output = latest_race[
        ["driver_name", "constructor_name", "grid", "podium_probability"]
    ].sort_values("podium_probability", ascending=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Wrote {len(output)} predictions to {args.output}")
    else:
        print(output.to_string(index=False, float_format=lambda value: f"{value:.1%}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

