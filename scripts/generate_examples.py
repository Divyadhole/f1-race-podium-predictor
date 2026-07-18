"""Generate deterministic inference examples from the final artifact."""

import json

import pandas as pd

from f1_podium.config import ROOT, SETTINGS
from f1_podium.features import FEATURE_COLUMNS
from f1_podium.model import PodiumModel


if __name__ == "__main__":
    output = ROOT / "examples"
    output.mkdir(exist_ok=True)
    frame = pd.read_csv(SETTINGS.processed_path)
    latest = frame.loc[frame["race_id"] == frame["race_id"].max()].copy()
    model = PodiumModel.load(SETTINGS.model_path)
    latest["podium_probability"] = model.predict_proba(latest)
    ranking = latest[
        ["driver_name", "constructor_name", "grid", "podium_probability"]
    ].sort_values("podium_probability", ascending=False)
    ranking.to_csv(output / "latest-race-ranking.csv", index=False)

    row = latest.sort_values("grid").iloc[0]
    request = {
        "driver_name": row["driver_name"],
        "race_name": row["name"],
        **{column: float(row[column]) for column in FEATURE_COLUMNS},
    }
    request["grid"] = int(request["grid"])
    response = model.predict_one(
        request["driver_name"], request["race_name"], request
    ).as_dict()
    (output / "prediction-request.json").write_text(json.dumps(request, indent=2) + "\n")
    (output / "prediction-response.json").write_text(json.dumps(response, indent=2) + "\n")
    print(f"Wrote examples to {output}")

