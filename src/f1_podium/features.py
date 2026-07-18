"""Leakage-safe joins, temporal features, and train/test splitting."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "grid",
    "driver_podium_rate_5",
    "driver_points_5",
    "constructor_points_5",
    "circuit_finish_history",
    "driver_reliability_5",
    "season_round_fraction",
]


def build_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join relational tables and compute features using prior events only."""
    results = tables["results"].merge(
        tables["qualifying"],
        on=["race_id", "driver_id", "constructor_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(results) != len(tables["results"]):
        raise ValueError("Qualifying and result joins are incomplete")
    frame = (
        results.merge(tables["races"], on="race_id", validate="many_to_one")
        .merge(tables["drivers"], on="driver_id", validate="many_to_one")
        .merge(tables["constructors"], on="constructor_id", validate="many_to_one")
        .sort_values(["date", "race_id", "driver_id"])
        .reset_index(drop=True)
    )
    frame["podium"] = (frame["position"] <= 3).astype(int)
    frame["finished"] = (frame["status"] == "Finished").astype(int)
    driver = frame.groupby("driver_id", sort=False)
    frame["driver_podium_rate_5"] = driver["podium"].transform(
        lambda series: series.shift().rolling(5, min_periods=1).mean()
    )
    frame["driver_points_5"] = driver["points"].transform(
        lambda series: series.shift().rolling(5, min_periods=1).mean()
    )
    frame["driver_reliability_5"] = driver["finished"].transform(
        lambda series: series.shift().rolling(5, min_periods=1).mean()
    )

    constructor_race = (
        frame.groupby(["constructor_id", "race_id", "date"], as_index=False)["points"]
        .sum()
        .sort_values(["constructor_id", "date"])
    )
    constructor_race["constructor_points_5"] = constructor_race.groupby(
        "constructor_id"
    )["points"].transform(lambda series: series.shift().rolling(5, min_periods=1).mean())
    frame = frame.merge(
        constructor_race[["constructor_id", "race_id", "constructor_points_5"]],
        on=["constructor_id", "race_id"],
        validate="many_to_one",
    )
    frame["circuit_finish_history"] = frame.groupby(
        ["driver_id", "circuit_id"], sort=False
    )["position"].transform(lambda series: series.shift().expanding(min_periods=1).mean())
    frame["season_round_fraction"] = frame["round"] / frame.groupby("year")["round"].transform(
        "max"
    )
    frame["driver_name"] = frame["forename"] + " " + frame["surname"]
    defaults = {
        "driver_podium_rate_5": 0.3,
        "driver_points_5": 8.0,
        "constructor_points_5": 16.0,
        "circuit_finish_history": 5.5,
        "driver_reliability_5": 0.9,
    }
    return frame.fillna(defaults).sort_values(["date", "race_id", "grid"]).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame, test_start_year: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["year"] < test_start_year].copy()
    test = frame.loc[frame["year"] >= test_start_year].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition")
    if train["date"].max() >= test["date"].min():
        raise AssertionError("Temporal overlap detected")
    return train, test


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, FEATURE_COLUMNS].astype(float)

