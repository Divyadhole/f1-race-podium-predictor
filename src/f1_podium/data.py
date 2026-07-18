"""Ergast-compatible table loading, validation, and offline sample generation."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


TABLE_COLUMNS = {
    "races": {"race_id", "year", "round", "circuit_id", "name", "date"},
    "drivers": {"driver_id", "driver_code", "forename", "surname", "nationality"},
    "constructors": {"constructor_id", "constructor_name"},
    "qualifying": {"race_id", "driver_id", "constructor_id", "grid"},
    "results": {"race_id", "driver_id", "constructor_id", "position", "points", "status"},
}


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Validate schemas, keys, and relational integrity."""
    missing_tables = set(TABLE_COLUMNS).difference(tables)
    if missing_tables:
        raise ValueError(f"Missing tables: {sorted(missing_tables)}")
    clean = {name: frame.copy() for name, frame in tables.items()}
    for name, required in TABLE_COLUMNS.items():
        missing = required.difference(clean[name].columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    clean["races"]["date"] = pd.to_datetime(clean["races"]["date"], errors="raise")
    if clean["races"]["race_id"].duplicated().any():
        raise ValueError("races.race_id must be unique")
    if clean["drivers"]["driver_id"].duplicated().any():
        raise ValueError("drivers.driver_id must be unique")
    if clean["constructors"]["constructor_id"].duplicated().any():
        raise ValueError("constructors.constructor_id must be unique")
    for name in ("qualifying", "results"):
        if clean[name].duplicated(["race_id", "driver_id"]).any():
            raise ValueError(f"{name} has duplicate driver-race rows")
        if not set(clean[name]["race_id"]).issubset(set(clean["races"]["race_id"])):
            raise ValueError(f"{name} contains unknown race_id")
        if not set(clean[name]["driver_id"]).issubset(set(clean["drivers"]["driver_id"])):
            raise ValueError(f"{name} contains unknown driver_id")
        if not set(clean[name]["constructor_id"]).issubset(
            set(clean["constructors"]["constructor_id"])
        ):
            raise ValueError(f"{name} contains unknown constructor_id")
    if (clean["qualifying"]["grid"] < 1).any() or (clean["results"]["position"] < 1).any():
        raise ValueError("Grid and finish positions must be positive")
    return clean


def load_tables(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(raw_dir)
    tables = {name: pd.read_csv(root / f"{name}.csv") for name in TABLE_COLUMNS}
    return validate_tables(tables)


def generate_sample_tables(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Generate realistic, reproducible F1-style relational data."""
    rng = np.random.default_rng(seed)
    constructor_specs = [
        (1, "Apex GP", 1.20),
        (2, "Scuderia Veloce", 0.82),
        (3, "Silver Arrow Racing", 0.60),
        (4, "Papaya Motorsport", 0.35),
        (5, "Emerald Racing", -0.05),
    ]
    driver_specs = [
        (1, "NOR", "Noah", "Norris", "British", 0.85, 1),
        (2, "LEC", "Leo", "Leclerc", "Monegasque", 0.76, 1),
        (3, "VER", "Max", "Vermeer", "Dutch", 0.96, 2),
        (4, "SAI", "Carlos", "Santos", "Spanish", 0.55, 2),
        (5, "RUS", "George", "Russell", "British", 0.60, 3),
        (6, "HAM", "Lewis", "Hamilton", "British", 0.72, 3),
        (7, "PIA", "Oscar", "Piastri", "Australian", 0.48, 4),
        (8, "ALO", "Fernando", "Alonso", "Spanish", 0.52, 4),
        (9, "TSU", "Yuki", "Tsunoda", "Japanese", 0.12, 5),
        (10, "ALB", "Alex", "Albon", "Thai", 0.18, 5),
    ]
    circuits = [
        (1, "Bahrain", "Sakhir", 0.10),
        (2, "Monaco", "Monte Carlo", 0.45),
        (3, "Silverstone", "Silverstone", -0.05),
        (4, "Monza", "Monza", -0.20),
        (5, "Suzuka", "Suzuka", 0.15),
        (6, "Interlagos", "Sao Paulo", 0.20),
        (7, "Spa", "Stavelot", -0.10),
        (8, "Marina Bay", "Singapore", 0.30),
        (9, "Austin", "Austin", 0.00),
        (10, "Mexico City", "Mexico City", 0.12),
        (11, "Albert Park", "Melbourne", 0.05),
        (12, "Yas Marina", "Abu Dhabi", -0.02),
    ]
    constructors = pd.DataFrame(
        [(item[0], item[1]) for item in constructor_specs],
        columns=["constructor_id", "constructor_name"],
    )
    drivers = pd.DataFrame(
        [item[:5] for item in driver_specs],
        columns=["driver_id", "driver_code", "forename", "surname", "nationality"],
    )
    circuit_frame = pd.DataFrame(
        [(item[0], item[1], item[2]) for item in circuits],
        columns=["circuit_id", "circuit_name", "location"],
    )
    constructor_strength = {item[0]: item[2] for item in constructor_specs}
    driver_strength = {item[0]: item[5] for item in driver_specs}
    driver_team = {item[0]: item[6] for item in driver_specs}
    circuit_variance = {item[0]: item[3] for item in circuits}
    races: list[dict[str, object]] = []
    qualifying: list[dict[str, int]] = []
    results: list[dict[str, object]] = []
    points_scale = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    race_id = 1
    for year in range(2017, 2026):
        for round_number, circuit in enumerate(circuits, start=1):
            circuit_id = circuit[0]
            date = pd.Timestamp(year, 3, 1) + timedelta(days=(round_number - 1) * 18)
            races.append(
                {
                    "race_id": race_id,
                    "year": year,
                    "round": round_number,
                    "circuit_id": circuit_id,
                    "name": f"{circuit[1]} Grand Prix",
                    "date": date.date().isoformat(),
                }
            )
            pace = []
            for driver_id in driver_strength:
                team_id = driver_team[driver_id]
                season_trend = 0.04 * (year - 2017) * (1 if team_id in {1, 4} else -0.2)
                base = driver_strength[driver_id] + constructor_strength[team_id] + season_trend
                qualifying_score = base + rng.normal(0, 0.38)
                pace.append((driver_id, team_id, base, qualifying_score))
            grid_order = sorted(pace, key=lambda row: row[3], reverse=True)
            grid_by_driver = {row[0]: grid + 1 for grid, row in enumerate(grid_order)}
            finish_scores = []
            for driver_id, team_id, base, _ in pace:
                grid = grid_by_driver[driver_id]
                reliability = rng.random() > (0.035 + 0.008 * grid)
                score = base - 0.12 * grid + circuit_variance[circuit_id] * driver_strength[driver_id]
                score += rng.normal(0, 0.48)
                if not reliability:
                    score -= 5
                finish_scores.append((driver_id, team_id, grid, score, reliability))
            finish_order = sorted(finish_scores, key=lambda row: row[3], reverse=True)
            for position, (driver_id, team_id, grid, _, reliable) in enumerate(finish_order, start=1):
                qualifying.append(
                    {"race_id": race_id, "driver_id": driver_id, "constructor_id": team_id, "grid": grid}
                )
                results.append(
                    {
                        "race_id": race_id,
                        "driver_id": driver_id,
                        "constructor_id": team_id,
                        "position": position,
                        "points": points_scale[position - 1],
                        "status": "Finished" if reliable else "Retired",
                    }
                )
            race_id += 1
    return validate_tables(
        {
            "races": pd.DataFrame(races),
            "drivers": drivers,
            "constructors": constructors,
            "qualifying": pd.DataFrame(qualifying),
            "results": pd.DataFrame(results),
            "circuits": circuit_frame,
        }
    )


def write_sample_tables(raw_dir: str | Path, seed: int = 42) -> list[Path]:
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    tables = generate_sample_tables(seed)
    paths = []
    for name, frame in tables.items():
        path = destination / f"{name}.csv"
        frame.assign(**({"date": frame["date"].dt.date} if "date" in frame else {})).to_csv(
            path, index=False
        )
        paths.append(path)
    return paths
