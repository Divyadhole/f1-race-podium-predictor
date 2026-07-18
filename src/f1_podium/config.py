"""Project settings and paths."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    test_start_year: int = 2024
    raw_dir: Path = ROOT / "data" / "raw"
    processed_path: Path = ROOT / "data" / "processed" / "driver_race_features.csv"
    model_path: Path = ROOT / "artifacts" / "podium_model.joblib"
    metadata_path: Path = ROOT / "artifacts" / "model_metadata.json"


SETTINGS = Settings()

