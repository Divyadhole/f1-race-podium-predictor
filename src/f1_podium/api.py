"""FastAPI inference service for driver and race podium predictions."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from f1_podium.config import SETTINGS
from f1_podium.features import FEATURE_COLUMNS
from f1_podium.model import PodiumModel


class DriverRaceFeatures(BaseModel):
    driver_name: str = Field(min_length=1)
    race_name: str = Field(min_length=1)
    grid: int = Field(ge=1, le=30)
    driver_podium_rate_5: float = Field(ge=0, le=1)
    driver_points_5: float = Field(ge=0)
    constructor_points_5: float = Field(ge=0)
    circuit_finish_history: float = Field(ge=1)
    driver_reliability_5: float = Field(ge=0, le=1)
    season_round_fraction: float = Field(gt=0, le=1)

    def model_features(self) -> dict[str, float]:
        payload = self.model_dump()
        return {name: float(payload[name]) for name in FEATURE_COLUMNS}


class RaceRequest(BaseModel):
    entrants: list[DriverRaceFeatures] = Field(min_length=2, max_length=30)


@lru_cache(maxsize=1)
def load_default_model() -> PodiumModel:
    if not SETTINGS.model_path.exists():
        raise RuntimeError("Model artifact missing. Run `f1-podium train` first.")
    return PodiumModel.load(SETTINGS.model_path)


def create_app(model: PodiumModel | None = None) -> FastAPI:
    app = FastAPI(
        title="F1 Race Podium Predictor",
        description="Leakage-safe driver podium probabilities from qualifying and recent form.",
        version="1.0.0",
    )
    app.state.model = model

    def get_model() -> PodiumModel:
        return app.state.model or load_default_model()

    @app.get("/health", tags=["Operations"])
    def health() -> dict[str, str]:
        try:
            loaded = get_model()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {"status": "healthy", "model": loaded.model_name}

    @app.post("/predict", tags=["Inference"])
    def predict(request: DriverRaceFeatures) -> dict[str, str | int | float]:
        return get_model().predict_one(
            request.driver_name, request.race_name, request.model_features()
        ).as_dict()

    @app.post("/race-ranking", tags=["Inference"])
    def race_ranking(request: RaceRequest) -> dict[str, object]:
        grids = [entrant.grid for entrant in request.entrants]
        if len(grids) != len(set(grids)):
            raise HTTPException(status_code=422, detail="Grid positions must be unique")
        predictions = [
            get_model()
            .predict_one(entrant.driver_name, entrant.race_name, entrant.model_features())
            .as_dict()
            for entrant in request.entrants
        ]
        predictions.sort(key=lambda item: item["podium_probability"], reverse=True)
        return {"ranked_predictions": predictions}

    return app


app = create_app()

