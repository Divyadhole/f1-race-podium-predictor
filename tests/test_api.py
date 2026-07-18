from fastapi.testclient import TestClient

from f1_podium.api import create_app
from f1_podium.data import generate_sample_tables
from f1_podium.features import FEATURE_COLUMNS, build_feature_table, chronological_split
from f1_podium.model import PodiumModel


def client_and_payload():
    frame = build_feature_table(generate_sample_tables())
    train, test = chronological_split(frame)
    client = TestClient(create_app(PodiumModel.fit_gradient_boosting(train)))
    row = test.iloc[0]
    payload = {
        "driver_name": row["driver_name"],
        "race_name": row["name"],
        **{column: float(row[column]) for column in FEATURE_COLUMNS},
    }
    payload["grid"] = int(payload["grid"])
    return client, payload


def test_health_and_prediction_contract() -> None:
    client, payload = client_and_payload()
    assert client.get("/health").status_code == 200
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 < response.json()["podium_probability"] < 1


def test_invalid_grid_is_rejected() -> None:
    client, payload = client_and_payload()
    payload["grid"] = 0
    assert client.post("/predict", json=payload).status_code == 422

