import json

from f1_podium.cli import main
from f1_podium.data import generate_sample_tables
from f1_podium.features import FEATURE_COLUMNS, build_feature_table, chronological_split
from f1_podium.model import PodiumModel


def test_predict_cli_outputs_json(tmp_path, capsys) -> None:
    frame = build_feature_table(generate_sample_tables())
    train, test = chronological_split(frame)
    model_path = PodiumModel.fit_logistic(train).save(tmp_path / "model.joblib")
    row = test.iloc[0]
    payload = {
        "driver_name": row["driver_name"],
        "race_name": row["name"],
        **{column: float(row[column]) for column in FEATURE_COLUMNS},
    }
    feature_path = tmp_path / "features.json"
    feature_path.write_text(json.dumps(payload))
    assert main(["predict", str(feature_path), "--model", str(model_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["driver_name"] == row["driver_name"]

