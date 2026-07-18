import json

from f1_podium.config import Settings
from f1_podium.data import write_sample_tables
from f1_podium.model import PodiumModel
from f1_podium.training import train_and_evaluate


def test_training_writes_reloadable_artifacts(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    write_sample_tables(raw_dir, seed=42)
    settings = Settings(
        raw_dir=raw_dir,
        processed_path=tmp_path / "processed.csv",
        model_path=tmp_path / "model.joblib",
        metadata_path=tmp_path / "metadata.json",
    )
    metadata = train_and_evaluate(settings)
    saved = json.loads(settings.metadata_path.read_text())

    assert metadata["train_end"] < metadata["test_start"]
    assert metadata["selected_model"] in {"gradient_boosting", "logistic_regression"}
    assert saved["metrics"]["gradient_boosting"]["roc_auc"] > 0.5
    assert settings.processed_path.exists()
    model = PodiumModel.load(settings.model_path)
    assert model.model_name in {"gradient-boosting-v1", "logistic-v1"}

