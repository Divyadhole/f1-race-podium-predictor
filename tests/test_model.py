import pytest

from f1_podium.data import generate_sample_tables
from f1_podium.features import FEATURE_COLUMNS, build_feature_table, chronological_split
from f1_podium.model import PodiumModel, grid_baseline_probabilities


@pytest.fixture(scope="module")
def train_and_test():
    frame = build_feature_table(generate_sample_tables())
    return chronological_split(frame)


def test_probabilities_are_bounded(train_and_test) -> None:
    train, test = train_and_test
    model = PodiumModel.fit_gradient_boosting(train)
    probabilities = model.predict_proba(test)
    assert ((probabilities > 0) & (probabilities < 1)).all()


def test_grid_baseline_declines_with_position(train_and_test) -> None:
    _, test = train_and_test
    ordered = test.sort_values("grid").drop_duplicates("grid")
    probabilities = grid_baseline_probabilities(ordered)
    assert all(left > right for left, right in zip(probabilities, probabilities[1:]))


def test_model_reload_and_single_inference(train_and_test, tmp_path) -> None:
    train, test = train_and_test
    model = PodiumModel.fit_logistic(train)
    loaded = PodiumModel.load(model.save(tmp_path / "model.joblib"))
    features = test.iloc[0][FEATURE_COLUMNS].to_dict()
    prediction = loaded.predict_one("Test Driver", "Test Grand Prix", features)
    assert prediction.driver_name == "Test Driver"
    assert 0 < prediction.podium_probability < 1


def test_single_inference_validates_features(train_and_test) -> None:
    train, _ = train_and_test
    model = PodiumModel.fit_logistic(train)
    with pytest.raises(ValueError, match="Missing"):
        model.predict_one("Driver", "Race", {"grid": 1})
