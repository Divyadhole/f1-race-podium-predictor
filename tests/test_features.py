from f1_podium.data import generate_sample_tables
from f1_podium.features import build_feature_table, chronological_split, feature_matrix


def test_join_integrity_and_feature_shape() -> None:
    tables = generate_sample_tables()
    frame = build_feature_table(tables)
    assert len(frame) == len(tables["results"])
    assert not frame.duplicated(["race_id", "driver_id"]).any()
    assert feature_matrix(frame).shape == (1080, 7)
    assert not feature_matrix(frame).isna().any().any()


def test_chronological_split_has_no_leakage() -> None:
    frame = build_feature_table(generate_sample_tables())
    train, test = chronological_split(frame)
    assert train["date"].max() < test["date"].min()
    assert set(train["year"]) == set(range(2017, 2024))
    assert set(test["year"]) == {2024, 2025}


def test_first_race_uses_cold_start_defaults() -> None:
    frame = build_feature_table(generate_sample_tables())
    first = frame.loc[frame["race_id"] == 1]
    assert (first["driver_podium_rate_5"] == 0.3).all()
    assert (first["constructor_points_5"] == 16.0).all()

