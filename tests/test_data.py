import pandas as pd
import pytest

from f1_podium.data import generate_sample_tables, validate_tables


def test_sample_tables_are_reproducible_and_relational() -> None:
    first = generate_sample_tables(seed=9)
    second = generate_sample_tables(seed=9)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert len(first["races"]) == 108
    assert len(first["results"]) == 1080


def test_duplicate_driver_race_is_rejected() -> None:
    tables = generate_sample_tables()
    tables["results"] = pd.concat([tables["results"], tables["results"].head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        validate_tables(tables)

