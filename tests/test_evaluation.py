import numpy as np
import pandas as pd
import pytest

from f1_podium.evaluation import evaluate


def test_evaluation_rewards_good_probabilities() -> None:
    labels = pd.Series([0, 0, 1, 1])
    good = evaluate(labels, np.array([0.05, 0.15, 0.85, 0.95]))
    weak = evaluate(labels, np.array([0.4, 0.45, 0.55, 0.6]))
    assert good["log_loss"] < weak["log_loss"]
    assert good["roc_auc"] == 1.0


def test_evaluation_rejects_boundary_probabilities() -> None:
    with pytest.raises(ValueError, match="strictly"):
        evaluate(pd.Series([0, 1]), np.array([0.0, 1.0]))

