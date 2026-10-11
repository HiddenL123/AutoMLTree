"""Test the shared S6E10 data contract without requiring the Kaggle CSVs."""
import numpy as np
import pandas as pd
import pytest
from competition_data import load_competition


def test_competition_loader_aligns_features_and_keeps_ids(tmp_path):
    n = 50
    train = pd.DataFrame({
        "id": np.arange(100, 100 + n),
        "Gender": ["Female", "Male"] * (n // 2),
        "Class": ["Eco", "Business"] * (n // 2),
        "Customer Type": ["Loyal", "New"] * (n // 2),
        "Type of Travel": ["Work", "Personal"] * (n // 2),
        "Age": np.arange(n) + 20,
        "satisfaction": [True, False] * (n // 2),
    })
    test = train.drop(columns=["satisfaction"]).iloc[:7].copy()
    test["id"] = range(900, 907)
    test.loc[test.index[0], "Class"] = "UnseenInTrain"
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    result = load_competition(tmp_path, holdout_fraction=.2)
    assert len(result.X_dev) == 40
    assert len(result.X_holdout) == 10
    assert result.X_dev.shape[1] == result.X_test.shape[1]
    assert "id" not in result.feature_names
    assert "satisfaction" not in result.feature_names
    assert result.test_ids.tolist() == list(range(900, 907))
    assert set(result.y_dev) == {0, 1}


def test_competition_requires_matching_columns(tmp_path):
    pd.DataFrame({"id": [1, 2], "Age": [20, 30], "satisfaction": [True, False]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"id": [3], "Other": [20]}).to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="Feature columns differ"):
        load_competition(tmp_path)
