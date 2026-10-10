"""Competition data preprocessing and submission integrity tests."""
import numpy as np
import pandas as pd
import pytest
from kaggle_s6e10_data import load_s6e10, make_submission


def test_categorical_columns_align_and_id_is_not_a_feature(tmp_path):
    n = 40
    train = pd.DataFrame({
        'id': range(n),
        'Gender': ['Female', 'Male'] * (n // 2),
        'Class': ['Business', 'Eco'] * (n // 2),
        'Customer Type': ['Loyal Customer', 'disloyal Customer'] * (n // 2),
        'Type of Travel': ['Business travel', 'Personal Travel'] * (n // 2),
        'Age': np.arange(n),
        'satisfaction': [True, False] * (n // 2),
    })
    test = train.drop(columns=['satisfaction']).head(4).copy()
    test['Gender'] = 'Female'
    test['id'] = range(100, 104)
    train.to_csv(tmp_path / 'train.csv', index=False)
    test.to_csv(tmp_path / 'test.csv', index=False)
    data = load_s6e10(tmp_path)
    assert data.X_dev.shape[1] == data.X_test.shape[1] == len(data.feature_names)
    assert 'id' not in data.feature_names
    result = make_submission(data, np.array([.1, .2, .3, .4]))
    assert result['id'].tolist() == [100, 101, 102, 103]
    assert result['satisfaction'].tolist() == [.1, .2, .3, .4]


def test_submission_rejects_wrong_number_of_scores(tmp_path):
    n = 40
    train = pd.DataFrame({
        'id': range(n),
        'Gender': ['Female', 'Male'] * (n // 2),
        'Class': ['Business', 'Eco'] * (n // 2),
        'Customer Type': ['Loyal Customer', 'disloyal Customer'] * (n // 2),
        'Type of Travel': ['Business travel', 'Personal Travel'] * (n // 2),
        'Age': np.arange(n),
        'satisfaction': [True, False] * (n // 2),
    })
    train.to_csv(tmp_path / 'train.csv', index=False)
    train.drop(columns=['satisfaction']).iloc[:3].to_csv(tmp_path / 'test.csv', index=False)
    data = load_s6e10(tmp_path)
    with pytest.raises(ValueError):
        make_submission(data, np.array([.1, .2]))
