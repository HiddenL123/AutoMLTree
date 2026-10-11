"""Shared, deterministic CSV preprocessing for the S6E10 Kaggle runner.

No target-derived statistics are fitted to the test data. Categorical dummy
columns are aligned jointly, without ever including the target or ID column.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "satisfaction"
ID = "id"


@dataclass
class CompetitionData:
    X_dev: np.ndarray
    y_dev: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    X_test: np.ndarray
    test_ids: pd.Series
    feature_names: list[str]
    sample_submission: pd.DataFrame | None


def project_path(value: str | Path) -> Path:
    """Interpret relative paths relative to the repository root, not the shell CWD."""
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def _binary_target(values: pd.Series) -> np.ndarray:
    labels = values.astype("string").str.strip().str.lower()
    mapped = labels.map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "satisfied": 1, "neutral or dissatisfied": 0,
    })
    if mapped.isna().any() or set(mapped.unique()) != {0, 1}:
        bad = values.loc[mapped.isna()].astype(str).unique()[:5]
        raise ValueError(f"Unsupported or non-binary {TARGET} labels: {list(bad)}")
    return mapped.to_numpy(dtype=np.int64)


def load_competition(data_dir="playground-series-s6e10", *,
                     holdout_fraction=0.20, split_seed=0) -> CompetitionData:
    folder = project_path(data_dir)
    train_path, test_path = folder / "train.csv", folder / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv inside {folder}. "
            "Supply --data-dir if they are somewhere else."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if TARGET not in train or ID not in train or ID not in test:
        raise ValueError(f"Training CSV must include {TARGET!r} and both CSVs must include {ID!r}.")
    if TARGET in test:
        raise ValueError(f"test.csv must not contain the target column {TARGET!r}.")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    y = _binary_target(train[TARGET])
    train_features = train.drop(columns=[TARGET, ID])
    test_features = test.drop(columns=[ID])
    if set(train_features.columns) != set(test_features.columns):
        missing = sorted(set(train_features) - set(test_features))
        extra = sorted(set(test_features) - set(train_features))
        raise ValueError(f"Feature columns differ. Missing from test: {missing}; extra in test: {extra}")
    test_features = test_features.reindex(columns=train_features.columns)

    # One-hot encoding for all categorical inputs, including the four nominal
    # columns from the original notebook (Gender, Class, Customer Type, Travel).
    # Using one combined *unlabeled* feature frame keeps column ordering equal.
    combined = pd.concat([train_features, test_features], ignore_index=True)
    categoricals = list(combined.select_dtypes(include=["object", "category", "string"]).columns)
    for column in ("Gender", "Class", "Customer Type", "Type of Travel"):
        if column in combined and column not in categoricals:
            categoricals.append(column)
    encoded = pd.get_dummies(combined, columns=categoricals,
                             drop_first=True, dtype=np.float64)
    encoded = encoded.apply(pd.to_numeric, errors="raise")
    names = [str(col) for col in encoded.columns]
    if len(set(names)) != len(names):
        raise ValueError("Feature encoding produced duplicate column names")
    array = encoded.to_numpy(dtype=np.float64)
    if not np.isfinite(array[~np.isnan(array)]).all():
        raise ValueError("Input features contain infinite values")
    X, X_test = array[:len(train)], array[len(train):]
    dev_idx, hold_idx = train_test_split(
        np.arange(len(train)), test_size=holdout_fraction,
        stratify=y, random_state=split_seed)

    submission_path = folder / "sample_submission.csv"
    sample = pd.read_csv(submission_path) if submission_path.exists() else None
    return CompetitionData(
        X_dev=X[dev_idx], y_dev=y[dev_idx],
        X_holdout=X[hold_idx], y_holdout=y[hold_idx],
        X_test=X_test, test_ids=test[ID].copy(),
        feature_names=names, sample_submission=sample,
    )
