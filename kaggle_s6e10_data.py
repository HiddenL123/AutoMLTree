"""Shared, deterministic preprocessing for Kaggle Playground Series S6E10.

Both scripts use the same dataset and folds to restore saved AutoMLTree models.
IDs are kept for submission output, never used as prediction features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent / "playground-series-s6e10"
RUN_DIR = Path(__file__).resolve().parent / "runs" / "s6e10_baseline"
TARGET = "satisfaction"
NOMINAL_COLUMNS = ("Gender", "Class", "Customer Type", "Type of Travel")
SEED = 42
N_FOLDS = 5
HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 0


@dataclass
class CompetitionData:
    X_dev: np.ndarray
    y_dev: np.ndarray
    X_holdout: np.ndarray
    y_holdout: np.ndarray
    X_test: np.ndarray
    test_ids: pd.Series
    feature_names: tuple[str, ...]
    sample_submission: pd.DataFrame | None


def _target_to_binary(series: pd.Series) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f"Training target {TARGET!r} contains missing labels")
    as_text = series.astype(str).str.strip().str.lower()
    values = as_text.map({
        "true": 1, "false": 0, "1": 1, "0": 0,
        "satisfied": 1, "neutral or dissatisfied": 0,
    })
    if values.isna().any() or values.nunique() != 2:
        raise ValueError(f"Expected two known classes for {TARGET!r}; found {series.unique()!r}")
    return values.to_numpy(dtype=np.int64)


def load_s6e10(data_dir: str | Path = DATA_DIR) -> CompetitionData:
    data_dir = Path(data_dir)
    train_path, test_path = data_dir / "train.csv", data_dir / "test.csv"
    for path in (train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Place the Kaggle CSVs in {data_dir}")
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    if TARGET not in train_raw or "id" not in test_raw:
        raise ValueError("train.csv needs satisfaction and test.csv needs id")
    y = _target_to_binary(train_raw[TARGET])
    test_ids = test_raw["id"].reset_index(drop=True)
    train_features = train_raw.drop(columns=[TARGET, "id"], errors="ignore").copy()
    test_features = test_raw.drop(columns=[TARGET, "id"], errors="ignore").copy()

    # The same categorical vocabulary and dummy baseline are used for both sets.
    for col in NOMINAL_COLUMNS:
        if col not in train_features or col not in test_features:
            raise ValueError(f"Missing expected categorical column {col!r}")
        categories = sorted(train_features[col].astype("string").fillna("__MISSING__").unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train_features[col] = train_features[col].astype("string").fillna("__MISSING__").astype(dtype)
        test_features[col] = test_features[col].astype("string").fillna("__MISSING__").astype(dtype)

    train_encoded = pd.get_dummies(train_features, columns=list(NOMINAL_COLUMNS),
                                   drop_first=True, dtype=float)
    test_encoded = pd.get_dummies(test_features, columns=list(NOMINAL_COLUMNS),
                                  drop_first=True, dtype=float)
    if set(train_encoded.columns) != set(test_encoded.columns):
        raise ValueError("Train and test feature columns differ after encoding")
    test_encoded = test_encoded.reindex(columns=train_encoded.columns)

    try:
        X = train_encoded.to_numpy(dtype=np.float64)
        X_test = test_encoded.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Non-numeric features remain; update preprocessing in kaggle_s6e10_data.py") from exc
    if np.isinf(X).any() or np.isinf(X_test).any():
        raise ValueError("Feature matrices contain infinite values")

    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=HOLDOUT_FRACTION, stratify=y, random_state=HOLDOUT_SEED,
    )
    submission_path = data_dir / "sample_submission.csv"
    sample = pd.read_csv(submission_path) if submission_path.is_file() else None
    return CompetitionData(X_dev, y_dev, X_holdout, y_holdout, X_test,
                           test_ids, tuple(train_encoded.columns), sample)


def make_submission(data: CompetitionData, scores: np.ndarray) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(data.test_ids),) or not np.isfinite(scores).all():
        raise ValueError("Test predictions must be a finite vector with one score per row")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Submission scores must lie in [0,1]")
    if data.sample_submission is not None:
        submission = data.sample_submission.copy()
        if len(submission) != len(data.test_ids):
            raise ValueError("Sample submission and test.csv have different row counts")
        if "id" in submission and not submission["id"].reset_index(drop=True).equals(data.test_ids):
            raise ValueError("Sample submission IDs differ from test.csv or are out of order")
    else:
        submission = pd.DataFrame({"id": data.test_ids})
    submission[TARGET] = scores
    return submission
