"""Load the SAVED AutoMLTree ensemble and export a Kaggle submission.

Usage:
    python predict_submission.py
    python predict_submission.py --work-dir runs/s6e10 --output submission.csv

No model training takes place. The original train.csv is needed to reconstruct
and validate the same folds and OOF rank reference used by the saved ensemble.
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from automl_tree import AutoMLTree
from competition_data import load_competition, project_path, TARGET, ID


def create_submission(data_dir="playground-series-s6e10",
                      work_dir="runs/s6e10", output="submission.csv"):
    folder = project_path(work_dir)
    config_path = folder / "competition_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No saved run configuration: {config_path}. Run example_competition.py first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = load_competition(data_dir,
                            holdout_fraction=config["holdout_fraction"],
                            split_seed=config["split_seed"])
    if data.feature_names != config["feature_names"]:
        raise ValueError("Feature encoding differs from the training run")
    auto = AutoMLTree(
        work_dir=folder, backends=tuple(config["backends"]),
        n_folds=config["n_folds"], seed=config["seed"],
        corr_max=config["corr_max"],
    )
    # Restores and validates saved ensemble. This does NOT fit any models.
    auto.initialize(data.X_dev, data.y_dev)
    predictions = auto.predict_scores(data.X_test, prediction_set='test')
    if predictions.shape != (len(data.test_ids),) or not np.isfinite(predictions).all():
        raise ValueError("Missing, nonfinite or misaligned test predictions")
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Expected discrimination scores between 0 and 1")

    if data.sample_submission is not None:
        sub = data.sample_submission.copy()
        if ID not in sub or TARGET not in sub:
            raise ValueError("sample_submission.csv must contain id and satisfaction columns")
        if len(sub) != len(predictions) or not sub[ID].reset_index(drop=True).equals(
                data.test_ids.reset_index(drop=True)):
            raise ValueError("Sample submission IDs do not match test.csv order")
    else:
        sub = pd.DataFrame({ID: data.test_ids})
    sub[TARGET] = predictions
    destination = project_path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(destination, index=False)
    print(f"Saved {len(sub):,} predictions to {destination}")
    print(f"Selected ensemble CV AUC: {auto.best_ensemble_.score:.6f}")
    print(f"Holdout AUC (diagnostic only): "
          f"{roc_auc_score(data.y_holdout, auto.predict_scores(data.X_holdout, prediction_set='hold')):.6f}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="playground-series-s6e10")
    parser.add_argument("--work-dir", default="runs/s6e10")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    create_submission(args.data_dir, args.work_dir, args.output)


if __name__ == "__main__":
    main()
