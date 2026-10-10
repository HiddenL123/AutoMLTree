"""Restore the best saved AutoMLTree ensemble and create a Kaggle submission.

Run after kaggle_s6e10_train.py. No new models are trained here.
    python kaggle_s6e10_submit.py
    python kaggle_s6e10_submit.py --output submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score

from automl_tree import AutoMLTree
from kaggle_s6e10_data import DATA_DIR, RUN_DIR, SEED, N_FOLDS, load_s6e10, make_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "submission.csv")
    parser.add_argument("--corr-max", type=float, default=0.99,
                        help="Must match the training configuration")
    args = parser.parse_args()
    if not (args.work_dir / "best_ensemble.json").is_file():
        parser.error(f"No saved ensemble in {args.work_dir}. Run kaggle_s6e10_train.py first")

    data = load_s6e10(args.data_dir)
    # initialize() checks the original development-data and fold fingerprint
    # and restores the saved ensemble WITHOUT starting new Optuna trials.
    auto = AutoMLTree(work_dir=args.work_dir, backends=("lgb",),
                      n_folds=N_FOLDS, seed=SEED, corr_max=args.corr_max)
    auto.initialize(data.X_dev, data.y_dev)
    scores = auto.predict_scores(data.X_test)
    submission = make_submission(data, scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)

    holdout_auc = roc_auc_score(data.y_holdout, auto.predict_scores(data.X_holdout))
    print(f"Restored {len(auto.best_ensemble_.members)} models; CV AUC "
          f"{auto.best_ensemble_.score:.7f}; holdout AUC {holdout_auc:.7f}")
    print(f"Wrote {len(submission):,} rows to {args.output}")
    print("Note: scores are optimized for AUC; they are not calibrated probabilities.")


if __name__ == "__main__":
    main()
