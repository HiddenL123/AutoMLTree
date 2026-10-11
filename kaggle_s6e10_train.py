"""Train the original Optuna-model-pool + ensemble-search AutoML baseline.

Examples:
    python kaggle_s6e10_train.py --quick --families lgb
    python kaggle_s6e10_train.py --to-total --families lgb xgb cat
    python kaggle_s6e10_train.py --trials 10 --blend-trials 30

Each ordinary invocation adds trials to existing studies; --to-total instead
interprets trial counts as lifetime targets. Unlike example_scheduler.py,
this script calls AutoMLTree.fit() / fit_to_total(), not fit_scheduled().
"""
from __future__ import annotations

import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score

from automl_tree import AutoMLTree
from kaggle_s6e10_data import DATA_DIR, RUN_DIR, SEED, N_FOLDS, load_s6e10

ORIGINAL_TRIALS = {"lr": 5, "leaf_lr": 5, "lgb": 10, "xgb": 10, "cat": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--families", nargs="+", choices=list(ORIGINAL_TRIALS),
                        default=list(ORIGINAL_TRIALS))
    parser.add_argument("--quick", action="store_true", help="Three trials per selected family")
    parser.add_argument("--trials", type=int, default=None,
                        help="Override trial count for each selected family")
    parser.add_argument("--to-total", action="store_true",
                        help="Use lifetime trial targets instead of adding new trials")
    parser.add_argument("--min-candidate-auc", type=float, default=None,
                        help="Optional standalone AUC filter; original script used 0.95")
    parser.add_argument("--corr-max", type=float, default=0.99)
    parser.add_argument("--blend-trials", type=int, default=30)
    parser.add_argument("--candidate-limit", type=int, default=30,
                        help="Maximum models examined by search; 0 = all")
    args = parser.parse_args()
    if args.trials is not None and args.trials < 0:
        parser.error("--trials must be nonnegative")
    if args.blend_trials < 0 or args.candidate_limit < 0:
        parser.error("--blend-trials and --candidate-limit must be nonnegative")

    data = load_s6e10(args.data_dir)
    counts = {name: (args.trials if args.trials is not None
                     else 3 if args.quick else ORIGINAL_TRIALS[name])
              for name in args.families}
    print(f"Development rows: {len(data.y_dev):,} | holdout rows: {len(data.y_holdout):,} "
          f"| features: {len(data.feature_names)}")
    print("Trials:", counts, "(lifetime totals)" if args.to_total else "(additional)")

    auto = AutoMLTree(
        work_dir=args.work_dir, backends=tuple(args.families),
        n_folds=N_FOLDS, seed=SEED, early_stop=200,
        min_candidate_auc=args.min_candidate_auc, corr_max=args.corr_max,
        beam_width=5, search_depth=2, candidate_limit=args.candidate_limit,
        min_member_gain=1e-6, blend_trials_per_batch=args.blend_trials,
    )
    if args.to_total:
        auto.fit_to_total(data.X_dev, data.y_dev, target_trials_per_family=counts)
    else:
        auto.fit(data.X_dev, data.y_dev, trials_per_family=counts)

    best = auto.best_ensemble_
    holdout_auc = roc_auc_score(data.y_holdout, auto.predict_scores(data.X_holdout, prediction_set='hold'))
    print(f"Best selection CV AUC: {best.score:.7f} | untouched holdout AUC: {holdout_auc:.7f}")
    print(f"Blend mode: {best.mode}; selected members: {len(best.members)}")
    print("Models and weights:")
    for name, weight in zip(best.members, best.weights):
        print(f"  {name:20s} {weight:.5f}")
    print(f"Saved persistent run to {args.work_dir}")
    print("Generate submission: python kaggle_s6e10_submit.py")


if __name__ == "__main__":
    main()
