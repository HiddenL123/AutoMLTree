"""Train the original (non-scheduler) AutoMLTree on Kaggle S6E10 CSVs.

Examples:
    python example_competition.py
    python example_competition.py --trials 5 --backends lgb xgb cat lr leaf_lr
    python example_competition.py --full --backends lr leaf_lr lgb xgb cat

Studies and selected ensemble persist in --work-dir; reruns add new trials,
while --full resumes only until the original per-family trial totals are met.
Run predict_submission.py afterwards to write submission.csv.
"""
from __future__ import annotations
import argparse
import json

from sklearn.metrics import roc_auc_score
from automl_tree import AutoMLTree
from competition_data import load_competition, project_path

ORIGINAL_TRIAL_TARGETS = {"lr": 40, "leaf_lr": 30, "lgb": 80, "xgb": 80, "cat": 40}
BACKENDS = tuple(ORIGINAL_TRIAL_TARGETS)
CONFIG_NAME = "competition_config.json"


def run(args):
    data = load_competition(args.data_dir,
                            holdout_fraction=args.holdout_fraction,
                            split_seed=args.split_seed)
    work_dir = project_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / CONFIG_NAME
    config = {
        "competition": "playground-series-s6e10", "seed": args.seed,
        "n_folds": args.folds, "split_seed": args.split_seed,
        "holdout_fraction": args.holdout_fraction,
        "corr_max": args.corr_max,
        "feature_names": data.feature_names,
        "backends": list(args.backends),
    }
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("competition", "seed", "n_folds", "split_seed",
                    "holdout_fraction", "corr_max", "feature_names"):
            if previous[key] != config[key]:
                raise ValueError(f"{key} changed for the existing run. Use a NEW --work-dir.")
        # Keep other previously used families available for the submission script.
        config["backends"] = sorted(set(previous["backends"]) | set(args.backends))

    auto = AutoMLTree(
        work_dir=work_dir, backends=tuple(args.backends),
        n_folds=args.folds, seed=args.seed,
        early_stop=args.early_stop,
        min_candidate_auc=args.min_candidate_auc,
        corr_max=args.corr_max,
        beam_width=args.beam_width, search_depth=args.search_depth,
        candidate_limit=args.candidate_limit,
        blend_trials_per_batch=args.blend_trials,
    )
    # Initialize first: the dataset fingerprint guards against mixing saved trials.
    auto.initialize(data.X_dev, data.y_dev)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.full:
        targets = {name: ORIGINAL_TRIAL_TARGETS[name] for name in args.backends}
        auto.fit_to_total(data.X_dev, data.y_dev, targets)
    else:
        auto.fit(data.X_dev, data.y_dev,
                 trials_per_family={name: args.trials for name in args.backends})

    try:
        state = auto.best_ensemble_
    except RuntimeError:
        print("No eligible ensemble yet. Try more trials or reduce --min-candidate-auc.")
        return
    holdout = auto.predict_scores(data.X_holdout, prediction_set='hold')
    print(f"Best selection CV AUC: {state.score:.6f}")
    print(f"Holdout AUC: {roc_auc_score(data.y_holdout, holdout):.6f}")
    print("Members:")
    for name, weight in zip(state.members, state.weights):
        print(f"  {name}: {weight:.4f}")
    print(f"Blending mode: {state.mode}")
    print("Run 'python predict_submission.py' to generate submission.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="playground-series-s6e10")
    p.add_argument("--work-dir", default="runs/s6e10")
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=["lgb"])
    p.add_argument("--trials", type=int, default=5,
                   help="New trials per chosen family each invocation (default: 5)")
    p.add_argument("--full", action="store_true",
                   help="Reach original lifetime targets: LR40, Leaf30, LGB80, XGB80, Cat40")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--holdout-fraction", type=float, default=.20)
    p.add_argument("--min-candidate-auc", type=float, default=None,
                   help="Original script used 0.95; default None avoids excluding complementary models")
    p.add_argument("--corr-max", type=float, default=.99)
    p.add_argument("--early-stop", type=int, default=200)
    p.add_argument("--beam-width", type=int, default=5)
    p.add_argument("--search-depth", type=int, default=2)
    p.add_argument("--candidate-limit", type=int, default=30)
    p.add_argument("--blend-trials", type=int, default=20)
    args = p.parse_args()
    if args.trials < 0:
        p.error("--trials must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
