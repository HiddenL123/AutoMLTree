"""One-time conversion of saved Kaggle Optuna trials into native AutoMLTree studies.

Requires old files in:
    legacy/ens.db
    legacy/oof/lgb_0.npz, xgb_0.npz, ...

Typical:
    python convert_previous_models.py --source legacy --dry-run
    python convert_previous_models.py --source legacy --work-dir runs/s6e10

The old DB/folder is not needed by AutoMLTree after successful conversion.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from automl_tree import AutoMLTree
from automl_tree.evaluation import array_fingerprint, prepare_folds

FAMILIES = ("lr", "leaf_lr", "lgb", "xgb", "cat")


def load_data(loader, data_dir):
    if loader == "competition":
        from competition_data import load_competition
        return load_competition(data_dir, holdout_fraction=.2, split_seed=0)
    from kaggle_s6e10_data import load_s6e10
    return load_s6e10(data_dir)


def convert(source, work_dir, data_dir, loader="competition",
            seed=42, n_folds=5, dry_run=False, select=True):
    source, work_dir = Path(source), Path(work_dir)
    source_db, source_oof = source / "ens.db", source / "oof"
    if not source_db.is_file() or not source_oof.is_dir():
        raise FileNotFoundError(f"Expected both {source_db} and {source_oof}/")
    if source.resolve() == work_dir.resolve():
        raise ValueError("Keep legacy source and destination directories separate")

    data = load_data(loader, data_dir)
    X, y, folds = prepare_folds(data.X_dev, data.y_dev, seed=seed, n_splits=n_folds)
    source_storage = f"sqlite:///{source_db.resolve().as_posix()}"
    checked = []
    for name in FAMILIES:
        try:
            original = optuna.load_study(study_name=name, storage=source_storage)
        except KeyError:
            continue
        for trial in original.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
                continue
            source_path = source_oof / f"{name}_{trial.number}.npz"
            if not source_path.exists():
                print(f"Skipping {name}_{trial.number}: OOF file not found")
                continue
            with np.load(source_path, allow_pickle=False) as stored:
                if not {"oof", "hold", "test"}.issubset(stored.files):
                    raise ValueError(f"Missing oof/hold/test in {source_path}")
                oof, hold, test = [np.asarray(stored[k], dtype=float)
                                   for k in ("oof", "hold", "test")]
            for label, values, size in [
                ("oof", oof, len(y)),
                ("hold", hold, len(data.y_holdout)),
                ("test", test, len(data.X_test))
            ]:
                if values.shape != (size,) or not np.isfinite(values).all():
                    raise ValueError(f"{source_path}: invalid {label} predictions")
            # The strongest available alignment check: the stored original
            # fold scores must reproduce with the current y and fold indices.
            scores = [float(roc_auc_score(y[va], oof[va])) for _, va in folds]
            if abs(float(np.mean(scores)) - float(trial.value)) > 2e-6:
                raise ValueError(
                    f"{source_path}: AUC mismatch against current labels/folds. "
                    "Original CV rows may be differently ordered; refusing to import."
                )
            prior = trial.user_attrs.get("fold_auc")
            if prior is not None and (
                len(prior) != len(scores) or
                not np.allclose(prior, scores, rtol=0, atol=2e-6)
            ):
                raise ValueError(f"{source_path}: stored per-fold AUC mismatch")
            checked.append((name, trial, oof, hold, test))
    if not checked:
        raise ValueError("No completed legacy trials with compatible prediction arrays")
    print(f"Validated {len(checked)} candidates using the current labels and folds.")
    print("Holdout/test row ordering still relies on using unchanged original CSVs.")
    if dry_run:
        return 0

    auto = AutoMLTree(work_dir=work_dir, backends=FAMILIES, n_folds=n_folds,
                      seed=seed, min_candidate_auc=None)
    auto.initialize(X, y)
    hold_hash, test_hash = array_fingerprint(data.X_holdout), array_fingerprint(data.X_test)
    imported = 0
    for name, trial, oof, hold, test in checked:
        study = auto.runner.study(name)
        origin = f"previous:{name}_{trial.number}"
        previous = [t for t in study.trials if t.user_attrs.get("conversion_source") == origin]
        if previous:
            existing_path = auto.runner.prediction_path(f"{name}_{previous[0].number}")
            with np.load(existing_path, allow_pickle=False) as old:
                if not np.array_equal(old["oof"], oof):
                    raise ValueError(f"{origin} has changed since its previous import")
            print(f"Already imported: {origin}")
            continue
        attributes = dict(trial.user_attrs)
        attributes.update(conversion_source=origin,
                          cached_splits=["hold", "test"],
                          original_cv_auc=float(trial.value))
        new_trial = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE, value=float(trial.value),
            params=dict(trial.params), distributions=dict(trial.distributions),
            user_attrs=attributes,
        )
        new_number = len(study.trials)
        target = auto.runner.prediction_path(f"{name}_{new_number}")
        if target.exists():
            raise FileExistsError(f"{target}: unregistered existing OOF; refusing overwrite")
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            np.savez_compressed(tmp, oof=oof, hold=hold, test=test,
                                hold_hash=hold_hash, test_hash=test_hash)
        try:
            os.replace(tmp_path, target)
            study.add_trial(new_trial)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        imported += 1
        print(f"{origin} -> {name}_{new_number} | AUC {trial.value:.6f}")

    # The standard competition submission script requires saved preprocessing
    # configuration in addition to the dataset fingerprint.
    if loader == "competition":
        path = work_dir / "competition_config.json"
        config = {
            "competition": "playground-series-s6e10", "seed": seed,
            "n_folds": n_folds, "split_seed": 0, "holdout_fraction": .2,
            "corr_max": .99, "feature_names": list(data.feature_names),
            "backends": list(FAMILIES),
        }
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8"))
            for key in ("competition", "seed", "n_folds", "split_seed",
                        "holdout_fraction", "corr_max", "feature_names"):
                if current.get(key) != config[key]:
                    raise ValueError(f"Incompatible existing competition config: {key}")
            config["backends"] = sorted(set(current["backends"]) | set(FAMILIES))
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    if select:
        auto.select_ensemble()
        if auto.ensemble.best is not None:
            print(f"Starting ensemble: {len(auto.best_ensemble_.members)} models, "
                  f"CV AUC {auto.best_ensemble_.score:.6f}")

    print(f"Imported {imported} new candidates into {work_dir}/")
    print("Conversion complete. You may archive the old source directory.")
    return imported


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("legacy"))
    p.add_argument("--work-dir", type=Path, default=Path("runs/s6e10"))
    p.add_argument("--data-dir", type=Path, default=Path("playground-series-s6e10"))
    p.add_argument("--loader", choices=("competition", "kaggle"), default="competition",
                   help="Use kaggle for kaggle_s6e10_train.py / runs/s6e10_baseline")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-select", action="store_true")
    args = p.parse_args()
    convert(args.source, args.work_dir, args.data_dir,
            loader=args.loader, dry_run=args.dry_run, select=not args.no_select)


if __name__ == "__main__":
    main()
