"""Persistent Optuna studies; model generation happens in BATCHES."""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import joblib
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from .backends import predict_positive, REGISTRY


class StudyRunner:
    def __init__(self, work_dir, X, y, folds, seed=42, early_stop=75,
                 registry=None, verbose=True):
        self.work_dir = Path(work_dir)
        self.X, self.y, self.folds = X, y, folds
        self.seed, self.early_stop = seed, early_stop
        self.registry = registry or REGISTRY
        self.pred_dir = self.work_dir / 'oof'
        self.models_dir = self.work_dir / 'models'
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.storage = f'sqlite:///{(self.work_dir / "ens.db").resolve().as_posix()}'
        self.verbose = verbose

    def study(self, name):
        if name not in self.registry:
            raise ValueError(f'Unknown backend: {name}')
        study = optuna.create_study(
            study_name=name, direction='maximize', storage=self.storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1))
        # SQLite saves previous trials, NOT the sampler RNG state. On resume,
        # reseed from the number of existing trials to avoid replaying the
        # same startup TPE/random suggestions after process restarts.
        study.sampler = optuna.samplers.TPESampler(seed=self.seed + len(study.trials))
        return study

    def prediction_path(self, member):
        return self.pred_dir / f'{member}.npz'

    def model_path(self, member, fold):
        return self.models_dir / member / f'fold_{fold}.joblib'

    def _objective(self, name):
        backend = self.registry[name]

        def objective(trial):
            params = backend.suggest(trial)
            member = f'{name}_{trial.number}'
            tmp_dir = self.models_dir / f'.pending_{member}'
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True)
            oof = np.zeros(len(self.y), dtype=float)
            scores, best_iters = [], []
            try:
                for fold, (tr, va) in enumerate(self.folds):
                    model, n_iter = backend.fit(self.X, self.y, tr, params,
                                                self.seed + fold, self.early_stop)
                    pv = predict_positive(model, self.X[va])
                    if len(pv) != len(va) or not np.all(np.isfinite(pv)):
                        raise ValueError(f'{member} fold {fold}: bad predictions')
                    oof[va] = pv
                    scores.append(float(roc_auc_score(self.y[va], pv)))
                    best_iters.append(int(n_iter))
                    joblib.dump(model, tmp_dir / f'fold_{fold}.joblib')
                    trial.report(float(np.mean(scores)), fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                trial.set_user_attr('best_iters', best_iters)
                trial.set_user_attr('fold_auc', scores)
                # Models and OOF are saved before Optuna marks the trial COMPLETE.
                final_models = self.models_dir / member
                if final_models.exists():
                    raise FileExistsError(f'{final_models}: refusing to overwrite')
                os.replace(tmp_dir, final_models)
                final_oof = self.prediction_path(member)
                if final_oof.exists():
                    raise FileExistsError(f'{final_oof}: refusing to overwrite')
                temp_oof = self.pred_dir / f'.{member}.tmp'
                with open(temp_oof, 'wb') as f:
                    np.savez_compressed(f, oof=oof)
                os.replace(temp_oof, final_oof)
                return float(np.mean(scores))
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        return objective

    def run_batch(self, trials_per_family: dict[str, int]):
        """Every invocation ADDS the specified count of trials to existing studies."""
        results = {}
        for name, additional in trials_per_family.items():
            if additional < 0:
                raise ValueError('Trial counts must be nonnegative')
            study = self.study(name)
            before = len(study.trials)
            if self.verbose:
                print(f'[{name}] continuing at trial count {before}, adding {additional}')
            if additional:
                study.optimize(self._objective(name), n_trials=additional)
            completed = [t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
            if self.verbose:
                print(f'[{name}] completed {len(completed)} / {len(study.trials)}; '
                      f'best {max((t.value for t in completed), default=float("nan")):.6f}')
            results[name] = study
        return results

    def run_to_total(self, total_trials_per_family: dict[str, int]):
        """Continue existing studies until each has TARGET TOTAL TERMINAL trials.

        COMPLETE + PRUNED + FAIL count toward the target; RUNNING do not.
        """
        add = {}
        for name, total in total_trials_per_family.items():
            study = self.study(name)
            finished = sum(t.state.is_finished() for t in study.trials)
            add[name] = max(0, int(total) - finished)
        return self.run_batch(add)

    def candidates(self, family_names, min_auc=None):
        out = []
        for name in family_names:
            for trial in self.study(name).trials:
                if trial.state != optuna.trial.TrialState.COMPLETE or trial.value is None:
                    continue
                if min_auc is not None and trial.value < min_auc:
                    continue
                member = f'{name}_{trial.number}'
                if not self.prediction_path(member).exists():
                    raise FileNotFoundError(f'{member} is COMPLETE in Optuna but OOF file '
                                            f'{self.prediction_path(member)} is missing.')
                out.append((float(trial.value), member))
        return sorted(out, key=lambda z: (-z[0], z[1]))

    def load_oof(self, member):
        with np.load(self.prediction_path(member)) as saved:
            result = np.asarray(saved['oof'], dtype=float)
        if result.shape != (len(self.y),) or not np.isfinite(result).all():
            raise ValueError(f'{member}: OOF shape/values mismatch')
        return result

    def predict_member(self, member, X):
        preds = []
        for fold in range(len(self.folds)):
            path = self.model_path(member, fold)
            if not path.exists():
                raise FileNotFoundError(
                    f'Missing fitted model {path}. Older studies may have OOF predictions '
                    'but no serialized fold models. Those models must be refitted for '
                    'arbitrary new-data inference.')
            model = joblib.load(path)
            preds.append(predict_positive(model, X))
        return np.mean(preds, axis=0)
