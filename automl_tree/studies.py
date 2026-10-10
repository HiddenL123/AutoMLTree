"""Persistent Optuna studies; model generation happens in BATCHES."""
from __future__ import annotations
import os
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from .backends import predict_positive, REGISTRY, FeatureMaskModel


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

    def feature_importances(self):
        """Use saved LightGBM importances for feature-ablation suggestions."""
        trials = sorted((t for t in self.study('lgb').trials
                         if t.state == optuna.trial.TrialState.COMPLETE
                         and t.value is not None), key=lambda t: -t.value)
        for trial in trials:
            member = f'lgb_{trial.number}'
            scores = []
            for fold in range(len(self.folds)):
                path = self.model_path(member, fold)
                if not path.exists():
                    continue
                model = joblib.load(path)
                if isinstance(model, FeatureMaskModel):
                    columns = model.columns
                    model = model.model
                else:
                    columns = np.arange(self.X.shape[1])
                values = getattr(model, 'feature_importances_', None)
                if values is None or len(values) != len(columns):
                    continue
                importance = np.zeros(self.X.shape[1])
                importance[columns] = values
                scores.append(importance)
            if scores:
                return np.mean(scores, axis=0)
        return None

    def _experiment_setup(self, trial, name, operator):
        params = self.registry[name].suggest(trial)
        columns = np.arange(self.X.shape[1], dtype=np.int64)
        note = None
        if operator != 'standard' and name != 'lgb':
            raise ValueError(f'{operator} not supported for backend {name}')
        if name == 'lgb':
            if operator == 'dart':
                params['boosting_type'] = 'dart'
            elif operator == 'rf':
                params['boosting_type'] = 'rf'
                params['subsample'] = min(float(params['subsample']), .9)
                params['colsample_bytree'] = min(float(params['colsample_bytree']), .9)
                params['subsample_freq'] = 1
            elif operator == 'extra_trees':
                params['extra_trees'] = True
            elif operator == 'feature_subset':
                fraction = trial.suggest_float('feature_subset_fraction', .55, .85)
                rng = np.random.default_rng(self.seed + trial.number * 104729 + 71)
                k = max(1, min(len(columns)-1, round(len(columns)*fraction)))
                if len(columns) > 1:
                    columns = np.sort(rng.choice(columns, k, replace=False))
            elif operator == 'feature_ablation':
                fraction = trial.suggest_float('feature_ablation_fraction', .05, .25)
                importance = self.feature_importances()
                if importance is None:
                    raise ValueError('A trained LightGBM model is required for ablation')
                if len(columns) > 1:
                    count = max(1, min(len(columns)-1, round(len(columns)*fraction)))
                    removed = np.argsort(-importance, kind='stable')[:count]
                    columns = np.setdiff1d(columns, removed, assume_unique=True)
                note = 'Dropped highest-importance features from prior LightGBM model'
            elif operator != 'standard':
                raise ValueError(f'Unknown LightGBM operator: {operator}')
        return params, columns, note

    def _objective(self, name, operator='standard'):
        backend = self.registry[name]

        def objective(trial):
            start = time.monotonic()
            trial.set_user_attr('operator', operator)
            member = f'{name}_{trial.number}'
            tmp_dir = self.models_dir / f'.pending_{member}'
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True)
            oof = np.zeros(len(self.y), dtype=float)
            scores, best_iters = [], []
            try:
                params, columns, note = self._experiment_setup(trial, name, operator)
                trial.set_user_attr('feature_indices', columns.tolist())
                if note:
                    trial.set_user_attr('feature_note', note)
                X_view = self.X[:, columns]
                for fold, (tr, va) in enumerate(self.folds):
                    model, n_iter = backend.fit(X_view, self.y, tr, params,
                                                self.seed + fold, self.early_stop)
                    wrapped = FeatureMaskModel(model, columns)
                    pv = predict_positive(wrapped, self.X[va])
                    if len(pv) != len(va) or not np.all(np.isfinite(pv)):
                        raise ValueError(f'{member} fold {fold}: invalid prediction')
                    oof[va] = pv
                    scores.append(float(roc_auc_score(self.y[va], pv)))
                    best_iters.append(int(n_iter))
                    joblib.dump(wrapped, tmp_dir / f'fold_{fold}.joblib')
                    trial.report(float(np.mean(scores)), fold)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                trial.set_user_attr('best_iters', best_iters)
                trial.set_user_attr('fold_auc', scores)
                final_models = self.models_dir / member
                if final_models.exists():
                    raise FileExistsError(f'{final_models}: refusing overwrite')
                os.replace(tmp_dir, final_models)
                final_oof = self.prediction_path(member)
                if final_oof.exists():
                    raise FileExistsError(f'{final_oof}: refusing overwrite')
                tmp_oof = self.pred_dir / f'.{member}.tmp'
                with open(tmp_oof, 'wb') as f:
                    np.savez_compressed(f, oof=oof)
                os.replace(tmp_oof, final_oof)
                return float(np.mean(scores))
            finally:
                trial.set_user_attr('training_seconds', time.monotonic()-start)
                shutil.rmtree(tmp_dir, ignore_errors=True)
        return objective

    def run_experiment_batch(self, experiments):
        """Perform one newly recommended trial per action in persistent studies."""
        touched, outcomes = {}, []
        for action in experiments:
            study = self.study(action.backend)
            if self.verbose:
                print(f'[scheduler] {action.backend}:{action.operator} -> '
                      f'trial {len(study.trials)}')
            study.optimize(self._objective(action.backend, action.operator),
                           n_trials=1, catch=(Exception,))
            trial = study.trials[-1]
            member = f'{action.backend}_{trial.number}'
            outcomes.append((action, trial, member))
            touched[action.backend] = study
        return touched, outcomes

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
