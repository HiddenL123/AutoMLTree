"""Public interface for batch-oriented, resumable binary classification AutoML."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from .backends import REGISTRY
from .evaluation import numeric_array, prepare_folds, ensure_manifest, rank_new
from .studies import StudyRunner
from .ensemble import EnsembleSearch
from .scheduler import ExperimentScheduler


class AutoMLTree:
    def __init__(self, work_dir='automl_run', backends=('lgb',), n_folds=5,
                 seed=42, early_stop=75, min_candidate_auc=None,
                 corr_max=.99, beam_width=5, search_depth=2,
                 candidate_limit=30, min_member_gain=1e-6,
                 blend_trials_per_batch=20, verbose=True, allow_legacy=False,
                 registry=None):
        self.work_dir = Path(work_dir)
        self.backends = tuple(backends)
        self.registry = registry or REGISTRY
        if not self.backends or any(b not in self.registry for b in self.backends):
            raise ValueError('All selected backends must be in the registry.')
        self.n_folds, self.seed, self.early_stop = n_folds, seed, early_stop
        self.min_candidate_auc = min_candidate_auc
        self.corr_max, self.beam_width = corr_max, beam_width
        self.search_depth, self.candidate_limit = search_depth, candidate_limit
        self.min_member_gain = min_member_gain
        self.blend_trials_per_batch = blend_trials_per_batch
        self.verbose, self.allow_legacy = verbose, allow_legacy
        self.studies = {}
        self.runner = None
        self.ensemble = None
        self.scheduler = None

    def initialize(self, X, y, folds=None):
        """Bind data to a persistent run, validating the dataset/fold fingerprint."""
        X, y, folds = prepare_folds(X, y, self.seed, self.n_folds, folds)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        ensure_manifest(self.work_dir, X, y, folds, allow_legacy=self.allow_legacy)
        self.runner = StudyRunner(self.work_dir, X, y, folds, seed=self.seed,
                                  early_stop=self.early_stop,
                                  registry=self.registry, verbose=self.verbose)
        self.ensemble = EnsembleSearch(self.runner, corr_max=self.corr_max,
                                       beam_width=self.beam_width,
                                       search_depth=self.search_depth,
                                       candidate_limit=self.candidate_limit,
                                       min_gain=self.min_member_gain)
        self.scheduler = ExperimentScheduler(self.work_dir, verbose=self.verbose)
        return self

    def fit(self, X, y, trials_per_family=5, folds=None):
        """Initialize if needed, then ADD one batch of Optuna trials per family."""
        if self.runner is None:
            self.initialize(X, y, folds)
        # A new call cannot silently switch datasets/folds.
        else:
            XX, yy, ff = prepare_folds(X, y, self.seed, self.n_folds, folds)
            ensure_manifest(self.work_dir, XX, yy, ff, allow_legacy=self.allow_legacy)
        counts = (dict.fromkeys(self.backends, int(trials_per_family))
                  if isinstance(trials_per_family, int) else dict(trials_per_family))
        self.studies = self.runner.run_batch(counts)
        return self.select_ensemble()

    def try_model(self, trials_per_family=5):
        """Generate one additional BATCH then update ensemble; needs initialize()."""
        if self.runner is None:
            raise RuntimeError('Call initialize(X, y) before try_model().')
        counts = (dict.fromkeys(self.backends, int(trials_per_family))
                  if isinstance(trials_per_family, int) else dict(trials_per_family))
        self.studies = self.runner.run_batch(counts)
        return self.select_ensemble()

    def fit_scheduled(self, X, y, batch_size=5, folds=None,
                      blend_trials_per_mode=3):
        """Run a recommended batch; evaluate and checkpoint its ensemble."""
        if self.runner is None:
            self.initialize(X, y, folds)
        else:
            XX, yy, ff = prepare_folds(X, y, self.seed, self.n_folds, folds)
            ensure_manifest(self.work_dir, XX, yy, ff, allow_legacy=self.allow_legacy)
        if batch_size < 0:
            raise ValueError('batch_size must be nonnegative')
        previous_best = self.ensemble.best
        plan = self.scheduler.plan_batch(
            self.backends, batch_size,
            allow_ablation=(self.runner.feature_importances() is not None)
            if 'lgb' in self.backends else False)
        self.studies, outcomes = self.runner.run_experiment_batch(plan)
        for action, trial, member in outcomes:
            import optuna
            if (trial.state == optuna.trial.TrialState.COMPLETE
                    and self.runner.prediction_path(member).exists()):
                gain = self.ensemble.estimate_counterfactual_gain(member, previous_best)
            else:
                gain = 0.
            self.scheduler.record(
                member, action, gain,
                trial.user_attrs.get('training_seconds', 0.),
                auc=trial.value,
                baseline_auc=previous_best.score if previous_best else .5)
        candidates = self.runner.candidates(self.backends, self.min_candidate_auc)
        if candidates:
            self.ensemble.search_batch(candidates)
            self.ensemble.compare_blend_modes(trials_per_mode=blend_trials_per_mode,
                                              seed=self.seed)
            self.ensemble.refine_weights(self.blend_trials_per_batch, seed=self.seed)
        if self.verbose:
            print('[scheduler] planned:', ', '.join(self.scheduler.label(a) for a in plan))
            if self.ensemble.best is not None:
                print(f'[scheduler] best AUC {self.ensemble.best.score:.6f}, '
                      f'mode {self.ensemble.best.mode}, '
                      f'members {len(self.ensemble.best.members)}')
        return self

    def run_scheduled_batches(self, X, y, n_batches, batch_size=5, folds=None,
                              max_seconds=None, blend_trials_per_mode=3):
        """Run bounded scheduled batches synchronously; restart to resume."""
        start = time.monotonic()
        for _ in range(n_batches):
            if max_seconds is not None and time.monotonic()-start >= max_seconds:
                break
            self.fit_scheduled(X, y, batch_size=batch_size, folds=folds,
                               blend_trials_per_mode=blend_trials_per_mode)
        return self

    @property
    def experiment_summary_(self):
        if self.scheduler is None:
            raise RuntimeError('Initialize first')
        return self.scheduler.summary()

    def run_batches(self, X, y, n_batches, trials_per_family=5, folds=None,
                    max_seconds=None):
        """Synchronous anytime search; checkpoint after each completed batch."""
        if self.runner is None:
            self.initialize(X, y, folds)
        else:
            XX, yy, ff = prepare_folds(X, y, self.seed, self.n_folds, folds)
            ensure_manifest(self.work_dir, XX, yy, ff, allow_legacy=self.allow_legacy)
        start = time.monotonic()
        for _ in range(n_batches):
            if max_seconds is not None and time.monotonic() - start >= max_seconds:
                break
            self.try_model(trials_per_family)
        return self

    def fit_to_total(self, X, y, target_trials_per_family, folds=None):
        """Resume studies, doing only the trials needed to reach lifetime totals."""
        if self.runner is None:
            self.initialize(X, y, folds)
        else:
            XX, yy, ff = prepare_folds(X, y, self.seed, self.n_folds, folds)
            ensure_manifest(self.work_dir, XX, yy, ff, allow_legacy=self.allow_legacy)
        counts = (dict.fromkeys(self.backends, int(target_trials_per_family))
                  if isinstance(target_trials_per_family, int)
                  else dict(target_trials_per_family))
        self.studies = self.runner.run_to_total(counts)
        return self.select_ensemble()

    def select_ensemble(self):
        if self.runner is None:
            raise RuntimeError('Call initialize or fit first.')
        candidates = self.runner.candidates(self.backends, self.min_candidate_auc)
        if not candidates:
            if self.verbose:
                print('No eligible completed trials yet.')
            return self
        prior_score = self.ensemble.best.score if self.ensemble.best else None
        self.ensemble.search_batch(candidates)
        self.ensemble.refine_weights(self.blend_trials_per_batch, seed=self.seed)
        if self.verbose and self.ensemble.best:
            best = self.ensemble.best
            print(f'Best ensemble: AUC {best.score:.6f}, members {len(best.members)}'
                  + (f', previous {prior_score:.6f}' if prior_score is not None else ''))
            for member, weight in zip(best.members, best.weights):
                print(f'  {member}: {weight:.4f}')
        return self

    @property
    def best_ensemble_(self):
        if self.ensemble is None or self.ensemble.best is None:
            raise RuntimeError('No ensemble available. Run fit first.')
        return self.ensemble.best

    def predict_proba(self, X, prediction_set=None):
        """Discrimination scores compatible with sklearn shape, NOT calibrated."""
        X = numeric_array(X)
        if X.shape[1] != self.runner.X.shape[1]:
            raise ValueError('Feature count differs from training data.')
        state = self.best_ensemble_
        score = np.zeros(len(X), dtype=float)
        from scipy.special import expit, logit
        for member, weight in zip(state.members, state.weights):
            raw = self.runner.predict_member(member, X, prediction_set=prediction_set)
            if state.mode == 'rank':
                transformed = rank_new(raw, self.runner.load_oof(member))
            elif state.mode == 'probability':
                transformed = raw
            elif state.mode == 'logit':
                transformed = logit(np.clip(raw, 1e-6, 1.-1e-6))
            else:
                raise ValueError(f'Unknown blending mode: {state.mode}')
            score += weight * transformed
        if state.mode == 'logit':
            score = expit(score)
        score = np.clip(score, 0., 1.)
        return np.column_stack((1.-score, score))

    def predict_scores(self, X, prediction_set=None):
        return self.predict_proba(X, prediction_set=prediction_set)[:, 1]

    def predict(self, X, threshold=.5, prediction_set=None):
        """Threshold rank scores. For calibrated probabilities, add calibration separately."""
        return (self.predict_scores(X, prediction_set=prediction_set) >= threshold).astype(int)
