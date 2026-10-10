"""Lightweight graph/beam search over sets of pre-trained prediction columns."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import optuna
from scipy.special import logit
from .evaluation import ranks, fold_auc, corr_spearman


@dataclass(frozen=True)
class EnsembleState:
    members: tuple[str, ...]
    weights: tuple[float, ...]
    score: float
    mode: str = 'rank'


def state_from(members, weights, score, mode='rank'):
    if mode not in ('rank', 'probability', 'logit'):
        raise ValueError('Unknown blending mode')
    pairs = sorted(zip(members, weights))
    if not pairs:
        raise ValueError('An ensemble must have at least one model.')
    w = np.array([v for _, v in pairs], dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Invalid ensemble weights.')
    w /= w.sum()
    return EnsembleState(tuple(k for k, _ in pairs), tuple(float(x) for x in w),
                         float(score), mode)


class EnsembleSearch:
    def __init__(self, runner, corr_max=.99, beam_width=5,
                 alpha_grid=None, min_gain=1e-6, candidate_limit=30,
                 max_swaps=3, search_depth=2):
        if not -1 < corr_max <= 1:
            raise ValueError('corr_max must be > -1 and <= 1')
        self.runner = runner
        self.y, self.folds = runner.y, runner.folds
        self.corr_max = corr_max
        self.beam_width = beam_width
        self.min_gain = min_gain
        self.candidate_limit = candidate_limit
        self.max_swaps = max_swaps
        self.search_depth = search_depth
        self.alpha_grid = np.asarray(
            alpha_grid if alpha_grid is not None
            else np.r_[np.geomspace(.002, .10, 10), np.linspace(.12, .80, 12)],
            dtype=float)
        if np.any((self.alpha_grid <= 0) | (self.alpha_grid >= 1)):
            raise ValueError('alphas must be strictly between 0 and 1')
        self.best = None
        self.visited = {}  # subset -> best weighted state
        self.checkpoint = runner.work_dir / 'best_ensemble.json'
        self._restore()

    @lru_cache(maxsize=128)
    def raw(self, member):
        return self.runner.load_oof(member)

    @lru_cache(maxsize=128)
    def scaled(self, member, mode='rank'):
        pred = self.raw(member)
        if mode == 'rank':
            return ranks(pred)
        if mode == 'probability':
            return pred
        if mode == 'logit':
            return logit(np.clip(pred, 1e-6, 1.-1e-6))
        raise ValueError(f'Unknown blending mode: {mode}')

    @lru_cache(maxsize=16384)
    def corr(self, a, b):
        return corr_spearman(self.raw(a), self.raw(b))

    def score(self, members, weights, mode='rank'):
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
        blended = sum(w * self.scaled(m, mode) for m, w in zip(members, weights))
        return fold_auc(self.y, self.folds, blended)

    def _save(self):
        if self.best is None:
            return
        obj = {'members': list(self.best.members), 'weights': list(self.best.weights),
               'score': self.best.score, 'mode': self.best.mode}
        tmp = self.checkpoint.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(obj, indent=2))
        tmp.replace(self.checkpoint)

    def _restore(self):
        if self.checkpoint.exists():
            x = json.loads(self.checkpoint.read_text())
            for name in x['members']:
                if not self.runner.prediction_path(name).exists():
                    raise FileNotFoundError(f'Checkpoint references missing {name}')
            s = state_from(x['members'], x['weights'], x['score'], x.get('mode', 'rank'))
            # Check saved result really belongs to these predictions and folds.
            if abs(self.score(s.members, s.weights, s.mode) - s.score) > 1e-8:
                raise ValueError('Saved ensemble score does not match current data/folds.')
            if not self._legal(s.members):
                raise ValueError('Saved ensemble violates current corr_max. '
                                 'Use a new work_dir when changing ensemble constraints.')
            self.best = s

    def _register(self, state):
        key = (state.members, state.mode)
        old = self.visited.get(key)
        if old is None or state.score > old.score + 1e-12:
            self.visited[key] = state
        if self.best is None or state.score > self.best.score + self.min_gain:
            self.best = state
            self._save()

    def _legal(self, members):
        return all(self.corr(a, b) < self.corr_max
                   for i, a in enumerate(members) for b in members[i+1:])

    def _add_best_alpha(self, parent, newcomer, remove=()):
        survivors = [m for m in parent.members if m not in remove]
        if newcomer in survivors:
            return None
        if not self._legal(survivors + [newcomer]):
            return None
        if not survivors:
            return state_from([newcomer], [1], self.score([newcomer], [1], parent.mode), parent.mode)
        pw = np.array([w for m, w in zip(parent.members, parent.weights)
                       if m not in remove])
        pw /= pw.sum()
        base = sum(w * self.scaled(m, parent.mode) for m, w in zip(survivors, pw))
        cand = self.scaled(newcomer, parent.mode)
        result = None
        tested = set()

        def evaluate_alphas(values, current):
            for alpha in values:
                alpha = float(alpha)
                if alpha in tested or not 0 < alpha < 1:
                    continue
                tested.add(alpha)
                blend = (1-alpha)*base + alpha*cand
                value = fold_auc(self.y, self.folds, blend)
                if current is None or value > current.score:
                    current = state_from(survivors+[newcomer],
                                         list((1-alpha)*pw)+[alpha], value, parent.mode)
            return current

        result = evaluate_alphas(self.alpha_grid, result)
        # Cheap local refinement around the strongest grid location. AUC is
        # stepwise (not differentiable), so variance is not an alpha optimum.
        alpha_star = dict(zip(result.members, result.weights))[newcomer]
        idx = int(np.argmin(np.abs(self.alpha_grid-alpha_star)))
        left = self.alpha_grid[max(0, idx-1)]
        right = self.alpha_grid[min(len(self.alpha_grid)-1, idx+1)]
        result = evaluate_alphas(np.linspace(left, right, 9), result)
        return result

    def _remove(self, parent, discarded):
        names = [m for m in parent.members if m != discarded]
        if not names:
            return None
        weights = [w for m, w in zip(parent.members, parent.weights) if m != discarded]
        return state_from(names, weights, self.score(names, weights, parent.mode), parent.mode)

    def _neighbors(self, state, candidates):
        generated = []
        for m in state.members:
            candidate = self._remove(state, m)
            if candidate is not None:
                generated.append(candidate)
        # Try removing low-impact members first for ordinary single swaps.
        contributions = []
        for m in state.members:
            without = self._remove(state, m)
            contributions.append((state.score - (without.score if without else -np.inf), m))
        removable = [m for _, m in sorted(contributions)[:self.max_swaps]]
        for m in candidates:
            if m in state.members:
                continue
            conflicts = [old for old in state.members
                         if self.corr(old, m) >= self.corr_max]
            if not conflicts:
                generated.append(self._add_best_alpha(state, m))
                for old in removable:
                    generated.append(self._add_best_alpha(state, m, remove=[old]))
            else:
                # Your conflict-replacement rule: remove ALL violating members.
                generated.append(self._add_best_alpha(state, m, remove=conflicts))
        return [s for s in generated if s is not None]

    def search_batch(self, candidate_scores):
        """Search after a generation batch; models already stored in persistent Optuna DB."""
        if not candidate_scores:
            return self.best
        ranked = [name for _, name in candidate_scores]
        # Default is compute-limited; 0/None means all eligible candidates.
        if self.candidate_limit:
            ranked = ranked[:self.candidate_limit]
        singles = [state_from([m], [1], self.score([m], [1]))
                   for m in ranked[:max(self.beam_width, 1)]]
        if self.best is not None:
            singles.append(self.best)
        for state in singles:
            self._register(state)
        beam = sorted(singles, key=lambda s: -s.score)[:self.beam_width]
        for depth in range(self.search_depth):
            new = {}
            for parent in beam:
                for state in self._neighbors(parent, ranked):
                    key = (state.members, state.mode)
                    if key not in new or state.score > new[key].score:
                        new[key] = state
            for state in new.values():
                self._register(state)
            # Keep some alternative subsets even when they temporarily score worse.
            beam = sorted({**{(s.members, s.mode): s for s in beam}, **new}.values(),
                          key=lambda s: -s.score)[:self.beam_width]
            if not beam:
                break
        return self.best

    def estimate_counterfactual_gain(self, member, reference):
        """Measure candidate reward against FROZEN pre-batch ensemble."""
        if reference is None:
            return max(0., self.score([member], [1]) - .5)
        if member in reference.members:
            return 0.
        conflicts = [m for m in reference.members
                     if self.corr(m, member) >= self.corr_max]
        state = self._add_best_alpha(reference, member, remove=conflicts)
        return max(0., state.score-reference.score) if state is not None else 0.

    def compare_blend_modes(self, modes=('rank', 'probability', 'logit'),
                            trials_per_mode=3, seed=42):
        """Cheap combination experiments using cached OOF predictions only."""
        if self.best is None or len(self.best.members) <= 1:
            return self.best
        reference = self.best
        for mode in modes:
            state = state_from(reference.members, reference.weights,
                               self.score(reference.members, reference.weights, mode),
                               mode)
            self._register(state)
            if trials_per_mode:
                self.refine_weights(trials_per_mode, seed=seed, base_state=state)
        return self.best

    def refine_weights(self, additional_trials=30, seed=42, base_state=None):
        """Resume an Optuna blend study for each (mode, membership) pair."""
        start = base_state or self.best
        if start is None or len(start.members) <= 1 or additional_trials <= 0:
            return self.best
        members = start.members
        # Preserve the original rank blend study name for existing users.
        key = hashlib.sha256((start.mode + ':' + '|'.join(members)).encode()).hexdigest()[:16]
        study = optuna.create_study(
            direction='maximize', study_name=f'blend_{key}',
            storage=self.runner.storage, load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=seed))
        if len(study.trials) == 0:
            study.enqueue_trial({f'w{i}': float(w) for i, w in enumerate(start.weights)})

        def objective(trial):
            w = np.array([trial.suggest_float(f'w{i}', 0., 1.)
                          for i in range(len(members))], dtype=float)
            if w.sum() < 1e-9:
                raise optuna.TrialPruned()
            return self.score(members, w, start.mode)

        study.optimize(objective, n_trials=int(additional_trials))
        if study.best_value > start.score + self.min_gain:
            w = [study.best_params[f'w{i}'] for i in range(len(members))]
            self._register(state_from(members, w, study.best_value, start.mode))
        return self.best
