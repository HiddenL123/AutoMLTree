"""Consistent shared folds, AUC and stable rank-based blending."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def numeric_array(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or np.isinf(X).any():
        raise ValueError('X must be a 2-D numeric matrix (NaN permitted, inf forbidden).')
    return X


def dataset_fingerprint(X, y, folds):
    h = hashlib.sha256()
    for arr in [X, y] + [a for pair in folds for a in pair]:
        z = np.ascontiguousarray(arr)
        h.update(str(z.shape).encode())
        h.update(str(z.dtype).encode())
        h.update(z.tobytes())
    return h.hexdigest()


def prepare_folds(X, y, seed, n_splits, folds=None):
    X = numeric_array(X)
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError('y must be a vector with one value per row.')
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError('V1 supports binary targets encoded as 0 and 1 only.')
    if folds is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y))
    folds = [(np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
             for tr, va in folds]
    observed = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        if np.intersect1d(tr, va).size or not len(tr) or not len(va):
            raise ValueError('Fold train/validation indices overlap or are empty.')
        if (np.min(tr) < 0 or np.max(tr) >= len(y) or
            np.min(va) < 0 or np.max(va) >= len(y)):
            raise ValueError('Fold indices out of bounds.')
        if len(np.unique(y[tr])) != 2 or len(np.unique(y[va])) != 2:
            raise ValueError('Every train and validation fold must contain both classes.')
        observed[va] += 1
    if not np.all(observed == 1):
        raise ValueError('Each row must occur in exactly one validation fold.')
    return X, y.astype(int), folds


def fold_auc(y, folds, oof):
    return float(np.mean([roc_auc_score(y[va], oof[va]) for _, va in folds]))


def ranks(oof):
    # Stable training-reference empirical CDF (does not depend on inference batch).
    return rankdata(oof, method='average') / len(oof)


def rank_new(new_predictions, reference_oof):
    ref = np.sort(reference_oof)
    # Average-rank convention for ties, consistent with ranks(reference_oof).
    left = np.searchsorted(ref, new_predictions, side='left')
    right = np.searchsorted(ref, new_predictions, side='right')
    return np.clip((left + right + 1) / (2 * len(ref)), 0.0, 1.0)


def corr_spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 1.0
    c = float(spearmanr(a, b).statistic)
    return c if np.isfinite(c) else 1.0   # constant vectors are conservatively redundant


def ensure_manifest(work_dir: Path, X, y, folds, allow_legacy=False):
    path = work_dir / 'dataset.json'
    fp = dataset_fingerprint(X, y, folds)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['fingerprint'] != fp:
            raise ValueError('Dataset/folds differ from this run. Use a NEW work_dir.')
        return
    # Legacy Optuna DBs cannot identify which dataset/folds produced old trials.
    if (work_dir / 'ens.db').exists() and not allow_legacy:
        import optuna
        try:
            studies = optuna.study.get_all_study_summaries(storage=f'sqlite:///{work_dir / "ens.db"}')
        except Exception:
            studies = []
        if studies:
            raise ValueError('Found legacy studies without dataset fingerprint; '
                             'pass allow_legacy=True ONLY if the dataset and folds match.')
    path.write_text(json.dumps({'fingerprint': fp, 'rows': len(y),
                                'features': X.shape[1], 'n_folds': len(folds)}, indent=2))
