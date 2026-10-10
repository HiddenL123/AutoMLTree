"""Backend adapters. All backends see the same numeric feature matrix and shared folds."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scipy import sparse


@dataclass(frozen=True)
class Backend:
    suggest: Callable
    fit: Callable


def suggest_lgb(t):
    return dict(learning_rate=t.suggest_float('learning_rate', .01, .1, log=True),
                num_leaves=t.suggest_int('num_leaves', 8, 128, log=True),
                min_child_samples=t.suggest_int('min_child_samples', 10, 200),
                colsample_bytree=t.suggest_float('colsample_bytree', .5, 1.),
                subsample=t.suggest_float('subsample', .5, 1.), subsample_freq=1,
                reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True),
                reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True))


def _inner_split(train_indices, y, seed):
    """Early-stop on INNER data, leaving the outer OOF fold untouched."""
    a, b = train_test_split(train_indices, test_size=.18,
                            stratify=y[train_indices], random_state=seed)
    return a, b


def fit_lgb(X, y, tr, params, seed, early_stop):
    import lightgbm as lgb
    a, b = _inner_split(tr, y, seed)
    inner = lgb.LGBMClassifier(n_estimators=2000, metric='auc', random_state=seed,
                              verbosity=-1, **params)
    inner.fit(X[a], y[a], eval_set=[(X[b], y[b])],
              callbacks=[lgb.early_stopping(early_stop, verbose=False)])
    n_iter = max(1, int(inner.best_iteration_ or inner.n_estimators_))
    model = lgb.LGBMClassifier(n_estimators=n_iter, metric='auc', random_state=seed,
                              verbosity=-1, **params)
    model.fit(X[tr], y[tr])
    return model, n_iter


def suggest_xgb(t):
    return dict(learning_rate=t.suggest_float('learning_rate', .01, .1, log=True),
                max_depth=t.suggest_int('max_depth', 3, 9),
                min_child_weight=t.suggest_float('min_child_weight', 1, 50, log=True),
                subsample=t.suggest_float('subsample', .5, 1.),
                colsample_bytree=t.suggest_float('colsample_bytree', .5, 1.),
                gamma=t.suggest_float('gamma', 1e-3, 5, log=True),
                reg_lambda=t.suggest_float('reg_lambda', 1e-3, 10, log=True),
                reg_alpha=t.suggest_float('reg_alpha', 1e-3, 10, log=True))


def fit_xgb(X, y, tr, params, seed, early_stop):
    import xgboost as xgb
    a, b = _inner_split(tr, y, seed)
    inner = xgb.XGBClassifier(n_estimators=2000, eval_metric='auc',
                              early_stopping_rounds=early_stop,
                              tree_method='hist', random_state=seed, n_jobs=-1, **params)
    inner.fit(X[a], y[a], eval_set=[(X[b], y[b])], verbose=False)
    n_iter = int(inner.best_iteration) + 1
    model = xgb.XGBClassifier(n_estimators=n_iter, eval_metric='auc',
                              tree_method='hist', random_state=seed, n_jobs=-1, **params)
    model.fit(X[tr], y[tr], verbose=False)
    return model, n_iter


def suggest_cat(t):
    return dict(learning_rate=t.suggest_float('learning_rate', .02, .1, log=True),
                depth=t.suggest_int('depth', 4, 10),
                l2_leaf_reg=t.suggest_float('l2_leaf_reg', 1, 30, log=True),
                random_strength=t.suggest_float('random_strength', 1e-2, 10, log=True),
                bagging_temperature=t.suggest_float('bagging_temperature', 0, 5),
                border_count=t.suggest_categorical('border_count', [64, 128, 254]))


def fit_cat(X, y, tr, params, seed, early_stop):
    from catboost import CatBoostClassifier
    a, b = _inner_split(tr, y, seed)
    inner = CatBoostClassifier(iterations=2000, eval_metric='AUC',
                               early_stopping_rounds=early_stop,
                               random_seed=seed, verbose=False, thread_count=-1, **params)
    inner.fit(X[a], y[a], eval_set=(X[b], y[b]), use_best_model=True)
    n_iter = max(1, int(inner.get_best_iteration()) + 1)
    model = CatBoostClassifier(iterations=n_iter, eval_metric='AUC', random_seed=seed,
                               verbose=False, thread_count=-1, **params)
    model.fit(X[tr], y[tr])
    return model, n_iter


def suggest_lr(t):
    return dict(C=t.suggest_float('C', 1e-3, 10, log=True),
                n_knots=t.suggest_int('n_knots', 3, 8))


def fit_lr(X, y, tr, params, seed, early_stop):
    model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                          SplineTransformer(n_knots=params['n_knots'], degree=3,
                                            extrapolation='linear'),
                          LogisticRegression(C=params['C'], max_iter=3000,
                                             random_state=seed))
    model.fit(X[tr], y[tr])
    return model, 0


class LeafLogistic:
    """LightGBM leaf one-hot encoding + scaled raw features + logistic regression."""
    def __init__(self, g, encoder, imputer, scaler, lr):
        self.g, self.encoder, self.imputer, self.scaler, self.lr = (
            g, encoder, imputer, scaler, lr)

    def _features(self, X):
        leaves = self.g.predict(X, pred_leaf=True)
        dense = self.scaler.transform(self.imputer.transform(X))
        return sparse.hstack([self.encoder.transform(leaves),
                              sparse.csr_matrix(dense)], format='csr')

    def predict_proba(self, X):
        return self.lr.predict_proba(self._features(X))


def suggest_leaf(t):
    return dict(n_trees=t.suggest_int('n_trees', 50, 300),
                num_leaves=t.suggest_int('num_leaves', 4, 16),
                min_child=t.suggest_int('min_child', 20, 200),
                C=t.suggest_float('C', 1e-3, 1, log=True))


def fit_leaf(X, y, tr, params, seed, early_stop):
    import lightgbm as lgb
    g = lgb.LGBMClassifier(n_estimators=params['n_trees'],
                           num_leaves=params['num_leaves'],
                           min_child_samples=params['min_child'], learning_rate=.1,
                           subsample=.8, subsample_freq=1, colsample_bytree=.8,
                           random_state=seed, verbosity=-1)
    g.fit(X[tr], y[tr])
    leaves = g.predict(X[tr], pred_leaf=True)
    encoder = OneHotEncoder(handle_unknown='ignore').fit(leaves)
    imputer = SimpleImputer(strategy='median').fit(X[tr])
    scaler = StandardScaler().fit(imputer.transform(X[tr]))
    model = LeafLogistic(g, encoder, imputer, scaler, None)
    lr = LogisticRegression(C=params['C'], max_iter=3000, random_state=seed)
    lr.fit(model._features(X[tr]), y[tr])
    model.lr = lr
    return model, 0


REGISTRY = {
    'lgb': Backend(suggest_lgb, fit_lgb),
    'xgb': Backend(suggest_xgb, fit_xgb),
    'cat': Backend(suggest_cat, fit_cat),
    'lr': Backend(suggest_lr, fit_lr),
    'leaf_lr': Backend(suggest_leaf, fit_leaf),
}


def predict_positive(model, X, chunk=50_000):
    parts = []
    for start in range(0, len(X), chunk):
        parts.append(np.asarray(model.predict_proba(X[start:start+chunk])[:, 1]))
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)
