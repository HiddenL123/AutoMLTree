"""Run: python example.py. Re-running resumes the same saved studies."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from automl_tree import AutoMLTree

X, y = make_classification(n_samples=1500, n_features=12,
                           n_informative=7, random_state=14)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=42)

auto = AutoMLTree(
    work_dir='demo_run', backends=('lgb',),
    n_folds=3, seed=42, min_candidate_auc=None,
    corr_max=.99, beam_width=3, search_depth=2,
    candidate_limit=12, blend_trials_per_batch=5)

# Each invocation trains up to two NEW trials, reusing earlier Optuna trials.
auto.fit(X_train, y_train, trials_per_family=2)
print('Best CV AUC:', auto.best_ensemble_.score)
print('Example scores:', auto.predict_scores(X_test[:5]))

# Alternative: run to a LIFETIME total, not +N:
# auto.fit_to_total(X_train, y_train, target_trials_per_family={'lgb': 10})

# To reuse legacy ens.db / oof files (only with exactly matching data + folds):
# auto = AutoMLTree(work_dir='.', backends=('lgb', 'xgb', 'cat'),
#                   allow_legacy=True)
# auto.initialize(Xd, yd, folds=FOLDS)
# auto.select_ensemble()
# Note: old OOF-only trials cannot predict new X until their fitted models are restored.
