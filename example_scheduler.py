"""Run scheduled LightGBM experiment batches; rerunning resumes the study."""
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from automl_tree import AutoMLTree

X, y = make_classification(n_samples=1200, n_features=15, n_informative=8,
                           n_redundant=3, random_state=31)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=31)

auto = AutoMLTree(work_dir='scheduled_run', backends=('lgb',),
                  n_folds=3, seed=31, corr_max=.995, beam_width=3,
                  search_depth=2, blend_trials_per_batch=5)
auto.fit_scheduled(X_train, y_train, batch_size=5, blend_trials_per_mode=2)
print('CV AUC:', auto.best_ensemble_.score)
print('Blend mode:', auto.best_ensemble_.mode)
print('Scheduler history:', auto.experiment_summary_)
print('Holdout AUC:', roc_auc_score(y_test, auto.predict_scores(X_test)))
