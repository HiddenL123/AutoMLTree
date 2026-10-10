import numpy as np
from sklearn.datasets import make_classification
from automl_tree import AutoMLTree
from automl_tree.evaluation import ranks, rank_new


def test_rank_mapping_is_stable():
    a = np.array([.2, .4, .4, .8])
    np.testing.assert_allclose(rank_new(a, a), ranks(a))
    assert 0 <= rank_new(np.array([.1]), a)[0] <= 1
    assert 0 <= rank_new(np.array([1.]), a)[0] <= 1


def test_resume_and_inference(tmp_path):
    X, y = make_classification(n_samples=250, n_features=8, n_informative=6,
                               n_redundant=0, random_state=13)
    conf = dict(work_dir=tmp_path, n_folds=2, backends=('lgb',),
                min_candidate_auc=None, beam_width=2, search_depth=1,
                candidate_limit=5, blend_trials_per_batch=2, early_stop=20,
                verbose=False)
    a = AutoMLTree(**conf).fit(X, y, trials_per_family=1)
    assert len(a.runner.study('lgb').trials) == 1
    pred_a = a.predict_scores(X[:3])
    assert np.isfinite(pred_a).all()
    b = AutoMLTree(**conf).fit_to_total(X, y, target_trials_per_family=2)
    assert len(b.runner.study('lgb').trials) == 2
    c = AutoMLTree(**conf).fit_to_total(X, y, target_trials_per_family=2)
    assert len(c.runner.study('lgb').trials) == 2
    assert np.isfinite(c.predict_scores(X[:5])).all()
    assert c.best_ensemble_.score >= a.best_ensemble_.score - 1e-9


def test_no_dataset_mix(tmp_path):
    X, y = make_classification(n_samples=200, n_features=5, n_redundant=0,
                               random_state=13)
    a = AutoMLTree(work_dir=tmp_path, n_folds=2, verbose=False).initialize(X, y)
    X2 = X.copy()
    X2[0, 0] += 1
    import pytest
    with pytest.raises(ValueError, match='Dataset/folds differ'):
        AutoMLTree(work_dir=tmp_path, n_folds=2, verbose=False).initialize(X2, y)
