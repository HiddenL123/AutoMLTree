import numpy as np
import optuna
from sklearn.datasets import make_classification
from automl_tree import AutoMLTree
from automl_tree.scheduler import ExperimentScheduler, Experiment


def test_scheduler_persists_once(tmp_path):
    s = ExperimentScheduler(tmp_path)
    batch = s.plan_batch(('lgb',), 5)
    assert {a.operator for a in batch} == {'standard', 'extra_trees', 'dart',
                                          'rf', 'feature_subset'}
    s.record('lgb_0', Experiment('lgb', 'standard'), .001, 20)
    s.record('lgb_0', Experiment('lgb', 'standard'), .001, 20)
    assert ExperimentScheduler(tmp_path).summary()['lgb:standard']['trials'] == 1


def test_scheduler_resume_and_inference(tmp_path):
    X, y = make_classification(n_samples=180, n_features=10, n_informative=7,
                               n_redundant=0, random_state=71)
    opts = dict(work_dir=tmp_path, backends=('lgb',), n_folds=2, early_stop=12,
                seed=71, beam_width=2, search_depth=1, candidate_limit=6,
                blend_trials_per_batch=2, verbose=False)
    a = AutoMLTree(**opts)
    a.fit_scheduled(X, y, batch_size=1, blend_trials_per_mode=0)
    a.fit_scheduled(X, y, batch_size=5, blend_trials_per_mode=2)
    trials = a.runner.study('lgb').trials
    assert len(trials) == 6
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
    assert {'standard', 'extra_trees', 'dart', 'rf',
            'feature_subset', 'feature_ablation'} <= {
                t.user_attrs.get('operator') for t in trials}
    for t in trials:
        m = f'lgb_{t.number}'
        assert np.isfinite(a.runner.load_oof(m)).all()
        assert np.isfinite(a.runner.predict_member(m, X[:5])).all()
    assert a.best_ensemble_.mode in ('rank', 'probability', 'logit')
    b = AutoMLTree(**opts).fit_scheduled(X, y, batch_size=0,
                                        blend_trials_per_mode=0)
    assert len(b.runner.study('lgb').trials) == 6
    assert len(b.scheduler.records) == 6
    np.testing.assert_allclose(a.predict_scores(X[:3]), b.predict_scores(X[:3]))
