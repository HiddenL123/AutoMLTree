"""One-time migration smoke test with synthetic saved predictions."""
import numpy as np
import optuna
import pytest
from types import SimpleNamespace
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from automl_tree import AutoMLTree
from automl_tree.evaluation import prepare_folds
from convert_previous_models import convert


def test_cached_trial_conversion_and_idempotence(tmp_path, monkeypatch):
    X, y = make_classification(n_samples=90, n_features=6, n_informative=4,
                               n_redundant=0, random_state=42)
    dev, hold, yd, yh = train_test_split(X, y, stratify=y, test_size=.2,
                                         random_state=0)
    Xt = X[:7].copy()
    data = SimpleNamespace(X_dev=dev, y_dev=yd, X_holdout=hold,
                           y_holdout=yh, X_test=Xt,
                           feature_names=[str(i) for i in range(X.shape[1])])
    monkeypatch.setattr("convert_previous_models.load_data", lambda *_: data)
    source = tmp_path / "legacy"
    (source / "oof").mkdir(parents=True)
    _, _, folds = prepare_folds(dev, yd, 42, 2)
    oof = np.where(yd == 1, .9, .1) + np.arange(len(yd)) * 1e-7
    scores = [roc_auc_score(yd[va], oof[va]) for _, va in folds]
    study = optuna.create_study(study_name="lgb",
                                storage=f"sqlite:///{source / 'ens.db'}",
                                direction="maximize")
    study.add_trial(optuna.trial.create_trial(
        params={"num_leaves": 12},
        distributions={"num_leaves": optuna.distributions.IntDistribution(8, 128, log=True)},
        value=float(np.mean(scores)), user_attrs={"fold_auc": scores}))
    np.savez_compressed(source / "oof/lgb_0.npz", oof=oof,
                        hold=np.linspace(.1, .9, len(hold)),
                        test=np.linspace(.2, .8, len(Xt)))
    target = tmp_path / "native"
    assert convert(source, target, "unused", n_folds=2, loader="kaggle", select=False) == 1
    assert convert(source, target, "unused", n_folds=2, loader="kaggle", select=False) == 0
    auto = AutoMLTree(work_dir=target, n_folds=2, verbose=False).initialize(dev, yd)
    assert len(auto.runner.candidates(("lgb",), None)) == 1
    np.testing.assert_allclose(auto.runner.predict_member("lgb_0", Xt, prediction_set="test"),
                               np.linspace(.2, .8, len(Xt)))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        auto.runner.predict_member("lgb_0", Xt+1, prediction_set="test")
    with pytest.raises(FileNotFoundError):
        auto.runner.predict_member("lgb_0", Xt)
    auto.select_ensemble()
    assert np.isfinite(auto.predict_scores(Xt, prediction_set="test")).all()
