# AutoMLTree — Experiment Scheduler (V2)

Current version supports a **resumable, LLM-free experiment scheduler** that selects LightGBM standard GBDT, DART, random-forest mode, extra-trees, feature-subset training, or top-feature ablation. After each batch, ensemble search evaluates additions, removals and replacements, then compares rank, probability and logit blending.

## Start scheduled training

```python
from automl_tree import AutoMLTree

auto = AutoMLTree(work_dir="runs/ev", backends=("lgb",), n_folds=5)
auto.fit_scheduled(X_train, y_train, batch_size=5)
auto.fit_scheduled(X_train, y_train, batch_size=5)  # 5 more trials
print(auto.experiment_summary_)
print(auto.best_ensemble_)
scores = auto.predict_scores(X_test)
```

Run the example with `python example_scheduler.py`. All Optuna studies, ensemble checkpoints and scheduler rewards persist across runs. The scheduler learns operator-level rewards from marginal ensemble AUC gain adjusted for training time. It is a heuristic UCB policy, not yet cross-dataset meta-learning. CV ensemble-selection scores are subject to selection overfitting; confirm on untouched holdout data. `predict_scores()` returns discrimination scores, not calibrated probabilities.

**The earlier V1 documentation follows below for compatibility.** Some descriptions of future scheduler work below are superseded by the V2 notes above.

---

# AutoMLTree — batch-driven, resumable prototype

An experimental binary-classification AutoML optimizer organized around four independent concepts:

- **Backends** (`automl_tree/backends.py`): Optuna parameter spaces and fitted models. LightGBM is the default. XGBoost, CatBoost, spline logistic regression and GBDT-leaf logistic regression are also available through lazy imports.
- **Study runner** (`automl_tree/studies.py`): persistent Optuna studies in SQLite, one shared CV scheme and cached OOF predictions, serialized fitted fold models.
- **Ensemble optimizer** (`automl_tree/ensemble.py`): rank-based AUC blend, correlation constraint, beam-limited ADD / REMOVE / SWAP / multi-conflict REPLACE, and per-subset Optuna weight refinement.
- **Public API** (`automl_tree/api.py`): train a batch, select the best available ensemble, resume studies and predict on new data.

## Quick start

```bash
pip install -r requirements.txt
python example.py
```

```python
from automl_tree import AutoMLTree

auto = AutoMLTree(work_dir='runs/ev', backends=('lgb',),
                  n_folds=5, min_candidate_auc=.90, corr_max=.99)
auto.fit(X_train, y_train, trials_per_family=5)
auto.fit(X_train, y_train, trials_per_family=5)  # 5 MORE trials
# Or auto.run_batches(X_train, y_train, n_batches=10, trials_per_family=5)
# OR auto.fit_to_total(X_train, y_train, {'lgb': 25})
print(auto.best_ensemble_)
scores = auto.predict_scores(X_test)
```

Use a **unique work_dir for every dataset/fold setup**. The manifest hashes feature matrix, target labels and CV splits and refuses incompatible new data. It does not validate changes in code or preprocessing semantics; use a new `work_dir` for material changes to those too.

## Resume semantics

`ens.db` is kept in `work_dir`. Study names are `lgb`, `xgb`, `cat`, `lr`, `leaf_lr` so `optuna.create_study(..., load_if_exists=True)` truly resumes instead of creating duplicate studies.

- `fit(..., trials_per_family=5)` always requests **5 additional trials per selected family**.
- `fit_to_total(..., {'lgb': 25})` requests enough extra trials to reach **25 FINISHED trials** (complete, pruned or failed; unfinished running trials not counted).
- The best ensemble is checkpointed as `best_ensemble.json` and is restored on restart.
- `blend_<hash-of-membership>` studies are persistent and resumed whenever *that exact subset* is refined again.
- For compatibility with old `ens.db` and `oof/<name>_<trial>.npz`, set `work_dir='.'`, `allow_legacy=True`, and **provide exactly the original `folds`** (Optuna alone does not retain fold indices). This trust switch cannot verify what data was used historically, and old `.npz` predictions do not include serialized models. Such legacy candidates can participate in selection but require their original models or refitting for `predict_scores(new_X)`.

## Why deterministic alpha search before expensive Optuna blending?

An AUC blend with one new candidate is a one-dimensional problem:

`blend(alpha) = (1-alpha)*existing + alpha*new`.

A coarse grid (with a local refinement around its best value) is cheap, reproducible, and finds candidates worth exploring without a multi-dimensional optimizer for every model. There is no guaranteed closed-form optimum based on variance/correlation for rank-based AUC: AUC changes at score-order crossings. After a batch has been evaluated, Optuna refines weights **for the selected subset only**. This follows the intent of your second idea without its per-candidate high-dimensional optimization overhead.

## Important statistical and engineering limitations (V1)

1. **OOF is a model-selection score, not an unbiased final generalization estimate.** The system repeatedly selects and tunes on the same folds. Assess a completed system on untouched test data or a nested outer CV loop.
2. **Early stopping** uses an inner split within each training fold. Fitted models are refit on the whole outer-training fold with the inner-selected tree count, so outer OOF labels never influence their training or stopping iteration.
3. **AUC rank-based ensemble:** OOF training scores are converted to ranks. For inference, each candidate prediction is mapped through its saved OOF empirical CDF. These are *not calibrated probabilities*, despite the sklearn-style `predict_proba` method name. Use `predict_scores()` for AUC.
4. **Inputs:** V1 expects a *numeric, already-encoded* feature matrix and a binary 0/1 target. Preprocessing and feature transforms are not automatically generated yet.
5. **Search is heuristic:** finite-width beam search and `candidate_limit` do not guarantee globally optimal subsets, even after unlimited batches. The completeness scheduler and learned experiment recommender are future features.
6. **Performance filters:** `min_candidate_auc=None` includes every completed model. A hard threshold can miss complementary weak models. `corr_max` is a hard constraint for the search, not a statistically safe overfitting bound.
7. **Model persistence:** after a new completed trial, all outer-fold estimators are serialized, potentially using significant disk space. Older trials from your original script have only OOF arrays; model files must be recovered or refitted before arbitrary new-data predictions.
8. **Optuna resume:** Optuna persists trial history, not necessarily the complete sampler/pruner internal RNG state. This supports study continuation, not bit-for-bit equivalence to an uninterrupted run.
9. **Training budget:** `fit()` currently budgets by **trial count**, not wall clock or energy. A compute-aware scheduler is a later stage.
10. **Score optimizer:** coordinate- or joint-weight nonlinear optimization can be investigated later; the present per-membership Optuna refinement is a small-budget heuristic.


## Kaggle Playground Series S6E10: original AutoML baseline

Place competition files in the repository root as:

```text
playground-series-s6e10/
    train.csv
    test.csv
    sample_submission.csv  # optional; submission writer uses test IDs if missing
```

Run the **original multi-model Optuna search** (not the adaptive experiment scheduler):

```bash
python kaggle_s6e10_train.py --quick --families lgb
python kaggle_s6e10_train.py --families lgb xgb cat lr leaf_lr --trials 10
# Or train to a lifetime total without repeating completed trials:
python kaggle_s6e10_train.py --to-total --families lgb --trials 30
```

Without `--quick` or `--trials`, the runner uses the previous per-family allocations (40 LR, 30 leaf LR, 80 LGB, 80 XGB, 40 CatBoost trials) **per invocation**. XGBoost and CatBoost are optional dependencies; install them separately when selecting those families. Additional runs using the same data and work directory resume existing Optuna studies. The working directory defaults to `runs/s6e10_baseline/`. Unlike the old script, the `id` column is not treated as a predictor and a candidate AUC cutoff is optional (`--min-candidate-auc 0.95` reproduces the old filter).

Generate `submission.csv` from the **saved best ensemble without retraining**:

```bash
python kaggle_s6e10_submit.py
# Alternative input and output directories:
python kaggle_s6e10_submit.py --data-dir playground-series-s6e10 --output my_submission.csv
```

Both entry points reuse `kaggle_s6e10_data.py` for deterministic categorical encoding and the original stratified development/holdout split (80/20, split seed 0). Prediction scores are suitable for AUC but are not calibrated probabilities. The holdout is reported as a diagnostic, not used to choose members. If a dataset or preprocessing changes, use a fresh work directory; the dataset fingerprint prevents accidental mixing.


## Kaggle S6E10: CSV training and submission (original strategy)

Put the competition files in the repository's local \`playground-series-s6e10/\` folder:

- \`train.csv\`
- \`test.csv\`
- \`sample_submission.csv\` (optional; if absent, \`id,satisfaction\` is created)

These files are not required for the package's unit tests.

Run the **original Optuna model-generation + ensemble-selection** workflow (not the adaptive experiment scheduler):

\`\`\`bash
python example_competition.py
\`\`\`

This trains five *additional* LightGBM trials by default, saves the studies under \`runs/s6e10/\`, and prints an untouched holdout AUC diagnostic. Repeat the command to add five more trials and retain existing models and ensemble checkpoints. Or select all previous model families:

\`\`\`bash
python example_competition.py --backends lr leaf_lr lgb xgb cat --trials 5
\`\`\`

For the original per-family lifetime trial targets (LR 40, leaf LR 30, LGB 80, XGB 80, CatBoost 40), use:

\`\`\`bash
python example_competition.py --backends lr leaf_lr lgb xgb cat --full
\`\`\`

The \`--full\` option resumes existing studies **up to** those totals, rather than adding the totals on every invocation. XGBoost and CatBoost require their optional Python packages. The original AUC cutoff of 0.95 can be restored with \`--min-candidate-auc 0.95\`; the runner defaults to no hard cutoff to retain potentially complementary models.

Generate a submission from the **already saved ensemble**, without training again:

\`\`\`bash
python predict_submission.py
\`\`\`

This writes \`submission.csv\` with the target column \`satisfaction\`. You can use \`--data-dir\`, \`--work-dir\`, and \`--output\` to override the default paths. The submission-only script rereads \`train.csv\` to restore the original development folds and rank reference; keep the training CSV and preprocessing consistent. The \`id\` column is not used as a predictive feature. The prediction values are AUC discrimination scores, not calibrated probabilities.

Selection CV AUC is **not** an unbiased generalization estimate after repeated ensemble selection. Do not use the holdout score to repeatedly tune the experiment policy.
