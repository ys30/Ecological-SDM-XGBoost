# v6.23 exact model specification (code-verified)

This text documents the **fixed** model configurations implemented in `scripts/102_v623_fixed_xgb_outer_oof.py` and `scripts/108_v623_matched_buffer_glm.py`. It can replace placeholder model descriptions in the Methods. It is not a claim that hyperparameters were optimally tuned.

## Ready-to-integrate English Methods text

### Predictors and spatial validation

All three models used the same four environmental predictors: annual rainfall (`rainann`), soil depth (`soildepth`), annual temperature (`tempann`), and topography (`topo`). Predictor units require confirmation from the authoritative source documentation before publication. The five outer folds were derived from a 4 × 4 longitude–latitude quantile-block partition, with entire geographic blocks assigned to folds. For each held-out fold, candidate training plots within **10 km or less** of any test plot were excluded, using great-circle (Haversine) distances computed from longitude and latitude. The exclusion was plot-based and therefore shared across all taxa. Predictions were produced once per held-out observation and pooled within taxon before calculating the primary scores.

### Gradient-boosted trees

We fitted an XGBoost binary logistic classifier (`XGBClassifier`) with histogram-based tree construction (`tree_method='hist'`), 100 boosting trees, maximum tree depth 2, learning rate 0.03, minimum child weight 1, row subsampling 0.8 and column subsampling per tree 0.8. The random seed was 622 and two threads were requested. The four predictors entered on their original numeric scales without fold-level standardization. Predicted event probabilities were clipped to [1 × 10⁻¹², 1 − 1 × 10⁻¹²] for numerical stability. No inner-loop hyperparameter search, early-stopping selection or nested tuning was performed in this matched experiment.

### Logistic regression comparators

Both logistic regression comparators were fitted with scikit-learn's `LogisticRegression` using L2 regularization (default penalty), `C=100`, the `lbfgs` solver, `max_iter=3000`, `tol=1e-8`, and `random_state=20261104`. Each fold's `StandardScaler` was fitted exclusively on the eligible training observations and applied to the held-out observations. The linear model contained the four standardized predictors. The quadratic model added **two** squared standardized terms, for `rainann` and `tempann`; it did not add pairwise interactions or squares for soil depth and topography. The quadratic terms were specified before this v6.23 comparison rather than selected on its held-out predictions. Predicted probabilities were clipped to the same bounds as XGBoost.

### Matched design and limitations

Across the three model arms, the held-out rows, eligible training rows, outer-fold assignments and 10-km exclusion rule were matched. The fitted algorithms and their transformations differ by design. This is a **fixed-configuration model comparison**, not a nested spatial hyperparameter-tuning study. A matched training-membership audit does not rule out leakage introduced before fold assignment by site selection or environmental covariate construction. The primary reported scores are pooled out-of-fold metrics computed separately for each taxon and averaged equally across the eight taxa.

## Source code and reproducibility

- XGBoost: `scripts/102_v623_fixed_xgb_outer_oof.py`
- Linear and quadratic GLMs: `scripts/108_v623_matched_buffer_glm.py`
- Strict row alignment and macro summaries: `scripts/109_v623_matched_three_model_comparison.py`
- Verified numerical run: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44763794431
- Verified figure technical QC: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44770268244

**Open metadata:** original sampling dates/design, covariate units, source redistribution rights, and any upstream selection/preprocessing leakage.
