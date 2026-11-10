# EMS v6.22 historical quadratic formula audit

Status: PRE-FREEZE. Inspected repository source code; no fitting or performance calculations.

## Direct source evidence

- `scripts/57_bayes_extension_spatial_cv.R` computes training-fold `mu <- mean(tr[[nm]])` and `sig <- sd(tr[[nm]])` for `rainann`, `tempann`, `soildepth`, `topo`, and applies the **same training moments** to outer test rows.
- Its quadratic hierarchical Bayesian formula uses `z_rainann + z_tempann + z_soildepth + z_topo + I(z_rainann^2) + I(z_tempann^2) + (1 + z_rainann + z_tempann || species)`.
- `scripts/15_bayes_hierarchical_spatial.R` has the linear analogue with the same taxon-varying intercept and rain/temperature slopes.
- `scripts/00_baselines_logit_rf.R` has an earlier logistic baseline with a different feature scope; **do not** assume that baseline establishes the v6.22 GLM quadratic specification.

## Implementation implications

1. `design_matrix.py` already squares standardized values when `standardize=True`, matching the **shape** of the historical quadratic Bayesian fixed-effect terms. It is **not** yet an exact replication of the R fit or hierarchical random effects.
2. The historical R `sd()` is sample standard deviation (denominator n-1). The initial Python implementation used population standard deviation (n). This is a numerical discrepancy and must be fixed before claiming exact moment parity.
3. The historical R code does not impute missing data, whereas the new Python builder has train-median imputation. Missingness policy requires explicit audit before declaring equivalence.
4. Model matrix specifies GLM/GLMM/glmnet quadratic variants, but their full historical model-fitting code, link function and family-specific design conventions must be independently verified; do not infer equivalence from a Bayesian formula alone.
5. Taxon random effects in GLMM/Bayes cannot be replaced by simple one-hot fixed effects without changing the model.

No model runs or new manuscript results were generated.
