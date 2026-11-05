# Nested Bayesian 10-km fold-3 pilot: preregistered interpretation

## Status
GitHub Actions [#42124372697](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42124372697) was running at the time this protocol was committed. No Bayesian numerical results are claimed here.

## Methods
- Source data: anonymized eight-taxon, 1,167-plot dataset; no raw taxon identities are published.
- Outer spatial test: alternative spatial fold 3; training plots within the nominal 10-km coordinate buffer are removed.
- Four inner spatial validation folds are clustered from the remaining candidate outer-training plots, matching the deterministic GLM design.
- Inner training is further buffered against inner validation plots by 10 km.
- Hierarchical Bayesian quadratic logistic model retains the original `brms` formula, priors and four-chain settings from `scripts/57_bayes_extension_spatial_cv.R`; fit diagnostics are required before publishing results.
- Inner out-of-fold **negative** predictions determine cutoffs at nominal FPR targets 1%, 2%, 5%. Outer labels are used only to score predictions after thresholds are fixed.
- Outputs are aggregate confusion-matrix counts for nsw19, nsw20 and nsw22, not row-level predictions.

## Comparison planned
Compare Bayesian fold-3 outcomes with the previously verified GLM fold-3 confusion matrices for the same outer fold, 10-km buffer and nominal FPR targets. The threshold selection rule is analogous, but model-fitting procedures and posterior predictive probability construction differ. This is a **single-fold pilot**, not a five-fold performance comparison; improvement in one fold would not establish overall superiority.

## Known limitations
No external geographic test dataset; nominal XY-to-km scale has not been formally CRS-verified. Rare taxa have very few held-out positives. The inner threshold depends on observed inner negatives, which is valid for validation-based selection but is not a threshold available without labeled calibration data. If MCMC fails convergence checks, do not interpret counts.
