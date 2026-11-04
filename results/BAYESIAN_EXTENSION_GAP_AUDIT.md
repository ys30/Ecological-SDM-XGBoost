# Bayesian model-extension gap audit — 2026-11-03

This is an evidence-based public research planning note, not evidence that new Bayesian fits were run.

## Verified baseline
The implemented model in [scripts/15_bayes_hierarchical_spatial.R](../scripts/15_bayes_hierarchical_spatial.R) is Bernoulli-logit with linear fixed effects for fold-standardized rain, temperature, soil depth, and topography, and taxon-specific varying intercepts plus rain and temperature slopes (uncorrelated via `||`). No quadratic terms or explicit spatial-block random effect appear in the archived formula.

Five-fold MCMC confirmation is complete: [convergence report](BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md). Rare-taxon probability scores, including Brier and log loss, exist in [rare-taxon evaluation](BAYESIAN_RARE_SPECIES_PROBABILITY_EVALUATION.md). Matched Bayesian–XGBoost top-k retrieval has a [verified comparison](RARE_TAXON_MATCHED_BAYES_XGBOOST_VERIFIED.md).

## Extension gaps not evidenced by current tracked scripts/results
1. Quadratic environmental response terms (e.g., standardized rainfall squared and temperature squared), fitted and evaluated under the same five spatial folds.
2. Spatial-block or spatially structured random effects with a defensible prediction rule for **unseen** blocks. A naive held-out block factor with a new-level intercept set to zero is not a validated spatial extrapolation mechanism.
3. Non-hierarchical Bayesian comparator using the same predictors and held-out folds to isolate the contribution of partial pooling.
4. Held-out spatial residual autocorrelation diagnostic (e.g., Moran's I with explicit neighbor graph and sensitivity to distance/weights).
5. Posterior predictive calibration and uncertainty coverage under comparable prediction targets, especially for sparse taxa.
6. Appropriate model comparison: report held-out spatial predictive scores as primary; use LOO/WAIC only with an explicitly justified dependence structure, not as a substitute for spatial holdouts.
7. Independent external geographic transfer test, which would require another authorized dataset.

These are **not verified completed**, rather than proof no experiments were ever run outside the tracked repository.

## Prioritized minimum next experiment
- Baseline: archived linear hierarchical Bayesian model.
- Variant Q: add a small prespecified set of quadratic fixed effects (rain and temperature) with regularizing priors.
- Variant N: non-hierarchical comparator with identical environmental variables and training-only scaling.
- Keep identical plot-grouped five-fold splits; report held-out taxon-fold AUC/AP eligibility, Brier/log loss, and rare-taxon top-k at equal fold budgets.
- Only after Q/N checks, evaluate an explicit spatial effect with a defined prediction policy for unseen held-out blocks; then evaluate residual spatial structure.
- Require posterior diagnostics, no divergent transitions, and traceable fold-level artifacts before claiming results.

## Scientific boundaries
The 23 eligible discrimination cells are not independent replicates; three sparse taxa need probability and fixed-budget retrieval metrics. Existing confirmed long-chain diagnostics do not imply that Q/N/spatial variants have been fitted. No manuscript is published here.
