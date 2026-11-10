# EMS v6.22: class-support decision gate (pre-freeze)

## Verified diagnostics

- Inner AUC feasibility: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42751441252
- Outer grid class support: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42751723413
- Inner: 160 validation taxon-fold combinations; 37 undefined validation AUC (all zero positives), 5 undefined training AUC; 118/160 both training and validation AUC-defined.
- Outer: 40 test taxon-fold combinations; 10 undefined test AUC (all zero positives); 0 undefined outer-training AUC.
- Outer valid AUC folds by species: nsw18 5, nsw19 3, nsw20 2, nsw21 3, nsw22 2, nsw23 5, nsw24 5, nsw25 5.
- Total positives by species: nsw18 244, nsw19 8, nsw20 18, nsw21 39, nsw22 6, nsw23 270, nsw24 109, nsw25 53.

## Mandatory scientific rules before freeze

1. **No imputation of undefined fold AUC** (neither 0.5 nor zero); no silent deletion of invalid folds. Report numerator and denominator of valid taxon-folds for every summary.
2. Inner model selection should use the already proposed relative log-loss score on held-out records, which remains mathematically defined for single-class validation partitions when predictions are clipped to (epsilon, 1-epsilon). The null baseline must be estimated from the **inner-training partition only**, not from the validation or outer-test prevalence. For taxon-specific null log-loss, evaluate constant training prevalence on the validation records. Use a fixed epsilon and explicit failure rules for zero-prevalence training partitions; do not let an undefined null baseline silently enter a mean.
3. Check inner-training positive support before fitting any candidate. Some inner folds contain no positive examples. Such folds cannot fit standard binary classifiers as specified. A documented fallback, redesign, or exclusion policy must be frozen before running tuning. Merely changing the scoring metric does **not** resolve single-class training.
4. AUC on the 5 outer folds is a *secondary diagnostic with incomplete support*. If pooled OOF AUC is later reported, construct it from all held-out predictions once per plot/taxon, identify it as pooled OOF (not mean fold AUC), and accompany it with uncertainty and prevalence information. Do not claim it restores missing fold-level AUC or replaces the preregistered estimand.
5. Do not select spatial fold schemes or k-means seeds based on observed AUC feasibility or eventual performance. The current 12-cluster greedy balancing implementation was introduced as a reproducible plumbing test; verify whether its algorithm exactly matches the preregistered protocol before using it for final tuning.
6. For rare taxa, prioritize proper scores (held-out log loss/Brier), calibration, class support and confidence intervals. Avoid interpreting unstable rare-species AUC differences as conclusive.
7. Keep MaxEnt presence-background interpretation separate from presence-absence AUC; preserve the specified target definition and evaluation population.
8. Formal CRS metadata, km15 source availability, full model runner, and complete leakage tests remain open gates.

## Status

PRE-FREEZE. No new fitted models, performance metrics, or manuscript conclusions. This document records the observed class-support problem and the decisions still requiring preregistration; it does not unilaterally change the analysis plan.
