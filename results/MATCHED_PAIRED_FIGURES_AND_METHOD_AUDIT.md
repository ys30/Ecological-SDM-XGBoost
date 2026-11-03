# Matched spatial-fold paired differences — public audit

This report uses the existing `results/matched_bayesian_maxent_xgboost_folds.csv` (23 eligible taxon–fold records across five anonymous taxa). The two companion SVG figures display **XGBoost minus MaxEnt** for ROC AUC and average precision (AP).

- [Paired ROC AUC by taxon and fold](figures/25_paired_xgboost_maxent_auc.svg)
- [Paired AP by taxon and fold](figures/26_paired_xgboost_maxent_ap.svg)
- [Matched fold-level source CSV](matched_bayesian_maxent_xgboost_folds.csv)
- [Earlier paired-difference summary](MATCHED_MODEL_PAIRED_DIFFERENCES.md)
- [Weighting sensitivity](MATCHED_MODEL_WEIGHTING_SENSITIVITY.md)

## Verified descriptive contrasts

| Comparison | Mean delta ROC AUC | ROC AUC cells favoring XGBoost | Mean delta AP | AP cells favoring XGBoost |
|---|---:|---:|---:|---:|
| XGBoost − MaxEnt | +0.01508 | 14 / 23 | +0.02658 | 13 / 23 |

Taxon-level differences vary in direction: MaxEnt has higher mean ROC AUC for `nsw21` and `nsw23`; XGBoost has higher mean ROC AUC for `nsw18`, `nsw24` and `nsw25`. The diagrams show every eligible fold as a dot and each taxon's mean as a diamond. Values are descriptive, **not independent replicates or confidence intervals**.

## Reproducibility status and caveats

1. Saved matched predictions and metrics were previously cross-checked in CI (138 AUC/AP comparisons), but **the AP tie-handling definition still requires an independent sklearn-style sensitivity check on per-record scores**. Do not claim that this check has passed.
2. `scripts/41_xgboost_spatial_encoding_sensitivity.R` implements fold-local categorical encoding, but the expected outputs `results/xgboost_spatial_encoding_sensitivity.csv` and `results/xgboost_spatial_species_fold_sensitivity.csv` were **not present on main when audited**. Treat statements about its empirical effect as pending archival verification.
3. The original pooled XGBoost script `scripts/11_spatial_2d_block_cv.R` builds the dummy-variable vocabulary before splitting folds. It does not use held-out outcomes in this encoding step, but the strict train-only preprocessing sensitivity is therefore relevant.
4. MaxEnt uses survey-derived non-detections as background approximations, not independent landscape background, and its cloglog score is not an externally calibrated occupancy probability.
5. Models differ in predictor sets; these contrasts describe complete modeling pipelines, not an algorithm-only causal comparison.
6. The Bayesian long-chain confirmation assessed fold-level stability, not a regenerated per-record matched prediction set.
7. The spatial folds are unbuffered and within-region; do not describe these as independent-region validation.

## Disclosure

Only anonymous `nsw18`–`nsw25` identifiers are used in these public outputs. No private manuscript or taxon-identity mapping is included. This audit does not resolve permissions or data redistribution terms for historical repository content.
