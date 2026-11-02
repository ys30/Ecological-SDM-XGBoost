# Paired 23-cell sensitivity and spatial-fold bootstrap — verified

Successful CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41390769760

## Paired mean AUC differences across 23 matched species-fold cells

| XGBoost sensitivity | ΔAUC vs MaxEnt | ΔAUC vs Bayesian | Cells with AUC > MaxEnt | Cells with AUC > Bayesian |
|---|---:|---:|---:|---:|
| Full fold-local encoding | +0.01248540 | +0.03800373 | 14/23 | 17/23 |
| Four environmental predictors + Species | +0.01628069 | +0.04179902 | 13/23 | 15/23 |
| Four environmental predictors only | -0.16516653 | -0.13964820 | 2/23 | 3/23 |

## Descriptive 2,000-replicate fold-block bootstrap ranges (2.5–97.5 percentiles)

| XGBoost sensitivity | ΔAUC vs MaxEnt | ΔAUC vs Bayesian |
|---|---|---|
| Full fold-local encoding | [-0.00906, +0.03104] | [+0.01886, +0.05085] |
| Four environmental predictors + Species | [-0.01139, +0.04145] | [+0.01641, +0.06162] |
| Four environmental predictors only | [-0.19263, -0.12905] | [-0.16514, -0.10318] |

**Caution:** Only five spatial folds are resampled. These percentile ranges are descriptive sensitivity intervals and are not confirmatory confidence intervals or hypothesis tests; overlapping species-fold cells are not independent.

## Species heterogeneity
- Full XGBoost vs MaxEnt mean ΔAUC: `nsw18` +0.04442, `nsw21` -0.02159, `nsw23` -0.02303, `nsw24` +0.04368, `nsw25` +0.00532.
- Hence, a universal XGBoost advantage is not supported.
- Four-predictor-only pooled XGBoost omits Species identity; its poorer within-species ranking does not imply the environmental predictors are intrinsically uninformative.

CI artifact includes paired cell, species, summary and bootstrap CSVs. The manuscript is private; do not copy unpublished text or original species identities into the public repository.
