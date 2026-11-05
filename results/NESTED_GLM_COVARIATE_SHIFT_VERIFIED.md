# Verified buffered outer-fold covariate shift audit

Successful [CI #42121825010](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42121825010), aggregate artifact ID **13459176285**. Script: [85_nested_glm_covariate_shift_audit.py](../scripts/85_nested_glm_covariate_shift_audit.py). Four predictors (rainann, tempann, soildepth, topo) compared between each held-out outer fold and its 5/10/20-km-buffered outer-training plots. Standardized mean difference (SMD) uses training sample SD. No model probabilities or species outcomes are used for covariate shift calculation.

## 10 km shift summary

| Outer fold | Max absolute SMD | Mean absolute SMD | Max fraction outside training feature range | Median nearest retained training XY distance (coordinate units) |
|---|---:|---:|---:|---:|
| 1 | 0.272728 | 0.208068 | 0.004405 | 23616.56 |
| 2 | 1.402460 | 0.526769 | 0.056769 | 24525.63 |
| 3 | 0.608720 | 0.335098 | 0.008772 | 26379.72 |
| 4 | 0.388747 | 0.257379 | 0.041841 | 22800.33 |
| 5 | 0.916066 | 0.516602 | 0.004098 | 28161.45 |

Largest signed SMD per fold: fold 1 tempann +0.272728; fold 2 tempann −1.402460; fold 3 rainann −0.608720; fold 4 rainann +0.388747; fold 5 tempann +0.916066.

## Descriptive comparison with nested GLM fold transfer (inner FPR target 2%)

| Fold | nsw19 outer FPR | nsw20 outer FPR | nsw22 outer FPR |
|---|---:|---:|---:|
| 1 | 0.00% | 0.00% | 6.17% |
| 2 | 2.20% | 9.39% | 2.20% |
| 3 | 30.09% | 2.65% | 14.98% |
| 4 | 10.59% | 0.00% | 11.02% |
| 5 | 0.00% | 0.00% | 0.00% |

The fold with the largest univariate mean shift (fold 2) is **not** the fold with the largest nsw19 FPR (fold 3). Univariate SMD alone does not explain the pattern. This does not rule out multivariate covariate shift, prevalence shift, calibration effects or threshold variation; no causal test has been run. The XY distance unit has not been independently CRS-verified, so it is reported as coordinate units. Only three rare taxa are examined for FPR comparisons, and all FPR estimates are within-dataset spatial holdouts.
