# Verified nested GLM outer-fold transfer audit

Successful [CI #42116521486](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42116521486). Source [84_nested_glm_fold_transfer_audit.py](../scripts/84_nested_glm_fold_transfer_audit.py); artifact ID 13457510797 contains aggregate per-fold transfer and summary tables, covering 5/10/20-km buffers and inner targets 1/2/5%.

## 10-km buffer, inner target FPR 2%

| Taxon | Folds above 2% target | Worst fold FPR | Median fold FPR | Threshold min–max |
|---|---:|---:|---:|---:|
| nsw19 | 3/5 | 30.09% | 2.20% | 0.001694–0.189575 |
| nsw20 | 2/5 | 9.39% | 0.00% | 0.001290–0.386123 |
| nsw22 | 4/5 | 14.98% | 6.17% | 0.001613–0.041264 |

**Specific folds:** nsw19 outer fold 3 has 2 positives, TP 2, FP 68, TN 158, outer FPR 30.09%; fold 4 has TP 3, FP 25, TN 211, FPR 10.59%. nsw20 fold 2 contains 16 of 18 positives, with TP 11, FP 20, TN 193, FPR 9.39%; folds 1, 4 and 5 contain no positives. nsw22 folds 3 and 4 have FPR 14.98% and 11.02% respectively; 4 of 5 folds exceed the inner target.

These are **observed patterns**, not causal explanations. Threshold range differences may reflect spatial distribution, prevalence, calibration, or fitting instability; this audit does not distinguish mechanisms. Zero outer-fold recall is undefined for folds with no positives, and those folds must not be interpreted as perfect negative-only detection performance. A five-fold 10-km Bayesian nested comparator was subsequently completed; see [verified Bayesian–GLM report](NESTED_BAYESIAN_FIVE_FOLD_VERIFIED.md). Coordinate CRS remains unverified.
