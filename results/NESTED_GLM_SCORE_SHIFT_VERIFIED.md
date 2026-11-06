# Verified nested spatial GLM negative-score shift

[Successful CI #42123004894](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42123004894); artifact **13459412847** contains `outer_fold_score_shift.csv` plus repeated threshold metrics. Script [86_nested_glm_score_shift.py](../scripts/86_nested_glm_score_shift.py) reruns the nested buffered GLM and compares inner-OOF negative prediction score quantiles with untouched outer-fold negative scores. The latter use held-out labels **for retrospective diagnostic evaluation only**, never to choose thresholds.

## 10-km buffer, inner target 2% FPR

| Taxon / outer fold | Selected cutoff | Inner negative P98 | Outer negative P98 | Outer FP / negatives | Outer FPR |
|---|---:|---:|---:|---:|---:|
| nsw19 / 3 | 0.001694 | 0.001681 | 0.027055 | 68 / 226 | 30.09% |
| nsw19 / 4 | 0.021363 | 0.021041 | 0.242378 | 25 / 236 | 10.59% |
| nsw22 / 3 | 0.001613 | 0.001613 | 0.009166 | 34 / 227 | 14.98% |
| nsw22 / 4 | 0.009336 | 0.009108 | 0.050803 | 26 / 236 | 11.02% |
| nsw20 / 2 | 0.001290 | 0.001290 | 0.896625 | 20 / 213 | 9.39% |

At the same cutoff, the high-score tail of held-out negatives is heavier than the inner validation tail in these folds, directly accounting for increased observed false-positive counts. This is a **distributional explanation**, not a causal identification of why model scores changed. Different inner/outer training fits and spatial shift may contribute. In nsw22 fold 3, the inner negative above-cutoff count is **4**, rather than approximately 2% of 910, because many probabilities are tied around the selected order statistic. The inner target is an upper allowance, not a promise of exact 2% realized rate.

The P98 ratio is not a calibrated effect size and should not be used as a formal significance measure. This is an internal spatial holdout analysis, not independent external validation. A five-fold 10-km Bayesian nested comparator was subsequently completed; see [verified Bayesian–GLM report](NESTED_BAYESIAN_FIVE_FOLD_VERIFIED.md).
