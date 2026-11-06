# Five-fold nested Bayesian vs GLM: verified 10-km comparison

**CI:** [Bayesian folds 1/2/4/5 #42129195010](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42129195010) (four matrix jobs SUCCESS), [Bayesian fold 3 #42124372697](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42124372697) (SUCCESS), [nested GLM #42112283643](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42112283643) (SUCCESS).

## Outer pooled counts, inner target FPR 2%, buffer 10 km

| Taxon | Model | TP | FP | FN | TN | Recall | FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| nsw19 | GLM | 5 | 98 | 3 | 1061 | 62.50% | 8.46% |
| nsw19 | Bayesian | 4 | 119 | 4 | 1040 | 50.00% | 10.27% |
| nsw20 | GLM | 13 | 26 | 5 | 1123 | 72.22% | 2.26% |
| nsw20 | Bayesian | 1 | 5 | 17 | 1144 | 5.56% | 0.44% |
| nsw22 | GLM | 2 | 79 | 4 | 1082 | 33.33% | 6.80% |
| nsw22 | Bayesian | 4 | 87 | 2 | 1074 | 66.67% | 7.49% |

All numbers are sums of *outer* confusion matrices, not averages of fold percentages. No claim of statistical significance. Bayesian fit: hierarchical quadratic logistic, shared partial pooling; GLM: separate taxon-wise quadratic logistic. Both use the same outer and inner spatial partition logic and training-only 10-km exclusion, with distinct probability models and slightly different numeric threshold boundary implementations. Four inner plus one outer Bayesian refit per outer fold, 25 Bayesian fits total. Script gates check Rhat <=1.01, bulk and tail ESS >=400, zero divergences; not all diagnostics are archived separately. Data consist of internal spatial folds only, with no independent external region; CRS-to-km interpretation not independently certified.

## Bayesian per-fold outer confusion counts at 2%

| Fold | nsw19 TP/FP/FN/TN | nsw20 TP/FP/FN/TN | nsw22 TP/FP/FN/TN |
|---|---|---|---|
| 1 | 1/13/0/213 | 0/0/0/227 | 0/11/0/216 |
| 2 | 0/10/2/217 | 0/0/16/213 | 1/18/1/209 |
| 3 | 2/65/0/161 | 1/5/1/221 | 1/47/0/180 |
| 4 | 1/31/2/205 | 0/0/0/239 | 2/11/1/225 |
| 5 | 0/0/0/244 | 0/0/0/244 | 0/0/0/244 |

Folds without positives have undefined recall, even when FPR=0. For nsw20, 16 of 18 positives occur in fold 2; Bayesian misses all 16 there at 2% inner target. The Bayes/GLM trade-off depends on species and threshold: for nsw19 Bayes has lower recall and higher FPR; for nsw20 Bayes has far lower FPR but also much lower recall; for nsw22 Bayes has higher recall and slightly higher FPR. Do not describe this as general Bayesian superiority. The 1% and 5% target results are retained in the CI artifacts but not discussed as the primary comparison.

## Interpretation for manuscript

> Across five spatially held-out folds, neither hierarchical Bayesian partial pooling nor taxon-wise regularized logistic regression consistently maintained the nominal inner-validation false-positive target in geographically shifted outer folds. Differences in detection sensitivity and false-positive burden varied by rare taxon, underscoring that probability modeling and deployment-oriented threshold transfer are separate evaluation questions.

This remains descriptive. The work does not establish causal mechanisms for fold shift or external deployment accuracy. Preserve anonymous taxon IDs in all public outputs.
