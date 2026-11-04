# Bayesian extension: verified probability calibration and taxon heterogeneity

Source: successful GitHub Actions run https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772352167 (9,336 matched OOF records per model).

| Model | Observed prevalence | Mean predicted | Predicted minus observed | ECE, 10 equal-width bins | Brier |
|---|---:|---:|---:|---:|---:|
| Archived linear hierarchical Bayesian | 0.080013 | 0.085014 | +0.005001 | 0.019853 | 0.060842 |
| Quadratic hierarchical Bayesian | 0.080013 | 0.083302 | +0.003289 | 0.012553 | 0.054037 |
| Species-specific no-pooling Bayesian | 0.080013 | 0.086516 | +0.006503 | 0.020273 | 0.061226 |

The quadratic model reduces pooled ECE by about 36.8% and Brier by about 11.2% relative to the archived baseline. ECE depends on the choice of probability bins and should be treated as descriptive. Brier is a proper scoring rule. These pooled metrics may be dominated by more prevalent species.

## Species-specific quadratic minus baseline Brier

| Taxon | ΔBrier |
|---|---:|
| nsw18 | -0.045024 |
| nsw19 | +0.000121 |
| nsw20 | -0.000375 |
| nsw21 | +0.000131 |
| nsw22 | +0.000006 |
| nsw23 | -0.001959 |
| nsw24 | -0.006206 |
| nsw25 | -0.001135 |

Five taxa improve Brier, three slightly worsen. The nsw18 gain is disproportionately large, so avoid claims of uniform benefit. ECE worsens for nsw19, nsw21, nsw22, and nsw25, although the pooled ECE improves. All reported findings are internal unbuffered spatial block holdout, not external transfer.

## Interpretation for private manuscript

The five-fold quadratic Bayesian extension improves matched OOF discrimination, average precision, Brier score, log loss, and pooled reliability relative to the archived linear-response hierarchical Bayesian pipeline. This suggests environmental response specification is an important axis of SDM pipeline design. However, taxon-level heterogeneity and dependence among spatial folds constrain generalization. The non-pooled model has no comparable across-metric improvement, though it improves one rare-taxon top-25 retrieval outcome. The manuscript must remain private; only reproducible code and verified results are public.
