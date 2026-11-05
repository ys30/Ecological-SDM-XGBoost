# Verified nested spatial GLM threshold experiment

**Successful CI:** [#42112283643](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42112283643). Script: [83_nested_spatial_glm_threshold.py](../scripts/83_nested_spatial_glm_threshold.py). Artifact ID **13454522585** contains `outer_fold_threshold_counts.csv` and `nested_glm_pooled_metrics.csv`. All results are aggregated; no restricted species names or plot-level records published.

Five outer spatial folds, with 5/10/20-km training exclusion buffers; four inner spatial k-means folds on the outer-training plots. Inner training is additionally buffered from its inner validation fold. Quadratic regularized per-taxon GLM is refitted in each inner split; inner OOF negative probabilities select thresholds for target inner false-positive rates 1%, 2%, 5%. Final models refit on outer-training plots, and the frozen thresholds are evaluated on outer held-out plots. Spatial distances use projected XY units assumed meters, without verified CRS metadata.

## Verified 10-km outer-fold pooled results

| Taxon | Inner FPR target | TP | FP | FN | TN | Outer recall | Outer FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| nsw19 | 1% | 5 | 63 | 3 | 1096 | 62.50% | 5.44% |
| nsw19 | 2% | 5 | 98 | 3 | 1061 | 62.50% | 8.46% |
| nsw19 | 5% | 6 | 130 | 2 | 1029 | 75.00% | 11.22% |
| nsw20 | 1% | 6 | 8 | 12 | 1141 | 33.33% | 0.70% |
| nsw20 | 2% | 13 | 26 | 5 | 1123 | 72.22% | 2.26% |
| nsw20 | 5% | 13 | 45 | 5 | 1104 | 72.22% | 3.92% |
| nsw22 | 1% | 1 | 48 | 5 | 1113 | 16.67% | 4.13% |
| nsw22 | 2% | 2 | 79 | 4 | 1082 | 33.33% | 6.80% |
| nsw22 | 5% | 4 | 123 | 2 | 1038 | 66.67% | 10.59% |

## Interpretation and limitations

For nsw20, the inner 2%-FPR rule produces 13/18 detected positives and 26 false positives (outer FPR 2.26%). In contrast, the same nominal inner FPR target produces 8.46% and 6.80% outer FPR for nsw19 and nsw22. **Inner FPR targets do not guarantee outer FPR control under spatial shift.** These findings are within-dataset spatial validation, not external validation or operational performance guarantees.

**No nested Bayesian comparator has been fitted.** Do not compare this newly refitted nested GLM against the previous Bayesian oracle FP-budget results as though thresholds had been selected identically. Rare positive counts (6–18), alternative fold geometry, spatial buffers, and unverified coordinate CRS limit inference. The threshold selector uses inner-validation negative scores only, and outer test labels are used only for evaluation. The current inner clustering is four coordinate-based k-means clusters, not independently audited as a spatial-block design; validate its geometry and training exclusions before claiming fully leakage-free geographic generalization.
