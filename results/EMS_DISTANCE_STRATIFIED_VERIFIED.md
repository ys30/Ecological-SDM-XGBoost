# Distance-stratified matched out-of-fold model audit

Source: `results/matched_distance_stratified_oof_metrics.csv`. Calculated from archived spatially held-out matched predictions, without refitting models. [Successful workflow](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41752468152).

**Distance** is each held-out plot's nearest training-plot distance, based on the coordinate fields and a haversine calculation. Confirm coordinate CRS/provenance before asserting physical km validity. The `ge1km` and `ge5km` sets are nested subsets of `all`; `lt5km` and `ge5km` partition the plots. Results are **unweighted arithmetic means over eligible taxon-fold cells**, not pooled-record metrics.

| Band | Eligible AUC/AP cells | Bayesian AUC | MaxEnt AUC | XGBoost AUC | Bayesian AP | MaxEnt AP | XGBoost AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 23 | 0.761028 | 0.786546 | 0.800696 | 0.283965 | 0.313715 | 0.336851 |
| ge1km | 23 | 0.757156 | 0.783492 | 0.793265 | 0.276528 | 0.311546 | 0.316367 |
| ge5km | 23 | 0.737361 | 0.762010 | 0.784918 | 0.263884 | 0.295874 | 0.313497 |
| lt5km | 21 | 0.843065 | 0.858886 | 0.795376 | 0.509400 | 0.513348 | 0.464675 |

**Interpretation:** XGBoost has the highest descriptive mean AUC and AP in the all/ge1km/ge5km sets; MaxEnt has the highest mean AUC and AP in the lt5km subset. This is an observed ranking change, not evidence of a statistically significant interaction or superior geographic transfer. Short-distance results omit two single-class cells, so their composition differs. Prevalence, sample size, and the restricted-score range differ by distance band; AP is especially prevalence-dependent. Distances to nearest training plots are descriptive diagnostics, not an independent region holdout or spatial-buffer CV.

**Limitations and next step:** Assess paired per-cell band contrasts and report fold/taxon uncertainty without treating the 23 cells as independent. Confirm geographic coordinate CRS. Keep original model input comparability limitations in view; Bayesian, MaxEnt and XGBoost are pipeline comparisons. This document is public result reporting, not manuscript text.
