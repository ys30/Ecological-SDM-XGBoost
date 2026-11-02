# Species-stratified XGBoost sensitivity: verified spatial-fold CI

Successful CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41388369247

The workflow computes per-species, per-fold ROC AUC and average precision for three XGBoost configurations, and uploads the CSV artifact `results/xgboost_spatial_species_fold_sensitivity.csv`. The table below is taken from the successful workflow log.

| Configuration | Mean pooled AUC across five folds | Mean species-fold AUC among estimable cells | Mean species-fold AP among estimable cells |
|---|---:|---:|---:|
| Full fold-local categorical encoding | 0.907595 | 0.7956090 | 0.3285771 |
| Four environment variables + Species | 0.907429 | 0.8052382 | 0.3294149 |
| Four environment variables only | 0.574066 | 0.5397489 | 0.1654571 |

Pooled AUC averages within-fold AUC over all species observations, and can reflect differences in species prevalence. Species-fold AUC evaluates ranking within each species and spatial fold. These estimands are not interchangeable.

**Important:** These species-fold summaries include all estimable species-fold cells for each configuration, not necessarily the exact same 23 species-fold cells used in the existing three-model matched comparison. A join by `species` and `fold` against the matched-cell key set is required before claiming a direct comparison.

The R script marks cells without both held-out classes as `insufficient_test_classes`, and the workflow uploads both pooled and species-fold CSVs. All species names in this report remain anonymized.

Next: perform an explicit 23-cell matched join and paired differences against archived MaxEnt and Bayesian, and check cell counts, alignment, and output scales before updating private manuscript.
