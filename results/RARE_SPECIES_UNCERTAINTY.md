# Bayesian spatial-CV uncertainty for rare species

This audit summarizes **held-out predictions**, not training-fit uncertainty. Source: [species summary](bayes_species_uncertainty_summary.csv), derived from [record-level predictions](bayes_spatial_cv_predictions.csv) by [script 19](../scripts/19_bayes_species_uncertainty.py).

| Species | Positive records / 1,167 | Mean predicted presence | Mean 95% posterior interval width | Mean width among positives | Mean width among negatives |
|---|---:|---:|---:|---:|---:|
| *Corymbia gummifera* | 8 | 0.0104 | 0.0253 | 0.0387 | 0.0252 |
| *Corymbia intermedia* | 18 | 0.0187 | 0.0279 | 0.3115 | 0.0235 |
| *Eucalyptus carnea* | 6 | 0.0041 | 0.0126 | 0.0738 | See CSV |

## Interpretation

The rarest taxa have only 6–18 observed positive records across all five held-out spatial folds. Species-level ROC AUC/AP are not reliably estimable for several species–fold subsets. Hierarchical partial pooling permits prediction, but **does not establish predictive accuracy** for these taxa.

For *C. intermedia*, the average 95% interval width for held-out positive records (0.3115) is much greater than for negatives (0.0235). This is a **descriptive comparison**, not a significance test or a posterior coverage assessment. The overall mean interval width (0.0279) is dominated by negatives because the positive prevalence is only 1.54%.

The intervals summarize posterior uncertainty in predicted probabilities; they should not be interpreted as calibrated confidence intervals or as proof of improved detection. The underlying Bayesian sampling audit still has R-hat and bulk-ESS flags, and divergence/treedepth/BFMI checks remain outstanding.

## Next checks

1. Inspect per-fold positive counts and uncertainty to detect geographic concentration.
2. Audit interval calibration and observed-vs-predicted probabilities by species, with explicit small-sample caveats.
3. Export Stan sampler divergences, treedepth and BFMI in future Bayesian fits before making convergence claims.
