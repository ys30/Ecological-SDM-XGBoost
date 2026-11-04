# Bayesian extension: taxon-clustered robustness and rare-taxon retrieval

Verified successful Actions run: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41772026458

## Taxon-equal mean paired deltas against archived shorter-chain Bayesian baseline

| Variant | Metric | Mean Δ | 95% taxon bootstrap percentile interval | Exploratory exact taxon sign-flip p |
|---|---|---:|---|---:|
| quadratic | AUC | +0.032667 | [0.013010, 0.055893] | 0.007812 |
| quadratic | AP | +0.081739 | [0.036770, 0.126937] | 0.015625 |
| quadratic | Brier | -0.006775 | [-0.017992, -0.000340] | 0.062500 |
| quadratic | Log loss | -0.020342 | [-0.049610, -0.002349] | 0.007812 |
| no_pool | AUC | +0.012027 | [-0.007242, 0.035873] | 0.320312 |
| no_pool | AP | +0.021890 | [-0.003717, 0.052338] | 0.242188 |
| no_pool | Brier | +0.000389 | [-0.000476, 0.001180] | 0.445312 |
| no_pool | Log loss | -0.000658 | [-0.003194, 0.001919] | 0.656250 |

All analyses treat the eight taxa as clusters. The five folds are not independent replications. Exact sign flips are exploratory and no multiplicity adjustment has been applied. In particular, the Brier bootstrap interval excludes zero while the exact sign-flip p exceeds .05; do not overstate that result.

## Rare taxa: top-25 observed presences over five held-out folds

| Variant | nsw19 | nsw20 | nsw22 | Total |
|---|---:|---:|---:|---:|
| archived Bayesian baseline | 3 | 17 | 4 | 24 |
| quadratic | 4 | 17 | 5 | 26 |
| no_pool | 8 | 17 | 4 | 29 |

These are retrospective ranked retrieval counts over 125 selections per taxon; do not claim prospective surveillance performance. Probability calibration-bin outputs were produced as an artifact, but have not yet been interpreted quantitatively.

## Manuscript-safe interpretation

A quadratic environmental response substantially improves held-out discrimination and precision-recall performance compared with the archived linear hierarchical Bayesian baseline. These gains are consistent across taxon-clustered exploratory analyses. Nonhierarchical species-specific fitting shows weaker overall evidence of improvement, although it retrieves more positives for one very rare taxon in a fixed-budget top-25 comparison. These are comparisons of fitted pipeline variants, not controlled isolation of the pooling mechanism. No independent external transfer validation is available. Manuscript remains private.
