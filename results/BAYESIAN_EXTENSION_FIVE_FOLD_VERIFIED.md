# Bayesian extension: verified five-fold paired comparison

Source: GitHub Actions [41771738887](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41771738887), successful; five-fold fits [41769286020](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41769286020), all 10 jobs successful.

## Paired differences against archived shorter-chain hierarchical Bayesian baseline

| Variant | Eligible taxon-fold AUC cells | Mean ΔAUC | Mean ΔAP | Mean ΔBrier | Mean ΔLog loss |
|---|---:|---:|---:|---:|---:|
| no_pool | 30 | +0.010708 | +0.013707 | +0.000389 | -0.000658 |
| quadratic | 30 | +0.037831 | +0.081273 | -0.006775 | -0.020342 |

Positive AUC/AP differences are better; negative Brier/log-loss differences are better. AUC and AP average eligible taxon-fold cells; Brier and log-loss average all 40 taxon-fold cells. The comparison uses matched OOF records (9,336 per variant), and the baseline is the archived shorter-chain run. The new quadratic model adds squared standardized rainfall and temperature terms, whereas no_pool uses species-specific fixed coefficients. Both retain the original four environmental predictors.

**Interpretation:** Quadratic response terms improve all four descriptive metrics. Nonpooled fitting slightly improves discrimination/ranking while worsening Brier. These results do not establish a statistical significance claim or a causal isolation of hierarchy, since the nonpooled model also changes coefficient structure and prior specification.

**Next:** taxon-clustered paired uncertainty intervals and taxon-wise heterogeneity; rare-species top-25 retrieval; calibration and geographic residual diagnostics; check full-fold outputs and retain OOF files for reproducibility. Do not publicly commit the private manuscript.
