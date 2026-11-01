# Bayesian hierarchical SDM: parameter convergence review

Source: the five published `results/bayes_spatial_fold_{1..5}_diagnostics.csv` files, 35 diagnostic variables per fold (175 parameter-fold records). Thresholds are screening rules, not formal proof of convergence.

## R-hat > 1.01 (3 records)

| Fold | Variable | R-hat | Bulk ESS |
|---|---|---:|---:|
| 2 | `sd_species__z_rainann` | 1.014528 | 350.83 |
| 2 | `r_species[nsw22,z_rainann]` | 1.011172 | 492.64 |
| 5 | `b_z_topo` | 1.011098 | 1075.71 |

## Bulk ESS < 400 (12 records)

| Fold | Variable | Bulk ESS |
|---|---|---:|
| 3 | `lp__` | 190.99 |
| 4 | `lp__` | 266.84 |
| 1 | `lp__` | 319.71 |
| 2 | `lp__` | 328.91 |
| 3 | `sd_species__z_rainann` | 337.76 |
| 5 | `lp__` | 340.80 |
| 2 | `sd_species__z_rainann` | 350.83 |
| 5 | `b_z_tempann` | 351.92 |
| 5 | `r_species[nsw24,z_tempann]` | 362.22 |
| 5 | `r_species[nsw23,z_tempann]` | 369.03 |
| 5 | `r_species[nsw18,z_tempann]` | 373.34 |
| 4 | `r_species[nsw23,Intercept]` | 391.87 |

The 12 low-ESS records consist of 5 log posterior `lp__`, 4 species-specific effects, 2 population-level effects, and 1 hierarchical standard deviation.

## Interpretation and next validation

- The sampler-level audit reports zero divergences, no treedepth saturation, and minimum BFMI 0.679 across five folds.
- These results are encouraging but do not resolve parameter-level R-hat/ESS flags.
- Prioritize fold 2 rainfall hierarchical standard deviation and species-specific rainfall effect; then fold 5 topographic population coefficient and temperature effects.
- Recommended confirmatory refit: 4 chains, at least 2,000 iterations per chain with 1,000 warmup, fixed fold assignments and unchanged priors/predictors, record R-hat, bulk/tail ESS, divergences, treedepth, BFMI, and out-of-fold prediction metrics. Do not replace the published five-fold results until reruns are verified.
- The existing diagnostics do not contain tail ESS, so tail convergence is not yet established.
