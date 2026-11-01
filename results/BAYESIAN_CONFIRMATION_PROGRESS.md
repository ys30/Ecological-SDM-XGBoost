# Bayesian long-chain confirmation: verified progress

Confirmation uses four chains, 2,000 iterations per chain, 1,000 warmup per chain. The following values were read directly from the published `results/bayes_spatial_fold_{k}_diagnostics_confirm.csv` and `results/bayes_spatial_fold_{k}_sampler_diagnostics_confirm.csv` files.

| Fold | Parameters checked | Max R-hat | Min bulk ESS | Min tail ESS | Total divergences | Treedepth-limit hits | Min BFMI | Assessment |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 35 | 1.005731 | 1112.68 | 1368.01 | 0 | 0 | 0.8020 | Pass screening |
| 3 | — | — | — | — | — | — | — | Confirmation CSV not yet published |
| 5 | 35 | 1.004522 | 1008.98 | 1744.61 | 0 | 0 | 0.7356 | Pass screening |

Screening thresholds: R-hat <= 1.01, bulk/tail ESS >= 400, no divergent transitions or treedepth-limit hits, BFMI >= 0.3. These are diagnostic screening thresholds, not a guarantee of inferential validity.

Original short-chain runs for folds 1 and 4 had low bulk ESS in some parameters, so **full five-fold convergence has not been established**. Keep original cross-validation metrics and matched model comparisons separate from confirmation diagnostics unless confirmed predictions are explicitly regenerated and compared.

Next: obtain fold 3 confirmation output, then decide whether to confirm folds 1 and 4 to complete a uniform five-fold four-chain analysis.
