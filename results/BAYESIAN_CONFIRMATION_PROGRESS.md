# Bayesian four-chain confirmation: five-fold diagnostic audit

All five archived confirmation diagnostic files are now present and were checked directly:
`results/bayes_spatial_fold_{1..5}_diagnostics_confirm.csv` and
`results/bayes_spatial_fold_{1..5}_sampler_diagnostics_confirm.csv`.

Confirmation configuration: 4 chains per fold, 2,000 iterations per chain, 1,000 warmup per chain.

| Fold | Parameters | Max R-hat | Min bulk ESS | Min tail ESS | Divergences | Treedepth-limit hits | Min BFMI | Screening |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 35 | 1.004643 | 959.01 | 1599.85 | 0 | 0 | 0.6449 | Pass |
| 2 | 35 | 1.005731 | 1112.68 | 1368.01 | 0 | 0 | 0.8020 | Pass |
| 3 | 35 | 1.005064 | 1068.78 | 1615.97 | 0 | 0 | 0.7577 | Pass |
| 4 | 35 | 1.004493 | 871.31 | 1611.59 | 0 | 0 | 0.7156 | Pass |
| 5 | 35 | 1.004522 | 1008.98 | 1744.61 | 0 | 0 | 0.7356 | Pass |

**Result:** all 175 parameter-fold diagnostics pass R-hat <= 1.01 and bulk/tail ESS >= 400. Across the 20 confirmed chains, no divergent transitions or treedepth-limit hits are reported; minimum chain BFMI is 0.6449, above the 0.3 screening threshold.

**Interpretation and provenance:** These are diagnostic-screening results for the archived four-chain confirmation fits, not proof of model validity. The original shorter two-chain runs have 3 R-hat flags and 12 bulk-ESS flags; those original runs produced the published out-of-fold predictions and matched model comparisons. Do not imply that four-chain confirmed predictions replaced the original predictions unless a separate prediction and metric comparison is completed. Four-chain confirmation does not establish ecological transferability or superiority over competing models.

**Fold-level performance reconciliation:** `results/bayesian_five_fold_confirmed_metrics.csv` contains all five four-chain fold-level AUC/AP/logloss results. These have already been reconciled against the original metrics in `results/BAYESIAN_LONG_CHAIN_METRIC_RECONCILIATION.md`: maximum absolute AUC difference < 0.00033. The historical confirmation artifacts do not include four-chain per-record held-out predictions, so matched three-model figures remain based on original predictions. Do not claim per-record agreement or four-chain-derived three-model matched metrics.

**Automated verification:** GitHub Actions [Validate Bayesian four-chain metric stability, run 41356374575](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41356374575) succeeded using `scripts/39_validate_bayesian_confirmation_metrics.py`. Maximum absolute original-vs-confirmed changes across five folds: AUC 0.0003276484, AP 0.0013288474, logloss 0.0001763619. This CI verifies archived fold-level metrics only, not long-chain per-record predictions.

Next: update private manuscript Bayesian robustness wording using the confirmed diagnostics and fold-level metric stability; keep original three-model matched comparisons clearly labeled.
