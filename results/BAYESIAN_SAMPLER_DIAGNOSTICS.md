# Bayesian spatial CV: Stan sampler diagnostics audit

Verified from `results/bayes_spatial_fold_{1..5}_sampler_diagnostics.csv`. Each of five spatial folds used 2 chains with 600 post-warmup draws per chain.

| Fold | Divergences (both chains) | Treedepth-at-limit (both chains) | Minimum chain BFMI | Maximum observed treedepth |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 0.8143 | 8 |
| 2 | 0 | 0 | 0.7067 | 8 |
| 3 | 0 | 0 | 0.6794 | 8 |
| 4 | 0 | 0 | 0.7021 | 8 |
| 5 | 0 | 0 | 0.8469 | 9 |

All 10 chains had zero divergent transitions and zero iterations at the configured treedepth limit of 10. BFMI exceeded 0.3 for all chains. These results indicate no obvious *sampler-pathology* warning in the exported Stan diagnostics.

**Important qualification:** These checks are not proof of complete posterior convergence or sufficient effective sample size. The existing parameter audit (`results/bayes_mcmc_diagnostics_summary.csv`) reports 3 parameter-fold R-hat values above 1.01 and 12 bulk-ESS values below 400. Tail ESS is not yet comprehensively summarized across folds. These limitations must remain visible in manuscripts and recruiter-facing descriptions.

Next steps: audit per-parameter tail ESS and R-hat, identify weakly sampled coefficients, then consider longer chains / more warmup for affected folds. Do not claim full MCMC convergence until those checks pass.
