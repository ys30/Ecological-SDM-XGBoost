# Bayesian hierarchical SDM: completed five-fold MCMC convergence audit

All values below were verified against the published `results/bayes_spatial_fold_{1..5}_diagnostics_confirm.csv` and `results/bayes_spatial_fold_{1..5}_sampler_diagnostics_confirm.csv` files. Each spatial fold has 4 chains, 2,000 iterations per chain, 1,000 warmup per chain (1,000 retained draws per chain).

| Fold | Parameters | Max R-hat | Min bulk ESS | Min tail ESS | Chains | Divergences | Treedepth-limit hits | Min BFMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35 | 1.004643 | 959.01 | 1599.85 | 4 | 0 | 0 | 0.644910 |
| 2 | 35 | 1.005731 | 1112.68 | 1368.01 | 4 | 0 | 0 | 0.802047 |
| 3 | 35 | 1.005064 | 1068.78 | 1615.97 | 4 | 0 | 0 | 0.757668 |
| 4 | 35 | 1.004493 | 871.31 | 1611.59 | 4 | 0 | 0 | 0.715574 |
| 5 | 35 | 1.004522 | 1008.98 | 1744.61 | 4 | 0 | 0 | 0.735583 |

**Overall:** 175/175 parameter-fold records meet R-hat <= 1.01, bulk ESS >= 400, tail ESS >= 400. All 20 chains show zero divergent transitions, zero treedepth-limit hits, and BFMI >= 0.3. These checks support satisfactory MCMC convergence diagnostics for the five independently refitted spatial folds.

**Scope and limitations:** This is a convergence-diagnostics result, not a model-validation guarantee. The original matched three-model comparison and published predictive metrics were computed from the original shorter-chain fits; do **not** silently relabel them as confirmed-fit predictive performance. To compare long-chain predictions, collect the corresponding `*_confirm.csv` prediction files and recompute metrics with consistent held-out records. Statistical and ecological limitations remain.
