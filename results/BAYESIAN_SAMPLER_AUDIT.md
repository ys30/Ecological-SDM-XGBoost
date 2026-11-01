# Bayesian five-fold Stan sampler audit

Verified from published `results/bayes_spatial_fold_{1..5}_sampler_diagnostics.csv`. Each fold has two chains with 600 post-warmup draws per chain.

| Fold | Divergences (total) | Max treedepth | At treedepth limit | Minimum chain BFMI |
|---|---:|---:|---:|---:|
| 1 | 0 | 8 | 0 | 0.8143 |
| 2 | 0 | 8 | 0 | 0.7067 |
| 3 | 0 | 8 | 0 | 0.6794 |
| 4 | 0 | 8 | 0 | 0.7021 |
| 5 | 0 | 9 | 0 | 0.8469 |

**Sampler-level finding:** no divergent transitions in 6,000 total post-warmup draws; no transitions at the configured treedepth limit (10); all chain-level BFMI estimates exceed the common warning threshold of 0.3.

**Important limitation:** sampler diagnostics alone do not establish full convergence. The existing parameter summary reports three parameter-fold R-hat values above 1.01 and twelve parameter-fold bulk ESS values below 400. Tail ESS was added to the fitting script for subsequent runs, but the archived aggregate diagnostic summary has not yet been refreshed with tail ESS. These flags should be investigated before treating Bayesian estimates as fully converged or presenting strong inferential claims.

The matched three-model predictive comparison remains descriptive and should not be interpreted as a statistically significant model ranking. The model fit is an exploratory spatial-CV research extension, not a fully validated Bayesian inferential analysis.
