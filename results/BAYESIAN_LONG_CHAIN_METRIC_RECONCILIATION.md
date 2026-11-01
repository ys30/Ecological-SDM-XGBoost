# Bayesian spatial CV: original versus long-chain refit metrics

The confirmed four-chain metrics were recovered from artifacts of successful GitHub Actions runs [41279140198](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41279140198) and [41277076588](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41277076588). Original metrics are from `results/bayes_spatial_cv_metrics.csv`. Both results use the same five held-out spatial folds, but independent MCMC fits.

| Fold | N | Original AUC | Long-chain AUC | Original AP | Long-chain AP | Original logloss | Long-chain logloss |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1896 | 0.881808 | 0.881812 | 0.270027 | 0.270180 | 0.203829 | 0.204005 |
| 2 | 1872 | 0.910336 | 0.910445 | 0.438132 | 0.438311 | 0.193517 | 0.193438 |
| 3 | 1872 | 0.868213 | 0.868243 | 0.345772 | 0.345660 | 0.239139 | 0.239074 |
| 4 | 1832 | 0.902645 | 0.902818 | 0.415813 | 0.417142 | 0.176866 | 0.176781 |
| 5 | 1864 | 0.903416 | 0.903744 | 0.346244 | 0.346422 | 0.180696 | 0.180526 |

All five per-fold AUC changes are smaller than 0.00033 in magnitude. This supports *numerical stability of fold-level predictive metrics* between short- and long-chain fits, not equivalence of per-record predictions or intervals.

**Artifact limitation:** Six inspected historical confirmation artifacts contain metrics and convergence diagnostics but **no `*_predictions_*_confirm.csv` files**. The workflow at execution time did not upload predictions. The updated confirmation workflow now lists prediction files for future artifacts, but this cannot retroactively recover absent historical files. Do not relabel original short-chain predictions or matched Bayesian–MaxEnt–XGBoost figures as long-chain predictions. Avoid costly re-fitting solely for an unneeded figure update; any future three-model long-chain matched comparison needs actual long-chain held-out predictions and a consistent join by observation identifier.
