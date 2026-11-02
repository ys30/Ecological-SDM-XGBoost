# XGBoost spatial sensitivity: first verified CI results

CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41386760979 (success; artifact `xgboost-spatial-encoding-sensitivity`).

## Held-out AUC by fold

| Fold | Archived XGBoost | Fold-local one-hot XGBoost | Four environmental predictors only |
|---|---:|---:|---:|
| 1 | 0.889918 | 0.891747 | 0.551041 |
| 2 | 0.937308 | 0.935531 | 0.585666 |
| 3 | 0.888425 | 0.886871 | 0.555769 |
| 4 | 0.903382 | 0.906314 | 0.568753 |
| 5 | 0.918692 | 0.917509 | 0.609103 |

Fold-local one-hot AUC differs from archived model by at most ~0.00294 across folds. However, **the encoding schemes are not exactly identical**: the archived `model.matrix(~ . - 1)` uses R's contrast coding for some factor columns, while the new sensitivity uses explicit all-level one-hot coding. Thus the differences do not isolate train-only versus full-data vocabulary effects.

The four-variable model omits `Species`, `disturb`, and `soilfert` together. Its low AUC must **not** be attributed to removing two environmental predictors alone, nor interpreted as a fair comparison against species-specific MaxEnt or hierarchical Bayesian models. The pooled four-variable XGBoost model cannot express species-specific prevalence without species identity.

## Next experiment
Fit a four-environmental-predictor **plus Species identity** XGBoost pipeline, with fold-local encoding and the same folds and hyperparameters. This better separates environmental-variable reduction from the loss of species identity. It remains a pipeline sensitivity, not a fully controlled algorithm comparison.

Source: archived `results/spatial_2d_block_cv_metrics.csv` and successful CI logs for the sensitivity run.
