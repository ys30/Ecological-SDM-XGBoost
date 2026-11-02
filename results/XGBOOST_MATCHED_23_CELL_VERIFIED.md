# Verified 23-cell matched XGBoost sensitivity comparison

CI: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41389562346 (success).

The script `scripts/42_compare_xgboost_species_fold_sensitivity.R` joined each sensitivity configuration to exactly 23 archived species × spatial-fold cells, checking species/fold keys, held-out sample counts, and positive counts. Four CSVs were uploaded as the workflow artifact `xgboost-spatial-encoding-sensitivity`.

## Mean held-out metrics on the same 23 cells

| Model | AUC | AP |
|---|---:|---:|
| XGBoost, 4 environmental predictors + Species | 0.8028266 | 0.3451846 |
| XGBoost, full fold-local encoding | 0.7990313 | 0.3493455 |
| Archived XGBoost | 0.8016246 | 0.3401105 |
| MaxEnt | 0.7865459 | 0.3135298 |
| Hierarchical Bayesian | 0.7610276 | 0.2838372 |
| XGBoost, 4 environmental predictors only (no Species) | 0.6213794 | 0.2104910 |

## Interpretation and limitations

The near-equality of the first three XGBoost AUC means is descriptive; no equivalence test has been performed. Pooled environment-only XGBoost does not encode species identity, whereas the other models use species identity either as a predictor or through species-specific/hierarchical modeling. Consequently, the environment-only result is not a controlled model-family comparison and does not imply environmental predictors are unimportant.

The original matched comparison includes only 23 eligible species-fold cells; three rare taxa are handled separately. These means are unweighted across cells and should not be confused with pooled observation-level AUC.

MaxEnt outputs relative suitability, so compare discrimination/ranking metrics rather than treating its scores as calibrated probabilities. Spatial CV uses non-buffered blocks; the minimum distance from held-out to training plots can be small.

Next: compute per-cell paired deltas and species-stratified summaries; consider uncertainty with only five spatial folds and correlated cells. Keep unpublished manuscript private.
