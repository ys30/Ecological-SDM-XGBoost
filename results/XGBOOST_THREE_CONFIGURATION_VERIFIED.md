# Three-configuration XGBoost spatial sensitivity: verified CI

Successful run: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41387403515

Five held-out spatial folds; 1,167 plots, 9,336 records. The workflow uploads `xgboost_spatial_encoding_sensitivity.csv` as an artifact (it does not publish it into the tracked results tree).

| Fold | Full fold-local AUC | Environment + Species AUC | Environment-only AUC |
|---|---:|---:|---:|
| 1 | 0.8917472 | 0.8891244 | 0.5510408 |
| 2 | 0.9355313 | 0.9350747 | 0.5856656 |
| 3 | 0.8868710 | 0.8878875 | 0.5557685 |
| 4 | 0.9063138 | 0.9047007 | 0.5687534 |
| 5 | 0.9175092 | 0.9203559 | 0.6091027 |

| Fold | Full fold-local AP | Environment + Species AP | Environment-only AP |
|---|---:|---:|---:|
| 1 | 0.37701015 | 0.37915894 | 0.08018330 |
| 2 | 0.57905375 | 0.58252168 | 0.11484631 |
| 3 | 0.41292026 | 0.43431048 | 0.11556275 |
| 4 | 0.39086597 | 0.41269578 | 0.09418521 |
| 5 | 0.45444318 | 0.49188640 | 0.10471136 |

## Interpretation
- The four environmental variables alone yield substantially weaker discrimination in a **pooled model that cannot distinguish species identities**.
- Retaining Species while removing disturbance and soil fertility nearly preserves full-model AUC and modestly improves mean AP in these five folds. These are descriptive results, not a formal equivalence test or causal feature importance estimate.
- This does **not** establish that environmental variables are ecologically unimportant, nor that XGBoost is intrinsically superior to species-specific MaxEnt or hierarchical Bayesian SDMs.
- The full model uses four numerical environmental predictors and three categorical predictors (`Species`, `disturb`, `soilfert`). Avoid describing it as six environmental predictors.
- MaxEnt and Bayesian use the four numerical environmental predictors; the models differ in species structure and output semantics.
- This is a spatial block validation, not distance-buffered CV. Nearest-training distance audit: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41385657460

## Next verification
Run a **species-stratified** held-out comparison (per species-fold AUC/AP) for the three XGBoost configurations, rather than relying on pooled AUC that can be driven by between-species prevalence differences. Do not alter the private manuscript until the comparison is interpreted appropriately.
