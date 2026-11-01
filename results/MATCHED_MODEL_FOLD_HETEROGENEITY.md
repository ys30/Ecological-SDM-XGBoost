# Fold-level heterogeneity and model-ranking stability

This audit examines **which model ranked first within each eligible species–fold**, separately for ROC AUC and average precision. It uses the committed `results/matched_bayesian_maxent_xgboost_folds.csv`, containing 23 matched species–fold rows. Ranking is based on the largest observed metric in each row, with no inferential claim.

| Species | Eligible folds | Bayesian AUC wins | MaxEnt AUC wins | XGBoost AUC wins | Bayesian AP wins | MaxEnt AP wins | XGBoost AP wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Angophora costata* | 5 | 0 | 1 | 4 | 0 | 0 | 5 |
| *Eucalyptus blakelyi* | 3 | 0 | 2 | 1 | 0 | 2 | 1 |
| *Eucalyptus campanulata* | 5 | 0 | 2 | 3 | 0 | 1 | 4 |
| *Eucalyptus fastigata* | 5 | 2 | 1 | 2 | 2 | 2 | 1 |
| *Eucalyptus nova-anglica* | 5 | 0 | 1 | 4 | 1 | 2 | 2 |
| **Total** | **23** | **2** | **7** | **14** | **3** | **7** | **13** |

## Implications

- The best model varies across spatial folds within a species. In particular, *E. fastigata* has no dominant AUC winner (Bayesian 2, MaxEnt 1, XGBoost 2).
- XGBoost leads on AUC in 14/23 individual species–fold comparisons, but this does **not** mean it has the highest species-mean AUC for 14 species; the species-mean ranking uses a different unit of aggregation.
- MaxEnt leads two of three eligible *E. blakelyi* folds for both AUC and AP. The two ineligible folds must **not** be imputed as losses or wins for any model.
- Winner counts are descriptive and omit the size of model differences. Spatial folds are dependent and class counts vary. No binomial p-values or confidence intervals are warranted from these counts.

## Missing-fold sensitivity

The matched dataset excludes two *E. blakelyi* folds because they are not evaluable under the common comparison rule. Since AUC is undefined for a fold lacking one outcome class, simply assigning AUC=0.5 or carrying forward the observed-fold mean would fabricate data. The existing matched comparison therefore identifies performance only for **eligible folds**, not all spatial folds for all eight species. A fuller assessment requires documenting outcome-class counts for every species–fold and reporting which exclusion rule was applied before scoring.

## Reproducibility

The row-wise winner is the model with the largest value among `bayesian_auc`, `maxent_auc`, and `xgboost_auc` (or the corresponding `*_ap` columns). Values are from original shorter-chain Bayesian predictions, not long-chain confirmation predictions.
