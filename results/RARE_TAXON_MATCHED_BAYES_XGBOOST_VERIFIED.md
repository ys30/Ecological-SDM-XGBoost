# Verified matched rare-taxon top-k retrieval: Bayesian versus XGBoost

This is a public, anonymized result audit, **not** a manuscript. Source data: `results/bayesian_rare_species_within_fold_retrieval.csv` and `results/v614_common_input/rare_taxon_within_fold_retrieval.csv`. The counts below were independently recomputed by summing the archived five spatial folds for each taxon and budget. Top-k is applied **per fold**, so total inspected is 5k, not k.

| Taxon | Positives | Budget per fold | Bayesian captured | Taxon-specific XGBoost captured | Difference in recall (XGB - Bayes) |
|---|---:|---:|---:|---:|---:|
| nsw19 | 8 | 10 | 1 | 5 | +0.5000 |
| nsw19 | 8 | 25 | 3 | 7 | +0.5000 |
| nsw19 | 8 | 50 | 6 | 8 | +0.2500 |
| nsw20 | 18 | 10 | 9 | 12 | +0.1667 |
| nsw20 | 18 | 25 | 17 | 16 | -0.0556 |
| nsw20 | 18 | 50 | 18 | 18 | 0.0000 |
| nsw22 | 6 | 10 | 2 | 0 | -0.3333 |
| nsw22 | 6 | 25 | 4 | 1 | -0.5000 |
| nsw22 | 6 | 50 | 6 | 1 | -0.8333 |

## Interpretation and limitations

- XGBoost retrieves more nsw19 presences across all budgets; Bayesian retrieves more nsw22 presences across all budgets.
- The model ranking can reverse with taxon and budget (nsw20).
- These are **retrospective out-of-fold retrieval** outcomes, not prospective field-trial outcomes.
- There are only three rare taxa and five spatial folds. Do not treat individual presences or folds as independent replicates or infer population-wide model superiority.
- The two model pipelines may differ in predictors and fitting strategy; the result is a **pipeline-level comparison**, not proof of an isolated algorithm effect.
- No original taxon names or unpublished manuscript text are included.

Reproduction script: `scripts/54_compare_rare_taxon_retrieval.py`. Figure generation and automatic publishing are defined in `.github/workflows/compare-rare-taxon-bayes-xgboost.yml`. The figure is not claimed to be CI-verified here.
