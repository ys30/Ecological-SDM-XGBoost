# EMS v6.14 — archived common-input experiment results

This report summarizes **CSV files already committed** under `results/v614_common_input/`. It is a results audit, not a manuscript. No original species identities are disclosed.

## Common-input pooling and species-identity ablation

Source: `three_way_taxon_cluster_uncertainty.csv`. Estimates are **equal-taxon means** over eight anonymous taxa; AUC/AP use 30 eligible taxon–fold cells, while Brier uses all 40.

| Contrast | Metric | Equal-taxon mean | Descriptive taxon bootstrap 95% range | Exact taxon sign-flip tail fraction |
|---|---|---:|---:|---:|
| Pooled four predictors + species minus taxon-specific four predictors | AUC | -0.012310 | [-0.051011, 0.021995] | 0.585938 |
| Same contrast | AP | +0.001549 | [-0.030417, 0.033557] | 0.937500 |
| Same contrast | Brier (lower is better) | -0.005340 | [-0.009052, -0.002137] | 0.007812 |
| Species-blind pooled four predictors minus pooled four + species | AUC | -0.313131 | [-0.461943, -0.179326] | 0.007812 |
| Same contrast | AP | -0.182210 | [-0.302884, -0.085682] | 0.007812 |
| Same contrast | Brier | +0.019168 | [0.004741, 0.038439] | 0.007812 |

**Interpretation:** On this dataset, pooled versus taxon-specific ranking performance is mixed, whereas removing species identity from the pooled model substantially degrades AUC/AP. These are *not* independent 30-cell experiments: five spatial folds overlap in training data, and the taxon count is only eight. Sign-flip fractions and bootstrap ranges are exploratory, not confirmatory generalization guarantees. The Brier comparison includes single-class folds; AUC/AP cannot.

## Rare-taxon retrieval under a common per-fold inspection budget

Source: `rare_taxon_retrieval_summary.csv`. Each budget applies **separately in each of five spatial folds**, not once globally. Captured positives are summed across folds.

| Anonymous taxon | Total positives | Top 10 per fold | Top 25 per fold | Top 50 per fold |
|---|---:|---:|---:|---:|
| nsw19 | 8 | 5 (62.5%) | 7 (87.5%) | 8 (100%) |
| nsw20 | 18 | 12 (66.7%) | 16 (88.9%) | 18 (100%) |
| nsw22 | 6 | 0 (0%) | 1 (16.7%) | 1 (16.7%) |

The retrieval experiment is **retrospective** and uses out-of-fold predictions. Taxon nsw22 demonstrates substantial heterogeneity: no universal rare-taxon retrieval success claim is justified. These are taxon-specific XGBoost results. A **separate verified matched comparison against Bayesian** is now available in [the rare-taxon comparison report](RARE_TAXON_MATCHED_BAYES_XGBOOST_VERIFIED.md); a matched MaxEnt top-k comparison is not established here.

## Public artifacts and reproducibility

- `results/v614_common_input/pooled_vs_specific_four_input.csv`
- `results/v614_common_input/three_way_xgboost_ablation.csv`
- `results/v614_common_input/three_way_taxon_cluster_uncertainty.csv`
- `results/v614_common_input/rare_taxon_retrieval_summary.csv`
- `results/v614_common_input/rare_taxon_within_fold_retrieval.csv`
- `.github/workflows/ems-v614-common-input.yml`

**Metric eligibility audit:** the archived v6.14 table has 40 taxon–fold rows, of which 30 contain both outcome classes and support AUC/AP; the other 10 are single-class folds with undefined AUC/AP but valid Brier scores. The source table labels all 40 rows `ok`; this is a status-label limitation, not evidence that AUC/AP were calculated in single-class folds. [Verified row-level eligibility audit](v614_common_input/metric_eligibility_audit.csv).

## Limitations before EMS submission

1. No independently validated second-region or second-dataset experiment.
2. No direct algorithm-only claim from comparisons using unequal predictors or pooled taxon information.
3. Check prediction calibration, geographic separation, and top-k evaluation under identical held-out units.
4. Keep manuscript drafts private; this report is an anonymized, public result summary.
