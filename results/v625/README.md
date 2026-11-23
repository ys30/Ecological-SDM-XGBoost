# v6.25 nested spatial tuning — public summary

This directory documents the exploratory nested spatial XGBoost analysis reported in manuscript Sections 2.8 and 3.6.

## Scope
- Designs: 4×4 quantile-grid blocks, 15-cluster k-means, and 10-km buffered quantile-grid blocks.
- Outer test records remain untouched during inner tuning.
- Tuned per-taxon and pooled XGBoost are compared against quadratic per-taxon logistic regression and MaxEnt.
- The expanded search and buffered sensitivity work are exploratory, not a prespecified confirmatory test.

## Selected matched AUC results (from author-provided v6.25b summary)

| Spatial design | Quadratic GLM | Tuned per-taxon XGBoost | Tuned pooled XGBoost | MaxEnt |
| --- | ---: | ---: | ---: | ---: |
| 4×4 blocks | 0.8188 | 0.7984 | 0.7981 | 0.7866 |
| 15 k-means clusters | 0.8201 | 0.7937 | 0.7896 | 0.8102 |
| 4×4 blocks + 10 km buffer | 0.8146 | 0.7801 | 0.7888 | 0.7813 |

Source: `EMS_v625b_GitHub_Review_Package.zip/results/v625/table_tuned_summary.csv`, supplied by the author. These are aggregate summary values, not an independently rerun CI result.

## Public/restricted separation
The full author-supplied package also contains hyperparameter search logs, fold-specific selections, calibration summaries, and per-record out-of-fold predictions. This directory currently includes only the public-facing summary. The complete v6.25 results package has **not** been uploaded to this GitHub directory. Do not cite this commit as proof that every file in the author archive is available here.

For full reproducibility, consult the associated author-held archive and its license/data-provenance review. No Zenodo DOI is asserted here.
