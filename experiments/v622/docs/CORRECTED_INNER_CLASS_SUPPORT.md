# EMS v6.22 — corrected inner-fold class support (pre-freeze)

Source: [preregistered random-order greedy fold audit, run 42752158765](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42752158765), commit 9baeb28c2668d3c62697d62743ab472cf7505282.

The current implementation uses 12 k-means clusters from outer-training plots only, seed `6220 + outer_fold`, random cluster order and greedy assignment by plot count. It is a Python implementation of the draft plan; not a frozen study.

| Taxon | Validation AUC defined / 20 | Both train and validation AUC defined / 20 |
| --- | ---: | ---: |
| nsw18 | 20 | 20 |
| nsw19 | 13 | 13 |
| nsw20 | 7 | 4 |
| nsw21 | 14 | 14 |
| nsw22 | 9 | 7 |
| nsw23 | 20 | 20 |
| nsw24 | 20 | 20 |
| nsw25 | 20 | 20 |
| Total | 123 / 160 | 118 / 160 |

There are 37 validation taxon-folds with zero positives and 5 training taxon-folds without both classes. No validation partition is zero-negative. Counts match the previous aggregate by coincidence; the species-level distribution differs. All counts are diagnostics, **not model results**.

## Already specified in config/v622_experiment.yaml

- Fallback when fewer than 5 positive training examples: constant prediction at **training prevalence**, flagged.
- Inner probabilistic tuning: equal-taxon mean relative log-loss skill with training-only smoothed null prevalence and 1e-6 probability clipping.
- AUC contrasts require both classes and neither paired model in fallback.
- MaxEnt selection uses inner within-taxon AUC; folds without both classes require a documented, consistent handling policy before tuning. Do not silently substitute a different objective.

## Open gates

1. Verify predictor matrix is taxon-specific: `Species` is the taxon identifier and `pres.abs` is the binary response. Do not include the response or plot ID as predictors. Verify whether pooled models may encode Species under their separate model matrix.
2. Ensure all inner folds honor fallback and use train-only encoders; audit each model family.
3. Check MaxEnt tuning feasibility and freeze its treatment of AUC-undefined folds.
4. Confirm km15 pinned files, formal projected CRS, full runner, and independent code review.
5. No formal fitting, outer predictions, or performance estimates before a distinct freeze commit.
