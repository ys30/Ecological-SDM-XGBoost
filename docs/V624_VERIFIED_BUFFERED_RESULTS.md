# v6.24 buffered analysis results

## C1

[Actions 44908588964](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44908588964) completed all 32 jobs: the inner-only pilot, 30 outer-fold-by-seed jobs, and the OOF validator. The validator confirmed 9,336 uniquely keyed held-out observations per seed. Every fold and seed selected candidate 0, the same depth-2, 100-tree, learning-rate-0.03 configuration as the fixed comparator. Five of the 40 fold-by-taxon combinations use the prespecified fixed fallback and remain in this policy comparison.

| Seed | Log loss in both arms | Brier in both arms |
| ---: | ---: | ---: |
| 17 | 0.185571 | 0.055649 |
| 42 | 0.185753 | 0.055648 |
| 101 | 0.187270 | 0.055952 |
| 202 | 0.186311 | 0.055804 |
| 622 | 0.186905 | 0.055984 |
| 2026 | 0.186184 | 0.055792 |

Paired predictions are identical. The paired log-loss difference and its conditional 16-block bootstrap interval are zero for each seed. This result describes the frozen grid and buffer. It does not estimate the benefit of searching a different grid or support claims about transfer to an independent region. The six-seed log-loss range is 0.001699. At seed 622 the pinned environment reproduces the archived v6.23 score to six decimals. The cause of the separately reported seed-622 value remains unassigned without its exact environment and implementation.

## Buffered exploratory C2–C4

The adjusted [comparator refit run](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44909037033) succeeded. Its predecessor failed because a glmnet sparse-class warning used unexpected whitespace; the corrected run handles sparse folds and uses a training-prevalence fallback for per-taxon ridge fits below five detections. The [paired audit](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44909075142) verified observation keys and matched training membership against the C1 hashes.

| Contrast | First arm minus second | Mean ΔAUC | Eligible cells |
| --- | --- | ---: | ---: |
| C2 | Per-taxon fixed ridge, quadratic minus linear | +0.03633 | 30 |
| C2 | Partially shared ridge slopes, quadratic minus linear | +0.00636 | 30 |
| C2 | Fully shared ridge slopes, quadratic minus linear | −0.02538 | 30 |
| C3 | Quadratic partially shared ridge slopes minus per-taxon ridge | −0.12102 | 30 |
| C3 | Quadratic fully shared ridge slopes minus per-taxon ridge | −0.30711 | 30 |
| C4 | XGBoost tuning policy minus per-taxon fixed quadratic ridge | −0.00407 | 30 |
| C4 | XGBoost tuning policy minus fixed MaxEnt | +0.00353 | 23 |
| C4 | Fixed MaxEnt minus per-taxon fixed quadratic ridge | −0.01920 | 23 |

The fixed ridge and MaxEnt specifications delimit these results. They are exploratory single-design comparisons and do not complete the prospective GLMM and all-tuned C4 contrasts. The C3 estimates combine sharing structure, regularization and fallback policy. MaxEnt outputs are relative suitability and were evaluated only on rank metrics. For MaxEnt, 6,297 records had scores and 23 paired cells had both classes. The 23 cells do not replace the 23 spatial schemes referenced in the v6.22 plan. See the [evidence inventory](V624_RESTORED_EVIDENCE_INDEX.md) for recovered historical analyses.
