# v6.22 C1 — execution gate (2026-11-14)

Status: **protocol / audit only; not evidence of a completed XGBoost C1 run**.

## Verified starting point
- The repository contains archived v6.14 common-input XGBoost comparisons, including a metric eligibility audit (30/40 taxon-fold cells eligible for AUC/AP; Brier defined for all 40).
- A separate five-outer-fold, 10-km buffered nested Bayesian vs quadratic GLM threshold study has verified aggregate results for three rare taxa. This is **not** an XGBoost nested hyperparameter study.
- Keep archived v6.14 results, v6.21 manuscript, and Bayesian/GLM validation immutable.

## C1 pilot acceptance checks (fail closed)
1. Load the actual canonical observation-level data and original plot IDs; record source paths, SHA-256, row counts, taxon IDs, and outcome counts. No synthetic replacement data.
2. Reuse the designated outcome-blind five-fold spatial partition; verify every plot's records share a fold and no plot appears in both outer train/test.
3. For the selected pilot outer fold, construct all inner spatial folds *only from outer training plots*. Apply any 10-km buffer to inner training points relative to inner validation points, and outer training points relative to outer test points. Record counts and coordinate CRS; do not label a distance “km” unless units are established.
4. Any scaling, imputation, feature selection, early stopping, class weighting, and tuning must be fitted inside the relevant training fold. No outer-test labels, prevalence, predictions, or thresholds may influence tuning.
5. Freeze 30 candidate configurations and a deterministic seed before running. Optimize the mean across eight taxa of relative log-loss improvement, with null log loss computed using **inner training** prevalence and clipped probabilities; document what happens when a training fold is single-class or has zero null loss.
6. Select parameters using only inner validation; refit on the permitted outer training data and predict untouched outer test plots exactly once.
7. Emit per-observation OOF predictions keyed by anonymous taxon, plot ID and outer fold; emit fold metrics (log loss, Brier, AUC/AP only when both test classes exist), parameter selections, sample counts, buffer exclusions and software versions.
8. Fail the job on duplicate keys, missing predictions, data overlap, nonfinite probabilities, unexpected taxon counts or changed input hashes.

## Deliverables
- `c1_manifest.json`: inputs, hashes, seed, git SHA, software versions, spatial partition and CRS evidence.
- `c1_inner_scores.csv`: 30 candidates × inner folds × taxon, with counts and relative log loss.
- `c1_outer_oof.csv`: held-out row keys, truth, predicted probability, model/config, fold.
- `c1_metric_eligibility.csv`: explicit reason for undefined AUC/AP.
- `c1_gate_report.md`: PASS/FAIL for each safeguard and exact CI URL.

## Stop rules
Do not launch C2–C4 or update manuscript performance claims until the C1 gate passes with real data and an inspectable CI artifact. Do not equate successful workflow dispatch with completed validation.
