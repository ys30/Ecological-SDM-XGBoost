# EMS v6.22 — cross-branch implementation reconciliation

Status: PRE-FREEZE / no new outer model results.
Reviewed: existing `ems-v622-experiments` branch against exploratory `research/v6.22-nested-spatial-tuning` branch.

## Decision

**The established `ems-v622-experiments` branch and its pinned YAML/model matrix/contrasts are authoritative.** The exploratory branch is not an alternative preregistration and its pilot must not be used as a confirmatory v6.22 run without conforming to the authoritative specifications.

## Verified mismatches

| Topic | Authoritative v6.22 YAML | Exploratory scripts/91_v622_c1_runtime_pilot.R | Required action |
| --- | --- | --- | --- |
| Inner cluster assignment | Random cluster order then greedy balance by plot count | Descending cluster size then greedy balance | Implement random-order allocation exactly; assert matching reference fold IDs |
| XGBoost candidate generation | 30 configurations; depth 1–6; learning rate loguniform 0.01–0.3; child weight loguniform 0.5–20; subsample/colsample uniform 0.5–1; rounds in 50,100,200,400,800 | Discrete choices with different ranges and 100,200,300,500 rounds | Generate identical frozen candidates across outer folds and variants |
| Relative log loss | Smoothed null prevalence (positives+0.5)/(n+1); clip 1e-6 | Unsmoothed training prevalence; clip 1e-15 | Match YAML exactly, including fallback handling |
| Aggregation | Equal taxon weighting over evaluable inner-validation folds | Fold means of taxon means | Define/verify taxon-first aggregation and missing-cell coverage |
| Failure handling | Retry once with documented policy then flagged fallback | No fit-error retry | Implement and test retry/fallback with diagnostics |
| Pilot scope | All tuned models, inner tuning and final fit, no outer test predictions/metrics | Only pooled and per-taxon XGBoost, no final fit | Label exploratory pilot as partial C1 smoke only; not protocol-compliant full pilot |
| km15 input | results/ems_revision/kmeans15_five_fold_assignments.csv, pinned SHA256 | Regenerated results/alternative_spatial_validation/kmeans15_five_fold_assignments.csv | Use exact pinned source, never silently substitute generated file |

## Existing supporting evidence

- `experiments/v622/docs/CORRECTED_INNER_CLASS_SUPPORT.md`: 123/160 inner validation AUC-defined cells; 118/160 have both training and validation AUC defined. These are class-support diagnostics, not fitted-model results.
- `experiments/v622/docs/CRS_SCALE_DIAGNOSTIC_RESULT.md`: median projected/geodesic nearest-neighbor ratio 1.0041124; plausible metre scale, formal CRS not verified.
- `experiments/v622/docs/FREEZE_READINESS.md`: full preregistration preflight passed; substantive scientific readiness and model-runner verification remain open.

## Blocking gates before formal fitting

1. Confirm the exact 19-model/20-contrast frozen candidate and all pinned input hashes.
2. Audit and test every model's train-only feature encoding, imputation and scaling.
3. Verify projected CRS from authoritative metadata before interpreting distance buffers.
4. Resolve MaxEnt undefined-AUC inner-fold selection policy and probabilistic comparability.
5. Implement the authoritative candidate sampler and taxon-first objective with unit tests.
6. Execute a protocol-compliant runtime-only pilot after an explicit freeze commit; no outer performance metrics during pilot.
7. Preserve matched taxon-fold coverage and all fallback logs in subsequent analysis.

This audit changes no scientific result and authorizes no interpretation of new AUC/AP/log-loss values.
