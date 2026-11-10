# EMS v6.22 — C1 real-input smoke and pooled fallback audit (2026-11-10)

## Verified execution

Workflow: https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42918001456

The run verified SHA-256 of `data/train.csv` and `results/spatial_2d_block_assignments.csv`, selected grid outer fold 1, excluded its test plots, and generated the preregistered train-only KMeans inner folds (12 clusters, 4 folds, seed 6221). It used **one** sampled XGBoost configuration, not the preregistered 30-candidate tuning search.

| Variant | Outer-training plots | Inner taxon-fold cells | Successful prediction cells | Insufficient-positive fallback cells | Actual model fits | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| Per-taxon | 930 | 32 | 26 | 6 | 26 | 0.33 s |
| Pooled | 930 | 32 | 26 | 6 | 4 | 0.20 s |

These are **engineering execution counts**, not accuracy estimates. There were no outer-test predictions or manuscript metrics.

## Scientific policy interpretation requiring sign-off

The frozen draft says: *Taxon-level training groups with fewer than five positives use a training-prevalence constant predictor*. The current C1 runner checks each taxon's training positives even when the model variant is pooled. Accordingly, six taxon-fold cells use constant predictions for both variants, **despite the pooled training set potentially having adequate positives overall**.

This is consistent with a literal taxon-level fallback rule but is not equivalent to a fully shared pooled model that supplies predictions for all taxa. It can change C3 sharing conclusions and comparison coverage. **Do not silently change the rule.** Before preregistration freeze, explicitly choose whether pooled model predictions are allowed for rare taxa, and whether the minimum-five-positives rule is taxon-specific or model-fitting-group-specific. If choosing a different rule, amend the plan, YAML, contrast eligibility and tests together.

## Remaining engineering and validity gates

1. The C1 runner currently constructs raw numeric feature matrices directly. It rejects nonfinite values but does not call `FoldEncoder`; verify whether the raw four-predictor XGBoost contract allows this, and ensure no train/test leakage if preprocessing is added.
2. The backend's retry doubles boosting rounds. Confirm this is the intended XGBoost-specific interpretation of the generic retry policy.
3. Check candidate scores under all 30 configurations and both spatial schemes; this smoke only exercised one candidate and one outer fold.
4. Confirm formal coordinate reference system and 10-km buffer before claiming physical distance validity.
5. Check the pooled-cache key against fold identity, training contents and configuration, and ensure retry diagnostics are archived.
6. Keep all performance claims out of the manuscript until full nested evaluation and paired contrasts are complete.

**Status:** Real-data C1 execution smoke passed; scientific freeze remains blocked.
