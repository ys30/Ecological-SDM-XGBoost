# C2a preregistration reconciliation gate — 2026-11-13

## Verified
- All 320 nested spatial taxon–inner partitions passed CI run 43553777585; 47 training partitions require the prespecified class-support fallback.
- Native R glmnet diagnostic outer smoke passed run 43555550986: nsw18, grid4, outer fold 1, 237 held-out plots and two model arms (474 prediction rows).
- Paired diagnostic AUC: linear 0.652240; quadratic rain/temperature 0.866853. These numbers **must not** be described as confirmatory or manuscript results.
- Lambda-boundary audit passed run 43561881305: quadratic selected the minimum lambda 1e-6.

## Critical protocol differences
The configuration file `experiments/v622/config/v622_experiment.yaml` is explicitly marked DRAFT. Its glmnet search space specifies **30-value path per alpha (glmnet default ratio)** and inner selection, not the diagnostic fixed numeric grid `exp(seq(log(10),log(1e-6),length.out=30))`.

The configuration also specifies a pilot with **runtime only**: outer-fold predictions are not written and outer metrics are not computed. Our diagnostic smoke intentionally did produce held-out predictions and metrics, so it is **outside the proposed pilot protocol**. Keep all smoke outputs labeled `DIAGNOSTIC_ONLY` and separate from confirmatory OOF outputs.

The shared-grid exploratory audit derived lambda endpoints from all outer-training outcomes before inner validation. This avoids outer-test leakage but makes the candidate range depend on inner-validation outcomes, so it should not be silently substituted for a strictly nested tuning procedure.

## Required gate before confirmatory C2a
1. Freeze the actual protocol, or explicitly record an approved deviation. Do not rewrite existing frozen C1 or C2b outputs.
2. Choose and implement a leakage-safe interpretation of the preregistered glmnet default-ratio path: construct each inner candidate path from **its own inner-training labels only**, select alpha plus relative lambda index/ratio using inner-fold validation, then derive the final path using full outer-training labels only. Log the actual lambda in every fold and final fit.
3. Confirm how glmnet automatic path truncation and low-class-support fallback are scored; keep every failed fit/retry auditable.
4. Run two spatial designs × five folds × eight taxa × two models; check 37,344 total OOF prediction rows before primary inference.
5. Only then calculate paired taxon-fold AUC, 2,000 taxon-cluster bootstrap draws (seed 622), and SESOI decision gate.

**Status:** diagnostic engineering evidence only; confirmatory C2a not started.
