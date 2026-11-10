# EMS v6.22 execution checkpoint

Status: IMPLEMENTATION PENDING — no v6.22 experimental outcomes claimed.

Base branch commit: 4c8cb67c654c379f0c8768fcbbbd9d4e682a7a3d
Research branch: research/v6.22-nested-spatial-tuning

## Scientific contrasts
- C1: nested tuned versus fixed XGBoost, per-taxon and pooled.
- C2: quadratic versus linear response within matched estimation families.
- C3: partial/full versus no cross-taxon information sharing.
- C4: tuned per-taxon XGBoost, glmnet and MaxEnt with matched four numeric inputs.

## Validation
Primary outer designs: grid4_s2026_b0 and km15_b0.
Inner: 12 geographic k-means clusters and four folds, seed 6220 + outer-fold index.
Never expose outer test observations to feature scaling, tuning, null prevalence or model selection.
Use one shared candidate set of 30 XGBoost configurations, random seed 622.
For each inner validation fold, compute null log loss using *inner training* prevalence for each taxon; average relative log-loss improvement equally across taxa.
Select configurations using only inner validation predictions. Refit on the full outer training data; write one held-out prediction per eligible outer observation.
Prespecify fallback for fewer than five training detections and log every fallback.
Preserve raw prediction rows, plot IDs, taxon, fold, model, probability/score, fallback flag, selected hyperparameters, package versions, seed and runtime.

## Required gates before running
1. Verify dataset SHA-256, plot-level group integrity, CRS and spatial distances.
2. Verify primary fold definitions and absence of outer/inner leakage.
3. Run runtime-only pilot on outer fold 1 of grid4_s2026_b0; do not save pilot predictions or evaluate outer metrics.
4. Freeze plan and configuration before examining new outer results.
5. Run primary comparisons; report paired within-taxon-fold AUC, AP, log loss, Brier and evaluable coverage.
6. Keep pooled cross-taxon AUC descriptive only; label dependent spatial partitions as sensitivity analyses.
7. Record all deviations and failures.

This file is an execution checklist, not evidence of completed experiments.
