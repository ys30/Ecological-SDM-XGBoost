# C1 pilot versus publication protocol — method audit (2026-11-14)

## Execution status
- Pilot code and push-triggered workflow are committed.
- **No successful GitHub Actions run, logs, or artifact verified in this audit.** Do not claim training complete.
- Existing pilot is an engineering smoke test, not the publication-ready five-fold nested experiment.

## Critical differences requiring correction
1. **Hyperparameter objective:** `96_v622_c1_nested_xgboost_pilot.py` selects the best candidate *separately for each taxon*. The registered manuscript target is a **single candidate chosen using equal-taxon mean relative log-loss improvement**. For formal C1, aggregate each candidate across valid inner folds within each taxon, then equally across all eight taxa, before choosing one global candidate per outer fold.
2. **Spatial protocol:** pilot uses KMeans on longitude/latitude within outer training plots; there is no 10-km exclusion buffer and no CRS certification. It cannot be compared directly to buffered Bayesian/GLM results.
3. **Scope:** one outer fold only; full OOF requires all five outer folds with identical prediction keys and held-out observations.
4. **Rare classes:** fail-closed diagnostics are archived before fitting; a single-class inner training split must be handled through a *predeclared* spatial split strategy, not by silently changing labels, dropping taxa or choosing a favorable split.
5. **Hyperparameter matrix:** 30 configurations vary depth, learning rate and tree count, while min_child_weight, subsample and colsample_bytree are fixed. Report the actual search space; do not imply a broader optimization.
6. **Reproducibility:** dependency ranges are not exact pins. Archive resolved versions, Git SHA, hashes, seeds and fold memberships.
7. **CI evidence:** GitHub workflow configuration is not proof of a run. Require run URL, job conclusion, logs and OOF artifact.

## Acceptance sequence
- Stage A: canonical input gate PASS.
- Stage B: single-fold pilot artifact and class diagnostics verified.
- Stage C: globally pooled equal-taxon tuning and matched spatial protocol implemented.
- Stage D: five-fold OOF complete, all observation keys unique, metric eligibility explicit.
- Stage E: matched GLM/Bayesian/XGBoost evaluation on the **same** outer-fold units and features, with algorithm-only claims restricted to matched inputs.
- Stage F: update manuscript only from verified results.

**Stop:** do not promote pilot estimates to manuscript or label v6.22 C1 complete before stages A–E pass.
