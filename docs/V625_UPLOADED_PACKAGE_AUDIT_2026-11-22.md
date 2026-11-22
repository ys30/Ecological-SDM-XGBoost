# v6.25 uploaded-package review — 2026-11-22

**Status: HOLD for publication / not yet a verified reproduction release.** This review concerns the author-provided files `EMS_v625_Manuscript_Tracked.docx`, `EMS_v625_SI_Addendum.docx`, and `EMS_v625_code_results.zip`. The ZIP was locally tested for corruption (89 members; archive test passed). **The ZIP's scripts and result binaries have not been pushed to this branch by this audit.** Do not claim otherwise.

## Findings

1. The manuscript is a tracked-changes document, not a clean submission version. Accepting the changes resolves visible concatenations and duplicated old/new text, but the accepted version still needs final copyediting and author sign-off.
2. The manuscript's new §2.8/§3.6 describes exploratory nested tuning on grid4, km15, and grid4 + 10 km, not the original unfrozen v6.22 design. Supplement S10 explicitly documents changes to the search space, inner folds, seeds and selection metric. This distinction must be retained.
3. The expanded search uses 180 combinations (5 depths × 3 rates × 2 child weights × 6 rounds), with a separate extended pooled search. The main fixed-setting 23-scheme analysis is distinct from these three nested-tuning designs.
4. The supplied `results/v625/table_paired_mixed_models.csv` contains 56 contrast/metric rows, `table_tuned_summary.csv` 28 rows, and `paired_cells.csv` 674 rows. These files were inspected for shape, **not** independently reproduced from the raw observations and code.
5. Important statistical caveat: some mixed-model fits are marked `singular=True` in the supplied results, and only five or six taxa contribute to the reported cell-level AUC contrasts. Treat Wald intervals as approximate; the taxon-cluster bootstrap is a sensitivity analysis, not independent spatial replication.
6. The supplementary table shows tuned per-taxon XGBoost vs quadratic GLM mean AUC contrasts of approximately -0.020 (grid4), -0.034 (grid4+10 km), and -0.026 (km15). These are observational summaries of the supplied result files, not fresh CI validation.
7. The ZIP contains individual held-out predictions and archived brms prediction data. Public redistribution requires checking the underlying dataset's reuse/privacy terms and the intended repository disclosure policy.
8. The main manuscript still contains a placeholder for the v6.25 results commit SHA; do not fill it with the audit-only commit.
9. The uploaded ZIP README refers to an `ems-revision` branch, which is not among the two current branches exposed by the GitHub branch listing. Its source history must be reconciled before merging.

## Release gate

- [ ] Author approves accepted-changes clean manuscript and verifies figures/tables.
- [ ] Push complete source scripts and an appropriately licensed, provenance-documented results subset.
- [ ] Verify all v6.25 numerical outputs by a fresh CI run from pinned dependencies and input checksums.
- [ ] Verify source-data license, privacy and archived Bayesian prediction redistribution rights.
- [ ] Add immutable code/results commit SHA and manuscript/SI cross-references.
- [ ] Confirm journal-specific formatting and final declarations.

**This document is a review log only. It does not assert successful upload or reproduction of v6.25 experiment outputs.**
