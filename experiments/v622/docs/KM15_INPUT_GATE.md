# C1 km15 primary-design input gate (2026-11-10)

The grid4 formal OOF run succeeded (Actions 42955194925), and its descriptive
taxon-cluster bootstrap succeeded (42958175994). Its pooled-minus-species
matched AUC mean was -0.0090659 (28/40 evaluable taxon-fold cells), 95% taxon
bootstrap percentile CI [-0.0366913, 0.0069632]. This is descriptive, not a
two-design C3d decision and not a C1a/C1b fixed-vs-tuned contrast.

## Blocker before km15 formal execution

The preregistration config pins
`results/ems_revision/kmeans15_five_fold_assignments.csv` at SHA-256
`04686634473d51b5c63dcbd4474659fe864a114ae0885c5c5f55736522c798ce`,
but this path is **not present in the current branch Git tree**. A tree
inventory of `ems-v622-experiments` found no `kmeans15`, `km15` or
`fold_assignments` path. Do not fabricate assignments, substitute the grid4
folds, or claim km15 was executed.

## Required resolution

1. Recover the exact pinned km15 assignment file from the original
   archived results / CI artifacts, or reproduce it from its documented
   original generation procedure with verified provenance.
2. Verify the file's SHA-256 against the preregistration, uniqueness of
   `plot`, exact coverage of all 1,167 plots, and five nonempty folds.
3. Inspect the actual column names and map to the outer-runner's
   `plot,fold` interface without changing fold memberships.
4. Commit a **separate, explicitly scoped km15 freeze** before launching
   formal OOF; keep grid4 outputs immutable.

C1a/C1b additionally require fixed-parameter XGBoost arms; tuned pooled vs
tuned species alone does not test hyperparameter optimization.
