# C1 rare-class feasibility finding (verified from canonical CSVs)

Date: 2026-11-17. Scope: read-only counts from `data/train.csv` and `results/spatial_2d_block_assignments.csv` on the research branch. This is **not** an XGBoost fit, KMeans execution, or CI pass.

All 40 outer training cells (8 taxa x 5 outer folds) contain both classes. However, the rare taxon **nsw22 has only six positive plots overall**; five occur in outer test fold 5, leaving **one positive plot in outer training fold 5** (934 total training plots for that taxon).

Under the currently specified **four nonempty inner validation groups**, the one remaining positive belongs to exactly one inner validation group. In that group's training complement, the positive count is **zero**. Therefore, at least one inner training split is necessarily single-class, independent of KMeans initialization, and the current 4-fold inner XGBoost protocol **cannot complete across all five outer folds**.

Additional high-risk outer training counts:

| Outer fold | Taxon | Train N | Train positives | Test positives |
| --- | --- | ---: | ---: | ---: |
| 1 | nsw19 | 930 | 4 | 4 |
| 1 | nsw20 | 930 | 9 | 9 |
| 1 | nsw22 | 930 | 6 | 0 |
| 2 | nsw19 | 933 | 7 | 1 |
| 2 | nsw22 | 933 | 5 | 1 |
| 3 | nsw19 | 933 | 5 | 3 |
| 3 | nsw20 | 933 | 9 | 9 |
| 3 | nsw22 | 933 | 6 | 0 |
| 4 | nsw19 | 938 | 8 | 0 |
| 4 | nsw22 | 938 | 6 | 0 |
| 5 | nsw19 | 934 | 8 | 0 |
| 5 | nsw22 | 934 | **1** | **5** |

## Protocol decision required before five-fold tuning

Do **not** silently omit failing inner splits, alter the rare taxon's labels, or claim five-fold nested tuning succeeded. Options for a separately labeled design amendment include: (A) replace inner four-group CV with a predeclared alternative capable of fitting rare classes, such as spatially structured repeated holdouts subject to a minimum training-positive constraint, explicitly acknowledging the selection/coverage limitations; (B) use a documented fixed hyperparameter configuration for the infeasible outer/taxon cell, treating it as an exception rather than a tuned model; or (C) restrict the primary tuned comparison to feasible taxa, with nsw22 as a clearly separate sensitivity analysis. None is adopted automatically.

The current outer-fold-1 pilot may still be feasible; its KMeans inner assignments require actual execution. Preserve frozen historical results and do not use these counts as model performance evidence.
