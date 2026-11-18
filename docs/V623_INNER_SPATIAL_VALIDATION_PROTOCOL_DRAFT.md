# v6.23 inner spatial validation — preregistration draft (NOT executed)

Status: **PROPOSED / NOT APPROVED / NO MODEL FITS**. Preserves v6.22 outputs and all frozen historical Bayesian/GLM comparisons.

## Evidence requiring change

GitHub Actions run [44217152367](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44217152367) checked 160 inner cells; four had zero training positives. Pilot outer 1 / nsw20 had all nine training positives in a single inner KMeans validation group. Outer 5 / nsw22 has only one training positive, making exhaustive nonempty four-group inner CV mathematically impossible. No v6.22 pilot fits were performed.

## Candidate design (to evaluate without outcome-based partition search)

1. Preserve the existing **five outer spatial folds**, feature set, 30 candidate XGBoost grid, and equal-taxon target. Outer test labels and metrics must never influence tuning or inner partition selection.
2. For taxa with at least two positive outer-training plots, evaluate **predeclared spatially separated holdouts** within outer training, with minimum positive count in training, minimum separation/buffer rules, and fixed geographical seeds. Validation groups need not partition all observations, but document the change from nested KMeans and the geographic coverage of held-out samples. Do not choose partitions based on candidate model performance; do not claim independence unless spatial separation is quantified.
3. **One-positive outer-training taxon (nsw22, outer 5)**: no valid two-class cross-validation can hold out the only positive while retaining a positive for fitting. Predeclare a **fixed-parameter fallback**, selected independently of that taxon's outer-test data and without using its unavailable inner validation; report separately as a protocol exception. A model trained on one positive may be unstable; compare to prevalence-only baseline and present uncertainty prominently.
4. Do not silently drop failed inner folds. If a taxon lacks enough feasible validation replicates, mark its tuning score **not estimable**. For a global equal-taxon selector, require a fully defined rule for unavailable taxon scores **before** training (e.g., exclude that taxon from tuning with an explicitly renamed estimand and apply global parameters afterward). Never call a seven-taxon objective an eight-taxon equal-weight objective.
5. Report per-taxon outer-fold counts, train/validation positives, spatial distance/buffer diagnostics, held-out plot IDs, candidate selection rule, baseline log loss, and eligibility for AUC/AP. AUC/AP is undefined for outer test folds with only one observed class.
6. Keep the original v6.22 design as an audited negative result. Any v6.23 results must live under separate paths and be labeled **not directly comparable** to frozen 10-km-buffered Bayesian/GLM analyses unless matching is established.

## Decision gate before training

Implement a **no-fit feasibility matrix** for each candidate design, including every outer/taxon split, held-out geographic extent and train-positive minimum. Select one design from scientific requirements, **not** from which produces the most favorable model scores. Obtain explicit approval for the exception and changed tuning estimand, then freeze config and run a pilot. Until approved, do not trigger training.

## Immediate work

- Build an outcome-blind geographic coverage and distance diagnostic using existing plot coordinates, with class counts assessed **after** geographic partitions are fixed.
- Audit whether a single fixed geographic scheme can support the intended global selection; do not use labels to move boundaries.
- Preserve CI run evidence and hashes of canonical inputs.
