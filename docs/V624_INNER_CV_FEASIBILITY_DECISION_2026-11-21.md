# v6.24 spatial nested-tuning decision record — 2026-11-21

## Evidence and scope
Repository: `ys30/Ecological-SDM-XGBoost`, branch `research/v6.22-nested-spatial-tuning`. Evidence: [Actions run 44907840896](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44907840896), commit `7163d043b19b749125ed7a1140bcd48abb698358`, job `no-fit-feasibility`. Its uploaded artifact `v624-inner-feasibility` is the machine-readable source.

The no-fit audit traversed five outer geographic folds, eight taxa and 640 outer-fold/taxon/cell combinations. It found 3 cells whose remaining training labels are invalid for binary fitting; 297 validation cells contain both classes. The mathematical impossibility is **outer fold 5 / nsw22**, with **one positive observation in outer training**: an inner validation fold cannot contain a positive and still leave a positive for the inner fit.

These numbers describe *candidate unbuffered 4-by-4 longitude/latitude quantile cells, constructed from outer-training plots only*. They are **not** results of buffered inner spatial cross-validation or hyperparameter tuning. Completion of this diagnostic CI does not authorize the claimed nested study.

## Binding design constraints and deviations
1. **Preserve all five pre-registered outer geographic holdouts**, never tune on outer test labels. Do not change outer folds or pool test labels to manufacture feasible rare-class inner CV.
2. **Do not run per-taxon nested tuning for nsw22 in outer fold 5**: it is mathematically impossible under a positive-present inner-training and positive-present inner-validation requirement, irrespective of chosen spatial partitions or number of folds. This is an unavoidable deviation from the original v6.22 proposed uniform nested tuning.
3. For any other taxon/fold, inner splits must be built exclusively from outer-training plot coordinates, include the intended distance buffer (with coordinate-reference-system and distance-unit verification), and pass a *post-buffer* binary-class fit gate. Select tuning hyperparameters using training-only validation data; never use outer holdout performance for model selection.
4. **Fallback for infeasible taxon/fold:** use a frozen, externally specified baseline hyperparameter configuration without outcome-driven selection; train on available outer-training data when both classes exist. If outer training has just one positive, fitting may be possible but fragile; explicitly document the model's suitability and never call its hyperparameters nested-tuned. If no binary fit is possible, mark that model/fold as unfit and keep the corresponding outer evaluations missing (not zero).
5. **Comparison:** C1–C4 models must use identical plot–taxon outer test records and equal predictor information when claiming an algorithmic contrast. Report the tuned-eligible population separately from the full eight-taxon descriptive cohort; no pooled apparent superiority from comparing different eligible subsets.
6. **Metrics:** Brier/log loss can be evaluated on single-class test folds if probabilities are defined; AUC/AP are undefined for single-class test folds and must be omitted. Report taxon-wise metrics, equal-taxon aggregates, eligible cell counts, and paired differences. The minimum meaningful-effect threshold must be pre-specified *before* examining C1–C4 effect sizes.
7. **Uncertainty:** seed sensitivity and repeated outcome-blind spatial-partition sensitivity are separate analyses. Eight-taxon bootstrap intervals are exploratory; neither geographic cells nor fold–taxon cells are independent population replicates.
8. **Evidence threshold:** manuscript/EMS submission package remains *blocked* until actual post-buffer feasibility, all planned fitting/OOF outputs, C1–C4 matched audits, multi-seed checks, and uncertainty analyses have verified artifacts and reproducible hashes.

## Immediate execution gates
- [x] Establish a versioned no-fit feasibility audit; CI success and artifact preserved.
- [x] Identify the nsw22 / outer-fold-5 impossibility, and prohibit misleading all-taxon nested-tuning claims.
- [ ] Validate source coordinate reference system and a physically meaningful projected distance; then run post-buffer **inner-split** feasibility checks for each candidate taxon/fold.
- [ ] Freeze viable inner split manifest and candidate hyperparameter grid before fits; record excluded cells and fixed-parameter fallback decisions.
- [ ] Run nested fits for feasible cells and pre-fixed baselines for ineligible cells; export aligned OOF predictions and full provenance.
- [ ] Execute C1–C4 comparisons, multiple seeds, spatial uncertainty, threshold sensitivity, and independently verify each artifact.
- [ ] Revise manuscript, SI and reproducibility package, including a table of protocol deviations from v6.22.

**Status:** protocol-amendment decision documented; no claim of completed tuning, C1–C4 computations or submission readiness.
