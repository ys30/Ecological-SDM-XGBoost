# C2a diagnostic results — manuscript-ready working draft (NOT confirmatory)

Status: **DIAGNOSTIC_ONLY**. Based on GitHub Actions run [43667019171](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43667019171). This text is a working draft, not approved manuscript claims. The preregistration remains DRAFT and the implemented train-only lambda-path search differs from the preregistered tuning plan.

## Results — spatial validation sensitivity

Across eight taxa, two spatial partition designs (grid4 and km15), five outer folds per design, and two glmnet model specifications (linear and quadratic RT), 160 model arms were fitted. The diagnostic audit yielded 80 paired taxon–design–fold comparisons, of which 59 were AUC-comparable after excluding five pairs with outer fallback and 16 nonfallback pairs with single-class held-out labels. AUC was not imputed for single-class test folds.

At the paired-cell level, quadratic RT improved AUC in 23 of 28 comparable grid4 cells (mean difference +0.0431) but only 15 of 31 comparable km15 cells (mean difference −0.0213). These cell-level means weight taxa in proportion to the number of evaluable folds and must not be interpreted as independent biological replicates.

After averaging evaluable folds within each taxon and then weighting taxa equally, the mean AUC difference (quadratic minus linear) was +0.0313 under grid4 and −0.0219 under km15. The within-taxon paired difference between designs was +0.0532. An exploratory taxon-cluster bootstrap (2,000 resamples, seed 622) produced percentile intervals of [+0.0034, +0.0592] for grid4, [−0.0684, +0.0166] for km15, and [+0.0276, +0.0832] for their paired difference. These are descriptive uncertainty summaries, not preregistered confirmatory confidence intervals.

For taxa with at least three evaluable AUC folds in **both** designs (n=5), the paired mean design difference was +0.0549 (exploratory percentile interval [+0.0248, +0.0982]); requiring all five folds in both designs (n=4) yielded +0.0598 ([+0.0223, +0.1072]). These subset analyses change the target population of taxa and should not be treated as equivalent estimates of the full eight-taxon mean.

## Discussion — cautious interpretation

Quadratic RT's apparent discrimination benefit depended strongly on the spatial validation design. nsw24 had positive AUC differences in all five folds under both designs, whereas nsw25 reversed direction between designs. For rare taxa, several outer folds lacked positive test examples or required model fallback, reducing the number of interpretable AUC comparisons. Log loss and Brier score should be presented separately from AUC because their evaluability and weighting differ.

These patterns motivate a methodological claim about spatial evaluation sensitivity, **not** a universal superiority claim for either model. Possible explanations include altered geographic extrapolation difficulty, different positive-class support, and the interaction between regularization and spatial training partitions; none is established causally by these diagnostics.

## Mandatory gates before manuscript result freeze

1. Reconcile the actual train-only lambda-path procedure with the DRAFT preregistration; either rerun the frozen confirmatory protocol or transparently designate these analyses exploratory.
2. Audit fallback reasons, training-class support, per-fold test prevalence, and regularization path-boundary selections.
3. Report log loss and Brier score by taxon and design, with appropriate valid-denominator rules, in addition to discrimination.
4. Check independence of the eight taxa and whether taxon-cluster resampling is scientifically justified; do not claim nominal frequentist coverage from n=8 alone.
5. Freeze source SHA, fold assignments, code SHA, environment, and artifact manifests before journal submission.

Data products: `all_arm_metrics_DIAGNOSTIC_ONLY.csv`, `paired_cell_metrics_DIAGNOSTIC_ONLY.csv`, `taxon_design_robustness_DIAGNOSTIC_ONLY.csv`, `auc_coverage_sensitivity_DIAGNOSTIC_ONLY.csv`, `taxon_cluster_bootstrap_EXPLORATORY_ONLY.csv`.
