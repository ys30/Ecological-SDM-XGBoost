# Audit of matched three-model AUC and AP SVG figures

**Source of truth:** `results/matched_bayesian_maxent_xgboost_species.csv` and `results/matched_three_model_plot_data.csv`.

The two published charts `results/figures/22_three_model_auc.svg` and `results/figures/22_three_model_ap.svg` use a shared 0–1 horizontal scale, five eligible species and three model bars per species. Their printed three-decimal AUC and AP values are consistent with the corresponding five-species source table (rounding to three decimals). There are 23 eligible species–fold comparisons, but each figure plots **15 species–model mean values**, not 23 fold observations.

**Critical interpretation:** bars represent unweighted species-specific means across eligible folds, not pooled predictions, confidence intervals or inferential uncertainty. The fold count varies (five folds for four species; three for *nsw21*). These graphics describe the **original shorter-chain Bayesian matched comparison**, not the confirmed long-chain predictions. Do not add error bars without a valid dependence-aware variance estimator.

**Suggested publication caption (AUC):** Mean held-out ROC AUC for Bayesian hierarchical, MaxEnt and XGBoost models evaluated on identical spatial-fold observations within five eligible species. Bars represent unweighted means of species–fold AUCs (23 eligible combinations across five species). Results are descriptive; no statistical significance is implied. Bayesian values use original-fit posterior predictions.

**Suggested publication caption (AP):** Mean held-out average precision for the same matched species–fold comparisons. Species differ in observed prevalence, and AP should be interpreted in relation to class imbalance. Bars are unweighted fold means, not confidence intervals; Bayesian values use original-fit posterior predictions.

**Improvement priority:** move fold counts and positive-case counts into a companion table; ensure species names are italicized in the final journal artwork; preserve an explicit legend and axis label; and keep all model colors consistent between AUC and AP. Rebuild figures from the committed CSV rather than manually editing bar geometry if revisions are required.


## Publication figure update (2026-11-01)

New source-data-driven SVGs are available at [AUC](figures/23_three_model_auc_publication.svg) and [AP](figures/23_three_model_ap_publication.svg). Both use the committed species-level means, annotate the eligible fold count and positive cases, and retain the same model palette. The original figures are preserved. An [R plotting implementation](../scripts/23_three_model_publication_figures.R) is committed, but has **not** been executed in a verified R environment; the published SVGs were constructed directly from the source CSV rather than exported by that R script. Consequently pixel-for-pixel reproducibility of the SVGs from R is not yet established. The source-data values can be checked against the CSV independently.
