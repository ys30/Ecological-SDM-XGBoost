# Pre-submission verification matrix — integrated SDM manuscript

**Status:** Evidence-linked editorial audit, not journal submission clearance. The manuscript is at [INTEGRATED_SDM_MANUSCRIPT_DRAFT.md](INTEGRATED_SDM_MANUSCRIPT_DRAFT.md).

| Manuscript claim | Source-of-truth | Status | Required action |
|---|---|---|---|
| 9,336 records, 1,167 plots, eight species | `data/train.csv`; `results/species_spatial_fold_presence_counts.csv` | Recorded and cross-checked in prior audits | Verify study population and original collection metadata |
| Five two-dimensional geographic folds | `results/spatial_2d_block_assignments.csv`; `scripts/25_audit_species_spatial_folds.py` | Fold presence audit reproducible | Document block construction and coordinate reference system |
| Five eligible species and 23 species–fold comparisons | `results/matched_bayesian_maxent_xgboost_folds.csv` | Matched summary published | Audit per-record identity and background treatment across algorithms |
| Three-model AUC/AP rankings | `results/matched_bayesian_maxent_xgboost_species.csv`; `scripts/37_validate_three_model_matched_metrics.py`; [successful CI run 41288740752](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41288740752) | **Verified**: 138 AUC/AP values independently recomputed from held-out predictions across 23 matched species–fold cells | Retain descriptive interpretation; this is not model refitting or inferential significance |
| Bayesian formula, prior, fold-specific scaling | `scripts/15_bayes_hierarchical_spatial.R` | Directly checked against script | Record brms/Stan package versions and computational environment |
| Bayesian confirmation diagnostics | `results/BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md` | Historical long-chain confirmation documented | Do not substitute confirmed fold metrics for original per-record predictions |
| Rare-species probability scoring | `results/BAYESIAN_RARE_SPECIES_BASELINE_AND_RETRIEVAL.md` | Original-fit Bayesian only | Clarify that this is not matched MaxEnt/XGBoost comparison |
| Fold-local Top-k and random expectation | `results/bayesian_rare_species_within_fold_retrieval.csv`; `results/bayesian_rare_species_random_baseline.csv` | Independently recomputed; no row mismatches | Maintain 5k-plot budget and non-inferential interpretation |
| Publication three-model figure source values | `results/matched_three_model_plot_data.csv`; `results/matched_bayesian_maxent_xgboost_species.csv`; `scripts/38_validate_three_model_figure_data.py` | **Direct repository-content audit passed (2026-11-01):** 15 plot rows, 30/30 AUC/AP values identical within 1e-12, 5 species × 3 models, no missing/duplicate combinations; source blob SHAs `391dbf34565f52c1fd31559f4bc669bd5515e2ae` and `3a98f3157227e6a9c1b9e5ab7fa521b9a13718e` | Confirm the separate GitHub Actions workflow run and SVG rendering/labels; direct audit does not establish CI success |
| Supplementary figure S1 | `results/figures/24_rare_species_retrieval_publication.svg` | 18 SVG data bars independently matched CSV | Check visual readability and accessibility |
| Matched three-model metric CI | `.github/workflows/validate-three-model-matched-metrics.yml`; [run 41288740752](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41288740752) | **Success confirmed**: `validate` job and 138-metric recalculation step passed | Separately confirm other training/figure workflows; no complete model rerun claimed |
| Original study area and environmental predictor provenance | `results/NSW_DISDAT_EXECUTED_PROVENANCE_AUDIT.md`; [successful site/label identity audit](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532) | **Source identity verified**: 1,167 coordinate matches, six covariates, unique species mapping, zero discrepancies across 9,336 labels | Confirm survey dates, license, original environmental-layer sources |
| XGBoost hyperparameters and MaxEnt background strategy | `scripts/11_spatial_2d_block_cv.R`; `scripts/13_maxent_spatial_2d_cv.R`; `scripts/14_matched_maxent_xgboost.R` | **Verified against committed scripts** | XGBoost fixed 300 rounds, eta .03, depth 4, subsample/colsample .8; MaxEnt `lq`, regmult 2, surveyed non-detection background. Record dependency versions |
| References, author declarations and target journal | Author-provided metadata | **Open** | Verify and insert literature citations, authorship, funding, ethics, data/code availability and journal format |

## Scientific interpretation safeguards

1. AUC and AP values for the five-species matched benchmark are **means of eligible species–fold metrics**; do not equate them with pooled all-species Bayesian fold metrics.
2. Zero-positive folds have undefined ROC AUC, but can still contribute Brier/log loss and survey budget.
3. Pooled Top 25 uses **25 plots total**; within-fold Top 25 uses **125 plots total** across five folds.
4. A random-selection expected count is not a p-value, uncertainty interval or independent external validation.
5. The NSW provenance and site/label identity workflows have verified successful CI runs; this does **not** verify all model-training or figure-generation workflows.

## Recommended manuscript gate

The integrated manuscript is structurally complete (Abstract–Conclusions and supplementary references) but should remain labeled **draft** until remaining source-license and survey-date metadata, computational environment, and author declaration items above are resolved. Publication claims must not be strengthened solely because scripts or CI workflows exist.

## Critical dataset-split provenance risk

The `disdat` reference manual describes NSW `disPa('NSW','ot')` and `disEnv('NSW','ot')` as **evaluation (presence–absence) survey records**, separate from presence-only/background model-training records (`disPo` and `disBg`). The present repository has 1,167 plots × eight species with binary labels, resembling the evaluation-data structure. If exact record identity is confirmed, the manuscript must disclose that its spatial CV is an **internal reanalysis of the published evaluation survey**, not an untouched evaluation of models trained on the package's prescribed presence-only/background training split. This is not necessarily invalid for a self-contained comparative study, but it changes the scientific question and limits claims of external validation. The source identity was subsequently **confirmed** by the executed [site-and-label audit](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532): all 1,167 local plots matched geographically and all 9,336 binary labels agreed under a unique eight-species code mapping. The manuscript now explicitly describes internal spatial CV reanalysis of the published evaluation survey. Reference: [disdat package manual](https://cran.r-universe.dev/disdat/doc/manual.html); [Elith et al. (2020), DOI:10.17161/bi.v15i2.13384](https://doi.org/10.17161/bi.v15i2.13384).

## Direct SVG numerical audit (2026-11-01)

Read the two committed SVG files and `matched_bayesian_maxent_xgboost_species.csv` directly from the repository. All **30 bar widths** matched 650 × their underlying AUC/AP value within 0.012 SVG units, all **30 numeric text labels** matched three-decimal rounding, and all five species labels were present. **60/60 numerical checks passed**, with no mismatches. Source blob SHAs: species CSV `3a98f3157227e6a9c1b9e5ab7fa521b9a13718e6`; AUC SVG `cbc0fbfaac9727023bce1e19fb4f54ca5c65ec2e`; AP SVG `7d383541c0e2359af388893e50b0f9508b0a63e8`. This was an independent read-and-check of committed files, not a confirmed CI run or a visual rendering inspection.
