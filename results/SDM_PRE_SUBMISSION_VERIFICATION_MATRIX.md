# Pre-submission verification matrix — integrated SDM manuscript

**Status:** Evidence-linked editorial audit, not journal submission clearance. The manuscript is at [INTEGRATED_SDM_MANUSCRIPT_DRAFT.md](INTEGRATED_SDM_MANUSCRIPT_DRAFT.md).

| Manuscript claim | Source-of-truth | Status | Required action |
|---|---|---|---|
| 9,336 records, 1,167 plots, eight species | `data/train.csv`; `results/species_spatial_fold_presence_counts.csv` | Recorded and cross-checked in prior audits | Verify study population and original collection metadata |
| Five two-dimensional geographic folds | `results/spatial_2d_block_assignments.csv`; `scripts/25_audit_species_spatial_folds.py` | Fold presence audit reproducible | Document block construction and coordinate reference system |
| Five eligible species and 23 species–fold comparisons | `results/matched_bayesian_maxent_xgboost_folds.csv` | Matched summary published | Audit per-record identity and background treatment across algorithms |
| Three-model AUC/AP rankings | `results/matched_bayesian_maxent_xgboost_species.csv` | Descriptive results published | Do not claim inferential significance |
| Bayesian formula, prior, fold-specific scaling | `scripts/15_bayes_hierarchical_spatial.R` | Directly checked against script | Record brms/Stan package versions and computational environment |
| Bayesian confirmation diagnostics | `results/BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md` | Historical long-chain confirmation documented | Do not substitute confirmed fold metrics for original per-record predictions |
| Rare-species probability scoring | `results/BAYESIAN_RARE_SPECIES_BASELINE_AND_RETRIEVAL.md` | Original-fit Bayesian only | Clarify that this is not matched MaxEnt/XGBoost comparison |
| Fold-local Top-k and random expectation | `results/bayesian_rare_species_within_fold_retrieval.csv`; `results/bayesian_rare_species_random_baseline.csv` | Independently recomputed; no row mismatches | Maintain 5k-plot budget and non-inferential interpretation |
| Supplementary figure S1 | `results/figures/24_rare_species_retrieval_publication.svg` | 18 SVG data bars independently matched CSV | Check visual readability and accessibility |
| Current CI success | `.github/workflows/rare-species-end-to-end.yml` | **Not independently confirmed** | Inspect an actual successful GitHub Actions run before claiming pass |
| Original study area and environmental predictor provenance | Dataset documentation not yet verified | **Open** | Confirm New South Wales extent, survey source, years, licenses, rainfall/temperature/soil/topography sources |
| XGBoost hyperparameters and MaxEnt background strategy | `scripts/11_spatial_2d_block_cv.R`; `scripts/13_maxent_spatial_2d_cv.R`; `scripts/14_matched_maxent_xgboost.R` | **Verified against committed scripts** | XGBoost fixed 300 rounds, eta .03, depth 4, subsample/colsample .8; MaxEnt `lq`, regmult 2, surveyed non-detection background. Record dependency versions |
| References, author declarations and target journal | Author-provided metadata | **Open** | Verify and insert literature citations, authorship, funding, ethics, data/code availability and journal format |

## Scientific interpretation safeguards

1. AUC and AP values for the five-species matched benchmark are **means of eligible species–fold metrics**; do not equate them with pooled all-species Bayesian fold metrics.
2. Zero-positive folds have undefined ROC AUC, but can still contribute Brier/log loss and survey budget.
3. Pooled Top 25 uses **25 plots total**; within-fold Top 25 uses **125 plots total** across five folds.
4. A random-selection expected count is not a p-value, uncertainty interval or independent external validation.
5. No verified current CI result is claimed in this audit.

## Recommended manuscript gate

The integrated manuscript is structurally complete (Abstract–Conclusions and supplementary references) but should remain labeled **draft** until the open provenance, exact algorithm configuration, literature and author declaration items above are resolved. Publication claims must not be strengthened solely because scripts or CI workflows exist.
