# Rare-species retrieval reproducibility audit

## Source and evaluation scope

- Input: `results/bayes_spatial_cv_predictions.csv` (9,336 original-fit Bayesian held-out prediction records; 1,167 plots per species).
- Rare species: *Corymbia gummifera*, *C. intermedia*, *Eucalyptus carnea*.
- Fold-local rankings are computed separately for each of the five spatial holdout folds, with deterministic probability-descending and numeric-ID tie breaking.
- Fold-local retrieval CSV: `results/bayesian_rare_species_within_fold_retrieval.csv` (15 species–fold rows).
- Stratified random expected retrieval: `results/bayesian_rare_species_random_baseline.csv` (9 species–budget rows).
- Figure: `results/figures/24_rare_species_retrieval_publication.svg` (18 data bars, two per CSV row).

## Independent consistency check

A separate read-back and recomputation directly from the committed prediction CSV found **zero discrepancies** among the 15 fold-level output rows (including all top-10/25/50 captured counts and recall fields). The 18 SVG data-bar heights also matched the nine random-baseline CSV rows, to within the declared numeric rendering tolerance (0.00002 SVG units). This verifies **stored data consistency**, not the latest GitHub Actions run status.

## Expected capture reference

For each species and k in {10, 25, 50}, expected random captures = Σ_f k × P_f / N_f, where P_f is the number of observed presences in fold f and N_f is its number of plots. Recall is actual captured / Σ_f P_f. Enrichment is actual captured / expected random captures. All five folds contribute survey costs, including zero-presence folds. This reference is a descriptive expectation and does not establish statistical significance.

## Automated checks

- `scripts/27_rare_species_within_fold_retrieval.py` and `.github/workflows/validate-rare-species-retrieval.yml`
- `scripts/28_rare_species_random_baseline.py` and `.github/workflows/validate-rare-species-random-baseline.yml`
- `scripts/29_validate_rare_species_retrieval_svg.py` and `.github/workflows/validate-rare-species-figure.yml`

**CI status:** Not verified here. A successful workflow run must be inspected separately before reporting the checks as CI-passed.

## Scientific limitations

These are original shorter-chain Bayesian predictions, not long-chain confirmed per-record predictions. The same folds are used to describe observed out-of-fold retrieval; this is not a prospective survey, external-region validation, matched three-algorithm rare-species comparison, or estimate of uncertainty in future capture efficiency. Small positive counts make fold-specific recall unstable.
