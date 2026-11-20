# v6.23 data and code availability — manuscript draft

**Status:** Conditional publication wording. Verify data licensing and decide on a citable immutable release before pasting into a journal submission.

## Data availability (draft)

> The analysis uses a verified subset of presence–absence records and environmental predictors from the New South Wales (NSW) survey distributed through the `disdat` R package and documented by Valavi et al. (2023). Identity with the source records was checked against 1,167 geographic sites, six shared environmental covariates and 9,336 binary site–taxon outcomes. The current manuscript evaluates eight anonymized taxa. The original data provider, dataset access conditions, source citation and any redistribution restrictions should be consulted before reuse. We will provide the exact data-access instructions and source version in the final manuscript. **Do not claim a public archival dataset DOI until an archive has been created and verified.**

## Code availability (draft)

> Reproducibility code for the matched five-fold spatially buffered comparison is maintained at https://github.com/ys30/Ecological-SDM-XGBoost (development branch `research/v6.22-nested-spatial-tuning`). The workflow `.github/workflows/v623-matched-three-model.yml` fits fixed-configuration XGBoost and linear/quadratic logistic GLMs, validates matched training and held-out observation membership, and generates model scores and figures. The spatial fold allocation can be replayed independently using `scripts/120_v623_replay_spatial_folds.R`. A stable release tag and permanent archive identifier will be supplied for the final published version; the development branch is not an immutable citation.

## Reproducibility evidence

| Evidence | Run |
|---|---|
| Source record identity | https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532 |
| R spatial fold replay | https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44681382736 |
| Three-model numerical consistency | https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44763794431 |
| Figure technical QC | https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44770268244 |
| Supplement artifact inventory | https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44780027918 (check status before citing as passed) |

## Release and licensing checks

1. Confirm original NSW data license and whether this repository may redistribute the exact rows, coordinates and predictors.
2. Confirm permission to publish anonymized taxa and location data; do not disclose original species identities without authorization.
3. Cite the primary field survey, Valavi et al. (2023) and `disdat` version/source appropriately after verifying metadata.
4. Record exact package versions, git commit, data/fold SHA-256 hashes and CI artifact checksums.
5. Create a versioned GitHub release and archival deposit only after licensing review. Cite a real DOI, not a placeholder.
6. Keep manuscript claims limited to within-dataset spatial validation, fixed model configurations and descriptive contrasts.

**Do not publicly upload a data-bearing Zenodo bundle until source reuse rights are confirmed.**
