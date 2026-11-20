# v6.23 submission-readiness register — 2026-11-20

**Assessment:** Analytical protocol reproducible and manuscript headline metrics CI-checked; not yet submission-ready.

## Evidence completed

| Gate | Status | Evidence |
|---|---|---|
| Original NSW dataset identity | PASS | [#41287426532](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41287426532), `results/NSW_DISDAT_EXECUTED_PROVENANCE_AUDIT.md` |
| Spatial-fold R replay | PASS | [#44681382736](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44681382736), `scripts/120_v623_replay_spatial_folds.R` |
| Three-model matched 10-km protocol | PASS | [#44763794431](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44763794431) |
| Frozen metric/figure-file consistency | PASS | [#44763794431](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44763794431), `scripts/121_v623_manuscript_consistency_gate.py` |
| Main figure visual QC | OPEN | Inspect actual PNG/SVG plots, not just file presence |
| Original survey dates and sampling design | OPEN | Verify original primary source |
| Predictor units, especially `rainann` | OPEN | Verify source data dictionary |
| Data reuse/redistribution permissions | OPEN | Check original dataset license and public archive contents |
| Upstream leakage audit | OPEN | Inspect label-dependent selection, covariate construction, joins |
| Complete formatted journal manuscript | OPEN | Title, author, abstract, citations, tables, captions, bibliography and supplement |

## Figure and numerical freeze

The latest successful full comparison is [#44763794431](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44763794431). Its uploaded artifact `v623-matched-three-model-comparison` contains the outputs generated in the same run. The manuscript consistency gate passed against `taxon_pooled_oof_scores.csv` and the four main SVG/PNG figure files. This is **not** a claim of visual inspection.

| Equal-taxon mean | XGBoost | Linear GLM | Quadratic GLM |
|---|---:|---:|---:|
| Log loss | 0.186905 | 0.204591 | 0.187524 |
| Brier | 0.055984 | 0.062737 | 0.054642 |

**Recommended next work:** inspect/download the CI artifact for figure-level visual QC, prepare a source-backed data-availability statement, and assemble a complete Word manuscript only after source metadata and licensing are settled.

## Language guardrails

- Do not call the within-dataset spatial folds independent external validation.
- Do not call the fixed-configuration comparison nested hyperparameter tuning.
- Do not describe exploratory sign-flip fractions as confirmatory significance tests.
- Do not claim XGBoost universally outperforms quadratic GLM.
- Do not publish underlying taxonomic names without appropriate authorization.
