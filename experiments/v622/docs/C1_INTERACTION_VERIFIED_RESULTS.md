# EMS v6.22 C1 — verified results and interpretation (2026-11-11)

**Status:** descriptive analysis complete; NOT a confirmatory interaction claim.
**Source:** GitHub Actions [42965741281](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42965741281), paired fixed-versus-tuned cells [42964164261](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42964164261), fixed OOF [42962847733](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42962847733).

## Main interaction

For metric M, interaction I = (M_species,tuned - M_species,fixed) - (M_pooled,tuned - M_pooled,fixed).
A positive AUC interaction, or negative log-loss/Brier interaction, favors a larger tuning benefit in per-species models.

| Metric | grid4 mean [95% taxon-bootstrap interval] | km15 mean [95% taxon-bootstrap interval] |
|---|---|---|
| AUC | +0.005923 [−0.047252, +0.043001] | +0.017427 [−0.033949, +0.059565] |
| Log loss | −0.018115 [−0.030568, −0.005479] | −0.023495 [−0.039065, −0.007780] |
| Brier | −0.005411 [−0.008833, −0.002047] | −0.005063 [−0.009536, −0.001380] |

Bootstrap resampled the same eight taxa in each of 2,000 replicates, retaining their available outer-fold cells and pairing species and pooled within each taxon-fold. The two spatial designs are **not independent replications**, since they reuse the same plots with different fold boundaries.

## Manuscript-ready Results paragraph (English)

Across both prespecified spatial partitions, hyperparameter tuning showed a larger improvement in proper scoring rules for taxon-specific than for pooled XGBoost models. The difference-in-differences in log loss (taxon-specific tuning effect minus pooled tuning effect) was −0.0181 (taxon-cluster bootstrap 95% interval −0.0306 to −0.0055) under grid4 and −0.0235 (−0.0391 to −0.0078) under km15. The corresponding Brier-score interactions were −0.00541 (−0.00883 to −0.00205) and −0.00506 (−0.00954 to −0.00138). In contrast, the AUC interaction intervals included zero under both partitions. These results suggest that the value of tuning may depend on the model's degree of cross-taxon sharing, particularly for probability accuracy, but are descriptive rather than confirmatory.

## Essential caveats before journal submission

1. Only eight taxon clusters; percentile bootstrap intervals can be unstable and do not address all spatial dependence.
2. No multiplicity adjustment; avoid unqualified claims of statistical significance.
3. The fixed arms are Python XGBoost reimplementations of historical R xgboost settings. Match software versions, random seeds, defaults, and predictions or explicitly label a reimplementation sensitivity.
4. AUC is evaluated only on evaluable matched nonfallback taxon-fold cells (grid4 28; km15 31); log loss and Brier include fallback predictions in all 40 taxon-fold cells per design. Thus the metric-specific estimands differ.
5. The current interaction is calculated as the unweighted mean of per-cell metric differences, not the record-weighted pooled loss; disclose both if later added.
6. Reusing the same taxon resamples across designs does not make the designs independent. Do not combine their p-values as if independent.
7. Check the locked preregistration for a declared interaction test and primary endpoint before elevating this contrast from exploratory to primary.

## Next analysis gates

- Compare fallback-excluded versus all-record proper scores; report both.
- Add leave-one-taxon-out interaction sensitivity, given n=8 taxa.
- Verify original R fixed-model equivalence (or state that it was not achieved).
- Reconcile preregistered primary endpoints and multiple comparisons; update full manuscript and figures only after this audit.

## Verified nonfallback and LOTO sensitivities (2026-11-11)

Nonfallback run [43050953967](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43050953967): retain only OOF rows for which all four arms avoided fallback. grid4 retained 8,866/9,336 records (38 taxon-fold cells); km15 retained 8,639/9,336 (37 cells).

| Nonfallback interaction | grid4 mean [95% CI] | km15 mean [95% CI] |
|---|---|---|
| Log loss | −0.018200 [−0.030736, −0.004413] | −0.024566 [−0.040621, −0.007265] |
| Brier | −0.005687 [−0.009006, −0.002272] | −0.005453 [−0.009883, −0.001520] |

LOTO run [42967072873](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42967072873): log-loss interaction ranges grid4 [−0.021143, −0.015174], km15 [−0.026785, −0.019331]; Brier ranges grid4 [−0.006191, −0.004254], km15 [−0.005785, −0.003255]. All 16 leave-one-taxon-out values per proper-score metric remain negative. AUC interaction signs are **not** stable under leave-one-taxon-out.

## Explicit preregistration reconciliation

Read `experiments/v622/docs/EMS_v6.22_analysis_plan.md`: **the prespecified primary evaluation is paired ROC AUC**, with log loss and Brier **secondary**. Its supported-effect gate requires same AUC sign in both designs, |mean AUC effect| >= 0.01 in both designs, and a 95% interval excluding zero in at least one. The tuning-by-sharing difference-in-differences in this note is an **exploratory post-freeze interaction**, not an automatically preregistered primary contrast. Its AUC intervals include zero in both designs, and grid4 mean +0.0059 falls below 0.01. Therefore **the preregistered AUC support gate is not met**; robust secondary proper-score patterns must not be described as confirmation of the primary hypothesis. Multiple comparisons remain unadjusted.

Suggested manuscript wording: “Exploratory interaction analyses indicated more favorable changes in log loss and Brier score following tuning of taxon-specific versus pooled XGBoost models; this pattern persisted after exclusion of fallback predictions and in leave-one-taxon-out analyses. However, the prespecified primary AUC-based support criterion was not satisfied, and the interaction findings should be interpreted as secondary and hypothesis-generating.”


## R fixed baseline implementation sensitivity (2026-11-11)

Actions run **43108252039** succeeded (artifact **13894209261**). This *grid4-only*, post-freeze exploratory comparison substitutes actual R `xgb.train` fixed predictions with explicitly specified `tree_method="hist"` for the Python fixed baseline, while keeping tuned Python predictions unchanged. It does **not** reproduce the historical unspecified-tree-method R runtime exactly.

The four-arm common nonfallback population contains **8,866/9,336 observations**, 38 taxon-fold cells; AUC is evaluable in 28 cells. Both baselines are scored on the **same retained records**. The interaction is (species tuned − species fixed) − (pooled tuned − pooled fixed), with 2,000 paired taxon bootstrap resamples (seed 622).

| Metric | Python fixed mean [95% CI] | R-hist fixed mean [95% CI] |
|---|---|---|
| AUC | +0.005923 [−0.047252, +0.043001] | −0.004352 [−0.081923, +0.036301] |
| Log loss | −0.018200 [−0.030736, −0.004413] | −0.019375 [−0.031705, −0.005729] |
| Brier | −0.005687 [−0.009006, −0.002272] | −0.005992 [−0.009221, −0.002669] |

**Interpretation:** The direction and bootstrap interval exclusion of zero for *secondary* proper-score interactions persist when using R-hist fixed controls. AUC interaction changes sign and both intervals cross zero; **no primary AUC support claim is justified**. This analysis is not a new independent spatial replication (grid4 only), and its exploratory status, eight-taxon uncertainty, multiple testing, and R/Python numerical non-equivalence must be disclosed. The historical R default tree method is not established by this experiment.

Suggested manuscript wording: “In a grid4 sensitivity analysis using fixed controls refitted with the R XGBoost interface, the exploratory tuning-by-sharing interactions remained negative for log loss and Brier score, whereas the AUC interaction changed sign and remained statistically uncertain. This strengthens the interpretation of the probability-score pattern as a secondary, implementation-robust observation, without satisfying the prespecified primary AUC criterion.”


## Publication figure generated (2026-11-11)

CI run **43123540439** completed successfully and uploaded artifact **13900989599**, `ems-v622-c1-publication-figure-43123540439`, containing vector PDF and 220-dpi PNG of the C1 interaction forest plot. The figure shows grid4 and km15 exploratory Python-baseline effects and grid4 R-hist fixed-baseline sensitivity (the latter is **not** an independent third spatial validation). The source is `experiments/v622/plot_c1_interaction_implementation.py`.

**Proposed caption — Figure C1:** “Exploratory tuning-by-model-sharing interaction in spatially blocked outer-fold predictions. Points indicate the mean taxon-fold interaction, defined as (taxon-specific tuned − taxon-specific fixed) − (pooled tuned − pooled fixed), and whiskers indicate 95% intervals from 2,000 taxon-cluster bootstrap resamples. The two Python-fixed estimates use grid4 and km15 spatial partitions of the same observations; the R-hist fixed estimate is a grid4 implementation sensitivity, not a third spatial replicate. AUC was the preregistered primary endpoint, and its support criterion was not met. Log loss and Brier are exploratory secondary outcomes. AUC excludes non-evaluable taxon-fold cells; the plotted grid4 R-hist comparison uses shared nonfallback records, whereas the original Python estimates use their respective analysis populations. Cross-row numerical contrasts should therefore be interpreted with care.”

**Presentation check:** The CI confirms files were created, not a manual visual inspection of the rendered plot. Before submission, verify legibility, axis scales, and exact population definitions against the JSON artifacts.
