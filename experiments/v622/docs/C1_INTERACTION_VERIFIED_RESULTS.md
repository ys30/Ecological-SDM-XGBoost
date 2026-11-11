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
