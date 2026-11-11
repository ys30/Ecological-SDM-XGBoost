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
