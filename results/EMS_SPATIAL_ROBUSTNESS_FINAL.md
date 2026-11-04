# EMS spatial robustness: verified final audit

This is a public reproducibility and interpretation note, not the private manuscript.

## Paired near/far comparisons

Source: [paired-distance summary](paired_distance_interaction_summary.csv), [paired cell audit](paired_distance_interaction_audit.csv), and [distance-stratified OOF metrics](matched_distance_stratified_oof_metrics.csv). [CI run 41754175515](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41754175515) succeeded.

For each eligible anonymous taxon × spatial fold, the interaction is:
`(model − MaxEnt) at <5 km − (model − MaxEnt) at ≥5 km`.
Only 21 paired cells across five taxa have defined AUC in both distance bands. The exploratory exact two-sided sign-flip test flips the five taxon-mean interaction signs (32 sign configurations); it does **not** assume 21 independent replicates.

| Comparison | Metric | Cell-weighted mean interaction | Equal-taxon mean interaction | Exploratory taxon sign-flip p |
|---|---|---:|---:|---:|
| XGBoost − MaxEnt | AUC | −0.091546 | −0.083496 | 0.1250 |
| Bayesian − MaxEnt | AUC | +0.007685 | +0.016195 | 0.3750 |
| XGBoost − MaxEnt | AP | −0.066834 | −0.088637 | 0.1875 |
| Bayesian − MaxEnt | AP | +0.027342 | +0.038405 | 0.0625 |

**Interpretation:** A descriptive change in XGBoost's relative performance is present, but taxon-level evidence is limited and does not support a definitive spatial interaction claim. The five spatial folds and distance bands are not independent external regions. Restricted subset prevalence and sample size influence AP and AUC; these analyses are not buffered retraining or extrapolation tests.

## Spatial-encoding sensitivity: archived output

[Successful sensitivity CI run 41752359890](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41752359890). Source: [23-cell sensitivity summary](xgboost_matched_23_cell_paired_summary.csv).

| XGBoost feature mode | Eligible cells | Mean ΔAUC vs MaxEnt | Mean ΔAP vs MaxEnt |
|---|---:|---:|---:|
| Common four environmental variables (no taxon identity) | 23 | −0.165167 | −0.103039 |
| Common four environmental variables + taxon identity | 23 | +0.016281 | +0.031655 |
| Fold-local full feature encoding | 23 | +0.012485 | +0.035816 |

These are descriptive matched-cell means. Removing taxon identity changes the prediction task, and these contrasts should not be interpreted as an algorithm-only causal effect. Do not claim these 23 cells are independent replicates.

## Decision

No additional model families are required for the current analysis narrative. Before manuscript finalization: check coordinate CRS and variable provenance, review all figure references, and ensure the private manuscript's Methods specify the common-input protocol, taxon identity, fold eligibility, and limitations. Do not promote exploratory p-values as confirmatory significance.
