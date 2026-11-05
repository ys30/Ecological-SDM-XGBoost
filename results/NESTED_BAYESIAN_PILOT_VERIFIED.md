# Verified nested Bayesian pilot: 10 km, outer fold 3

**CI:** [#42124372697](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/42124372697) SUCCESS; aggregate artifact **13460529481**. Five hierarchical quadratic `brms` fits (four inner + one outer) passed script-enforced Rhat <=1.01, bulk/tail ESS >=400 and zero divergent transitions. Full sampler diagnostics were not exported as an artifact; CI logs show each fit passed. The comparison is **one outer fold only**.

## At inner target FPR 2%

| Taxon | Model | TP | FP | FN | TN | Recall | Outer FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| nsw19 | GLM | 2 | 68 | 0 | 158 | 100% | 30.09% |
| nsw19 | Bayesian | 2 | 65 | 0 | 161 | 100% | 28.76% |
| nsw20 | GLM | 0 | 6 | 2 | 220 | 0% | 2.65% |
| nsw20 | Bayesian | 1 | 5 | 1 | 221 | 50% | 2.21% |
| nsw22 | GLM | 1 | 34 | 0 | 193 | 100% | 14.98% |
| nsw22 | Bayesian | 1 | 47 | 0 | 180 | 100% | 20.70% |

Bayesian cutoffs at 2%: nsw19 0.013182922; nsw20 0.081454257; nsw22 0.013245183. Bayesian at 1%: nsw19 TP1 FP54; nsw20 TP1 FP2; nsw22 TP1 FP36. At 5%: nsw19 TP2 FP100; nsw20 TP2 FP19; nsw22 TP1 FP80.

## Interpretation

The Bayesian model marginally reduces false positives for nsw19 and nsw20 in this one fold, but increases them for nsw22. For nsw19, Bayesian FPR remains nearly 29% against the 2% inner target. Therefore partial pooling alone does not guarantee threshold transport. This is not evidence of overall superiority or inferiority; other folds and buffer radii are untested for nested Bayesian. Positive counts are only 2, 2, and 1 in this outer fold. No external data, formal CRS certification, statistical significance, or causal explanation is claimed.

**Reproducibility:** [pilot assignments](../scripts/87_nested_bayesian_pilot_assignments.py), [Bayesian pilot](../scripts/88_nested_bayesian_pilot.R), [pre-registered protocol](NESTED_BAYESIAN_PILOT_PROTOCOL.md).
