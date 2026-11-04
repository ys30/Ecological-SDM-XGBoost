# Verified buffered GLM–Bayesian matched comparison (2026-11-04)

Source: [successful GitHub Actions audit #41917257355](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41917257355), artifact `audited-buffered-model-comparison`; Bayesian fits [#41907588922](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41907588922) (30/30 successful); GLM fits [#41874313829](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41874313829). All metrics below were printed by the successful audit job. Matched OOF: 9,336 records per model/radius, same test records. Within-dataset sensitivity, **not** external validation.

| Buffer (km) | Variant | Model | AUC | AP | Brier | Log loss |
|---:|---|---|---:|---:|---:|---:|
| 5 | linear | GLM | .889448 | .342662 | .062220 | .203370 |
| 5 | linear | Bayesian | .885958 | .336785 | .062156 | .203709 |
| 5 | quadratic | GLM | .902657 | .432188 | .056115 | .195095 |
| 5 | quadratic | Bayesian | .900844 | .410856 | .055640 | .186285 |
| 10 | linear | GLM | .886364 | .338635 | .062702 | .206564 |
| 10 | linear | Bayesian | .884745 | .336281 | .062573 | .205666 |
| 10 | quadratic | GLM | .899413 | .423774 | .056561 | .197525 |
| 10 | quadratic | Bayesian | .899297 | .408757 | .055940 | .187725 |
| 20 | linear | GLM | .877064 | .328384 | .063701 | .216353 |
| 20 | linear | Bayesian | .874214 | .328796 | .063242 | .212655 |
| 20 | quadratic | GLM | .891263 | .415261 | .056655 | .209761 |
| 20 | quadratic | Bayesian | .888301 | .408172 | .055899 | .197969 |

### Interpretation
* Quadratic GLM has the higher pooled AUC and AP at each radius; quadratic hierarchical Bayesian has lower Brier and log loss at each radius. The metrics reward different properties; avoid declaring a universal winner.
* Relative to 5 km, 20 km pooled AUC decreases for both quadratic GLM (.902657 → .891263) and Bayesian (.900844 → .888301). AP declines for GLM (.432188 → .415261) but Bayesian AP is relatively stable (.410856 → .408172). **No statistical significance is claimed**.
* More aggressive buffers remove additional training plots, so changes are not solely attributable to reduced spatial leakage.
* Projected easting/northing coordinate units are **assumed to be meters**, not yet independently CRS-verified. Do not make strong physical-distance claims until CRS is checked.
* Mean paired GLM-minus-Bayesian taxon×fold quadratic differences (34 eligible AUC/AP cells; 40 Brier/log-loss cells): at 5/10/20 km respectively: AUC +.007695/+ .007101/+ .016005; AP +.005699/+ .012135/+ .024849; Brier +.000399/+ .000537/+ .000682; log loss +.008518/+ .009524/+ .011177. These are descriptive means without interval estimates.
* Public repo contains analysis code and anonymized outputs only; **manuscript remains private**.
