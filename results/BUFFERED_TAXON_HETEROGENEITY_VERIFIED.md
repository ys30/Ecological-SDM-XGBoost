# Verified buffered taxon-level model heterogeneity

**Execution:** [GitHub Actions #41965224654 (success)](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/41965224654). Artifact `buffered-taxon-heterogeneity` includes `taxon_heterogeneity.csv` and `detection_heterogeneity_associations.csv`. Source: [scripts/76_buffered_taxon_heterogeneity.py](../scripts/76_buffered_taxon_heterogeneity.py).

The source dataset has 1,167 observations per anonymized taxon. Detections: nsw18=244, nsw19=8, nsw20=18, nsw21=39, nsw22=6, nsw23=270, nsw24=109, nsw25=53 (total 747).

The following are **within-taxon averages over eligible folds** of paired quadratic GLM minus quadratic hierarchical Bayesian differences at a nominal 10 km training buffer. Positive AUC/AP differences favor GLM.

| Taxon | Detections | Eligible AUC/AP folds | ΔAUC | ΔAP |
|---|---:|---:|---:|---:|
| nsw18 | 244 | 5 | +0.00275 | −0.01355 |
| nsw19 | 8 | 4 | +0.09379 | +0.13842 |
| nsw20 | 18 | 2 | +0.04038 | +0.23915 |
| nsw21 | 39 | 5 | +0.00524 | +0.03954 |
| nsw22 | 6 | 3 | −0.09349 | −0.19509 |
| nsw23 | 270 | 5 | −0.01314 | −0.01537 |
| nsw24 | 109 | 5 | +0.03546 | −0.02165 |
| nsw25 | 53 | 5 | −0.01711 | +0.00421 |

At 10 km, exploratory Spearman association between total detections and quadratic ΔAUC: ρ=−0.14286 (nominal p=0.73576); with quadratic ΔAP: ρ=−0.23810 (p=0.57016). For 5 km, quadratic ΔAUC ρ=−0.07143 (p=0.86653) and ΔAP ρ=−0.26190 (p=0.53092). For 20 km, quadratic ΔAUC ρ=+0.04762 (p=0.91085) and ΔAP ρ=−0.23810 (p=0.57016). No association is established.

**Interpretation:** The very rare nsw19/nsw20 favor GLM on the displayed discrimination metrics while nsw22 favors Bayesian. Thus rarity alone does not predict partial-pooling benefit. However, 2–4 eligible folds for rare taxa and extremely few positive observations make taxon-specific AP particularly volatile. Fold-level differences are not independent biological replicates. These are descriptive, within-dataset results, not confirmatory evidence, causal mechanisms, or prospective performance. No taxonomic identities are disclosed. The formal source CRS remains unverified; [distance-scale audit](COORDINATE_SCALE_VERIFIED.md) supports approximate metre units.

See [buffered matched model comparison](BUFFERED_FOUR_MODEL_MATCHED_VERIFIED.md) and [eight-taxon bootstrap intervals](BUFFERED_TAXON_BOOTSTRAP_VERIFIED.md), all of which include zero.
