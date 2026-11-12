# EMS v6.22 C1 — manuscript-ready supplementary insert (verified)

**Editorial status:** supplementary exploratory C1 analysis only. This insert is not a replacement for the historical integrated XGBoost/MaxEnt/Bayesian benchmark. Keep v6.22 distinct from older experiments; the frozen 19-model/20-contrast matrix has not been fully evaluated here.

## Methods — nested spatial tuning and implementation sensitivity

We analysed 1,167 NSW open-forest plots, comprising 9,336 plot–taxon binary outcomes across eight anonymised taxa. Fixed geographic grid4 and 15-cluster spatial partitions each defined five outer evaluation folds; tuning used training-only inner spatial folds. For C1, we compared fixed-parameter and nested-tuned species-specific and pooled XGBoost models using matched outer-fold predictions. The fixed Python models reimplemented the historical R parameter values but were not assumed numerically equivalent to the original R models. For the exploratory interaction, the effect was defined as the species-specific tuned-minus-fixed difference minus the pooled tuned-minus-fixed difference, calculated at taxon–fold level. Primary uncertainty was estimated with 2,000 paired taxon-cluster bootstrap draws (seed 622), preserving within-taxon folds. The prospective v6.22 analysis plan designated ROC AUC the primary endpoint, with log loss and Brier score secondary; the tuning-by-sharing interaction was explored after the freeze and is not a preregistered confirmatory contrast.

## Results — exploratory C1 interaction

In the initial Python-fixed analyses, the mean interaction in grid4 was +0.00592 for AUC (95% bootstrap interval −0.04725 to +0.04300), −0.01811 for log loss (−0.03057 to −0.00548), and −0.00541 for Brier score (−0.00883 to −0.00205). Corresponding km15 interactions were +0.01743 for AUC (−0.03395 to +0.05956), −0.02350 for log loss (−0.03907 to −0.00778), and −0.00506 for Brier score (−0.00954 to −0.00138). Both AUC intervals spanned zero, and the grid4 mean AUC magnitude was below the preregistered 0.01 threshold. Accordingly, the prespecified primary AUC support criterion was **not met**.

Leave-one-taxon-out analysis preserved the negative log-loss and Brier interaction directions for all 16 design-by-omission combinations; AUC signs were not stable. When excluding fallback predictions in all paired arms, 8,866 grid4 records (38 taxon-fold cells) and 8,639 km15 records (37 cells) remained. The nonfallback log-loss interactions were −0.01820 (95% CI −0.03074 to −0.00441) and −0.02457 (−0.04062 to −0.00726), respectively; Brier interactions were −0.00569 (−0.00901 to −0.00227) and −0.00545 (−0.00988 to −0.00152).

An additional grid4 implementation sensitivity compared Python fixed controls with actual R `xgb.train` fixed predictions fitted using explicitly specified histogram tree construction, on an identical set of 8,866 nonfallback records. R and Python fixed predictions were correlated but not numerically equivalent (species mean absolute difference 0.00758; pooled 0.00630). With R-hist fixed controls, the exploratory interaction was −0.00435 for AUC (95% CI −0.08192 to +0.03630), −0.01938 for log loss (−0.03171 to −0.00573), and −0.00599 for Brier (−0.00922 to −0.00267). Thus secondary proper-score interaction directions persisted across fixed-model implementations, whereas AUC changed sign and remained uncertain.

## Discussion — limits of interpretation

These results suggest that nested tuning of species-specific XGBoost models may improve probabilistic scoring more than nested tuning of pooled models in these internal spatial holdouts. They do **not** establish an effect on the preregistered primary AUC endpoint. The secondary interaction was explored after freezing the plan, confidence intervals were not adjusted for multiple comparisons, and inference is based on only eight taxon clusters. The two spatial designs reuse the same observations and are not independent external validations. Fixed R-hist models are an implementation sensitivity, not an exact reproduction of the historical R defaults; differences in R/Python versions, RNG or training details remain unresolved. Primary interaction AUC excludes non-evaluable taxon-fold cells, while the original proper-score interaction includes fallback predictions; nonfallback results were therefore reported separately. Avoid claims of causal benefit from pooling or tuning.

## Figure C1 caption

**Figure C1. Exploratory interaction between tuning and model sharing under spatially blocked cross-validation.** Points show mean taxon–fold interactions, defined as (species-specific tuned − fixed) − (pooled tuned − fixed); horizontal whiskers show percentile 95% intervals from 2,000 taxon-cluster bootstrap draws. Grid4 and km15 represent different spatial partitions of the same dataset, while the R-hist fixed-control row is a grid4-only implementation sensitivity, not an independent third spatial replicate. The preregistered AUC support criterion was not satisfied; log loss and Brier are secondary exploratory outcomes. Population and fallback handling differ between original Python and R-hist sensitivity rows; avoid interpreting between-row distances as isolated implementation effects.

## Reproducibility and placement

- Figure artifact: GitHub Actions run [43123540439](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43123540439), artifact ID `13900989599`, PNG and vector PDF.
- Original interaction: run `42965741281`; LOTO `42967072873`; nonfallback `43050953967`.
- R/Python parity: run `43079394306`; R-baseline sensitivity: run `43108252039`.
- Full source audit: [C1 interaction verified results](C1_INTERACTION_VERIFIED_RESULTS.md), [R/Python equivalence audit](C1_R_PYTHON_EQUIVALENCE_AUDIT.md).
- Recommended placement: Supplementary Methods C1, Supplementary Results C1, Supplementary Figure C1, and a concise main-text Discussion sentence. Do not silently merge into the historical main manuscript's matched three-model claims.

**Editorial checks still required:** visual inspection of PDF/PNG, manuscript-wide numbering and citations, disclosure/redistribution permissions, software-version tables, and final verification of model-count coverage.
