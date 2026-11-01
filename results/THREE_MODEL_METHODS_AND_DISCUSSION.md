# Three-model spatial SDM benchmark: Methods and Discussion draft

## Methods

### Study design and validation

The study analyzed 9,336 binary species–plot observations for eight tree species across 1,167 forest plots. A five-fold two-dimensional spatial-block split was used to separate geographic blocks between model training and testing. Model comparison was restricted to identical held-out records and species–fold combinations with both observed presences and absences, yielding 23 evaluable species–fold combinations across five species. The remaining three species were not assigned comparative AUC rankings because sparse positive observations precluded reliable scoring.

### Predictive models

The benchmark compared XGBoost, species-specific MaxEnt fitted with the R package `maxnet`, and a hierarchical Bayesian Bernoulli model fitted with `brms`. The Bayesian logit model included annual rainfall, annual temperature, soil depth and topography, standardized using training-fold means and standard deviations. It included species-level varying intercepts and varying rainfall and temperature slopes without estimated correlations between the species effects. Priors were Normal(-2.5, 2) on the population intercept, Normal(0, 1) on population slopes, and Exponential(2) on varying-effect standard deviations. Geographic holdout observations did not contribute to fold-specific predictor scaling or Bayesian fitting. Other model-specific settings should be read from their executable scripts before a final journal submission; this document does not infer undocumented hyperparameters.

### Performance estimation

ROC AUC and average precision (AP) were evaluated within eligible held-out species–fold combinations. Species-level comparisons report arithmetic means across eligible folds; these are not pooled-observation estimates. Model rankings are descriptive, without a formal statistical test of differences. AP was included because presence prevalence differs substantially among species and ROC AUC alone can obscure minority-class detection performance.

### Bayesian convergence audit

Original Bayesian runs used two chains, 1,200 iterations per chain and 600 warmup. Independent confirmation runs used four chains, 2,000 iterations per chain and 1,000 warmup. Per-fold parameter diagnostics were screened against R-hat <= 1.01 and bulk/tail effective sample size >= 400; sampler diagnostics were screened for divergences, treedepth-limit hits and BFMI < 0.3. The five confirmed fits passed these thresholds. The matched three-model predictions, however, were produced by the original fits; confirmed-fit per-record predictions were not retained in historical artifacts.

## Discussion

### Species-dependent model performance

XGBoost had the highest species-mean ROC AUC for three of five evaluable species, while MaxEnt had the highest for two. These patterns suggest that no single approach consistently dominates spatial transfer across the observed species. Differences may reflect variation in prevalence, environmental response shape and the geographic composition of held-out folds, but these mechanisms have not been independently tested.

### Discrimination versus presence retrieval

Average precision changes the ranking for some species. In particular, a model can lead on ROC AUC without leading on AP. Ecological surveillance and prioritization decisions should therefore report both metrics and, when deployment thresholds matter, evaluate sensitivity and precision under explicitly defined decision costs.

### What Bayesian partial pooling contributes

The hierarchical model shares information across species and yields posterior predictive uncertainty. Its five confirmation fits meet the pre-specified MCMC screening criteria, and their fold-level AUC values are numerically stable relative to the original fits. This establishes computational diagnostic adequacy for the evaluated fits; it does not establish ecological calibration, unbiased prediction or superiority over the alternative algorithms. Sparse species may benefit from partial pooling in estimation, but cannot be declared externally validated when held-out positive cases are insufficient.

### Sparse-species geographic validation limits

The raw plot-by-fold audit confirms that three species omitted from the main ranking have exceptionally sparse and geographically uneven presences: *Corymbia gummifera* has eight presences distributed across three folds (4, 1 and 3), *Corymbia intermedia* has 18 distributed across two folds (nine in each), and *Eucalyptus carnea* has six distributed across two folds (one and five). Consequently, zero-presence folds cannot yield ROC AUC, and folds with a single presence yield fragile discrimination estimates. This is an evaluation-support limitation rather than evidence of intrinsically poor ecological predictability or a failure of a particular algorithm. A distinct rare-species analysis could use held-out Brier scores and log loss, which are defined in zero-presence folds, alongside prespecified presence-retrieval metrics. The Bayesian-only probability scoring and a training-fold prevalence comparison have now been conducted using stored out-of-fold predictions; a matched three-algorithm rare-species comparison remains unavailable. See [rare-species evaluation plan](RARE_SPECIES_EVALUATION_PLAN.md) and [baseline and retrieval results](BAYESIAN_RARE_SPECIES_BASELINE_AND_RETRIEVAL.md).

### Rare-species probability scoring and practical retrieval

A secondary analysis of existing Bayesian out-of-fold probabilities compared the hierarchical model against a simple species-specific prevalence baseline estimated exclusively from the other four spatial folds. Bayesian log loss was lower for all three rare species, but its Brier score was higher than the baseline for *Corymbia gummifera* (0.007294 versus 0.006835). The Bayesian model captured zero of its eight presences among the 50 highest-probability plots; for *Corymbia intermedia*, 13 of 18 presences appeared among the top 50, and for *Eucalyptus carnea*, two of six. These rankings pool predictions across geographic folds and may reflect fold-level differences; they are not a prospective external-validation estimate. See [baseline and retrieval results](BAYESIAN_RARE_SPECIES_BASELINE_AND_RETRIEVAL.md). These mixed findings reinforce that low overall probability error in highly imbalanced ecological data does not necessarily imply useful rare-species detection.

### Spatial heterogeneity and weighting sensitivity

Across the 23 eligible matched species–fold comparisons, XGBoost led in AUC in 14 folds, MaxEnt in seven and Bayesian in two. Within-species rankings varied between geographic holdouts, so a species-average advantage is not consistent dominance in each held-out region. Reweighting contrasts equally across the five species rather than across the 23 fold cells preserved the sign of all three aggregate AUC and AP differences, but changed their magnitudes. Neither ranking counts nor weighting sensitivity provide statistical significance because geographic folds and species share data structure and are not independent replications.

### Limitations and next steps

The benchmark covers a restricted region and five evaluable species, with different numbers of eligible folds across species. Species-level means treat folds equally rather than weighting by record counts. No independent external-region test or inferential uncertainty for paired model differences is available. The comparison uses original Bayesian predictions, not long-chain predictions, and differences should not be called statistically significant. Priority next analyses are a code-verified comparison of exact matched records, spatially appropriate uncertainty analysis for model differences, and a sensitivity analysis of species-fold eligibility. These should be completed before a journal-level claim of comparative model superiority.

## Source-of-truth files

- `scripts/15_bayes_hierarchical_spatial.R` — Bayesian formula, priors, fold-specific scaling and MCMC settings
- `results/matched_bayesian_maxent_xgboost_folds.csv` — matched species-fold metrics
- `results/matched_bayesian_maxent_xgboost_species.csv` — species-level means
- `results/BAYESIAN_FIVE_FOLD_CONVERGENCE_FINAL.md` — confirmed MCMC diagnostics
- `results/BAYESIAN_LONG_CHAIN_METRIC_RECONCILIATION.md` — original versus confirmed fold-level metrics
