# Verified foundational literature for the SDM manuscript

The following bibliographic records and DOI landing pages were checked against publisher pages. These are methodological citations, **not** evidence of the original dataset's source.

1. Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. https://doi.org/10.1111/ecog.02881
2. Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019). blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models. *Methods in Ecology and Evolution*, 10(2), 225–232. https://doi.org/10.1111/2041-210X.13107
3. Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum entropy modeling of species geographic distributions. *Ecological Modelling*, 190(3–4), 231–259. https://doi.org/10.1016/j.ecolmodel.2005.03.026
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

**Usage:** Roberts and Valavi support the need for spatially structured validation; Phillips describes the original MaxEnt species distribution modeling framework, while Chen and Guestrin describe XGBoost. The current project uses a custom R quantile-grid spatial partition, not the `blockCV` package, and its MaxEnt uses surveyed non-detections as approximate background rather than independently sampled landscape background. Cite these studies as methodological context, not as proof that their implementations exactly match this repository.

**Still needed:** verify the hierarchical Bayesian/brms methodological reference, precision–recall under imbalance, MaxEnt interpretation/calibration references, and the original study dataset citation before final journal formatting.
