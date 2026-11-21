# Recovered v6.21 spatial robustness

The author-supplied EMS revision archive restored fold_schemes.csv, train_exclusions.csv, kmeans15_five_fold_assignments.csv and oof_bayes_km15.csv.gz. All four SHA-256 values match experiments/v622/config/v622_experiment.yaml on ems-v622-experiments.

An independent local replay checked 1,643,136 archived prediction records for unique scheme/model/observation keys and agreement with the frozen fold assignment. It recomputed ROC AUC and average precision on shared held-out taxon–fold cells for seven common model variants across 23 schemes. All 161 scheme/model summaries reproduced. Maximum absolute discrepancies were 0 for AUC and 5.6e-17 for average precision. Quadratic per-taxon GLM ranked first by AUC in 18 schemes, full pooled XGBoost in three, environmental-plus-taxon pooled XGBoost in one and categorical MaxEnt in one.

These are archived fits with independently replayed scoring. The [buffered C1 CI](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/44908588964) evaluates a distinct protocol. The 23-scheme replay ran locally on the author-supplied archive.
