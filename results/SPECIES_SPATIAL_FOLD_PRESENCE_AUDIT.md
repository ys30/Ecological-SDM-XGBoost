# Species-by-fold positive-case audit

Computed by joining `data/train.csv` to `results/spatial_2d_block_assignments.csv` on `plot`, then counting `pres.abs == 1` per species and spatial fold.

| Species | Fold 1 (n=237) | Fold 2 (n=234) | Fold 3 (n=234) | Fold 4 (n=229) | Fold 5 (n=233) | Total positives | Folds with positives |
|---|---:|---:|---:|---:|---:|---:|---:|
| *Angophora costata* | 47 | 70 | 40 | 19 | 68 | 244 | 5 |
| *Corymbia gummifera* | 4 | 1 | 3 | 0 | 0 | 8 | 3 |
| *Corymbia intermedia* | 9 | 0 | 9 | 0 | 0 | 18 | 2 |
| *Eucalyptus blakelyi* | 5 | 10 | 24 | 0 | 0 | 39 | 3 |
| *Eucalyptus campanulata* | 15 | 19 | 21 | 35 | 19 | 109 | 5 |
| *Eucalyptus carnea* | 0 | 1 | 0 | 0 | 5 | 6 | 2 |
| *Eucalyptus fastigata* | 34 | 58 | 67 | 73 | 38 | 270 | 5 |
| *Eucalyptus nova-anglica* | 19 | 8 | 14 | 6 | 6 | 53 | 5 |

All folds contain at least 229 plot records per species; when a species–fold has zero positives, ROC AUC is undefined because only the negative outcome class is observed. This directly explains the two excluded *E. blakelyi* folds. It also explains why three rare species cannot be assessed consistently across all five folds; however, **their exclusion from the matched three-model table is a benchmark eligibility decision**, not proof that no individual fold can be scored. For *C. gummifera* the three positive-containing folds have only 4, 1 and 3 positives; for *C. intermedia* only two folds contain positives (9 each); and for *E. carnea* only two folds contain positives (1 and 5). The study should avoid extrapolating strong model rankings from these tiny event counts.

AUC requires both positive and negative classes in the test fold. Average precision can be computed in positive-containing folds but becomes uninformative for zero-positive folds; comparisons must state eligibility rules explicitly.

**Source audit scope:** Counts were recalculated from the committed raw CSV and plot–fold assignment. This audit does not refit models or infer significance.
