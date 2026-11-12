# C2a real-data spatial inner pilot — verified engineering result

**Verified CI:** [run 43286664343](https://github.com/ys30/Ecological-SDM-XGBoost/actions/runs/43286664343), successful. Pilot uses only outer-training observations for grid4 outer fold 1 and taxon nsw18. It is **not** an outer test or a confirmatory result.

The run completed 24 candidate settings (two response families × three alpha × four lambda), each scored across four spatial inner folds: **96 inner fits**. The selected candidate per response family:

| Model | alpha | lambda | Mean inner relative log-loss skill | fallback inner folds | failed inner folds |
|---|---:|---:|---:|---:|---:|
| glmnet_lin | 1.0 | 0.01 | 0.2207593924 | 0 | 0 |
| glmnet_quad_rt | 1.0 | 0.001 | 0.4446404596 | 0 | 0 |

Both winners used lasso (`alpha=1`). Candidate values are training-fold specific and must **not** be reused across other outer folds or taxa. The Python sklearn SAGA adapter is a computational approximation, not certified R glmnet parity. The 12-candidate grid is the engineering smoke subset; before formal outer fitting, confirm the frozen preregistered grid and selection/tie-break convention. The pilot reports inner validation only, never outer AUC/AP or OOF predictions.

## Next gate

1. Verify frozen full alpha/lambda grid, class-support policy and fitting retry behavior.
2. Execute nested tuning separately for all 8 taxa × 5 outer folds × 2 spatial designs × 2 model families, with fold-specific traces.
3. Store complete paired outer OOF predictions, then evaluate prespecified C2a primary AUC and secondary AP/proper scores using 2,000 taxon-cluster bootstrap draws (seed 622).
4. Do not claim confirmation of C2b AP improvements from this inner-only smoke.
