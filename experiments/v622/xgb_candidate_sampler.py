"""Frozen-candidate generator for EMS v6.22 XGBoost random search.

Pure candidate construction: no data access, model fitting or outer-test scoring.
Uses NumPy Generator(PCG64) seeded 622; record NumPy version with output.
"""
import hashlib
import json
import numpy as np

SEED = 622
ROUNDS = (50, 100, 200, 400, 800)
N_CONFIGS = 30

def generate_candidates(seed=SEED, n_configs=N_CONFIGS):
    if seed != SEED or n_configs != N_CONFIGS:
        raise ValueError("v6.22 candidate count and seed are preregistered (30, 622)")
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n_configs):
        result.append({
            "candidate_id": i + 1,
            "max_depth": int(rng.integers(1, 7)),
            "learning_rate": float(np.exp(rng.uniform(np.log(.01), np.log(.3)))),
            "min_child_weight": float(np.exp(rng.uniform(np.log(.5), np.log(20.)))),
            "subsample": float(rng.uniform(.5, 1.)),
            "colsample_bytree": float(rng.uniform(.5, 1.)),
            "n_estimators": int(rng.choice(ROUNDS)),
        })
    return result

def canonical_candidate_bytes(candidates=None):
    if candidates is None:
        candidates = generate_candidates()
    return (json.dumps(candidates, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()

def candidate_sha256(candidates=None):
    return hashlib.sha256(canonical_candidate_bytes(candidates)).hexdigest()

if __name__ == "__main__":
    print(canonical_candidate_bytes().decode(), end="")
    print("sha256:", candidate_sha256())
