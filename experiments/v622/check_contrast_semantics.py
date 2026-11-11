"""Scientific comparison semantics gate for the v6.22 model matrix.

Validates claims supported by matrix metadata, without fitting models.
A CI pass is not a substitute for CRS verification or study freeze.
"""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parent
with (ROOT/"config/v622_model_matrix.csv").open(newline="") as f:
    models={r["model_id"]:r for r in csv.DictReader(f)}
with (ROOT/"config/v622_contrasts.csv").open(newline="") as f:
    contrasts=list(csv.DictReader(f))
assert len(models)==19 and len(contrasts)==20
errors=[]
for c in contrasts:
    a,b=models[c["model_a"]],models[c["model_b"]]
    key=c["contrast_id"]
    if key.startswith("C1") and not (a["family"]==b["family"]=="XGBoost" and a["predictors"]==b["predictors"] and a["sharing"]==b["sharing"]):
        errors.append(f"{key}: tuning contrast changes more than tuning")
    if key.startswith("C2") and a["response"]==b["response"]:
        errors.append(f"{key}: response-shape contrast has identical response")
    if key.startswith("C3") and a["sharing"]==b["sharing"]:
        errors.append(f"{key}: sharing contrast has identical sharing")
    if key.startswith("C4") and key in ("C4a","C4b","C4c") and a["predictors"]!=b["predictors"]:
        errors.append(f"{key}: matched raw predictor claim contradicted")
    if key in ("C2d",) and not ("archived" in (a["status"]+" "+a["notes"]+" "+a["schemes"]) and "archived" in (b["status"]+" "+b["notes"]+" "+b["schemes"])):
        errors.append(f"{key}: archived Bayesian result mislabeled")
assert not errors,"; ".join(errors)
print(f"PASS semantics: {len(models)} models, {len(contrasts)} contrasts; no metadata contradiction")
print("LIMITATIONS: C3d paired AUC excludes per-taxon fallback cells; report matched coverage and rare-taxon stratum.")
print("LIMITATIONS: C4 MaxEnt background response semantics are not interchangeable with calibrated occurrence probabilities.")
print("LIMITATIONS: projected CRS and 10-km buffered designs are not validated here.")
