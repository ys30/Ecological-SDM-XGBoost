#!/usr/bin/env python3
"""Inventory the actual v6.23 CI outputs without inventing missing supplement files."""
import csv,hashlib,json
from pathlib import Path
BASE=Path("results/v623_matched_three_model")
ROOT=Path("results")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def count(p):
    if p.suffix!=".csv":return None
    with p.open(newline="",encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))
def main():
    assert (BASE/"manuscript_consistency_gate.json").exists()
    candidates=[]
    for folder in (BASE,ROOT/"v623_matched_glm_10km"):
        if not folder.exists():continue
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".csv",".json",".png",".svg",".txt"}:
                candidates.append({"path":p.as_posix(),"bytes":p.stat().st_size,
                                   "sha256":sha(p),"csv_rows":count(p)})
    assert candidates
    required=["taxon_pooled_oof_scores.csv","paired_taxon_contrasts.csv",
              "figure_matched_taxon_scores.svg","figure_paired_probability_scores.svg"]
    paths={Path(x["path"]).name for x in candidates}
    assert all(x in paths for x in required)
    figures=[x for x in candidates if x["path"].endswith((".svg",".png"))]
    report={"status":"PASS_ACTUAL_ARTIFACT_INVENTORY","file_count":len(candidates),
            "figure_count":len(figures),"files":candidates,
            "warning":"Inventory and checksums do not establish visual quality, licenses, or external validation."}
    out=BASE/"supplement_artifact_inventory.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"file_count":len(candidates),
                      "figure_count":len(figures),"inventory":str(out)},indent=2))
if __name__=="__main__":main()
