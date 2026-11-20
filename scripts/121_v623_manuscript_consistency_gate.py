#!/usr/bin/env python3
"""Guard the manuscript's frozen headline values against freshly computed CI outputs."""
import csv,json,math
from pathlib import Path
P=Path("results/v623_matched_three_model")
EXPECTED={
"xgboost":(.186905,.055984),
"glm_linear":(.204591,.062737),
"glm_quadratic":(.187524,.054642),
}
def main():
    manifest=json.loads((P/"manifest.json").read_text())
    assert manifest["status"]=="SAME_OUTER_FOLDS_AND_BUFFER_VERIFIED"
    with (P/"taxon_pooled_oof_scores.csv").open(newline="") as f: rows=list(csv.DictReader(f))
    assert len(rows)==24
    taxa={r["taxon"] for r in rows}
    assert len(taxa)==8
    report={}
    for model,(ell,brier) in EXPECTED.items():
        selected=[r for r in rows if r["model"]==model]
        assert len(selected)==8 and {r["taxon"] for r in selected}==taxa
        assert all(int(r["n"])==1167 for r in selected)
        a=sum(float(r["logloss"]) for r in selected)/8
        b=sum(float(r["brier"]) for r in selected)/8
        assert math.isclose(a,ell,abs_tol=0.00000051),f"{model} log loss changed: {a}"
        assert math.isclose(b,brier,abs_tol=0.00000051),f"{model} Brier changed: {b}"
        report[model]={"mean_logloss":round(a,6),"mean_brier":round(b,6)}
    figs=["figure_matched_taxon_scores.svg","figure_matched_taxon_scores.png",
          "figure_paired_probability_scores.svg","figure_paired_probability_scores.png"]
    for name in figs:
        assert (P/name).is_file() and (P/name).stat().st_size>1000,name
    report["status"]="PASS_MANUSCRIPT_HEADLINE_NUMBERS_AND_FIGURE_FILES"
    report["figure_files"]=figs
    report["limitations"]="File presence and metric consistency only; visual quality and figure-content review remain manual."
    (P/"manuscript_consistency_gate.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
