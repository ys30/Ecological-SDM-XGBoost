"""Report source missingness and numeric type, without fitting models."""
import json
from pathlib import Path
import pandas as pd
d=pd.read_csv("data/train.csv")
cols=["rainann","soildepth","tempann","topo","soilfert","disturb","Species","pres.abs","plot"]
report={"rows":len(d),"columns":{}}
for c in cols:
    s=d[c]
    n=pd.to_numeric(s,errors="coerce") if c not in ("Species","plot") else None
    report["columns"][c]={"missing":int(s.isna().sum()),"unique":int(s.nunique(dropna=True)),
      "non_numeric_nonmissing":int((s.notna() & n.isna()).sum()) if n is not None else None}
Path("results/v622").mkdir(parents=True,exist_ok=True)
Path("results/v622/source_missingness_audit.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
