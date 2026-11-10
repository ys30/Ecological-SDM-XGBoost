"""Summarize pre-model AUC feasibility; do not redefine the registered estimand."""
import json
from pathlib import Path
import pandas as pd

source=Path("results/v622/inner_species_class_counts.csv")
df=pd.read_csv(source)
required={"outer_fold","inner_fold","partition","Species","n","positive","negative","auc_defined"}
assert required <= set(df), f"missing columns: {required-set(df)}"
assert len(df)==320, f"expected 320 audit records, got {len(df)}"
assert not df[list(required)].isna().any().any(), "missing audit cells"
assert (df["n"]==df["positive"]+df["negative"]).all()
assert ((df["positive"]>0)&(df["negative"]>0)).eq(df["auc_defined"]).all()
df["reason"]="both_classes"
df.loc[df["positive"].eq(0),"reason"]="zero_positive"
df.loc[df["negative"].eq(0),"reason"]="zero_negative"
df.loc[df["positive"].eq(0)&df["negative"].eq(0),"reason"]="empty_partition"
validation=df.loc[df["partition"]=="validation"].copy()
training=df.loc[df["partition"]=="train"].copy()
keys=["outer_fold","inner_fold","Species"]
pair=validation.merge(training[keys+["auc_defined"]].rename(columns={"auc_defined":"train_auc_defined"}),
                      on=keys,validate="one_to_one")
pair["eligible_auc_pair"]=pair["auc_defined"] & pair["train_auc_defined"]
per_species=pair.groupby("Species",as_index=False).agg(
    validation_folds=("auc_defined","size"),
    valid_validation_auc=("auc_defined","sum"),
    eligible_auc_pairs=("eligible_auc_pair","sum"),
    zero_positive_validation=("reason",lambda s:int((s=="zero_positive").sum())),
    zero_negative_validation=("reason",lambda s:int((s=="zero_negative").sum()))
)
per_species["invalid_validation_auc"]=per_species["validation_folds"]-per_species["valid_validation_auc"]
per_species=per_species.sort_values("Species")
out=Path("results/v622")
out.mkdir(parents=True,exist_ok=True)
per_species.to_csv(out/"inner_auc_feasibility_by_species.csv",index=False)
pair.to_csv(out/"inner_auc_feasibility_pairs.csv",index=False)
summary={
 "total_validation_species_folds":int(len(pair)),
 "valid_validation_auc":int(pair["auc_defined"].sum()),
 "invalid_validation_auc":int((~pair["auc_defined"]).sum()),
 "invalid_training_auc":int((~pair["train_auc_defined"]).sum()),
 "eligible_train_validation_pairs":int(pair["eligible_auc_pair"].sum()),
 "zero_positive_validation":int((pair["reason"]=="zero_positive").sum()),
 "zero_negative_validation":int((pair["reason"]=="zero_negative").sum()),
 "per_species":per_species.to_dict(orient="records"),
 "decision":"NO AUTOMATIC EXCLUSION OR AUC IMPUTATION. Requires preregistered handling and full outer-fold class-support audit before model selection."
}
(out/"inner_auc_feasibility_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
