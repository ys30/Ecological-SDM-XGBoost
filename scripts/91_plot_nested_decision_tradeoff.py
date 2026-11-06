"""Generate publication-ready, aggregate-only nested decision comparison figure."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=[
("nsw19","GLM",5,98,3,1061),("nsw19","Bayesian",4,119,4,1040),
("nsw20","GLM",13,26,5,1123),("nsw20","Bayesian",1,5,17,1144),
("nsw22","GLM",2,79,4,1082),("nsw22","Bayesian",4,87,2,1074)]
d=pd.DataFrame(rows,columns=["species","model","tp","fp","fn","tn"])
d["recall"]=d.tp/(d.tp+d.fn)
d["fpr"]=d.fp/(d.fp+d.tn)
out=Path("results/nested_decision_figures");out.mkdir(parents=True,exist_ok=True)
d.to_csv(out/"five_fold_10km_2pct_aggregate.csv",index=False)
fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharex=True)
for ax,(key,title) in zip(axes,[("recall","Sensitivity (recall)"),("fpr","False-positive rate")]):
 for model,marker in [("GLM","o"),("Bayesian","s")]:
  sub=d[d.model==model]
  ax.plot(sub.species,sub[key]*100,marker=marker,linewidth=1.6,label=model)
 ax.set_title(title)
 ax.set_ylabel("Percent (%)")
 ax.set_ylim(bottom=0)
 ax.grid(axis="y",alpha=.2)
 if key=="fpr":ax.axhline(2,color="gray",linestyle="--",linewidth=1,label="Inner target (2%)")
axes[0].legend(frameon=False);axes[1].legend(frameon=False)
fig.suptitle("Nested spatial validation | 10 km buffer | five outer folds")
fig.tight_layout()
fig.savefig(out/"nested_bayesian_glm_tradeoff.png",dpi=220,bbox_inches="tight")
fig.savefig(out/"nested_bayesian_glm_tradeoff.pdf",bbox_inches="tight")
print(d.to_string(index=False))
