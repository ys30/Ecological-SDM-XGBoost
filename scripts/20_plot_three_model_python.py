"""Render matched three-model species AUC/AP charts without an R dependency."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

source=Path("results/matched_three_model_plot_data.csv")
with source.open(newline="") as f:
    data=list(csv.DictReader(f))
species=list(dict.fromkeys(row["species"] for row in data))
models=["Bayesian","MaxEnt","XGBoost"]
out=Path("results/figures");out.mkdir(parents=True,exist_ok=True)
for metric in ("AUC","AP"):
    fig,ax=plt.subplots(figsize=(11,6.3))
    y=np.arange(len(species))
    width=0.25
    for i,model in enumerate(models):
        lookup={(row["species"],row["model"]):float(row[metric]) for row in data}
        values=[lookup[(sp,model)] for sp in species]
        ax.barh(y+(i-1)*width,values,height=width,label=model)
    ax.set_yticks(y,labels=[sp.replace("."," ") for sp in species])
    ax.invert_yaxis()
    ax.set_xlim(0,1)
    ax.set_xlabel("ROC AUC" if metric=="AUC" else "Average precision")
    ax.set_title(f"Matched spatial cross-validation: {metric} by species")
    ax.legend(loc="lower right")
    ax.grid(axis="x",alpha=.2)
    fig.text(.5,.015,"Same held-out species-fold records; five eligible species, 23 species-folds. Descriptive comparison only.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    path=out/f"22_three_model_{metric.lower()}.png"
    fig.savefig(path,dpi=180)
    plt.close(fig)
    print(path)
