"""Synthetic R glmnet vs sklearn logistic elastic-net probability comparison.

Engineering diagnostic, not an assertion of solver identity or formal OOF.
Both sides receive identical numeric design; standardize=FALSE in R.
"""
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .elasticnet_prototype import run_elasticnet_inner

def main():
    rng=np.random.default_rng(622)
    x=rng.normal(size=(300,4))
    eta=-.4+.65*x[:,0]-.5*x[:,1]+.3*x[:,2]
    y=rng.binomial(1,1/(1+np.exp(-eta)))
    v=rng.normal(size=(50,4))
    with tempfile.TemporaryDirectory() as folder:
        folder=Path(folder)
        pd.DataFrame(x).to_csv(folder/"x.csv",index=False)
        pd.DataFrame({"y":y}).to_csv(folder/"y.csv",index=False)
        pd.DataFrame(v).to_csv(folder/"v.csv",index=False)
        rcode=r"""
suppressPackageStartupMessages(library(glmnet))
a <- commandArgs(TRUE)
x <- as.matrix(read.csv(a[1], check.names=FALSE))
y <- read.csv(a[2])$y
v <- as.matrix(read.csv(a[3], check.names=FALSE))
alpha <- as.numeric(a[5]); lam <- as.numeric(a[6])
fit <- glmnet(x,y,family="binomial",alpha=alpha,lambda=lam,
              standardize=FALSE,intercept=TRUE,thresh=1e-10,
              maxit=100000)
p <- as.numeric(predict(fit,newx=v,s=lam,type="response"))
write.csv(data.frame(p=p),a[4],row.names=FALSE)
"""
        (folder/"compare.R").write_text(rcode)
        for alpha in (0.,.5,1.):
            for lam in (.01,.1):
                result=run_elasticnet_inner(y,np.zeros(len(v),dtype=int),x,v,
                                            alpha=alpha,lam=lam)
                if result.fallback:
                    raise AssertionError(f"sklearn fallback: {result.status}")
                subprocess.run(["Rscript",str(folder/"compare.R"),
                    str(folder/"x.csv"),str(folder/"y.csv"),str(folder/"v.csv"),
                    str(folder/"out.csv"),str(alpha),str(lam)],check=True)
                rp=pd.read_csv(folder/"out.csv")["p"].to_numpy()
                diff=float(np.max(np.abs(result.predictions-rp)))
                print(f"alpha={alpha} lambda={lam} max_abs_prob_diff={diff:.6g}")
                if diff>0.015:
                    raise AssertionError("R glmnet vs sklearn discrepancy exceeds 0.015")
    print("PASS synthetic R glmnet vs sklearn probability diagnostic; no outer scoring")

if __name__=="__main__":
    main()
