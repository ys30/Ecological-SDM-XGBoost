"""Synthetic R glm vs statsmodels binomial GLM prediction parity; no outer scoring."""
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .glm_adapter import run_glm_inner

def main():
    rng=np.random.default_rng(622)
    x=rng.normal(size=(240,4))
    eta=-0.5+0.4*x[:,0]-0.6*x[:,1]+0.25*x[:,2]+0.2*x[:,3]
    p=1/(1+np.exp(-eta))
    y=rng.binomial(1,p)
    v=rng.normal(size=(40,4))
    result=run_glm_inner(y,np.zeros(len(v),dtype=int),x,v)
    if result.fallback:
        raise AssertionError(f"unexpected fallback: {result.status} {result.first_error}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        names=["rainann","soildepth","tempann","topo"]
        tr=pd.DataFrame(x,columns=names)
        tr["response"]=y
        te=pd.DataFrame(v,columns=names)
        tr.to_csv(tmp/"train.csv",index=False)
        te.to_csv(tmp/"test.csv",index=False)
        rcode = """
tr <- read.csv(commandArgs(TRUE)[1])
te <- read.csv(commandArgs(TRUE)[2])
fit <- glm(response ~ rainann + soildepth + tempann + topo,
           data=tr, family=binomial(link="logit"),
           control=glm.control(maxit=100))
stopifnot(fit$converged)
write.csv(data.frame(p=predict(fit,newdata=te,type="response")),
          commandArgs(TRUE)[3],row.names=FALSE)
"""
        (tmp/"parity.R").write_text(rcode)
        subprocess.run(["Rscript",str(tmp/"parity.R"),str(tmp/"train.csv"),
                        str(tmp/"test.csv"),str(tmp/"pred.csv")],check=True)
        rpred=pd.read_csv(tmp/"pred.csv")["p"].to_numpy()
    maxdiff=float(np.max(np.abs(result.predictions-rpred)))
    np.testing.assert_allclose(result.predictions,rpred,atol=1e-6,rtol=1e-6)
    print(f"PASS R/Python GLM parity; max absolute probability difference={maxdiff:.3e}")
if __name__=="__main__":
    main()
