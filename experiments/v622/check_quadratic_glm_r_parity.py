"""R/Python parity for train-standardized rain/temp quadratic GLM (synthetic only)."""
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from .design_matrix import FoldDesign
from .glm_adapter import run_glm_inner

def main():
    rng=np.random.default_rng(6222)
    names=["rainann","soildepth","tempann","topo"]
    raw=rng.normal(size=(280,4))
    raw[:,0]=100+20*raw[:,0]
    raw[:,2]=20+3*raw[:,2]
    eta=-0.8+0.4*raw[:,1]-0.25*raw[:,3]+0.3*((raw[:,0]-100)/20)**2
    y=rng.binomial(1,1/(1+np.exp(-eta)))
    test=rng.normal(size=(35,4))
    test[:,0]=100+20*test[:,0]
    test[:,2]=20+3*test[:,2]
    tr=pd.DataFrame(raw,columns=names)
    te=pd.DataFrame(test,columns=names)
    design=FoldDesign("glm_quad_rt",standardize=True,missing_policy="error").fit(tr)
    x=design.transform(tr).to_numpy()
    v=design.transform(te).to_numpy()
    result=run_glm_inner(y,np.zeros(len(te),dtype=int),x,v)
    if result.fallback:
        raise AssertionError(f"unexpected GLM fallback: {result.status} {result.first_error}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        tr["response"]=y
        tr.to_csv(tmp/"train.csv",index=False)
        te.to_csv(tmp/"test.csv",index=False)
        rcode = """
tr <- read.csv(commandArgs(TRUE)[1])
te <- read.csv(commandArgs(TRUE)[2])
for (nm in c("rainann","soildepth","tempann","topo")) {
  mu <- mean(tr[[nm]])
  sig <- sd(tr[[nm]])
  tr[[paste0("z_",nm)]] <- (tr[[nm]]-mu)/sig
  te[[paste0("z_",nm)]] <- (te[[nm]]-mu)/sig
}
fit <- glm(response ~ z_rainann + z_soildepth + z_tempann + z_topo +
           I(z_rainann^2) + I(z_tempann^2),
           data=tr,family=binomial(link="logit"),
           control=glm.control(maxit=100))
stopifnot(fit$converged)
write.csv(data.frame(p=predict(fit,newdata=te,type="response")),
          commandArgs(TRUE)[3],row.names=FALSE)
"""
        (tmp/"parity.R").write_text(rcode)
        subprocess.run(["Rscript",str(tmp/"parity.R"),str(tmp/"train.csv"),
                        str(tmp/"test.csv"),str(tmp/"pred.csv")],check=True)
        rp=pd.read_csv(tmp/"pred.csv")["p"].to_numpy()
    diff=float(np.max(np.abs(result.predictions-rp)))
    np.testing.assert_allclose(result.predictions,rp,atol=1e-6,rtol=1e-6)
    print(f"PASS quadratic R/Python GLM parity; max absolute probability difference={diff:.3e}")
if __name__=="__main__":
    main()
