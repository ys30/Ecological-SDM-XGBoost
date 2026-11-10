"""Audit glmnet vs sklearn penalty normalization, without asserting solver parity.

glmnet binomial objective: mean negative log-likelihood +
lambda * (alpha*|beta|_1 + (1-alpha)*||beta||_2^2/2).
sklearn LogisticRegression saga objective (after normalization):
mean negative log-likelihood + 1/(n*C) * same penalty.
Intercept is unpenalized. This checks objective scaling, not R parity.
"""
import unittest
import numpy as np
from scipy.special import expit

def glmnet_objective(x,y,beta,intercept,alpha,lam):
    x=np.asarray(x,float); y=np.asarray(y,float); beta=np.asarray(beta,float)
    z=x@beta+intercept
    loss=np.logaddexp(0,z).mean()-(y*z).mean()
    return float(loss+lam*(alpha*np.abs(beta).sum()+
                            (1-alpha)*np.dot(beta,beta)/2))

def sklearn_normalized_objective(x,y,beta,intercept,alpha,C):
    n=len(y)
    return glmnet_objective(x,y,beta,intercept,alpha,1/(n*C))

class PenaltyConvention(unittest.TestCase):
    def test_objective_mapping_all_alphas(self):
        rng=np.random.default_rng(622)
        x=rng.normal(size=(37,4)); y=rng.binomial(1,.3,size=37)
        beta=np.array([-.5,.2,0,.7]); b0=-.2
        for alpha in (0.,.5,1.):
            for lam in (.001,.1,2.):
                C=1/(len(y)*lam)
                a=glmnet_objective(x,y,beta,b0,alpha,lam)
                b=sklearn_normalized_objective(x,y,beta,b0,alpha,C)
                self.assertAlmostEqual(a,b,places=13)
    def test_current_prototype_mapping_is_not_equivalent(self):
        n=37; lam=.1
        self.assertNotAlmostEqual(1/(n*lam),1/lam)

if __name__=="__main__":
    unittest.main()
