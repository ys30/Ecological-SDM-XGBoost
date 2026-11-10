"""Pre-freeze GLMnet design contract tests (not model fitting or tuning)."""
import unittest
import numpy as np
import pandas as pd
from experiments.v622.design_matrix import FoldDesign,prepare_design_fold

class GLMnetDesignContract(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({
            "plot":np.arange(1,41),
            "Species":["taxon"]*40,
            "rainann":np.arange(40,dtype=float)+1,
            "soildepth":np.linspace(10,30,40),
            "tempann":np.linspace(5,25,40),
            "topo":np.sin(np.arange(40)),
        })
    def test_feature_family_and_training_only_moments(self):
        for model,n in (("glmnet_lin",4),("glmnet_quad_rt",6),
                        ("glmnet_quad_env",8)):
            train,val,test,design=prepare_design_fold(
                self.frame,{39,40},{37,38},model,standardize=True)
            self.assertEqual(train.shape[1],n)
            self.assertEqual(train.columns.tolist(),val.columns.tolist())
            self.assertEqual(train.columns.tolist(),test.columns.tolist())
            self.assertEqual(len(train),36)
            self.assertNotIn("plot",train.columns)
            self.assertNotIn("Species",train.columns)
            changed=self.frame.copy()
            changed.loc[changed["plot"].isin([37,38,39,40]),"rainann"]=1e12
            _,_,_,other=prepare_design_fold(
                changed,{39,40},{37,38},model,standardize=True)
            self.assertEqual(design.mean_,other.mean_)
            self.assertEqual(design.scale_,other.scale_)
    def test_quadratic_uses_standardized_variables(self):
        train,_,_,_=prepare_design_fold(
            self.frame,{39,40},{37,38},"glmnet_quad_rt",standardize=True)
        np.testing.assert_allclose(train["rainann_sq"],train["rainann"]**2)
        np.testing.assert_allclose(train["tempann_sq"],train["tempann"]**2)
    def test_all_four_quadratics(self):
        train,_,_,_=prepare_design_fold(
            self.frame,{39,40},{37,38},"glmnet_quad_env",standardize=True)
        for name in ("rainann","soildepth","tempann","topo"):
            np.testing.assert_allclose(train[name+"_sq"],train[name]**2)

if __name__=="__main__":
    unittest.main()
