import unittest
import pandas as pd
from experiments.v622.design_matrix import FoldDesign,prepare_design_fold
class TestDesignMatrix(unittest.TestCase):
    def sample(self):
        return pd.DataFrame({"plot":[1,2,3,4,5],"Species":["a","a","b","b","new"],
            "rainann":[1.,2.,3.,4.,999.],"soildepth":[1.,2.,3.,4.,999.],
            "tempann":[5.,6.,7.,8.,999.],"topo":[1.,2.,3.,4.,999.]})
    def test_quadratic_contract(self):
        d=self.sample()
        for m,n in [("glm_lin",0),("glm_quad_rt",2),("glm_quad_env",4),
                    ("glmnet_quad_rt",2),("glmnet_quad_env",4)]:
            tr,val,te,design=prepare_design_fold(d,{5},{4},m,standardize=True)
            self.assertEqual(len(tr.columns),4+n)
            self.assertEqual(tr.columns.tolist(),val.columns.tolist())
            self.assertEqual(tr.columns.tolist(),te.columns.tolist())
            self.assertNotIn("Species=a",tr.columns)
            self.assertNotIn("pres.abs",tr.columns)
        tr,_,_,_=prepare_design_fold(d,{5},{4},"glm_quad_rt",standardize=True)
        self.assertTrue((tr["rainann_sq"]==tr["rainann"]**2).all())
    def test_holdout_does_not_change_fit(self):
        d=self.sample()
        _,_,_,a=prepare_design_fold(d,{5},{4},"glm_quad_rt",standardize=True)
        d.loc[d["plot"]==5,"rainann"]=1e10
        d.loc[d["plot"]==4,"rainann"]=-1e10
        _,_,_,b=prepare_design_fold(d,{5},{4},"glm_quad_rt",standardize=True)
        self.assertEqual(a.mean_,b.mean_)
        self.assertEqual(a.scale_,b.scale_)
    def test_pooled_taxon_levels_training_only(self):
        d=self.sample()
        tr,val,te,_=prepare_design_fold(d,{5},{4},"glm_shared_lin")
        self.assertIn("Species=a",tr.columns)
        self.assertIn("Species=b",tr.columns)
        self.assertNotIn("Species=new",tr.columns)
        self.assertEqual(te.filter(like="Species=").sum().sum(),0)
    def test_reject_unknown_family(self):
        with self.assertRaises(ValueError):
            FoldDesign("maxent_tuned")
if __name__=="__main__":
    unittest.main()
