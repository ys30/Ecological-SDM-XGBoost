import unittest
import numpy as np
from experiments.v622.elasticnet_path_prototype import lambda_path,select_alpha_lambda,resolve_refit_lambda,equal_taxon_mean_skill,select_alpha_lambda_by_taxon

class PathPrototype(unittest.TestCase):
    def test_paths_and_monotonicity(self):
        rng=np.random.default_rng(622)
        x=rng.normal(size=(80,4)); y=np.array([0,1]*40)
        for a in (0.,.5,1.):
            p=lambda_path(x,y,a)
            self.assertEqual(len(p),30)
            self.assertTrue(np.all(np.diff(p)<0))
            self.assertAlmostEqual(p[-1]/p[0],.001)
    def test_selection_no_outer_scoring(self):
        rng=np.random.default_rng(622)
        x=rng.normal(size=(100,3)); y=np.array([0,1]*50)
        splits=[(x[:70],y[:70],x[70:],y[70:])]
        result=select_alpha_lambda(splits)
        self.assertIn(result["alpha"],(0.,.5,1.))
        self.assertEqual(result["n_candidates"],90)
        self.assertTrue(np.isfinite(result["mean_skill"]))
    def test_invalid_one_class_path(self):
        with self.assertRaises(ValueError):
            lambda_path(np.ones((20,3)),np.zeros(20),.5)
    def test_refit_lambda_uses_only_outer_training(self):
        rng=np.random.default_rng(622)
        x=rng.normal(size=(80,4)); y=np.array([0,1]*40)
        selected={"alpha":.5,"lambda_index":7}
        a=resolve_refit_lambda(x,y,selected)
        b=resolve_refit_lambda(x,y,selected)
        self.assertEqual(a,b)
        self.assertAlmostEqual(a["lambda"],lambda_path(x,y,.5)[7])
        with self.assertRaises(ValueError):
            resolve_refit_lambda(x,y,{"alpha":.5,"lambda_index":30})
    def test_equal_taxon_not_record_weighted(self):
        rows=[("common",0,.8),("common",1,.6),("rare",0,-.4)]
        result=equal_taxon_mean_skill(rows)
        self.assertAlmostEqual(result["mean_skill"],.15)
        self.assertEqual(result["n_taxa"],2)
        self.assertEqual(result["n_taxon_folds"],3)
        with self.assertRaises(ValueError):
            equal_taxon_mean_skill(rows+[("rare",0,.1)])
    def test_equal_taxon_joint_selection(self):
        rng=np.random.default_rng(622)
        x=rng.normal(size=(60,3))
        y=np.array([0,1]*30)
        splits={
            "common":[(0,x[:40],y[:40],x[40:50],y[40:50]),
                      (1,x[:40],y[:40],x[50:],y[50:])],
            "rare":[(0,x[:40],y[:40],x[40:50],y[40:50])]}
        result=select_alpha_lambda_by_taxon(splits)
        self.assertEqual(result["n_candidates"],90)
        self.assertEqual(result["n_taxa"],2)
        self.assertEqual(result["n_taxon_folds"],3)
        self.assertTrue(np.isfinite(result["mean_skill"]))
    def test_duplicate_taxon_fold_rejected(self):
        x=np.ones((12,2)); y=np.array([0,1]*6)
        with self.assertRaises(ValueError):
            select_alpha_lambda_by_taxon({
                "a":[(0,x,y,x[:2],y[:2]),(0,x,y,x[:2],y[:2])]})
    def test_rare_taxon_one_class_fallback_is_scored(self):
        rng=np.random.default_rng(622)
        x=rng.normal(size=(60,3))
        y=np.array([0,1]*30)
        rare=np.zeros(40,dtype=int)
        rare[:3]=1
        result=select_alpha_lambda_by_taxon({
            "common":[(0,x[:40],y[:40],x[40:50],y[40:50])],
            "rare":[(0,x[:40],rare,x[40:50],np.zeros(10,dtype=int))]})
        self.assertEqual(result["n_candidates"],90)
        self.assertEqual(result["n_fallback_taxon_folds"],1)
        self.assertEqual(result["fit_status_counts"]["insufficient_positives"],1)
        self.assertEqual(result["n_taxa"],2)
        self.assertTrue(np.isfinite(result["mean_skill"]))
if __name__=="__main__":
    unittest.main()
