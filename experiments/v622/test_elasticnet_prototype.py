import unittest
import numpy as np
from experiments.v622.elasticnet_prototype import run_elasticnet_inner

class TestElasticNetPrototype(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(622)
        self.x=rng.normal(size=(120,6))
        self.y=np.array([0,1]*60)
        self.v=rng.normal(size=(16,6))
    def test_three_alpha_choices(self):
        for alpha in (0.,.5,1.):
            result=run_elasticnet_inner(self.y,[0,1]*8,self.x,self.v,
                                        alpha=alpha,lam=.5)
            self.assertFalse(result.fallback)
            self.assertIn(result.status,("fit_success","retry_success"))
            self.assertEqual(result.predictions.shape,(16,))
            self.assertTrue(((result.predictions>=0)&(result.predictions<=1)).all())
    def test_rare_taxon(self):
        result=run_elasticnet_inner([0]*119+[1],[0,1],self.x,self.v[:2])
        self.assertEqual(result.status,"insufficient_positives")
        self.assertEqual(result.attempt_count,0)
    def test_bad_lambda(self):
        with self.assertRaises(ValueError):
            run_elasticnet_inner(self.y,[0,1]*8,self.x,self.v,lam=0)
    def test_bad_dimensions(self):
        with self.assertRaises(ValueError):
            run_elasticnet_inner(self.y,[0,1]*8,self.x,self.v[:,:5])
if __name__=="__main__":
    unittest.main()
