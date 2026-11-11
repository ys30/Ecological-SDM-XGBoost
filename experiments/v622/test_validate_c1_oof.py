"""Regression tests for strict C1 OOF completeness checks."""
import unittest
import pandas as pd
from experiments.v622.validate_c1_oof import validate_oof

class TestOOF(unittest.TestCase):
    def setUp(self):
        self.ref=pd.DataFrame({"plot":[1,2],"Species":["a","a"],"pres.abs":[1,0]})
        self.folds=pd.DataFrame({"plot":[1,2],"fold":[1,2]})
        self.oof=pd.DataFrame({"plot":[1,2],"Species":["a","a"],"outer_fold":[1,2],
                               "pres.abs":[1,0],"pred":[0.7,0.2],"fallback":[0,1],
                               "candidate_id":[1,2]})
    def test_complete(self):
        self.assertEqual(validate_oof(self.oof,self.ref,self.folds,expected_folds=2)["rows"],2)
    def test_fold_mismatch(self):
        d=self.oof.copy();d.loc[0,"outer_fold"]=2
        with self.assertRaisesRegex(ValueError,"fold mismatch"):
            validate_oof(d,self.ref,self.folds,expected_folds=2)
    def test_duplicate(self):
        d=self.oof.copy();d.loc[1,"plot"]=1
        with self.assertRaisesRegex(ValueError,"coverage or uniqueness"):
            validate_oof(d,self.ref,self.folds,expected_folds=2)
    def test_label_mismatch(self):
        d=self.oof.copy();d.loc[0,"pres.abs"]=0
        with self.assertRaisesRegex(ValueError,"outcome mismatch"):
            validate_oof(d,self.ref,self.folds,expected_folds=2)
    def test_bad_probability(self):
        d=self.oof.copy();d.loc[0,"pred"]=1.2
        with self.assertRaisesRegex(ValueError,"outside"):
            validate_oof(d,self.ref,self.folds,expected_folds=2)

if __name__=="__main__":
    unittest.main()
