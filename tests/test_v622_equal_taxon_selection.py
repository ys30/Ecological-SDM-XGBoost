#!/usr/bin/env python3
"""Unit tests for equal-taxon C1 tuning aggregation; no model fit or network."""
import unittest
import numpy as np

def choose_global(scores):
    """One row per taxon; one column per candidate; maximize equal-taxon mean."""
    x=np.asarray(scores,dtype=float)
    if x.ndim!=2 or x.shape[0]!=8 or x.shape[1]!=30:
        raise ValueError("Expected 8 taxa by 30 candidates")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite candidate scores")
    return int(np.argmax(x.mean(axis=0)))

class EqualTaxonTuningTests(unittest.TestCase):
    def test_equal_taxon_not_pooled_by_counts(self):
        x=np.zeros((8,30))
        x[0,0]=8.0
        x[1:,1]=2.0
        self.assertEqual(choose_global(x),1)
    def test_tie_break_is_first_candidate(self):
        self.assertEqual(choose_global(np.zeros((8,30))),0)
    def test_rejects_missing_taxa(self):
        with self.assertRaises(ValueError):
            choose_global(np.zeros((7,30)))
    def test_rejects_nonfinite(self):
        x=np.zeros((8,30));x[0,0]=np.nan
        with self.assertRaises(ValueError):
            choose_global(x)
if __name__=="__main__": unittest.main()
