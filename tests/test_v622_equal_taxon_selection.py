#!/usr/bin/env python3
"""Unit tests for equal-taxon C1 tuning aggregation; no model fit or network."""
import unittest
import numpy as np

import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("c1",Path(__file__).resolve().parents[1]/"scripts"/"97_v622_c1_equal_taxon_global_pilot.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
choose_global=module.choose_global

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
