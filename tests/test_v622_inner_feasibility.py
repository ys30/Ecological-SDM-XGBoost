#!/usr/bin/env python3
"""Regression tests for structural inner-CV impossibility, no model fitting."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("c1_gate",Path(__file__).resolve().parents[1]/"scripts"/"98_v622_c1_inner_spatial_preflight.py")
gate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

class StructuralFeasibilityTests(unittest.TestCase):
    def test_one_positive_is_structurally_impossible(self):
        rows=[
            {"plot":"1","Species":"rare","pres.abs":"1"},
            {"plot":"2","Species":"rare","pres.abs":"0"},
            {"plot":"3","Species":"rare","pres.abs":"0"},
            {"plot":"4","Species":"rare","pres.abs":"0"},
            {"plot":"5","Species":"rare","pres.abs":"0"},
        ]
        fmap={str(i):i for i in range(1,6)}
        cert=gate.one_positive_certificates(rows,fmap,["rare"])
        self.assertEqual(len(cert),4)
        self.assertEqual({x["outer_fold"] for x in cert},{2,3,4,5})
        self.assertTrue(all(x["train_positive"]==1 for x in cert))
    def test_no_certificate_with_two_positives_in_training(self):
        rows=[
            {"plot":"1","Species":"rare","pres.abs":"1"},
            {"plot":"2","Species":"rare","pres.abs":"1"},
            {"plot":"3","Species":"rare","pres.abs":"1"},
            {"plot":"4","Species":"rare","pres.abs":"0"},
            {"plot":"5","Species":"rare","pres.abs":"0"},
        ]
        fmap={str(i):i for i in range(1,6)}
        self.assertEqual(gate.one_positive_certificates(rows,fmap,["rare"]),[])

if __name__=="__main__":
    unittest.main()
