import unittest
from experiments.v622.xgb_candidate_sampler import (
    SEED, N_CONFIGS, ROUNDS, generate_candidates, candidate_sha256
)

class TestXGBCandidates(unittest.TestCase):
    def test_deterministic_frozen_grid(self):
        a = generate_candidates()
        self.assertEqual(len(a), 30)
        self.assertEqual(a, generate_candidates())
        self.assertEqual(len({tuple(sorted(x.items())) for x in a}), N_CONFIGS)
        self.assertEqual(candidate_sha256(a), candidate_sha256(generate_candidates()))
        self.assertEqual([x["candidate_id"] for x in a], list(range(1, 31)))

    def test_prespecified_bounds(self):
        for x in generate_candidates():
            self.assertTrue(1 <= x["max_depth"] <= 6)
            self.assertTrue(.01 <= x["learning_rate"] <= .3)
            self.assertTrue(.5 <= x["min_child_weight"] <= 20.)
            self.assertTrue(.5 <= x["subsample"] <= 1.)
            self.assertTrue(.5 <= x["colsample_bytree"] <= 1.)
            self.assertIn(x["n_estimators"], ROUNDS)

    def test_reject_unregistered_search(self):
        with self.assertRaises(ValueError):
            generate_candidates(seed=SEED+1)
        with self.assertRaises(ValueError):
            generate_candidates(n_configs=29)

if __name__ == "__main__":
    unittest.main()
