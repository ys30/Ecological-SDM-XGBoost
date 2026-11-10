import unittest
from experiments.v622.xgb_selection_objective import select_candidate_score, choose_candidate

class TestSelectionObjective(unittest.TestCase):
    def test_taxon_first_not_fold_first(self):
        rows = [
            {"taxon":"a","inner_fold":0,"relative_log_loss_skill":0.9},
            {"taxon":"a","inner_fold":1,"relative_log_loss_skill":0.7},
            {"taxon":"b","inner_fold":0,"relative_log_loss_skill":-0.2},
        ]
        result = select_candidate_score(rows, expected_taxa=["a","b"])
        self.assertAlmostEqual(result["score"], 0.3)
        self.assertEqual(result["folds_per_taxon"], {"a":2,"b":1})
        self.assertEqual(result["cells"], 3)

    def test_duplicate_and_missing_taxon_rejected(self):
        a = {"taxon":"a","inner_fold":0,"relative_log_loss_skill":0.1}
        with self.assertRaises(ValueError):
            select_candidate_score([a,a])
        with self.assertRaises(ValueError):
            select_candidate_score([a], expected_taxa=["a","b"])

    def test_maximize_and_deterministic_tie(self):
        choice = choose_candidate([
            {"candidate_id":3,"score":0.5},
            {"candidate_id":2,"score":0.5},
            {"candidate_id":1,"score":0.4},
        ])
        self.assertEqual(choice["candidate_id"],2)

if __name__ == "__main__":
    unittest.main()
