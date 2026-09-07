"""Known statistical examples and confounding safeguards, no dataset fixtures."""
import unittest

import numpy as np

from scripts.audit_relationships import (
    categorical, correlate, design_matrix, distribution_shift, eta_squared,
    group_residual, js_divergence, partial_r_squared, rank, same_speaker_contrast,
)


class RelationshipTests(unittest.TestCase):
    def test_average_ranks_preserve_ties_and_perfect_reverse_order(self):
        np.testing.assert_allclose(rank([3, 1, 1]), [3, 1.5, 1.5])
        self.assertAlmostEqual(correlate(rank([1, 2, 3]), rank([3, 2, 1])), -1)
        self.assertIsNone(correlate([1, 1], [1, 2]))

    def test_ks_and_cliff_delta_have_expected_sign_and_tie_handling(self):
        self.assertEqual(distribution_shift([1, 2], [3, 4])["ks_d"], 1)
        self.assertEqual(distribution_shift([1, 2], [3, 4])["cliff_delta_dev_minus_train"], 1)
        self.assertEqual(distribution_shift([3, 4], [1, 2])["cliff_delta_dev_minus_train"], -1)
        result = distribution_shift([1, 1, 2], [1, 1, 2])
        self.assertEqual(result["ks_d"], 0)
        self.assertEqual(result["cliff_delta_dev_minus_train"], 0)

    def test_nested_speakers_and_conversations_cannot_be_independently_estimated(self):
        rows = [{"speaker": s, "conversation": "a" if s in "12" else "b"}
                for s in "11223344"]
        design = design_matrix(rows, ["speaker", "conversation"])
        self.assertEqual(design.shape[1], 5)
        self.assertEqual(np.linalg.matrix_rank(design), 4)
        self.assertAlmostEqual(categorical(rows, "speaker", "conversation")["cramers_v"], 1)

    def test_group_adjustment_removes_between_group_offset(self):
        groups = ["a", "a", "b", "b"]
        np.testing.assert_allclose(group_residual([1, 3, 101, 103], groups), [-1, 1, -1, 1])
        self.assertEqual(eta_squared([1, 1, 3, 3], groups), 1)
        self.assertEqual(eta_squared([1, 3, 1, 3], groups), 0)

    def test_partial_effect_detects_within_speaker_language_difference(self):
        rows = [{"speaker": s, "language": lang, "value": offset + effect}
                for s, offset in [("a", 0), ("b", 100)]
                for lang, effect in [("x", 0), ("y", 2)]]
        self.assertAlmostEqual(partial_r_squared(rows, "value", ["speaker"], ["language"]), 1)

    def test_shared_speaker_comparison_eliminates_composition_only_gap(self):
        rows = ([{"speaker": "a", "language": "ind", "value": 0}] * 9 +
                [{"speaker": "a", "language": "javind", "value": 0}] +
                [{"speaker": "b", "language": "ind", "value": 100}] +
                [{"speaker": "b", "language": "javind", "value": 100}] * 9)
        result = same_speaker_contrast(rows, "value")
        self.assertEqual(result["raw_mean_difference_ind_minus_mixed"], -80)
        self.assertEqual(result["same_speaker_difference_weighted_by_ind_counts"], 0)

    def test_js_divergence_bounds_and_symmetry(self):
        self.assertEqual(js_divergence({"a": 1}, {"a": 2}), 0)
        self.assertEqual(js_divergence({"a": 1}, {"b": 1}), 1)
        a, b = {"a": 2, "b": 3}, {"b": 1, "c": 3}
        self.assertAlmostEqual(js_divergence(a, b), js_divergence(b, a))


if __name__ == "__main__":
    unittest.main()
