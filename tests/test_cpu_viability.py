"""Checks for the CPU audit's edit arithmetic and held-out supervision boundary."""

import itertools
import random
import unittest

import numpy as np

from scripts.cpu_viability import (
    PIVOT, SMALL_POOL, decompose, distance, feature_data, fit_selector, group_folds,
    minimum_edits, tokens, vote, word_errors,
)


class CPUViabilityTests(unittest.TestCase):
    def test_lattice_matches_exhaustive_paths(self):
        rng = random.Random(311)
        for _ in range(250):
            ref = rng.choices(["a", "b", "c"], k=rng.randrange(6))
            network = [{tuple(rng.choices(["a", "b", "c"], k=rng.randrange(3)))
                        for _ in range(3)} for _ in range(rng.randrange(1, 5))]
            expected = min(distance(ref, [w for seq in path for w in seq])
                           for path in itertools.product(*network))
            self.assertEqual(minimum_edits(ref, network), expected)

    def test_insertion_voting_and_pivot_ties(self):
        pivot = ["a", "b"]
        parts = {PIVOT: decompose(pivot, pivot),
                 "other1": decompose(pivot, ["a", "missing", "b"]),
                 "other2": decompose(pivot, ["a", "missing", "b"])}
        self.assertEqual(vote(parts, list(parts)), "a missing b")
        self.assertEqual(vote(parts, list(parts), insertions=False), "a b")
        self.assertEqual(vote(parts, [PIVOT, "other1"]), "a b")
        self.assertEqual(decompose([], ["a"]), [("a",)])

    def test_emitted_text_uses_official_normalization(self):
        examples = [("Iku Jakarta. Bener!", "iku jakarta bener"),
                    ("e... iki, gak", "e... iki nggak"),
                    ("(?) ABC [noise]", "ABC"),
                    ("a missing b", "a b")]
        for reference, hypothesis in examples:
            self.assertEqual(distance(tokens(reference), tokens(hypothesis)),
                             word_errors(reference, hypothesis).errors)

    def test_group_folds_never_split_sessions_or_development(self):
        records = [{"source": "jember", "group": f"jember:{i}"}
                   for i in range(20) for _ in range(2)]
        records += [{"source": "development", "group": "development:5"}] * 5
        for _, train, test in group_folds(records):
            self.assertFalse({records[i]["group"] for i in train} &
                             {records[i]["group"] for i in test})
            self.assertTrue(all(records[i]["source"] == "jember" for i in train))

    def test_held_out_labels_cannot_change_fitted_weights(self):
        data = {"x": np.asarray([[1., 0., 0., 0., 1.], [0., 1., 0., 0., 0.],
                                  [1., 0., 0., 0., 1.], [0., 1., 0., 0., 0.]]),
                "y": np.asarray([1., 0., 0., 1.]),
                "owner": np.asarray([0, 0, 1, 1]),
                "slot": np.asarray([0, 0, 1, 1]),
                "offsets": np.asarray([0, 2, 4])}
        first = fit_selector(data, [0])
        data["y"][2:] = [1., 0.]
        second = fit_selector(data, [0])
        np.testing.assert_array_equal(first, second)

    def test_reference_changes_cannot_change_features_or_candidates(self):
        pivot = ["a", "b"]
        hypotheses = {m: pivot if i == 0 else ["a", "missing", "b"]
                      for i, m in enumerate(SMALL_POOL)}
        record = {"hyp": hypotheses, "ref": ["a", "missing", "b"],
                  "parts": {m: decompose(pivot, h) for m, h in hypotheses.items()}}
        first = feature_data([record], [{"transcript": "a b missing"}])
        record["ref"] = ["completely", "different"]
        second = feature_data([record], [{"transcript": "a b missing"}])
        for name in ["x", "offsets", "slot", "owner"]:
            np.testing.assert_array_equal(first[name], second[name])
        self.assertEqual(first["choices"], second["choices"])
        self.assertFalse(np.array_equal(first["y"], second["y"]))


if __name__ == "__main__":
    unittest.main()
