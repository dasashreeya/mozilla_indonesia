import json
import unittest

from scripts.train_recovery import training_result


class FinalFitResultTests(unittest.TestCase):
    def test_no_validation_is_null_and_serializes_with_strict_json(self):
        result = training_result(2040, float("inf"), float("inf"), "merged", "adapter",
                                 [{"step": 2040, "wer": None}], final_fit=True)
        json.dumps(result, allow_nan=False)
        self.assertIsNone(result["best_wer"])
        self.assertIsNone(result["baseline_wer"])
        self.assertIsNone(result["improves_baseline"])
        self.assertEqual(result["steps"], 2040)

    def test_validated_fit_keeps_measured_numbers(self):
        result = training_result(900, 0.4, 0.23, "best", None, [])
        self.assertEqual(result["best_wer"], 0.23)
        self.assertTrue(result["improves_baseline"])
        self.assertFalse(result["final_fit"])
