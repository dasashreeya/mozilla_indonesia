"""Regression tests for training supervision and validated long windows."""
import ast
import importlib.util
import unittest
from pathlib import Path

from scripts.prepare_recovery import contiguous_windows, target_text


class PreparationTests(unittest.TestCase):
    def test_windows_do_not_cross_removed_rows_or_timestamp_gaps(self):
        rows = [{"id": f"1:{i}", "start": (i - 1) * 5., "end": i * 5.} for i in range(1, 9)]
        windows = contiguous_windows(rows)
        self.assertEqual([[r["id"] for r in w] for w in windows],
                         [["1:1", "1:2", "1:3", "1:4"], ["1:5", "1:6", "1:7", "1:8"]])
        windows = contiguous_windows([r for r in rows if r["id"] != "1:4"])
        self.assertEqual([[r["id"] for r in w] for w in windows], [["1:5", "1:6", "1:7", "1:8"]])
        rows[3]["start"] += .5
        self.assertEqual(contiguous_windows(rows)[0][0]["id"], "1:4")

    def test_jember_style_does_not_rewrite_development_accents_or_mid_sentence_case(self):
        self.assertEqual(target_text("Wè met Émile.", "jember", "dev_style"), "we met Émile")
        self.assertEqual(target_text("Wè met Émile.", "development", "dev_style"), "wè met Émile")
        self.assertEqual(target_text("Wè met Émile.", "jember", "raw"), "wè met Émile")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for tensor regression")
class LabelMaskTests(unittest.TestCase):
    def test_left_and_right_padding_keep_only_response_and_real_eos(self):
        import torch
        # Extract only the production function to avoid importing model libraries.
        path = Path(__file__).resolve().parents[1] / "vendor/qwen3_asr_sft.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "response_only_labels")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        mask = namespace["response_only_labels"]
        inputs = torch.tensor([[0, 0, 11, 12, 21, 0], [11, 12, 21, 0, 0, 0]])
        attention = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])
        expected = torch.tensor([[-100, -100, -100, -100, 21, 0], [-100, -100, 21, 0, -100, -100]])
        torch.testing.assert_close(mask(inputs, attention, [2, 2]), expected)
        torch.testing.assert_close(inputs, torch.tensor([[0, 0, 11, 12, 21, 0], [11, 12, 21, 0, 0, 0]]))


if __name__ == "__main__":
    unittest.main()
