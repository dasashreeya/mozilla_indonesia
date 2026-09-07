"""Regression tests for training supervision and validated long windows."""
import ast
import importlib.util
import unittest
from pathlib import Path

from scripts.prepare_recovery import (contiguous_runs, contiguous_windows, language_tag,
                                      review_queue_ids, sliding_windows, target_text)


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

    def test_sliding_windows_overlap_and_fill_toward_the_maximum(self):
        rows = [{"id": f"1:{i}", "start": (i - 1) * 5., "end": i * 5.} for i in range(1, 9)]
        # Greedy closes at the minimum; sliding fills to the cap and overlaps.
        self.assertEqual([len(w) for w in contiguous_windows(rows, 18., 30.)], [4, 4])
        windows = sliding_windows(rows, 18., 30., stride=.5)
        self.assertEqual([[r["id"] for r in w] for w in windows],
                         [["1:1", "1:2", "1:3", "1:4", "1:5", "1:6"],
                          ["1:4", "1:5", "1:6", "1:7", "1:8"]])
        self.assertEqual(len(sliding_windows(rows, 18., 30., stride=1.)), 1)
        with self.assertRaises(ValueError):
            sliding_windows(rows, 18., 30., stride=0.)

    def test_sliding_windows_never_bridge_a_gap(self):
        rows = [{"id": f"1:{i}", "start": (i - 1) * 5., "end": i * 5.} for i in range(1, 9)]
        rows[4]["start"] += .5
        self.assertEqual([len(r) for r in contiguous_runs(rows)], [4, 4])
        self.assertTrue(all(len({r["id"].split(":")[0] for r in w}) == 1
                            for w in sliding_windows(rows, 15., 30., stride=.5)))

    def test_language_tags_survive_the_qwen_normalizer_unchanged(self):
        self.assertEqual(language_tag("ind", False), "Indonesian")
        self.assertEqual(language_tag("javind", False), "None")
        self.assertEqual([language_tag(x, True) for x in ("jav", "ind", "javind")],
                         ["Jav", "Ind", "Javind"])
        # qwen-asr normalizes a forced language to first-upper/rest-lower, so a
        # tag that does not survive it would not match the trained prefix.
        for tag in ("Jav", "Ind", "Javind"):
            self.assertEqual(tag[:1].upper() + tag[1:].lower(), tag)
        with self.assertRaises(ValueError):
            language_tag("unknown", True)

    def test_review_queue_filter_selects_only_requested_training_flags(self):
        path = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "q.csv"
        path.write_text("dataset,id,flags\n"
                        "train,1:1,mostly_silent\n"
                        "train,1:2,very_quiet\n"
                        "train,1:3,low_energy_contrast;low_token_rate\n"
                        "dev,x.mp3,mostly_silent\n")
        self.assertEqual(review_queue_ids(path, ["mostly_silent", "low_token_rate"]), {"1:1", "1:3"})
        self.assertEqual(review_queue_ids(path, []), set())

    def test_jember_style_does_not_rewrite_development_accents_or_mid_sentence_case(self):
        self.assertEqual(target_text("Wè met Émile.", "jember", "dev_style"), "we met Émile")
        self.assertEqual(target_text("Wè met Émile.", "development", "dev_style"), "wè met Émile")
        self.assertEqual(target_text("Wè met Émile.", "jember", "raw"), "wè met Émile")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for tensor regression")
class SpecAugmentTests(unittest.TestCase):
    def _augment(self, **kwargs):
        from scripts.train_recovery import spec_augment
        return spec_augment(**kwargs)

    def test_masks_only_inside_real_frames_and_respects_probability(self):
        import torch
        # (batch, mel, frames): the mel axis is the smaller of the two.
        features = torch.ones(4, 8, 100)
        lengths = [40, 40, 40, 40]
        generator = torch.Generator().manual_seed(7)
        out = self._augment(features=features.clone(), lengths=lengths, freq_width=3,
                            time_width=.25, probability=1.0, generator=generator)
        self.assertTrue((out[:, :, 40:] == 1).all(), "padding frames must never be masked")
        self.assertTrue((out == 0).any(), "nothing was masked at probability 1")
        untouched = self._augment(features=features.clone(), lengths=lengths, freq_width=3,
                                  time_width=.25, probability=0.0, generator=generator)
        torch.testing.assert_close(untouched, features)

    def test_handles_frames_first_layout_and_rejects_wrong_rank(self):
        import torch
        features = torch.ones(2, 100, 8)  # (batch, frames, mel)
        generator = torch.Generator().manual_seed(7)
        out = self._augment(features=features.clone(), lengths=[50, 50], freq_width=3,
                            time_width=.25, probability=1.0, generator=generator)
        self.assertTrue((out[:, 50:, :] == 1).all())
        with self.assertRaises(ValueError):
            self._augment(features=torch.ones(2, 8), lengths=None, freq_width=3,
                          time_width=.25, probability=1.0, generator=generator)


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
