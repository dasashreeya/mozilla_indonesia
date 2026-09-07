"""Regression tests for checkpoint averaging and the evaluation harness."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from scripts.eval_checkpoint import QWEN_LANGUAGES, WHISPER_LANGUAGES, detect_backend, load_rows


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _model(self, model_type):
        (self.root / model_type).mkdir()
        (self.root / model_type / "config.json").write_text(json.dumps({"model_type": model_type}))
        return self.root / model_type

    def test_backend_detection_matches_the_submission_entrypoint(self):
        self.assertEqual(detect_backend(self._model("qwen3_asr")), "qwen")
        self.assertEqual(detect_backend(self._model("whisper")), "whisper")
        with self.assertRaises(ValueError):
            detect_backend(self._model("wav2vec2"))

    def test_every_competition_label_maps_to_a_token_for_both_backends(self):
        for label in ("jav", "ind", "javind"):
            self.assertIn(label, QWEN_LANGUAGES)
            self.assertIn(label, WHISPER_LANGUAGES)
        # Whisper's Javanese token is the reason that backend is worth measuring.
        self.assertEqual(WHISPER_LANGUAGES["jav"], "jw")

    def test_split_selection_and_limit(self):
        (self.root / "metadata.csv").write_text(
            "audio_filename,transcript,language,split,source,duration_seconds\n"
            "a.flac,x,jav,train,jember,5\n"
            "b.flac,y,jav,validation,jember,25\n"
            "c.flac,z,javind,validation,development,26\n")
        self.assertEqual(len(load_rows(self.root, "validation", 0)), 2)
        self.assertEqual(len(load_rows(self.root, "validation", 1)), 1)
        with self.assertRaises(ValueError):
            load_rows(self.root, "nonexistent", 0)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("safetensors"),
                     "torch and safetensors required for weight averaging")
class SoupTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _checkpoint(self, name, value, extra=None):
        import torch
        from safetensors.torch import save_file

        directory = self.root / name
        directory.mkdir()
        tensors = {"a": torch.full((2, 3), value, dtype=torch.bfloat16)}
        if extra:
            tensors.update(extra)
        save_file(tensors, str(directory / "model.safetensors"), metadata={"format": "pt"})
        (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_asr"}))
        return directory

    def test_uniform_and_weighted_averages_round_trip_through_bfloat16(self):
        import torch
        from safetensors.torch import load_file

        from scripts.model_soup import average_state_dicts

        members = [self._checkpoint("one", 1.0), self._checkpoint("two", 3.0)]
        receipt = average_state_dicts(members, self.root / "uniform")
        self.assertEqual(receipt["parameters"], 1)
        averaged = load_file(str(self.root / "uniform" / "model.safetensors"))["a"]
        self.assertEqual(averaged.dtype, torch.bfloat16)
        torch.testing.assert_close(averaged.float(), torch.full((2, 3), 2.0))
        # Config files travel with the weights so the soup is loadable as-is.
        self.assertTrue((self.root / "uniform" / "config.json").is_file())

        average_state_dicts(members, self.root / "weighted", weights=[0.25, 0.75])
        weighted = load_file(str(self.root / "weighted" / "model.safetensors"))["a"]
        torch.testing.assert_close(weighted.float(), torch.full((2, 3), 2.5))

    def test_rejects_mismatched_members_and_bad_weights(self):
        import torch

        from scripts.model_soup import average_state_dicts

        one = self._checkpoint("one", 1.0)
        wide = self._checkpoint("wide", 1.0, extra={"b": torch.ones(4, dtype=torch.bfloat16)})
        with self.assertRaises(ValueError):
            average_state_dicts([one], self.root / "single")
        with self.assertRaises(ValueError):
            average_state_dicts([one, wide], self.root / "unseen")
        with self.assertRaises(ValueError):
            average_state_dicts([wide, one], self.root / "missing")
        with self.assertRaises(ValueError):
            average_state_dicts([one, self._checkpoint("two", 3.0)], self.root / "bad",
                                weights=[0.5, 0.9])


if __name__ == "__main__":
    unittest.main()
