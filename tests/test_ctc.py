import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.train_ctc import read_split, summarize, target_text, training_vocabulary


class CTCTests(unittest.TestCase):
    def test_targets_preserve_mid_sentence_case_and_accents(self):
        self.assertEqual(target_text("Aku menyang Jember [laugh] karo caf\u00e9!"),
                         "aku menyang Jember karo caf\u00e9")

    def test_vocabulary_excludes_unseen_validation_characters(self):
        vocab = training_vocabulary(["training caf\u00e9"])
        self.assertIn("\u00e9", vocab)
        self.assertNotIn("\u03bb", vocab)
        self.assertEqual(vocab["[PAD]"], 0)
        self.assertEqual(len(vocab), len(set(vocab.values())))

    def test_metrics_use_reference_words_and_preserve_case(self):
        rows = [{"transcript": "aku Jember", "language": "jav", "session": "a"},
                {"transcript": "ini itu sana sini", "language": "javind", "session": "b"}]
        result = summarize(rows, ["aku jember", "ini itu sana sini"])
        self.assertAlmostEqual(result["wer"], 1 / 6)
        self.assertAlmostEqual(result["selection_wer"], 0.49 / 2)
        self.assertEqual(result["reference_words"], 6)

    def test_session_overlap_is_rejected_even_with_distinct_filenames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fields = ["audio_filename", "split", "session", "source", "speaker"]
            with (root / "metadata.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    dict(zip(fields, ["a.flac", "train", "same", "jember", ""])),
                    dict(zip(fields, ["b.flac", "validation", "same", "jember", ""]))])
            receipt = {"validation_rows": 1, "metadata_sha256":
                       hashlib.sha256((root / "metadata.csv").read_bytes()).hexdigest()}
            (root / "PREPARATION_RECEIPT.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "overlap in session"):
                read_split(root)


if __name__ == "__main__":
    unittest.main()
