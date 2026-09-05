import csv
import tempfile
import unittest
from pathlib import Path

from mdc_asr.wer import ErrorCounts, corpus_errors, score_files, word_errors


class WordErrorRateTests(unittest.TestCase):
    def test_substitution_and_insertion(self):
        counts = word_errors("aku mangan sega", "aku nedha sega maneh")
        self.assertEqual(counts.errors, 2)
        self.assertEqual(counts.reference_words, 3)
        self.assertAlmostEqual(counts.wer, 2 / 3)

    def test_normalization_applies_before_alignment(self):
        counts = word_errors("Halo, [laugh] kabeh!", "halo kabeh")
        self.assertEqual(counts, ErrorCounts(reference_words=2))

    def test_score_aligns_by_filename_and_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth = root / "truth.csv"
            prediction = root / "prediction.csv"
            with truth.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("audio_filename", "transcript", "language"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"audio_filename": "a.mp3", "transcript": "aku lunga", "language": "jav"},
                        {"audio_filename": "b.mp3", "transcript": "saya pergi", "language": "ind"},
                    ]
                )
            with prediction.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("audio_filename", "transcript"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"audio_filename": "b.mp3", "transcript": "saya pergi"},
                        {"audio_filename": "a.mp3", "transcript": "aku"},
                    ]
                )
            scores = score_files(truth, prediction, "language")
            self.assertEqual(scores["all"].deletions, 1)
            self.assertEqual(scores["all"].reference_words, 4)
            self.assertEqual(scores["jav"].wer, 0.5)
            self.assertEqual(scores["ind"].wer, 0.0)


if __name__ == "__main__":
    unittest.main()

