import csv
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_whisper import read_rows  # noqa: E402


class EvaluateWhisperTests(unittest.TestCase):
    def test_filters_split_before_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("audio_filename", "transcript", "split"))
                writer.writeheader()
                writer.writerows(
                    [
                        {"audio_filename": "a.flac", "transcript": "a", "split": "train"},
                        {"audio_filename": "b.flac", "transcript": "b", "split": "validation"},
                        {"audio_filename": "c.flac", "transcript": "c", "split": "validation"},
                    ]
                )
            rows = read_rows(path, "validation", 1)
            self.assertEqual([row["audio_filename"] for row in rows], ["b.flac"])


if __name__ == "__main__":
    unittest.main()
