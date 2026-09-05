import csv
import json
import tempfile
import unittest
from pathlib import Path

from submission_src import main as submission


class SubmissionTests(unittest.TestCase):
    def test_backend_auto_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
            old_source = submission.SOURCE_DIR
            try:
                submission.SOURCE_DIR = root
                self.assertEqual(
                    submission.detect_backend({"backend": "auto", "model_path": "model"}),
                    "whisper_transformers",
                )
            finally:
                submission.SOURCE_DIR = old_source

    def test_manifest_and_quoted_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            with (data / "test_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("audio_filename", "language"))
                writer.writeheader()
                writer.writerow({"audio_filename": "a.mp3", "language": "javind"})
            old_data, old_output = submission.DATA_DIR, submission.OUTPUT_PATH
            try:
                submission.DATA_DIR = data
                submission.OUTPUT_PATH = root / "submission" / "submission.csv"
                rows = submission.read_manifest()
                submission.write_submission(rows, ['aku, "lunga"'])
                with submission.OUTPUT_PATH.open("r", newline="", encoding="utf-8") as handle:
                    output = list(csv.DictReader(handle))
                self.assertEqual(output[0]["audio_filename"], "a.mp3")
                self.assertEqual(output[0]["transcript"], 'aku, "lunga"')
            finally:
                submission.DATA_DIR, submission.OUTPUT_PATH = old_data, old_output


if __name__ == "__main__":
    unittest.main()
