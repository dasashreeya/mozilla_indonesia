import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.make_qwen_jsonl import main, qwen_target


class MakeQwenJsonlTests(unittest.TestCase):
    def test_language_prefix(self):
        self.assertEqual(
            qwen_target({"language": "ind", "transcript": "Saya pergi."}, False),
            "language Indonesian<asr_text>Saya pergi.",
        )
        self.assertEqual(
            qwen_target({"language": "javind", "transcript": "Aku pergi."}, False),
            "language None<asr_text>Aku pergi.",
        )

    def test_conversation_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "clips"
            clips.mkdir()
            (clips / "a.mp3").write_bytes(b"a")
            (clips / "b.mp3").write_bytes(b"b")
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("audio_filename", "transcript", "language", "convo_id"),
                )
                writer.writeheader()
                writer.writerow(
                    {"audio_filename": "a.mp3", "transcript": "aku", "language": "jav", "convo_id": "A"}
                )
                writer.writerow(
                    {"audio_filename": "b.mp3", "transcript": "saya", "language": "ind", "convo_id": "B"}
                )
            output = root / "folds"
            self.assertEqual(main([str(metadata), str(clips), str(output), "--fold-column", "convo_id"]), 0)
            receipt = json.loads((output / "JSONL_RECEIPT.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["folds"]["A"], {"train": 1, "eval": 1})
            self.assertEqual(receipt["folds"]["B"], {"train": 1, "eval": 1})


if __name__ == "__main__":
    unittest.main()

