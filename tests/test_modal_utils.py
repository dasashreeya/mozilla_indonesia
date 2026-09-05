import json
import tempfile
import unittest
from pathlib import Path

from mdc_asr.modal_utils import latest_checkpoint, rebase_qwen_jsonl, safe_volume_path


class ModalUtilsTests(unittest.TestCase):
    def test_safe_volume_path_accepts_relative_inside_root(self):
        self.assertEqual(
            safe_volume_path(Path("/vol"), "models/qwen3-asr-1.7b"),
            Path("/vol/models/qwen3-asr-1.7b"),
        )

    def test_safe_volume_path_rejects_escape(self):
        with self.assertRaises(ValueError):
            safe_volume_path(Path("/vol"), "../secret")
        with self.assertRaises(ValueError):
            safe_volume_path(Path("/vol"), "/tmp/secret")

    def test_latest_checkpoint_uses_highest_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-5").mkdir()
            (root / "checkpoint-20").mkdir()
            (root / "not-a-checkpoint").mkdir()
            self.assertEqual(latest_checkpoint(root), root / "checkpoint-20")

    def test_rebase_qwen_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "a.flac").write_bytes(b"a")
            source = root / "source.jsonl"
            source.write_text(
                json.dumps({"audio": "/old/machine/a.flac", "text": "language None<asr_text>aku"})
                + "\n",
                encoding="utf-8",
            )
            destination = root / "rebased.jsonl"
            result = rebase_qwen_jsonl(source, destination, audio_dir)
            self.assertEqual(result, {"written": 1, "missing_audio": 0})
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(payload["audio"], str(audio_dir / "a.flac"))


if __name__ == "__main__":
    unittest.main()
