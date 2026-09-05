import tempfile
import unittest
import zipfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_openslr35 import prepare  # noqa: E402


class PrepareOpenSlr35Tests(unittest.TestCase):
    def test_deterministic_speaker_disjoint_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "data.zip"
            transcript_path = root / "utt_spk_text.tsv"
            rows = []
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("asr_javanese/LICENSE", "test license")
                for speaker_index in range(4):
                    for utterance_index in range(3):
                        utterance_id = "{:02x}{:08x}".format(speaker_index, utterance_index)
                        archive.writestr("asr_javanese/data/00/{}.flac".format(utterance_id), b"audio")
                        rows.append("{}\tspk{}\ttext {}\n".format(utterance_id, speaker_index, utterance_id))
            transcript_path.write_text("".join(rows), encoding="utf-8")

            first = prepare(archive_path, transcript_path, root / "out1", 8, 0.25, 42, None)
            second = prepare(archive_path, transcript_path, root / "out2", 8, 0.25, 42, None)
            self.assertEqual(first["train_speakers"], second["train_speakers"])
            self.assertEqual(first["validation_speakers"], second["validation_speakers"])
            self.assertEqual(first["speaker_overlap"], [])
            self.assertEqual(first["train_utterances"] + first["validation_utterances"], 8)
            self.assertEqual(
                (root / "out1" / "train.qwen.jsonl").read_text(encoding="utf-8"),
                (root / "out2" / "train.qwen.jsonl").read_text(encoding="utf-8").replace(
                    str(root / "out2"), str(root / "out1")
                ),
            )


if __name__ == "__main__":
    unittest.main()
