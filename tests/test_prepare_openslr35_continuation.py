import csv
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_openslr35 import sha256_file  # noqa: E402
from prepare_openslr35_continuation import prepare_continuation  # noqa: E402


class PrepareOpenSlr35ContinuationTests(unittest.TestCase):
    def test_excludes_fixed_validation_speakers(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            archive_path = root / "asr_javanese_1.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("data/train-one.flac", b"one")
                archive.writestr("data/train-two.flac", b"two")
                archive.writestr("data/held-out.flac", b"held")
            transcripts = root / "utt_spk_text.tsv"
            transcripts.write_text(
                "train-one\ttrain-speaker\tone\n"
                "train-two\tother-speaker\ttwo\n"
                "held-out\tvalidation-speaker\theld\n",
                encoding="utf-8",
            )
            checksums = root / "asr_javanese.sha256"
            checksums.write_text(
                "{}  {}\n".format(sha256_file(archive_path), archive_path.name),
                encoding="utf-8",
            )
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("speaker_id", "split"))
                writer.writeheader()
                writer.writerow({"speaker_id": "validation-speaker", "split": "validation"})

            manifest = prepare_continuation(
                archives_dir=root,
                transcript_path=transcripts,
                checksums_path=checksums,
                exclude_metadata=metadata,
                output_dir=root / "out",
                shards=("1",),
            )

            self.assertEqual(manifest["train_utterances"], 2)
            self.assertEqual(manifest["skipped_validation_speaker_utterances"], 1)
            self.assertEqual(manifest["speaker_overlap"], [])
            self.assertFalse((root / "out" / "audio" / "held-out.flac").exists())


if __name__ == "__main__":
    unittest.main()
