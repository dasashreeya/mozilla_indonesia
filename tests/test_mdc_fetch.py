import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.mdc_fetch import MDCError, _safe_member_path, extract_subset, resolve_dataset


class MDCFetchTests(unittest.TestCase):
    def test_alias_resolution(self):
        config = {"datasets": {"dev": {"id": "c123456", "name": "Development"}}}
        self.assertEqual(resolve_dataset("dev", config)["id"], "c123456")
        self.assertEqual(resolve_dataset("c123456", config)["alias"], "dev")

    def test_rejects_parent_archive_path(self):
        with self.assertRaises(MDCError):
            _safe_member_path(Path("/tmp/safe"), "../escape")

    def test_deterministic_subset_keeps_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sample.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for name, contents in {
                    "dataset/clips/a.mp3": b"a",
                    "dataset/clips/b.mp3": b"b",
                    "dataset/clips/c.mp3": b"c",
                    "dataset/metadata.tsv": b"audio_filename\ttranscript\na.mp3\ta\n",
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(contents)
                    archive.addfile(info, io.BytesIO(contents))
            output = root / "out"
            manifest_path = extract_subset(
                archive_path,
                output,
                max_audio_files=2,
                seed=17,
                dataset={"id": "c123456", "alias": "sample", "name": "Sample"},
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["audio_files_in_archive"], 3)
            self.assertEqual(manifest["audio_files_extracted"], 2)
            self.assertTrue((output / "dataset" / "metadata.tsv").exists())
            extracted_audio = list((output / "dataset" / "clips").glob("*.mp3"))
            self.assertEqual(len(extracted_audio), 2)


if __name__ == "__main__":
    unittest.main()

