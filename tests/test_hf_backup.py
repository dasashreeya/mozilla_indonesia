import tempfile
import unittest
from pathlib import Path

from mdc_asr.hf_backup import BackupItem, default_local_items, parse_item_specs


class HfBackupTests(unittest.TestCase):
    def test_parse_item_specs(self):
        self.assertEqual(
            parse_item_specs(["artifacts/a.zip=submissions/a.zip"]),
            [BackupItem(Path("artifacts/a.zip"), "submissions/a.zip")],
        )

    def test_default_local_items_excludes_raw_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("readme", encoding="utf-8")
            (root / "data" / "downloads").mkdir(parents=True)
            (root / "data" / "downloads" / "raw.zip").write_bytes(b"raw")
            (root / "artifacts").mkdir()
            (root / "artifacts" / "submission.zip").write_bytes(b"zip")
            items = default_local_items(root)
            paths = {item.path_in_repo for item in items}
            self.assertIn("source/README.md", paths)
            self.assertIn("artifacts", paths)
            self.assertNotIn("data/downloads", paths)

    def test_default_local_items_can_skip_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("readme", encoding="utf-8")
            (root / "artifacts").mkdir()
            items = default_local_items(root, include_artifacts=False)
            self.assertNotIn("artifacts", {item.path_in_repo for item in items})


if __name__ == "__main__":
    unittest.main()
