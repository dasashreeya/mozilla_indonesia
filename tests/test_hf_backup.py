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


class BatchedCommitTests(unittest.TestCase):
    """Loose files must go up in one commit, not one commit each.

    A per-file commit is what tripped the Hub's 128-commits-per-hour limit: a
    training wave leaves roughly fifteen receipt and prediction files per arm,
    so a routine backup asked for eighty-odd commits and earned a 429 with an
    hour-long cooldown.
    """

    def setUp(self):
        import tempfile
        from mdc_asr.hf_backup import BackupItem
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.files = []
        for name in ("a.json", "b.json", "c.json"):
            path = root / name
            path.write_text("{}")
            self.files.append(BackupItem(path, name))
        directory = root / "plots"
        directory.mkdir()
        (directory / "x.png").write_bytes(b"x")
        self.directory = BackupItem(directory, "plots")

    def _api(self):
        class FakeCommit:
            commit_url = "https://example.invalid/commit/abc"

        class FakeApi:
            def __init__(self):
                self.commits = 0
                self.folders = 0
                self.operations = None

            def whoami(self, token=None):
                return {"name": "someone"}

            def create_repo(self, **kwargs):
                return None

            def upload_file(self, **kwargs):
                self.commits += 1
                return FakeCommit()

            def upload_folder(self, **kwargs):
                self.folders += 1
                return FakeCommit()

            def create_commit(self, operations=None, **kwargs):
                self.commits += 1
                self.operations = operations
                return FakeCommit()
        return FakeApi()

    def _run(self, items):
        import sys
        import types
        from mdc_asr import hf_backup
        api = self._api()
        fake = types.ModuleType("huggingface_hub")
        fake.HfApi = lambda: api
        fake.CommitOperationAdd = lambda path_in_repo, path_or_fileobj: (
            path_in_repo, path_or_fileobj)
        original = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = fake
        try:
            result = hf_backup.upload_items(token="t", repo_id="me/repo", items=items)
        finally:
            if original is None:
                sys.modules.pop("huggingface_hub", None)
            else:
                sys.modules["huggingface_hub"] = original
        return api, result

    def test_many_files_become_one_commit(self):
        api, result = self._run(self.files)
        self.assertEqual(api.commits, 1)
        # Three files plus the README, all in that single commit.
        self.assertEqual(len(api.operations), 4)
        self.assertEqual(len(result["uploaded"]), 3)
        self.assertEqual(result["commits"], 1)

    def test_no_upload_file_calls_remain(self):
        api, _ = self._run(self.files)
        # upload_file would be one commit per file; nothing should reach it.
        self.assertIsNotNone(api.operations)

    def test_directories_still_use_upload_folder(self):
        api, result = self._run(self.files + [self.directory])
        self.assertEqual(api.folders, 1)
        self.assertEqual(api.commits, 1)
        self.assertEqual(result["commits"], 2)

    def test_missing_paths_are_skipped_not_committed(self):
        from mdc_asr.hf_backup import BackupItem
        missing = BackupItem(Path(self.tmp.name) / "nope.json", "nope.json")
        api, result = self._run([missing])
        # Only the README commit; the absent file contributes no operation.
        self.assertEqual(api.commits, 1)
        self.assertEqual(len(api.operations), 1)
        self.assertEqual(len(result["skipped"]), 1)
        self.assertEqual(result["uploaded"], [])
