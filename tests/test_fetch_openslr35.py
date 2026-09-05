import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "fetch_openslr35", ROOT / "scripts" / "fetch_openslr35.py"
)
assert SPEC and SPEC.loader
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class FetchOpenSlr35Tests(unittest.TestCase):
    def test_parse_shard_ranges_and_deduplicates(self):
        self.assertEqual(FETCH.parse_shards("0,2-4,3"), ("0", "2", "3", "4"))
        self.assertEqual(FETCH.parse_shards("all"), tuple(format(i, "x") for i in range(16)))

    def test_load_checksums(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "checksums"
            path.write_text("{}  one.zip\n".format("a" * 64), encoding="utf-8")
            self.assertEqual(FETCH.load_checksums(path), {"one.zip": "a" * 64})

    def test_rejects_invalid_shards(self):
        with self.assertRaises(ValueError):
            FETCH.parse_shards("g")
        with self.assertRaises(ValueError):
            FETCH.parse_shards("f-1")


if __name__ == "__main__":
    unittest.main()
