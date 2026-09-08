import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_submission", ROOT / "scripts" / "build_submission.py"
)
assert SPEC and SPEC.loader
BUILD_SUBMISSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD_SUBMISSION)


class BuildSubmissionTests(unittest.TestCase):
    def test_checkpoint_training_state_is_not_packaged(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source"
            model = root / "checkpoint"
            source.mkdir()
            model.mkdir()
            (source / "main.py").write_text("print('ok')\n", encoding="utf-8")
            (source / "__pycache__").mkdir()
            (source / "__pycache__/main.cpython-312.pyc").write_bytes(b"old bytecode")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            for name in BUILD_SUBMISSION.TRAINING_ONLY_FILENAMES:
                (model / name).write_bytes(b"training state")

            output = root / "submission.zip"
            result = BUILD_SUBMISSION.main(
                ["--source", str(source), "--model-dir", str(model), "--output", str(output)]
            )

            self.assertEqual(result, 0)
            with zipfile.ZipFile(output) as archive:
                names = set(archive.namelist())
            self.assertIn("main.py", names)
            self.assertIn("model/config.json", names)
            self.assertIn("model/model.safetensors", names)
            self.assertFalse(any("__pycache__" in name for name in names))
            self.assertTrue(
                all("model/" + name not in names for name in BUILD_SUBMISSION.TRAINING_ONLY_FILENAMES)
            )


if __name__ == "__main__":
    unittest.main()
