import unittest

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_qwen import choose_device  # noqa: E402


class _Backends:
    @staticmethod
    def is_available():
        return False


class _Torch:
    cuda = _Backends()

    class backends:
        mps = _Backends()

    bfloat16 = "bf16"
    float16 = "f16"
    float32 = "f32"


class EvaluateQwenTests(unittest.TestCase):
    def test_cpu_auto_dtype(self):
        self.assertEqual(choose_device(_Torch(), "auto"), ("cpu", "f32"))


if __name__ == "__main__":
    unittest.main()
