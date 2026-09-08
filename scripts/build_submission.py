#!/usr/bin/env python3
"""Build a ZIP64 offline submission with main.py at the archive root."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
STORE_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".mp3", ".zip", ".gz"}
TRAINING_ONLY_FILENAMES = {
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
}


def files_below(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink():
                raise ValueError("Refusing to package symlink: {}".format(path))
            yield path


def inference_model_files(root: Path) -> Iterable[Path]:
    """Yield model assets while omitting Trainer state that inference cannot use."""
    for path in files_below(root):
        if path.name not in TRAINING_ONLY_FILENAMES:
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "submission_src")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "submission.zip")
    args = parser.parse_args(argv)

    if not (args.source / "main.py").is_file():
        raise ValueError("{} must contain main.py".format(args.source))
    if not (args.model_dir / "config.json").is_file():
        raise ValueError("Model directory lacks config.json: {}".format(args.model_dir))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    entries: Sequence[Tuple[Path, str]] = list(
        [(path, path.relative_to(args.source).as_posix()) for path in files_below(args.source)]
        + [
            (path, "model/" + path.relative_to(args.model_dir).as_posix())
            for path in inference_model_files(args.model_dir)
        ]
    )
    with zipfile.ZipFile(args.output, "w", allowZip64=True) as archive:
        for path, archive_name in entries:
            compression = zipfile.ZIP_STORED if path.suffix.lower() in STORE_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, archive_name, compress_type=compression)

    with zipfile.ZipFile(args.output) as archive:
        names = set(archive.namelist())
        if "main.py" not in names or not any(name.startswith("model/") for name in names):
            raise ValueError("Built archive failed layout validation")
    receipt = {
        "archive": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "entries": len(entries),
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".json")
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
