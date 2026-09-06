#!/usr/bin/env python3
"""Run a local Qwen3-ASR baseline and report MDC-normalized WER."""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_whisper import batches, read_rows, write_predictions  # noqa: E402
from mdc_asr.wer import ErrorCounts, word_errors  # noqa: E402


def choose_device(torch, requested: str):
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    dtype = torch.bfloat16 if requested.startswith("cuda") else torch.float16
    if requested == "cpu":
        dtype = torch.float32
    return requested, dtype


def evaluate(args: argparse.Namespace) -> ErrorCounts:
    import torch
    from qwen_asr import Qwen3ASRModel

    rows = read_rows(args.metadata, args.split, args.limit)
    device, dtype = choose_device(torch, args.device)
    model = Qwen3ASRModel.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
    )
    predictions: List[str] = []
    started = time.monotonic()
    for batch_number, batch in enumerate(batches(rows, args.batch_size), start=1):
        paths = [str(args.audio_dir / row["audio_filename"]) for row in batch]
        language = None if args.language == "auto" else args.language
        results = model.transcribe(audio=paths, language=language)
        if len(results) != len(batch):
            raise RuntimeError("Model returned {} results for {} clips".format(len(results), len(batch)))
        predictions.extend(" ".join(str(result.text or "").splitlines()).strip() for result in results)
        done = min(batch_number * args.batch_size, len(rows))
        elapsed = time.monotonic() - started
        print("transcribed {}/{} ({:.2f} clips/s)".format(done, len(rows), done / elapsed), file=sys.stderr)

    write_predictions(args.output, rows, predictions)
    total = ErrorCounts()
    for row, prediction in zip(rows, predictions):
        total += word_errors(row["transcript"], prediction)
    elapsed = time.monotonic() - started
    print(
        "WER={:.6f} S={} D={} I={} N={} clips={} device={} language={} runtime_seconds={:.2f}".format(
            total.wer,
            total.substitutions,
            total.deletions,
            total.insertions,
            total.reference_words,
            len(rows),
            device,
            args.language,
            elapsed,
        )
    )
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "qwen3-asr-1.7b")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "subsets" / "openslr35" / "shard0-3000-seed-20260905" / "metadata.csv",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=ROOT / "data" / "subsets" / "openslr35" / "shard0-3000-seed-20260905" / "audio",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "predictions" / "openslr35-qwen-auto.csv",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--language", default="auto", help="Qwen language name, or auto")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda:0"), default="auto")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        evaluate(build_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
