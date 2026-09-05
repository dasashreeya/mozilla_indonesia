#!/usr/bin/env python3
"""Run a local Whisper baseline and report MDC-normalized WER."""

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdc_asr.wer import ErrorCounts, word_errors  # noqa: E402


def read_rows(path: Path, split: Optional[str], limit: Optional[int]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"audio_filename", "transcript"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("{} must contain audio_filename and transcript".format(path))
    if split is not None:
        if "split" not in rows[0]:
            raise ValueError("--split was given but metadata has no split column")
        rows = [row for row in rows if row["split"] == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No rows selected")
    return rows


def batches(values: Sequence[Dict[str, str]], size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_audio(path: Path, target_rate: int = 16000):
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        audio = resample_poly(audio, target_rate // divisor, rate // divisor).astype(np.float32)
    return audio


def choose_device(torch, requested: str) -> Tuple[str, object]:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    dtype = torch.float16 if requested in {"cuda", "mps"} else torch.float32
    return requested, dtype


def write_predictions(path: Path, rows: Sequence[Dict[str, str]], predictions: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio_filename", "transcript"))
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({"audio_filename": row["audio_filename"], "transcript": prediction})


def evaluate(args: argparse.Namespace) -> ErrorCounts:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    rows = read_rows(args.metadata, args.split, args.limit)
    device, dtype = choose_device(torch, args.device)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(args.model),
        dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    predictions: List[str] = []
    started = time.monotonic()
    for batch_number, batch in enumerate(batches(rows, args.batch_size), start=1):
        audio = [load_audio(args.audio_dir / row["audio_filename"]) for row in batch]
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            # Auto language detection requires Whisper's full 30-second (3000
            # frame) feature window, even when every clip is shorter.
            padding="max_length",
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        generate_kwargs = {"task": "transcribe", "max_new_tokens": args.max_new_tokens}
        if args.language != "auto":
            generate_kwargs["language"] = args.language
        with torch.inference_mode():
            generated = model.generate(
                input_features,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))
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
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "whisper-large-v3")
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
        default=ROOT / "experiments" / "predictions" / "openslr35-whisper-jw.csv",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--language", default="jw", help="Whisper language code/name, or auto")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
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
