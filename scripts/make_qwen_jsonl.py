#!/usr/bin/env python3
"""Build Qwen3-ASR train/eval JSONL, optionally as group-held-out folds."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdc_asr.normalization import normalize_text  # noqa: E402


AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".webm"}


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
        rows = list(csv.DictReader(handle, dialect=dialect))
    if not rows:
        raise ValueError("No metadata rows found")
    required = {"audio_filename", "transcript"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError("Metadata is missing: {}".format(", ".join(sorted(missing))))
    return rows


def index_audio(root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            if path.name in result:
                raise ValueError("Duplicate audio basename: {}".format(path.name))
            result[path.name] = path.resolve()
    return result


def qwen_target(row: Mapping[str, str], apply_normalizer: bool) -> str:
    transcript = row.get("transcript", "") or ""
    if apply_normalizer:
        transcript = normalize_text(transcript).strip()
    label = (row.get("language", "") or "").strip().lower()
    language = "Indonesian" if label in {"ind", "id", "id-id", "indonesian"} else "None"
    return "language {}<asr_text>{}".format(language, transcript)


def safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return label or "missing"


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, str]],
    audio: Mapping[str, Path],
    apply_normalizer: bool,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            filename = row.get("audio_filename", "")
            if filename not in audio or not (row.get("transcript", "") or "").strip():
                continue
            json.dump(
                {"audio": str(audio[filename]), "text": qwen_target(row, apply_normalizer)},
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")
            written += 1
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("audio_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fold-column", help="Create one train/eval pair per value, e.g. convo_id")
    parser.add_argument("--normalize-target", action="store_true", help="Ablation: train on MDC-normalized targets")
    args = parser.parse_args(argv)

    rows = read_rows(args.metadata)
    audio = index_audio(args.audio_dir)
    if args.fold_column:
        if args.fold_column not in rows[0]:
            raise ValueError("Fold column is missing: {}".format(args.fold_column))
        groups = sorted({row.get(args.fold_column, "") or "<missing>" for row in rows})
        if len(groups) < 2:
            raise ValueError("Fold column must contain at least two groups")
        receipt = {"fold_column": args.fold_column, "folds": {}}
        for group in groups:
            label = safe_label(group)
            train = [row for row in rows if (row.get(args.fold_column, "") or "<missing>") != group]
            evaluate = [row for row in rows if (row.get(args.fold_column, "") or "<missing>") == group]
            train_count = write_jsonl(args.output_dir / (label + "_train.jsonl"), train, audio, args.normalize_target)
            eval_count = write_jsonl(args.output_dir / (label + "_eval.jsonl"), evaluate, audio, args.normalize_target)
            receipt["folds"][group] = {"train": train_count, "eval": eval_count}
    else:
        count = write_jsonl(args.output_dir / "all.jsonl", rows, audio, args.normalize_target)
        receipt = {"all": count}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "JSONL_RECEIPT.json").open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)

