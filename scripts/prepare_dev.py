#!/usr/bin/env python3
"""Convert extracted competition dev data to the official local runtime layout."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".webm"}
REQUIRED_COLUMNS = {"audio_filename", "transcript", "language", "speaker", "convo_id"}


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("No metadata rows found in {}".format(path))
    missing = REQUIRED_COLUMNS.difference(rows[0])
    if missing:
        raise ValueError("{} is missing columns: {}".format(path, ", ".join(sorted(missing))))
    return rows


def find_metadata(root: Path) -> Path:
    candidates = sorted(root.rglob("metadata.tsv"))
    for candidate in candidates:
        try:
            rows = read_tsv(candidate)
        except (OSError, ValueError):
            continue
        if rows:
            return candidate
    raise ValueError("Could not find competition metadata.tsv below {}".format(root))


def audio_index(root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            if path.name in result:
                raise ValueError("Duplicate audio basename: {}".format(path.name))
            result[path.name] = path
    return result


def duration_seconds(path: Path) -> int:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    process = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(round(float(process.stdout.strip())))


def place_file(source: Path, target: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()))
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Extracted dev archive or subset root")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    args = parser.parse_args(argv)

    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError("Destination is not empty: {}".format(args.destination))
    metadata_path = find_metadata(args.source)
    rows = read_tsv(metadata_path)
    available = audio_index(args.source)
    selected = [row for row in rows if row["audio_filename"] in available]
    if not selected:
        raise ValueError("No metadata rows match available audio")

    clips = args.destination / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    durations: Dict[str, int] = {}
    for row in selected:
        filename = row["audio_filename"]
        source = available[filename]
        place_file(source, clips / filename, args.mode)
        durations[filename] = duration_seconds(source)

    truth_rows = [
        {
            "audio_filename": row["audio_filename"],
            "transcript": row["transcript"],
            "language": row["language"],
            "speaker": row["speaker"],
            "convo_id": row["convo_id"],
        }
        for row in selected
    ]
    manifest_rows = [
        {
            "audio_filename": row["audio_filename"],
            "file_duration_seconds": durations[row["audio_filename"]],
            "language": row["language"],
        }
        for row in selected
    ]
    format_rows = [{"audio_filename": row["audio_filename"], "transcript": ""} for row in selected]
    write_csv(
        args.destination / "ground_truth.csv",
        ("audio_filename", "transcript", "language", "speaker", "convo_id"),
        truth_rows,
    )
    write_csv(
        args.destination / "test_metadata.csv",
        ("audio_filename", "file_duration_seconds", "language"),
        manifest_rows,
    )
    write_csv(args.destination / "submission_format.csv", ("audio_filename", "transcript"), format_rows)
    with (args.destination / "PREPARATION_RECEIPT.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(args.source.resolve()),
                "metadata": str(metadata_path.resolve()),
                "rows": len(selected),
                "missing_audio_rows": len(rows) - len(selected),
                "placement_mode": args.mode,
                "duration_zero_means": "ffprobe was unavailable or reported a rounded sub-second clip",
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    if not shutil.which("ffprobe"):
        print("warning: ffprobe not found; durations were written as 0", file=sys.stderr)
    print("Prepared {} clips in {}".format(len(selected), args.destination))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)

