#!/usr/bin/env python3
"""Segment Jember session MP3s from TSV timestamps into ASR training utterances."""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".webm"}


def normalized_key(value: str) -> str:
    return "_".join(value.strip().lower().replace("/", " ").replace("-", " ").split())


def canonical_row(row: Mapping[str, str]) -> Dict[str, str]:
    return {normalized_key(key): value for key, value in row.items() if key is not None}


def first_value(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return ""


def parse_timestamp(value: str) -> float:
    """Parse H:MM:SS(.sss), MM:SS(.sss), or raw seconds."""

    parts = value.strip().replace(",", ".").split(":")
    try:
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            seconds = float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError("Invalid timestamp: {!r}".format(value)) from exc
    if seconds < 0:
        raise ValueError("Timestamp must be non-negative: {!r}".format(value))
    return seconds


def find_tsv(root: Path) -> Path:
    for path in sorted(root.rglob("*.tsv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            keys = {normalized_key(key) for key in (reader.fieldnames or [])}
        if keys.intersection({"start", "start_time"}) and keys.intersection({"end", "end_time"}):
            return path
    raise ValueError("No timestamp TSV found below {}".format(root))


def audio_by_stem(root: Path) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            result.setdefault(path.stem, path)
            result.setdefault(path.name, path)
    return result


def load_jobs(root: Path, limit: Optional[int]) -> List[Dict[str, object]]:
    metadata_path = find_tsv(root)
    audio = audio_by_stem(root)
    jobs: List[Dict[str, object]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, raw_row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
            row = canonical_row(raw_row)
            session = first_value(
                row,
                ("audio_file_name", "audio_filename", "audio_file", "file_name", "filename", "session", "id"),
            )
            start_text = first_value(row, ("start", "start_time"))
            end_text = first_value(row, ("end", "end_time"))
            transcript = first_value(row, ("text", "transcript", "sentence"))
            source = audio.get(session) or audio.get(Path(session).stem)
            if not source or not start_text or not end_text or not transcript:
                continue
            start = parse_timestamp(start_text)
            end = parse_timestamp(end_text)
            if end <= start:
                continue
            jobs.append(
                {
                    "row_number": row_number,
                    "session": Path(session).stem,
                    "source": source,
                    "start": start,
                    "duration": end - start,
                    "transcript": transcript,
                }
            )
            if limit is not None and len(jobs) >= limit:
                break
    if not jobs:
        raise ValueError("No usable timestamp rows matched an audio session")
    return jobs


def segment_one(ffmpeg: str, clips: Path, job: Mapping[str, object]) -> Tuple[str, Dict[str, object]]:
    filename = "{}_{:06d}.wav".format(job["session"], job["row_number"])
    output = clips / filename
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "{:.3f}".format(job["start"]),
            "-i",
            str(job["source"]),
            "-t",
            "{:.3f}".format(job["duration"]),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-y",
            str(output),
        ],
        check=True,
    )
    return filename, dict(job)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--limit", type=int, help="Maximum utterances, for pipeline smoke tests")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError("ffmpeg is required but was not found on PATH")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    if args.destination.exists() and any(args.destination.iterdir()):
        raise ValueError("Destination is not empty: {}".format(args.destination))
    clips = args.destination / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    jobs = load_jobs(args.source, args.limit)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        completed = list(executor.map(lambda job: segment_one(ffmpeg, clips, job), jobs))

    metadata_path = args.destination / "metadata.tsv"
    jsonl_path = args.destination / "train.jsonl"
    with metadata_path.open("w", encoding="utf-8", newline="") as metadata, jsonl_path.open(
        "w", encoding="utf-8"
    ) as jsonl:
        writer = csv.DictWriter(
            metadata,
            fieldnames=("audio_filename", "speaker", "transcript", "language", "convo_id"),
            delimiter="\t",
        )
        writer.writeheader()
        for filename, job in completed:
            writer.writerow(
                {
                    "audio_filename": filename,
                    "speaker": job["session"],
                    "transcript": job["transcript"],
                    "language": "unknown",
                    "convo_id": job["session"],
                }
            )
            json.dump(
                {
                    "audio": str((clips / filename).resolve()),
                    "text": "language None<asr_text>{}".format(job["transcript"]),
                },
                jsonl,
                ensure_ascii=False,
            )
            jsonl.write("\n")
    print("Segmented {} utterances into {}".format(len(completed), args.destination))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
