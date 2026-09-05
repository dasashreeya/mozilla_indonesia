#!/usr/bin/env python3
"""Extract a deterministic, speaker-disjoint training subset from OpenSLR 35."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "downloads" / "openslr35" / "asr_javanese_0.zip"
DEFAULT_TRANSCRIPTS = ROOT / "data" / "downloads" / "openslr35" / "utt_spk_text.tsv"
DEFAULT_OUTPUT = ROOT / "data" / "subsets" / "openslr35" / "shard0-3000-seed-20260905"
EXPECTED_SHA256 = "a871c8b71ff8fa9d95955447ca0c388e8c6f925aecfce92e1880bda2da113bcb"


class PrepareError(RuntimeError):
    """A user-actionable archive or metadata error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_transcripts(path: Path) -> Dict[str, Dict[str, str]]:
    records: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3 or not all(parts):
                raise PrepareError("Malformed transcript row {} in {}".format(line_number, path))
            utterance_id, speaker_id, transcript = parts
            if utterance_id in records:
                raise PrepareError("Duplicate utterance ID: {}".format(utterance_id))
            records[utterance_id] = {
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "transcript": transcript,
            }
    return records


def archive_audio(zip_file: zipfile.ZipFile) -> Dict[str, str]:
    members: Dict[str, str] = {}
    for info in zip_file.infolist():
        member = PurePosixPath(info.filename)
        if info.is_dir() or member.suffix.lower() not in {".flac", ".wav"}:
            continue
        if member.is_absolute() or ".." in member.parts:
            raise PrepareError("Unsafe archive path: {}".format(info.filename))
        utterance_id = member.stem
        if utterance_id in members:
            raise PrepareError("Duplicate audio utterance ID: {}".format(utterance_id))
        members[utterance_id] = info.filename
    if not members:
        raise PrepareError("Archive contains no FLAC or WAV audio")
    return members


def choose_speaker_disjoint_rows(
    records: Iterable[Mapping[str, str]],
    max_utterances: int,
    validation_fraction: float,
    seed: int,
) -> Tuple[List[Mapping[str, str]], List[Mapping[str, str]]]:
    if max_utterances < 2:
        raise PrepareError("--max-utterances must be at least 2")
    if not 0.0 < validation_fraction < 1.0:
        raise PrepareError("--validation-fraction must be between 0 and 1")

    by_speaker: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        by_speaker[record["speaker_id"]].append(record)
    if len(by_speaker) < 2:
        raise PrepareError("At least two speakers are required for a disjoint split")

    rng = random.Random(seed)
    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    target_validation = max(1, round(max_utterances * validation_fraction))
    validation_speakers = set()
    validation_available = 0
    for speaker in speakers[:-1]:
        validation_speakers.add(speaker)
        validation_available += len(by_speaker[speaker])
        if validation_available >= target_validation:
            break

    train_pool = [row for speaker in speakers if speaker not in validation_speakers for row in by_speaker[speaker]]
    validation_pool = [row for speaker in speakers if speaker in validation_speakers for row in by_speaker[speaker]]
    rng.shuffle(train_pool)
    rng.shuffle(validation_pool)

    validation_count = min(target_validation, len(validation_pool), max_utterances - 1)
    train_count = min(max_utterances - validation_count, len(train_pool))
    if train_count + validation_count < max_utterances:
        validation_count = min(len(validation_pool), max_utterances - train_count)
    train = sorted(train_pool[:train_count], key=lambda row: row["utterance_id"])
    validation = sorted(validation_pool[:validation_count], key=lambda row: row["utterance_id"])
    if not train or not validation:
        raise PrepareError("Could not create non-empty speaker-disjoint splits")
    return train, validation


def write_jsonl(path: Path, rows: Iterable[Mapping[str, str]], audio_dir: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "audio": str((audio_dir / (row["utterance_id"] + row["suffix"])).resolve()),
                "text": "language None<asr_text>" + row["transcript"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def prepare(
    archive_path: Path,
    transcript_path: Path,
    output_dir: Path,
    max_utterances: int,
    validation_fraction: float,
    seed: int,
    expected_sha256: Optional[str],
) -> Dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PrepareError("Output directory is not empty: {}".format(output_dir))
    actual_sha256 = sha256_file(archive_path)
    if expected_sha256 and actual_sha256.lower() != expected_sha256.lower():
        raise PrepareError("SHA-256 mismatch for {}".format(archive_path))

    transcripts = load_transcripts(transcript_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()

    with zipfile.ZipFile(archive_path) as archive:
        audio_members = archive_audio(archive)
        available = []
        for utterance_id, member in audio_members.items():
            if utterance_id not in transcripts:
                continue
            record = dict(transcripts[utterance_id])
            record["member"] = member
            record["suffix"] = PurePosixPath(member).suffix.lower()
            available.append(record)
        train, validation = choose_speaker_disjoint_rows(available, max_utterances, validation_fraction, seed)
        selected = [("train", row) for row in train] + [("validation", row) for row in validation]
        for _, row in selected:
            destination = audio_dir / (row["utterance_id"] + row["suffix"])
            with archive.open(row["member"]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        license_members = [name for name in archive.namelist() if PurePosixPath(name).name.upper() == "LICENSE"]
        if license_members:
            with archive.open(license_members[0]) as source, (output_dir / "LICENSE").open("wb") as target:
                shutil.copyfileobj(source, target)

    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("audio_filename", "utterance_id", "speaker_id", "transcript", "language", "split"),
        )
        writer.writeheader()
        for split, row in selected:
            writer.writerow(
                {
                    "audio_filename": row["utterance_id"] + row["suffix"],
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "transcript": row["transcript"],
                    "language": "jav",
                    "split": split,
                }
            )
    write_jsonl(output_dir / "train.qwen.jsonl", train, audio_dir)
    write_jsonl(output_dir / "validation.qwen.jsonl", validation, audio_dir)

    train_speakers = sorted({row["speaker_id"] for row in train})
    validation_speakers = sorted({row["speaker_id"] for row in validation})
    manifest: Dict[str, object] = {
        "source": "OpenSLR SLR35 shard 0",
        "source_url": "https://openslr.org/35/",
        "license": "CC-BY-SA-4.0",
        "competition_external_data_action": "Publish a permitted redistribution to MDC at the conclusion of the competition.",
        "archive": archive_path.name,
        "archive_sha256": actual_sha256,
        "seed": seed,
        "archive_audio_count": len(audio_members),
        "transcribed_audio_count": len(available),
        "train_utterances": len(train),
        "validation_utterances": len(validation),
        "train_speakers": train_speakers,
        "validation_speakers": validation_speakers,
        "speaker_overlap": sorted(set(train_speakers) & set(validation_speakers)),
    }
    with (output_dir / "SUBSET_MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-utterances", type=int, default=3000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare(
            args.archive,
            args.transcripts,
            args.output,
            args.max_utterances,
            args.validation_fraction,
            args.seed,
            args.expected_sha256,
        )
    except (OSError, zipfile.BadZipFile, PrepareError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "manifest": str(args.output / "SUBSET_MANIFEST.json"),
                "archive_audio_count": manifest["archive_audio_count"],
                "train_utterances": manifest["train_utterances"],
                "validation_utterances": manifest["validation_utterances"],
                "train_speakers": len(manifest["train_speakers"]),
                "validation_speakers": len(manifest["validation_speakers"]),
                "speaker_overlap": len(manifest["speaker_overlap"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
