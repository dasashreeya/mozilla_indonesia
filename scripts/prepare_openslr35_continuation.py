#!/usr/bin/env python3
"""Prepare additional OpenSLR 35 shards without leaking fixed validation speakers."""

import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

from fetch_openslr35 import load_checksums, parse_shards
from prepare_openslr35 import (
    PrepareError,
    archive_audio,
    load_transcripts,
    sha256_file,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOADS = ROOT / "data" / "downloads" / "openslr35"
DEFAULT_EXCLUDE_METADATA = (
    ROOT / "data" / "subsets" / "openslr35" / "shard0-11000-seed-20260905" / "metadata.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "subsets" / "openslr35" / "shards1-f-continuation"


def validation_speakers(path: Path) -> Set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"speaker_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise PrepareError("Exclude metadata must contain speaker_id and split: {}".format(path))
    speakers = {row["speaker_id"] for row in rows if row["split"] == "validation"}
    if not speakers:
        raise PrepareError("Exclude metadata has no validation speakers: {}".format(path))
    return speakers


def prepare_continuation(
    *,
    archives_dir: Path,
    transcript_path: Path,
    checksums_path: Path,
    exclude_metadata: Path,
    output_dir: Path,
    shards: Sequence[str],
    max_utterances: int = 0,
) -> Dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PrepareError("Output directory is not empty: {}".format(output_dir))
    if max_utterances < 0:
        raise PrepareError("--max-utterances cannot be negative")

    transcripts = load_transcripts(transcript_path)
    checksums = load_checksums(checksums_path)
    excluded_speakers = validation_speakers(exclude_metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir()

    selected = []
    shard_counts: Counter[str] = Counter()
    skipped_validation_speaker = 0
    seen_utterances: Set[str] = set()
    for shard in shards:
        archive_path = archives_dir / "asr_javanese_{}.zip".format(shard)
        expected_hash = checksums.get(archive_path.name)
        if not expected_hash or sha256_file(archive_path) != expected_hash:
            raise PrepareError("Missing or unverified archive: {}".format(archive_path))
        with zipfile.ZipFile(archive_path) as archive:
            members = archive_audio(archive)
            for utterance_id, member in sorted(members.items()):
                if utterance_id in seen_utterances:
                    raise PrepareError("Duplicate utterance across shards: {}".format(utterance_id))
                seen_utterances.add(utterance_id)
                transcript = transcripts.get(utterance_id)
                if transcript is None:
                    continue
                if transcript["speaker_id"] in excluded_speakers:
                    skipped_validation_speaker += 1
                    continue
                if max_utterances and len(selected) >= max_utterances:
                    break
                suffix = Path(member).suffix.lower()
                destination = audio_dir / (utterance_id + suffix)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                row = dict(transcript)
                row["suffix"] = suffix
                row["shard"] = shard
                selected.append(row)
                shard_counts[shard] += 1
        if max_utterances and len(selected) >= max_utterances:
            break

    if not selected:
        raise PrepareError("No training utterances were selected")
    selected.sort(key=lambda row: row["utterance_id"])
    with (output_dir / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("audio_filename", "utterance_id", "speaker_id", "transcript", "language", "split", "shard")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "audio_filename": row["utterance_id"] + row["suffix"],
                    "utterance_id": row["utterance_id"],
                    "speaker_id": row["speaker_id"],
                    "transcript": row["transcript"],
                    "language": "jav",
                    "split": "train",
                    "shard": row["shard"],
                }
            )
    write_jsonl(output_dir / "train.qwen.jsonl", selected, audio_dir)

    train_speakers = sorted({row["speaker_id"] for row in selected})
    manifest: Dict[str, object] = {
        "source": "OpenSLR SLR35 continuation shards",
        "source_url": "https://openslr.org/35/",
        "license": "CC-BY-SA-4.0",
        "competition_external_data_action": "Publish a permitted redistribution to MDC at the conclusion of the competition.",
        "shards": list(shards),
        "shard_utterances": dict(sorted(shard_counts.items())),
        "train_utterances": len(selected),
        "train_speakers": train_speakers,
        "excluded_validation_speakers": sorted(excluded_speakers),
        "skipped_validation_speaker_utterances": skipped_validation_speaker,
        "speaker_overlap": sorted(set(train_speakers) & excluded_speakers),
    }
    with (output_dir / "SUBSET_MANIFEST.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives-dir", type=Path, default=DEFAULT_DOWNLOADS)
    parser.add_argument("--transcripts", type=Path, default=DEFAULT_DOWNLOADS / "utt_spk_text.tsv")
    parser.add_argument("--checksums", type=Path, default=DEFAULT_DOWNLOADS / "asr_javanese.sha256")
    parser.add_argument("--exclude-metadata", type=Path, default=DEFAULT_EXCLUDE_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shards", default="1-f")
    parser.add_argument("--max-utterances", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = prepare_continuation(
            archives_dir=args.archives_dir,
            transcript_path=args.transcripts,
            checksums_path=args.checksums,
            exclude_metadata=args.exclude_metadata,
            output_dir=args.output,
            shards=parse_shards(args.shards),
            max_utterances=args.max_utterances,
        )
    except (OSError, ValueError, zipfile.BadZipFile, PrepareError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(args.output / "SUBSET_MANIFEST.json"), **manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
