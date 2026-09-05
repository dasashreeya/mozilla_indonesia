#!/usr/bin/env python3
"""Download and verify pinned OpenSLR 35 audio shards and metadata."""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from mdc_fetch import MDCError, download_file, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "external_data.json"
OUTPUT = ROOT / "data" / "downloads" / "openslr35"
ALL_SHARDS = tuple(format(index, "x") for index in range(16))


def parse_shards(value: str) -> Tuple[str, ...]:
    value = value.strip().lower()
    if value == "all":
        return ALL_SHARDS
    selected = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start not in ALL_SHARDS or end not in ALL_SHARDS:
                raise ValueError("OpenSLR shard range must use hexadecimal IDs 0-f")
            start_index = ALL_SHARDS.index(start)
            end_index = ALL_SHARDS.index(end)
            if start_index > end_index:
                raise ValueError("OpenSLR shard range must be ascending: {}".format(part))
            selected.extend(ALL_SHARDS[start_index : end_index + 1])
        elif part in ALL_SHARDS:
            selected.append(part)
        else:
            raise ValueError("Unknown OpenSLR shard: {}".format(part))
    if not selected:
        raise ValueError("At least one OpenSLR shard is required")
    return tuple(dict.fromkeys(selected))


def load_checksums(path: Path) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("Malformed checksum row {} in {}".format(line_number, path))
        digest, filename = parts
        int(digest, 16)
        checksums[filename] = digest.lower()
    return checksums


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--shards",
        default="0",
        help="Comma-separated hexadecimal IDs/ranges (for example 0,2-4 or 1-f), or all",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))["openslr35_shard0"]
        shards = parse_shards(args.shards)
        supporting_files = (
            ("utt_spk_text.tsv", config["transcripts_url"], config["transcripts_sha256"]),
            ("asr_javanese.sha256", config["checksums_url"], config["checksums_sha256"]),
            ("LICENSE", config["license_url"], config["license_sha256"]),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, url, expected_hash in supporting_files:
            destination = args.output_dir / filename
            if destination.exists() and sha256_file(destination) == expected_hash:
                print("Using verified cached file: {}".format(destination))
                continue
            download_file(url, destination)
            if sha256_file(destination) != expected_hash:
                raise MDCError("SHA-256 mismatch for {}".format(destination))
            print("Downloaded and verified: {}".format(destination))
        checksums = load_checksums(args.output_dir / "asr_javanese.sha256")
        base_url = config["archive_url"].rsplit("/", 1)[0]
        for shard in shards:
            filename = "asr_javanese_{}.zip".format(shard)
            expected_hash = checksums.get(filename)
            if not expected_hash:
                raise MDCError("Pinned checksum manifest has no entry for {}".format(filename))
            destination = args.output_dir / filename
            if destination.exists() and sha256_file(destination) == expected_hash:
                print("Using verified cached file: {}".format(destination))
                continue
            download_file("{}/{}".format(base_url, filename), destination)
            if sha256_file(destination) != expected_hash:
                raise MDCError("SHA-256 mismatch for {}".format(destination))
            print("Downloaded and verified: {}".format(destination))
    except (KeyError, OSError, ValueError, json.JSONDecodeError, MDCError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
