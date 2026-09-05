#!/usr/bin/env python3
"""Download and verify the pinned OpenSLR 35 starter shard and metadata."""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional, Sequence

from mdc_fetch import MDCError, download_file, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "external_data.json"
OUTPUT = ROOT / "data" / "downloads" / "openslr35"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))["openslr35_shard0"]
        files = (
            ("asr_javanese_0.zip", config["archive_url"], config["archive_sha256"]),
            ("utt_spk_text.tsv", config["transcripts_url"], config["transcripts_sha256"]),
            ("asr_javanese.sha256", config["checksums_url"], config["checksums_sha256"]),
            ("LICENSE", config["license_url"], config["license_sha256"]),
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, url, expected_hash in files:
            destination = args.output_dir / filename
            if destination.exists() and sha256_file(destination) == expected_hash:
                print("Using verified cached file: {}".format(destination))
                continue
            download_file(url, destination)
            if sha256_file(destination) != expected_hash:
                raise MDCError("SHA-256 mismatch for {}".format(destination))
            print("Downloaded and verified: {}".format(destination))
    except (KeyError, OSError, json.JSONDecodeError, MDCError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
