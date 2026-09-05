#!/usr/bin/env python3
"""Validate output columns and exact filename coverage before upload."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def read(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["audio_filename", "transcript"]:
            raise ValueError("{} must have exactly: audio_filename,transcript".format(path))
        return list(reader)


def unique_filenames(rows: Sequence[Dict[str, str]], path: Path) -> set:
    names = [row["audio_filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("{} contains duplicate audio_filename values".format(path))
    return set(names)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("expected", type=Path, help="Official submission_format.csv")
    args = parser.parse_args(argv)

    submission = read(args.submission)
    expected = read(args.expected)
    actual_names = unique_filenames(submission, args.submission)
    expected_names = unique_filenames(expected, args.expected)
    missing = expected_names - actual_names
    extra = actual_names - expected_names
    if missing or extra:
        raise ValueError("Filename mismatch: {} missing, {} extra".format(len(missing), len(extra)))
    blank = sum(not (row.get("transcript") or "").strip() for row in submission)
    if blank:
        raise ValueError("Submission contains {} blank transcripts".format(blank))
    print("Valid submission: {} unique rows".format(len(submission)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)

