"""Dependency-free corpus WER scorer using the official MDC normalizer."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .normalization import normalize_text


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return float("inf") if self.errors else 0.0
        return self.errors / self.reference_words

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.reference_words + other.reference_words,
        )


def word_errors(reference: str, hypothesis: str) -> ErrorCounts:
    """Return Levenshtein S/D/I counts after MDC normalization."""

    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    previous: List[Tuple[int, int, int, int]] = [(j, 0, 0, j) for j in range(len(hyp) + 1)]

    for i, ref_word in enumerate(ref, start=1):
        current: List[Tuple[int, int, int, int]] = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hyp, start=1):
            if ref_word == hyp_word:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            choices = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            current.append(min(choices))
        previous = current

    _, substitutions, deletions, insertions = previous[-1]
    return ErrorCounts(substitutions, deletions, insertions, len(ref))


def corpus_errors(pairs: Iterable[Tuple[str, str]]) -> ErrorCounts:
    total = ErrorCounts()
    for reference, hypothesis in pairs:
        total += word_errors(reference, hypothesis)
    return total


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel_tab if path.suffix.lower() == ".tsv" else csv.excel
        rows = list(csv.DictReader(handle, dialect=dialect))
    if not rows:
        raise ValueError("No rows found in {}".format(path))
    return rows


def _by_filename(rows: Sequence[Mapping[str, str]], path: Path) -> Dict[str, Mapping[str, str]]:
    result: Dict[str, Mapping[str, str]] = {}
    for row in rows:
        filename = row.get("audio_filename", "")
        if not filename:
            raise ValueError("{} is missing audio_filename".format(path))
        if filename in result:
            raise ValueError("{} has duplicate audio_filename {}".format(path, filename))
        result[filename] = row
    return result


def score_files(reference_path: Path, prediction_path: Path, group_by: Optional[str] = None) -> Dict[str, ErrorCounts]:
    references = _read_rows(reference_path)
    predictions = _by_filename(_read_rows(prediction_path), prediction_path)
    if "transcript" not in references[0] or "transcript" not in next(iter(predictions.values())):
        raise ValueError("Both files need a transcript column")

    missing = [row["audio_filename"] for row in references if row["audio_filename"] not in predictions]
    if missing:
        raise ValueError("Missing {} predictions (first: {})".format(len(missing), missing[0]))

    groups: Dict[str, List[Tuple[str, str]]] = {"all": []}
    for row in references:
        predicted = predictions[row["audio_filename"]].get("transcript", "") or ""
        pair = (row.get("transcript", "") or "", predicted)
        groups["all"].append(pair)
        if group_by:
            label = row.get(group_by, "") or "<missing>"
            groups.setdefault(label, []).append(pair)
    return {name: corpus_errors(pairs) for name, pairs in groups.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--group-by", help="Optional reference column for diagnostic WER")
    args = parser.parse_args(argv)

    results = score_files(args.reference, args.prediction, args.group_by)
    for name, counts in results.items():
        print(
            "{}: WER={:.6f} S={} D={} I={} N={}".format(
                name,
                counts.wer,
                counts.substitutions,
                counts.deletions,
                counts.insertions,
                counts.reference_words,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

