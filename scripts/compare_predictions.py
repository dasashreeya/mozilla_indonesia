#!/usr/bin/env python3
"""Paired WER differences with session-cluster bootstrap intervals."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdc_asr.wer import ErrorCounts, word_errors


def read_predictions(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {r["audio_filename"]: r["transcript"] for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prediction filenames in {path}")
    return result


def compare(rows, baseline, candidate, samples=10000):
    import numpy as np
    expected = {r["audio_filename"] for r in rows}
    if set(baseline) != expected or set(candidate) != expected:
        raise ValueError("Prediction filenames must exactly match the validation split")
    by_session = defaultdict(lambda: [ErrorCounts(), ErrorCounts()])
    by_language = defaultdict(lambda: [ErrorCounts(), ErrorCounts()])
    aggregate = [ErrorCounts(), ErrorCounts()]
    for row in rows:
        for i, predictions in enumerate((baseline, candidate)):
            error = word_errors(row["transcript"], predictions[row["audio_filename"]])
            aggregate[i] += error
            by_session[row.get("session") or row.get("convo_id") or row["audio_filename"]][i] += error
            by_language[row["language"]][i] += error
    sessions = sorted(by_session)
    values = np.array([[sum((c.substitutions, c.deletions, c.insertions)) for c in by_session[s]]
                       + [by_session[s][0].reference_words] for s in sessions], dtype=float)
    rng = np.random.default_rng(20260908)
    indices = rng.integers(0, len(sessions), size=(samples, len(sessions)))
    boot = values[indices].sum(axis=1)
    delta = (boot[:, 1] - boot[:, 0]) / boot[:, 2]
    return {"clips": len(rows), "sessions": len(sessions),
            "baseline": {"wer": aggregate[0].wer, **asdict(aggregate[0])},
            "candidate": {"wer": aggregate[1].wer, **asdict(aggregate[1])},
            "delta_wer": aggregate[1].wer - aggregate[0].wer,
            "paired_session_bootstrap_95pct": np.quantile(delta, [0.025, 0.975]).tolist(),
            "languages": {language: {"baseline": pair[0].wer, "candidate": pair[1].wer,
                                      "delta": pair[1].wer - pair[0].wer}
                          for language, pair in by_language.items()},
            "caveat": "Descriptive interval after model selection; one held-out codeswitch conversation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.metadata.open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "validation"]
    result = compare(rows, read_predictions(args.baseline), read_predictions(args.candidate))
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
