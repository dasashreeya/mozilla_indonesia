#!/usr/bin/env python3
"""Build local review queues and supplementary statistics from completed audits."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict

from audit_data import (ROOT, canonical_inputs, interval_audit, overlap_candidates,
                        read_tsv, strip_accents, text_key, text_summary, timestamp, write_json)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    reports = ROOT / "reports"
    private = reports / "private"
    rows, _, structure = canonical_inputs(ROOT / "data/subsets/dev/all", ROOT / "data/subsets/jember/all")
    _, raw = read_tsv(next((ROOT / "data/subsets/jember/all").rglob("*.tsv")))
    metrics = json.loads((private / "clip_metrics.json").read_text())
    files = json.loads((private / "source_files.json").read_text())
    issues = json.loads((private / "timestamp_anomalies.json").read_text())
    acoustic = json.loads((private / "acoustic_overlap.json").read_text())
    review = defaultdict(set)
    for issue in structure["invalid_rows"]:
        r = raw[issue["row"] - 1]
        review[("train", f'{r["Audio file name"]}:{issue["row"]}')].add("invalid_timestamp")
    for issue in issues["bounds"]:
        review[("train", issue["id"])].add(issue["reason"])
    for group in issues["intervals"].values():
        for r in group["overlap_rows"]:
            review[("train", r["id"])].add("overlapping_timestamp")
    for dataset, clips in metrics.items():
        for r in clips:
            for flag, hit in [("near_full_scale_samples", r["clipped_sample_ratio"] > 1e-4),
                              ("mostly_silent", r["silence_ratio"] > 0.5),
                              ("low_energy_contrast", r["estimated_snr_db"] < 15),
                              ("very_quiet", r["rms_dbfs"] < -35),
                              ("low_token_rate", r["tokens_per_second"] < 0.5),
                              ("high_token_rate", r["tokens_per_second"] > 6)]:
                if hit:
                    review[(dataset, r["id"])].add(flag)
    write_csv(private / "review_queue.csv", ["dataset", "id", "flags"],
              [{"dataset": d, "id": i, "flags": ";".join(sorted(flags))} for (d, i), flags in sorted(review.items())])
    timing = []
    for (dataset, rid), flags in review.items():
        if dataset != "train" or not flags.intersection({"invalid_timestamp", "end_beyond_decoded_audio", "start_beyond_decoded_audio"}):
            continue
        number = int(rid.split(":")[-1])
        r = raw[number - 1]
        timing.append({"tsv_data_row": number, "session": r["Audio file name"],
                       "start": r["start"], "end": r["end"], "flags": ";".join(sorted(flags)),
                       "recommendation": "quarantine until reviewed; original data unchanged"})
    write_csv(private / "timestamp_quarantine_candidates.csv",
              ["tsv_data_row", "session", "start", "end", "flags", "recommendation"],
              sorted(timing, key=lambda r: r["tsv_data_row"]))
    durations = {r["filename"]: r["decoded_duration_seconds"] for r in files["train"]}
    intervals = defaultdict(list)
    for r in rows["train"]:
        end = min(r["end"], durations[r["session"] + ".mp3"])
        if end > r["start"]:
            intervals[r["session"]].append(dict(r, end=end))
    valid_union = sum(interval_audit(rs)["union_seconds"] for rs in intervals.values()) / 3600
    transforms = {"scoring": lambda s: s, "casefold": str.casefold,
                  "accent_and_casefold": lambda s: strip_accents(s).casefold()}
    dv = [w for r in rows["dev"] for w in text_key(r["text"]).split()]
    coverage = {}
    for name, transform in transforms.items():
        vocab = {transform(w) for r in rows["train"] for w in text_key(r["text"]).split()}
        coverage[name] = sum(transform(w) not in vocab for w in dv) / len(dv)
    relaxed = overlap_candidates([dict(r, text=strip_accents(r["text"]).casefold()) for r in rows["train"]],
                                 [dict(r, text=strip_accents(r["text"]).casefold()) for r in rows["dev"]])
    write_json(private / "relaxed_text_overlap.json", relaxed)
    folds = []
    for heldout in sorted({r["conversation"] for r in rows["dev"]}):
        train_speakers = {r["speaker"] for r in rows["dev"] if r["conversation"] != heldout}
        eval_speakers = {r["speaker"] for r in rows["dev"] if r["conversation"] == heldout}
        if train_speakers & eval_speakers:
            raise ValueError("Conversation folds would leak speakers")
        folds.extend({"heldout_conversation": heldout, "audio_filename": r["id"],
                      "role": "validation" if r["conversation"] == heldout else "development_training",
                      "speaker": r["speaker"], "language": r["language"]} for r in rows["dev"])
    write_csv(private / "development_folds.csv",
              ["heldout_conversation", "audio_filename", "role", "speaker", "language"], folds)
    summary = {"all_training_metadata_text": text_summary([{"text": r["text"]} for r in raw]),
               "zero_duration_rows": sum(timestamp(r["start"]) == timestamp(r["end"]) for r in raw),
               "reversed_timestamp_rows": sum(timestamp(r["start"]) > timestamp(r["end"]) for r in raw),
               "timestamp_quarantine_candidate_rows": len(timing),
               "clipped_to_audio_interval_union_hours": valid_union,
               "vocabulary_oov_fraction_of_dev_tokens": coverage,
               "unique_review_rows": dict(Counter(d for d, _ in review)),
               "relaxed_text_overlap": {k: v for k, v in relaxed.items() if not isinstance(v, list)},
               "acoustic_screen": {k: v for k, v in acoustic.items() if k not in {"matches", "known_positive_controls"}},
               "acoustic_positive_control_passes": sum(r["correct_recording"] and r["within_half_second"] for r in acoustic["known_positive_controls"]),
               "acoustic_positive_controls": len(acoustic["known_positive_controls"]),
               "scope": "Train text in data_audit.json covers 6,638 rows with positive timestamps; all_training_metadata_text here covers all 6,679 rows. No data cleaned or deleted."}
    write_json(reports / "audit_supplement.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
