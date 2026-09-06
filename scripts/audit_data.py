#!/usr/bin/env python3
"""Audit extracted Indonesian–Javanese development and Jember data locally.

Requires numpy and ffmpeg/ffprobe. Writes aggregate JSON plus private anomaly
records, never transcript strings. Signal probes are shared with the Spanish
workspace by a vendored, language-independent mdc_asr.audio_probe module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdc_asr.audio_probe import decode, downmix, probe_container, require_ffmpeg, signal_metrics
from mdc_asr.normalization import normalize_text

FIELDS = ["decoded_duration_seconds", "rms_dbfs", "peak_dbfs", "estimated_snr_db",
          "silence_ratio", "clipped_sample_ratio", "effective_bandwidth_hz",
          "spectral_centroid_hz", "tokens_per_second"]


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_key(text):
    return " ".join(normalize_text(text).split())


def describe(values):
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not a.size:
        return {"count": 0}
    return {"count": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
            **{k: float(np.percentile(a, p)) for k, p in
               [("min", 0), ("p01", 1), ("p05", 5), ("median", 50),
                ("p95", 95), ("p99", 99), ("max", 100)]}}


def timestamp(value):
    parts = value.strip().replace(",", ".").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError("invalid timestamp component count")
    numbers = [float(x) for x in parts]
    if any(not math.isfinite(x) or x < 0 for x in numbers):
        raise ValueError("nonfinite or negative timestamp")
    if len(numbers) > 1 and any(x >= 60 for x in numbers[1:]):
        raise ValueError("minute/second component outside range")
    return sum(n * 60 ** i for i, n in enumerate(reversed(numbers)))


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        columns = reader.fieldnames or []
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Ragged metadata rows")
    return columns, rows


def interval_audit(rows):
    """Rows have validated start/end values and belong to one recording."""
    ordered = sorted(rows, key=lambda r: (r["start"], r["end"], r["id"]))
    frontier = 0.0
    union = 0.0
    overlaps = []
    for row in ordered:
        start, end = row["start"], row["end"]
        if start < frontier:
            overlaps.append({"id": row["id"], "overlap_seconds": min(end, frontier) - start})
        union += max(0.0, end - max(start, frontier))
        frontier = max(frontier, end)
    return {"union_seconds": union, "overlap_rows": overlaps}


def canonical_inputs(dev_root, train_root):
    dev_meta = next(dev_root.rglob("metadata.tsv"))
    train_metas = list(train_root.rglob("*.tsv"))
    if len(train_metas) != 1:
        raise ValueError("Expected exactly one Jember timestamp TSV")
    train_meta = train_metas[0]
    dev_cols, dev_raw = read_tsv(dev_meta)
    train_cols, train_raw = read_tsv(train_meta)
    if set(dev_cols) != {"audio_filename", "speaker", "transcript", "language", "convo_id"}:
        raise ValueError("Unexpected development columns")
    if set(train_cols) != {"Audio file name", "start", "end", "text"}:
        raise ValueError("Unexpected Jember columns")
    inventory = {}
    for label, root in [("dev", dev_root), ("train", train_root)]:
        audio = list(root.rglob("*.mp3"))
        counts = Counter(p.name for p in audio)
        if any(v > 1 for v in counts.values()):
            raise ValueError("Ambiguous audio basenames")
        inventory[label] = {p.name: p for p in audio}
    rows = {"dev": [], "train": []}
    invalid = []
    for i, r in enumerate(dev_raw, 1):
        name = r["audio_filename"]
        if Path(name).name != name or name not in inventory["dev"]:
            invalid.append({"dataset": "dev", "row": i, "reason": "unsafe or missing audio"})
            continue
        rows["dev"].append({"id": name, "source": str(inventory["dev"][name]),
                            "text": r["transcript"], "language": r["language"],
                            "speaker": r["speaker"], "conversation": r["convo_id"]})
    for i, r in enumerate(train_raw, 1):
        session = Path(r["Audio file name"].strip()).stem
        name = session + ".mp3"
        try:
            start, end = timestamp(r["start"]), timestamp(r["end"])
            if end <= start:
                raise ValueError("end is not after start")
            if name not in inventory["train"]:
                raise ValueError("missing audio")
        except ValueError as exc:
            invalid.append({"dataset": "train", "row": i, "reason": str(exc)})
            continue
        rows["train"].append({"id": f"{session}:{i}", "source": str(inventory["train"][name]),
                              "session": session, "start": start, "end": end, "text": r["text"]})
    structural = {"invalid_rows": invalid}
    for label, raw, cols, meta in [("dev", dev_raw, dev_cols, dev_meta),
                                    ("train", train_raw, train_cols, train_meta)]:
        references = {Path(r["source"]).name for r in rows[label]}
        structural[label] = {"metadata_sha256": digest_file(meta), "metadata_columns": cols,
                             "metadata_rows": len(raw), "audio_files": len(inventory[label]),
                             "empty_cells": {c: sum(not r[c].strip() for r in raw) for c in cols},
                             "untrimmed_cells": {c: sum(r[c] != r[c].strip() for r in raw) for c in cols},
                             "orphan_audio_files": sorted(set(inventory[label]) - references),
                             "duplicate_ids": {k: v for k, v in Counter(r["id"] for r in rows[label]).items() if v > 1}}
    structural["dev"]["unexpected_language_rows"] = sum(r["language"] not in {"ind", "jav", "javind"} for r in rows["dev"])
    return rows, inventory, structural


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def text_summary(rows):
    raw = [r["text"] for r in rows]
    normalized = [text_key(t) for t in raw]
    tokens = [t.split() for t in normalized]
    counts = Counter(normalized)
    nonempty = {k: v for k, v in counts.items() if k}
    vocabulary = Counter(w for ts in tokens for w in ts)
    total = sum(vocabulary.values())
    lower_errors = sum(a != b for s in raw for a, b in zip(text_key(s).split(), text_key(s.lower()).split()))
    accent_errors = sum(a != b for s in raw for a, b in zip(text_key(s).split(), text_key(strip_accents(s)).split()))
    return {"rows": len(rows), "scoring_words": total, "vocabulary_types": len(vocabulary),
            "words_per_row": describe([len(t) for t in tokens]),
            "empty_raw_rows": sum(not t.strip() for t in raw),
            "empty_after_normalization_rows": sum(not t for t in normalized),
            "rows_changed_by_normalizer": sum(a.strip() != b for a, b in zip(raw, normalized)),
            "rows_with_bracket_annotations": sum(bool(re.search(r"\[[^\]]+\]", t)) for t in raw),
            "rows_with_unintelligibility_markers": sum(bool(re.search(r"\(\?+\)", t)) for t in raw),
            "rows_with_digits": sum(bool(re.search(r"\d", t)) for t in raw),
            "rows_with_hyphens": sum("-" in t for t in raw),
            "rows_with_ellipsis": sum("..." in t for t in raw),
            "rows_not_nfc": sum(t != unicodedata.normalize("NFC", t) for t in raw),
            "non_ascii_character_counts": dict(Counter(c for t in raw for c in t if ord(c) > 127)),
            "duplicate_normalized_text_groups": sum(v > 1 for v in nonempty.values()),
            "duplicate_normalized_text_excess_rows": sum(v - 1 for v in nonempty.values()),
            "reference_lowercase_token_change_fraction": lower_errors / total if total else None,
            "reference_accent_strip_token_change_fraction": accent_errors / total if total else None}


def overlap_candidates(train, dev):
    exact = defaultdict(list)
    ngrams = defaultdict(set)
    train_tokens = {}
    for row in train:
        key = text_key(row["text"])
        if key:
            exact[key].append(row["id"])
        ts = key.casefold().split()
        train_tokens[row["id"]] = len(ts)
        for i in range(len(ts) - 4):
            ngrams[tuple(ts[i:i + 5])].add(row["id"])
    exact_hits, phrase_hits = [], []
    for row in dev:
        key = text_key(row["text"])
        if key in exact:
            exact_hits.append({"dev_id": row["id"], "train_ids": exact[key], "words": len(key.split())})
        ts = key.casefold().split()
        grams = {tuple(ts[i:i + 5]) for i in range(len(ts) - 4)}
        matches = Counter(tid for gram in grams for tid in ngrams.get(gram, ()))
        for tid, count in matches.most_common(3):
            phrase_hits.append({"dev_id": row["id"], "train_id": tid, "shared_unique_5grams": count,
                                "dev_unique_5grams": len(grams), "dev_words": len(ts),
                                "train_words": train_tokens[tid]})
    return {"exact_normalized_text_matches": exact_hits, "shared_phrase_candidates": phrase_hits,
            "dev_rows_with_exact_text_match": len(exact_hits),
            "dev_rows_with_shared_5gram": len({r["dev_id"] for r in phrase_hits}),
            "note": "Text overlap generates review candidates; it does not establish audio identity."}


def probe_source(job):
    label, path, rows = job
    path = Path(path)
    try:
        container = probe_container(path)
        samples, sr = decode(path, container=container)
        mono = downmix(samples)
        duration = len(mono) / sr
        file_result = {"dataset": label, "filename": path.name, **container,
                       "decoded_duration_seconds": duration, "file_sha256": digest_file(path),
                       "pcm_sha256": hashlib.sha256(str(sr).encode() + mono.astype("<f4").tobytes()).hexdigest()}
        clips, anomalies = [], []
        for row in rows:
            start, end = row.get("start", 0), row.get("end", duration)
            if end > duration:
                anomalies.append({"id": row["id"], "reason": "end_beyond_decoded_audio",
                                  "overrun_seconds": end - duration})
            if start >= duration:
                anomalies.append({"id": row["id"], "reason": "start_beyond_decoded_audio"})
                continue
            segment = mono[round(start * sr):min(len(mono), round(end * sr))]
            if not segment.size:
                anomalies.append({"id": row["id"], "reason": "empty_decoded_segment"})
                continue
            metrics = signal_metrics(segment, sr)
            words = len(text_key(row["text"]).split())
            clips.append({"id": row["id"], "sample_rate_hz": sr, "channels": container["channels"],
                          **metrics, "words": words, "tokens_per_second": words / (len(segment) / sr),
                          "pcm_sha256": hashlib.sha256(str(sr).encode() + segment.astype("<f4").tobytes()).hexdigest()})
        return file_result, clips, anomalies
    except Exception as exc:
        return {"dataset": label, "filename": path.name, "decode_error": type(exc).__name__}, [], []


def audio_summary(files, clips):
    good = [f for f in files if "decode_error" not in f]
    return {"source_files": len(files), "source_decode_failures": sum("decode_error" in f for f in files),
            "source_audio_hours": sum(f["decoded_duration_seconds"] for f in good) / 3600,
            "source_bytes": sum(f["bytes"] for f in good),
            "segment_rows_probed": len(clips),
            "segment_audio_hours_sum": sum(c["decoded_duration_seconds"] for c in clips) / 3600,
            "sample_rates": dict(Counter(f["declared_sample_rate_hz"] for f in good)),
            "channel_counts": dict(Counter(f["channels"] for f in good)),
            "codec_counts": dict(Counter(f["codec_name"] for f in good)),
            "metrics": {k: describe([c.get(k) for c in clips]) for k in FIELDS},
            "under_2_seconds": sum(c["decoded_duration_seconds"] < 2 for c in clips),
            "over_30_seconds": sum(c["decoded_duration_seconds"] > 30 for c in clips),
            "over_40_seconds": sum(c["decoded_duration_seconds"] > 40 for c in clips),
            "over_40_1_seconds": sum(c["decoded_duration_seconds"] > 40.1 for c in clips),
            "any_clipping": sum(c["clipped_sample_ratio"] > 0 for c in clips),
            "heavy_clipping": sum(c["clipped_sample_ratio"] > 1e-4 for c in clips),
            "mostly_silent": sum(c["silence_ratio"] > 0.5 for c in clips),
            "low_energy_contrast": sum(c["estimated_snr_db"] < 15 for c in clips),
            "very_quiet": sum(c["rms_dbfs"] < -35 for c in clips),
            "low_token_rate": sum(c["tokens_per_second"] < 0.5 for c in clips),
            "high_token_rate": sum(c["tokens_per_second"] > 6 for c in clips)}


def group_summary(rows, clips, key):
    by_id = {c["id"]: c for c in clips}
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = {}
    for label, rs in sorted(groups.items()):
        cs = [by_id[r["id"]] for r in rs if r["id"] in by_id]
        out[label] = {"rows": len(rs), "words": sum(len(text_key(r["text"]).split()) for r in rs),
                      "audio_hours": sum(c["decoded_duration_seconds"] for c in cs) / 3600,
                      "speakers": sorted({r["speaker"] for r in rs}),
                      "languages": dict(Counter(r["language"] for r in rs)),
                      "metrics": {k: describe([c.get(k) for c in cs]) for k in FIELDS}}
    return out


def duplicates(items, hash_field):
    groups = defaultdict(list)
    for r in items:
        if hash_field in r:
            groups[r[hash_field]].append(r.get("id", r.get("filename")))
    return [ids for ids in groups.values() if len(ids) > 1]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dev", type=Path, default=ROOT / "data/subsets/dev/all")
    p.add_argument("--train", type=Path, default=ROOT / "data/subsets/jember/all")
    p.add_argument("--output", type=Path, default=ROOT / "reports")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers must be positive")
    require_ffmpeg()
    rows, inventory, structural = canonical_inputs(args.dev, args.train)
    by_source = defaultdict(list)
    for label in rows:
        for r in rows[label]:
            by_source[(label, r["source"])].append(r)
    jobs = [(label, str(path), by_source[(label, str(path))])
            for label in inventory for path in inventory[label].values()]
    files = {"dev": [], "train": []}
    clips = {"dev": [], "train": []}
    anomalies = []
    private = args.output / "private"
    cache = private / "audio_cache"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for job in jobs:
            label, path, rs = job
            # Include source and metadata hashes so cache is invalidated on input changes.
            identity = json.dumps([digest_file(Path(path)), structural[label]["metadata_sha256"],
                                   digest_file(Path(__file__)), digest_file(ROOT / "src/mdc_asr/audio_probe.py"),
                                   digest_file(ROOT / "src/mdc_asr/normalization.py")]).encode()
            cached = cache / (hashlib.sha256(identity).hexdigest() + ".json")
            if cached.exists():
                f, cs, issues = json.loads(cached.read_text())
                files[label].append(f); clips[label].extend(cs); anomalies.extend(issues)
            else:
                pending[pool.submit(probe_source, job)] = (label, cached)
        print(f"Audio files: {len(jobs)}; uncached: {len(pending)}", flush=True)
        for i, future in enumerate(as_completed(pending), 1):
            label, cached = pending[future]
            f, cs, issues = future.result()
            files[label].append(f); clips[label].extend(cs); anomalies.extend(issues)
            if "decode_error" not in f:
                write_json(cached, [f, cs, issues])
            if i % 25 == 0 or i == len(pending):
                print(f"Decoded {i}/{len(pending)} files", flush=True)
    for label in clips:
        clips[label].sort(key=lambda c: c["id"])
        files[label].sort(key=lambda f: f["filename"])
    interval_results = {Path(source).name: interval_audit(rs)
                        for (label, source), rs in by_source.items() if label == "train"}
    overlap = overlap_candidates(rows["train"], rows["dev"])
    train_vocab = {w for r in rows["train"] for w in text_key(r["text"]).split()}
    dev_words = [w for r in rows["dev"] for w in text_key(r["text"]).split()]
    dev_pcm = {c["pcm_sha256"] for c in clips["dev"]}
    overlap["train_segments_with_exact_dev_pcm_hash"] = sum(c["pcm_sha256"] in dev_pcm for c in clips["train"])
    overlap["pcm_limit"] = "Hashes include native sample rate. Different rates/encodings/timing prevent matches; zero is not proof of no leakage."
    summary = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "structural": structural,
               "text": {label: text_summary(rs) for label, rs in rows.items()},
               "audio": {label: audio_summary(files[label], clips[label]) for label in rows},
               "development_groups": {k: group_summary(rows["dev"], clips["dev"], k) for k in ["language", "conversation", "speaker"]},
               "training_intervals": {"sessions_referenced": len(interval_results),
                                      "union_hours": sum(v["union_seconds"] for v in interval_results.values()) / 3600,
                                      "rows_overlapping_preceding_coverage": sum(len(v["overlap_rows"]) for v in interval_results.values()),
                                      "audio_boundary_anomalies": len(anomalies)},
               "training_vocabulary_coverage": {"dev_token_oov_fraction": sum(w not in train_vocab for w in dev_words) / len(dev_words),
                                                "dev_token_oov_fraction_casefolded": None},
               "overlap": {k: v for k, v in overlap.items() if not isinstance(v, list)},
               "duplicates": {label: {"file_hash_groups": len(duplicates(files[label], "file_sha256")),
                                      "segment_pcm_groups": len(duplicates(clips[label], "pcm_sha256"))} for label in rows},
               "methodology": {"signal_sample_rate": "native; no resampling", "clipping_threshold": 0.999,
                               "heavy_clipping_fraction": 0.0001, "silence_relative_db": 35,
                               "snr": "90th minus 10th percentile frame RMS; energy contrast, not absolute SNR",
                               "quality_flags": "heuristic review candidates; no automatic filtering",
                               "training_hours": "segment sum may double-count overlapping timestamps; union hours also reported",
                               "transcript_privacy": "no transcript strings written to audit outputs",
                               "speaker_limit": "Jember TSV has no speaker/conversation labels; sessions are not speaker IDs"}}
    folded = {w.casefold() for w in train_vocab}
    summary["training_vocabulary_coverage"]["dev_token_oov_fraction_casefolded"] = sum(w.casefold() not in folded for w in dev_words) / len(dev_words)
    write_json(args.output / "data_audit.json", summary)
    write_json(private / "clip_metrics.json", clips)
    write_json(private / "source_files.json", files)
    write_json(private / "timestamp_anomalies.json", {"parse": structural["invalid_rows"], "bounds": anomalies, "intervals": interval_results})
    write_json(private / "overlap_candidates.json", overlap)
    write_json(private / "duplicates.json", {label: {"files": duplicates(files[label], "file_sha256"), "pcm": duplicates(clips[label], "pcm_sha256")} for label in rows})
    print(json.dumps({label: summary["audio"][label] for label in rows}, indent=2))


if __name__ == "__main__":
    main()
