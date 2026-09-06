#!/usr/bin/env python3
"""Build audited, conversation-held-out short/long Qwen adaptation inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_data import canonical_inputs, digest_file, text_key, write_json


def contiguous_windows(rows, minimum=18.0, maximum=28.0):
    """Nonoverlapping groups; never bridge a removed row or a timestamp gap."""
    windows, pending = [], []
    for row in rows:
        if pending:
            last = pending[-1]
            consecutive = int(row["id"].split(":")[-1]) == int(last["id"].split(":")[-1]) + 1
            if not consecutive or abs(row["start"] - last["end"]) > 1e-6 or row["end"] - pending[0]["start"] > maximum:
                if pending[-1]["end"] - pending[0]["start"] >= minimum and len(pending) > 1:
                    windows.append(pending)
                pending = []
        pending.append(row)
        if row["end"] - pending[0]["start"] >= minimum:
            if len(pending) > 1:
                windows.append(pending)
            pending = []
    if pending and len(pending) > 1 and pending[-1]["end"] - pending[0]["start"] >= minimum:
        windows.append(pending)
    return windows


def target_text(text, source, style):
    cleaned = text_key(text)
    if source == "jember" and style == "dev_style":
        cleaned = cleaned.translate(str.maketrans({"è": "e", "ê": "e", "ě": "e", "ḍ": "d",
                                                   "È": "E", "Ê": "E", "Ě": "E", "Ḍ": "D"}))
    return cleaned


def encode_session(job):
    source, candidates, audio_dir = job
    process = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", source,
                              "-ar", "16000", "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True)
    signal = np.frombuffer(process.stdout, dtype="<f4")
    # Headroom is applied once per recording only if decoded lossy peaks exceed
    # FLAC's integer range. Avoid introducing clipping during lossless staging.
    peak = float(np.max(np.abs(signal)))
    gain = min(1.0, .999 / max(peak, 1e-12))
    written = []
    for row in candidates:
        start, end = round(row["start"] * 16000), round(row["end"] * 16000)
        if end > len(signal) or start < 0 or end <= start:
            raise ValueError("Prepared interval exceeds resampled source")
        sf.write(audio_dir / row["audio_filename"], signal[start:end] * gain, 16000, format="FLAC", subtype="PCM_24")
        written.append({**row, "duration_seconds": (end - start) / 16000, "source_gain": gain})
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "data/recovery-v1")
    p.add_argument("--heldout", choices=["2", "5"], default="5")
    p.add_argument("--style", choices=["raw", "dev_style"], default="dev_style")
    p.add_argument("--dev-repeats", type=int, default=8)
    p.add_argument("--heldout-jember-sessions", type=int, default=0,
                   help="Hold out whole Jember sessions to measure Javanese-only WER. "
                        "Dev folds are 100%% code-switched, so without this the "
                        "Javanese half of the task is never evaluated.")
    p.add_argument("--seed", type=int, default=20260905)
    args = p.parse_args()
    if args.dev_repeats < 1:
        p.error("dev-repeats must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing preparation")
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True)
    rows, _, _ = canonical_inputs(ROOT / "data/subsets/dev/all", ROOT / "data/subsets/jember/all")
    with (ROOT / "reports/private/timestamp_quarantine_candidates.csv").open() as f:
        excluded = {f'{r["session"]}:{r["tsv_data_row"]}' for r in csv.DictReader(f)}
    anomalies = json.loads((ROOT / "reports/private/timestamp_anomalies.json").read_text())
    excluded.update(r["id"] for session in anomalies["intervals"].values() for r in session["overlap_rows"])
    sessions = defaultdict(list)
    for row in rows["train"]:
        if row["id"] not in excluded:
            sessions[row["source"]].append(row)
    heldout_sessions = set()
    if args.heldout_jember_sessions:
        keys = sorted(sessions)
        if args.heldout_jember_sessions >= len(keys):
            raise ValueError("Cannot hold out every Jember session")
        random.Random(args.seed).shuffle(keys)
        heldout_sessions = set(keys[: args.heldout_jember_sessions])
    jobs = []
    for source, rs in sessions.items():
        held = source in heldout_sessions
        rs.sort(key=lambda r: int(r["id"].split(":")[-1]))
        candidates = []
        for row in (() if held else rs):
            candidates.append({"audio_filename": "j-" + row["id"].replace(":", "-") + ".flac",
                               "source": "jember", "view": "short", "source_rows": [row["id"]],
                               "start": row["start"], "end": row["end"], "speaker": "", "convo_id": "",
                               "language": "unknown", "transcript": target_text(row["text"], "jember", args.style),
                               "original_transcript": row["text"], "split": "train"})
        for window in contiguous_windows(rs):
            identity = "-".join(r["id"] for r in window)
            original = " ".join(r["text"] for r in window)
            candidates.append({"audio_filename": "w-" + hashlib.sha256(identity.encode()).hexdigest()[:20] + ".flac",
                               "source": "jember", "view": "long", "source_rows": [r["id"] for r in window],
                               "start": window[0]["start"], "end": window[-1]["end"], "speaker": "", "convo_id": "",
                               "language": "unknown", "transcript": target_text(original, "jember", args.style),
                               "original_transcript": original,
                               "split": "validation" if held else "train"})
        jobs.append((source, candidates, audio_dir))
    prepared = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(encode_session, jobs), 1):
            prepared.extend(result)
            if i % 25 == 0:
                print(f"Prepared {i}/{len(jobs)} Jember sessions", flush=True)
    dev_metrics = {r["id"]: r for r in json.loads((ROOT / "reports/private/clip_metrics.json").read_text())["dev"]}
    for row in rows["dev"]:
        name = "dev-" + row["id"]
        shutil.copy2(row["source"], audio_dir / name)
        prepared.append({"audio_filename": name, "source": "development", "view": "original", "source_rows": [row["id"]],
                         "start": 0, "end": dev_metrics[row["id"]]["decoded_duration_seconds"],
                         "duration_seconds": dev_metrics[row["id"]]["decoded_duration_seconds"], "source_gain": 1.0,
                         "speaker": row["speaker"], "convo_id": row["conversation"], "language": row["language"],
                         "transcript": target_text(row["text"], "development", args.style), "original_transcript": row["text"],
                         "split": "validation" if row["conversation"] == args.heldout else "train"})
    train = [r for r in prepared if r["split"] == "train"]
    validation = [r for r in prepared if r["split"] == "validation"]
    train_speakers = {r["speaker"] for r in train if r["source"] == "development"}
    val_speakers = {r["speaker"] for r in validation}
    assert not train_speakers & val_speakers
    assert not {r["audio_filename"] for r in train} & {r["audio_filename"] for r in validation}
    leaked = {r["source_rows"][0].split(":")[0] for r in train if r["source"] == "jember"} & heldout_sessions
    assert not leaked, f"held-out Jember sessions leaked into training: {sorted(leaked)}"
    training_views = [r for r in train for _ in range(args.dev_repeats if r["source"] == "development" else 1)]
    for split, selected in [("train", training_views), ("validation", validation)]:
        with (args.output / f"{split}.qwen.jsonl").open("w") as f:
            for r in selected:
                language = "Indonesian" if r["language"] == "ind" else "None"
                f.write(json.dumps({"audio": "audio/" + r["audio_filename"],
                                    "text": f'language {language}<asr_text>{r["transcript"]}'}, ensure_ascii=False) + "\n")
    columns = ["audio_filename", "transcript", "speaker", "convo_id", "language", "split", "source", "view", "duration_seconds"]
    with (args.output / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(prepared)
    write_json(args.output / "private_provenance.json", prepared)
    receipt = {"heldout_conversation": args.heldout,
               "heldout_jember_sessions": sorted(heldout_sessions),
               "heldout_jember_session_count": len(heldout_sessions),
               "validation_javanese_rows": sum(r["source"] == "jember" for r in validation),
               "validation_codeswitch_rows": sum(r["source"] == "development" for r in validation),
               "train_development_speakers": sorted(train_speakers),
               "validation_speakers": sorted(val_speakers), "style": args.style, "dev_repeats": args.dev_repeats,
               "excluded_source_rows": sorted(excluded), "unique_training_examples": len(train),
               "training_rows_with_repeats": len(training_views), "validation_rows": len(validation),
               "training_audio_hours_with_views_and_repeats": sum(r["duration_seconds"] for r in training_views) / 3600,
               "validation_audio_hours": sum(r["duration_seconds"] for r in validation) / 3600,
               "jember_short": sum(r["view"] == "short" for r in train), "jember_long": sum(r["view"] == "long" for r in train),
               "training_development_unique": sum(r["source"] == "development" for r in train),
               "metadata_sha256": digest_file(args.output / "metadata.csv"),
               "train_jsonl_sha256": digest_file(args.output / "train.qwen.jsonl"),
               "validation_jsonl_sha256": digest_file(args.output / "validation.qwen.jsonl"),
               "preparation_script_sha256": digest_file(Path(__file__)),
               "audio_note": "Jember resampled to 16 kHz, per-recording peak headroom where needed, PCM24 FLAC; development originals copied unchanged."}
    write_json(args.output / "PREPARATION_RECEIPT.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != "excluded_source_rows"}, indent=2))


if __name__ == "__main__":
    main()
