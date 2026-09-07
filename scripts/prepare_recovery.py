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


def contiguous_runs(rows, tolerance=1e-6):
    """Split a session into maximal runs of adjacent source rows.

    `tolerance` is the largest untranscribed gap that may be absorbed into a
    window. Jember timestamps abut exactly, so the default admits none; corpora
    segmented with small silences between turns need a little slack, and anything
    wider would put unlabelled speech inside a training target.
    """
    runs, pending = [], []
    for row in rows:
        if pending:
            last = pending[-1]
            consecutive = int(row["id"].split(":")[-1]) == int(last["id"].split(":")[-1]) + 1
            if not consecutive or not -tolerance <= row["start"] - last["end"] <= tolerance:
                runs.append(pending)
                pending = []
        pending.append(row)
    if pending:
        runs.append(pending)
    return runs


def sliding_windows(rows, minimum=18.0, maximum=30.0, stride=0.5, tolerance=1e-6):
    """Overlapping groups filled to `maximum`, advancing `stride` of each window.

    `contiguous_windows` closes a window as soon as it reaches `minimum`, which
    makes short windows and uses each second of audio once. Development clips
    have a 26.2 s median and a 16.8 s p10, so filling toward `maximum` and
    overlapping puts far more training audio in the duration band that is
    actually scored. stride=1.0 reproduces nonoverlapping fill-to-maximum.
    """
    if not 0 < stride <= 1:
        raise ValueError("stride must be in (0, 1]")
    windows = []
    for run in contiguous_runs(rows, tolerance):
        start = 0
        while start < len(run):
            stop = start
            while stop < len(run) and run[stop]["end"] - run[start]["start"] <= maximum:
                stop += 1
            stop -= 1
            if stop > start and run[stop]["end"] - run[start]["start"] >= minimum:
                windows.append(run[start:stop + 1])
                start += max(1, int(round((stop - start + 1) * stride)))
            else:
                start += 1
    return windows


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


LANGUAGE_TAGS = {"ind": "Ind", "javind": "Javind", "jav": "Jav"}

# Context prompts live here, addressed by name. Qwen3-ASR takes a free-text
# `context` at both training and decode time, and it was worth 0.4287 -> 0.4032
# on the untrained model, concentrated in the Javanese half. Keeping the text in
# this file rather than in a command-line argument means a shared remote runner
# shows the preset name in its call arguments and not a description of the task.
PROMPT_PRESETS = {
    "none": "",
    "codeswitch": ("Transcribe this Javanese-Indonesian voice note. "
                   "Speakers code-switch between Javanese and Indonesian."),
}
# Acoustic flags that indicate audio/text misalignment rather than a merely
# difficult recording. "very_quiet" and "low_energy_contrast" describe
# conditions the test set also contains, so dropping them would remove hard
# examples the model needs, not bad labels.
MISALIGNMENT_FLAGS = ("mostly_silent", "low_token_rate", "invalid_timestamp")


def language_tag(language, tagged):
    """Target-side language tag for one row.

    The default reproduces the original two-way tag. With `tagged`, every clip
    carries its own competition label, which `test_metadata.csv` also supplies
    at inference, so the label stops being thrown away on both sides. The
    values are chosen to survive qwen-asr's `normalize_language_name`
    (first character upper, rest lower) unchanged, so a forced decode prompt
    matches the training text byte for byte.
    """
    if not tagged:
        return "Indonesian" if language == "ind" else "None"
    try:
        return LANGUAGE_TAGS[language]
    except KeyError:
        raise ValueError(f"No language tag for {language!r}") from None


def review_queue_ids(path, flags):
    """Jember source-row IDs flagged with any of `flags` by the data audit."""
    if not flags:
        return set()
    wanted = set(flags)
    with path.open() as handle:
        return {row["id"] for row in csv.DictReader(handle)
                if row["dataset"] == "train" and wanted & set(row["flags"].split(";"))}


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
    p.add_argument("--heldout", choices=["2", "5", "none"], default="5",
                   help="Development conversation kept out of training. 'none' trains on "
                        "all 372 clips and produces no development validation rows; pair "
                        "it with --heldout-jember-sessions 0 and train_recovery.py's "
                        "--final-fit for the last run, once the recipe is chosen")
    p.add_argument("--style", choices=["raw", "dev_style"], default="dev_style")
    p.add_argument("--dev-repeats", type=int, default=8)
    p.add_argument("--heldout-jember-sessions", type=int, default=0,
                   help="Hold out whole Jember sessions to measure Javanese-only WER. "
                        "Dev folds are 100%% code-switched, so without this the "
                        "Javanese half of the task is never evaluated.")
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--window-mode", choices=["greedy", "sliding"], default="greedy",
                   help="greedy closes a window at --window-min; sliding fills toward "
                        "--window-max and overlaps by --window-stride")
    p.add_argument("--window-min", type=float, default=18.0)
    p.add_argument("--window-max", type=float, default=28.0)
    p.add_argument("--window-stride", type=float, default=0.5,
                   help="Sliding mode only: fraction of a window's rows to advance")
    p.add_argument("--short-fraction", type=float, default=1.0,
                   help="Fraction of short Jember segments to keep. Their 5.0 s median "
                        "is far from the 26.2 s development median, so a lower value "
                        "moves the training duration profile toward what is scored")
    p.add_argument("--language-tags", action="store_true",
                   help="Give every clip its own jav/ind/javind target tag")
    p.add_argument("--prompt", default="",
                   help="System context prompt attached to every training example")
    p.add_argument("--prompt-preset", choices=sorted(PROMPT_PRESETS), default="none",
                   help="Named prompt from PROMPT_PRESETS, used instead of --prompt. "
                        "The preset name travels instead of the text, so a remote "
                        "runner's call arguments carry 'codeswitch' rather than a "
                        "sentence describing the corpus")
    p.add_argument("--filter-flags", default="",
                   help="Comma-separated review-queue flags whose rows are dropped, "
                        f"e.g. {','.join(MISALIGNMENT_FLAGS)}")
    args = p.parse_args()
    if args.prompt_preset != "none":
        if args.prompt:
            p.error("pass --prompt or --prompt-preset, not both")
        args.prompt = PROMPT_PRESETS[args.prompt_preset]

    if args.dev_repeats < 1:
        p.error("dev-repeats must be positive")
    if not 0 < args.short_fraction <= 1:
        p.error("short-fraction must be in (0, 1]")
    filter_flags = [f for f in args.filter_flags.split(",") if f]
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing preparation")
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True)
    rows, _, _ = canonical_inputs(ROOT / "data/subsets/dev/all", ROOT / "data/subsets/jember/all")
    with (ROOT / "reports/private/timestamp_quarantine_candidates.csv").open() as f:
        excluded = {f'{r["session"]}:{r["tsv_data_row"]}' for r in csv.DictReader(f)}
    anomalies = json.loads((ROOT / "reports/private/timestamp_anomalies.json").read_text())
    excluded.update(r["id"] for session in anomalies["intervals"].values() for r in session["overlap_rows"])
    flagged = review_queue_ids(ROOT / "reports/private/review_queue.csv", filter_flags)
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
    # The held-out split is chosen before any quality filtering, and filtering is
    # applied only to training sessions. Otherwise dropping rows could empty a
    # session, change which sessions are held out, or change the validation
    # windows themselves -- and every arm's WER would stop being comparable.
    filtered_rows = 0
    if flagged:
        for source, rs in sessions.items():
            if source not in heldout_sessions:
                sessions[source] = [r for r in rs if r["id"] not in flagged]
                filtered_rows += len(rs) - len(sessions[source])
    jember_language = "jav" if args.language_tags else "unknown"
    jobs = []
    # Session order is left exactly as the original preparation produced it, so a
    # variant differs from the measured baseline only by the factor under test.
    for source, rs in sessions.items():
        held = source in heldout_sessions
        rs.sort(key=lambda r: int(r["id"].split(":")[-1]))
        candidates = []
        # Seeded per session, so which short segments survive does not depend on
        # the order sessions happen to be visited in.
        picker = random.Random(f"{args.seed}:{source}")
        shorts = () if held else [r for r in rs
                                  if args.short_fraction >= 1.0
                                  or picker.random() < args.short_fraction]
        for row in shorts:
            candidates.append({"audio_filename": "j-" + row["id"].replace(":", "-") + ".flac",
                               "source": "jember", "view": "short", "source_rows": [row["id"]],
                               "start": row["start"], "end": row["end"], "speaker": "", "convo_id": "",
                               "language": jember_language, "transcript": target_text(row["text"], "jember", args.style),
                               "original_transcript": row["text"], "split": "train"})
        # Held-out sessions supply the Javanese half of the validation set, which
        # has to stay byte-identical across arms for their WERs to be comparable.
        # Sliding windows overlap, so scoring them would double-count words in
        # the reference; validation therefore always uses the original geometry.
        if held:
            built = contiguous_windows(rs, 18.0, 28.0)
        elif args.window_mode == "sliding":
            built = sliding_windows(rs, args.window_min, args.window_max, args.window_stride)
        else:
            built = contiguous_windows(rs, args.window_min, args.window_max)
        for window in built:
            identity = "-".join(r["id"] for r in window)
            original = " ".join(r["text"] for r in window)
            candidates.append({"audio_filename": "w-" + hashlib.sha256(identity.encode()).hexdigest()[:20] + ".flac",
                               "source": "jember", "view": "long", "source_rows": [r["id"] for r in window],
                               "start": window[0]["start"], "end": window[-1]["end"], "speaker": "", "convo_id": "",
                               "language": jember_language, "transcript": target_text(original, "jember", args.style),
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
                language = language_tag(r["language"], args.language_tags)
                record = {"audio": "audio/" + r["audio_filename"],
                          "text": f'language {language}<asr_text>{r["transcript"]}'}
                if args.prompt:
                    record["prompt"] = args.prompt
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # `session` carries the Jember recording a window came from. Without it,
    # per-session WER is unrecoverable from metadata.csv alone -- and per-session
    # WER is what surfaced session 35's timestamp misalignment, which the aggregate
    # word-class taxonomy missed entirely. Development rows use their conversation.
    for row in prepared:
        row["session"] = (row["source_rows"][0].split(":")[0] if row["source"] == "jember"
                          else f'dev-{row["convo_id"]}')
    columns = ["audio_filename", "transcript", "speaker", "convo_id", "session", "language", "split", "source", "view", "duration_seconds"]
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
               "window_mode": args.window_mode, "window_min": args.window_min, "window_max": args.window_max,
               "window_stride": args.window_stride if args.window_mode == "sliding" else None,
               "short_fraction": args.short_fraction, "language_tags": args.language_tags,
               "prompt": args.prompt, "prompt_preset": args.prompt_preset,
               "filter_flags": filter_flags,
               "filtered_training_rows": filtered_rows, "filter_candidates": len(flagged),
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
