#!/usr/bin/env python3
"""Turn the permissively licensed MDC corpora into extra training rows.

The competition's own data is small: Jember is 9.97 h and development is 372
clips, and validation shows the Javanese half carries most of the remaining
error. This script adds two kinds of audio that the attempt has never used:

* Common Voice Spontaneous Javanese (CC0) -- spontaneous Javanese whose 21 s
  median already sits in the scored duration band;
* two CC-BY-SA Indonesian podcast corpora -- 23.6 h of transcribed spontaneous
  conversation, more than double the whole Jember corpus, aimed at the
  code-switched half.

The output holds training rows only. Validation always comes from the primary
prepared dataset, so adding data cannot move the measuring stick.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_data import digest_file, text_key, timestamp, write_json
from scripts.prepare_recovery import language_tag, sliding_windows, target_text

# Segment boundaries in the podcast transcripts are whole seconds around natural
# turns, so a small gap between consecutive rows is silence rather than dropped
# speech. Anything wider starts a new run.
PODCAST_GAP_TOLERANCE = 1.0

SOURCES = {
    # East Javanese read speech. Lumajang Regency borders Jember, and Bojonegoro
    # is the same province, so these are the closest dialect match on the
    # platform. Read speech transferred poorly on its own in an earlier attempt
    # (OpenSLR-35, standard Central Javanese), so these are only ever a
    # supplement to spontaneous in-domain audio, never the primary source.
    "tts_lumajang": {"kind": "nested_zip", "language": "jav", "style": "dev_style",
                     "licence": "CC-BY-SA-4.0"},
    "tts_bojonegoro": {"kind": "nested_zip", "language": "jav", "style": "dev_style",
                       "licence": "CC-BY-SA-4.0"},
    "cv_jv": {"kind": "clips", "language": "jav", "style": "dev_style",
              "transcripts": "sps-corpus-4.0-2026-06-12-jv/ss-corpus-jv.tsv",
              "audio_dir": "sps-corpus-4.0-2026-06-12-jv/audios",
              "columns": {"name": "audio_file", "text": "transcription"},
              "licence": "CC0-1.0"},
    "cv_id_scripted": {"kind": "clips", "language": "ind", "style": "raw",
                       "transcripts": "cv-corpus-26.0-2026-06-12/id/validated.tsv",
                       "audio_dir": "cv-corpus-26.0-2026-06-12/id/clips",
                       "columns": {"name": "path", "text": "sentence"},
                       "licence": "CC0-1.0"},
    "podcast_hari": {"kind": "segments", "language": "ind", "style": "raw",
                     "transcripts": "2/Transcript_Hari Minggoean.tsv",
                     "audio_dir": "2", "suffix": ".mp3", "licence": "CC-BY-SA-4.0"},
    "podcast_homostoria": {"kind": "segments", "language": "ind", "style": "raw",
                           "transcripts": "homostoria/transcript_homostoria.tsv",
                           "audio_dir": "homostoria", "suffix": ".mp3",
                           "licence": "CC-BY-SA-4.0"},
}


def member_index(archive: tarfile.TarFile):
    """Case-insensitive member lookup.

    Three Hari Minggoean episodes are stored as "Hari MInggoean_04x.mp3" while
    the transcript spells them "Minggoean", so an exact lookup silently loses
    them. tarfile also raises KeyError rather than returning None for a name it
    does not hold, which is why callers go through this index.
    """
    return {name.lower(): name for name in archive.getnames()}


def open_member(archive: tarfile.TarFile, index, name: str):
    resolved = index.get(name.lower())
    if resolved is None:
        return None
    try:
        return archive.extractfile(resolved)
    except KeyError:
        return None


def read_tsv_member(archive: tarfile.TarFile, name: str):
    handle = archive.extractfile(name)
    if handle is None:
        raise ValueError(f"Missing transcript table {name}")
    return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"),
                               delimiter="\t"))


def decode_pcm(data: bytes) -> np.ndarray:
    """Decode any container ffmpeg understands to 16 kHz mono float32."""
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0", "-ar", "16000",
         "-ac", "1", "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, check=True)
    return np.frombuffer(process.stdout, dtype="<f4")


def write_segment(signal, start, end, gain, path):
    begin, stop = round(start * 16000), round(end * 16000)
    if begin < 0 or stop <= begin or stop > len(signal):
        return None
    sf.write(path, signal[begin:stop] * gain, 16000, format="FLAC", subtype="PCM_24")
    return (stop - begin) / 16000


def prepare_clips(archive, spec, audio_dir, name, args):
    """Whole-clip corpora: one training row per transcribed recording."""
    columns = spec["columns"]
    index = member_index(archive)
    rows = []
    for record in read_tsv_member(archive, spec["transcripts"]):
        text = (record.get(columns["text"]) or "").strip()
        if not text:
            continue  # Spontaneous-speech corpora ship prompts without answers.
        member = f"{spec['audio_dir']}/{record[columns['name']]}"
        handle = open_member(archive, index, member)
        if handle is None:
            print(f"{name}: no audio member for {member}", flush=True)
            continue
        signal = decode_pcm(handle.read())
        duration = len(signal) / 16000
        # Whole-clip corpora are kept over a wider band than windows are built to.
        # The window bounds exist to match the scored duration profile when
        # assembling segments; applying them here would just discard recordings,
        # and this is the only spontaneous Javanese audio outside Jember.
        if not args.clip_min_seconds <= duration <= args.clip_max_seconds:
            continue
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        gain = min(1.0, .999 / max(peak, 1e-12))
        filename = f"{name}-{Path(record[columns['name']]).stem}.flac"
        written = write_segment(signal, 0, duration, gain, audio_dir / filename)
        if written is None:
            continue
        rows.append({"audio_filename": filename, "source": name, "view": "clip",
                     "language": spec["language"], "speaker": "", "convo_id": "",
                     "transcript": target_text(text, "jember" if spec["language"] == "jav"
                                               else "development", spec["style"]),
                     "original_transcript": text, "duration_seconds": written,
                     "split": "train"})
    return rows


def prepare_nested_zips(archive, spec, audio_dir, name, args):
    """TTS corpora shipped as a tar of per-speaker zips, each with its own mapping.

    Every zip holds hash-named .webm recordings plus a `mapping.tsv` of
    `audio_filename` and `sentence`. Layouts differ between corpora -- one
    nests under a directory, the other is flat -- so audio members are matched
    on basename rather than on a constructed path.
    """
    import io
    import zipfile

    rows = []
    zip_members = [m for m in archive.getnames() if m.lower().endswith(".zip")]
    skipped = [m for m in archive.getnames()
               if m.lower().endswith(".rar")]
    if skipped:
        print(f"{name}: skipping {len(skipped)} rar archive(s), no extractor available: "
              f"{skipped[:2]}", flush=True)
    for member in sorted(zip_members):
        handle = archive.extractfile(member)
        if handle is None:
            continue
        try:
            bundle = zipfile.ZipFile(io.BytesIO(handle.read()))
        except zipfile.BadZipFile:
            print(f"{name}: unreadable zip {member}", flush=True)
            continue
        names = bundle.namelist()
        mapping = next((n for n in names if n.lower().endswith("mapping.tsv")), None)
        if mapping is None:
            print(f"{name}: no mapping.tsv in {member}", flush=True)
            continue
        by_base = {PurePosixPath(n).name: n for n in names}
        table = list(csv.DictReader(
            io.TextIOWrapper(bundle.open(mapping), encoding="utf-8", errors="replace"),
            delimiter="\t"))
        kept = 0
        for record in table:
            text = (record.get("sentence") or "").strip()
            audio_name = (record.get("audio_filename") or "").strip()
            if not text or not audio_name:
                continue
            inner = by_base.get(PurePosixPath(audio_name).name)
            if inner is None:
                continue
            signal = decode_pcm(bundle.read(inner))
            duration = len(signal) / 16000
            if not args.clip_min_seconds <= duration <= args.clip_max_seconds:
                continue
            peak = float(np.max(np.abs(signal))) if signal.size else 0.0
            gain = min(1.0, .999 / max(peak, 1e-12))
            filename = f"{name}-{PurePosixPath(member).stem}-{PurePosixPath(audio_name).stem}.flac"
            written = write_segment(signal, 0, duration, gain, audio_dir / filename)
            if written is None:
                continue
            rows.append({"audio_filename": filename, "source": name, "view": "clip",
                         "language": spec["language"], "speaker": PurePosixPath(member).stem,
                         "convo_id": "", "transcript": target_text(text, "jember", spec["style"]),
                         "original_transcript": text, "duration_seconds": written,
                         "split": "train"})
            kept += 1
        print(f"{name}: {member} -> {kept}/{len(table)} clips", flush=True)
    return rows


def prepare_segments(archive, spec, audio_dir, name, args):
    """Timestamped corpora: windows built the same way as the Jember long view."""
    grouped = {}
    for index, record in enumerate(read_tsv_member(archive, spec["transcripts"])):
        text = (record.get("Text") or "").strip()
        episode = (record.get("Audio file") or "").strip()
        if not text or not episode or not record.get("Start") or not record.get("End"):
            continue
        try:
            start, end = timestamp(record["Start"]), timestamp(record["End"])
        except ValueError:
            continue
        if end <= start:
            continue
        grouped.setdefault(episode, []).append(
            {"id": f"{episode}:{index}", "start": start, "end": end, "text": text})
    index = member_index(archive)
    rows = []
    for episode, records in sorted(grouped.items()):
        member = f"{spec['audio_dir']}/{episode}{spec.get('suffix', '.mp3')}"
        handle = open_member(archive, index, member)
        if handle is None:
            print(f"{name}: no audio for {episode}", flush=True)
            continue
        signal = decode_pcm(handle.read())
        peak = float(np.max(np.abs(signal))) if signal.size else 0.0
        gain = min(1.0, .999 / max(peak, 1e-12))
        windows = sliding_windows(records, args.min_seconds, args.max_seconds,
                                  args.stride, tolerance=PODCAST_GAP_TOLERANCE)
        for window in windows:
            start, end = window[0]["start"], window[-1]["end"]
            if end * 16000 > len(signal):
                continue  # Transcript runs past the audio; drop rather than pad.
            text = " ".join(r["text"] for r in window)
            filename = f"{name}-{episode.replace(' ', '_')}-{int(start)}-{int(end)}.flac"
            written = write_segment(signal, start, end, gain, audio_dir / filename)
            if written is None:
                continue
            rows.append({"audio_filename": filename, "source": name, "view": "window",
                         "language": spec["language"], "speaker": "", "convo_id": "",
                         "transcript": target_text(text, "development", spec["style"]),
                         "original_transcript": text, "duration_seconds": written,
                         "split": "train"})
        print(f"{name}: {episode} -> {len(windows)} windows", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--archive", action="append", required=True, metavar="NAME=PATH",
                   help=f"Repeat per corpus; NAME one of {sorted(SOURCES)}")
    p.add_argument("--min-seconds", type=float, default=18.0)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--stride", type=float, default=0.5)
    p.add_argument("--clip-min-seconds", type=float, default=5.0)
    p.add_argument("--clip-max-seconds", type=float, default=30.0)
    p.add_argument("--language-tags", action="store_true")
    p.add_argument("--prompt", default="")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing preparation")
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True)

    jobs = []
    for entry in args.archive:
        name, _, path = entry.partition("=")
        if name not in SOURCES:
            raise ValueError(f"Unknown corpus {name!r}; expected one of {sorted(SOURCES)}")
        archive_path = Path(path)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing archive for {name}: {archive_path}")
        jobs.append((name, archive_path))

    def build(job):
        name, archive_path = job
        spec = SOURCES[name]
        with tarfile.open(archive_path) as archive:
            builder = {"clips": prepare_clips, "segments": prepare_segments,
                       "nested_zip": prepare_nested_zips}[spec["kind"]]
            rows = builder(archive, spec, audio_dir, name, args)
        print(f"{name}: {len(rows)} rows, "
              f"{sum(r['duration_seconds'] for r in rows) / 3600:.2f} h", flush=True)
        return name, rows, digest_file(archive_path)

    prepared, digests = [], {}
    with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        for name, rows, digest in pool.map(build, jobs):
            prepared.extend(rows)
            digests[name] = {"archive_sha256": digest, "rows": len(rows),
                             "hours": sum(r["duration_seconds"] for r in rows) / 3600,
                             "licence": SOURCES[name]["licence"],
                             "language": SOURCES[name]["language"]}
    if not prepared:
        raise ValueError("No external rows were produced")
    names = [r["audio_filename"] for r in prepared]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate audio filenames in the external preparation")

    with (args.output / "train.qwen.jsonl").open("w") as handle:
        for row in prepared:
            record = {"audio": "audio/" + row["audio_filename"],
                      "text": f'language {language_tag(row["language"], args.language_tags)}'
                              f'<asr_text>{row["transcript"]}'}
            if args.prompt:
                record["prompt"] = args.prompt
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    columns = ["audio_filename", "transcript", "speaker", "convo_id", "language",
               "split", "source", "view", "duration_seconds"]
    with (args.output / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(prepared)
    receipt = {"sources": digests, "rows": len(prepared), "validation_rows": 0,
               "training_rows_with_repeats": len(prepared),
               "training_audio_hours_with_views_and_repeats":
                   sum(r["duration_seconds"] for r in prepared) / 3600,
               "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
               "clip_min_seconds": args.clip_min_seconds, "clip_max_seconds": args.clip_max_seconds,
               "stride": args.stride, "language_tags": args.language_tags,
               "prompt": args.prompt,
               "metadata_sha256": digest_file(args.output / "metadata.csv"),
               "train_jsonl_sha256": digest_file(args.output / "train.qwen.jsonl"),
               "preparation_script_sha256": digest_file(Path(__file__)),
               "audio_note": "Resampled to 16 kHz mono, per-recording peak headroom, PCM24 FLAC."}
    write_json(args.output / "PREPARATION_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
