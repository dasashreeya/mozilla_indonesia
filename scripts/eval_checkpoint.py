#!/usr/bin/env python3
"""Score any checkpoint on a prepared holdout without spending a training run.

One harness for both backends so a Qwen arm and a Whisper arm are measured on
exactly the same clips, references and grouping. Everything here is decode-side,
so it answers questions -- does beam search help, does forcing the competition
language label help, is Whisper actually better in domain -- for the cost of a
decode pass rather than a fine-tune.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "submission_src"))

SAMPLE_RATE = 16000
# Whisper's encoder is fixed at 30 s, so longer clips are decoded in windows.
WHISPER_WINDOW_SECONDS = 30.0
CHUNK_SECONDS = 28.0
# Competition label -> Whisper language token. Whisper has a native Javanese
# token, which is the reason this backend is worth measuring at all.
WHISPER_LANGUAGES = {"jav": "jw", "ind": "id", "javind": "id"}
QWEN_LANGUAGES = {"jav": "Jav", "ind": "Ind", "javind": "Javind"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def detect_backend(model_dir: Path) -> str:
    model_type = json.loads((model_dir / "config.json").read_text()).get("model_type")
    if model_type == "qwen3_asr":
        return "qwen"
    if model_type == "whisper":
        return "whisper"
    raise ValueError(f"Unsupported model_type {model_type!r} in {model_dir}")


def load_rows(data: Path, split: str, limit: int):
    with (data / "metadata.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == split]
    if not rows:
        raise ValueError(f"No rows in split {split!r}")
    return rows[:limit] if limit else rows


def qwen_decoder(args, model_dir: Path, tokens):
    import torch
    from qwen_asr import Qwen3ASRModel

    from train_recovery import register_competition_languages

    wrapper = Qwen3ASRModel.from_pretrained(
        str(model_dir), dtype=torch.bfloat16, device_map="cuda:0",
        max_inference_batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        local_files_only=True)
    if args.force_language:
        register_competition_languages()
    from qwen_decoding import install_probability_ensemble, set_beams
    set_beams(wrapper.model, args.beams)
    extra_wrappers = []
    for path in args.ensemble_model:
        extra = Qwen3ASRModel.from_pretrained(str(path), dtype=torch.bfloat16,
                    device_map="cuda:0", local_files_only=True)
        if extra.processor.tokenizer.get_vocab() != wrapper.processor.tokenizer.get_vocab():
            raise ValueError("Ensemble tokenizer mapping mismatch")
        set_beams(extra.model, 1)
        extra_wrappers.append(extra)
    if extra_wrappers:
        install_probability_ensemble(wrapper.model, [w.model for w in extra_wrappers],
                                     args.ensemble_weight or None)

    def decode(batch):
        paths = [str(args.data / "audio" / r["audio_filename"]) for r in batch]
        languages = [tokens[r["language"]] for r in batch] if args.force_language else None
        outputs = wrapper.transcribe(audio=paths, context=[args.prompt] * len(batch),
                                     language=languages)
        return [" ".join((o.text or "").splitlines()).strip() for o in outputs]

    return decode


def whisper_decoder(args, model_dir: Path, tokens):
    import librosa
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    model.generation_config.forced_decoder_ids = None
    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
    minimum_samples = SAMPLE_RATE // 2

    def generate(waveforms, language):
        batch = processor.feature_extractor(waveforms, sampling_rate=SAMPLE_RATE,
                                            return_tensors="pt", return_attention_mask=True)
        kwargs = {"task": "transcribe", "num_beams": args.beams, "do_sample": False,
                  "max_new_tokens": min(args.max_new_tokens, 440)}
        if language:
            kwargs["language"] = language
        mask = batch.get("attention_mask")
        if mask is not None:
            kwargs["attention_mask"] = mask.to("cuda:0")
        with torch.no_grad():
            ids = model.generate(batch.input_features.to("cuda:0", dtype=torch.bfloat16), **kwargs)
        return [" ".join(str(t).split()) for t in processor.tokenizer.batch_decode(ids, skip_special_tokens=True)]

    def decode(batch):
        texts = []
        for row in batch:
            waveform, _ = librosa.load(str(args.data / "audio" / row["audio_filename"]),
                                       sr=SAMPLE_RATE, mono=True)
            language = tokens[row["language"]] if args.force_language else None
            if len(waveform) / SAMPLE_RATE > WHISPER_WINDOW_SECONDS + 0.5:
                windows = [waveform[s:s + chunk_samples]
                           for s in range(0, len(waveform), chunk_samples)]
                windows = [w for w in windows if len(w) >= minimum_samples]
                texts.append(" ".join(generate(windows, language)).strip())
            else:
                texts.append(generate([waveform], language)[0])
        return texts

    return decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--label", default="eval")
    p.add_argument("--split", default="validation")
    p.add_argument("--backend", choices=["auto", "qwen", "whisper"], default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--beams", type=int, default=1)
    p.add_argument("--ensemble-model", type=Path, action="append", default=[])
    p.add_argument("--ensemble-weight", type=float, action="append", default=[])
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--prompt", default="")
    p.add_argument("--force-language", action="store_true")
    p.add_argument("--language-map", default="",
                   help='JSON overriding the label-to-token map, e.g. \'{"unknown": "jw"}\'. '
                        "Datasets prepared before --language-tags label Jember rows "
                        '"unknown", and guessing a token for them would silently '
                        "mis-condition half the holdout")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--volume-name", default="")
    args = p.parse_args()
    if args.beams < 1:
        p.error("beams must be at least 1")

    from mdc_asr.wer import ErrorCounts, word_errors

    volume = None
    if args.volume_name:
        import modal
        volume = modal.Volume.from_name(args.volume_name)

    backend = detect_backend(args.model) if args.backend == "auto" else args.backend
    rows = load_rows(args.data, args.split, args.limit)
    tokens = dict(QWEN_LANGUAGES if backend == "qwen" else WHISPER_LANGUAGES)
    tokens.update(json.loads(args.language_map) if args.language_map else {})
    unknown = {r["language"] for r in rows} - set(tokens)
    if args.force_language and unknown:
        raise ValueError(f"No {backend} language token for {sorted(unknown)}; "
                         "prepare the dataset with --language-tags")
    print(f"{args.label}: backend={backend} clips={len(rows)} beams={args.beams} "
          f"force_language={args.force_language} prompt={args.prompt!r}", flush=True)
    decode = (qwen_decoder if backend == "qwen" else whisper_decoder)(args, args.model, tokens)

    aggregate = ErrorCounts()
    groups = defaultdict(ErrorCounts)
    predictions = []
    empty = 0
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = decode(batch)
        if len(texts) != len(batch):
            raise ValueError("Prediction count mismatch")
        for row, text in zip(batch, texts):
            counts = word_errors(row["transcript"], text)
            aggregate += counts
            empty += not bool(text)
            groups[f"language:{row['language']}"] += counts
            groups[f"source:{row['source']}"] += counts
            # Older preparations predate the column; skip rather than mislabel.
            if row.get("session"):
                groups[f"session:{row['session']}"] += counts
            groups["duration:over30" if float(row["duration_seconds"]) > 30 else "duration:up_to30"] += counts
            predictions.append({"audio_filename": row["audio_filename"], "transcript": text})
        done = min(start + args.batch_size, len(rows))
        if (start // args.batch_size) % 5 == 0:
            print(f"{args.label}: decoded {done}/{len(rows)} "
                  f"elapsed={time.monotonic() - started:.0f}s", flush=True)
    if not aggregate.reference_words or not math.isfinite(aggregate.wer):
        raise RuntimeError("Invalid validation denominator or WER")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_filename", "transcript"])
        writer.writeheader(); writer.writerows(predictions)
    result = {"label": args.label, "model": str(args.model), "backend": backend,
              "ensemble_models": [str(path) for path in args.ensemble_model],
              "ensemble_weights": args.ensemble_weight or None,
              "beams": args.beams, "force_language": args.force_language, "prompt": args.prompt,
              "language_map": tokens if args.force_language else None,
              "split": args.split, "clips": len(rows), "empty_predictions": empty,
              "wer": aggregate.wer, **asdict(aggregate),
              "seconds": time.monotonic() - started,
              "groups": {g: {"wer": c.wer, **asdict(c)} for g, c in sorted(groups.items())}}
    save_json(args.output, result)
    if volume is not None:
        volume.commit()
    print(f"{args.label}: WER={result['wer']:.6f}; S={aggregate.substitutions} "
          f"D={aggregate.deletions} I={aggregate.insertions} N={aggregate.reference_words}", flush=True)
    for name, counts in sorted(groups.items()):
        if name.startswith("language:"):
            print(f"  {name}: WER={counts.wer:.6f} N={counts.reference_words}", flush=True)


if __name__ == "__main__":
    main()
