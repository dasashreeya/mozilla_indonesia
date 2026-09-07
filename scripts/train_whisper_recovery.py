#!/usr/bin/env python3
"""Fine-tune Whisper large-v3 on the same data and holdout as the Qwen arms.

Whisper is worth a real measurement here for one concrete reason: it has a
native Javanese language token, `<|jw|>`, and roughly half the test set is
Javanese-only, which is where the remaining error lives. Qwen3-ASR's supported
language list contains no Javanese at all. The previous attempt dismissed
Whisper on a five-clip smoke test -- 22 errors against 23 -- which cannot
separate two systems. This settles it on 509 clips instead.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16000
# Whisper's encoder takes a fixed 30 s window; longer clips are decoded in
# consecutive chunks and their transcripts joined.
WHISPER_WINDOW_SECONDS = 30.0
CHUNK_SECONDS = 28.0
WHISPER_LANGUAGES = {"jav": "jw", "ind": "id", "javind": "id", "unknown": "jw"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    import librosa
    import torch
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
                              WhisperForConditionalGeneration, WhisperProcessor, set_seed)
    from mdc_asr.wer import ErrorCounts, word_errors

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-acc", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--force-language", action="store_true")
    p.add_argument("--volume-name", default="")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing training run")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(20260905)
    torch.backends.cuda.matmul.allow_tf32 = True

    volume = None
    if args.volume_name:
        import modal
        volume = modal.Volume.from_name(args.volume_name)

    def commit():
        if volume is not None:
            volume.commit()

    receipt = json.loads((args.data / "PREPARATION_RECEIPT.json").read_text())
    with (args.data / "metadata.csv").open() as handle:
        metadata = list(csv.DictReader(handle))
    validation = [r for r in metadata if r["split"] == "validation"]
    assert len(validation) == receipt["validation_rows"]
    train_rows = [r for r in metadata if r["split"] == "train"]
    # The encoder truncates at 30 s, so a longer training clip would be paired
    # with a transcript whose tail was never heard. Those rows are dropped
    # rather than silently mistrained on.
    keep = [r for r in train_rows if float(r["duration_seconds"]) <= WHISPER_WINDOW_SECONDS]
    dropped = len(train_rows) - len(keep)
    repeats = receipt.get("dev_repeats", 1)
    train_rows = [r for r in keep for _ in range(repeats if r["source"] == "development" else 1)]
    print(f"Whisper training rows={len(train_rows)} (dropped {dropped} over "
          f"{WHISPER_WINDOW_SECONDS:.0f}s); validation={len(validation)}", flush=True)

    processor = WhisperProcessor.from_pretrained(str(args.model), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(args.model), local_files_only=True, dtype=torch.bfloat16).to("cuda:0")
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    save_json(args.output / "RUN_CONFIG.json",
              {"model": str(args.model), "data": receipt, "epochs": args.epochs,
               "batch_size": args.batch_size, "grad_acc": args.grad_acc, "learning_rate": args.lr,
               "force_language": args.force_language, "dropped_over_window": dropped,
               "training_rows": len(train_rows), "seed": 20260905,
               "method": "full fine-tuning, BF16, gradient checkpointing, per-clip language token"})

    feature_extractor, tokenizer = processor.feature_extractor, processor.tokenizer

    def language_of(row):
        return WHISPER_LANGUAGES[row["language"]] if args.force_language else None

    def collate(rows):
        waveforms = [librosa.load(str(args.data / "audio" / r["audio_filename"]),
                                  sr=SAMPLE_RATE, mono=True)[0] for r in rows]
        features = feature_extractor(waveforms, sampling_rate=SAMPLE_RATE,
                                     return_tensors="pt").input_features
        sequences = []
        for row in rows:
            language = language_of(row)
            if language:
                tokenizer.set_prefix_tokens(language=language, task="transcribe")
            sequences.append(tokenizer(row["transcript"]).input_ids)
        width = max(len(s) for s in sequences)
        labels = torch.full((len(sequences), width), -100, dtype=torch.long)
        for i, sequence in enumerate(sequences):
            labels[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        # The decoder start token is prepended by the model, so it is stripped here.
        if (labels[:, 0] == tokenizer.convert_tokens_to_ids("<|startoftranscript|>")).all():
            labels = labels[:, 1:]
        return {"input_features": features.to(torch.bfloat16), "labels": labels}

    chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)

    def transcribe(rows):
        texts = []
        for row in rows:
            waveform, _ = librosa.load(str(args.data / "audio" / row["audio_filename"]),
                                       sr=SAMPLE_RATE, mono=True)
            windows = ([waveform] if len(waveform) / SAMPLE_RATE <= WHISPER_WINDOW_SECONDS + .5
                       else [w for w in (waveform[s:s + chunk_samples]
                                         for s in range(0, len(waveform), chunk_samples))
                             if len(w) >= SAMPLE_RATE // 2])
            features = feature_extractor(windows, sampling_rate=SAMPLE_RATE,
                                         return_tensors="pt").input_features
            kwargs = {"task": "transcribe", "num_beams": 1, "do_sample": False, "max_new_tokens": 440}
            language = language_of(row)
            if language:
                kwargs["language"] = language
            with torch.no_grad():
                ids = model.generate(features.to("cuda:0", dtype=torch.bfloat16), **kwargs)
            parts = tokenizer.batch_decode(ids, skip_special_tokens=True)
            texts.append(" ".join(" ".join(str(part).split()) for part in parts).strip())
        return texts

    def evaluate(label):
        prior = model.training
        model.eval()
        aggregate, groups, preds, empty = ErrorCounts(), defaultdict(ErrorCounts), [], 0
        started = time.monotonic()
        try:
            for start in range(0, len(validation), 8):
                batch = validation[start:start + 8]
                for row, text in zip(batch, transcribe(batch)):
                    counts = word_errors(row["transcript"], text)
                    aggregate += counts
                    empty += not bool(text)
                    groups[f"language:{row['language']}"] += counts
                    groups[f"source:{row['source']}"] += counts
                    preds.append({"audio_filename": row["audio_filename"], "transcript": text})
                if (start // 8) % 10 == 0:
                    print(f"{label}: decoded {min(start + 8, len(validation))}/{len(validation)}", flush=True)
        finally:
            model.train(prior)
        if not aggregate.reference_words or not math.isfinite(aggregate.wer):
            raise RuntimeError("Invalid validation denominator or WER")
        with (args.output / f"{label}.predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["audio_filename", "transcript"])
            writer.writeheader(); writer.writerows(preds)
        result = {"wer": aggregate.wer, **asdict(aggregate), "empty_predictions": empty,
                  "clips": len(validation), "seconds": time.monotonic() - started,
                  "groups": {g: {"wer": c.wer, **asdict(c)} for g, c in sorted(groups.items())}}
        save_json(args.output / f"{label}.metrics.json", result)
        print(f"{label}: WER={result['wer']:.6f}; S={aggregate.substitutions} "
              f"D={aggregate.deletions} I={aggregate.insertions} N={aggregate.reference_words}", flush=True)
        commit()
        return result

    baseline = evaluate("baseline")

    class EpochWER(TrainerCallback):
        def __init__(self):
            self.best = baseline["wer"]
            self.best_dir = str(args.model)
            self.history = []

        def on_epoch_end(self, training_args, state, control, **kwargs):
            epoch = int(round(state.epoch))
            metrics = evaluate(f"epoch-{epoch}")
            self.history.append({"epoch": epoch, "step": state.global_step, "wer": metrics["wer"]})
            if metrics["wer"] < self.best:
                self.best = metrics["wer"]
                destination = args.output / f"best-epoch-{epoch}"
                model.save_pretrained(destination, safe_serialization=True)
                processor.save_pretrained(destination)
                self.best_dir = str(destination)
            save_json(args.output / "BEST_MODEL.json",
                      {"path": self.best_dir, "wer": self.best, "baseline_wer": baseline["wer"],
                       "improves_baseline": self.best < baseline["wer"], "epochs": self.history})
            commit()
            return control

    callback = EpochWER()
    train_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr,
        num_train_epochs=args.epochs, lr_scheduler_type="cosine", warmup_ratio=.08,
        bf16=True, fp16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, max_grad_norm=1.0,
        logging_steps=10, save_strategy="no", eval_strategy="no", dataloader_num_workers=4,
        remove_unused_columns=False, report_to="none", seed=20260905)
    trainer = Seq2SeqTrainer(model=model, args=train_args, train_dataset=train_rows,
                             data_collator=collate, callbacks=[callback])
    print("Starting Whisper optimizer steps", flush=True)
    trainer.train()
    result = {"status": "completed", "steps": trainer.state.global_step,
              "baseline_wer": baseline["wer"], "best_wer": callback.best,
              "best_model": callback.best_dir,
              "improves_baseline": callback.best < baseline["wer"], "history": callback.history}
    save_json(args.output / "TRAIN_RESULT.json", result)
    commit()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
