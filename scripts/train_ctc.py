#!/usr/bin/env python3
"""Train a complementary CTC recognizer on the existing, fixed session split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdc_asr.normalization import normalize_text
from mdc_asr.wer import ErrorCounts, word_errors


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def target_text(text):
    # Preserve the scorer's case and accents, while removing annotation markup.
    return " ".join(normalize_text(text).split())


def training_vocabulary(texts):
    chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'-.")
    for text in texts:
        chars.update(target_text(text))
    chars.discard(" ")
    chars.discard("|")
    tokens = ["[PAD]", "[UNK]", "|"] + sorted(chars)
    return {token: index for index, token in enumerate(tokens)}


def read_split(data):
    receipt = json.loads((data / "PREPARATION_RECEIPT.json").read_text())
    raw = (data / "metadata.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["metadata_sha256"]:
        raise ValueError("Prepared metadata checksum mismatch")
    with (data / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not train or len(validation) != receipt["validation_rows"]:
        raise ValueError("Invalid prepared split")
    for key in ("audio_filename", "session"):
        left = {r[key] for r in train if r.get(key)}
        right = {r[key] for r in validation if r.get(key)}
        if left & right:
            raise ValueError(f"Training/validation overlap in {key}")
    speakers = lambda selected: {r["speaker"] for r in selected
                                 if r["source"] == "development" and r.get("speaker")}
    if speakers(train) & speakers(validation):
        raise ValueError("Development speakers overlap")
    for row in rows:
        name = row["audio_filename"]
        if Path(name).name != name or not (data / "audio" / name).is_file():
            raise ValueError(f"Invalid audio path: {name}")
    return receipt, train, validation


def summarize(rows, texts):
    if len(rows) != len(texts):
        raise ValueError("Prediction count mismatch")
    total = ErrorCounts()
    groups = defaultdict(ErrorCounts)
    for row, text in zip(rows, texts):
        errors = word_errors(row["transcript"], text)
        total += errors
        for key in ("language", "session", "source"):
            if row.get(key):
                groups[f"{key}:{row[key]}"] += errors
    if total.reference_words == 0:
        raise ValueError("Empty validation reference")
    result = {"wer": total.wer, **asdict(total), "clips": len(rows),
              "empty_predictions": sum(not text.strip() for text in texts),
              "groups": {k: {"wer": v.wer, **asdict(v)} for k, v in sorted(groups.items())}}
    jav = groups.get("language:jav")
    other = ErrorCounts()
    for key, value in groups.items():
        if key.startswith("language:") and key != "language:jav":
            other += value
    result["selection_wer"] = (0.49 * jav.wer + 0.51 * other.wer
                               if jav and jav.reference_words and other.reference_words else total.wer)
    result["selection_note"] = "49% Javanese heuristic; not a public-score forecast"
    return result


def prepare(data, base, output):
    import librosa
    import numpy as np
    from datasets import Dataset, DatasetDict
    from transformers import (SeamlessM4TFeatureExtractor, Wav2Vec2CTCTokenizer,
                              Wav2Vec2BertProcessor)

    receipt, train, validation = read_split(data)
    if output.exists():
        raise ValueError("Refusing to replace a prepared CTC dataset")
    output.mkdir(parents=True)
    vocabulary = training_vocabulary(r["transcript"] for r in train)
    save_json(output / "processor/vocab.json", vocabulary)
    tokenizer = Wav2Vec2CTCTokenizer(str(output / "processor/vocab.json"),
                unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
                bos_token=None, eos_token=None, do_lower_case=False)
    extractor = SeamlessM4TFeatureExtractor.from_pretrained(str(base), local_files_only=True)
    processor = Wav2Vec2BertProcessor(feature_extractor=extractor, tokenizer=tokenizer)
    processor.save_pretrained(output / "processor")

    def features(row):
        audio, _ = librosa.load(data / "audio" / row["audio_filename"], sr=16000, mono=True)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError(f"Invalid waveform {row['audio_filename']}")
        inputs = extractor(audio, sampling_rate=16000).input_features[0]
        labels = tokenizer(target_text(row["transcript"]), add_special_tokens=False).input_ids
        required_frames = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
        return {"input_features": inputs, "labels": labels, "length": len(inputs),
                "ctc_valid": bool(labels) and required_frames <= len(inputs)}

    built = {}
    dropped = []
    for name, rows in (("train", train), ("validation", validation)):
        dataset = Dataset.from_list(rows).map(features, num_proc=8,
                    remove_columns=list(rows[0]), desc=f"CTC features: {name}")
        if name == "train":
            indices = []
            for index, row in enumerate(rows):
                if not dataset[index]["ctc_valid"]:
                    dropped.append(row["audio_filename"])
                    continue
                repeat = receipt.get("dev_repeats", 1) if row["source"] == "development" else 1
                indices.extend([index] * repeat)
            dataset = dataset.select(indices)
        built[name] = dataset.remove_columns("ctc_valid")
    DatasetDict(built).save_to_disk(str(output / "features"))
    save_json(output / "validation_rows.json", validation)
    save_json(output / "PREPARATION_RECEIPT.json", {
        "source": str(data), "source_metadata_sha256": receipt["metadata_sha256"],
        "vocabulary_sha256": hashlib.sha256((output / "processor/vocab.json").read_bytes()).hexdigest(),
        "training_rows": len(built["train"]), "validation_rows": len(validation),
        "vocabulary_size": len(tokenizer), "dropped_unalignable_training": dropped,
        "vocabulary_source": "training transcripts only plus fixed ASCII alphabet",
        "normalization": "official scorer, preserving noninitial case and accents"})
    # A language model, when used, must be built only from these training texts.
    (output / "lm_train.txt").write_text(
        "\n".join(target_text(r["transcript"]) for r in train if target_text(r["transcript"])) + "\n",
        encoding="utf-8")
    print((output / "PREPARATION_RECEIPT.json").read_text(), flush=True)


def train(args):
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import (Trainer, TrainerCallback, TrainingArguments,
                              Wav2Vec2BertForCTC, Wav2Vec2BertProcessor, set_seed)

    set_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.output.exists() and not args.resume:
        raise ValueError("Refusing to overwrite a training run; use --resume explicitly")
    args.output.mkdir(parents=True, exist_ok=True)
    volume = None
    if args.volume_name:
        import modal
        volume = modal.Volume.from_name(args.volume_name)

    def commit():
        if volume:
            volume.commit()

    dataset = load_from_disk(str(args.data / "features"))
    validation = json.loads((args.data / "validation_rows.json").read_text())
    if not validation:
        raise ValueError("This experimental trainer requires held-out validation")
    processor = Wav2Vec2BertProcessor.from_pretrained(args.data / "processor", local_files_only=True)
    model = Wav2Vec2BertForCTC.from_pretrained(str(args.model), local_files_only=True,
        vocab_size=len(processor.tokenizer), pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean", ctc_zero_infinity=True,
        mask_time_prob=0.05, mask_time_length=10, mask_feature_prob=0.0,
        attention_dropout=0.0, hidden_dropout=0.0, final_dropout=0.1,
        layerdrop=0.0, ignore_mismatched_sizes=True)
    model.to("cuda")

    def collate(rows):
        batch = processor.feature_extractor.pad(
            [{"input_features": r["input_features"]} for r in rows],
            padding=True, return_attention_mask=True, return_tensors="pt")
        labels = processor.tokenizer.pad([{"input_ids": r["labels"]} for r in rows],
                                         padding=True, return_tensors="pt")
        batch["labels"] = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return batch

    def evaluate():
        model.eval()
        texts, emissions = [], []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(validation), args.batch_size):
                rows = [dataset["validation"][i] for i in
                        range(start, min(start + args.batch_size, len(validation)))]
                batch = collate(rows)
                batch.pop("labels")
                batch = {k: v.to("cuda") for k, v in batch.items()}
                logits = model(**batch).logits.float()
                lengths = model._get_feat_extract_output_lengths(batch["attention_mask"].sum(-1))
                for item, length in zip(logits, lengths):
                    item = item[:int(length)].cpu().numpy()
                    texts.append(processor.tokenizer.decode(item.argmax(-1)))
                    emissions.append(item.astype(np.float16))
        model.train()
        return summarize(validation, texts), texts, emissions

    started = time.monotonic()
    receipt = json.loads((args.data / "PREPARATION_RECEIPT.json").read_text())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["prepared_data"] = receipt
    save_json(args.output / "RUN_CONFIG.json", config)
    history_path = args.output / "HISTORY.json"
    history = json.loads(history_path.read_text()) if args.resume and history_path.exists() else []

    class Validation(TrainerCallback):
        def __init__(self):
            self.best = min((r["selection_wer"] for r in history), default=float("inf"))

        def on_epoch_end(self, training_args, state, control, **kwargs):
            metrics, texts, emissions = evaluate()
            metrics.update({"epoch": state.epoch, "step": state.global_step,
                            "seconds": time.monotonic() - started})
            history.append(metrics)
            save_json(history_path, history)
            save_json(args.output / f"eval-{state.global_step}.json", metrics)
            if metrics["selection_wer"] < self.best:
                self.best = metrics["selection_wer"]
                model.save_pretrained(args.output / "best", safe_serialization=True)
                processor.save_pretrained(args.output / "best")
                np.savez_compressed(args.output / "best-emissions.npz",
                                    **{str(i): item for i, item in enumerate(emissions)})
                with (args.output / "best.predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["audio_filename", "transcript"])
                    writer.writeheader()
                    writer.writerows({"audio_filename": r["audio_filename"], "transcript": t}
                                     for r, t in zip(validation, texts))
                save_json(args.output / "BEST_MODEL.json", {"path": str(args.output / "best"), **metrics})
            commit()
            print(json.dumps({k: v for k, v in metrics.items() if k != "groups"}), flush=True)
            if time.monotonic() - started > args.max_hours * 3600:
                control.should_training_stop = True
            return control

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                finite = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                save_json(args.output / "PROGRESS.json", {"step": state.global_step, "epoch": state.epoch,
                          "max_steps": state.max_steps, "seconds": time.monotonic() - started, **finite})
                commit()

        def on_save(self, training_args, state, control, **kwargs):
            processor.save_pretrained(args.output / f"checkpoint-{state.global_step}")
            commit()

    train_args = TrainingArguments(output_dir=str(args.output), num_train_epochs=args.epochs,
        max_steps=args.max_steps, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr,
        warmup_ratio=0.1, lr_scheduler_type="linear", weight_decay=0.01,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch", save_total_limit=2, eval_strategy="no",
        logging_steps=25, logging_first_step=True, disable_tqdm=True,
        dataloader_num_workers=4, dataloader_pin_memory=True,
        group_by_length=True, length_column_name="length", remove_unused_columns=False,
        report_to="none", seed=args.seed)
    trainer = Trainer(model=model, args=train_args, train_dataset=dataset["train"],
                      data_collator=collate, processing_class=processor, callbacks=[Validation()])
    trainer.model_accepts_loss_kwargs = False
    probe = collate([dataset["train"][0], dataset["train"][1]])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**{k: v.to("cuda") for k, v in probe.items()}).loss
    if not torch.isfinite(loss) or loss.item() <= 0:
        raise ValueError("Invalid CTC preflight loss")
    print(f"CTC preflight loss={loss.item():.5f}; rows={len(dataset['train'])}", flush=True)
    del probe, loss
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    result = {"status": "completed", "steps": trainer.state.global_step,
              "best": json.loads((args.output / "BEST_MODEL.json").read_text()),
              "seconds": time.monotonic() - started}
    save_json(args.output / "TRAIN_RESULT.json", result)
    commit()
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-acc", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--volume-name", default="")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.data, args.model, args.output)
    else:
        train(args)


if __name__ == "__main__":
    main()
