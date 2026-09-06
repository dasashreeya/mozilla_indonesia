#!/usr/bin/env python3
"""Single-GPU Qwen adaptation with response-only loss and real epoch WER."""
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
sys.path.insert(0, str(ROOT / "vendor"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    import torch
    from datasets import load_dataset
    from qwen_asr import Qwen3ASRModel
    from transformers import TrainerCallback, TrainingArguments, set_seed
    from mdc_asr.wer import ErrorCounts, word_errors
    from qwen3_asr_sft import (
        CastFloatInputsTrainer, DataCollatorForQwen3ASRFinetuning,
        make_preprocess_fn_prefix_only, patch_outer_forward,
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-acc", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--volume-name", default="")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Refusing to overwrite an existing training run")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(20260905)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    volume = None
    if args.volume_name:
        import modal
        volume = modal.Volume.from_name(args.volume_name)

    def commit():
        if volume is not None:
            volume.commit()

    receipt = json.loads((args.data / "PREPARATION_RECEIPT.json").read_text())
    with (args.data / "metadata.csv").open() as f:
        metadata = list(csv.DictReader(f))
    validation = [r for r in metadata if r["split"] == "validation"]
    assert len(validation) == receipt["validation_rows"]
    assert not set(receipt["train_development_speakers"]) & set(receipt["validation_speakers"])
    rebased = args.output / "train.inputs.jsonl"
    count = 0
    with rebased.open("w") as f:
        for line in (args.data / "train.qwen.jsonl").read_text().splitlines():
            record = json.loads(line)
            path = args.data / record["audio"]
            if not path.is_file() or path.parent != args.data / "audio":
                raise ValueError("Invalid training audio path")
            record["audio"] = str(path)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    assert count == receipt["training_rows_with_repeats"]
    save_json(args.output / "RUN_CONFIG.json", {"model": str(args.model), "data": receipt,
              "epochs": args.epochs, "batch_size": args.batch_size, "grad_acc": args.grad_acc,
              "learning_rate": args.lr, "max_new_tokens": 512, "seed": 20260905,
              "method": "full fine-tuning, BF16, response-only supervision, gradient checkpointing",
              "selection": "held-out conversation generation WER; baseline must be beaten"})
    print(f"Loading pinned Qwen base; train rows={count}, validation={len(validation)}", flush=True)
    wrapper = Qwen3ASRModel.from_pretrained(str(args.model), dtype=torch.bfloat16,
              device_map="cuda:0", max_inference_batch_size=8, max_new_tokens=512, local_files_only=True)
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)
    # The pinned checkpoint ships generation_config.json with do_sample=false and
    # temperature=1e-06. transformers validates that pair strictly on every
    # save_pretrained, so both this callback's save and the Trainer's own epoch
    # save raise ValueError and abort the run. Drop the sampling-only flags
    # instead of enabling sampling: evaluation and submission both decode greedily.
    for _flag in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
        if not model.generation_config.do_sample and getattr(model.generation_config, _flag, None) is not None:
            setattr(model.generation_config, _flag, None)
    print(f"Processor padding side: {processor.tokenizer.padding_side}", flush=True)

    def evaluate(label):
        prior = model.training
        model.eval()
        aggregate = ErrorCounts()
        groups = defaultdict(ErrorCounts)
        preds = []
        empty = 0
        started = time.monotonic()
        try:
            with torch.inference_mode():
                for start in range(0, len(validation), 8):
                    batch = validation[start:start + 8]
                    outputs = wrapper.transcribe(audio=[str(args.data / "audio" / r["audio_filename"]) for r in batch], language=None)
                    if len(outputs) != len(batch):
                        raise ValueError("Prediction count mismatch")
                    for row, output in zip(batch, outputs):
                        prediction = " ".join((output.text or "").splitlines()).strip()
                        counts = word_errors(row["transcript"], prediction)
                        aggregate += counts
                        empty += not bool(prediction)
                        for key in ["language", "speaker", "convo_id"]:
                            groups[f"{key}:{row[key]}"] += counts
                        groups["duration:over30" if float(row["duration_seconds"]) > 30 else "duration:up_to30"] += counts
                        preds.append({"audio_filename": row["audio_filename"], "transcript": prediction})
                    if (start // 8) % 10 == 0:
                        print(f"{label}: decoded {min(start + 8, len(validation))}/{len(validation)}", flush=True)
        finally:
            model.train(prior)
        if not aggregate.reference_words or not math.isfinite(aggregate.wer):
            raise RuntimeError("Invalid validation denominator or WER")
        with (args.output / f"{label}.predictions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio_filename", "transcript"])
            w.writeheader(); w.writerows(preds)
        result = {"wer": aggregate.wer, **asdict(aggregate), "empty_predictions": empty,
                  "clips": len(validation), "seconds": time.monotonic() - started,
                  "groups": {g: {"wer": c.wer, **asdict(c)} for g, c in groups.items()}}
        save_json(args.output / f"{label}.metrics.json", result)
        print(f"{label}: WER={result['wer']:.6f}; S={aggregate.substitutions} D={aggregate.deletions} I={aggregate.insertions} N={aggregate.reference_words}", flush=True)
        commit()
        return result

    baseline = evaluate("baseline")
    raw = load_dataset("json", data_files={"train": str(rebased)})["train"]
    dataset = raw.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    collator = DataCollatorForQwen3ASRFinetuning(processor=processor)
    probe_batch = collator([dataset[0], dataset[len(dataset) - 1]])
    supervised = (probe_batch["labels"] != -100).sum(dim=1).tolist()
    if min(supervised) <= 0:
        raise RuntimeError("A training example has no supervised target tokens")
    with torch.no_grad():
        probe_gpu = {k: v.to(model.device, dtype=model.dtype if v.is_floating_point() else v.dtype) for k, v in probe_batch.items()}
        loss = model(**probe_gpu).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite preflight training loss")
        print(f"Supervision preflight: target lengths={supervised}; finite loss={loss.item():.5f}", flush=True)
    del probe_gpu, probe_batch, loss
    torch.cuda.empty_cache()

    class EpochWER(TrainerCallback):
        def __init__(self):
            self.best = baseline["wer"]
            self.best_dir = str(args.model)
            self.stale = 0
            self.history = []

        def on_epoch_end(self, training_args, state, control, **kwargs):
            epoch = int(round(state.epoch))
            metrics = evaluate(f"epoch-{epoch}")
            self.history.append({"epoch": epoch, "step": state.global_step, "wer": metrics["wer"]})
            if metrics["wer"] < self.best:
                self.best = metrics["wer"]
                self.stale = 0
                destination = args.output / f"best-epoch-{epoch}"
                model.save_pretrained(destination, safe_serialization=True)
                processor.save_pretrained(destination)
                self.best_dir = str(destination)
            else:
                self.stale += 1
            save_json(args.output / "BEST_MODEL.json", {"path": self.best_dir, "wer": self.best,
                      "baseline_wer": baseline["wer"], "improves_baseline": self.best < baseline["wer"], "epochs": self.history})
            commit()
            if self.stale >= 2:
                control.should_training_stop = True
            return control

        def on_save(self, training_args, state, control, **kwargs):
            processor.save_pretrained(args.output / f"checkpoint-{state.global_step}")
            commit()
            return control

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                safe = {k: float(v) for k, v in logs.items() if isinstance(v, (int, float)) and math.isfinite(v)}
                save_json(args.output / "PROGRESS.json", {"step": state.global_step, "epoch": state.epoch, **safe})
                commit()

    callback = EpochWER()
    train_args = TrainingArguments(output_dir=str(args.output), per_device_train_batch_size=args.batch_size,
                 gradient_accumulation_steps=args.grad_acc, learning_rate=args.lr, num_train_epochs=args.epochs,
                 max_steps=args.max_steps, lr_scheduler_type="cosine", warmup_ratio=.08, bf16=True, fp16=False,
                 gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                 max_grad_norm=1.0, logging_steps=5, logging_first_step=True, save_strategy="epoch", save_total_limit=1,
                 eval_strategy="no", dataloader_num_workers=4, dataloader_pin_memory=True,
                 dataloader_persistent_workers=True, remove_unused_columns=False, report_to="none", seed=20260905)
    trainer = CastFloatInputsTrainer(model=model, args=train_args, train_dataset=dataset, data_collator=collator,
                                    processing_class=processor.tokenizer, callbacks=[callback])
    # The patched forward intentionally consumes only model inputs; Trainer must
    # perform the standard per-microbatch gradient-accumulation loss scaling.
    trainer.model_accepts_loss_kwargs = False
    print("Starting optimizer steps on audited in-domain data", flush=True)
    trainer.train()
    trainer.save_state()
    result = {"status": "completed", "steps": trainer.state.global_step, "baseline_wer": baseline["wer"],
              "best_wer": callback.best, "best_model": callback.best_dir, "improves_baseline": callback.best < baseline["wer"],
              "history": callback.history}
    save_json(args.output / "TRAIN_RESULT.json", result)
    commit()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
