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


# Tags the fine-tune teaches the model to emit. qwen-asr validates a forced
# language against a module-level list, so the competition's own labels are
# registered there before any decode that forces one.
COMPETITION_LANGUAGES = ("Jav", "Ind", "Javind")
# Mirrors scripts/prepare_recovery.LANGUAGE_TAGS. Duplicated because the training
# image ships only src/, vendor/ and this file, not the preparation script.
LANGUAGE_TAGS = {"ind": "Ind", "javind": "Javind", "jav": "Jav"}


def register_competition_languages():
    """Let qwen-asr force `language Jav<asr_text>`-style prompts.

    `validate_language` reads `SUPPORTED_LANGUAGES` from its own module at call
    time, so extending that list in place is enough; the list is mutated rather
    than rebound so every importer sees the addition.
    """
    from qwen_asr.inference import utils

    missing = [tag for tag in COMPETITION_LANGUAGES if tag not in utils.SUPPORTED_LANGUAGES]
    utils.SUPPORTED_LANGUAGES.extend(missing)
    return missing


def spec_augment(features, lengths, freq_width, time_width, probability, generator):
    """Mask mel bands and time spans in place, as the 1st-place recipe does.

    `features` is (batch, mel, frames) or (batch, frames, mel); the mel axis is
    the smaller of the two. Masking is applied per example, only inside that
    example's real frames, so padding never becomes a training signal.
    """
    import torch

    if features.ndim != 3:
        raise ValueError(f"Expected 3-D input features, got {tuple(features.shape)}")
    mel_axis = 1 if features.shape[1] <= features.shape[2] else 2
    time_axis = 3 - mel_axis
    n_mel, n_frames = features.shape[mel_axis], features.shape[time_axis]
    for i in range(features.shape[0]):
        if torch.rand((), generator=generator).item() >= probability:
            continue
        usable = int(lengths[i]) if lengths is not None else n_frames
        usable = max(1, min(usable, n_frames))
        # Both masks are confined to the example's real frames. A band that also
        # spanned the padding would be a no-op there, so confining it keeps every
        # applied mask acting on actual signal.
        width = int(torch.randint(0, max(1, min(freq_width, n_mel)) + 1, (), generator=generator))
        if width:
            start = int(torch.randint(0, n_mel - width + 1, (), generator=generator))
            index = [i, slice(None), slice(None)]
            index[mel_axis] = slice(start, start + width)
            index[time_axis] = slice(0, usable)
            features[tuple(index)] = 0.0
        span = int(round(time_width * usable))
        span = int(torch.randint(0, max(1, span) + 1, (), generator=generator))
        if span:
            start = int(torch.randint(0, usable - span + 1, (), generator=generator))
            index = [i, slice(None), slice(None)]
            index[time_axis] = slice(start, start + span)
            features[tuple(index)] = 0.0
    return features


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
    p.add_argument("--lora-rank", type=int, default=0,
                   help="0 keeps full fine-tuning; >0 trains LoRA adapters and merges "
                        "them into the saved weights so evaluation and packaging are unchanged")
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--spec-augment", type=float, default=0.0,
                   help="Per-example probability of masking one mel band and one time span")
    p.add_argument("--spec-freq-width", type=int, default=15)
    p.add_argument("--spec-time-width", type=float, default=0.05)
    p.add_argument("--force-language", action="store_true",
                   help="Decode with each clip's own competition language tag, which "
                        "test_metadata.csv also supplies. Requires a tagged dataset")
    p.add_argument("--extra-data", type=Path, action="append", default=[],
                   help="Additional training-only dataset built by prepare_external.py. "
                        "Validation always comes from --data, so extra corpora cannot "
                        "move the measuring stick")
    p.add_argument("--extra-fraction", type=float, default=1.0,
                   help="Deterministically keep this fraction of each extra corpus, so "
                        "out-of-domain audio can be capped instead of dominating")
    p.add_argument("--keep-epochs", action="store_true",
                   help="Save every epoch's weights, not only improving ones, so the "
                        "checkpoints can be averaged afterwards")
    p.add_argument("--evals-per-epoch", type=int, default=1,
                   help="Validate this many times per epoch instead of only at epoch "
                        "ends. An epoch here is ~550 optimizer steps, so the default of "
                        "1 gives three measurements over a whole run and can miss the "
                        "WER-optimal point entirely; it also yields too few checkpoints "
                        "to average. See docs/kb/11-audit-and-methodology-v2.md F2/F3")
    p.add_argument("--final-fit", action="store_true",
                   help="Train on every labelled row with no validation split and no "
                        "checkpoint selection. For the last run only, after the recipe "
                        "and step count have been chosen by the validated arms")
    p.add_argument("--patience", type=int, default=2,
                   help="Stop after this many consecutive non-improving validations. "
                        "Must exceed the number of validations the learning-rate "
                        "schedule needs to finish, or the run stops before the "
                        "low-learning-rate phase the cosine decay was configured for")
    args = p.parse_args()
    if args.lora_rank < 0:
        p.error("lora-rank must be non-negative")
    if not 0 <= args.spec_augment <= 1:
        p.error("spec-augment must be a probability")
    if not 0 < args.extra_fraction <= 1:
        p.error("extra-fraction must be in (0, 1]")
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
    prompt = receipt.get("prompt", "")
    tagged = bool(receipt.get("language_tags"))
    if args.force_language and not tagged:
        raise ValueError("--force-language needs a dataset prepared with --language-tags")
    rebased = args.output / "train.inputs.jsonl"

    def rebase(directory, handle, fraction=1.0):
        """Copy one dataset's rows out with absolute, validated audio paths."""
        import random as _random
        picker = _random.Random(f"20260905:{directory.name}")
        written = 0
        for line in (directory / "train.qwen.jsonl").read_text().splitlines():
            if fraction < 1.0 and picker.random() >= fraction:
                continue
            record = json.loads(line)
            path = directory / record["audio"]
            if not path.is_file() or path.parent != directory / "audio":
                raise ValueError(f"Invalid training audio path in {directory}")
            record["audio"] = str(path)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        return written

    def rebase_metadata(directory, handle, fraction, tagged, prompt, style_map=None):
        """Emit an extra corpus's training rows in the primary arm's convention.

        `style_map` comes from the extra corpus's own preparation receipt and maps
        a source name to its transcription-convention tag. When it covers a row,
        it wins: the whole point of a style tag is that the external corpus is
        *not* written in the primary arm's convention, so folding it back into
        the in-domain language tag would undo the separation. Rows from a corpus
        the map does not cover keep the language tag.
        """
        import random as _random
        style_map = style_map or {}
        picker = _random.Random(f"20260905:{directory.name}")
        with (directory / "metadata.csv").open() as source:
            rows = [r for r in csv.DictReader(source) if r["split"] == "train"]
        written = 0
        for row in rows:
            if fraction < 1.0 and picker.random() >= fraction:
                continue
            path = directory / "audio" / row["audio_filename"]
            if not path.is_file():
                raise ValueError(f"Missing extra training audio: {path}")
            tag = style_map.get(row["source"]) or (
                LANGUAGE_TAGS[row["language"]] if tagged else
                ("Indonesian" if row["language"] == "ind" else "None"))
            record = {"audio": str(path), "text": f'language {tag}<asr_text>{row["transcript"]}'}
            if prompt:
                record["prompt"] = prompt
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        return written

    extra_receipts = {}
    with rebased.open("w") as f:
        count = rebase(args.data, f)
        assert count == receipt["training_rows_with_repeats"]
        for directory in args.extra_data:
            extra = json.loads((directory / "PREPARATION_RECEIPT.json").read_text())
            if extra.get("validation_rows"):
                raise ValueError(f"{directory} carries validation rows; extra data must be train-only")
            # Rebuilt from the extra corpus's metadata rather than copied from its
            # own jsonl, so one decoded copy of the audio serves every arm whatever
            # tagging and prompt that arm uses.
            style_map = extra.get("style_tag_by_source") or {}
            added = rebase_metadata(directory, f, args.extra_fraction, tagged, prompt, style_map)
            extra_receipts[directory.name] = {"rows": added, "sources": extra.get("sources"),
                                              "style_tags": style_map}
            count += added
            print(f"Added {added} rows from {directory.name}", flush=True)
    method = ("LoRA r={} alpha={}".format(args.lora_rank, args.lora_alpha) if args.lora_rank
              else "full fine-tuning")
    save_json(args.output / "RUN_CONFIG.json", {"model": str(args.model), "data": receipt,
              "epochs": args.epochs, "batch_size": args.batch_size, "grad_acc": args.grad_acc,
              "learning_rate": args.lr, "max_new_tokens": 512, "seed": 20260905,
              "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
              "spec_augment": args.spec_augment, "force_language": args.force_language,
              "prompt": prompt, "language_tags": tagged, "keep_epochs": args.keep_epochs,
              "extra_data": extra_receipts, "extra_fraction": args.extra_fraction,
              "total_training_rows": count,
              "method": f"{method}, BF16, response-only supervision, gradient checkpointing",
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
    if args.force_language:
        print(f"Registered forced language tags: {register_competition_languages()}", flush=True)

    peft_model = None
    if args.lora_rank:
        from peft import LoraConfig, get_peft_model
        # The 1st-place Pasketti recipe trains the audio encoder too, so the
        # targets are matched by suffix across both towers rather than being
        # restricted to the decoder.
        config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout, bias="none", task_type=None,
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj"])
        peft_model = get_peft_model(model, config)
        # With every base weight frozen, the embedding output carries no grad, so
        # gradient checkpointing would hand the backward pass a graph with nothing
        # to differentiate ("element 0 of tensors does not require grad").
        for target in (model, getattr(model, "thinker", None)):
            if target is not None and hasattr(target, "enable_input_require_grads"):
                target.enable_input_require_grads()
        adapted = sum(1 for name, _ in model.named_modules() if name.endswith("lora_A.default"))
        if not adapted:
            raise RuntimeError("LoRA matched no modules; target_modules does not fit this model")
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in peft_model.parameters())
        print(f"LoRA: {adapted} adapted modules, {trainable/1e6:.1f}M of {total/1e6:.1f}M "
              f"parameters trainable ({100*trainable/total:.2f}%)", flush=True)

    def save_weights(destination):
        """Persist one epoch. LoRA saves the adapter; the merge happens once, at the end."""
        if peft_model is not None:
            peft_model.save_pretrained(str(destination))
        else:
            model.save_pretrained(destination, safe_serialization=True)
            processor.save_pretrained(destination)
        return destination

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
                    paths = [str(args.data / "audio" / r["audio_filename"]) for r in batch]
                    languages = [LANGUAGE_TAGS[r["language"]] for r in batch] if args.force_language else None
                    outputs = wrapper.transcribe(audio=paths, context=[prompt] * len(batch),
                                                 language=languages)
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

    if args.final_fit:
        # The recipe and step count are already chosen by the validated arms, so
        # the last run trains on every labelled row -- including the 294
        # development clips and the 20 Jember sessions each earlier arm had to
        # withhold. There is nothing left to select on, so validation is skipped
        # rather than faked; checkpoints are kept on a step schedule instead.
        if validation:
            raise ValueError("--final-fit needs a dataset prepared with no validation split; "
                             f"this one carries {len(validation)} validation rows")
        baseline = {"wer": float("inf"), "final_fit": True}
        print("Final fit: no validation, no checkpoint selection, keeping every save", flush=True)
    else:
        baseline = evaluate("baseline")
    raw = load_dataset("json", data_files={"train": str(rebased)})["train"]
    dataset = raw.map(make_preprocess_fn_prefix_only(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    collator = DataCollatorForQwen3ASRFinetuning(processor=processor)
    if args.spec_augment:
        base_collator = collator
        augment_generator = torch.Generator().manual_seed(20260905)

        def collator(features):
            batch = base_collator(features)
            mask = batch.get("feature_attention_mask")
            lengths = mask.sum(dim=-1).tolist() if mask is not None else None
            batch["input_features"] = spec_augment(
                batch["input_features"], lengths, args.spec_freq_width,
                args.spec_time_width, args.spec_augment, augment_generator)
            return batch

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
            self.best_epoch = 0
            self.stale = 0
            self.last_mark = 0
            self.history = []

        def record(self, label, epoch, state, control):
            """Validate once, keep the weights, and decide whether to stop.

            Shared by the epoch boundary and, when --evals-per-epoch > 1, by the
            intra-epoch marks. Epoch-end directories keep their historical
            `epoch-N` names because the evaluation and packaging scripts look
            them up by that name; intra-epoch ones are named by optimizer step.
            """
            if args.final_fit:
                # Nothing to select on: keep the checkpoint and carry on.
                save_weights(args.output / f"weights-{label}")
                self.best_dir = str(args.output / f"weights-{label}")
                self.best_epoch = epoch
                self.history.append({"label": label, "epoch": epoch, "step": state.global_step,
                                     "wer": None})
                save_json(args.output / "BEST_MODEL.json",
                          {"path": self.best_dir, "wer": None, "final_fit": True,
                           "baseline_wer": None, "improves_baseline": None,
                           "epochs": self.history})
                commit()
                return control
            metrics = evaluate(label)
            self.history.append({"label": label, "epoch": epoch, "step": state.global_step,
                                 "wer": metrics["wer"]})
            if args.keep_epochs:
                save_weights(args.output / f"weights-{label}")
            if metrics["wer"] < self.best:
                self.best = metrics["wer"]
                self.stale = 0
                self.best_epoch = epoch
                self.best_dir = str(save_weights(args.output / f"best-{label}"))
            else:
                self.stale += 1
            save_json(args.output / "BEST_MODEL.json", {"path": self.best_dir, "wer": self.best,
                      "baseline_wer": baseline["wer"], "improves_baseline": self.best < baseline["wer"], "epochs": self.history})
            commit()
            if self.stale >= args.patience:
                control.should_training_stop = True
            return control

        def on_step_end(self, training_args, state, control, **kwargs):
            # Fractional-epoch marks only. The mark that lands on an epoch
            # boundary is left to on_epoch_end so a boundary is never scored
            # twice, and so `best-epoch-N` keeps meaning what it always did.
            if args.evals_per_epoch <= 1:
                return control
            mark = int(state.epoch * args.evals_per_epoch + 1e-9)
            if mark <= self.last_mark or mark % args.evals_per_epoch == 0:
                return control
            self.last_mark = mark
            return self.record(f"step-{state.global_step}", state.epoch, state, control)

        def on_epoch_end(self, training_args, state, control, **kwargs):
            epoch = int(round(state.epoch))
            self.last_mark = max(self.last_mark, epoch * args.evals_per_epoch)
            return self.record(f"epoch-{epoch}", epoch, state, control)

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
    # PEFT injects the adapters into `model`'s own submodules, so evaluation keeps
    # using `wrapper`/`model` while the Trainer optimises the LoRA parameters.
    trainer = CastFloatInputsTrainer(model=peft_model or model, args=train_args, train_dataset=dataset,
                                    data_collator=collator, processing_class=processor.tokenizer,
                                    callbacks=[callback])
    # The patched forward intentionally consumes only model inputs; Trainer must
    # perform the standard per-microbatch gradient-accumulation loss scaling.
    trainer.model_accepts_loss_kwargs = False
    print("Starting optimizer steps on audited in-domain data", flush=True)
    trainer.train()
    trainer.save_state()
    merged = None
    if peft_model is not None and callback.best_epoch:
        # Package a plain model so evaluation, souping and the submission ZIP all
        # stay adapter-agnostic. Merging on CPU keeps the base copy off the GPU.
        from peft import PeftModel
        merged = args.output / f"merged-epoch-{callback.best_epoch}"
        base = Qwen3ASRModel.from_pretrained(str(args.model), dtype=torch.bfloat16,
               device_map="cpu", local_files_only=True).model
        adapter = PeftModel.from_pretrained(base, str(callback.best_dir))
        adapter.merge_and_unload().save_pretrained(merged, safe_serialization=True)
        processor.save_pretrained(merged)
        del base, adapter
        print(f"Merged best LoRA adapter into {merged}", flush=True)
        commit()
    result = {"status": "completed", "steps": trainer.state.global_step, "baseline_wer": baseline["wer"],
              "best_wer": callback.best, "best_model": str(merged) if merged else callback.best_dir,
              "best_adapter": callback.best_dir if merged else None,
              "improves_baseline": callback.best < baseline["wer"],
              "history": callback.history}
    save_json(args.output / "TRAIN_RESULT.json", result)
    commit()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
