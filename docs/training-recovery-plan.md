# Indonesian recovery: diagnosis and first training run

Upstream fetched and fast-forward checked at `d773727cc628a1c2aeaf366e21c2d328df76203a`; local audit work preserved. This plan implements a measured first adaptation run, not a prediction of a particular leaderboard rank.

## Phase 0 — evidence and allowed interfaces

- `experiments/results.csv`, L02: the recorded full-public WER 0.47555 is base Qwen3-ASR-1.7B. OpenSLR fine-tuning improved read-speech validation, but no corresponding full competition evaluation is recorded. The current authenticated leaderboard could not be inspected; "last" is the user's current report.
- `vendor/qwen3_asr_sft.py:128–150` assumes right padding. The tokenizer property reports right padding, but **the processor overrides it to left padding on actual calls**: a remote processor probe returned attention masks `[[0,0,0,1],[1,1,1,1]]`. Inspecting only `tokenizer.padding_side` was insufficient. The old mask can supervise prompt/audio tokens in shorter batch members. The replacement is regression-tested for both padding sides and real EOS tokens. This defect affects fine-tuning, not the recorded submission that used base weights.
- `scripts/evaluate_qwen.py:102` uses 128 new tokens; submission uses 256. Use 512 in both training validation and Qwen submission. Detect truncation/empty outputs in evaluation.
- `modal_app.py:365,415` chooses the highest checkpoint step. Save and select actual generation-WER-best weights instead.
- `docs/data-audit.md` and `docs/data-relationships.md`: 46 timestamp defects, two overlapping intervals, training median 5 s versus development 26.2 s, marked orthography mismatch, two development conversations/four speakers. Preserve whole conversations in validation.
- Spanish evidence: V4 public WER 0.4636 versus V3 0.5042, both Whisper turbo. Copy the principles of target-domain training, correct loss labels, consistent scoring, explicit long-form handling, and data balancing. Its subdivision of conversations and failure to restore the WER-best weights must not transfer.
- The connected Modal profile is `dasashreeya`. The old `govind-llm-data` volume is absent here. Use a new `mozilla-indonesian-recovery` volume and app; do not alter the Spanish volumes or require the other account's credentials.

Verified API references:

- [Qwen fine-tuning source and target format](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff/finetuning): `language None<asr_text>...` is valid for unsupported or mixed-language speech. Do not force unsupported `Javanese`/`jw` through the inference wrapper.
- [Qwen processor](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/processing_qwen3_asr.py): `_merge_kwargs` passes the processor's left-padding default to the tokenizer. Actual attention masks, rather than the tokenizer property alone, verify the effective padding side.
- Qwen wrapper: `Qwen3ASRModel.from_pretrained(path, dtype=..., device_map=..., max_inference_batch_size=..., max_new_tokens=512)` and `transcribe(audio=paths, context="", language=None)`. No invented beam-search wrapper argument.
- Existing vendor `patch_outer_forward`, `make_preprocess_fn_prefix_only`, `CastFloatInputsTrainer`, and HF `TrainingArguments` are the starting patterns. Disable KV caching on labelled forward calls; generation delegates directly to the thinker. Use non-reentrant gradient checkpointing.
- [Modal Volume](https://modal.com/docs/sdk/py/latest/Volume): `Volume.from_name(..., create_if_missing=True)`, explicit `commit()`, directory uploads. Use immutable run paths and never overwrite previous artifacts.

## Phase 1 — validated target-domain training inputs

Implement a new preparation script using the audit's validated IDs and timestamp exclusions. Copy original source audio to staged data only through generated 16 kHz PCM segments. Exclude the 46 timing defects and overlapping rows from initial training; record every exclusion. Preserve original data.

Include both short utterances and disjoint 18–28 s windows of exactly adjacent, valid rows from one recording. Break windows at missing/bad/overlapping rows or timestamp gaps. Do not infer text for clipped audio.

For the first selection run, hold out development conversation 5 (294 clips, speakers 1/6) and use conversation 2 (78 clips, speakers 5/7) only for training. Repeat development training rows to increase their exposure without using validation rows. Keep all long validation clips intact.

First Jember label hypothesis: normalize scoring punctuation/casing and fold Jember-only accent marks toward development's largely unaccented convention; preserve original targets in the local provenance manifest and preserve development accents. This is a training intervention whose benefit must be established by WER, not assumed from vocabulary overlap.

Verification: manifest row/file correspondence, no held-out speaker/conversation IDs in training, timestamp provenance, native-rate-aware decoding, and synthetic window-boundary tests. No arbitrary quality-flag filtering.

## Phase 2 — fix supervision and establish the baseline

Mask padding and the first N active prompt tokens using attention-mask cumulative positions. Do not mask real EOS tokens merely because EOS and padding IDs coincide. Preserve the vendor's original source receipt and document the local patch.

On one A100-80GB, load fresh pinned Qwen base weights and evaluate the complete held-out conversation at 512 tokens before training. Save predictions privately and aggregate normalized WER/S/D/I by language, conversation, speaker, and duration. Verify loss is finite and inspect a synthetic left-padding regression plus runtime target-token checks.

Verification: baseline receipt, prediction count, full reference word denominator, no reference-audio truncation, first optimizer step with finite loss. Do not reuse the OpenSLR checkpoint as the sole initialization.

## Phase 3 — initial supervised adaptation

Full fine-tuning of Qwen3-ASR-1.7B, BF16, gradient checkpointing, initial LR 1e-5, effective batch 16, three epochs, cosine decay and warmup. This corrects the training/data problem before spending on a model sweep. Use a separate immutable run directory.

At each epoch, generate held-out predictions with the same 512-token decoder and score the actual transcripts. Save inference-ready weights for the best WER so far, plus resumable training checkpoints. The base model is the score-to-beat; do not mark a worse checkpoint as an improvement. Stop early after two non-improving epochs. Persist progress and receipts to the private Modal volume.

The initial remote function is capped at six hours on one A100, with no automatic paid retries. Approximate published GPU/CPU/reserved-memory cost for the cap is around $20, excluding image builds/storage and any CPU staging. It is a ceiling on this first invocation, not permission for an unlimited architecture sweep.

## Phase 4 — verification and next experiments

Run unit tests, source checks, staging integrity checks, and confirm optimizer steps rather than reporting an enqueued job as training. Record the Modal app URL, function/run ID, data counts and baseline in a local run receipt.

Before any leaderboard submission, evaluate the selected recipe on the opposite conversation holdout. Compare orthography variants and then a Whisper candidate using identical folds and complete long-form decoding. Consider LoRA, checkpoint averaging, or ensembles only after measured comparisons; the installed `peft` package alone does not enable LoRA in the existing trainer.

Final refit on all permitted labelled data and packaging are downstream steps after validation establishes an improvement. No leaderboard submission, automatic publication, or cross-account data transfer is part of this first training launch.

## Reproduction and monitoring

Launch verification: [Modal run](https://modal.com/apps/dasashreeya/main/ap-WE2Z3uLcFewAJlRmJHdgSW)
completed the 294-clip baseline at WER **0.265920** (S=2769, D=533, I=402,
N=13929; no empty predictions), passed a finite-loss supervision preflight,
and reached optimizer step 5 with loss 3.1872. All 62 local tests passed.
This is a running training job, not a completed improvement claim.
The [launch receipt](../reports/training_recovery_launch.json) records source
hashes and the real-batch masking probe: the old implementation supervised
54 extra prompt/audio tokens for the shorter of two examples.

The baseline varies substantially by held-out speaker: WER 0.31115 for
speaker 1 versus 0.21496 for speaker 6. This split contains mixed-language
`javind` clips only; the opposite fold is necessary to assess Indonesian-only
clips. Clips over 30 seconds have baseline WER 0.22882 versus 0.27039 for
shorter clips on this fold, so duration mismatch alone does not establish
that long clips are the dominant remaining error source.

The prepared dataset is private and ignored by Git. Preparation refuses to
overwrite a directory; use a new output directory for a new recipe. The first
archive contains 8,818 training rows (8,272 unique examples), including 6,631
short Jember segments, 1,563 Jember windows, and 78 development training clips
repeated eight times. Validation contains 294 untouched development clips.

```bash
uv run --no-project --with numpy==2.5.2 --with soundfile==0.14.0 python scripts/prepare_recovery.py
tar -cf artifacts/recovery-v1.tar -C data recovery-v1
shasum -a 256 artifacts/recovery-v1.tar
uv run --no-project --with modal modal volume put mozilla-indonesian-recovery artifacts/recovery-v1.tar /uploads/recovery-v1.tar
uv run --no-project --with modal modal run --detach scripts/modal_recovery.py::run --archive-sha256 e4a91e335289ac958ff4f312437bcf0c1990a621a6a742485ea65b60c1064ad1

# Read-only progress and selection receipts (created as the run advances).
uv run --no-project --with modal modal volume get mozilla-indonesian-recovery /runs/qwen-recovery-v1-fold5/PROGRESS.json -
uv run --no-project --with modal modal volume get mozilla-indonesian-recovery /runs/qwen-recovery-v1-fold5/BEST_MODEL.json -
```

The archive hash above identifies this specific prepared archive, not a promise
that a regenerated tar has identical metadata. Always use the actual hash of
the archive being uploaded. Never overwrite or reuse a nonempty training run.
