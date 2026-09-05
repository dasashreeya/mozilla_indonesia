# Modal Training Runner

This repo uses Modal as the primary GPU environment for the competition. The
Modal app name is `govind-llm`; durable data, checkpoints, and submission zips
live in the Modal volume `govind-llm-data`.

Modal is being used only as controlled compute for open-weight models. Do not
turn these functions into web endpoints, and do not send competition audio to a
hosted model API or inference service.

## Authentication

This machine already has an authenticated Modal profile. To check it:

```bash
uvx --from modal modal token info
```

If you need to replace it later, Modal requires both a token ID and token
secret:

```bash
uvx --from modal modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

Do not put either value in this repository.

## Stage Public Assets

```bash
uvx --from modal modal run modal_app.py::stage_qwen
uvx --from modal modal run modal_app.py::stage_openslr
uvx --from modal modal run modal_app.py::volume_status
```

The OpenSLR staging step downloads the pinned shard, verifies SHA-256 hashes,
and prepares the deterministic speaker-disjoint `shard0-3000-seed-20260905`
subset on the Modal volume.

## Train

Start the default A100 fine-tune:

```bash
uvx --from modal modal run --detach modal_app.py::train_qwen
```

For an attached run with logs:

```bash
uvx --from modal modal run --timestamps modal_app.py::train_qwen
```

The default command trains Qwen3-ASR-1.7B for 2 epochs on the staged OpenSLR
subset with effective batch size 32. It resumes from the latest checkpoint if
the output directory already contains one.

Useful overrides:

```bash
uvx --from modal modal run modal_app.py::train_qwen --epochs 3 --lr 2e-5 --batch-size 4 --grad-acc 8
```

## Evaluate And Package

Evaluate the latest checkpoint on the staged validation split:

```bash
uvx --from modal modal run modal_app.py::evaluate_qwen
```

Build the offline submission zip from the latest checkpoint:

```bash
uvx --from modal modal run modal_app.py::package_submission
```

Download the ready-to-submit zip:

```bash
uvx --from modal modal volume get govind-llm-data \
  artifacts/modal-qwen-openslr35-stage1-submission.zip \
  artifacts/modal-qwen-openslr35-stage1-submission.zip
```

Download its receipt:

```bash
uvx --from modal modal volume get govind-llm-data \
  artifacts/modal-qwen-openslr35-stage1-submission.zip.json \
  artifacts/modal-qwen-openslr35-stage1-submission.zip.json
```

## Back Up To Hugging Face

Use the private model repo `YOUR_HF_USER/govind-llm-mdc-indonesian-javanese-asr`
for checkpoints, submission zips, receipts, source snapshots, and plots. Do not
upload raw MDC competition audio or extracted restricted datasets.

Local source/artifact backup:

```bash
export HF_TOKEN='...'
uv run --python 3.12 --with huggingface_hub python scripts/hf_backup.py
```

Modal volume backup after training/package:

```bash
uvx --from modal modal secret create hf-backup-token HF_TOKEN="$HF_TOKEN" --force
uvx --from modal modal run modal_app.py::backup_to_hf
```

To also mirror the public base Qwen weights from the Modal volume, add:

```bash
uvx --from modal modal run modal_app.py::backup_to_hf --include-base-model
```

## In-Domain MDC Data

The MDC API key authenticates requests, but dataset-specific terms still must be
accepted in the web UI. If `scripts/mdc_fetch.py` returns HTTP 403 for
development or Jember data, open the MDC dataset pages in the same account,
complete the Download/terms flow, then rerun the local fetch.

After the restricted data is available locally, prepare it locally and upload
only your permitted training assets to the private Modal volume:

```bash
python3 scripts/prepare_dev.py data/subsets/dev data/dev_runtime
python3 scripts/make_qwen_jsonl.py data/dev_runtime/ground_truth.csv data/dev_runtime/clips data/qwen_folds --fold-column convo_id

uvx --from modal modal volume put govind-llm-data data/dev_runtime /data/dev_runtime
uvx --from modal modal volume put govind-llm-data data/qwen_folds /data/qwen_folds
```

Then launch a train command with explicit JSONL/subset paths, or add a small
Modal function for the chosen fold once the exact prepared layout is present.
