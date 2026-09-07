# Lost in Transcription: Indonesian–Javanese

Competition workspace for a reproducible, rules-compliant attempt at the Mozilla Data Collective (MDC) Indonesian–Javanese ASR track.

The score target is **WER <= 0.20**. That would beat the 0.2196 public leader visible on 2026-09-05, but it cannot guarantee first place: the final ranking uses the private leaderboard and competitors can improve. The safer engineering gate used here is **mean held-out WER <= 0.185** across conversation/speaker splits before spending a full submission.

**Current status (2026-09-05): the repository's recorded full-test score is 0.47555 using base Qwen3-ASR-1.7B; current authenticated rankings were not reverified.** The development and Jember archives are locally audited. A new in-domain adaptation run is training on an isolated A100 after quarantining timing defects. Its complete conversation-5 baseline is **WER 0.265920** (294 clips, 13,929 reference words, no empty predictions); this development score is not comparable directly with the full-test leaderboard. See the [run receipt](reports/training_recovery_launch.json), [baseline breakdown](reports/recovery_baseline_fold5.json), and [training plan](docs/training-recovery-plan.md). No fine-tuned improvement or new submission has been established yet.

The **[full data audit](docs/data-audit.md)** covers all 580 source recordings: 372 development clips (2.562 hours) and 208 Jember sessions (10.015 hours). Key findings include mixed training sample rates, substantial transcription-convention differences, and a development set with no Javanese-only clips. Reproduction commands, aggregate JSON reports, and private review queues are linked in the report.

The **[relationship analysis and figures](docs/data-relationships.md)** extend this with speaker/language confounding, audio–text correlations, train/development shifts, session-level quality concentration, and vocabulary coverage under each validation split. It distinguishes measured associations from transcription correctness and model performance, which still require separate evaluation.

The 5-clip smoke test cannot rank systems — Whisper's 0.1913 and Qwen's 0.2000 are 22 versus 23 errors out of 115 reference words. Do not select a model on it.

Start at **[docs/kb/00-README.md](docs/kb/00-README.md)**: the knowledge base holds the full run history, the locally measured findings (a 6.4-point casing/diacritic lever; blanket lowercasing costs 11.8 points), the proven winning recipe from a comparable DrivenData ASR competition, the verified official spec, and the prioritised plan to the deadline.

## Quick start

Requirements for data preparation are Python 3.9+ and `ffmpeg`. Submission inference must use Python 3.12 in the official runtime.

The new audited training workflow uses the isolated Modal app/volume
`mozilla-indonesian-recovery` in the connected account. See the
[recovery plan](docs/training-recovery-plan.md) for the diagnosis, validation
split, cost cap, and checkpoint-selection gates. Its entrypoints are
`scripts/prepare_recovery.py`, `scripts/train_recovery.py`, and
`scripts/modal_recovery.py` (Python 3.12 on the GPU).

The commands below and [docs/modal.md](docs/modal.md) describe the older
`govind-llm` / `govind-llm-data` workflow. Its latest-checkpoint packaging is
not the recovery workflow's WER-best selection and requires access to that
separate account.

```bash
# Public catalog access (no key needed)
python3 scripts/mdc_fetch.py catalog --query Javanese --task ASR

# Create a key at https://mozilladatacollective.com/profile/credentials,
# accept any dataset terms in the web UI, then export it only in your shell.
export MDC_API_KEY='...'

# Full 72.8 MB development archive (recommended; only 372 clips)
python3 scripts/mdc_fetch.py download dev --extract

# Deterministic 24-session training subset. The MDC API serves whole archives,
# so the archive is cached and only 24 audio files are extracted.
python3 scripts/mdc_fetch.py download jember --extract --max-audio-files 24 --seed 20260905

# Reproduce the external OpenSLR 35 starter subset already prepared locally.
# It is CC BY-SA 4.0 and must be published to MDC at competition conclusion.
make openslr-subset

# Prepare dev data in the official runtime layout, then score predictions.
python3 scripts/prepare_dev.py data/subsets/dev data/dev_runtime
PYTHONPATH=src python3 -m mdc_asr.wer data/dev_runtime/ground_truth.csv predictions.csv --group-by language

# Generate leak-resistant, conversation-held-out Qwen training JSONLs.
python3 scripts/make_qwen_jsonl.py \
  data/dev_runtime/ground_truth.csv data/dev_runtime/clips data/qwen_folds \
  --fold-column convo_id

# Run tests.
make test

# Modal A100 path: stage public assets, launch training, package latest checkpoint.
make modal-stage-qwen
make modal-stage-openslr
make modal-train
make modal-eval
make modal-package
make modal-download

# Backup source, reusable artifacts, and generated weights to private HF.
# Set HF_TOKEN only in your shell; never commit it.
make hf-backup-local
make modal-backup-hf

# Optional: exercise Whisper locally on 10 held-out OpenSLR clips (not an
# in-domain score and therefore not evidence that the 0.20 target is met).
make whisper-smoke
make qwen-smoke
```

Set `PYTHONPATH=src` if invoking modules without `make`; the Makefile does this automatically.

An ignored clone of the official runtime is already present at `runtime/`. Refresh it with `make runtime`; record the resulting commit before a scored submission.

Download model weights only after the data pipeline is working:

```bash
python3 scripts/fetch_hf_model.py qwen3-asr-1.7b
python3 scripts/build_submission.py --model-dir models/qwen3-asr-1.7b

# Build the comparison model with the same auto-detecting entrypoint.
python3 scripts/fetch_hf_model.py whisper-large-v3
python3 scripts/build_submission.py \
  --model-dir models/whisper-large-v3 \
  --output artifacts/whisper-submission.zip
```

The model downloader pins the researched Hugging Face revision and verifies every published LFS SHA-256. Model weights, data, and built ZIP files are deliberately gitignored.
The official Qwen supervised fine-tuning helper is pinned with provenance under `vendor/`; see [vendor/README.md](vendor/README.md).

MDC API keys authenticate requests but do not accept dataset-specific legal terms. A 403 from `mdc_fetch.py` means the account must open each dataset page, complete the Download/terms flow (and any approval request), then retry the same command. A bad or missing key returns 401. Never commit the key.

## What to build to reach 0.20

The recommended sequence is evidence-driven:

1. Measure zero-shot Qwen3-ASR-1.7B and Whisper large-v3 on the same conversation-held-out folds. Note that the base-model question is largely settled by external evidence: all three Word Track winners of DrivenData's comparable "On Top of Pasketti" ASR challenge used a fine-tuned Qwen3-ASR-1.7B, and their code is public. See [docs/kb/research/winning-solutions.md](docs/kb/research/winning-solutions.md).
2. Fine-tune on Jember spontaneous speech — the corpus that both the dev and test sets are drawn from — then Common Voice Javanese/Indonesian and the competition development training folds. Keep each conversation and speaker wholly within one fold. Use the winners' LoRA configuration as the starting point and average checkpoints.
3. Tune decoding separately for `ind`, `jav`, and `javind`, using the manifest language label only. Use normalized WER—not loss—as the selection criterion.
4. Add a Javanese–Indonesian text LM or hypothesis rescoring only if it improves both conversation folds. The official runtime now includes `kenlm` and `pyctcdecode`.
5. Use a model router or ensemble only when its full-fold WER beats the best single model by at least 0.005. Retrain on all permitted labeled audio for the final candidate, run the official offline Docker smoke test, then submit.

See [docs/strategy.md](docs/strategy.md) for the experiment matrix and [docs/resources.md](docs/resources.md) for the researched datasets, models, licenses, and pinned sources.

## Competition rules and constraints

This is a working summary verified against the official pages on **2026-09-05**, not legal advice. The [official rules](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/rules/) and competition website control if anything here differs.

### Dates, prizes, ranking, and entry

- End: **2026-09-25 at 00:00 UTC**.
- Track prizes: **$4,000 first** and **$2,000 second**. A separate cross-track bonus requires the same team membership in all three tracks.
- Lower corpus-level MDC-normalized WER is better. Running scores use public test data, while winners are determined by the private leaderboard. Identical final scores are broken in favor of the earlier submission.
- One platform account per person. A participant may enter once, either alone or on one team, and cannot belong to multiple teams. Attempts to evade submission limits can disqualify the participant/team.
- Entrants must satisfy the official age, residency, sanctions, employment, and conflict-of-interest eligibility terms. Legal entities are not eligible.

### Permitted data and models

- MDC datasets may be used for training. External data is allowed only when the participant has all necessary rights and permissions; externally sourced training data must be published to MDC when the competition ends. Keep license/provenance evidence for every source.
- Open-weight pretrained models are allowed when weights are freely/publicly available, the license permits competition use, and the model runs in an environment controlled by the participant.
- **Never upload competition data, confidential information, or nonpublic derivatives to hosted models, model APIs, or inference services.** Hosted coding tools are allowed only without that material. This repository's tooling processes audio locally.
- Each test clip must be processed independently. Do not train, pseudo-label, adapt, or derive cross-sample features from the hidden test set. Inference must work automatically on new clips without retraining, and test data cannot be manually annotated.
- Training data may be annotated if those annotations ship with the solution for reproducibility and do not overfit the test set.

### Data handling and code sharing

- Competition data must be used only as permitted and protected from people who have not accepted the rules. Do not identify speakers. Report accidental identification or unauthorized access to the organizers.
- Do not publish or redistribute the competition data. The `.gitignore` keeps downloads, extracted subsets, models, and artifacts out of Git.
- Private sharing of competition-related code or data outside the registered team is prohibited. Public code sharing is allowed, subject to third-party rights, and is deemed licensed under MPL 2.0.
- A winning solution, its code, weights/configuration, dependencies, and reproducibility documentation must be delivered to DrivenData. Winning code must be available under **MPL 2.0**; supporting software must allow free commercial use unless the competition says otherwise. Confirm any ambiguous dataset/model license with the organizers before relying on it.

### Metric details

The primary metric is corpus WER:

`WER = (substitutions + deletions + insertions) / reference words`

Before scoring, MDC removes bracketed annotations and unintelligible markers, unwraps ordinary parentheses, strips specified punctuation, preserves attached ellipses, and collapses repeated spaces. It lowercases sentence-initial letters but **does not lowercase all text**, so mid-sentence case errors count. Use `mdc_asr.wer`; it reproduces the published normalizer and aligns rows by `audio_filename`.

### Code-execution submission

- Submit a ZIP with `main.py` at the archive root and every required model asset included. The runtime has no network access.
- Read clips from `/code_execution/data/clips` and `test_metadata.csv`; write exactly one quoted CSV row per clip to `/code_execution/submission/submission.csv`, with columns `audio_filename,transcript`.
- Runtime: Python 3.12, one NVIDIA A100 80 GB, 24 vCPUs, 220 GB RAM, no root, and a **2-hour** full-run limit. Smoke tests have a **1-minute** limit.
- Do not log information about test data. Normal jobs cap logs at 500 lines and 300 characters per line.
- Test locally in the [official runtime](https://github.com/drivendataorg/lost-in-transcription-runtime), then use a smoke test. Failed, cancelled, and smoke jobs do not count against the submission limit.

## Repository map

```text
configs/             pinned dataset and model manifests
data/reference/      public full/smoke submission schemas
docs/                strategy and source/license notes
scripts/             API download, preparation, model, and packaging tools
src/mdc_asr/         exact normalization and dependency-free WER scorer
submission_src/      offline Qwen3-ASR submission entrypoint
tests/               unit tests that do not require audio or model weights
```

## Important links

- [Competition home](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/1/)
- [Problem and metric](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/)
- [Code submission specification](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/16/)
- [MDC API reference](https://mozilladatacollective.com/api-reference)
- [Competition forum](https://mdc-competitions.discourse.group/c/lost-in-transcription/5)
