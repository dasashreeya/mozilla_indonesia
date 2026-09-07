# Wave 2 runbook — the depth path, end to end

_2026-09-07. The operational companion to
[11-audit-and-methodology-v2](11-audit-and-methodology-v2.md). That document
argues what to do and why; this one is the command sequence, the gate at each
step, and where the outputs land._

Everything below runs in the Modal app `govind-llm` against the volume
`govind-llm-data`.

## 0. Names, because the account is shared

Anyone with dashboard access to that Modal account can see app names, volume
names, function names and call arguments. None of them say what this is:

| What shows | Value |
| --- | --- |
| app | `govind-llm` |
| volume | `govind-llm-data` |
| prepared data | `ds-base`, `ds-prompt`, `ds-full`, `ext-a`, `ext-b` |
| runs | `arm-lora`, `arm-lora-long`, `arm-lr3`, `arm-prompt`, `arm-extb`, `arm-extall`, `arm-final` |
| context prompt | passed as the preset name `codeswitch`, never as its text |

`tests/test_wave2.py::ModalNamingTests` fails the build if a new name carries
"mozilla", "javanese", "jember", "indonesia", "jav", "mdc", "competition" or
"asr". Keep the descriptive name in a comment — Modal does not render comments.

Override the two top-level names with `MODAL_APP_NAME` and `MODAL_VOLUME_NAME`
if the account already uses them for something else.

**What this does not hide.** Run *logs* still print per-language WER tables with
the labels `jav` / `javind` / `ind`, because those come from the data. Somebody
who opens a log and reads it carefully can work out the domain. Hiding that too
would mean obfuscating the metrics you need to read, which is a bad trade. The
default Hugging Face backup repo is also named
`govind-llm-mdc-indonesian-javanese-asr` — that is a private repo in your own
account, not the shared one, but pass `--repo-id` to `report` if you want it
neutral there as well.

## 1. One-time setup

```bash
# The HF token, as a Modal secret. Never as a function argument -- arguments
# are visible in the shared dashboard, secrets are not.
modal secret create hf-token HF_TOKEN=hf_...

# Push the four MDC archives to the volume. Resolves each corpus by directory,
# so a renamed download does not break anything, and maps the one alias that
# differs (cv_jv_spontaneous on disk -> cv_jv in the corpus builder).
python3 scripts/upload_external.py            # --dry-run first if unsure

# The ~350 MB of preparation sources, and the pinned base model.
modal volume put govind-llm-data <prep-sources.tar> uploads/prep-sources.tar
modal run scripts/modal_recovery.py::stage_base
```

`prep-sources.tar` is the audited local corpus plus `reports/private/`; take its
SHA-256 and pass it as `PREP_SHA` below. The hash is checked on the volume before
anything is built from it.

## 2. Build the data

```bash
make w2-prep PREP_SHA=<sha256>
```

Builds five things in parallel:

| Name | What it is |
| --- | --- |
| `ds-base` | the audited in-domain corpus, 20 sessions and development conversation 5 held out, per-clip language tags |
| `ds-prompt` | `ds-base` plus the `codeswitch` context prompt on every example |
| `ds-full` | **no holdout at all** — all 208 sessions, all 372 development clips. For the final fit only |
| `ext-a` | Common Voice Javanese plus the two Indonesian podcast corpora |
| `ext-b` | **the new data** — East Javanese TTS from Lumajang and Bojonegoro |

`ext-a` and `ext-b` are built with `--style-tags`, so each corpus is tagged by
its transcription convention (`Javtts`, `Javcv`, `Indpod`) rather than by its
language. Decoding always forces the in-domain tag, so external orthography
trains the encoder and never reaches the decoder. This is what makes read-speech
TTS safe to add; without it the decoder learns a blend of two spellings of
Javanese, which is the mechanism behind the earlier negative OpenSLR transfer.

**Gate.** Check `ext-b` reports roughly 10–15 h. If it reports under 2 h the
nested-zip reader did not find the recordings, and training on it would waste a
GPU-hour proving nothing.

## 3. Fix the instrument first

```bash
make w2-stage0
```

One decode pass. Prints the fold WER, the two language halves, the
**test-weighted** number and its public forecast, and then per-session WER with
anything past the 90th percentile flagged.

**This is the highest-value single command in the runbook.** Session 35 sits at
0.9352 WER against 0.16–0.44 for its neighbours and is a timestamp
misalignment, not hard audio. The 188 *training* sessions have never been
checked. If several carry the same defect, every arm in the previous wave
trained on partly-misaligned Javanese supervision — which would explain why
eight recipe variants all landed within 0.018 of each other.

**Gate.** Listen to the top two flagged sessions against their transcripts
before training. If the reference lags the audio by about one window, drop those
sessions and rebuild `ds-base`.

## 4. Train the arms

```bash
make w2-train              # detached; six arms in parallel
```

Every arm runs `--evals-per-epoch 4 --patience 8 --keep-epochs`. That matters as
much as the learning rate: at ~550 optimizer steps per epoch, epoch-only
validation gives three measurements per run and leaves one averageable
checkpoint. Patience 8 keeps the schedule from stopping before its
low-learning-rate phase.

| Arm | The question it answers |
| --- | --- |
| `arm-lora` | does the reference 1st-place recipe transfer? LoRA r=16 α=32, lr **1e-4**, 3 epochs, SpecAugment 0.5 |
| `arm-lora-long` | where does the optimum actually sit? Same, 6 epochs |
| `arm-lr3` | was the old full fine-tune simply under-stepped? lr 3e-5 against the old 1e-5 |
| `arm-prompt` | does the context prompt survive fine-tuning? |
| `arm-extb` | **the new East Javanese data, alone** |
| `arm-extall` | both external corpora |

Rank on **Javanese at the 49% weighting**, not the fold mean. Javanese is ~49%
of the test set and 59% of the remaining error.

**Gate.** If `arm-lora` does not beat 0.3399 Javanese, the "the winning recipe
does not transfer" conclusion becomes real — measured, for the first time — and
the fallback is [11 §3.3](11-audit-and-methodology-v2.md) (CTC + KenLM).

## 5. Average the checkpoints

```bash
make w2-soup
```

Picks the four best-scoring kept checkpoints across all Wave 2 runs, averages
them, and scores the average on the same fold. This is free at inference — a
soup is one model at one model's cost, which is why it, and not a decode-time
ensemble, is the combination technique this two-hour runtime can afford.

**Gate.** Promote only on **≥0.015** improvement in the test-weighted number.
Run-to-run noise is ±0.002 and the 509-clip fold cannot resolve less than ~0.015.

## 6. Look at it

```bash
make w2-plots
```

Writes four figures and `summary.csv` to `/vol/plots`:

* `arms.png` — every arm, split by language, sorted on the test weighting;
* `curves.png` — validation WER against optimizer step, optimum ringed;
* `sessions.png` — per-session WER with outliers flagged;
* `target.png` — every system's forecast against the leader and the target.

## 7. Final fit, then submit

```bash
make w2-final STEPS=<the step count the arms selected>
```

`STEPS` is required and has no default: `ds-full` has no validation split, so
there is nothing to early-stop on. Read the winning arm's `BEST_MODEL.json`
history, take the step where WER bottomed out, and scale it by the row-count
ratio between `ds-base` and `ds-full`.

This run trains on **100% of the labelled in-domain data** — including the 294
development clips (79% of the only in-domain code-switched audio) that every
model submitted so far has never seen.

Then package, smoke-test, and submit:

```bash
modal volume get govind-llm-data runs/arm-final/weights-epoch-1 models/final
python3 scripts/build_submission.py --model-dir models/final
```

Set `prompt` and `language_routing` in `submission_src/config.json` to match
whatever the winning arm trained with. If the arm used `ds-prompt`, the decode
prompt must be the same string, byte for byte, or the forced prefix will not
match what was trained.

**Always run the free smoke submission before spending a full one.** Smoke tests,
cancelled and failed jobs do not count against the 3-per-7-days limit.

## 8. Back everything up

```bash
make w2-report                                     # figures + results to HF
modal run scripts/modal_recovery.py::report \
    --weights runs/arm-final/weights-epoch-1,soups/soup-v1
```

`report` always pushes figures, evaluation JSONs, predictions and every run's
receipts — small, so run it after every wave. Weights go only when named,
because a run directory holds several multi-gigabyte checkpoints and pushing
them all by default turns a routine backup into an hour of upload.

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `Upload missing: /vol/uploads/external/<corpus>` | `upload_external.py` not run, or the archive landed under its download name — `cv_jv_spontaneous` must upload as `cv_jv` |
| `<corpus> has N archives; expected one` | two downloads in one directory; delete the stale one |
| `HF_TOKEN missing` | `modal secret create hf-token HF_TOKEN=...` |
| `--final-fit needs a dataset prepared with no validation split` | pointed at `ds-base`; the final fit needs `ds-full` |
| `Preparation flag not allowed` | add the flag to `PREP_FLAGS` in `modal_recovery.py` |
| Run dies and restarts into a permanent failure | fixed — a preempted attempt is moved aside rather than blocking the restart |
| `Uploaded training input hash mismatch` | the volume's prepared data was rebuilt; re-run `prep2` |
