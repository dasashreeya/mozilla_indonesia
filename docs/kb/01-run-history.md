# Run history and scoreboard

_Last updated 2026-09-05. Append to this file after every submission, training run, or
evaluation. Future iterations should be able to reconstruct the whole attempt from here._

This is the factual record of what has been built, submitted, and scored. It exists so
that no future session repeats an experiment that has already been run, or re-derives a
conclusion that has already been paid for with a submission.

## Scoreboard

| # | Artifact | Method | Size | Smoke WER (5 clips) | Full public WER (2,118 clips) | External val WER |
|---|---|---|---|---|---|---|
| 1 | `artifacts/submission.zip` | Qwen3-ASR-1.7B, **base weights, zero-shot** | 4.4 GB | 0.2000 | **0.47555** | — |
| 2 | `artifacts/whisper-submission.zip` | Whisper large-v3, **base weights, zero-shot** | 2.9 GB | 0.1913 | not run | — |
| 3 | `artifacts/modal-qwen-openslr35-full-v1-best-submission.zip` | Qwen3-ASR-1.7B fine-tuned on OpenSLR-35 shard 0 | 3.8 GB | 0.3130 | not run | 0.293670 |

Public leaderboard on 2026-09-05: #1 0.2196, #2 0.2247, #3 0.2290, #4 0.2506.
Our only full-test measurement, 0.47555, is **more than double the leader**.

Submissions consumed: 2 of the track allowance.

> The ZIP files above are retained on disk and are gitignored. Do not delete them: #1 is
> the only artifact with a known full-test score, and #2 is the current fallback
> candidate. See [09-risks-and-submission-strategy](09-risks-and-submission-strategy.md).

## The smoke test is 5 clips and 115 reference words

`data/reference/smoke_test_submission_format.csv` contains exactly 5 rows. Every smoke
score we have is an integer error count over a 115-word reference:

| System | Smoke WER | Reconstructed errors | Arithmetic |
|---|---|---|---|
| Base Whisper large-v3 | 0.1913 | 22 | 22 / 115 = 0.191304 |
| Base Qwen3-ASR-1.7B | 0.2000 | 23 | 23 / 115 = 0.200000 |
| Qwen fine-tuned on OpenSLR-35 | 0.3130 | 36 | 36 / 115 = 0.313043 |

Three consequences, and they change what we should do next:

1. **Whisper did not beat Qwen.** It produced one fewer wrong word out of 115. The 95%
   confidence interval on a WER of 0.20 measured over 115 words is roughly ±0.073
   (`sqrt(p(1-p)/N)` = 0.037, doubled), and that *understates* the uncertainty because
   word errors cluster within an utterance — the effective sample size is nearer 5 than
   115. Any claim that Whisper is the stronger base model is currently unsupported.
   Do not spend a submission on it for that reason alone.
2. **Smoke score does not predict full score.** Base Qwen went 0.2000 smoke →
   0.47555 full, a factor of 2.4. Smoke is a plumbing check: it proves the ZIP runs,
   reads the manifest, and writes a valid CSV. It is not a model-selection instrument
   and must never be used as one again.
3. **The OpenSLR fine-tune is the one smoke signal with any weight.** 36 errors vs 23 is
   a 13-word gap, far less likely to be noise than a 1-word gap. It is consistent with
   the read-speech fine-tune actively *hurting* on spontaneous conversational audio.
   Treated as a hypothesis, not a conclusion — see the open questions below.

## What was actually built

1. Offline code-execution ZIPs with `main.py` at the archive root, an auto-detecting
   backend in [submission_src/main.py](../../submission_src/main.py) that dispatches to
   either `qwen_asr.Qwen3ASRModel` or a Whisper `transformers` pipeline based on the
   bundled model's `config.json`.
2. Pinned, hash-verified base weights for Qwen3-ASR-1.7B (revision
   `7278e1e70fe206f11671096ffdd38061171dd6e5`) and Whisper large-v3 (revision
   `06f233fe06e710322aca913c1bc4249a0d71fce1`).
3. A dependency-free reimplementation of the official scorer in
   [src/mdc_asr/normalization.py](../../src/mdc_asr/normalization.py) and
   [src/mdc_asr/wer.py](../../src/mdc_asr/wer.py). **Verified byte-for-byte behaviourally
   identical to `runtime/score.py`** — see [02-measured-findings](02-measured-findings.md).
4. A deterministic, speaker-disjoint OpenSLR-35 Javanese subset (seed `20260905`):
   9,900 training and 1,100 validation utterances from shard 0, zero speaker overlap.
5. A Modal A100 training path ([modal_app.py](../../modal_app.py),
   [docs/modal.md](../modal.md)) covering staging, fine-tuning, evaluation, packaging,
   and Hugging Face backup.

## Training runs

### `openslr35-full-v1` — completed

* Base: Qwen3-ASR-1.7B. Data: 9,900 OpenSLR-35 Javanese **read-speech** utterances.
* Validation: 1,100 speaker-disjoint OpenSLR-35 utterances. Hardware: one A100-80GB.
* Receipts: `artifacts/openslr35-full-v1-TRAIN_RECEIPT.json`, best checkpoint `930`.

Validation WER, recomputed from the raw prediction CSVs with the repo scorer:

| Checkpoint | WER | S | D | I | N |
|---|---|---|---|---|---|
| base (no fine-tune) | 0.705934 | 1337 | 131 | 126 | 2258 |
| stage1 (3,000-clip subset) | 0.521701 | 1007 | 86 | 85 | 2258 |
| 310 | 0.333376 | 2269 | 135 | 182 | 7757 |
| 620 | 0.298311 | 2042 | 130 | 142 | 7757 |
| **930 (best)** | **0.293670** | 2015 | 117 | 146 | 7757 |

Fine-tuning cut read-speech Javanese WER from 0.706 to 0.294. The gain is real but the
curve is flattening (310→620 = −0.035, 620→930 = −0.005), so more epochs on this data
are near-exhausted. Errors are 88% substitutions — a lexical/orthographic problem, not a
decoding or hallucination problem.

### `openslr35-continuation` — staged, never completed

156,643 additional Javanese utterances from shards 1–f were staged. Only
`artifacts/openslr35-full-v1-TRAIN_INPUT_RECEIPT.json` exists; there is **no training
receipt and no checkpoint**. This run did not finish and produced nothing usable.

## Data actually used

| Source | Status | Used for |
|---|---|---|
| OpenSLR-35 shard 0 (Javanese read speech, CC-BY-SA-4.0) | downloaded, hash-verified | all fine-tuning and all validation to date |
| OpenSLR-35 shards 1–f | staged only | nothing — run never completed |
| MDC development data (372 clips, in-domain) | **HTTP 403** | nothing |
| MDC Jember spontaneous Javanese (~10 h) | **HTTP 403** | nothing |
| MDC Common Voice jv/id | not fetched | nothing |

**Zero in-domain data has ever been used, and zero in-domain WER has ever been
measured.** Every model decision so far rests on out-of-domain read speech or on a
5-clip smoke test. This is the central weakness of the attempt.

## Why the full score was 0.47555

In descending order of confidence:

1. **The wrong model was submitted.** `submission.zip` contains base Qwen3-ASR-1.7B with
   no fine-tuned weights. Qwen's official language table does not list Javanese at all.
   Its measured zero-shot Javanese WER on our own held-out read speech is 0.706.
2. **Domain mismatch.** The only training signal we have ever applied is OpenSLR-35 read
   speech — scripted, single-speaker, clean. The test set is spontaneous, conversational,
   code-switched Javanese/Indonesian. Read-speech training is known to transfer poorly to
   spontaneous speech, and our fine-tuned model's smoke score (36 errors vs base's 23) is
   weak evidence that it transferred *negatively* here.
3. **Smoke was mistaken for validation.** 0.2000 on 5 clips was read as a competitive
   signal. It was not; see above.
4. **No decoding, routing, or post-processing was applied.** The submitted ZIPs used
   greedy decoding, no language routing off the manifest label, no hallucination guard,
   and no text post-processing.

## Open questions this history cannot answer

These need in-domain data before they can be settled. They are the reason clearing the
MDC 403 is the critical path.

* Does the OpenSLR read-speech fine-tune help or hurt on spontaneous conversation?
  (Smoke says hurt, 13 words out of 115. Not decisive.)
* Is Whisper large-v3 genuinely better than Qwen3-ASR-1.7B here? (Currently unknown;
  the 1-word smoke difference proves nothing.)
* What orthographic convention do the MDC references use for Javanese diacritics
  (`dheweke` vs `dhèwèké`) and mid-sentence capitalisation? This is worth several WER
  points on its own — see [02-measured-findings](02-measured-findings.md).
* What is the actual per-language-label (`jav` / `ind` / `javind`) error breakdown?

## Changelog

* **2026-09-05** — Initial history compiled. Recomputed all checkpoint WERs from raw
  prediction CSVs. Verified the local scorer against `runtime/score.py`. Reconstructed
  the smoke-test error counts and retired the "Whisper is better" hypothesis as
  unsupported.
