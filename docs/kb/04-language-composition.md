# The leaderboard gap is language composition

_Measured 2026-09-06. Every number here is reproducible from this repository._

Public score moved from **0.47555 to 0.32590** (rank #37) on 2026-09-06. This document
records why the earlier score was what it was, what the fix was, and the calibration
that lets a local number forecast a leaderboard number for the first time.

## The finding

Qwen3-ASR is far worse on Javanese-only speech than on code-switched speech, and
roughly half the test set is Javanese. Every WER this project measured before
2026-09-06 came from a development set that is **0% Javanese-only**.

| Base Qwen3-ASR-1.7B, held-out | WER | Reference words |
| --- | ---: | ---: |
| Code-switched (`javind`, dev conversation 5) | 0.2655 | 13,929 |
| **Javanese only (20 held-out Jember sessions)** | **0.6953** | 8,524 |

Same weights, duration-matched (Jember windows 18–28 s against dev's 26.2 s median),
same corpus family. The only variable is language content. Solving for the Javanese
share that reproduces the measured public score:

```
0.47555 = s · 0.6953 + (1 − s) · 0.2655   →   s = 0.49
```

A ~49% Javanese share explains the entire pre-fix leaderboard score. Not the submission
harness, not orthography, not the base model.

### Why dev cannot see this

| `language` label | Dev clips | Share |
| --- | ---: | ---: |
| `javind` | 352 | 94.62% |
| `ind` | 20 | 5.38% |
| **`jav`** | **0** | **0%** |

Test `test_metadata.csv` carries the same `language` column with all three values.
Development carries none of the third. Any fold drawn from development measures at most
half the task.

## Fine-tuning on Jember fixes most of it, and it generalises

Run `qwen-jav-holdout-v1`: 20 whole Jember sessions held out, contributing only their
duration-matched windows to validation, with an assertion that no row from a held-out
session reaches training. One epoch.

| Held-out, never trained on | Base | After 1 epoch | Δ |
| --- | ---: | ---: | ---: |
| Javanese only | 0.6953 | **0.3368** | −0.3585 |
| Code-switched | 0.2655 | 0.2266 | −0.0389 |
| Combined | 0.4287 | 0.2685 | −0.1602 |

Javanese WER halved on sessions the model never saw. The absolute gain on Javanese is
**nine times** the gain on code-switched, which is where the headroom always was.

Contamination was real but not the whole story: on Jember segments it *had* trained on,
the same recipe scored 0.2427. The honest unseen number is 0.3368, so ~0.09 of the
optimistic figure was session memorisation and the remaining 0.36 improvement is real.

## Calibration: local WER now forecasts leaderboard WER

Two real leaderboard measurements, each paired with the composition-matched local score:

| Model | Local (mixed holdout) | Public test | Offset |
| --- | ---: | ---: | ---: |
| Base Qwen3-ASR-1.7B | 0.4287 | 0.47555 | +0.0468 |
| Jember fine-tune (v2, epoch 2) | 0.2685 | 0.32590 | +0.0574 |

```
test ≈ local_mixed_holdout + 0.052
```

The offset is stable across a 0.15 range of WER, which is the first forecasting tool this
project has had. It only holds for a validation set that contains Javanese: the
development-only folds understate test by ~0.21, not ~0.05.

**Forecasts must use the mixed holdout.** A dev-only number is not a leaderboard estimate.

### What the target costs

The mixed holdout is 37.96% Javanese by reference words; test is ~49%. To reach a public
score below 0.20, local combined must reach **0.148**:

| If Javanese reaches… | Code-switched must reach |
| ---: | ---: |
| 0.30 | 0.055 |
| 0.25 | 0.085 |
| 0.20 | 0.116 |
| 0.15 | 0.147 |

Both halves must roughly halve from their current 0.3368 / 0.2266. Bringing Javanese to
parity with code-switched alone lands near **0.27** public, not 0.20.

## What this rules out

Three hypotheses were retired with measurements rather than argument.

### The submitted decode config was never the problem

| Config | WER | D | Predicted words |
| --- | ---: | ---: | ---: |
| batch 8, 512 tokens | 0.266566 | 540 | 13,786 |
| batch 24, 256 tokens (as submitted) | 0.265130 | 538 | 13,795 |
| batch 24, 512 tokens | 0.265130 | 538 | 13,795 |

256 and 512 tokens give bit-identical WER, so nothing was ever truncated against a
13,929-word reference. Batch 24 is also 2.4× faster at identical quality.

### Orthography is worth ~0.004 in domain, not 0.064

| Measure | OpenSLR-35 | MDC dev |
| --- | ---: | ---: |
| Case + diacritic headroom | 0.0642 | **0.0044** |
| Mid-sentence capitalised words | 20.7% | 3.58% |

The lever [02-measured-findings](02-measured-findings.md) ranked second is negligible on
real competition data. Two rules remain confirmed harmful: `lowercase-all` costs
**+0.023** and `hyphen-to-space` costs **+0.056** — the latter far worse here than on
OpenSLR, where it was neutral.

### Spelling variants are 41% of error and are not post-processable

After fine-tuning, 57.6% of substitutions are alternate spellings of the same word
(1,277 of 2,216), accounting for 0.0917 of 0.2239 WER. They cannot be fixed with a map,
because the references use both spellings legitimately:

```
gak     74 in refs, 139 in hyps      nggak   78 in refs,  59 in hyps
itu    290 in refs, 276 in hyps      tuh     49 in refs,  76 in hyps
```

Cross-validated, a majority-vote map costs **+0.047**. Restricting it to spellings that
never appear in references at all is safe but worth only **−0.0024**. This error class
has to be learned from data.

Reproduce with `scripts/variant_analysis.py`.

## Statistical resolution

Run-to-run noise on the 294-clip code-switched fold is **±0.002**, measured by running an
identical recipe twice (v1 epoch 1 = 0.223921, v2 epoch 1 = 0.221696, baselines 0.26592
and 0.26657). Differences smaller than that are not real.

Training converges in **one epoch**: epoch 2 bought −0.000072, thirty times below that
noise floor. Every experiment should be single-epoch until the data mixture changes.

## Next levers, none of them yet attempted

1. **Javanese training data.** Current training is 10 hours of Jember and nothing else.
   OpenSLR-35 is public and needs no credentials (dialect-mismatched: Central Javanese,
   not Jember Pandhalungan). Common Voice Spontaneous Javanese (CC0) and the East
   Javanese TTS corpora need an MDC API key and are far closer to target.
2. **The `language` label.** Test metadata carries it and the rules permit routing on it.
   `submission_src/config.json` sets `force_language: null` for every clip.
3. **LoRA and checkpoint averaging.** All three Pasketti Word Track winners used
   r=16/α=32 plus model soup; the current recipe is a single full fine-tune.
   See [research/winning-solutions.md](research/winning-solutions.md).
4. **Data filtering.** The audit's review queue flags 1,471 training rows; none have been
   filtered.

## Changelog

* **2026-09-06** — Language composition identified as the cause of the dev/test gap.
  Javanese generalisation measured on held-out sessions. Local-to-leaderboard calibration
  established from two public scores. Decode-config, orthography and spelling-map
  hypotheses retired.
