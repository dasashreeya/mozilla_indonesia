# Measured findings

_Last updated 2026-09-05. Everything here was measured locally on this machine and can be
reproduced with the commands shown. Nothing in this file is a citation or an estimate
unless it is explicitly labelled as one._

## 1. The local scorer is exact

`src/mdc_asr/normalization.py` is behaviourally identical to `normalize_text` in the
official `runtime/score.py` (runtime commit `05b8938`).

Differential test: 14 hand-written adversarial cases (mixed case, `[brackets]`, `(?)`,
nested parens, em dashes, `...`, `#x27;`, accented capitals, acronyms, empty string) plus
20,000 random fuzz strings drawn from an alphabet containing every character the
normalizer special-cases.

```
tested=20014 divergences=0
```

**Consequence:** local WER numbers are trustworthy and can be used for model selection
without a submission. This is the foundation everything else rests on, so re-run the
differential test whenever `runtime/` is refreshed with `make runtime`.

## 2. Where the error actually lives: 22% of it is orthography, not acoustics

Measured on the best fine-tuned checkpoint (`openslr35-full-v1/checkpoint-930`) against
its 1,100-utterance speaker-disjoint OpenSLR-35 validation set, 7,757 reference words.

The rows below relax **both** the reference and the hypothesis. They are diagnostics, not
achievable scores — the real references are hidden and cannot be relaxed. What they
measure is how much WER is caused by a *convention mismatch* rather than by mishearing,
which bounds what a text-side fix could ever recover.

| Diagnostic view | WER | Headroom vs actual |
|---|---|---|
| actual (scored as-is) | 0.293670 | — |
| case-insensitive | 0.263633 | 0.030037 |
| diacritic-insensitive | 0.260797 | 0.032874 |
| case + diacritic-insensitive | 0.229470 | **0.064200** |

**6.4 WER points — 22% of all error — is pure casing and Javanese diacritic mismatch.**
The model heard the right word and wrote it in the wrong convention.

For scale: the entire public leaderboard, from #1 (0.2196) to #4 (0.2506), spans 3.1
points. This single lever is twice the width of the whole leaderboard.

The same measurement on base Qwen zero-shot (300 clips, 2,258 words) gives 5.9 points of
combined headroom, so this is a property of the task and the metric, not of one
checkpoint.

Reproduce:

```bash
PYTHONPATH=src python3 scripts/error_analysis.py <ground_truth.csv> <predictions.csv>
```

## 3. Why: the MDC normalizer preserves mid-sentence case

`runtime/score.py` lowercases **only** sentence-initial letters, and even then only when
the following letter is not also uppercase (to protect acronyms). Every other capital is
scored literally.

In the OpenSLR-35 validation references, **20.7% of words (1,606 of 7,757) are
mid-sentence capitalised.** Measured effect of the reflex every ASR practitioner has:

| Post-processing rule applied to predictions | WER | Delta |
|---|---|---|
| as-is | 0.293670 | — |
| **lowercase everything** | **0.412015** | **+0.118345** |
| strip punctuation | 0.293541 | −0.000129 |
| hyphen → space | 0.293412 | −0.000258 |
| collapse repeated words | 0.295991 | +0.002320 |

**Blanket lowercasing costs 11.8 WER points.** It would move a leader-grade 0.2196
submission to roughly 0.34. Do not do it, and check that no preprocessing step in any
future pipeline does it silently.

The corollary matters too: punctuation stripping and hyphen handling are worth
essentially nothing (±0.0003), because the normalizer already removes that punctuation.
Effort spent there is wasted.

## 4. The dominant error class is Javanese diacritics

Substitution analysis on checkpoint-930: **2,002 substitutions, of which 10.5% (211)
differ from the reference only by capitalisation.** The top confusion pairs:

| Count | Reference → hypothesis | Class |
|---|---|---|
| 11 | `dheweke` → `dhèwèké` | diacritic |
| 7 | `Indonésia` → `Indonesia` | diacritic |
| 4 | `gedhe` → `gedhé` | diacritic |
| 3 | `sawise` → `sawisé`, `akeh` → `akèh`, `nate` → `naté`, `laré` → `lare`, `dhèwèké` → `dheweke` | diacritic |
| 3 | `siji` → `Siji`, `azrul` → `Azrul`, `kelurahan` → `Kelurahan`, `Kiribati` → `kiribati` | case |
| 3 | `mesti` → `mesthi`, `duwé` → `dhuwé` | retroflex `dh`/`th` spelling |

Note that `dheweke → dhèwèké` (11 occurrences) and `dhèwèké → dheweke` (3 occurrences)
**both** appear. The OpenSLR-35 references are internally inconsistent about diacritics,
which caps how much a global rule can recover on that corpus.

**The actionable question is therefore not "should we strip diacritics" but "which
convention do the MDC references use".** That is a one-hour measurement on the dev set
the moment the 403 clears, and it is the single highest-value hour available:

* If MDC transcripts are diacritic-free, strip diacritics from predictions and train on
  diacritic-free text. Expect a large fraction of that 3.3-point headroom.
* If MDC transcripts use diacritics consistently, train on them and leave predictions
  alone — the model will learn the convention from in-domain data.
* If MDC transcripts are inconsistent, the headroom is genuinely unrecoverable and the
  effort belongs elsewhere.

Do the same for capitalisation: measure the share of mid-sentence capitalised words in
the MDC dev references before choosing any casing policy.

## 5. Error composition points away from decoding work

Checkpoint-930: **S=2,015, D=117, I=146** — substitutions are 88% of all errors.

* Low deletions mean the model is not dropping speech; the acoustic front end is working.
* Low insertions mean hallucination and repetition loops are **not** currently a problem
  on this data. A hallucination guard is cheap insurance for spontaneous audio but is not
  where the points are.
* Overwhelming substitutions mean the errors are lexical and orthographic: wrong word,
  wrong spelling, wrong case. That points at in-domain data, vocabulary and text
  convention — not at beam width or temperature fallback.

This is the empirical basis for prioritising data and text convention over decoding
parameter sweeps.

## 6. Fine-tuning works, and is near-exhausted on this corpus

| Stage | Data | WER (OpenSLR-35 held-out) |
|---|---|---|
| base Qwen3-ASR-1.7B | none | 0.705934 |
| stage1 | 2,700 clips | 0.521701 |
| full-v1 ckpt 310 | 9,900 clips | 0.333376 |
| full-v1 ckpt 620 | 9,900 clips | 0.298311 |
| full-v1 ckpt 930 | 9,900 clips | 0.293670 |

Fine-tuning is worth **0.41 absolute** on in-corpus data — by far the largest lever
observed. But the last third of training bought only 0.005, so additional epochs on
OpenSLR-35 are close to worthless. The next gain must come from *different* data,
specifically spontaneous conversational data, not more of the same.

The 156,643 staged utterances from shards 1–f would deepen the same read-speech
distribution. Given the flattening curve and the domain mismatch, finishing that run is
**not** the highest-value use of the remaining A100 time.

## 7. Statistical resolution: what differences are even measurable

| Evaluation set | Approx. reference words | 95% CI half-width at WER≈0.25 | Smallest trustworthy delta |
|---|---|---|---|
| Smoke test (5 clips) | 115 | ±0.079 | nothing — unusable |
| MDC dev (372 clips) | ~3,700 (estimate) | ±0.014 | ~0.015 |
| Full public test (2,118 clips) | ~21,000 (estimate) | ±0.006 | ~0.01 |

Computed as `1.96 * sqrt(p(1-p)/N)`. These are optimistic: word errors cluster within an
utterance, so the effective sample size is smaller and the true intervals are wider.
Bootstrap over utterances rather than trusting this closed form.

Three conclusions:

* The 5-clip smoke test cannot distinguish any two systems. Ever.
* On the 372-clip dev set, only differences above roughly 0.015 are real. Plan
  experiments that aim for gains larger than that, and do not chase 0.003 improvements.
* On the leaderboard, #1 (0.2196) vs #2 (0.2247) is inside the noise floor; #1 vs #4
  (0.2506) is not. The top three are effectively tied.

## Caveats

Every number above was measured on **OpenSLR-35 Javanese read speech**, which is not the
competition domain. The *mechanisms* are verified from the official scorer source and
will hold on the real test set. The *magnitudes* must be re-measured on MDC dev data as
soon as the 403 clears. Findings 1, 3 and 7 are domain-independent; findings 2, 4, 5 and
6 need re-measurement.

## Reproducing all of this

```bash
# Build ground truth for a prediction CSV from the OpenSLR transcript table, then:
PYTHONPATH=src python3 scripts/error_analysis.py <ground_truth.csv> <predictions.csv> \
    --group-by speaker --top-confusions 30
```

`scripts/error_analysis.py` reports corpus and per-group WER, the orthography headroom
table, the reference casing profile, the post-processing rule ablation, confusion pairs
tagged by class, and the ten worst clips.
