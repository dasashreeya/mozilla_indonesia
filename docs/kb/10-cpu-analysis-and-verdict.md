# CPU analysis, model survey, and the verdict on <0.20

_2026-09-06. Companion to [05-architecture-and-arms](05-architecture-and-arms.md),
which records the GPU wave. Everything here was computed on CPU from predictions
already paid for, or from transcripts. No GPU was used for anything in this file._

## Why this file exists

After the GPU wave produced no improvement (§05), the remaining questions were
answered without a GPU by mining the nine prediction sets the wave produced over
the same 509-clip fold. Several of those answers are stronger than anything the
wave itself established.

## 1. The decisive number: the candidate space is capped above the target

**Word-level oracle over all nine saved systems: 0.1641.**

At every reference position, credit a hit if *any* of the nine systems produced
that word. 18,769 of 22,453 reference words are recoverable; **3,684 words are
produced correctly by no system we own.** The figure ignores insertions
entirely, so a deployable system scores worse.

The working target is **0.148 local** (<0.20 public at the measured +0.052
offset). So a perfect oracle — one that reads the answers and picks the best word
at every position — **still misses the target by 0.016.**

An independent implementation reached 0.1731 using word choices plus seven
spelling-variant groups ([09-cpu-viability-results](09-cpu-viability-results.md)).
Two implementations, two methods, same conclusion.

**Consequence: no selector, corrector, ROVER, rescorer or ensemble built on the
current models can reach <0.20.** This bounds *combination*, not new training —
a better-trained model can produce words ours never produce. But eight variants
of one recipe sharing a 3,684-word blind spot is strong evidence that recipe
variation is exhausted.

## 2. Ensembling works, and is small

Word-level ROVER over the saved systems, pool selected on one half of the fold
and scored on the other:

| selected on | scored on (held out) | ROVER | single system |
| --- | --- | ---: | ---: |
| Javanese half | code-switched | **0.2210** | 0.2266 |
| code-switched half | Javanese | **0.3338** | 0.3368 |

Weighted ≈ **0.2638 (−0.005)**. Tuning the pool on the whole fold gives an
optimistic 0.2596; the held-out number is the honest one. A separately
implemented contextual selector reached 0.2631 with each evaluated session
excluded from fitting, removing 121 net errors and worsening no session.

Whisper large-v3 — the *worst* single system at 0.4503 — **improves** the
ensemble. The diversity is real even though the model is weak.

Utterance-level oracle over the nine systems is 0.2314, so ~0.037 of headroom
exists inside hypotheses we already generate; realistic combiners capture about
an eighth of it.

## 3. The Javanese orthography question, settled

Jember and the competition's development set use **different diacritic
conventions**:

| corpus | non-ASCII inventory |
| --- | --- |
| Jember | ê ×5,374, è ×4,833, ě ×820, ḍ ×17 — no é |
| Development | é ×54 only |

Our pipeline maps Jember's `è ê ě` to plain `e` to match development. Because
development contains **no Javanese-only speech**, that choice was never verified
against Javanese data. It is worth ±0.05: scoring our predictions against
Jember-native references instead costs **+0.0953** on the Javanese half.

**Resolved in favour of the current behaviour.** Of 158 word types Jember spells
with diacritics that also occur in development, development writes them **plain
1,601 times and accented 3 times** (99.8% plain):

```
jember 'lèk'   x813  -> dev 'lek'        jember 'bangêt' x209 -> dev 'banget'
jember 'ndèk'  x782  -> dev 'ndek'       jember 'akèh'   x207 -> dev 'akeh'
jember 'těrus' x757  -> dev 'terus'      jember 'nèng'   x452 -> dev 'neng'
```

A ±0.05 risk retired with transcripts alone.

## 4. Where the error is: convention, not acoustics

Errors on words the model has already seen (2,109 of them, Javanese half):

```
different seen word (mostly convention variants)  46.7%
invented word not in training vocabulary          27.8%
deleted                                           16.8%
near-miss spelling variant                         6.1%
case only                                          2.1%
diacritic only                                     0.5%
```

The top confusions are the same spoken word under a different transcription
convention, not mishearings:

```
18x ning->neng   14x nggak->gak   13x kaya->kayak   11x karna->karena
10x nang->neng    8x iki->ki       6x wektu->waktu   6x kandhang->kandang
 6x wedus->wedhus 6x lho->loh      6x ibukku->ibuku 30x e->e...
```

### Out-of-vocabulary words are not the bottleneck

| | share of reference | error rate | share of all errors |
| --- | ---: | ---: | ---: |
| seen in training | 90.8% | 27.2% | **81.9%** |
| never seen | 9.2% | 59.5% | 18.1% |

**81.9% of errors are on words the model has already seen.** The five candidate
external corpora *combined* cover only **28%** of the missed OOV tokens, which
bounds the entire external-data plan at roughly 18.1% × 28% ≈ 5% of Javanese
errors ≈ 0.007 local. That predicts, without training anything, the +0.0037
code-switched movement actually observed in `arm-kitchen`.

Missed OOV words are local proper nouns (`Ranuregula`, `Mloko`, `Aisyiyah`),
English loans (`shadow`, `teacher`, `camping-e`) and morphology (`maeme`,
`negatife`). Generic Javanese corpora do not contain Jember place names.

## 5. Two correction strategies, both measured dead

**Learned spelling-correction map.** Built from confusion pairs, applied to a
held-out half: **+0.09 to +0.23**. Worse even in-sample (+0.1972), which is the
diagnostic clue. Cause:

| word the model produced | correct | wrong | a global rewrite would |
| --- | ---: | ---: | --- |
| `gak` | 167 | 89 | **lose 78** |
| `waktu` | 54 | 7 | **lose 47** |
| `ki` | 53 | 19 | **lose 34** |
| `neng` | 37 | 64 | gain 27 |

The model produces these words correctly far more often than not, so any
context-free rewrite destroys more than it repairs. Only `neng`, `loh` and
`kandang` are net-wrong, worth ~43 words ≈ 0.002, below the noise floor. This
independently reproduces and explains the earlier +0.047 majority-vote result.

**Constraining output to the training vocabulary.** Out-of-vocabulary outputs are
correct 1,651 times and wrong 1,727 — nearly 1:1. Snap-to-nearest-vocabulary:
**+0.0131**.

## 6. The annotation-inconsistency floor

**≈0.05 WER as an upper estimate; realistically 0.02–0.03.**

Measured from word pairs confused in *both* directions and within edit distance 2
of each other, weighted by their reference frequencies: `lek` 1450 / `nek` 952,
`gak` 1719 / `nggak` 673, `nang` 416 / `neng` 868, `kaya` 1260 / `kayak` 142. The
references genuinely use both spellings, so some error is unavoidable.

A first attempt used transitive merging and produced a bogus 26,440-token class
containing `ya`, `dadi`, `wis`, `apa` and `yang` — plainly different words chained
by `ki↔iki↔iku↔kuwi`. That inflated the floor to 0.2013. **Do not merge
confusion pairs transitively.**

**The floor does not block the 0.148 target.** Combined with the leader's 0.2196
public, the gap is method, not data noise.

## 7. Session 35 is a timestamp misalignment

Per-session WER on the Javanese half is 0.16–0.44 for nineteen sessions and
**0.9352 for session 35** (432 reference words). Excluding it, Javanese improves
**0.3399 → 0.3081**.

It is not a recognition failure. Reference text lags the audio by roughly one
window:

```
window 0 REF tail : ...setengah enem kudu wis siap brangkat ning kampus
window 1 HYP head : jam setengah enam kudu wis ya berangkat neng kampus...
window 2 REF tail : ...mulai Senen nganti Sabtu ibu neng kampus terus
window 3 HYP head : dadi sed e mulai Senin nganti Sabtu ibuk neng kampus terus...
```

Session 35 is validation-only, so it does not poison training, but it inflates
the measuring stick by ~0.012 overall. Whether comparable offsets exist among the
188 *training* sessions is unknown and is the cheapest remaining GPU question:
one decode pass over the training audio, then flag sessions whose WER sits far
outside the distribution. No fluent speaker required to *find* them.

This was surfaced by per-session analysis; the aggregate word-class taxonomy in
§4 completely missed it.

## 8. Model survey: only two current ASR models support Javanese

Full Hugging Face sweep, 2026-09-06.

| model | Javanese | Indonesian | notes |
| --- | --- | --- | --- |
| `microsoft/VibeVoice-ASR` | yes | yes | 8.3B, MIT, native code-switching, hotwords, 16.7 GB |
| `facebook/seamless-m4t-v2-large` | yes | yes | 2.3B, 95 languages |
| Cohere transcribe 2026-06 | no | no | 13 languages |
| `mistralai/Voxtral-Mini-4B` | no | no | 12 languages |
| `nvidia/parakeet-tdt-0.6b-v3` | no | no | 24 languages |
| `nvidia/nemotron-3.5-asr` | no | no | 34 languages |
| `microsoft/Phi-4-multimodal` | no | no | 24 languages |

Qwen3-ASR ships only 0.6B and 1.7B — no larger variant. No gated model in the
top 100 ASR repositories is relevant. `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese`
is a small CTC model trained on exactly these languages and is architecturally
diverse, which makes it a candidate ensemble member rather than a base model.

VibeVoice-ASR is the only untested unknown large enough to change the verdict.
Its Javanese quality is unverified — Javanese is last in its language list, and
Whisper's native `<|jw|>` token turned out to buy nothing here.

## 9. Verdict on the <0.20 target

**Not reachable with anything currently in hand, and not by any amount of
post-processing applied to it.** The word-level oracle of 0.1641 caps the entire
combination family above the 0.148 requirement.

What survives as genuinely supported:

* **three-model selector, ≈0.2631–0.2638 local (−0.005)** — real, validated
  held-out, deployable with no new training;
* **channel/noise robustness is plausible but unmeasured** — estimated SNR
  correlates with session WER at −0.642, surviving removal of session 35, and
  83.8% of RMS variance is session-associated. No augmentation gain has been
  measured;
* **alignment auditing of the training set** — cheapest open question, one decode
  pass.

Earlier forecasts of a 0.04–0.09 aggregate improvement were **speculation stated
alongside measurements** and should not be relied on. The realistic ceiling of
everything measured is roughly 0.25–0.28 public. Reaching #1 requires a model
class that fails differently from every model tested here.

## Reproduction

All analyses in this file run on CPU from `runs/*/epoch-1.predictions.csv`,
`evals/w0-*.predictions.csv` and the corpus transcripts. See
[09-cpu-viability-results](09-cpu-viability-results.md) for the independently
implemented selector and audio-side SNR measurement.
