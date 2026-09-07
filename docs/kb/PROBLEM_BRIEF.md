# Javanese–Indonesian ASR: problem brief for an outside opinion

Self-contained. Every number is measured on our own data unless marked *estimate*.
Written 2026-09-06.

## 1. Task

Mozilla Data Collective "Lost in Transcription" competition, Javanese–Indonesian
track. Transcribe 2,118 held-out audio clips. Metric: corpus **word error rate**
after a normalizer that strips punctuation, lowercases only sentence-initial
letters (mid-sentence capitals are scored literally), and collapses whitespace.
We have a byte-exact local reimplementation of the official scorer, verified over
20,014 differential cases with zero divergences.

Submission is **code execution**: we ship a ZIP containing `main.py` plus model
weights; it runs offline in their container against the hidden test set, with a
**2-hour wall-clock limit** and no network. **3 submissions per 7 days.**
Deadline 2026-09-25.

## 2. Where we stand

| | WER |
| --- | ---: |
| Our public score (rank #38) | **0.3259** |
| Public leader | **0.2196** |
| Our local mixed-holdout score | 0.2697 |
| Target set by the project owner | **< 0.20 public** |

Local→public calibration, from two paired measurements: `public ≈ local + 0.052`.
So <0.20 public needs **local ≈ 0.148**, from 0.2697 today. The leader's 0.2196
implies local ≈ 0.168, so the target is achievable by *someone* — this is a
method gap, not a data-noise wall (see §7).

**Validation set** (fixed across all experiments): 509 clips / 22,453 reference
words = 215 Javanese windows from 20 held-out Jember sessions + 294 code-switched
development clips. Run-to-run noise **±0.002**; only differences above ~0.015 are
comfortably real.

Current best model: **Qwen3-ASR-1.7B**, full fine-tune, 1 epoch, lr 1e-5,
effective batch 16, on ~10 h Jember + 372 dev clips. Scores 0.2697 overall,
**0.3399 Javanese**, 0.2268 code-switched.

## 3. The test set is half Javanese, and that half is the problem

`test_metadata.csv` carries a `language` column with three values. Solving for
the share that reproduces our public score gives **~49% `jav` (Javanese-only)**,
~45% `javind` (code-switched), ~6% `ind`.

The development set is **0% `jav`** — 352 `javind` + 20 `ind`. So every
measurement made before we built a Jember-session holdout covered at most half
the task.

**Javanese WER has not moved across any experiment: 0.337–0.356.**

## 4. Data

| Source | Size | Licence | Used |
| --- | --- | --- | --- |
| Jember Javanese Spontaneous Speech | 9.97 h, 208 sessions, 6,679 segments | CC-BY-NC-SA-4.0 | yes — the only in-domain Javanese |
| Competition development set | 372 clips | CC-BY-SA-4.0 | yes |
| 2 Indonesian podcast corpora | 23.6 h transcribed conversation → 17.31 h of windows | CC-BY-SA-4.0 | tested |
| Common Voice Spontaneous Javanese | 221 clips / 1.33 h | CC0 | tested |
| TTS Lumajang + Bojonegoro (East Javanese read speech) | ~11.9k utts → 11.08 h | CC-BY-SA-4.0 | built, untested |
| OpenSLR-35 Javanese read speech | 156,643 utterances (~200 h) | CC-BY-SA-4.0 | staged, never trained |

Jember segments have a **5.0 s median**; dev/test clips have a **26.2 s median**
(p10 16.8, p90 29.9, max 37.8).

## 5. What has been measured and ruled out

Each tested as a single-factor arm against a same-machine control (0.2697) on the
fixed 509-clip fold, one epoch each. **None beat the control.**

| Lever | Δ WER | Verdict |
| --- | ---: | --- |
| Per-clip language tags (`jav`/`ind`/`javind`) forced at decode | +0.0001 | null |
| SpecAugment p=0.5 (freq 15, time 5%) | +0.0010 | null |
| Beam search, 4 beams | +0.0003 | null, 4× cost |
| Drop 166 misalignment-flagged rows | +0.0029 | harmful |
| Duration matching (sliding 18–30 s windows, shorts cut to 35%) | +0.0066 | harmful |
| All three stacked | +0.0051 | harmful |
| +17.31 h external conversational Indonesian | +0.0037 overall, **−0.0037 code-switched** | mixed |
| Whisper large-v3 as base (best of 3 configs) | +0.020 | rejected |

Also ruled out, on CPU using existing predictions:

* **Switching Javanese orthography.** Jember writes `ê è ě` (11,027 occurrences);
  dev writes only `é` (54). We strip Jember's diacritics to match dev. Verified
  correct: of 158 word types Jember spells with diacritics that also occur in
  dev, dev writes them **plain 1,601 times vs accented 3**. Scoring our output
  against Jember-native references would cost +0.0953 on the Javanese half, so
  this was a ±0.05 risk, now retired.
* **Spelling-correction maps.** Learned from confusions, applied held-out:
  +0.09 to +0.23. Worse even in-sample. Reason: the model produces these words
  **correctly more often than not** — `gak` is right 167×, wrong 89×; `waktu`
  right 54×, wrong 7×. Any context-free rewrite loses more than it gains.
* **Constraining output to the training vocabulary.** Out-of-vocabulary outputs
  are correct 1,651× vs wrong 1,727× — nearly 1:1. Snap-to-vocabulary: +0.0131.
* **Model choice.** Of every current ASR release, only two support Javanese at
  all: `microsoft/VibeVoice-ASR` (8.3B, MIT, native code-switching, hotwords)
  and `facebook/seamless-m4t-v2-large`. Cohere-transcribe-2026, Voxtral,
  Parakeet-v3, Nemotron-ASR and Phi-4-multimodal have **no Javanese**.
  Qwen3-ASR itself does not list Javanese, yet still beats Whisper here.
  Qwen3-ASR ships only 0.6B and 1.7B — no larger variant.

## 6. Where the error actually is

Error taxonomy on the Javanese half (2,109 errors on words the model has seen):

```
different seen word (mostly convention variants)  46.7%
invented word not in training vocabulary          27.8%
deleted (no word produced)                        16.8%
near-miss spelling variant                         6.1%
case only                                          2.1%
diacritic only                                     0.5%
```

Top confusions are **not mishearings** — they are the same spoken word under a
different transcription convention:

```
18x ning->neng   14x nggak->gak   13x kaya->kayak   11x karna->karena
10x nang->neng    8x iki->ki       6x wektu->waktu   6x kandhang->kandang
30x e->e...      (filler-annotation convention)
```

Out-of-vocabulary analysis: OOV words are **9.2% of the reference** but wrong
**59.5%** of the time, versus 27.2% for seen words. Still, OOV accounts for only
**18.1% of all errors** — **81.9% of errors are on words the model has already
seen.** The five candidate corpora combined cover only **28%** of the missed OOV
tokens, so the entire external-data plan is bounded at roughly
18.1% × 28% ≈ 5% of Javanese errors ≈ 0.007 local. Missed OOV words are local
proper nouns (`Ranuregula`, `Mloko`, `Aisyiyah`), English loans (`shadow`,
`teacher`, `camping-e`) and morphology (`maeme`, `negatife`).

## 7. The floor, and the biggest unexploited gap

**Annotation-inconsistency floor ≈ 0.05 WER upper estimate, realistically
0.02–0.03.** Measured from bidirectionally-confused, orthographically-close word
pairs and their reference frequencies: `lek` 1450 / `nek` 952, `gak` 1719 /
`nggak` 673, `nang` 416 / `neng` 868, `kaya` 1260 / `kayak` 142. The references
genuinely use both spellings, so some error is unavoidable — but **the floor does
not block a 0.148 local target.**

**The largest measured gap in the project is session generalisation:**

* The corpus audit: **83.8% of RMS variance is associated with session identity**.
* Jember sessions **seen** during training score **0.2427**; **held-out** sessions
  score **0.3368**. A ~**0.09 gap** from recording-condition familiarity alone,
  larger than every lever in §5 combined.

Nothing has been tried against this. SpecAugment masks time and frequency bands
but does **not** simulate channel variation — no RIR convolution, additive noise,
spectral tilt/EQ, or codec simulation has been tested.

## 8. Constraints

* 2-hour offline inference for 2,118 clips. Measured: Qwen3-ASR-1.7B decodes the
  full test set in **~23 minutes**, so there is room for 3–4 models. (An earlier
  note claiming ensembles "will not fit" was wrong.)
* Winning code must be MPL-2.0 with supporting software permitting commercial
  use. Jember is CC-BY-NC-SA — an unresolved licence question.
* Compute: Modal A100-80GB, on-demand, preemptible.
* ~13 GPU-jobs already spent this session; the owner wants no further waste.

## 9. What we already know works

**ROVER word-level ensembling of models we already have: −0.005 local**,
validated by selecting the pool on one half of the fold and scoring on the other
(0.2266→0.2210 code-switched; 0.3368→0.3338 Javanese). Notably, adding Whisper —
the *worst* single system at 0.4503 — improves the ensemble, so the diversity is
real. Per-clip oracle selection over the 9 systems we own would give **0.2314**,
so ~0.037 of headroom exists inside the hypotheses we already generate.

## 10. The question

**What method could take local WER from 0.2697 to ~0.148 (public 0.3259 → <0.20)?**

Constraints on a useful answer: it must not be one of the levers already measured
dead in §5, it must survive the observation in §6 that most errors are
convention choices on correctly-heard, already-seen words, and it must fit the
2-hour offline inference budget.

Specific things we would value an opinion on:

1. Is the **session-generalisation gap (§7)** the right target, and what is the
   best-evidenced way to close it — channel augmentation, speaker/session
   normalisation, invariance objectives, or something else?
2. Is there a principled way to attack **convention choice on seen words**, given
   that context-free mapping is measured harmful and the references are
   internally inconsistent?
3. Is a materially better base model available for Javanese, and is 8.3B
   affordable inside a 2-hour budget with vLLM?
4. Is there a way to exploit ~200 h of out-of-domain Javanese read speech
   (OpenSLR-35) given that a single-stage fine-tune on it transferred negatively,
   and the TTS corpora use a *different* orthographic convention than the target?
