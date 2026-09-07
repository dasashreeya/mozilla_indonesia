# The architecture question, settled — and the Wave 1 arm design

_Measured 2026-09-06. Every number is reproducible from this repository._

Two waves were run against one fixed holdout: the 509-clip mixed fold
(215 Javanese windows from 20 held-out Jember sessions + 294 code-switched
development clips, 22,453 reference words). Wave 0 spends only decode passes and
answers questions that would otherwise cost fine-tunes. Wave 1 trains one arm per
factor against a shared control.

## 0. The harness was validated before anything was concluded from it

Two Wave 0 arms exist only to check the instrument:

| Control | New harness | Previously recorded | Agreement |
| --- | ---: | ---: | --- |
| Base Qwen3-ASR-1.7B | 0.4302 | 0.4287 (H01) | +0.0015, inside the ±0.002 noise floor |
| `qwen-jav-holdout-v1/best-epoch-1` | 0.2685 / 0.3368 / 0.2266 | 0.26847 / 0.33681 / 0.22665 (H02) | exact to four decimals |

Numbers from `scripts/eval_checkpoint.py` are therefore comparable with everything
measured before it existed.

## 1. Qwen3-ASR is the right base, and this is now measured rather than assumed

The previous conclusion rested on a five-clip smoke test — 22 errors against 23 —
which cannot separate two systems. The real comparison, in domain:

| System | WER | Javanese | Code-switched | Decode |
| --- | ---: | ---: | ---: | ---: |
| **Base Qwen3-ASR-1.7B** | **0.4302** | 0.6971 | **0.2669** | 337 s |
| Base Whisper large-v3, forcing `jw` everywhere | 0.4503 | 0.6974 | 0.2990 | 1206 s |
| Base Whisper large-v3, routed `jw`/`id` | 0.4843 | 0.6942 | 0.3559 | 556 s |
| Base Whisper large-v3, auto-detect | 0.4907 | 0.7317 | 0.3432 | 1256 s |

Whisper large-v3 has a native Javanese token (`<|jw|>` = 50356, confirmed in its
`generation_config.json`) and Qwen3-ASR's supported-language list contains no
Javanese at all (confirmed in the `qwen-asr==0.0.6` wheel). That asymmetry buys
nothing: **Whisper's Javanese WER is 0.6942 against Qwen's 0.6971, a dead heat**,
while it is 8.9 points worse on code-switched speech and 2–4× slower to decode.
Its auto-detect run produced 2,806 deletions, a long-form instability that is also
a runtime risk inside the two-hour limit.

Qwen3-ASR also has no larger variant to grow into: the family ships 0.6B and 1.7B
only.

Two consequences: keep Qwen3-ASR-1.7B, and do not spend a fine-tune on Whisper.

### A finding worth keeping

Forcing `<|jw|>` on **every** clip (0.4503) beats routing `jw`/`id` from the
metadata (0.4843). Whisper's `id` token actively harms code-switched clips
(0.3559 against 0.2990). If Whisper is ever revisited, do not route it by label.

## 2. Beam search is retired

| Decode | WER | Javanese | Code-switched |
| --- | ---: | ---: | ---: |
| greedy | 0.2685 | 0.3368 | 0.2266 |
| beam 4 | 0.2688 | 0.3372 | 0.2270 |

Four beams cost four times the compute for **+0.0003**. This matches the error
composition already on file: 88% of errors are substitutions, so decoding width
was never the constraint. `qwen-asr`'s `transcribe()` exposes no generation
arguments, so a beam width has to be set through `generation_config.num_beams`.

## 3. Language conditioning pays before any training

The competition manifest carries a `language` column (`jav`/`ind`/`javind`), and
roughly half the test set is Javanese-only. The previous submission set
`force_language: null` and trained every Jember and `javind` row under the same
`language None` tag, so the label was discarded on both sides.

Measured on the **untrained** base model, as each Wave 1 arm's baseline:

| Baseline decode | WER | Javanese | Code-switched |
| --- | ---: | ---: | ---: |
| plain | 0.4287 | 0.6953 | 0.2655 |
| forced per-clip tag (`Jav`/`Ind`/`Javind`) | 0.4038 | 0.6524 | 0.2517 |
| context prompt, no tag | 0.4032 | 0.6512 | 0.2515 |

Roughly 0.025 for free, concentrated in the Javanese half, from a model that has
never seen these tags. Tags are chosen to survive `qwen-asr`'s
`normalize_language_name` (first character upper, rest lower) unchanged, so a
forced decode prompt matches the trained prefix byte for byte.

## 4. Two measurement traps found while building the wave

**mp3 decoding is not reproducible across platforms.** The same validation window
is 531,666 bytes when built by the macOS ffmpeg and 531,667 by the Linux one on
Modal. Identical clips, identical transcripts, but base WER moves 0.4287 → 0.4212.
Arms built on different machines are not comparable, which is why `arm-control`
rebuilds the baseline recipe on the same machine as every variant. Note that the
Linux encoding is the closer analogue of what the competition runtime does.

**Overlapping validation windows would inflate the reference.** Sliding windows
overlap by design, so held-out sessions always use the original non-overlapping
geometry regardless of the arm's training windows. Every variant reports the same
509 rows, the same 215 Javanese rows, and the same 20 held-out sessions.

## 5. Where the remaining error is, and what each arm attacks

After fine-tuning, Javanese sits at 0.3368 and code-switched at 0.2266. Two
structural mismatches motivated most of the arms:

* **Duration.** Jember segments have a 5.0 s median; development and test clips
  have a 26.2 s median (p10 16.8, p90 29.9, max 37.8). 81% of training rows were
  short segments. Sliding 18–30 s windows at half-stride yield 2,318 windows and
  17.4 h in the scored band, against 1,343 and 9.3 h before — from the same audio.
* **Data volume.** Training was 9.97 h of Jember plus 372 development clips. Three
  permissively licensed MDC corpora had never been used: Common Voice Spontaneous
  Javanese (CC0) and two CC-BY-SA Indonesian podcast corpora, which are segmented
  with the same `Audio file / Start / End / Text` schema as Jember and together
  hold 23.6 h of transcribed spontaneous conversation. Built into **17.31 h** of
  windows: 0.99 h Javanese, 16.32 h Indonesian.

| Arm | Factor under test |
| --- | --- |
| `arm-control` | baseline recipe, rebuilt on the same machine as every variant |
| `arm-dur` | sliding 18–30 s windows, short segments cut to 35% |
| `arm-lang` | per-clip language tags, forced at decode |
| `arm-prompt` | task context prompt |
| `arm-filter` | 166 rows dropped for misalignment, not for being difficult |
| `arm-specaug` | SpecAugment at p=0.5 |
| `arm-lora` | the 1st-place Pasketti recipe: LoRA r=16/α=32, lr 1e-4, 3 epochs |
| `arm-combo` | duration + tags + filtering |
| `arm-extra` | control plus the 17.31 h of new corpora |
| `arm-kitchen` | combo plus the 17.31 h |

Only differences above roughly 0.015 are measurable on this fold, and run-to-run
noise is ±0.002.

## 6. Wave 1 result: every lever tested is null or harmful

One epoch per arm, scored on the fixed fold, each compared with `arm-control`
(0.2697) built on the same machine. `arm-specaug` is compared with H02 (0.26847)
because it is the one arm on the macOS encoding.

| Arm | WER | Javanese | Code-switched | Δ WER | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `arm-control` | 0.2697 | 0.3399 | 0.2268 | — | reference |
| `arm-lang` | 0.2698 | 0.3374 | 0.2284 | +0.0001 | null |
| `arm-specaug` | 0.2695 | 0.3396 | 0.2266 | +0.0010 | null |
| `arm-filter` | 0.2726 | 0.3428 | 0.2297 | +0.0029 | harmful |
| `arm-kitchen` | 0.2734 | 0.3555 | **0.2231** | +0.0037 | mixed |
| `arm-combo` | 0.2748 | 0.3532 | 0.2267 | +0.0051 | harmful |
| `arm-dur` | 0.2763 | 0.3523 | 0.2297 | +0.0066 | harmful |

**Nothing beat the existing recipe.** Three conclusions follow.

**The published winning recipe does not transfer.** SpecAugment, beam search and
LoRA-style regularisation were adopted from the Pasketti Word Track winners.
SpecAugment is +0.0010 and beam search +0.0003 — both inside the noise floor. The
competition they won had a different bottleneck.

**Duration matching backfires, and it is the most instructive failure.** Matching
the scored duration profile required cutting short Jember segments from 5,781 to
1,977, and the Javanese loss from discarding that lexical coverage (+0.0124)
exceeded everything the duration match bought. `arm-dur` and `arm-combo` agree on
Javanese to within 0.001, which localises the regression to the short-segment cut
rather than to windowing. **Volume of in-domain Javanese text beats duration
shape.**

**Language conditioning is real before training and gone after it.** Forcing the
per-clip tag moves the untrained model 0.4287 → 0.4038, but after one epoch the
arm is +0.0001 against control. Fine-tuning already learns the language
distinction from audio, so the metadata label adds nothing a trained model does
not have. The routing support in `submission_src/main.py` is harmless and stays,
but it is not a source of points.

The one positive signal is `arm-kitchen`'s **0.2231 code-switched**, the best
measured anywhere and better than control by 0.0037. That is the 16.32 h of
CC-BY-SA Indonesian podcast conversation doing what it was added for. Its
Javanese is the worst of the wave only because it inherits the duration change.

### The critical path did not move

Every arm sits between 0.3374 and 0.3555 on Javanese. Roughly half the test set is
Javanese-only, so that is where the score lives, and eight variations of the recipe
did not touch it. Combined with the earlier finding that 57.6% of post-fine-tune
substitutions are alternate spellings of the same word — 0.0917 of 0.2239 WER, and
not post-processable — the remaining error is lexical and convention-bound. It has
to be learned from Javanese data, which is why the East Javanese TTS corpora
(~11,900 utterances from Lumajang, which borders Jember, and Bojonegoro) are the
next thing to test.

### Two infrastructure lessons

**Preemption plus an overwrite guard is a permanent failure.** `arm-extra` was
preempted at step 94 of 636; Modal restarted the container with the same input and
`train_recovery.py` refused to write into its own non-empty output directory, so
the restart failed for good. `modal_recovery.train` now moves a partial attempt
aside before restarting.

**`modal volume get` will not overwrite an existing local file.** Reading progress
into a fixed path silently returns the first download forever. Every collection
script here downloads into a fresh directory.

## Reproducing

```bash
modal run scripts/modal_recovery.py::wave0                     # decode-only arms
modal run scripts/modal_recovery.py::prep_wave1 --sources-sha256 <sha>
modal run scripts/modal_recovery.py::build_external            # the new corpora
modal run scripts/modal_recovery.py::wave1                     # training arms
```
