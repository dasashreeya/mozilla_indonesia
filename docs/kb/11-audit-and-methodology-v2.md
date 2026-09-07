# Audit of the 2026-09-06 verdict, and the route to first place

_2026-09-07. Independent audit of everything recorded through commit `2bbba2f`,
plus a live Hugging Face model survey run today. Read this after
[10-cpu-analysis-and-verdict](10-cpu-analysis-and-verdict.md), which it partly
overturns._

## Summary

The project stopped because [10](10-cpu-analysis-and-verdict.md) concluded that
`<0.20` is "not reachable with anything currently in hand". That conclusion is
correct *and has been misread as a verdict on the task*. It is a verdict on
**recombining nine near-identical systems**, which is exactly what it says on
close reading.

Three things this audit establishes:

1. **The reference winning recipe was never actually measured.** `arm-lora` —
   LoRA r=16/alpha=32, lr 1e-4, 3 epochs, the verbatim 1st-place Pasketti config
   — is fully defined at `scripts/modal_recovery.py:394` and has **no row in the
   Wave 1 results table and no row in `experiments/results.csv`**. Nor do
   `arm-prompt`, `arm-extra`, `arm-javtts` or `arm-alldata`. Twelve arms defined,
   seven reported. The sentence "the published winning recipe does not transfer"
   rests on SpecAugment (one component of it) and beam search (not part of it at
   all). The missing runs are one `modal run` away.

2. **The oracle bound is measured over eight clones of one recipe.** All eight
   Qwen arms share a base model, a tokenizer, a data manifest, a full-fine-tune
   method, lr 1e-5, and a single epoch. A 3,684-word shared blind spot is the
   expected outcome, not a discovery about the task.

3. **The leader is sitting almost exactly on our oracle.** 0.1641 local + the
   recorded +0.052 offset = **0.2161 public**, against the leader's **0.2196**.
   Whoever is in first place is not doing something unreachable — they are
   roughly where a *perfect* combination of our own current hypotheses would
   land. That is the most encouraging number in this repository, and it inverts
   the strategic reading: we do not need a miracle, we need better base systems
   and one honest combination pass.

The route below is built on that. It is not a promise of `<0.20` — see
[section 5](#5-honest-expectation). It is the highest-expected-value ordering of
what has provably never been tried.

---

## 0. Which path to take

Section 3 lays out five stages. **One person cannot execute all five in 18 days**,
so this is the decision, made 2026-09-07 with all data available.

**Take the depth path: one Qwen3-ASR-1.7B, trained properly and averaged. Not a
model zoo.**

Four reasons, in order of weight.

**The external evidence says depth, not breadth.** All three Word Track winners of
the closest comparable competition used **one** fine-tuned Qwen3-ASR-1.7B. None
built a second architecture. 2nd place souped **eighteen runs of the same model**.
We have never run that recipe once, never trained past ~550 optimizer steps at
lr 1e-5, and never averaged two checkpoints.

**The two-hour runtime kills the ensemble path.** Measured on the 509-clip fold at
batch 8: Qwen 337 s, Whisper large-v3 **1,206 s** at its best config — 3.6x. The
recorded Qwen full-run is ~23 minutes for 2,118 clips, so Whisper adds ~84
minutes on top. That is ~107 minutes against a **120-minute hard limit**, before
two model loads, with a 5-minute margin. A souped Qwen is a **single model at
single-model cost**, and checkpoint averaging is the only combination technique
that is free at inference.

**The measured ensemble payoff is small.** The held-out selector over the existing
pool was worth **-0.005**. Realistic combiners capture about an eighth of the
oracle headroom. That does not pay for a 3-5 day CTC build plus the packaging
risk of shipping two models in one ZIP.

**The LM idea is separable from the CTC model.** N-best rescoring against a KenLM
needs no new architecture — just n-best from Qwen and a scorer. Beam 4 on Qwen
cost +0.0003 WER for no prohibitive time penalty, so n-best is affordable inside
the limit. Keep the language model; drop the model that was going to carry it.

### The path

| Order | Do this | Section | Why it is on the list |
| ---: | --- | --- | --- |
| 1 | Stage 0 in full, **0.4 first** | [3.0](#30-stage-0--fix-the-instrument-no-training) | Nothing downstream is trustworthy until the measuring stick is |
| 2 | `arm-lora` + `arm-prompt` + LR sweep + **soup** | [3.1](#31-stage-a--close-the-recipe-gap) | The recipe that was never run; one `modal run` |
| 3 | East Java TTS + **style conditioning** | [3.2](#32-stage-b--get-the-javanese-data-and-condition-on-style) | The only genuinely new Javanese data; Javanese is 59% of the error |
| 4 | Speed perturbation (`A4`) | [3.1](#31-stage-a--close-the-recipe-gap) | Data-starved, not over-fit — multiplies data and steps |
| 5 | **KenLM n-best rescoring on Qwen** (`D1`) | [3.4](#34-stage-d--model-the-convention-explicitly) | The one bet. Gate it on the n-best oracle first |
| 6 | **Final fit on 100% of labelled data** | [3.5](#35-stage-e--combine-then-fit-on-everything) | Free, certain, and never done — recovers 294 dev clips |

### Deliberately not doing

| Dropped | Why |
| --- | --- |
| CTC + w2v-BERT (`C2`) | 3-5 days of new engineering, and the LM value is reachable through `D1` without it |
| Whisper fine-tune (`C1`) | Blows the two-hour budget in any ensemble that uses it |
| VibeVoice (`C3`) | Runtime-blocked (F11) |
| Conditional corrector (`D2`) | Depends on a diverse pool that the depth path does not build |
| `arm-dur` repeats | Measured harmful (+0.0066), and the cause is understood |

Revisit `C2` only if Stage A's soup and Stage B's data together fail to move the
Javanese half below ~0.30 with a week still on the clock. At that point the
depth path has been falsified and diversity is the remaining bet.

---

## 1. Reframing: what the numbers actually say

### 1.1 The three targets

| Goal | Public WER | Implied local mixed-holdout WER (at +0.052) |
| --- | ---: | ---: |
| Beat the current public leader | < 0.2196 | **< 0.1676** |
| The stated stretch goal | < 0.20 | 0.148 |
| Where we are now | 0.32590 | 0.2685 |
| Word-level oracle over the nine saved systems | *(0.2161)* | 0.1641 |

Beating the leader needs local **0.1676**. The nine-system oracle is **0.1641**.
The gap between "perfect combination of what we own" and "first place" is
**0.0035** — one third of the measurable resolution of the fold. `<0.20`
genuinely does require new hypotheses; **first place mostly requires better ones**.

### 1.2 The calibration pairs two different models

[04-language-composition](04-language-composition.md) builds the
`test = local + 0.052` rule from this table:

| Model | Local (mixed holdout) | Public test |
| --- | ---: | ---: |
| Base Qwen3-ASR-1.7B | 0.4287 | 0.47555 |
| "Jember fine-tune (v2, epoch 2)" | 0.2685 | 0.32590 |

The second row is not one model. `0.32590` is `L05`, submitted from
`qwen-recovery-v2-fold5/best-epoch-2`, which trained on **all 208 Jember
sessions** — `J02` is explicitly labelled contaminated for that reason.
`0.2685` is `H02`/`W05`, from `qwen-jav-holdout-v1/best-epoch-1`, a different run
on a different manifest (`data/recovery-jav1`) with 20 sessions held out.

The offset that every subsequent target is derived from — including the 0.148
figure the "not reachable" verdict is measured against — comes from a
**mismatched pair**. It may still be roughly right; it has never been measured.

**Fix, costing one decode pass:** score
`qwen-recovery-v2-fold5/best-epoch-2` on the 509-clip mixed fold. That single
number turns the calibration into a measurement.

### 1.3 The holdout is weighted wrong for the task

The mixed holdout is **37.96% Javanese** by reference words; the test set is
**~49%**. Reweighting the same per-language numbers to the test composition:

| Model | Jav | Code-switched | Local @37.96% | Local @49% | Public |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen | 0.6953 | 0.2655 | 0.4287 | **0.4761** | 0.47555 |
| `qwen-jav-holdout-v1` | 0.3368 | 0.2266 | 0.2685 | **0.2806** | — |

At the test composition, base Qwen's local score predicts its public score to
within 0.0006. Report both weightings from now on; the 49% column is the one
that means something.

### 1.4 Where the score lives

At the test composition, Javanese contributes `0.49 x 0.3368 = 0.165` of the
0.281 and code-switched contributes `0.51 x 0.2266 = 0.116`. **Javanese is 59% of
the remaining error.** Every arm in Wave 1 landed between 0.3374 and 0.3555 on
Javanese. The critical path has not moved since the first fine-tune.

---

## 2. Audit findings

Ordered by how much they change what to do next.

### F1 — Five of the twelve defined arms have no result, and they are the important five

`scripts/modal_recovery.py:386` defines **twelve** Wave 1 arms.
`experiments/results.csv` and the Wave 1 table in
[05](05-architecture-and-arms.md) report **seven**. The five with no result
anywhere:

| Arm | Definition | Why it matters |
| --- | --- | --- |
| **`arm-lora`** | `lora_rank 16, lr 1e-4, spec_augment 0.5, epochs 3, keep_epochs` — the verbatim 1st-place Pasketti recipe | the entire "the winning recipe does not transfer" conclusion |
| **`arm-prompt`** | dataset `jav-prompt`, built with `CONTEXT_PROMPT` | measured -0.025 before training; never checked after |
| `arm-extra` | control + `external-v1`, no other change | preempted at step 94/636; external data's isolated effect |
| `arm-javtts` | control + `external-jav1` | needs the East Javanese TTS archives (F6) |
| `arm-alldata` | control + `external-v1,external-jav1` | same |

The code is written, tested, and launchable. Nobody has the numbers.

This matters most for `arm-lora`. Every **reported** arm ran
`scripts/train_recovery.py` defaults: **full fine-tune** — `--lora-rank` defaults
to 0 — lr **1e-5**, cosine, warmup 0.08, **1 epoch**. `arm-lora` is LoRA r=16 at
lr **1e-4** for **3 epochs**. A 10x learning-rate difference between an adapter
and a full-parameter update is a different optimisation problem, not a
neighbouring point in hyperparameter space.

So the seven reported arms varied **data**, not **method** — and the method was
then declared not to transfer, on the strength of SpecAugment (one component of
it) and beam search (not part of it at all).

```bash
modal run scripts/modal_recovery.py::wave1 --only arm-lora,arm-prompt,arm-extra --whisper false
```

### F2 — Checkpoint averaging was never measured

`EpochWER` writes `best-epoch-N` weights **only when validation WER improves**,
and `weights-epoch-N` **only when `--keep-epochs` is passed**. Nothing in
`experiments/results.csv` records that flag being passed, so a typical run leaves
exactly one averageable checkpoint behind. The Trainer's own
`save_total_limit=1` (line 403) keeps one resumable checkpoint, which is right
for preemption recovery and wrong for souping. `scripts/model_soup.py` was
written, tested, and has **never produced a result row**.

This is the technique that separated 1st from 3rd in the reference competition;
2nd place souped **eighteen** LoRA runs. It costs nothing at inference.

### F3 — Evaluation granularity hides the optimum, and early-stop truncates the schedule

`EpochWER.on_epoch_end` evaluates once per epoch. At ~550 optimizer steps per
epoch that is three measurements over a run. The reference solution saved every
5,000 steps with `save_total_limit: 10` and picked by hand.

Worse, `self.stale >= 2` stops training after two non-improving epochs while the
**cosine schedule is still configured for three**. A run stopped at epoch 2 never
reaches the low-learning-rate refinement phase. "Training converges in one epoch;
epoch 2 bought -0.000072" is therefore a statement about a truncated schedule,
measured on `dev-conv5` — the **code-switched-only** fold — and never once on
Javanese, which is 59% of the error.

### F4 — SpecAugment being null is the diagnostic, and it was read backwards

SpecAugment is a regulariser. It does nothing when a model is not overfitting.
9.97 h of Jember, ~2.3k windows, ~550 steps per epoch, a 1.7B model: this system
is **data-starved, not over-fit**. The correct response to a null regulariser
here is more and more varied data and more effective steps — not the conclusion
that the recipe family fails.

Speed/tempo perturbation, the standard low-resource multiplier, has **never been
tried**. The winners disabled it deliberately — with 329k samples.

### F5 — `arm-extra` never completed, so external data's isolated effect is unknown

`arm-extra` (control + 17.31 h, no other change) was preempted at step 94 of 636
and permanently failed. Only `arm-kitchen` ran, which is external data **plus**
the duration change that independently cost +0.0066.

Despite carrying that regression, `arm-kitchen` produced **the best code-switched
WER measured anywhere in this project: 0.2231**. The one arm that added data is
the one arm with a positive signal on half the task, and its clean version was
never run.

### F6 — The dialect-matched data was never downloaded

Javanese training audio, in total, across every arm: **9.97 h Jember + 0.99 h
Common Voice Javanese**. `external-v1`'s 17.31 h is 16.32 h Indonesian.

`data/downloads/` contains only `cv_id_scripted` and `cv_jv_spontaneous`. The
East Javanese TTS corpora that [05](05-architecture-and-arms.md) closes by naming
"the next thing to test" were never fetched:

| Dataset | ID | Size | Licence |
| --- | --- | ---: | --- |
| TTS Javanese-Lumajang Dialect *(borders Jember)* | `cml5bgysg00bhkr07g23kewke` | 718 MB | CC-BY-SA-4.0 |
| Bojonegoro Javanese TTS *(East Java)* | `cmltnzkug0012mh07k1obic7v` | 492 MB | CC-BY-SA-4.0 |
| TTS Javanese-Gresik Dialect *(East Java)* | `cmpwi0ac200zbo007hebmkzqy` | 297 MB | CC-BY-**NC**-SA-4.0 |
| Jombang Dialect-Javanese TTS *(East Java)* | `cmpo53l5b000pku07cem453ac` | 295 MB | CC-BY-**NC**-SA-4.0 |

`scripts/prepare_external.py` already has `tts_lumajang` and `tts_bojonegoro`
entries in `SOURCES`. The code is written. The archives were never obtained.

### F7 — On the two datasets asked about: both are already in use

- `cmqi294n3004mmf07wvybp3fn` = **Common Voice Spontaneous Speech 4.0 - Javanese**.
  Downloaded, prepared, and used as `cv_jv` in `external-v1`. It is **232 clips /
  1.39 h, 0 hours validated, 3 speakers** — 0.99 h after windowing.
- `cmqinqwef00xonr07z4vbfovw` = **Common Voice Scripted Speech 26.0 - Indonesian**.
  Downloaded, prepared, and used as `cv_id_scripted`. 60,804 clips / 67 h
  (33.76 h validated) of **read** Indonesian, 671 speakers.

Neither is new, and neither can carry the Javanese half: one is 1.4 hours, the
other is read Indonesian. They are already counted in the +0.0037 that
`arm-kitchen` moved. **They are not the lever.**

### F8 — No model has ever trained on all the in-domain data

The submitted model held out development conversation 5: **294 of 372 dev clips,
79% of the only in-domain code-switched labelled audio in existence**. The
best-validated model additionally held out 20 of 208 Jember sessions.

A final-fit model over 100% of labelled in-domain data has never been built. This
is the most certain unclaimed gain in the repository and it needs no new method.

### F9 — Training-set alignment has never been audited

Session 35 sits at **0.9352 WER** against 0.16-0.44 for the other nineteen
held-out sessions, and [10, section 7](10-cpu-analysis-and-verdict.md) identifies
it as a timestamp misalignment, not a recognition failure. It is validation-only.

Nothing has ever checked the **188 training sessions** for the same defect.
Misaligned supervision on Javanese sessions is a direct hit on the critical path,
and [10](10-cpu-analysis-and-verdict.md) itself calls this "the cheapest
remaining GPU question". It is still open.

### F10 — The context channel is wired up and set to the empty string

`submission_src/config.json` carries no `prompt` key, so
`submission_src/main.py`'s `qwen_decoder` calls
`model.transcribe(..., context=[""] * n)`.

Meanwhile [05, section 3](05-architecture-and-arms.md) measures a context prompt
on the **untrained** model at 0.4287 -> **0.4032**, concentrated in the Javanese
half (0.6953 -> 0.6512). That is 0.025 for free. `arm-prompt` was designed to
test whether it survives fine-tuning and **was never reported**.

### F11 — VibeVoice is blocked by the runtime (new, from today's HF survey)

[07-route-to-target](07-route-to-target.md) makes a bounded VibeVoice probe
investment priority #2. Checked live today:

| Path | Blocker |
| --- | --- |
| `microsoft/VibeVoice-ASR-HF` | `config.json` declares `transformers_version: 5.3.0.dev0`. The official runtime pins **transformers 4.57.6** with constraint **`<5`**. `model_type: vibevoice_asr` does not exist in 4.57. |
| `microsoft/VibeVoice-ASR` | Needs the vLLM plugin. Microsoft's instructions target vLLM **0.14.1**; the runtime ships **0.23.0**. |

Shipping it means vendoring a transformers-5-era modeling stack into the ZIP and
running it against a 4.57 lock, for a model whose Javanese pretraining share is
**0.0025%** and whose Javanese quality is unmeasured. **Deprioritise.**

### F12 — The model survey missed the CTC family

Today's sweep of every ASR model tagged Javanese on Hugging Face:

| Model | Params | Licence | Notes |
| --- | ---: | --- | --- |
| **`facebook/w2v-bert-2.0`** | 580M | **MIT** | Pretrained on 4.5M h / 143 languages incl. `jv` + `id`. Encoder only — add a CTC head. **Absent from the [10, section 8](10-cpu-analysis-and-verdict.md) survey.** |
| `facebook/mms-1b-all` | 1B | CC-BY-**NC**-4.0 | 1,162-language CTC with a `jav` adapter. Same NC licence question as Jember. |
| `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` | 300M | Apache-2.0 | Trained on exactly these three languages. Noted in [10](10-cpu-analysis-and-verdict.md) as an ensemble candidate; never tested. |
| `facebook/seamless-m4t-v2-large` | 2.3B | CC-BY-**NC**-4.0 | NC. |
| `Qwen/Qwen3-ASR-{0.6B,1.7B}` | — | Apache-2.0 | Confirmed today: still no larger variant. |

The survey in [10](10-cpu-analysis-and-verdict.md) asked "which models support
Javanese?" and answered with generative models. It missed the **CTC** class
entirely — which matters for [section 3.3](#33-stage-c--build-a-system-that-fails-differently)
below, because a CTC model is the only architecture that can use the `kenlm` and
`pyctcdecode` the official runtime ships.

---

## 3. The methodology

Five stages. Stage 0 is free. Stages A and B are the highest expected value per
GPU-hour. Stage C is the structural bet. Do not skip Stage 0.

### 3.0 Stage 0 — Fix the instrument (no training)

| # | Action | Cost |
| --- | --- | --- |
| 0.1 | Decode `qwen-recovery-v2-fold5/best-epoch-2` on the 509-clip mixed fold. Re-derive the local-to-public offset from a **matched** pair (section 1.2). | 1 decode |
| 0.2 | Report every WER at **both** 37.96% and 49% Javanese weighting. Make the 49% column the selection criterion (section 1.3). | code |
| 0.3 | Keep every epoch checkpoint, evaluate **4x per epoch**, and raise the early-stop patience above the schedule length (F2, F3). | code |
| 0.4 | Decode the **188 training sessions** with the current best model. Rank by per-session WER; inspect anything past the 90th percentile for the session-35 boundary-lag signature. Drop or realign (F9). | 1 decode |
| 0.5 | Re-measure the context prompt **after** fine-tuning, on the mixed fold, both languages (F10). | 1 decode |

0.4 is the one to run first. If even 10% of training sessions carry session-35's
defect, every arm in Wave 1 was trained on partly-misaligned Javanese
supervision, and that alone would explain why eight recipe variants all landed
within 0.018 of each other on Javanese.

### 3.1 Stage A — Close the recipe gap

Every arm here is a *method* change, not a data change. Select on **Javanese WER
at 49% weighting**, not mixed. `A1`, `A2` and `A5` need no new code.

| Arm | Configuration | Status |
| --- | --- | --- |
| **`A1-lora-winner`** | `arm-lora` exactly as already defined: LoRA r=16 alpha=32 dropout 0, lr **1e-4**, **3 epochs**, SpecAugment p=0.5, `keep_epochs`. Add `--evals-per-epoch 4 --patience 6`. | **defined, never run** |
| `A2-lora-long` | `A1` at 6-8 epochs. Locates the WER-optimal step instead of assuming it. | one flag |
| `A3-fullft-sweep` | Full fine-tune at lr 3e-5 and lr 1e-4 (current is 1e-5). Two runs. Tests whether the existing method was simply under-stepped. | one flag |
| `A4-speed` | Speed perturbation 0.9 / 1.0 / 1.1 on the control recipe. Triples the data **and** the optimizer steps (F4). | new prep option |
| **`A5-prompt`** | `arm-prompt`, already defined against the `jav-prompt` dataset. Then a per-`language` variant. | **defined, never run** |
| `A6-noise` | RIR plus additive noise on at most 50% of training audio, calibrated to the observed session SNR range. Motivated by the -0.684 SNR/WER correlation. | new prep option |
| **`A7-soup`** | `scripts/model_soup.py` over the top-k checkpoints by held-out Javanese WER **within** `A1`/`A2`, then **across** `{A1, A3, A4, arm-control}`. The 1st- and 2nd-place finish; free at inference. | **written, never run** |

```bash
# A1 + A5, then re-score every surviving checkpoint on the fixed fold
modal run scripts/modal_recovery.py::wave1 --only arm-lora,arm-prompt --whisper false
modal run scripts/modal_recovery.py::wave0            # eval harness over the new weights

# A7: soup the kept checkpoints, then evaluate the average like any other arm
python3 scripts/model_soup.py --output runs/soup-a1 \
    --checkpoint runs/arm-lora/weights-epoch-2 \
    --checkpoint runs/arm-lora/weights-epoch-3 \
    --checkpoint runs/arm-lora/weights-step-1650
```

Run `A1` and `A7` first. If `A1` alone does not beat `arm-control`'s 0.3399
Javanese, the "recipe does not transfer" conclusion becomes real — and it will
have been *measured*, for the first time.

`--evals-per-epoch` and `--patience` were added to
[`scripts/train_recovery.py`](../../scripts/train_recovery.py) on 2026-09-07
specifically for this stage; without them a 3-epoch run yields three validation
points and, unless `--keep-epochs` is set, one averageable checkpoint (F2, F3).

### 3.2 Stage B — Get the Javanese data, and condition on style

**B1.** Accept terms and download the East Javanese TTS corpora (F6). Lumajang
and Bojonegoro first: both CC-BY-SA-4.0, both East Java, ~11,900 utterances.
Everything downstream already exists — `tts_lumajang` and `tts_bojonegoro` in
`prepare_external.py`'s `SOURCES`, and the `arm-javtts` / `arm-alldata` arms in
`WAVE1`. Only the archives are missing.

```bash
export MDC_API_KEY='...'                       # after accepting terms in the web UI
python3 scripts/mdc_fetch.py download cml5bgysg00bhkr07g23kewke --extract   # Lumajang
python3 scripts/mdc_fetch.py download cmltnzkug0012mh07k1obic7v --extract   # Bojonegoro
modal run scripts/modal_recovery.py::build_external \
    --name external-jav1 --corpora tts_lumajang,tts_bojonegoro
modal run scripts/modal_recovery.py::wave1 --only arm-javtts,arm-alldata --whisper false
```

**B2.** Download the Javanese text corpora for Stage D's LM: Panjebar Semangat
(`cmk719jvs02z1nt07tjswt52s`), the Lumajang parallel corpus
(`cmtbjsbum00vmnr0701e5imnc`), and the Semarang parallel corpus
(`cmtbjkm6400vbnr075arsqkst`).

**B3 — style conditioning.** This is the mechanism that makes out-of-domain
Javanese usable without importing its orthography, and it is the fix for the
"OpenSLR-35 transferred negatively" result — that run had no target-style
adaptation stage at all.

The training target already carries a tag slot:

```json
{"audio": "audio/w-...flac", "text": "language Jav<asr_text>ndek kene..."}
```

Extend `LANGUAGE_TAGS` so the tag encodes **corpus style, not just language**:
`Jav` / `Javind` / `Ind` for Jember and development, and distinct tags such as
`Javtts` / `Javcv` / `Indpod` for every external corpus. Then **always decode
with the in-domain tag**. External audio trains the encoder; only in-domain rows
teach the decoder how Jember spells things. Precedent: Tanaka et al., Interspeech
2021 ([arXiv:2107.05382](https://arxiv.org/abs/2107.05382)).

This reuses `language_tag()` and `--force-language`, which already exist and are
already tested.

**B4.** Re-run the arm that died: external data alone, no duration change (F5).
Cap external rows at 35% or less of each batch and hold Jember exposure fixed.

**Do not** repeat `arm-dur`. Cutting short Jember segments cost more Javanese
lexical coverage than the duration match bought; that result is sound.

### 3.3 Stage C — Build a system that fails differently

[07, "Word recombination"](07-route-to-target.md) contains the finding that
should have driven everything after it: adding **base, zero-shot** Whisper to
three Qwen systems moves the word oracle from 0.2267 to **0.1882**. The worst
single system contributes the most new coverage. Diversity, not accuracy, is the
scarce resource.

**C1 — Fine-tune Whisper large-v3.** `scripts/train_whisper_recovery.py` exists
and has never produced an evaluated checkpoint. Encoder-decoder, different
tokenizer, different failure modes. Handle the 40 s clips with chunked long-form
decoding (`chunk_length_s=30`, `stride_length_s=5`). Judge it on the oracle it
unlocks, not on its standalone WER.

**C2 — A CTC system with a KenLM. The structural bet.**

Why this specifically, and why now:

- **46.7%** of Javanese errors on already-seen words are *the correct word under
  the wrong transcription convention*: `ning` to `neng`, `nggak` to `gak`,
  `kaya` to `kayak`, `karna` to `karena`.
- Context-free rewrite maps are **measured dead** (+0.09 to +0.23) for a precise
  reason: the model emits `gak` correctly 167 times and wrongly 89 times. A
  global rewrite loses 78.
- A **word-level n-gram language model trained on the Jember and development
  transcripts** encodes exactly those frequencies *conditioned on context*. It is
  the conditional version of the intervention that failed unconditionally.
- The official runtime ships **`kenlm 0.3.0`** and **`pyctcdecode 0.5.0`**.
  Nothing in this repository uses either. They only work with a CTC model.

Build:

```
facebook/w2v-bert-2.0  (MIT, 580M, pretrained on jv + id)
  |- character-level CTC head over the development/Jember charset
  |    (plain e, no diacritics -- the convention settled in doc 10 section 3)
  |- train on Jember + dev + CV-jav + East Java TTS
  |- decode with pyctcdecode, beam 100,
       4-gram KenLM over Jember + dev + Panjebar + Lumajang text,
       alpha/beta tuned on the holdout
```

Expect a worse standalone WER than Qwen. That is not the point. Its errors are
near-orthogonal, and the KenLM attacks the single largest error class directly.

Prefer `w2v-bert-2.0` (MIT) over `facebook/mms-1b-all` (CC-BY-NC-4.0) — the NC
licence carries the same unresolved winner-obligation question as Jember, and
there is no reason to take that risk twice.

**C3 — Do not probe VibeVoice** (F11). It cannot run in the official runtime
without vendoring a transformers-5 stack against a 4.57 lock, and its Javanese
pretraining share is 0.0025%.

### 3.4 Stage D — Model the convention explicitly

Gate every item here on a cheap oracle measurement first. If the oracle is not
well below the target, do not build the machine.

**D1 — N-best plus LM rescoring.** greedy is approximately beam-4 on the
**1-best**, which says nothing about the n-best list. Emit 8-16 hypotheses per
clip from the best Qwen, then score:

```
score(h) = log P_qwen(h) + lambda * log P_kenlm(h) + mu * |h|
```

with lambda and mu tuned on the holdout. **Measure the n-best oracle before
building the rescorer** — one decode pass answers whether the headroom exists.

**D2 — Conditional corrector with an explicit KEEP option.** Train a small
seq2seq on session-disjoint out-of-fold hypotheses mapped to references, taking
the concatenated hypotheses of `{Qwen-soup, Whisper-FT, CTC+KenLM}` as input.
Score it on **net edits: corrected minus introduced**, under the official
normaliser. This is the method [07, section 2](07-route-to-target.md) proposes
and nothing has tested (Chen et al.,
[arXiv:2310.13013](https://arxiv.org/abs/2310.13013)).

**D3 — Context biasing at decode, free.** Fill in `config.json`'s missing
`prompt` (F10). Static and identical for every clip, so it satisfies per-sample
independence trivially.

### 3.5 Stage E — Combine, then fit on everything

**E1.** ROVER or a selector over `{Qwen-soup, Whisper-FT, CTC+KenLM}`, pool
selected on one half of the fold and scored on the other. Worth -0.005 with the
old, homogeneous pool; a genuinely diverse pool should do better. Report the
held-out number, never the tuned-on-everything number.

**E2 — the final fit.** Retrain the selected recipe on **100% of labelled
in-domain data**: all 208 Jember sessions, all 372 development clips, external
corpora at the capped weight, style tags in place. No holdout — the recipe is
already chosen by then. This recovers the 294 development clips the submitted
model has never seen (F8).

**E3.** Merge LoRA into the base weights, package, run the free smoke submission,
then spend a full submission.

---

## 4. Schedule and submission budget

18 days to 2026-09-25 00:00 UTC. **3 submissions per 7 days per person**; smoke
tests, cancelled and failed jobs are free. Roughly 7-8 full submissions remain
solo, 14-16 with a second team member.

| Days | Work | Submit? |
| --- | --- | --- |
| 1 | Stage 0 in full. Start the MDC terms flow for Stage B — approval can take days, so it must not be on the critical path. | no |
| 2-4 | `A1`, `A3`, `A4`, then `A7` soup. | 1 if 0.015 or better |
| 4-6 | Stage B: style tags, `B4`, dialect data. | — |
| 5-9 | `C1` Whisper fine-tune and `C2` CTC+KenLM, in parallel with Stage B. | 1 |
| 9-12 | `D1` n-best oracle, then the rescorer if the oracle justifies it. `D3` free. | 1 |
| 12-15 | `E1` selector, held-out validated. | 1 |
| 15-17 | `E2` final fit on 100% of the data. Smoke, then submit. | 1-2 |
| 18 | Reserve. Never leave the last submission for the last day. | 1 |

**Promotion gate:** no full submission unless the 49%-weighted local holdout
improves by **0.015 or more** over the current best. Run-to-run noise is +/-0.002
and only differences above ~0.015 are measurable on 509 clips.

---

## 5. Honest expectation

Stages 0, A, B and E are low-risk and mostly measure things that should already
have been measured. My estimate for that block alone is local **0.22-0.25**
(public ~0.27-0.30) — a clear improvement on 0.32590 and not a winning score.

Reaching the leader requires Stage C or D to work. The supporting argument is
[section 1.1](#11-the-three-targets): first place sits 0.0035 from the oracle of
systems we already own, so the requirement is one genuinely diverse system plus
an honest combination pass, not a new state of the art.

Rough odds, with the whole programme executed: **beating 0.2196, 35-45%.
Public `<0.20`, 15-25%.** Those are judgements, not measurements, and they sit
above the "not reachable" reading of [10](10-cpu-analysis-and-verdict.md) for one
reason: three of the ten designed arms — including the reference recipe itself —
have never been run.

## What this document does not overturn

These findings from [10](10-cpu-analysis-and-verdict.md) survive the audit intact
and should not be revisited:

* Blanket lowercasing costs +0.023 (+0.118 on OpenSLR). Never lowercase.
* `hyphen-to-space` costs +0.056.
* Context-free spelling maps cost +0.09 to +0.23; vocabulary snapping +0.0131.
* The Javanese diacritic convention as implemented is correct (99.8% plain).
* Duration matching by cutting short Jember segments is harmful (+0.0066).
* Beam search over greedy is +0.0003 on the 1-best.
* Base Whisper large-v3 is a worse standalone model than base Qwen3-ASR-1.7B.
* The annotation-inconsistency floor is 0.02-0.05 and does not block the target.
