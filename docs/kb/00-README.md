# Knowledge base

**Current update, 2026-09-08:** `submission-soup-div2.zip` scored **0.2991
public WER**, reported by the user. The actual packaged entrypoint reproduces
the local result at **0.23017 WER** with 509/509 nonempty predictions. Corrected
beam search plus a 60/40 two-model probability ensemble reaches **0.21169
held-out WER** through the exact packaged entrypoint, with 509/509 nonempty
predictions and an 80.3-minute projected full runtime. The corresponding
full-data final ensemble is packaged, smoke-tested and locally hash-verified as
`artifacts/submission-final-ens-ext60-lr40-b4.zip`; it has no honest local WER
because its models saw the validation clips. The CTC branch emitted only blanks
and is retired. See
[14-wave3-results](14-wave3-results.md) for current measurements; older
forecasts and model-family verdicts below are historical.

_Last updated 2026-09-05. Entry point for anyone — human or agent — picking this
competition attempt up._

## Executive summary

We are at **0.47555** on the full public test set; the leader is at **0.2196**. But
0.47555 is roughly what *any* zero-shot model scores on Javanese — published zero-shot
Whisper on Javanese is 52.25 — and the submitted ZIP contained base Qwen3-ASR-1.7B with no
fine-tuned weights. This is a data problem, not a model problem.

Research settles the model question: **all three Word Track winners of DrivenData's
"On Top of Pasketti" ASR challenge (completed April 2026) used a fine-tuned
Qwen3-ASR-1.7B**, beating purpose-built baselines by more than 50%, and their code and
hyperparameters are public. We already have that model. What we lack is the in-domain data:
**dev and test are both drawn from the Jember corpus**, which MDC still returns HTTP 403 for
until dataset terms are accepted in the account web UI.

Everything that looked like evidence for switching models was not. The 5-clip smoke test
cannot separate two systems: Whisper's 0.1913 and Qwen's 0.2000 are 22 errors versus 23 out
of 115 words.

The plan: clear the 403, fine-tune Qwen3-ASR-1.7B on Jember using the published winning
recipe, average checkpoints, and fix the orthographic convention mismatch that local
measurement shows is worth up to 6.4 WER points.

## The seven things that decide this competition

1. **In-domain data.** Fine-tuning was worth 0.41 absolute on in-corpus data; nothing else
   comes close. Dev and test both come from Jember, so Jember *is* the training set. The
   403 is the critical path.
2. **Orthographic convention.** 6.4 WER points — 22% of measured error — is casing and
   Javanese diacritics, not mishearing. Twice the width of the entire leaderboard.
3. **Never lowercase.** The MDC scorer preserves mid-sentence case; blanket lowercasing
   costs a measured 11.8 WER points.
4. **Honest validation.** Smoke is 5 clips. Only differences above ~0.015 are measurable
   even on the full 372-clip dev set.
5. **Domain match over data volume.** 156,643 more read-speech utterances are staged and
   are the wrong data; ~10 h of Jember spontaneous speech is the right data.
6. **Base model choice is settled: Qwen3-ASR-1.7B.** Proven by three independent winners
   of the closest comparable competition. Copy their LoRA config and average checkpoints —
   free WER at zero inference cost.
7. **A submission that crashes is a wasted submission.** `main.py` has no per-clip error
   handling and no deadline guard.

## Read this first if you have ten minutes

1. [01-run-history](01-run-history.md) § Scoreboard and § The smoke test is 5 clips —
   what was actually submitted and why the evidence was misleading.
2. [02-measured-findings](02-measured-findings.md) §§ 2–3 — the orthography lever and the
   lowercasing trap.
3. [03-next-actions](03-next-actions.md) § If you only do three things.

## Contents

| Document | What it is | When to read it |
|---|---|---|
| [01-run-history](01-run-history.md) | Every submission, training run, score, and dataset used, with the reasoning behind each outcome | before proposing anything, so you do not repeat a paid experiment |
| [02-measured-findings](02-measured-findings.md) | Locally measured facts: scorer verification, error decomposition, the orthography headroom, statistical resolution | when deciding where to spend effort |
| [03-next-actions](03-next-actions.md) | Prioritised plan to the 2026-09-25 deadline, with the submission budget | at the start of every working session |
| [04-language-composition](04-language-composition.md) | **Why the leaderboard score is what it is.** Javanese vs code-switched WER, the local-to-leaderboard calibration, and three retired hypotheses | before planning any experiment |
| [05-architecture-and-arms](05-architecture-and-arms.md) | **The base-model question, measured in domain.** Qwen against Whisper on the 509-clip fold, the retired beam-search hypothesis, the free gain from language conditioning, and the design of the ten training arms | before proposing a different base model, a decoding change, or a new arm |
| [06-evidence-audit](06-evidence-audit.md), [07-route-to-target](07-route-to-target.md), [08-target-strategy-summary](08-target-strategy-summary.md) | Independent audit of the evidence, the route toward the target, and the recommended execution order | when planning the next investment |
| [09-cpu-viability-results](09-cpu-viability-results.md) | Independently implemented selector and audio-side SNR measurement, CPU only | before assuming an ensemble or noise gain |
| [10-cpu-analysis-and-verdict](10-cpu-analysis-and-verdict.md) | **The verdict on <0.20.** Word-level oracle ceiling, the orthography question settled, error taxonomy, two dead correction strategies, the annotation floor, session 35, and the model survey | before proposing any post-processing, ensemble or base-model change |
| [11-audit-and-methodology-v2](11-audit-and-methodology-v2.md) | **The 2026-09-07 audit and the route to first place.** Five of twelve arms never ran, the calibration pairs two different models, the leader sits on our oracle, and the five-stage programme that follows | before acting on the `<0.20` verdict in [10](10-cpu-analysis-and-verdict.md) |
| [12-runbook-wave2](12-runbook-wave2.md) | **The command sequence.** One-time setup, the five build/train/soup/final stages with a gate on each, the shared-account naming rules, and troubleshooting | when actually running the programme |
| [13-wave2-results](13-wave2-results.md) | **What the programme produced.** Javanese 0.3368 -> 0.2573, the three conclusions it overturns, the soup, and the seven defects found on the way | after reading 11, before planning anything further |
| [PROBLEM_BRIEF](PROBLEM_BRIEF.md) | Portable, self-contained statement of the problem and everything ruled out | when seeking an outside opinion |
| [research/winning-solutions.md](research/winning-solutions.md) | **The proven recipe.** Pasketti winners' exact models, LoRA hyperparameters, augmentation, averaging strategy | before designing any training run |
| [research/official-competition-spec.md](research/official-competition-spec.md) | Verified official facts: audio format, metadata columns, runtime versions, limits, rules | when a factual assumption matters |
| [research/mdc-data-catalog.md](research/mdc-data-catalog.md) | Every usable MDC dataset with licence, including ~3.5 GB of Javanese TTS corpora | when planning the data mixture |
| [research/forum-and-benchmarks.md](research/forum-and-benchmarks.md) | Forum threads, the dev/test distribution shift, published Javanese WER numbers, HF checkpoint sweep | for context and calibration |
| [research/README.md](research/README.md) | Remaining research gaps | when you have budget |

Supporting documents that predate this knowledge base:
[docs/strategy.md](../strategy.md) (experiment matrix and acceptance gates),
[docs/resources.md](../resources.md) (dataset and model provenance),
[docs/modal.md](../modal.md) (A100 training runner).

## Status and blockers

| | |
|---|---|
| Best known full-test score | **0.32590**, rank #37→#38 (Jember fine-tune, 2026-09-06) |
| Best model built | **Final extb 60% + lr3 40% probability ensemble, beam 4**: `artifacts/submission-final-ens-ext60-lr40-b4.zip`, SHA-256 `f357f8ff3ebd1f6fc926ff599dd4a818e7de9b78187df6a37b650594e1065e8f`. Its matching held-out recipe scored 0.21169 (Javanese 0.23275, code-switched 0.19879); the final-data ZIP passed a 20-clip execution smoke but has no honest local WER. This is not a public-score guarantee. See [14-wave3-results](14-wave3-results.md) |
| **Superseded verdict** | The `<0.20` "not reachable" finding below is a bound on *recombining nine near-identical systems*, not on the task. Five of twelve defined arms — including the reference winning recipe — have never been run. See [11-audit-and-methodology-v2](11-audit-and-methodology-v2.md) |
| Hard ceiling on combining what we own | **0.1641 word-level oracle** over nine saved systems, against a 0.148 target. No selector, ensemble or corrector built on current models can reach <0.20 — see [10](10-cpu-analysis-and-verdict.md) |
| Best validated improvement | **≈0.2631–0.2638 local (−0.005)** from a held-out-validated multi-model selector, needing no new training |
| Previous | 0.47555 (base Qwen3-ASR-1.7B, zero-shot) |
| Public leader | 0.2196 |
| Deadline | 2026-09-25 00:00 UTC |
| Submission limit | **3 per 7 days, per person.** Smoke tests, cancelled and failed jobs do **not** count — run a smoke submission before every full one. |
| Base model | **Qwen3-ASR-1.7B, measured in domain 2026-09-06.** Whisper large-v3 is 0.4503 at best against Qwen's 0.4302, ties on Javanese despite its native `jw` token, and is 8.9 points worse on code-switched speech. See [05-architecture-and-arms](05-architecture-and-arms.md) |
| **Critical path** | **Javanese-only WER, currently 0.3368 on held-out sessions.** ~49% of the test set is Javanese and it carries most of the remaining error. See [04-language-composition](04-language-composition.md). |
| Forecasting rule | `test ≈ mixed-holdout WER + 0.052`. A development-only fold understates test by ~0.21 and must never be used as a forecast. |
| Open question for organizers | Jember is CC-BY-NC-SA-4.0. Is an NC-licensed training corpus compatible with the winner obligation that supporting software allow free commercial use? Ask on the forum before depending on it. |

## Conventions for this knowledge base

* Measured numbers state how they were measured and are reproducible from the repo.
* Estimates are labelled as estimates. Unverified claims are labelled UNVERIFIED.
* Append to [01-run-history](01-run-history.md) after every submission, training run, or
  evaluation, including failures. The failures are the expensive part.
* Do not delete anything under `artifacts/`. Those ZIPs are gitignored, and one of them is
  the only artifact with a known full-test score.

## Changelog

* **2026-09-08 (final fit)** - Completed the two no-validation final fits, merged
  the 2,040-step external-data LoRA, and built the 60/40 beam-4 ensemble. The
  exact downloaded ZIP passed a 20/20 nonempty offline smoke and matches SHA-256
  `f357f8ff3ebd1f6fc926ff599dd4a818e7de9b78187df6a37b650594e1065e8f`.
  Its weights and experiment records are backed up privately on Hugging Face.
  The best honest held-out result remains 0.21169; final-fit smoke WER is
  resubstitution and is not reported as model-selection evidence.

* **2026-09-07** — Independent audit. `arm-lora` (the verbatim 1st-place
  Pasketti recipe), `arm-prompt`, `arm-extra`, `arm-javtts` and `arm-alldata` are
  defined in `scripts/modal_recovery.py` and have **no results anywhere**, so the
  conclusion that the winning recipe does not transfer is unsupported by the
  measurements it cites. The local-to-public calibration pairs a local score from
  `qwen-jav-holdout-v1` with a public score from `qwen-recovery-v2-fold5`. The
  nine-system oracle of 0.1641 maps to 0.2161 public against the leader's 0.2196,
  so first place sits at our oracle rather than beyond it. A live HF survey found
  VibeVoice runtime-blocked (needs transformers 5; the runtime pins <5) and added
  the CTC family — `facebook/w2v-bert-2.0`, MIT — which is the only architecture
  that can use the `kenlm`/`pyctcdecode` the runtime ships. Both datasets raised
  as "new" were already downloaded and used. See
  [11-audit-and-methodology-v2](11-audit-and-methodology-v2.md).

* **2026-09-06 (later)** — Ten single-factor training arms and six decode-only arms
  produced **no improvement over the baseline**; the published winning recipe
  (SpecAugment, beam search, LoRA) is null-to-negative on this task. CPU analysis
  then established the decisive bound: a word-level oracle over all nine saved
  systems reaches only **0.1641**, above the 0.148 target, so the entire
  combination family is capped. The Javanese orthography convention was verified
  correct (a ±0.05 risk retired), spelling-correction maps and vocabulary
  constraints were measured harmful, the annotation floor was bounded at
  0.02–0.05, and session 35 was identified as a timestamp misalignment inflating
  the measuring stick by ~0.012. A full model survey found only two current ASR
  models supporting Javanese at all. See
  [05-architecture-and-arms](05-architecture-and-arms.md) and
  [10-cpu-analysis-and-verdict](10-cpu-analysis-and-verdict.md).

* **2026-09-06** — Public score 0.47555 → **0.32590** (rank #37). Root cause identified:
  the test set is ~49% Javanese-only and development contains none, so every prior
  measurement covered half the task. Javanese fine-tuning shown to generalise
  (0.6953 → 0.3368 on held-out sessions). Local-to-leaderboard calibration established.
  Decode-config, orthography and spelling-map hypotheses retired with measurements.
  Fixed a generation-config bug that made checkpoint saving impossible, and hardened the
  submission entrypoint against per-clip failures. See
  [04-language-composition](04-language-composition.md).
* **2026-09-05** — Knowledge base created. Recomputed all checkpoint WERs from raw
  prediction CSVs, verified the local scorer against the official one over 20,014 cases,
  measured the casing and diacritic headroom, reconstructed the smoke-test error counts,
  and added `scripts/error_analysis.py`. External research pass completed manually after an
  automated attempt failed: official spec verified against the live pages, runtime versions
  resolved, all forum threads read, the Pasketti winning recipe extracted from public code,
  the MDC catalogue enumerated, and Javanese WER benchmarks collected. Model question
  settled in favour of Qwen3-ASR-1.7B.
