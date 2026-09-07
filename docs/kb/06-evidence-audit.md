# Problem brief: evidence audit and next decisions

Inspected 2026-09-06. This is a second opinion based on the actual preparation
manifests, source code, experiment ledger, and recovered prediction files.
No GPU jobs or competition submissions were launched. Existing training and
submission code was left unchanged.

The current fine-tune is a sound baseline. The available evidence does **not**
identify a method known to reach <0.20 public WER, establish a required 0.148 local
WER, or prove that recording channels or annotation conventions cause most of
the remaining errors. Those are hypotheses to test, not constraints that should
exclude other approaches.

## What is actually present

The project is the nested Git repository `mozilla_indonesia/`, separate from the
parent Spanish–Nahuatl project. Both working trees contain pre-existing changes.

| Location | Contents verified locally |
| --- | --- |
| `docs/kb/PROBLEM_BRIEF.md` | Current problem statement and proposed bottlenecks |
| `experiments/results.csv` | Baselines, public submission records, Wave 0 and seven completed Wave 1 arms |
| `data/` | Seven source archives: development, Jember, two podcasts, Common Voice spontaneous Javanese, two TTS corpora |
| `data/recovery-jav1/` | Prepared mixed holdout, training/validation JSONL, audio, metadata, provenance and receipt; about 2.1 GB |
| `data/recovery-v1/` | Earlier preparation with all Jember sessions in training and conversation 5 held out; about 2.2 GB |
| `scripts/` | Preparation, full/LoRA training, checkpoint evaluation, model averaging, Modal orchestration, packaging, diagnostics |
| `src/mdc_asr/` | Normalizer, WER scorer and support utilities |
| `submission_src/` | Qwen/Whisper offline entrypoint and configuration |
| `artifacts/ckpt-epoch2/` | Locally available `best-epoch-2` model weights; about 3.8 GB |
| `artifacts/submission-qwen-recovery-v2-epoch2.zip` | Existing submission archive; 4,080,138,455 bytes in its receipt |
| `reports/private/` | Audio, overlap, session, transcript and earlier prediction diagnostics |
| `tests/` | Preparation, scoring, training-arm, submission and infrastructure tests |

Nine full-fold prediction CSVs, six evaluation receipts and several analyses were
found in a previous session's `/private/tmp/.../scratchpad`, outside the repository.
Twenty-three small files (1,399,370 bytes) have been copied to the ignored directory
`reports/private/problem-brief-audit-20260906/recovered/`. The recovery manifest
records original paths and SHA-256 hashes. This preserves evidence that otherwise
depends on temporary-directory retention. Most arm weights and prepared external
datasets are referenced on Modal; their continued availability was not checked.
The staged full OpenSLR continuation is documented, not verified locally here.

## Reproduced results

The recomputation checks duplicate/missing prediction IDs, hashes all three saved
manifest files against their preparation receipt, verifies Jember session
separation from row provenance, and scores every saved hypothesis with the repo
normalizer/scorer.

| System | Mixed WER | Jember WER | Development WER |
| --- | ---: | ---: | ---: |
| Earlier held-out Qwen fine-tune | 0.268472 | 0.336814 | 0.226649 |
| Same-machine control | 0.269719 | 0.339864 | 0.226793 |
| Language tags | 0.269808 | 0.337400 | 0.228444 |
| SpecAugment | 0.269541 | 0.339629 | 0.226649 |
| Filtering | 0.272614 | 0.342797 | 0.229665 |
| Duration change | 0.276266 | 0.352299 | 0.229737 |
| Combined changes | 0.274752 | 0.353238 | 0.226721 |
| Combined changes + external corpora | 0.273371 | 0.355467 | 0.223132 |
| Base Whisper, forced Javanese | 0.450274 | 0.697443 | 0.299016 |

All nine scores reproduce the experiment ledger to its displayed precision.
SpecAugment and the earlier fine-tune use the earlier audio encoding; comparisons
with the same-machine control do not isolate their training effects. This audit
re-scores saved text; it does not reproduce inference or validate Modal audio hashes.

The whole-clip oracle is **0.231417**: for each clip, use its reference to select
the least erroneous of nine hypotheses. This is unattainable reference-assisted
selection, not a deployable score. A selector restricted to these complete
hypotheses cannot reach 0.148. Word-level recombination or newly generated text
can exceed this oracle, so it is not a bound on every possible ensemble.

## Corrections that change the diagnosis

**The local-to-public calibration is not a matched checkpoint comparison.**
`H02`/`W05` evaluates `qwen-jav-holdout-v1/best-epoch-1` on a preparation that
excludes 20 Jember sessions. `L05` submits
`qwen-recovery-v2-fold5/best-epoch-2`, trained with all Jember sessions. Treating
their scores as one local/public pair confounds training data, checkpoint and
evaluation distribution. Even two correctly matched pairs would not establish
a constant offset for a different model family. The 0.148 target is a scenario,
not a validated requirement. A leader at 0.2196 does not demonstrate that <0.20
is attainable on this evaluation.

**The approximately 49% Javanese share is inferred, not counted.** The equation
assumes the local per-language error rates also hold on public test audio. For
corpus WER, its mixture coefficient represents reference-word mass, not clip
count. Domain difficulty can also explain a public/local gap. The actual hidden
metadata and reference-word shares were not available in this audit. Do not
treat the brief's 49/45/6 split as measured test composition.

**The nine-point gap does not isolate channel generalization.** `J02` explicitly
records 0.242690 on training-contaminated audio using the epoch-2 model. `H02`
uses another model on unseen sessions. Speaker, content, annotation, training
exposure and checkpoint all differ. The audio audit does independently confirm
that Jember session identity explains 83.76% of RMS variance, but variance in
audio level is not variance in recognition error. No causal channel effect of
0.09 WER has been measured. Even recovering 0.09 on Jember alone would improve
this mixed fold by only about 0.034 at its current 37.96% Jember word share.

**The error taxonomy does not prove that most errors are spelling conventions.**
Recovered `seen.py` labels any remaining substitution whose hypothesis word is
in the training vocabulary as `different seen word (confusion)`. It performs no
phonetic or contextual check. Its near-miss class uses the first four plain,
lowercase characters. Neither operation establishes that the audio was correctly
heard. The brief strengthens these heuristic categories into a linguistic claim
that the code does not support. Reference frequencies of alternative spellings
also do not identify an irreducible annotation-noise floor.

**The OOV calculation excludes insertions.** Its reference alignment yields
2,109 seen-word errors and 465 OOV errors: 465 / 2,574 = 18.07%. Official total
Jember edit errors are 2,897. Under that scratch alignment, 323 insertion errors
are outside the taxonomy, and OOV errors are 16.05% of total edit errors. Tied
alignments can change S/D/I attribution, although total WER is unchanged.
More importantly, vocabulary overlap places no upper bound on the benefit of
additional audio: repeated words can improve acoustic recognition, pronunciation
coverage and contextual selection. The claimed 0.007 ceiling on external-data
benefit is not established.

**Some supposedly isolated experiments were not isolated.** The duration arm
changes both window construction and short-segment retention. It rules out that
combined recipe; it does not identify which change caused the regression. The
completed external-data arm is `arm-kitchen`, which also includes duration,
language-tag and filtering changes. Its matched no-external counterpart is
`arm-combo`; the mixed difference there is -0.001381. The standalone `arm-extra`
is documented as preempted. No completed result for it, LoRA, or the prompt arm
appears in the inspected ledger/prediction bundle. A base-model Whisper comparison
also does not establish how an adapted Whisper would perform. These are reasons
to qualify the conclusions, not reasons to repeat all the experiments.

## Validation strengths and limits

Verified preparation: **188 training Jember sessions, 20 validation sessions, no
session overlap**. Validation contains 215 Jember windows and 294 development
clips, totalling 22,453 reference words. Training contains **78 unique development
clips repeated eight times**, not all 372. The 294 development validation clips
are one conversation (`5`) with two speakers (`1`, `6`). Jember speaker identities
are absent, so session separation is not evidence of speaker separation.

A paired bootstrap over the 20 Jember sessions, 5,000 draws with seed 20260906,
gives the following conditional 95% percentile intervals for WER differences
against the same-machine control:

| Arm | Jember delta | Interval |
| --- | ---: | ---: |
| Language tags | -0.00246 | [-0.00897, +0.00251] |
| Filtering | +0.00293 | [-0.00482, +0.01153] |
| Duration change | +0.01244 | [+0.00439, +0.01904] |
| Combined changes | +0.01337 | [+0.00430, +0.02134] |
| Combined + external | +0.01560 | [+0.00385, +0.02465] |

These intervals condition on the trained checkpoints and sampled sessions. They
do not include training-seed uncertainty or correct for searching many arms;
shared speakers could further reduce independence. There is insufficient
conversation diversity to estimate cross-conversation development uncertainty.
Avoid a universal rule that every delta below 0.015 is noise or that a small
positive delta proves a method harmful.

One defensive code issue remains: `prepare_recovery.py` compares numeric session
IDs from `source_rows` against full source paths in its final `leaked` assertion.
That assertion would miss leakage. The actual selection logic and independently
checked saved manifests are disjoint; this is a guard weakness, not evidence of
existing contamination. Correct the representations before new preparations.

## Recommended next decisions

1. **Establish the error mechanism before choosing the large experiment.** Have a
   fluent Jember/Pandhalungan speaker review about 100 stratified error spans from
   the 20 held-out sessions, listening to the audio and distinguishing recognition,
   dialect/word choice, transcription convention, alignment and unintelligibility.
   Keep original scored references intact. Frequencies and edit distance alone
   cannot substitute for this review. Use the acoustic audit to check whether
   session WER tracks level, noise or bandwidth after considering duration and
   vocabulary difficulty; treat any correlation as diagnostic, not causal proof.
2. **Use existing hypotheses for a bounded, inexpensive ensemble improvement.**
   The recovered ROVER code votes only at pivot-token positions and discards
   insertions from other systems, so it cannot restore a word missing from the
   pivot. An insertion-aware confusion network is an untested extension. Choose
   weights/pools on separate groups and evaluate on held-out groups. Preserve
   case and the official normalizer. The recovered cross-selection script chooses
   on Jember and tests on development, then reverses the roles; that is useful
   transfer evidence, but not another independent validation conversation.
3. **If the review supports acoustic sensitivity, test one controlled channel
   augmentation arm.** Keep source rows, targets, sampling, steps and validation
   audio fixed; alter only training audio, with a retained clean fraction and
   modest gain/EQ, codec or noise/RIR variation motivated by measured conditions.
   Preserve timing and intelligibility. First confirm that the transform does
   what it claims on a small sample. Use a matched control and per-session deltas.
   Do not promise that augmentation will close the observed nine-point gap.
4. **If convention choice dominates the reviewed sample, learn contextual
   decisions.** Use out-of-fold ASR hypotheses paired with references for a small
   candidate reranker or conservative correction model, with an explicit keep
   option. Include local linguistic context and, where affordable, audio-based
   candidate scores. Disjoint training/evaluation groups are essential; a
   corrector trained on hypotheses from its own evaluation clips would invalidate
   the estimate. Context-sensitive code-switching correction has published
   precedent, but no Jember result has been established here.
   [Primary research](https://arxiv.org/abs/2310.13013).
5. **Reopen staged external-data adaptation only as a distinct hypothesis.**
   A capped read-speech adaptation stage followed by a final Jember/dev adaptation
   stage differs from submitting a read-speech fine-tune directly. Compare with
   a compute-matched in-domain continuation and retain all in-domain short rows.
   Different spelling conventions make the final target-domain stage important.
   This is a candidate experiment, not a predicted gain or a reason to train all
   200 hours immediately.

For model replacement, Microsoft's VibeVoice-ASR is a candidate for an initial
accuracy/runtime benchmark. Its official card documents code-switching, hotwords
and an MIT license, and its official vLLM setup requires a plugin. That setup
documents vLLM 0.14.1, whereas the project's saved competition specification lists
0.23.0. Compatibility, offline startup and end-to-end throughput need measurement.
A parameter count or successful loading cannot establish the two-hour budget.
Benchmark both language slices and difficult/long clips before a full-fold run.
[Model card](https://huggingface.co/microsoft/VibeVoice-ASR),
[vLLM instructions](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md).

The claim that only two ASR releases support Javanese is too broad: Meta's MMS
model is another multilingual family to account for, and VibeVoice now has
additional variants. These are not demonstrated upgrades for this task.
MMS and SeamlessM4T-v2-large both list noncommercial model licenses, so they do
not resolve the brief's licensing concern. Confirm eligibility with the
organizers before making either a deployment dependency.
[MMS card](https://huggingface.co/facebook/mms-1b-all),
[Seamless card](https://huggingface.co/facebook/seamless-m4t-v2-large),
[VibeVoice streaming report, September 2](https://arxiv.org/abs/2609.02812).

The official competition pages could not be fetched through the web tool during
this audit. Runtime, deadline, test independence and submission-limit statements
remain attributed to the repository's previously researched specification, not
independently reverified current rules. Organizer clarification about Jember's
license remains unresolved; no messages were sent.

## Reproduction and open questions

```bash
python3 reports/private/problem-brief-audit-20260906/recompute.py
PYTHONPATH=src python3 -m unittest discover -s tests -p test_wer.py
PYTHONPATH=src python3 -m unittest discover -s tests -p test_normalization.py
```

The recomputation completed and all six scorer/normalizer tests passed. Outputs:
`reports/private/problem-brief-audit-20260906/AUDIT.json` and
`RECOVERY_MANIFEST.json`. The ignored bundle includes sensitive source-derived
predictions and stays local. The historical 20,014-case differential comparison
against the official runtime was not rerun in this audit.
All 27 aggregate/per-source score comparisons match the ledger within its
rounding precision. The new files pass whitespace checks; repository-wide
`git diff --check` reports pre-existing CRLF/trailing-whitespace issues in
`experiments/results.csv`, which this audit did not edit.

The user subsequently confirmed <0.20 leaderboard WER as the goal, with roughly
0.15 local as the working target, and supplied the status of the unfinished arms.
See [the target strategy](07-route-to-target.md) for the four requested answers
and new word-recombination, session-SNR and boundary diagnostics. GPU budget and
whether a fluent speaker has reviewed the audio remain open. No improvement to
the public score is claimed.
