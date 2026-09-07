# Route toward <0.20 leaderboard WER

2026-09-06. User-confirmed success criterion: <0.20 leaderboard WER, with roughly
0.15 on the present mixed holdout as a working target. The local/public offset
remains uncertain for the reasons in [the evidence audit](06-evidence-audit.md).

No method in the evidence currently demonstrates the required approximately 45%
error reduction. The strongest next program is to improve the accuracy and
diversity of the underlying hypotheses, investigate timing/noise sensitivity,
and explicitly learn target transcription conventions. Small variations of the
current voting pool should not be the main investment.

## New measurements from existing predictions

### Word recombination, beyond the whole-clip oracle

The earlier oracle (0.231417) chooses one complete transcript per clip. A new CPU
diagnostic aligns each hypothesis to the existing best Qwen pivot without using
references. Each pivot position offers observed token/deletion alternatives;
each gap offers the observed insertion sequences. Dynamic programming then uses
the reference to choose the lowest-error path through that fixed network.

| Reference-assisted diagnostic | Mixed WER | Jember | Code-switched |
| --- | ---: | ---: | ---: |
| Single pivot: actual saved predictions | 0.268472 | 0.336814 | 0.226649 |
| Three Qwen systems: word oracle | 0.226696 | 0.298451 | 0.182784 |
| Those three + base Whisper: word oracle | 0.188215 | 0.268771 | 0.138919 |
| All eight Qwen systems: word oracle | 0.214225 | 0.280971 | 0.173379 |
| All nine systems: word oracle | **0.179620** | **0.254693** | **0.133678** |

The three Qwen systems are `w0-qwen-ft-greedy`, `arm-combo`, `arm-kitchen`.
The fourth is `w0-whisper-jw`.

Only the first row is a deployable model score. The other rows use the answers
to select words, allow combinations not emitted by any model, and are optimistic
diagnostics. They do not measure a trained corrector or ROVER improvement. Each
source hypothesis is a valid network path; the algorithm was checked against
exhaustive enumeration on 250 small random networks plus insertion/empty-pivot
cases, and single-system distances reproduce the saved score.

Within this particular alignment, even ideal recombination of all nine systems
remains above the working target. Other alignments or a generative corrector
can produce paths this network does not contain, so this is not a universal
impossibility bound. It is strong practical evidence to prioritize systems that
provide new correct alternatives. Whisper contributes more new oracle coverage
than adding several similar Qwen arms, despite its poor standalone accuracy.

### Session difficulty: noise proxy versus amplitude

Using control predictions and the existing raw-corpus session acoustic summaries,
Spearman correlations across the 20 held-out Jember sessions are:

| Session property | Correlation with session WER |
| --- | ---: |
| Median estimated SNR | **-0.684** |
| Median RMS level | +0.150 |
| Reference OOV rate against actual training vocabulary | +0.130 |
| Median effective bandwidth | +0.511 |
| Median silence ratio | -0.539 |

Leaving out any one session keeps the SNR association between -0.797 and -0.635.
This makes noise sensitivity a more useful experimental lead than the original
RMS-variance argument. It still does not establish causation: SNR is a heuristic,
the summaries describe original segments rather than freshly measured prepared
windows, predictors correlate, session sizes vary, and only 20 sessions exist.
Topic, speaking style, transcription quality and recording condition can covary.

### A timing lead that a word-convention taxonomy misses

Session 35 has 404 errors over 432 reference words (0.935185 WER). In its first
two windows, several words at the end of the first reference appear near the
start of the second prediction. Scoring concatenated *contiguous* windows lowers
its diagnostic error count to 283; concatenating the entire session, including
unscored gaps, lowers it to 248. Across all sessions, contiguous concatenation
reduces errors from 2,897 to 2,768, with 121 of those 129 removed errors in session
35.

Concatenation forgives boundary mistakes and changes the evaluation problem; it
is not an allowed way to claim a better competition score. The pattern warrants
listening to the original recording and verifying timestamps. It could indicate
misalignment, model boundary effects or a mixture; text comparison alone cannot
decide. Repairing validation references would require a separately versioned
fold and reevaluation of controls, and would not by itself improve leaderboard
performance. The practical training question is whether similar timestamp
problems affect the training sessions. Realigning verified faulty segments is a
different intervention from dropping rows using the previous heuristic flags.

## Answers to the four questions

### 1. Session generalization

Target robustness to genuinely new speakers and recording conditions, but do
not treat the training-audio/held-out gap as nine points of recoverable channel
loss. The new SNR association supports one controlled multi-condition training
experiment. The established ASR literature supports simulated RIR/noise training
for environmental robustness, including transfer to close-talking conditions;
this is evidence for the method family, not a Qwen/Jember effect-size forecast.
[Ko et al., ICASSP 2017](https://danielpovey.com/files/2017_icassp_reverberation.pdf).

Proposed first arm: preserve a substantial clean fraction, all in-domain short
segments, transcript targets, optimizer steps and source sampling. Apply a
moderate noise/channel transform only to training audio, calibrated to the
observed conditions. Avoid heavily corrupting clips already estimated to be
noisy. Validate transformation timing and intelligibility before training.
Run a matched control if the environment or preparation has changed, and report
paired per-session deltas plus code-switched WER. A subsequent ablation can
separate noise from EQ/codec/RIR effects if the arm helps.

Simple level normalization does not change SNR. The weak RMS/WER association
does not justify making gain normalization the main experiment. Explicit
speaker/session invariance losses are a later option: forcing away every session
signal can also suppress useful accent or convention cues, and is more complex
than testing the suspected nuisance variation directly.

### 2. Convention selection

Use conditional correction with an explicit KEEP option. Train from session-
disjoint out-of-fold ASR predictions, retaining both correct occurrences and
errors. The model should select among plausible alternatives using nearby words,
agreement/disagreement across models, and acoustic candidate scores where useful.
Evaluate net edits corrected minus edits introduced, including insertions and
deletions, on untouched groups under the original scorer. If choices are truly
random conditional on all observable context/audio, no corrector can infer them.

Multi-system hypotheses-to-transcription learning has been studied specifically
for code-switched ASR. That work supports a distinct conditional objective; it
does not supply a Javanese gain prediction or authorize using a hosted model on
competition audio. A compact local selector/corrector is the deployment candidate.
[Chen et al.](https://arxiv.org/abs/2310.13013).

The failed global replacements do not test conditional correction. Similarly,
near-equal greedy/beam-4 top-one WER says nothing decisive about beam posterior
entropy or word alternatives supplied by other models. The new word oracle
quantifies some of that alternative coverage. However, coverage is insufficient
to assume that correction of existing hypotheses alone reaches 0.15.

### 3. VibeVoice: actual Javanese evidence and runtime

Javanese is explicitly listed in Microsoft's model card. Appendix A of the
technical report and the model card's training-distribution figure report
**0.002467% Javanese**, versus **0.575294% Indonesian**. The figure was inspected
visually. The report's benchmark table contains no Javanese evaluation, and its
limitations section acknowledges that English/Chinese-focused SFT can degrade
low-resource language performance. Thus Javanese pretraining exposure is real;
strong Jember recognition is unestablished.
[Official model card](https://huggingface.co/microsoft/VibeVoice-ASR),
[technical report, Appendix A and sections 3–4](https://arxiv.org/pdf/2601.18184v1).

An 8–9B BF16 model needs roughly 16–18 GB just for weights. Memory on an A100
80 GB therefore looks feasible at conservative context/batch settings; actual
memory includes the encoders, cache, workspaces and activations. End-to-end speed
and compatibility require a benchmark. The official vLLM plugin documentation
uses version 0.14.1; the saved competition specification lists 0.23.0. The full
offline path must include plugin registration, local assets, structured-output
parsing, audio decoding, cold startup and failure handling.
[Microsoft's vLLM instructions](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md).

Budget arithmetic, using the user's 23-minute Qwen measurement and a proposed
10-minute startup/IO/rescoring reserve:

```
7,200 s total - 1,380 s Qwen - 600 s reserve = 5,220 s for VibeVoice
5,220 s / 2,118 clips = 2.46 s per clip, amortized over batches
```

This is an acceptance threshold, not a speed estimate. Begin with 100 clips
stratified across language, duration and session, including all 20 Jember
sessions. If compatible and potentially useful, run the full 509-clip fold.
An 18-minute full-fold decode would extrapolate to approximately 75 minutes for
2,118 similarly distributed clips, leaving room for Qwen and the reserve. The
hidden duration distribution, startup and slow outliers still need a margin.
Judge both standalone accuracy and the new alternatives it contributes to Qwen;
another weak but diverse system can be useful without being the primary decoder.

### 4. OpenSLR without imposing the wrong convention

The cheapest informative probe is to reuse the existing OpenSLR-adapted
checkpoint, if still present, and apply the exact current in-domain Jember/dev
adaptation recipe. Compare the final adapted model against the Qwen-base-to-
Jember control. The recorded negative transfer measured an OpenSLR fine-tune
without the subsequent target-domain adaptation. A failure before that final
stage does not settle this experiment. No new 200-hour pass is needed for the
initial probe.

For scaling, explicitly condition on **transcription style/corpus**, preserving
source labels, then finish with target-style in-domain adaptation. A training
example from OpenSLR and one from Jember can ask for different written conventions
while sharing acoustic learning. Use bounded external sampling with retained
in-domain exposure; infer only in the target style. This differs from the tested
language tags: two corpora can both be Javanese while writing it differently.

Style-conditioned ASR has published precedent using small rich-transcription
data and much larger common-transcription data. Applying that idea to Javanese
orthography is a proposed transfer of the method, not a reported result.
[Tanaka et al., Interspeech 2021](https://arxiv.org/abs/2107.05382).

If preserving decoder conventions proves difficult, an auxiliary OpenSLR CTC
head with a shared encoder is a more substantial alternative: external labels
train that head/encoder while the main decoder learns target-style labels. Its
training complexity makes it a second-line option, and its effectiveness for
this Qwen architecture has not been tested here. The external-data benefit is
not bounded by novel-word coverage: recognition of already-seen words can also
improve through acoustic and contextual variation.

## Order of investment

1. Inspect session 35's boundaries and a stratified sample of errors; establish
   whether the training corpus has corresponding alignment problems. Keep the
   original validation set frozen for comparisons until a versioned correction
   is independently justified.
2. Run the bounded VibeVoice compatibility/accuracy/runtime probe; do not start a
   large fine-tune just because a language is listed as supported.
3. Test one controlled noise/channel arm and one reuse-OpenSLR-then-Jember arm,
   sequentially and within the user-provided budget. Archive full predictions and
   initialization/data hashes. Do not restart the previous broad arm sweep.
4. Scale whichever approach supplies new correct Javanese alternatives; evaluate
   the final candidate on more independent sessions/conversations where possible.
5. Train conditional correction/combination using honest out-of-fold hypotheses,
   then finalize a model on all eligible labeled data after selecting the recipe.
   Keep final-fit training scores separate from validation estimates.

No additive gain forecast is justified: these interventions can fix overlapping
errors. A small statistically credible gain can still be useful, but a route
to approximately 0.15 needs a substantial change in hypothesis quality. The
organizer's licensing clarification and the user's GPU budget remain unresolved.
No GPU work or submission was launched in this turn.

## Reproduction

```bash
python3 reports/private/problem-brief-audit-20260906/lattice_oracle.py
python3 reports/private/problem-brief-audit-20260906/session_diagnostics.py
```

Results: `LATTICE_ORACLE.json`, `SESSION_ASSOCIATIONS.json`, and
`BOUNDARY_DIAGNOSTIC.json` in the same ignored directory. These files contain
local diagnostic evidence, not improved competition submissions.
