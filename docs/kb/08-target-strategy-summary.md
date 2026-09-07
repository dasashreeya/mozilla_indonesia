# Summary: path toward <0.20 leaderboard WER

Date: 2026-09-06

**Update after CPU testing:** a three-Qwen contextual selector reaches
0.2631 local WER from 0.2685 in exploratory grouped evaluation. The tests
do not support an assured 0.04-0.09 improvement or a public-score guarantee.
See [the measured CPU results](09-cpu-viability-results.md).

## Goal and current position

- Goal: below **0.20 public leaderboard WER**.
- Current reported public score: **0.3259**.
- Current local control: **0.2697**.
- Working local target: approximately **0.15**. The local-to-public offset
  is uncertain, so this is a planning target, not a guaranteed conversion.

The strongest next bets are audio/transcript alignment, noise robustness,
and adaptation that preserves the target transcription style. No measured
result yet demonstrates the full improvement needed to reach the goal.

## What the new diagnostics tell us

**Noise is a stronger experimental lead than recording volume.** Across
20 held-out Jember sessions, estimated signal-to-noise ratio correlates with
WER at -0.684; RMS volume correlates at only +0.150. These are exploratory
associations, not proof that noise causes the errors. The previously reported
seen-versus-unseen gap does not isolate recording conditions either.

**Session 35 needs a timestamp check.** It has 404 errors over 432 words.
Joining contiguous windows reduces the diagnostic error count to 283, with
some words appearing in the neighboring prediction. Listen to the recording
and check boundaries, then investigate whether training data has similar
problems. Concatenation changes the evaluation and is not a score improvement.

**Existing predictions contain useful alternatives, but are insufficient to
assume 0.15 is reachable through voting alone.** A reference-assisted word
oracle scores 0.1796 across all nine systems, versus the earlier whole-clip
oracle of 0.2314. It uses the correct answers to choose words: this is an
optimistic diagnostic, not a deployable model result or a universal bound.
Whisper contributes useful alternatives despite weak standalone accuracy.

## Answers to the four main questions

### 1. Is session generalization the right target?

Yes, it deserves a controlled experiment, especially for noise sensitivity.
First inspect the alignment issue. Then test moderate noise/channel
augmentation while preserving a substantial clean fraction, short clips,
transcripts, training steps and sampling. Avoid excessive corruption of
already noisy audio.

Compare against a matched control and report both per-session changes and
code-switched WER. Keep the original validation fold frozen. Simple volume
normalization does not improve signal-to-noise ratio.

### 2. How can we improve transcription-convention choices?

Train a contextual selector/corrector with an explicit **KEEP** option.
Use predictions generated out of fold, with entire sessions held out, and
include correct predictions as well as mistakes. Inputs can include nearby
words, alternatives from other models, and acoustic candidate scores.

Measure errors fixed minus errors introduced under the original scorer.
Failed global replacements do not rule out contextual correction. However,
truly random annotation choices cannot be recovered from unavailable clues.
The claim that most errors are convention-only is not yet established by
listening or a reliable error classification.

### 3. Is VibeVoice worth trying, and can it fit two hours?

It merits a small benchmark. Microsoft's training figure lists Javanese at
**0.002467%** of training data, but provides no demonstrated Jember accuracy.
Javanese exposure is real; strong recognition remains unproven.

Memory appears feasible on an A100 80 GB. Runtime and offline compatibility
are unmeasured. Reserving 23 minutes for Qwen and 10 minutes for overhead
leaves **87 minutes for VibeVoice**, or **2.46 seconds per clip** on average.
That is a requirement, not a speed prediction.

Start with 100 representative clips. Measure cold startup, throughput,
accuracy and useful alternatives to Qwen before running the full holdout.

### 4. How should we use OpenSLR despite different spelling conventions?

First reuse the existing OpenSLR-adapted checkpoint, if available, and apply
the exact current Jember/dev adaptation recipe. Compare the resulting model
with the base-Qwen-to-Jember control. Negative transfer before this final
adaptation does not establish that the two-stage approach will fail.

If useful, scale with explicit corpus/transcription-style conditioning and
finish with target-style adaptation. This differs from language tags:
two Javanese corpora can require different written conventions.

External data's value is not limited to new vocabulary. It may improve
recognition of already-seen words through acoustic and contextual variation.

## Recommended order

Keep Qwen as the core model. My first substantial training investment would
be a controlled noise/channel experiment after checking transcript timing.
The broader strategy must improve both language slices and produce new
correct word alternatives before investing heavily in correction.

1. Inspect session 35 timestamps and about 100 stratified error spans with a
   fluent speaker. Check for similar training issues. Repair only verified
   training problems; keep the scored validation references unchanged.
2. Run the capped 100-clip VibeVoice benchmark as an early screening step.
   Continue only if it offers promising accuracy or useful complementary
   words and can plausibly fit the full offline runtime. An oracle gain
   alone is insufficient: a real combination must also improve held-out WER.
3. Test one controlled Qwen noise/channel arm. A reasonable initial design
   is half unmodified audio and half moderately perturbed audio, subject to
   listening checks. This fraction is a proposed starting point, not an
   established optimum. Keep initialization, examples, targets, training
   steps and sampling matched to the control.
4. Test the existing OpenSLR checkpoint followed by Jember/dev adaptation.
   Keep this separate from augmentation initially. If promising, scale the
   external stage with explicit transcription-style conditioning.
5. Confirm the best candidate on another independent split before scaling.
   Train both candidate and control with that split's evaluation audio
   excluded. The current code-switched validation is just one conversation;
   294 clips do not provide 294 independent conversations.
6. Add contextual correction or model combination only when it improves
   actual held-out predictions. Train correction on out-of-fold predictions,
   retain KEEP, and verify end-to-end offline inference before submission.

## Milestones and decisions

Jember contributes 8,524 of 22,453 local reference words, approximately 38%.
Even reducing its WER by 0.09 would move mixed WER only to about **0.236**.
The code-switched slice also needs substantial improvement.

Use local **0.25, 0.22, 0.19, then approximately 0.15** as planning milestones,
not predicted outcomes. For illustration, Jember WER 0.18 and code-switched
WER 0.13 would give approximately 0.149 on this fold. That arithmetic does
not predict a public score.

Prioritize scaling methods that show a repeatable gain across sessions and
an independently held-out conversation. Smaller credible gains can still
be retained for submission. If both training probes remain flat, prioritize
additional accurately transcribed, eligible in-domain conversations from
new speakers/recordings, or a demonstrably stronger model, before another
broad hyperparameter sweep. More data is also an experiment, not a guarantee.

Do not add hypothetical gains together: these methods may fix the same
errors. GPU budget is still needed before committing to paid experiments.
No GPU jobs or competition submissions were launched for this analysis.

## Further detail

- [Full strategy, measurements and research sources](07-route-to-target.md)
- [Audit of earlier assumptions and experiment evidence](06-evidence-audit.md)
