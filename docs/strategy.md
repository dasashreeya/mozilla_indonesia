# Experiment strategy for <= 0.20 WER

This is a target plan, not a claim of achieved performance. Every decision must be backed by MDC-normalized held-out WER.

## Validation that resists overfitting

The in-domain dev set is tiny: two conversations, four speakers, 372 clips. Random clip splits would leak speaker, channel, conversational topic, and adjacent context.

Use two primary folds:

- Fold A: hold out all clips from conversation A; train/adapt on conversation B plus external training data.
- Fold B: hold out all clips from conversation B; train/adapt on conversation A plus external training data.

No speaker may cross a fold. Report corpus WER for the combined out-of-fold predictions and diagnostic WER for `jav`, `ind`, and `javind`. Select on combined WER, not the easiest language group. Preserve one frozen fold report for the last model choice; repeated manual inspection can overfit even without gradient updates.

Acceptance gates:

| Gate | Requirement |
| --- | --- |
| Plumbing | Exact row count/order independence, official normalization, offline execution, no missing transcript |
| Competitive baseline | Combined out-of-fold WER <= 0.22 |
| Submission candidate | Combined out-of-fold WER <= 0.185 and neither conversation fold > 0.21 |
| Ensemble/router | At least 0.005 absolute improvement on combined OOF WER and no fold regression > 0.003 |
| Final package | Official Docker pass, smoke-test pass, <110 minutes measured runtime, reproducible hashes |

The 0.185 local gate leaves roughly 0.015 absolute headroom for dev/private distribution shift. It is a heuristic, not a statistical guarantee.

## Experiment matrix

Run in order and write every result to `experiments/results.csv`.

| ID | Model/data | Decoding or change | Decision question |
| --- | --- | --- | --- |
| E00 | Dummy output | End-to-end local runtime | Does packaging/output/scoring work? |
| E01 | Qwen3-ASR-1.7B zero-shot | Auto language, greedy, batch 24 | Fast primary baseline |
| E02 | Qwen3-ASR-1.7B zero-shot | Force Indonesian on `ind`; auto otherwise | Does safe manifest routing help? |
| E03 | Whisper large-v3 zero-shot | Auto language, sequential <=40 s | Does native `jw` coverage beat Qwen? |
| E04 | Whisper large-v3 | Force `id`/`jw` for monolingual labels, auto for `javind` | Value of language labels |
| E05 | Best base model | Transcript cleanup/normalization ablation | Reduce casing/markup errors without changing words |
| E10 | Best base + Jember | One epoch, low LR | Domain/accent adaptation gain |
| E11 | E10 + CV Javanese | Source-balanced sampling | Javanese acoustic coverage gain |
| E12 | E11 + CV Indonesian | Cap Indonesian read speech at 25–35% of batches | Indonesian gain without spontaneous-domain regression |
| E13 | E12 + dev train fold | Short final adaptation with early stopping | In-domain formatting and code-switch gain |
| E20 | MMS Javanese/Indonesian hypotheses | CTC beam + bilingual KenLM | Low-hallucination complement; only after license confirmation |
| E21 | Best two systems | Per-language router | Is one model consistently better by manifest label? |
| E22 | Best two systems | Word alignment / LM-rescored consensus | Does complementary error reduction exceed cost? |

Smoke-only observations (not model-selection evidence): both models completed local Apple-MPS inference on the first 10 held-out SLR35 subset clips. Whisper forced `jw` scored 0.794521 WER and Whisper auto scored 0.726027. Qwen auto scored 0.657534 and Qwen forced `Indonesian` scored 0.643836. This tiny, older read-speech slice is out of domain, but the result is enough to reject either unadapted model as a presumed winning setup and prioritize Qwen for the first in-domain run. Exact counts are in `experiments/results.csv`.

Do not submit E00 or every parameter trial. Public leaderboard feedback is scarce and can itself be overfit.

## Training recipe

1. Convert all audio to mono 16 kHz PCM for training. Keep the original files and immutable source checksums.
2. Segment Jember from its TSV timestamps with `scripts/prepare_jember.py`. Inspect duration and transcript-length distributions; discard corrupt, empty, and impossible alignments, documenting every rule.
3. Start with the model's published supervised fine-tuning recipe. For Qwen, use `language Indonesian<asr_text>` only on reliable Indonesian-only rows and `language None<asr_text>` on Javanese or mixed rows because Qwen does not advertise a Javanese language token.
4. Use source-balanced batches. A sensible starting mixture is 45% Jember, 20% Common Voice Javanese, 20% Indonesian sources, and 15% in-domain fold training; tune the mixture, not just total epochs.
5. Use speed perturbation around 0.9/1.0/1.1, modest gain/noise/reverb, and codec augmentation that resembles voice notes. Never augment validation. Run an ablation because aggressive noise can erase code-switch cues.
6. Early-stop on normalized OOF WER. Keep learning rate, effective batch size, seed, checkpoint hash, data hashes, and wall-clock/GPU details.
7. After selection, train one final checkpoint using all permitted labeled dev audio and the selected external mixture. The final training run must not change hyperparameters learned from the folds.

The [Qwen fine-tuning guide](https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning) expects JSONL rows such as:

```json
{"audio":"/absolute/path/clip.wav","text":"language None<asr_text>transcript here"}
```

Generate these with `scripts/make_qwen_jsonl.py`; the pinned upstream training program is `vendor/qwen3_asr_sft.py`.

Fine-tuning happens outside the inference container. Package the resulting complete checkpoint under `model/`; PEFT adapters are worthwhile only if the base model and adapter can be loaded offline and reproduced reliably.

## Decoding and text handling

- Score raw model text and MDC-normalized model text as an explicit ablation. Do not broadly lowercase by intuition: the official scorer compares mid-sentence case.
- Analyze substitutions/deletions/insertions and top confusion pairs by language group. A high deletion rate suggests decoding/acoustic issues; spelling substitutions suggest lexicon or LM work; insertions/repetitions suggest generative hallucination controls.
- Build the LM vocabulary only from permitted training text. Normalize it consistently, but retain a protected list of common acronyms and proper names if dev evidence supports that.
- Test `max_new_tokens` against the longest dev reference. Truncation is catastrophic WER, while an excessive limit increases repetition risk and runtime.
- A router may use `test_metadata.csv`'s provided language field and clip duration. It must not use predictions or aggregate statistics from other hidden test samples.

## Runtime budget

The public schema contains 2,118 full-test rows. Benchmark throughput on all 372 dev clips in the official network-disabled Docker image, then extrapolate conservatively. The go/no-go budget is 110 minutes, leaving 10 minutes for model load, filesystem overhead, and variance.

Prefer one batched model if quality is close. A two-model ensemble doubles model assets, load time, and inference; keep it only when the OOF gain clears the acceptance gate. Do not rely on online downloads, custom system packages, writable input data, or logs containing filenames/transcripts.

## Final checklist

- Re-fetch the official runtime and record its exact commit/image digest.
- Verify all dataset/model licenses and required attributions; ask the forum about any ambiguity.
- Freeze data manifests, hashes, seeds, hyperparameters, and source revisions.
- Generate predictions for both dev folds and archive the score report.
- Build with `scripts/build_submission.py`; confirm `main.py` is at ZIP root and all weights are present.
- Run the official local Docker harness with internet disabled and the input directory read-only.
- Confirm exactly two output columns, unique/all filenames, valid CSV quoting, no NaN transcripts, sparse non-sensitive logs, and runtime <110 minutes.
- Use a smoke test before a scored submission. Select final submissions deliberately; the private leaderboard decides winners.
