# Wave 3: instrument fixes and complementary models

Started 2026-09-08. All model-selection measurements use the unchanged
`ds-base` validation split: 509 clips, 22,453 reference words. Public scores
are recorded separately; no additive conversion is treated as a guarantee.

## Verified starting point

The user reports **0.2991 public WER** for
`artifacts/submission-soup-div2.zip`, SHA-256
`b217de4ffbed38fe8f02615a0bfdfdef8f0f9327ec6bba9fbbb3e128d5a8e2c5`.

The actual ZIP was extracted and its `main.py` executed with offline model
loading and a competition-format manifest, on an A100-80GB:

| Measurement | Result |
| --- | ---: |
| Held-out WER | 0.2301697 |
| Javanese WER | 0.2579775 |
| Code-switched WER | 0.2131524 |
| Nonempty outputs | 509 / 509 |
| Process time, including model load | 150.6 s |
| Linear projection to 2,118 clips | 626.5 s |

The time projection is a workload estimate, not an official full-run result.
No material packaging regression was found. Batch size and numerical rounding
account for a small difference from the previously recorded fold WER, but the
public generalization gap remains. The runtime measurement warrants testing
inference ensembles instead of assuming they exceed the two-hour limit.

Reproduce with:

```powershell
modal run scripts/modal_artifact_audit.py::audit
```

The new two-model beam archive was then tested by the same package-level audit:

| Measurement | Result |
| --- | ---: |
| Archive | `submission-ens-ext-lr3-b4.zip` |
| SHA-256 | `d59f701bde34c44c0a3a6c94f4db91e855b25703bd3705375f32f6a0783c68f4` |
| Bytes | 8,158,143,539 |
| Held-out WER | **0.2124438** |
| Javanese WER | **0.2304083** |
| Code-switched WER | **0.2014502** |
| Nonempty outputs | 509 / 509 |
| Process time, including model load | 1,244.3 s |
| Linear projection to 2,118 clips | 5,177.8 s (86.3 min) |

The local downloaded archive has the same byte count and SHA-256. The projected
runtime leaves about 28.7 minutes before the entrypoint's five-minute safety
margin, subject to the test audio workload matching the validation workload.

A coarse probability-weight sweep found 60% `arm-extb` / 40% `arm-lr3` best.
Its exact archive, `submission-ens-ext60-lr40-b4.zip`, is 8,158,143,556 bytes
with SHA-256
`2c73ef239380548985eb8817cb37727243b5c4618b797447a5d7403cba97bd09`.
The package audit produced **0.2116866 WER** (Javanese 0.2327546,
code-switched 0.1987939), 509/509 nonempty outputs, and a 4,817.1-second
(80.3-minute) linear projection to 2,118 clips.

## Final-data candidates

Two no-validation fits completed on all 11,170 labelled no-prompt rows. The
full-parameter `arm-final-lr3-np` fit reached its requested 1,223 steps. The
external-data LoRA fit, `arm-final-extb-np-v2`, used those rows plus 5,156 MDC
East Javanese/Common Voice rows and completed 2,040 steps; its epoch-2 adapter
was merged to `runs/arm-final-extb-np-v2/merged-epoch-2`. Both receipts contain
null validation scores by design.

The recommended final archive applies the held-out-selected 60/40 weighting to
the corresponding full-data models:

| Artifact | Bytes | SHA-256 | Exact ZIP smoke |
| --- | ---: | --- | --- |
| `artifacts/submission-final-ens-ext60-lr40-b4.zip` | 8,160,275,521 | `f357f8ff3ebd1f6fc926ff599dd4a818e7de9b78187df6a37b650594e1065e8f` | 20/20 nonempty, 63.3 s |
| `artifacts/submission-final-extb-b4.zip` | 4,080,140,603 | `0efb852f2d5c5b3b5f81318a5868b14399f03a2bf9e1c13b347195d4206fc0e1` | 20/20 nonempty, 37.5 s |
| `artifacts/submission-final-lr3-b4.zip` | 4,080,140,622 | `1af877919e059838543c976c4f6248d2c671d7241d05e8e90dcd79f715adca99` | 20/20 nonempty |

The ensemble and ext-b smokes produced 0.10943 and 0.14151 WER respectively,
but these are **resubstitution results, not held-out scores**: the final fits saw
these clips. They establish that the downloaded archives load offline, route
language labels, decode with beam 4 and write complete predictions. The honest
model-selection result remains the exact held-out package WER of 0.2116866.
The local archive sizes and hashes match the Modal build receipts. Final merged
weights and experiment outputs are also backed up to the private Hugging Face
repository.

## Bugs found

1. `eval_checkpoint.py` wrote beam width to the outer Qwen model's generation
   config. In the installed `qwen-asr==0.0.6`, outer `generate` delegates to
   `thinker.generate` without forwarding that config. The corrected path sets
   `model.thinker.generation_config.num_beams`. Earlier beam-search conclusions
   were not supported by the code that produced them.
2. A no-validation final fit holds infinity as its internal baseline and best
   WER. Its final receipt used those numbers with `allow_nan=False`, which
   raises after training and merging. Final-fit receipts now store null scores
   and never claim validation improvement. The running older job can have its
   receipt recovered only after its completed trainer state, requested step
   count, final checkpoint and merged model are verified.
3. The packager included Python bytecode caches. Those are now omitted.

The first artifact-audit launch failed at container import because it imported
an unmounted local `scripts` package. It was stopped and replaced by a
self-contained image definition; the successful audit is recorded above.

## Experiments

CTC uses `facebook/w2v-bert-2.0`, MIT, immutable revision
`da985ba0987f70aaeb84a80f2851cfac8c697a7b`. It sees only the training split:
7,753 examples including the established development repeats. Its 75-symbol
vocabulary uses training transcripts plus a fixed ASCII alphabet. It preserves
the official scorer's noninitial case and accents; labels and output frame
lengths are checked for CTC alignment. No validation or test transcript is used
to construct its vocabulary or language model.

The first run (`arm-c1`) used LR 5e-5, effective batch 16, BF16 autocast,
SpecAugment and full encoder fine-tuning. It emitted only blanks at both the
first- and second-epoch evaluations, with loss plateauing near 3.0, before the
Modal run stopped during the third epoch. This branch is retired rather than
spending more GPU time on an invalid candidate. Selection would have used
0.49 * Javanese WER + 0.51 * other WER as a heuristic, not as a prediction of
public WER.

KenLM is a training-text-only three-gram model with 7,473 observed word types.
The builder is pinned to `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`.
Emission caches allow decoding experiments without repeating GPU inference.

The probability ensemble uses Transformers' existing generation engine and
combines next-token probabilities from separate trained models. Each model
has an independent KV cache, reset at every batch. The beam-reorder hook applies
the same permutation to every cache; a regression test verifies this behavior.

```powershell
modal run --detach scripts/modal_ctc.py::run --build
modal run scripts/modal_ctc_decode.py::build
modal run scripts/modal_recovery.py::evaluate --label eval-soup-realbeam4 --model soups/soup-div2 --dataset ds-base --beams 4 --force-language
modal run scripts/modal_recovery.py::evaluate --label eval-ensemble-ext-lr3 --model runs/arm-extb/merged-epoch-2 --ensemble-models runs/arm-lr3/weights-step-849 --dataset ds-base --batch-size 12 --force-language
modal run scripts/modal_recovery.py::evaluate --label eval-ensemble-ext-lr3-beam4 --model runs/arm-extb/merged-epoch-2 --ensemble-models runs/arm-lr3/weights-step-849 --dataset ds-base --batch-size 6 --beams 4 --force-language
```

The first measured comparisons are:

| Model / decoding | Fold WER | Javanese | Code-switched |
| --- | ---: | ---: | ---: |
| Original soup, greedy | 0.229947 | 0.257274 | 0.213224 |
| extb + lr3 probability ensemble, greedy | 0.223890 | 0.242961 | 0.212219 |
| Soup, corrected beam 4 | **0.218813** | 0.243078 | **0.203963** |
| extb + lr3 probability ensemble, beam 4 | **0.212845** | **0.231347** | **0.201522** |
| extb 60% + lr3 40%, beam 4 | **0.211687** | 0.232755 | **0.198794** |
| extb + lr3 + soup, beam 4 | 0.211508 | 0.231816 | 0.199081 |

Correct beam search removes 250 errors from the soup on the same 22,453 words.
The two-model beam ensemble removes 384 errors and improves WER by 0.01710 over
the original soup. Its descriptive paired session-bootstrap interval is
[-0.03462, -0.01314]. The direct evaluator took about 756 seconds for 509 clips;
the exact package audit above is the authoritative runtime measurement.
These are local results, not public scores. The CTC model is not a candidate.

The 60/40 weighting gain over equal weights is 0.00116; its descriptive paired
session-bootstrap interval is [-0.00241, 0.00375]. The three-model gain is also
inside zero and adds about 50% decode cost. The 60/40 two-model package is the
deployment candidate because it has the lowest audited two-model WER without
the third model's runtime and archive-size penalty.

The final-data artifacts above are complete. They must never be scored on this
fold and described as held-out; those clips have been seen.

## Rules and validation

The [live competition rules](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/rules/)
were fetched on 2026-09-08. Public pretrained models and external data with
appropriate rights are allowed. Test clips must be processed independently.
All work here runs our own model code on allocated compute; no competition
audio is sent to a proprietary model API.

Focused tests cover split leakage, vocabulary construction, official WER,
beam routing, independent ensemble caches, and strict JSON for final fits.
The full Windows suite currently has two unrelated portability failures:
`test_modal_utils` assumes POSIX path equality and `test_prepare_openslr35`
compares raw JSON paths without accounting for escaped Windows separators.

Primary model and decoder references:
[HF CTC training guide](https://huggingface.co/blog/fine-tune-w2v2-bert),
[KenLM](https://github.com/kpu/kenlm),
[pyctcdecode](https://github.com/kensho-technologies/pyctcdecode).
