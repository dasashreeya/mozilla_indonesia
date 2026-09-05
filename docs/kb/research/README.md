# External research

_Last updated 2026-09-05._

An automated multi-agent research pass failed on an account session limit on 2026-09-05.
The **P0 topics were then researched manually** and are written up in this directory:

| Done | File |
|---|---|
| Official spec, runtime, rules, limits | [official-competition-spec.md](official-competition-spec.md) |
| The proven winning recipe and competitor intelligence | [winning-solutions.md](winning-solutions.md) |
| MDC dataset catalogue and licence risk | [mdc-data-catalog.md](mdc-data-catalog.md) |
| Forum threads, dev/test shift, published benchmarks, HF sweep | [forum-and-benchmarks.md](forum-and-benchmarks.md) |

What remains below is **not on the critical path**. Work it only after the training
pipeline is running on in-domain data.

## Remaining gaps

Ranked by how much the answer would change what we do next.

### Would sharpen the plan

| Topic | The question it answers | Target file |
|---|---|---|
| Runtime and inference budget | Pinned dependency versions; realistic clips/minute per model; how many models fit in 110 minutes; what silently fails offline. | `runtime-and-inference-budget.md` |
| External data landscape | Javanese/Indonesian speech and text beyond MDC, with licenses and the publish-to-MDC obligation. Recommended training mixture. | `external-data-landscape.md` |
| Fine-tuning recipes | Exact hyperparameters for low-resource ASR; LoRA vs full FT; how to fine-tune safely on only 372 in-domain clips. | `finetuning-recipes.md` |
| Javanese linguistics | Which orthographic convention the Jember/dev transcripts likely use; ngoko/krama register; East Javanese dialect. Directly feeds the orthography lever. | `javanese-linguistics.md` |

### Useful context

| Topic | The question it answers | Target file |
|---|---|---|
| Code-switching ASR | Whether forcing a decoder language token helps or hurts on code-switched audio — a direct decision, since the manifest carries `jav`/`ind`/`javind` labels. | `papers-code-switching-asr.md` |
| Low-resource ASR literature | The data-quantity/WER curve; augmentation deltas; checkpoint averaging. | `papers-lowresource-asr.md` |
| Open-source toolkits | Reusable code for training, fast inference, KenLM decoding, ROVER, forced alignment (needed to segment Jember from its TSV timestamps). | `projects-open-source-asr.md` |
| LM fusion and rescoring | How to get N-best from a seq2seq model and whether a shippable LM is worth the ZIP space. | `deep-lm-and-rescoring.md` |
| Hallucination and robustness | Detection and mitigation. Currently low priority: measured insertions are only 146 of 2,278 errors. | `deep-hallucination-and-robustness.md` |

## Standing constraints for any research done here

Carry these into every prompt or search, because they invalidate a lot of otherwise good
advice:

* No network at inference; every asset ships inside the ZIP. Python 3.12, one A100-80GB,
  2-hour full run, 1-minute smoke.
* Each test clip transcribed independently — no test-set adaptation, no pseudo-labelling
  the test set, no cross-sample statistics.
* Open weights only; license must permit competition use. Winning code must be
  MPL-2.0-releasable with supporting software allowing free commercial use.
* External training data must be published to MDC at the end, so permissive licenses
  (Apache-2.0 / CC0 / CC-BY / CC-BY-SA) are strongly preferred over NC.
* Never send competition audio to a hosted API or inference service.

## Standard for what lands here

Every file must carry inline source URLs, mark unverified claims UNVERIFIED, and end each
section with what we should actually do differently. A short accurate note beats a long
plausible one — the failure mode for this directory is confident fabrication, which is
worse than an admitted gap.
