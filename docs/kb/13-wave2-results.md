# Wave 2 results — the depth path, executed

_2026-09-07/08. What the programme in
[11-audit-and-methodology-v2](11-audit-and-methodology-v2.md) produced when it
was actually run. Every number is on the same 509-clip fold and the same 22,453
reference words as every measurement in
[experiments/results.csv](../../experiments/results.csv), so it is directly
comparable to Wave 1._

## The result

| | Fold | Javanese | Code-switched | Test-weighted | Forecast public |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base Qwen3-ASR-1.7B, forced tags | 0.4048 | — | — | — | — |
| Previous best (`qwen-jav-holdout-v1`) | 0.2685 | 0.3368 | 0.2266 | 0.2806 | 0.3326 |
| Best single arm (`arm-extb` epoch-2) | 0.2344 | 0.2616 | 0.2177 | 0.2392 | 0.2912 |
| **Soup of four (`soup-div2`)** | **0.2299** | **0.2573** | **0.2132** | **0.2348** | **0.2868** |
| Public leader | | | | | 0.2196 |

**Javanese fell 0.3368 → 0.2573, a 24% relative reduction**, on the half of the
task that is ~49% of the test set and carried 59% of the remaining error. Ten
Wave 1 arms had all landed between 0.3374 and 0.3555.

Shipped as `artifacts/submission-soup-div2.zip` — 4,078,016,826 bytes, SHA-256
`b217de4ffbed38fe8f02615a0bfdfdef8f0f9327ec6bba9fbbb3e128d5a8e2c5`.

## The three conclusions this overturns

### "The published winning recipe does not transfer"

It was never run. Every LoRA launch this project ever made died before step 1:

```
enable_input_require_grads() -> get_input_embeddings()
NotImplementedError: not auto-handled for Qwen3ASRForConditionalGeneration
```

transformers 4.57.6 does not implement that method for this architecture. The
conclusion that stopped this line of work was measuring a crash. Hooking the
embeddings directly (`enable_input_grads` in `train_recovery.py`) fixes it, and
on its first real execution the recipe produced the best Javanese WER in the
project's history within one epoch.

### "Training converges in one epoch"

All four arms peaked at **epoch 2** and regressed at epoch 3:

| Arm | epoch-1 | epoch-2 | epoch-3 |
| --- | ---: | ---: | ---: |
| `arm-lora` | 0.2479 | **0.2403** | 0.2453 |
| `arm-prompt` | 0.2462 | **0.2422** | 0.2483 |
| `arm-lora-long` | 0.2496 | **0.2471** | 0.2483 |
| `arm-extb` | 0.2421 | **0.2344** | 0.2362 |

The recorded "epoch 2 bought -0.000072" came from the code-switched-only
development fold under a schedule cut short by `stale >= 2` early stopping.
Neither 1 nor 3 is the answer; 2 is.

### "The East Javanese data is bounded at ~0.007"

`arm-extb` — the same recipe plus 12.06 h of Lumajang/Bojonegoro/Common Voice
under style tags — beat the identical recipe without it on **every** measure at
matched epochs, by 0.0067 test-weighted. That is outside the ±0.002 noise floor.
The corpora had never been downloaded.

The style tags mattered: the TTS corpora carry `Javtts`, Common Voice `Javcv`,
Jember `Jav`, and decoding always forces `Jav`. Read speech transferred
negatively here once before (OpenSLR-35) with no such separation.

## Averaging pays, and only because the pool was chosen for diversity

| | Fold | Javanese | Code-switched | Test-weighted |
| --- | ---: | ---: | ---: | ---: |
| `soup-div2` | 0.2299 | 0.2573 | 0.2132 | 0.2348 |
| Best member | 0.2344 | 0.2616 | 0.2177 | 0.2392 |
| **Gain** | **-0.0045** | **-0.0043** | **-0.0045** | **-0.0044** |

Ranking checkpoints by fold WER puts the top four *all inside `arm-extb`*,
within a few hundred steps of each other — nearly identical weights making
nearly identical mistakes. The pool that was actually used takes one checkpoint
per arm across three recipes, including `arm-lr3`, which ranks **last** on the
fold and **first** on Javanese (0.2577 against 0.2616) while being 0.021 worse
on code-switched. Full-parameter updates and low-rank adapters specialise in
opposite directions here; that is what the average exploits.

## Two levers that did not pay

**The context prompt does not survive fine-tuning.** Stacked on the forced
language tag it is worth **-0.0094 zero-shot** (0.4048 → 0.3954, a combination
the knowledge base had never measured — it had each lever alone at ~0.403). After
one epoch of training `arm-prompt` and `arm-lora` are tied at 0.2462/0.2479, and
at epoch 2 the prompt arm is *behind*. Same shape as the language-tag finding in
[05](05-architecture-and-arms.md): real before training, gone after it.

**Six epochs is worse than three.** `arm-lora-long` stretches the same cosine
over 6 epochs and is behind `arm-lora` at every shared epoch, because its
learning rate is still high where the shorter schedule has begun decaying.

## Seven defects found while executing

Each cost real time or would have shipped a broken artifact.

| # | Defect | Consequence |
| ---: | --- | --- |
| 1 | `get_input_embeddings` unimplemented for Qwen3ASR | every LoRA arm ever launched died at startup |
| 2 | Generation config sanitised on the trainer's model but not the merge's fresh base | every LoRA arm crashed at 99%, in the final merge |
| 3 | Failed merges left `merged-*` directories holding **only `config.json`** | passes an "is this a model?" check; fails opaquely at load |
| 4 | `merge_adapters` matched bare labels against prefixed directory names | merged nothing, reported success — a silent no-op |
| 5 | Merged models had no tokenizer assets | `AutoProcessor` dies with `IndexError` several steps downstream |
| 6 | `model_soup` copied a fixed file list | predates `chat_template.json` → `.jinja`; same opaque failure |
| 7 | `upload_items` committed once per file | HTTP 429, 128-commits-per-hour, one-hour cooldown |

Defects 3 and 5 are the dangerous class: a directory with a config and weights
looks valid to every guard, and the gap surfaces only at load time. What caught
them was insisting a checkpoint contain **both** `config.json` and weights
before anything consumes it, and refusing to average an empty pool.

## Honest position

Forecast **0.2868** against the leader's **0.2196**. This is a large, verified
improvement on the recorded 0.32590 and it is not first place.

The gap is ~0.07 and nothing left in the depth path closes it. What remains is
[11 §3.3](11-audit-and-methodology-v2.md): a model class whose errors are
uncorrelated with Qwen's — `facebook/w2v-bert-2.0` (MIT) with a CTC head and a
KenLM over the Jember and development transcripts, decoded through the
`pyctcdecode` the official runtime already ships. That attacks the 46.7% of
Javanese errors that are the right word under the wrong transcription
convention, which no amount of Qwen fine-tuning has moved. Budget 3-5 days.

## Reproduction

```bash
make w2-prep PREP_SHA=<sha>          # ds-base, ds-prompt, ds-full-np, ext-b
make w2-train                        # the five arms
modal run scripts/modal_recovery.py::soup_pick \
    --checkpoints "arm-extb/epoch-2,arm-lora/epoch-2,arm-prompt/epoch-2,arm-lr3/step-849" \
    --label soup-div2
modal run scripts/modal_recovery.py::package \
    --model soups/soup-div2 --label submission-soup-div2 --language-routing
```

Figures in [experiments/plots/](../../experiments/plots/); `curves.png` shows the
result that set the final fit's step count — the `ds-base` arms bottom out near
step 900 and climb, while `arm-extb` keeps improving to ~step 1600.
