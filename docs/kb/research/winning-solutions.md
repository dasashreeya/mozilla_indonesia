# Winning solutions from the closest comparable competition

_Researched 2026-09-05. Primary sources: DrivenData's winners blog and the public winners
code repository. This is the highest-value research note in the knowledge base._

## Why this competition is the right reference

DrivenData ran **"On Top of Pasketti: Children's Speech Recognition Challenge"**
(completed April 2026, 828 participants, $70,000 pool, 1,542 Word Track submissions). It is
the closest available analogue to our task:

* same platform, same code-execution submission model, same WER metric;
* a low-resource / hard-acoustics target population;
* a large train/test domain gap;
* and it concluded **five months before our deadline**, so its winners' stack reflects the
  current state of the art on this exact infrastructure.

Sources: [winners blog](https://drivendata.co/blog/on-top-of-pasketti-winners),
[winners code (MIT)](https://github.com/drivendataorg/on-top-of-pasketti).

## The headline result

**All three Word Track winners used a fine-tuned Qwen3-ASR-1.7B.** Nothing else placed.

| Place | Solver | WER | Approach |
|---|---|---|---|
| 1st | ktrw (Kotaro Watanabe) | 0.193693 | Qwen3-ASR-1.7B + LoRA, trained on several WER-filtered dataset variants, then **adapter averaging + model averaging** |
| 2nd | legend (Sunday) | 0.195281 | **18** Qwen3-ASR-1.7B LoRA fine-tunes combined by weight averaging ("model soup") |
| 3rd | chuxiliyixiaosa (Tang Yongqwei) | 0.198434 | Qwen3-ASR-1.7B fine-tune; focus on reconciling leaderboard/validation discrepancies |
| Noisy bonus | shiqi_47 (Shiqi Li) | — | Qwen3-ASR-1.7B **full-parameter** FT on one 24 GB GPU, 329k samples, bf16 + FlashAttention 2, 2 epochs, best at step 16,000, no augmentation, **vLLM greedy inference** |
| Noisy bonus | **mitchelld12345 (Mitchell DeHaven)** | 0.2248 | Qwen3-ASR fine-tune + **KL distillation** + **TTS-generated synthetic training data** (via Qwen3-TTS) |

Winners beat Whisper and KidWhisper — a model trained specifically on children's speech —
by **more than 50%**. Compute was modest: "1–2 days on 1–4 GPUs at an estimated $50–$250".

> **Mitchell DeHaven is #3 on our leaderboard at 0.2290.** His signature technique here was
> TTS-generated synthetic training data plus KL distillation. Expect him to be applying the
> same playbook to Javanese/Indonesian. See [mdc-data-catalog.md](mdc-data-catalog.md) — MDC
> hosts several gigabytes of Javanese TTS corpora, which is exactly the raw material for it.

## What this settles for us

**Keep Qwen3-ASR-1.7B.** The concern that Qwen "does not officially list Javanese" is
answered empirically: three independent winners took Qwen3-ASR-1.7B into a domain it was
never trained for and beat purpose-built models by 50%. The audio encoder plus LLM decoder
architecture adapts. Our 0.47555 is a *data* failure, not a *model* failure.

This also retires the "switch to Whisper" idea. Whisper appeared only in the Phonetic
Track's 2nd-place CTC ensemble, never as a Word Track winner. And Whisper's 30-second
window is a liability here: our clips run to 40 seconds.

## The exact 1st-place hyperparameters

From `word-track/1st Place/csrc_qwen/configs/qwen_exp022.yaml` in the winners repo:

```yaml
model_name_or_path: "qwen/Qwen3-ASR-1.7B"
train_audio_encoder: true          # the encoder IS trained
train_projector: false
prompt: "Transcribe only the children's voice."   # task-specific context prompt
max_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4     # effective batch 16
gradient_checkpointing: true
precision: "bf16"
save_steps: 5000
save_total_limit: 10
load_best_model_at_end: false      # checkpoints selected manually by validation WER
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
augment:
  spec_augment:
    freq_masks: 1
    freq_width: 15
    time_masks: 1
    time_width: 0.05
    prob: {train: 0.5, talkbank: 0.2}   # higher on in-domain, lower on out-of-domain
  speed_perturb: null                    # deliberately NOT used
optim:
  lr: 1e-4
  weight_decay: 0.01
  optim_name: "adamw_torch"
scheduler:
  name: "linear"
  warmup_ratio: 0.1
data:
  max_duration_sec: 30.0
```

Compare with our current run: full fine-tune, lr 2e-5, effective batch 32, no LoRA, no
SpecAugment, no prompt. Every one of those is a deviation from the proven recipe.

### Four techniques we are not using

**1. The context prompt.** Qwen3-ASR accepts a free-text prompt that conditions decoding.
The winner used `"Transcribe only the children's voice."` This is a zero-cost lever. Ours
should describe the actual task, e.g. *"Transcribe this Javanese-Indonesian voice note.
Speakers code-switch between Javanese and Indonesian."* It can also be varied **per clip
using the `language` label** — `jav` / `ind` / `javind` — which is legal, since it uses only
that clip's own metadata. Test three or four phrasings on the dev folds.

**2. WER-threshold data filtering.** The winner fine-tuned a *second* model (NVIDIA
Parakeet-CTC-1.1B), ran it over the training set, and dropped every training sample whose
WER against the reference exceeded a threshold (0.7 in exp022; exp026 used an `or` of
WER ≤ 0.9 and CER ≤ 0.5). This removes mis-transcribed and misaligned training data.
Directly applicable to Jember, whose ~10 h of spontaneous speech is segmented from
session-level recordings via TSV timestamps and will contain alignment errors.

**3. Checkpoint and model averaging.** Two stages: average several LoRA *adapters*
(`average_adapters.py`), then average the merged *models* (`average_models.py`). Select
"approximately 3 appropriate checkpoints" by validation WER. Second place scaled this to
18 fine-tunes souped together. **This costs nothing at inference** — the averaged model is
one model, so it does not touch the 2-hour budget. It is the cheapest ensembling available
and it is what separated the top three.

**4. vLLM greedy inference.** The bonus winner used the vLLM backend. `vllm==0.23.0` is in
our runtime lock, and `qwen-asr` supports a vLLM backend with `gpu_memory_utilization`,
`max_inference_batch_size` and `max_new_tokens`. Our `submission_src/main.py` uses the
slower transformers backend.

## The general pattern across both tracks

| Component | Evidence | Applies to us |
|---|---|---|
| One strong pretrained base, fine-tuned | all 6 winners | yes — Qwen3-ASR-1.7B |
| LoRA rather than full FT | 1st, 2nd, 3rd Word Track | yes; full FT also won a bonus, so both work |
| Weight/checkpoint averaging | 1st, 2nd Word Track | **yes — highest ROI, zero inference cost** |
| Training-data quality filtering by a second model | 1st Word Track | yes — Jember segmentation will be noisy |
| Task-specific context prompt | 1st Word Track | yes — free |
| SpecAugment, mild | 1st Word Track | yes |
| Speed perturbation | explicitly disabled by 1st | skip |
| Synthetic TTS training data | mitchelld12345 | yes — MDC has GBs of Javanese TTS corpora |
| KL distillation | mitchelld12345 | maybe, if time |
| Multi-model ROVER / reranker ensembles | Phonetic Track winners | **no** — 11–13 model ensembles will not fit 2 hours for 2,118 clips |
| Data augmentation generally | blog: "data augmentation drove the final edge" | yes |

The blog's own summary of what mattered: "Data augmentation drove the final edge",
and winners "relied almost entirely on competition data rather than external sources".

That last point is worth sitting with. In Pasketti, the in-competition data was the
winning data. Our analogue is **Jember**, which we have not yet downloaded.

## What to copy, in order

1. Fine-tune Qwen3-ASR-1.7B on **Jember**, using the 1st-place LoRA config verbatim as the
   starting point (r=16, α=32, lr 1e-4, 3 epochs, effective batch 16, SpecAugment, no
   speed perturb, `train_audio_encoder: true`).
2. Add a task-appropriate **context prompt**, tuned on the dev folds.
3. **Average 3+ checkpoints** by validation WER. Free WER, no inference cost.
4. **Filter training data** by a reference model's WER/CER before the final run.
5. Consider **synthetic data from the MDC Javanese TTS corpora**, which is what the
   competitor currently ahead of us used to win a bonus in this very competition.
6. Switch inference to the **vLLM backend** if throughput becomes a constraint.

Do not attempt an 11-model ROVER ensemble. It won the Phonetic Track under different
constraints and will not fit our 2-hour budget.
