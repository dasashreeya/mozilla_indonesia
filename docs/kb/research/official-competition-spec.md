# Official competition specification

_Researched 2026-09-05 directly from the competition site, the runtime repository, and the
official forum. Every fact below has a source URL. This supersedes the summary in the repo
README wherever they differ._

## Corrections to what this repo previously assumed

| Previously assumed | Actually true | Why it matters |
|---|---|---|
| audio properties unknown | **MP3, 48 kHz, mono, clips 2–40 seconds** | 40 s exceeds Whisper's 30 s window; Whisper needs chunking or long-form decoding, Qwen3-ASR does not |
| `test_metadata.csv` columns unknown | `audio_filename`, `file_duration_seconds`, `language` | no speaker or conversation id at test time — a speaker-adaptive router is impossible |
| submission limit unknown | **3 per 7 days, per individual team member** | ~8–9 submissions remain before the deadline; a 2-person team gets 6 per week |
| Jember was "the closest domain" | **dev and test are both drawn from the Jember corpus** | Jember is not adjacent data, it is *the* in-domain data |
| dialect unspecified | **Jember dialect, Pandhalungan variety, East Java** | OpenSLR-35 is standard/Central Javanese — a dialect mismatch, not just a domain mismatch |
| languages present: Javanese + Indonesian | **Javanese, Madurese, Indonesian and English code-mixing** | Madurese is a fourth language the corpus description names explicitly |

## Task and data

Source: [problem description](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/)

* Transcribe **natural, code-switched Indonesian–Javanese speech** — described on the
  homepage as "real voice-note dialogues between pairs of bilingual speakers, featuring
  heavy code-switching and lexical borrowing".
* Underlying corpus: **Jember Javanese Spontaneous Speech Corpus**, ~10 hours from 208
  session-level MP3 recordings, native speakers in Jember Regency, East Java. Captures
  "code-mixing and code-switching between Javanese, Madurese, Indonesian, and English".
* **Dev and test derive from the same underlying dataset, with no overlap in speakers or
  conversations.** Dev = 372 clips (~2.5 h), 2 conversations, 4 speakers, transcripts given.
  Test = 2,118 clips, withheld.
* Audio: segmented into utterance clips, normalised to **48 kHz mono MP3**, durations
  **2–40 seconds**.
* Register: Jember dialect and **Pandhalungan** variety; transcripts contain "reduplicated
  and truncated word forms".

### Language labels

The `language` column takes exactly three values:

| Value | Meaning |
|---|---|
| `javind` | both Indonesian and Javanese present |
| `jav` | Javanese only |
| `ind` | Indonesian only |

Dev `metadata.tsv` carries `audio_filename`, `speaker`, `transcript`, `language`,
`convo_id`. **Test `test_metadata.csv` carries only `audio_filename`,
`file_duration_seconds`, `language`** — so routing at inference time may use the language
label and the clip's own duration, and nothing else.

## Metric

Source: [problem description](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/)
and [`runtime/score.py`](https://github.com/drivendataorg/lost-in-transcription-runtime).

`WER = (S + D + I) / N`, corpus-level, computed with `jiwer` after normalisation.

Normalisation, in order: strip `~`; remove `[bracketed]` meta-linguistic tags; remove
unintelligibility markers `(?)`/`(???)`; unwrap ordinary parentheses; replace `#x27;` with
an apostrophe; strip `¿¡";:`; **lowercase sentence-initial letters only** (and not when the
next letter is also uppercase, protecting acronyms); em dash → comma; drop commas; drop
`!?`; drop periods except inside `...`; collapse multiple spaces.

> The official page states it directly: **"Casing is compared as written"** for
> non-sentence-initial words. This confirms the locally measured 11.8-point cost of
> blanket lowercasing — see [../02-measured-findings.md](../02-measured-findings.md).

The repo's `src/mdc_asr/normalization.py` is verified behaviourally identical to
`runtime/score.py` over 20,014 test strings.

## Submission contract

Source: [code submission format](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/16/)

```
submission.zip
├── main.py          # MUST be at the archive root, not nested
├── my_module/       # optional
└── model weights    # all assets, no network at runtime
```

Execution layout, working directory `/code_execution`:

```
/code_execution/
├── data/
│   ├── clips/              # read-only test audio
│   └── test_metadata.csv
├── src/                    # your unzipped submission
└── submission/
    └── submission.csv      # your output
```

Output columns: `audio_filename`, `transcript`. Standard CSV quoting; use the `csv` module
or `pandas.to_csv`.

## Runtime

Sources: [submission spec](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/16/),
[runtime repo](https://github.com/drivendataorg/lost-in-transcription-runtime),
and the local lockfile at `runtime/runtime/uv.lock`.

| | |
|---|---|
| Python | 3.12 only |
| GPU | 1× NVIDIA A100 80 GB |
| CPU / RAM | 24 vCPU / 220 GB |
| Network | **none** |
| Filesystem | `/code_execution/data/` read-only; `./submission/` writable |
| Root | not available |
| Full run limit | **2 hours** |
| Smoke limit | **1 minute** |
| Logs | 500 lines max, 300 chars per line |

Base image `nvidia/cuda:13.0.3-runtime-ubuntu22.04`. Resolved dependency versions:

| Package | Version |
|---|---|
| torch / torchaudio | 2.11.0+cu130 |
| transformers | 4.57.6 (constraint: `<5`) |
| vllm | 0.23.0 |
| qwen-asr | 0.0.6 |
| peft | 0.20.0 |
| numpy | 2.3.5 |
| librosa | 1.0.0 |
| datasets | 5.0.1 |
| torchcodec | 0.16.0 |
| soundfile | 0.14.0 |
| kenlm | 0.3.0 |
| pyctcdecode | 0.5.0 |
| accelerate | 1.12.0 |

Also present: tensorflow[and-cuda], keras, sentencepiece, sacremoses, sacrebleu,
scikit-learn, scipy, polars, pandas, mutagen, loguru, einops, tqdm.

**`flash-attn` is not in the lock.** Any code path requiring FlashAttention 2 must fall
back to SDPA. `peft` **is** present, so a LoRA adapter can be loaded at inference — though
merging the adapter into the base weights before packaging is simpler and faster.

Local testing uses `just pull`, `just pack-submission`, `just run`, `just score`.
Internet is blocked by default; `just block_internet=false run` overrides it for
debugging only.

## Limits, rules and prizes

Sources: [rules](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/rules/),
[homepage](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/1/),
[forum thread 23](https://mdc-competitions.discourse.group/t/submission-rule-clarification-for-teams/23).

* **Deadline: 2026-09-25, 00:00 UTC.**
* Prizes: $4,000 first, $2,000 second. Plus a cross-track bonus for the lowest average WER
  across all three tracks, requiring the same team members in each. Tracks:
  Indonesian–Javanese, Spanish–Nahuatl (active), Spanish–English (coming).
* **Submission limit: 3 per 7 days, per individual team member.** Confirmed by admin
  Mdaley: "The submission limit applies for individual team members. Each individual gets
  a limit of 3 submissions per 7 days." A 2-person team therefore has 6 per week, not 3.
* Smoke tests, cancelled jobs and failed jobs do **not** count against the limit.
* Winners are determined on the **private** leaderboard.
* **Test-set independence:** "Each test data sample should be processed independently
  during inference without the use of information from other cases in the test set."
  This forbids pseudo-labelling the test set and any cross-sample statistics.
* **External data with rights:** permitted if you hold all rights, and **any external
  training data must be published to Mozilla Data Collective when the competition ends.**
* Pre-trained models permitted if weights are freely and publicly available, the license
  permits competition use, and no proprietary hosted API is involved.
* Never upload competition data to any third-party hosted model, API, or inference service.
* Winners license under **MPL 2.0** and must deliver source, documentation, weights,
  configuration and dependencies sufficient to reproduce the result.
* Eligibility: 18+, permitted territories, one account per participant.

## Open items

* Exact **total** submission cap (as opposed to the 3-per-7-days rate) is not published;
  the rules say the number "will be displayed on the Competition Website". Check the
  submissions page.
* Whether an NC-licensed training corpus is compatible with the MPL-2.0 winner obligation
  — see [mdc-data-catalog.md](mdc-data-catalog.md).
