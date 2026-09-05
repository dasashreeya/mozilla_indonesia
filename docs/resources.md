# Researched resources and provenance

Snapshot date: **2026-09-05**. Recheck mutable pages and licenses before the final submission.

## Official competition material

| Resource | Why it matters |
| --- | --- |
| [Competition home](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/1/) | Task, 2026-09-25 deadline, prizes, external-data notice |
| [Problem description](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/) | Dev/test construction, language labels, audio properties, normalization and WER |
| [Submission specification](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/16/) | ZIP layout, paths, Python 3.12, A100, two-hour limit, offline execution |
| [Official rules](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/rules/) | Eligibility, test independence, licenses, data/model use, winner obligations |
| [Runtime repository](https://github.com/drivendataorg/lost-in-transcription-runtime) | Exact Docker image, locked packages, example entrypoint, official scorer |
| [MDC API reference](https://mozilladatacollective.com/api-reference) | Public discovery and authenticated presigned downloads |
| [MDC OpenAPI document](https://mozilladatacollective.com/api/openapi.json) | Machine-readable request/response contract |
| [Competition forum](https://mdc-competitions.discourse.group/c/lost-in-transcription/5) | Ask organizers about license or implementation ambiguity |

The runtime snapshot inspected was commit `05b89385334512c8bff98f1bf35616df3ab0bca9` dated 2026-09-04. It includes `qwen-asr`, Transformers, vLLM, PEFT, KenLM, and pyctcdecode. The live repository remains the source of truth.

## What the MDC API can and cannot do

Unauthenticated API calls can search the catalog and fetch public metadata. `POST /datasets/{id}/download` and `POST /datasets/{id}/sample-download` require `Authorization: Bearer <key>`. Even public/free datasets returned HTTP 401 without a key during this audit.

The API returns one short-lived URL for a complete archive (or a publisher-created sample); it does **not** expose record-level audio selection. `scripts/mdc_fetch.py` therefore downloads and checksum-verifies the complete archive, then deterministically extracts only the requested number of audio files plus metadata. This minimizes extracted data, not transfer size.

Create a key at [Profile → Credentials](https://mozilladatacollective.com/profile/credentials). Dataset terms, access requests, and payment where applicable must be completed in the web UI. Keep the key in `MDC_API_KEY`; never put it in Git, command-line arguments, notebooks, or logs.

An authenticated check on 2026-09-05 returned HTTP 403 for both the development and Jember full/sample endpoints. Per the API contract, this means dataset terms have not been agreed or an access request is not approved; it is not a failed-key response (401). The account owner must complete that legal/access step in the web UI.

## Recommended MDC data

The exact IDs, sizes, and SHA-256 values are pinned in `configs/datasets.json`.

| Priority | Dataset | Use | License / caution |
| --- | --- | --- | --- |
| 1 | [Competition development data](https://mozilladatacollective.com/datasets/cmsxiwqdo000fny07e5fx0yi5) | 372 in-domain clips, 2 conversations, 4 speakers, ~2.5 h | CC-BY-SA-4.0; never select hyperparameters on the same clips used for training |
| 1 | [Jember Javanese Spontaneous Speech](https://mozilladatacollective.com/datasets/cmlgm5a94008kny07nz2intus) | 6,679 timestamped utterances, ~10 h; closest spontaneous/code-mixed training domain | CC-BY-NC-SA-4.0; explicitly highlighted by the competition, but record its terms and attribution |
| 1 | [Common Voice Spontaneous 4.0 – Javanese](https://mozilladatacollective.com/datasets/cmqi294n3004mmf07wvybp3fn) | Extra spontaneous Javanese acoustic diversity | CC0-1.0 |
| 1 | [Common Voice Scripted 26.0 – Indonesian](https://mozilladatacollective.com/datasets/cmqinqwef00xonr07z4vbfovw) | Broad Indonesian coverage; subsample so read speech does not dominate | CC0-1.0 |
| 2 | [Panjebar Semangat Javanese text](https://mozilladatacollective.com/datasets/cmk719jvs02z1nt07tjswt52s) | Lexicon, spelling variants, and LM/rescoring text | CC-BY-SA-4.0; written-domain mismatch |
| 2 | [Podcast Hari Minggoean](https://mozilladatacollective.com/datasets/cmhlvswz200fio007c7szypw8) | Conversational Indonesian | CC-BY-SA-4.0; validate transcript alignment/quality |
| 2 | [Podcast Homostoria](https://mozilladatacollective.com/datasets/cmiepnyu1001jo207hzslvb5m) | Conversational Indonesian | CC-BY-SA-4.0; validate transcript alignment/quality |

Do not blindly combine everything. Inspect sample rate, clipping, transcript conventions, speaker overlap, and label quality; then cap each source's sampling weight. Domain-matched spontaneous speech should dominate late-stage adaptation.

## External starter subset already fetched

[OpenSLR SLR35](https://openslr.org/35/) contains about 185,000 transcribed Javanese utterances under CC BY-SA 4.0. Shard 0, the global transcript table, upstream checksum list, and license were downloaded from OpenSLR; the archive matched its published SHA-256 `a871c8b71ff8fa9d95955447ca0c388e8c6f925aecfce92e1880bda2da113bcb`.

`scripts/prepare_openslr35.py` deterministically extracted 3,000 clips using seed `20260905`: 2,700 training clips from 643 speakers and 300 validation clips from 19 other speakers, with zero overlap and no missing audio. The source is **external** to MDC. The competition rules therefore require a permitted redistribution to be published to MDC at the conclusion of the competition; retain the included license, attribution, source URLs, and hashes. Do not use it for a final submission if that publication obligation cannot be completed.

## Candidate models

| Model | Evidence and role | License / decision |
| --- | --- | --- |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Fast 1.7B generative ASR, official batch inference/fine-tuning, Indonesian is supported. The official language table does **not** list Javanese, so mixed/Javanese performance must be measured. Pinned revision: `7278e1e70fe206f11671096ffdd38061171dd6e5` (~4.70 GB of model shards). | Apache-2.0; primary baseline because `qwen-asr` is already in the runtime |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | Hugging Face metadata declares `id` and `jw`; useful zero-shot comparison, language-forcing sweep, and ensemble source. Pinned revision: `06f233fe06e710322aca913c1bc4249a0d71fce1` (~3.09 GB safetensors). | Apache-2.0; keep only if dev WER justifies its size/time |
| [MMS-1B-all](https://huggingface.co/facebook/mms-1b-all) | CTC model with a Javanese adapter; useful for a low-hallucination hypothesis and KenLM shallow fusion, but stock adapters are monolingual and code-switching is a weakness. | CC-BY-NC-4.0; ask organizers before making it a winning dependency |
| [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) | Native `jav_Latn` and `ind_Latn` coverage is attractive. | Not usable in the current official image: [runtime issue #4](https://github.com/drivendataorg/lost-in-transcription-runtime/issues/4) documents an unsatisfied fairseq2/Torch ABI conflict. Recheck only if the runtime changes. |

Official implementation references:

- [Qwen3-ASR repository and fine-tuning format](https://github.com/QwenLM/Qwen3-ASR)
- [Whisper model card and long-form decoding guidance](https://huggingface.co/openai/whisper-large-v3)
- [MMS model card](https://huggingface.co/facebook/mms-1b-all)

No hosted inference endpoint is part of this plan. Hugging Face is used only to download public weights locally before packaging. Both pinned Qwen and Whisper checkpoints were downloaded and their published LFS hashes verified during this setup.

## Files already fetched

The two schemas under `data/reference/` came from the competition's signed public S3 links:

| File | Rows including header | SHA-256 |
| --- | ---: | --- |
| `submission_format.csv` | 2,119 | `b50e5c4b0ea3de6eb2a10725b96a99fefc92c13887f62fe921f6259be4e472af` |
| `smoke_test_submission_format.csv` | 6 | `ab8b0ffe731c121ab370b90be1519a9c557cd8486056477676ad6056cc72b2a4` |

The placeholder text in these files is not a prediction baseline. The schemas reveal 2,118 full-test clip IDs and 5 smoke-test clip IDs; the audio remains available only inside the execution runtime.
