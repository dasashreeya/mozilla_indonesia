# Forum intelligence and published Javanese ASR benchmarks

_Researched 2026-09-05. Forum read via the Discourse JSON API; benchmarks from published
literature._

## Forum: every thread in the Lost in Transcription category

[mdc-competitions.discourse.group/c/lost-in-transcription/5](https://mdc-competitions.discourse.group/c/lost-in-transcription/5)

| Thread | Date | Substance |
|---|---|---|
| [Ind_Jav Dev/Test distribution shift](https://mdc-competitions.discourse.group/t/ind-jav-dev-test-distribution-shift-could-test-metadata-csv-aggregates-be-shared/25) (#25) | 2026-09-02 | **Directly about our track.** See below. |
| [MDC Datasets With "No AI Training" Clauses](https://mdc-competitions.discourse.group/t/mdc-datasets-with-no-ai-training-clauses/24) (#24) | 2026-09-01 | Participants must obtain explicit permission from dataset owners themselves; MDC will not do it centrally. |
| [Submission rule clarification for teams](https://mdc-competitions.discourse.group/t/submission-rule-clarification-for-teams/23) (#23) | 2026-09-01 | **3 submissions per 7 days, per individual team member.** A 2-person team gets 6/week. |
| [Runtime PR: kenlm + pyctcdecode](https://mdc-competitions.discourse.group/t/runtime-pr-kenlm-pyctcdecode-for-ctc-beam-search-with-lm-shallow-fusion/29) (#29) | 2026-09-04 | A participant added KenLM + pyctcdecode to the runtime — someone is pursuing a CTC + LM shallow-fusion path. |
| [Rules clarification: data firewall, aggregate metrics, private compute](https://mdc-competitions.discourse.group/t/rules-clarification-data-firewall-aggregate-metrics-and-private-compute/17) (#17) | 2026-08-31 | Rules questions on data handling and private compute. |
| [How to add team in competition](https://mdc-competitions.discourse.group/t/how-to-add-team-in-competition/28) (#28) | 2026-09-04 | Team formation mechanics. |
| [When will submissions open for the Indonesian-Javanese track?](https://mdc-competitions.discourse.group/t/when-will-submissions-open-for-the-indonesian-javanese-track/19) (#19) | 2026-08-31 | Track opened later than Nahuatl. |

### The distribution-shift thread is the most important one

Participant **Leopoldhillah** reported that the Indonesian–Javanese **dev and test sets have
substantially different clip-length distributions**, making offline validation poorly
predictive of leaderboard results. They asked the organizers either to publish
`test_metadata.csv` aggregate statistics, or to permit submitted code to log summary
statistics from it.

Organizer **Matt** declined both:

> "No. We don't make additional details of the test set publicly available."

and on logging aggregates from `test_metadata.csv`:

> "No. Per the competition rules… each test data sample should be processed independently
> during inference without the use of information from other cases in the test set."

**What this means for us.** Three things:

1. **Independent corroboration of our own experience.** Our smoke 0.2000 → full 0.47555 gap
   is the same phenomenon another team is reporting from a different angle. Dev-set WER will
   *systematically understate* test WER. Do not treat a dev number as a leaderboard forecast;
   treat it only as a *ranking* signal between our own candidates.
2. **Clip duration is the known axis of shift**, and `test_metadata.csv` gives us
   `file_duration_seconds` per clip. Using a clip's *own* duration is legal — it is that
   sample's metadata, not cross-sample information. So we can and should:
   * check dev WER **stratified by duration bucket**, since the test mix is weighted
     differently;
   * weight dev validation to approximate a flatter duration distribution when comparing
     candidates;
   * make sure long clips (up to 40 s) are handled correctly, since they are likely
     over-represented in test relative to dev.
3. **Aggregating anything across test clips is explicitly forbidden.** No batch statistics,
   no adaptive normalisation over the test set, no logging test aggregates.

That last point also constrains logging generally: 500 lines, 300 chars per line, and
nothing about test data.

## Published Javanese ASR benchmarks

These set expectations for what WER is achievable. Numbers are from published work, not
from our own runs.

| System | Task | WER | Source |
|---|---|---|---|
| Whisper zero-shot, Javanese | Javanese ASR | **52.25** | [Enhancing Zero-shot Whisper ASR for Under-resourced Javanese and Sundanese](https://scholar.its.ac.id/en/publications/enhancing-zero-shot-whisper-asr-for-under-resourced-javanese-and-/) (ITS, 2026) |
| Whisper + generative fusion decoding + logit calibration | Javanese ASR | **44.83** | same |
| Our base Qwen3-ASR-1.7B | this competition, full test | **47.56** | measured, [../01-run-history.md](../01-run-history.md) |
| Our base Qwen3-ASR-1.7B | OpenSLR-35 held-out read speech | 70.59 | measured |
| Our Qwen fine-tuned on OpenSLR-35 | OpenSLR-35 held-out read speech | 29.37 | measured |
| Leaderboard #1 | this competition, public test | **21.96** | leaderboard, 2026-09-05 |

Two conclusions:

* **Our 0.47555 is not anomalous — it is roughly what zero-shot models score on Javanese.**
  The published zero-shot Whisper Javanese number (52.25) is *worse* than our zero-shot Qwen
  (47.56). The base model is not the problem.
* **The leader at 0.2196 is far below any published zero-shot number.** That gap is not
  achievable by decoding tricks; it is what in-domain fine-tuning on Jember buys. It is
  consistent with the Pasketti winners' >50% improvement over strong baselines through
  fine-tuning alone.

Also relevant: an [XLS-R study on Indonesian, Javanese and Sundanese](https://arxiv.org/pdf/2401.06832)
and [ASR Under Noise: Robustness for Sundanese and Javanese](https://arxiv.org/pdf/2509.25878).
General finding across this literature: **MMS and w2v-BERT are more data-efficient in very
low-resource regimes, XLS-R scales better with more data, and Whisper is strongest at
mid-resource.** With ~10 h of Jember plus several GB of supplementary Javanese audio, we are
in the regime where a strong pretrained model fine-tuned on in-domain data wins — which is
exactly what the Pasketti result shows.

## Hugging Face: existing Javanese / Indonesian ASR checkpoints

Swept via the HF API, sorted by downloads. **There is no strong pre-existing checkpoint to
start from.** The field is dominated by 2022-era wav2vec2 models and small Whisper
fine-tunes on OpenSLR read speech — i.e. the same wrong domain we already trained on.

| Model | Downloads | Modified | Assessment |
|---|---|---|---|
| `indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` | 3.7 M | 2022-03 | old CTC architecture; would need KenLM to be competitive |
| `indonesian-nlp/wav2vec2-large-xlsr-indonesian` | 30.8 k | 2022-03 | Indonesian only, 2022 |
| `cahya/wav2vec2-large-xlsr-javanese` | 365 | 2022-03 | low usage, 2022 |
| `bagasshw/whisper-large-v2-javanese-openslr-v1/v2` | ~20 | 2025-04/05 | **trained on OpenSLR-35** — the same read-speech domain whose limits we have already measured |
| `bagasshw/whisper-tiny-javanese-openslr-v2..v8` | <60 | 2025 | tiny models, not competitive |
| `eldon922/whisper-large-v3-turbo-indonesian-ezra` | 9 | 2025-06 | Indonesian, unverified provenance |

**Verdict: no shortcut.** Fine-tuning Qwen3-ASR-1.7B on Jember ourselves is the path. This
is a useful negative result — it means no competitor has a public head start either, and the
differentiator is purely who exploits the in-domain data best.
