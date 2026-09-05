# Next actions

_Last updated 2026-09-05. 20 days to the 2026-09-25 00:00 UTC deadline._

Ordered by expected WER gain per hour. Claims trace to
[02-measured-findings](02-measured-findings.md) (our own measurements) or to
[research/](research/) (external sources, with URLs), or are labelled an estimate.

## If you only do three things

1. **Clear the MDC 403 today** — dev and test are both drawn from the Jember corpus, so
   Jember is not "adjacent data", it is *the* training set. Everything is blocked behind it.
2. **Copy the proven recipe:** Qwen3-ASR-1.7B + LoRA (r=16, α=32, lr 1e-4, 3 epochs) on
   Jember, then **average 3+ checkpoints**. All three winners of the closest comparable
   DrivenData ASR competition used exactly this. See [research/winning-solutions.md](research/winning-solutions.md).
3. **Never lowercase predictions.** Measured cost: +11.8 WER points.

## What the research changed

| Previous belief | Now known | Source |
|---|---|---|
| "Qwen may be the wrong base since it doesn't list Javanese" | **Qwen3-ASR-1.7B is the right base.** All three Word Track winners of DrivenData's Pasketti ASR challenge used it, beating purpose-built models by >50% | [winning-solutions](research/winning-solutions.md) |
| "Whisper large-v3 is the stronger candidate" | **Retired.** Based on 22 vs 23 errors out of 115 words. Whisper never won a Word Track; its 30 s window is also a liability given 40 s clips | [01-run-history](01-run-history.md), [official-spec](research/official-competition-spec.md) |
| "0.47555 means something is badly broken" | **0.47555 is normal for zero-shot Javanese.** Published zero-shot Whisper on Javanese is 52.25 | [forum-and-benchmarks](research/forum-and-benchmarks.md) |
| Submission limit unknown | **3 per 7 days per person** → ~8–9 remain | [forum thread 23](https://mdc-competitions.discourse.group/t/submission-rule-clarification-for-teams/23) |
| Jember is "closest domain" | **Dev and test are both drawn from Jember** | [problem description](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/) |
| Dev WER will predict test WER | **It will not.** Another participant reported a dev/test clip-length distribution shift; organizers refused to disclose test aggregates | [forum thread 25](https://mdc-competitions.discourse.group/t/ind-jav-dev-test-distribution-shift-could-test-metadata-csv-aggregates-be-shared/25) |

## P0 — today

### Clear the MDC 403

Log into MDC with the competition account and complete the Download/terms flow for each
dataset. Request them **all at once** — approval may take days.

Priority: dev data (`cmsxiwqdo000fny07e5fx0yi5`), **Jember**
(`cmlgm5a94008kny07nz2intus`), CV Spontaneous Javanese (`cmqi294n3004mmf07wvybp3fn`), CV
Scripted Indonesian (`cmqinqwef00xonr07z4vbfovw`), and the East Javanese TTS corpora listed
in [research/mdc-data-catalog.md](research/mdc-data-catalog.md).

```bash
export MDC_API_KEY='...'
python3 scripts/mdc_fetch.py download dev --extract
python3 scripts/mdc_fetch.py download jember --extract
```

Catalogue queries need no key, so the 403 is purely a terms/approval issue.

### Post two forum questions

1. **The Jember NC licence.** Jember is CC-BY-NC-SA-4.0 but is the competition's designated
   primary training resource, while winners must deliver MPL-2.0 code with supporting
   software permitting commercial use. Organizers have said participants must obtain
   permission from dataset owners themselves. Ask specifically about this dataset — the
   answer gates the entire training plan.
2. **The total submission cap**, as distinct from the 3-per-7-days rate.

### Consider adding a teammate

The limit is per *individual*. A second team member doubles the submission budget to 6 per
7 days. Rules constraints: one account per person, a participant may join only one team, and
the cross-track bonus requires identical team membership across tracks.

## P1 — the day the data lands

### Measure the reference conventions before training anything

On the MDC dev transcripts, answer:

1. What share of words are mid-sentence capitalised? (OpenSLR-35: 20.7%.)
2. Do Javanese words carry diacritics (`dhèwèké`) or not (`dheweke`)? Consistently?
3. Are disfluencies, fillers and `[bracketed]` annotations present?

`scripts/error_analysis.py` reports 1 and 2 directly. Worth up to 6.4 WER points
([02-measured-findings §2](02-measured-findings.md)). One hour, highest value on the list.

### Build honest validation, stratified by duration

Conversation- and speaker-disjoint folds, per [docs/strategy.md](../strategy.md). Then,
because of the known dev/test length shift, **also report WER by clip-duration bucket** and
compare candidates on a duration-reweighted score, not just the raw dev mean.

Only differences above ~0.015 are measurable on 372 clips. Bootstrap over utterances.

### Re-run baselines in-domain

Score base Qwen, base Whisper, and `openslr35-full-v1/checkpoint-930` on the same folds.
First honest comparison the project will have had. No submission required.

## P2 — the main training push, days 2–14

Start from the 1st-place Pasketti config verbatim
([research/winning-solutions.md](research/winning-solutions.md)) rather than our current
settings, which deviate on every axis:

| Parameter | Winner | Ours now |
|---|---|---|
| method | LoRA r=16, α=32, dropout 0 | full fine-tune |
| targets | q,k,v,o,gate,up,down proj | — |
| `train_audio_encoder` | true | — |
| lr | 1e-4 | 2e-5 |
| effective batch | 16 (4 × grad-acc 4) | 32 |
| epochs | 3 | 2 |
| precision | bf16 + grad checkpointing | bf16 |
| SpecAugment | freq 1×15, time 1×0.05, p=0.5 | none |
| speed perturb | explicitly disabled | none |
| context prompt | task-specific | none |
| checkpoint averaging | 3+ checkpoints, then model soup | none |

Then, in order:

1. **Fine-tune on Jember.** The in-domain data. Expected to be the dominant gain — the
   Pasketti winners beat strong baselines by >50% on fine-tuning alone. *Estimate.*
2. **Add a context prompt**, e.g. *"Transcribe this Javanese-Indonesian voice note.
   Speakers code-switch between Javanese and Indonesian."* Vary it per clip on the
   `language` label (`jav`/`ind`/`javind`) — legal, since it uses only that clip's own
   metadata. Free; tune on the dev folds.
3. **Average 3+ checkpoints** by validation WER, then soup the merged models. Zero
   inference cost, and it is what separated 1st from 3rd in Pasketti.
4. **Filter training data** by a reference model's WER/CER (winner used ≤0.7, or ≤0.9 WER
   or'd with ≤0.5 CER). Jember is segmented from session recordings via TSV timestamps and
   will contain alignment errors.
5. **Supplement with East Javanese TTS corpora** (Lumajang, Bojonegoro — CC-BY-SA, and
   Lumajang borders Jember Regency), capped so studio read speech does not dominate.
6. Optional if time: **synthetic TTS data**, the technique Mitchell DeHaven — currently #3
   on our leaderboard — used to win a Pasketti bonus.

Do **not** resume the OpenSLR read-speech run. Its gain curve is flat (620→930 bought
0.005), it is standard Javanese rather than the East Javanese Pandhalungan target, and the
156,643 staged shard-1–f utterances deepen the same wrong distribution.

## P3 — decoding and post-processing, days 14–19

| Change | Measured delta | Verdict |
|---|---|---|
| Lowercase all predictions | **+0.118** | never |
| Strip punctuation | −0.0001 | skip |
| Hyphen → space | −0.0003 | skip |
| Collapse repeated words | +0.002 | harmful here |
| Match reference diacritic convention | up to −0.033 | **do, once measured in domain** |
| Match reference casing convention | up to −0.030 | **do, once measured in domain** |

Error composition (88% substitutions; D=117, I=146) says decoding sweeps are low-yield and
hallucination is not currently a problem. Add a hallucination guard as cheap insurance for
spontaneous audio, not as a source of points.

Do **not** build an 11–13 model ROVER ensemble. It won the Pasketti *Phonetic* track under
different constraints and will not fit 2,118 clips in 2 hours. Weight averaging gives most
of the ensembling benefit at zero inference cost.

## P4 — packaging and submission, days 19–20

Harden [submission_src/main.py](../../submission_src/main.py):

* **No per-clip `try/except`** — one bad clip fails the whole run and burns a submission.
* **No global deadline guard** — if throughput lags, the run is killed at 2 h with no output.
* `transcribe_whisper` passes all 2,118 paths in one call; a single failure loses everything.
* No guarantee of a non-empty transcript per row.
* Consider the **vLLM backend** (`vllm==0.23.0` is in the runtime lock; the Pasketti bonus
  winner used vLLM greedy) if throughput is tight.
* Clips run to **40 s**; confirm `max_new_tokens=256` is not truncating long references.

Then run the official Docker harness offline, smoke test, and only then submit.

## Submission budget

3 per 7 days per person; 2 already spent. Roughly 8–9 remain before 2026-09-25.

| Submission | What it buys |
|---|---|
| 1 | first Jember-fine-tuned model — the first honest full-test datapoint |
| 2 | + context prompt and validated orthography post-processing |
| 3 | + checkpoint averaging and data filtering |
| 4 | + TTS-corpus supplementation |
| 5 | reserve for a regression or a runtime failure |
| final | best cross-validated candidate |

Because dev WER systematically understates test WER and the top three (0.2196 / 0.2247 /
0.2290) are separated by less than the noise floor, choose the final entry on
**cross-validated dev ranking**, not on the best public score. Keep
`artifacts/whisper-submission.zip` as a known-good fallback that at least runs.

## Research status

The external research pass is **done** for the official spec, the runtime, the forum,
competitor intelligence, the winning-solution recipe, the MDC catalogue, published Javanese
benchmarks, and the HF checkpoint landscape — see [research/](research/).

Still thin: code-switching ASR literature, KenLM/rescoring specifics, and Javanese
orthography scholarship. None of these are on the critical path.
