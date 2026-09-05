# MDC data catalogue for the Javanese–Indonesian track

_Researched 2026-09-05 via the public MDC catalogue API (no key required):_
`python3 scripts/mdc_fetch.py catalog --query Javanese`

## The finding that matters

**MDC hosts roughly 3.5 GB of Javanese TTS corpora that nobody is treating as ASR training
data.** A TTS corpus is a set of (audio, transcript) pairs — read speech, studio-clean, but
in-language and in-dialect. Several are from **East Java**, the same region as the Jember
target dialect, and most are **CC-BY-SA-4.0**, the permissive licence class.

| Dataset | ID | Size | Licence | Region |
|---|---|---|---|---|
| TTS Javanese-Lumajang Dialect | `cml5bgysg00bhkr07g23kewke` | 718 MB | CC-BY-SA-4.0 | **East Java, adjacent to Jember** |
| Bojonegoro Javanese TTS | `cmltnzkug0012mh07k1obic7v` | 492 MB | CC-BY-SA-4.0 | **East Java** |
| Javanese TTS of Banyumasan Dialect | `cmmamtaf104nemh07xa9e7sdx` | 586 MB | CC-BY-SA-4.0 | Central Java (Banyumasan) |
| TTS Javanese – Ngapak Dialect | `cmlgmf58l0096nx07tahttd6y` | 595 MB | CC-BY-SA-4.0 | Central Java |
| TTS Central Javanese | `cml5bn4k900aame07u0rwidcg` | 461 MB | CC-BY-SA-4.0 | Central Java |
| TTS Javanese-Gresik Dialect | `cmpwi0ac200zbo007hebmkzqy` | 297 MB | CC-BY-**NC**-SA-4.0 | **East Java** |
| Jombang Dialect-Javanese TTS | `cmpo53l5b000pku07cem453ac` | 295 MB | CC-BY-**NC**-SA-4.0 | **East Java** |
| TTS Javanese – Multidialect | `cmtimj5tm0045mj07qrjo74gc` | 136 MB | CC-BY-**NC**-SA-4.0 | mixed |

Lumajang borders Jember Regency. Bojonegoro, Gresik and Jombang are all East Javanese.
This is far closer to the target than OpenSLR-35, which is standard Javanese read speech
and is what we have trained on so far.

Caveats to verify on download: TTS corpora are usually few-speaker and studio-recorded, so
they add lexical and dialectal coverage but not conversational acoustics. Treat them as a
*supplement* to Jember with a capped sampling weight, never as the primary source. Confirm
transcript conventions (diacritics, casing) before mixing — see
[../02-measured-findings.md](../02-measured-findings.md) §4.

They are also the natural raw material for the **synthetic-TTS augmentation** that Mitchell
DeHaven — currently #3 on our leaderboard — used to win a bonus prize in the Pasketti
challenge. See [winning-solutions.md](winning-solutions.md).

## ASR datasets

| Priority | Dataset | ID | Size | Licence | Role |
|---|---|---|---|---|---|
| **0** | Lost In Transcription Development Data (Javanese-Indonesian) | `cmsxiwqdo000fny07e5fx0yi5` | 73 MB | CC-BY-SA-4.0 | the only in-domain labelled audio; 372 clips |
| **0** | Jember Javanese Spontaneous Speech Corpus | `cmlgm5a94008kny07nz2intus` | 285 MB | **CC-BY-NC-SA-4.0** | **dev and test are both drawn from this corpus** — the single most important training set |
| 1 | Common Voice Spontaneous Speech 4.0 – Javanese | `cmqi294n3004mmf07wvybp3fn` | 30 MB | CC0-1.0 | spontaneous Javanese, fully permissive |
| 2 | Common Voice Scripted Speech 26.0 – Indonesian | `cmqinqwef00xonr07z4vbfovw` | 1.5 GB | CC0-1.0 | Indonesian coverage; subsample heavily, read speech |
| 2 | Podcast Hari Minggoean | `cmhlvswz200fio007c7szypw8` | 355 MB | CC-BY-SA-4.0 | conversational Indonesian |
| 2 | Podcast Homostoria | `cmiepnyu1001jo207hzslvb5m` | 318 MB | CC-BY-SA-4.0 | conversational Indonesian |
| 3 | Bahasa Indonesia Audio Visual Speech Dataset | `cmtk31icr000lnv07v0ersyeq` | 1.0 GB | CC-BY-NC-SA-4.0 | Indonesian, NC |

## Text corpora for LM / rescoring / lexicon

| Dataset | ID | Size | Licence |
|---|---|---|---|
| Corpus of Panjebar Semangat Javanese-Language Magazine | `cmk719jvs02z1nt07tjswt52s` | 4.5 MB | CC-BY-SA-4.0 |
| Lumajang Dialect-Javanese Parallel Corpus (ind/jav/eng/arb) | `cmtbjsbum00vmnr0701e5imnc` | 536 KB | CC-BY-SA-4.0 |
| Semarang Dialect-Javanese Parallel Corpus | `cmtbjkm6400vbnr075arsqkst` | 566 KB | CC-BY-SA-4.0 |

The Lumajang parallel corpus is East Javanese text and is useful both for a KenLM and for
checking the orthographic convention question.

## Licence risk

**Jember, the most important dataset, is CC-BY-NC-SA-4.0.** The competition requires
winning code under MPL-2.0 with supporting software permitting free commercial use. Whether
an NC-licensed *training corpus* is compatible with that obligation is genuinely ambiguous:
a corpus is not software, and the competition's own problem description names Jember as the
primary training resource, which strongly implies it is intended to be used.

The organizers have addressed the general question but not this specific one. From
[forum thread 24](https://mdc-competitions.discourse.group/t/mdc-datasets-with-no-ai-training-clauses/24),
admin Mdaley:

> "If participants are unsure if they are permitted to use a dataset for this competition,
> they must obtain explicit permission from the dataset owner."

and, when asked whether MDC could seek permission collectively:

> "In this case, it is on the participants to obtain explicit consent from dataset owners
> if it is unclear whether a dataset is allowed to be used for the purposes of this
> competition."

**Action:** post on the forum asking specifically whether the NC clause on the Jember corpus
— the dataset the competition itself designates as the primary training resource — is
compatible with the winner obligations. Do this now; the answer gates the whole training
plan, and the same question applies to the Gresik, Jombang and Multidialect TTS corpora.

The CC-BY-SA and CC0 datasets above carry no such ambiguity. If the NC answer is
unfavourable, a viable all-permissive mixture still exists: dev data (CC-BY-SA) + CV
Javanese Spontaneous (CC0) + CV Indonesian (CC0) + the five CC-BY-SA Javanese TTS corpora +
the two CC-BY-SA podcasts + OpenSLR-35 (CC-BY-SA).

## Clearing the HTTP 403

The API key authenticates requests but does not accept dataset terms. A 403 means terms are
unaccepted or an access request is unapproved; a 401 means a bad key. Log into
[mozilladatacollective.com](https://mozilladatacollective.com) with the competition account,
open each dataset page, and complete the Download / terms flow including any approval
request. Request all datasets at once — approval may take days and it is the critical path.

```bash
export MDC_API_KEY='...'
python3 scripts/mdc_fetch.py download dev --extract
python3 scripts/mdc_fetch.py download jember --extract
```

Note that catalogue queries need no key, which is how this note was compiled — so the
403 is purely a terms/approval issue, not a credentials issue.

## External-data obligation

Anything used for training that is **not** from MDC must be published to MDC when the
competition ends. That applies to OpenSLR-35, which we have already trained on. It does not
apply to any of the datasets above, which are already on MDC — a further reason to prefer
them.
