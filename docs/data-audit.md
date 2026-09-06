# Indonesian–Javanese data audit

Completed 2026-09-05 on both full, checksum-verified archives. All 580 source audio files were decoded locally. The main preparation risks are faulty training timestamps, different transcription conventions, short training utterances versus long development clips, and very limited development coverage.

The follow-up **[relationship analysis](data-relationships.md)** quantifies speaker/conversation/language confounding, 91 audio/text feature pairs per dataset, recording-level quality concentration, vocabulary coverage by validation fold, and the consequences of filtering. It includes two figures and an explicit list of what automated auditing cannot establish.

## Measured corpus inventory

| Measure | Development | Jember training |
| --- | ---: | ---: |
| Source MP3 files | 372 | 208 |
| Metadata rows | 372 | 6,679 |
| Positive, parseable timestamp rows | Not applicable | 6,638 |
| Decoded source audio | 2.562 hours | 10.015 hours |
| Decoded utterance duration sum | 2.562 hours | 9.979 hours |
| Unique annotated time within audio bounds | 2.562 hours | 9.977 hours |
| Decode failures | 0 | 0 |
| Missing/orphan source audio | 0 / 0 | 0 / 0 |
| Channels | All mono | All mono |
| Sample rates | 372 at 48 kHz | **89 at 44.1 kHz; 119 at 48 kHz** |
| Median utterance duration | 26.21 seconds | 5.00 seconds |
| Utterance duration range | 6.29–37.81 seconds | 1.00–29.00 seconds |
| Scoring-normalized reference words | 17,485 | 65,739 across all rows; 65,645 in positive timestamp rows |
| Exact duplicate audio hash groups | 0 | 0 |

The [published problem description](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/) describes Jember recordings as 44.1 kHz. The actual archive is mixed-rate. Always read the native rate before resampling to the selected model's input rate.

Training segment metrics use all 6,638 positive timestamp rows, including five segments clipped to available audio for measurement. Those five are explicitly flagged; their shortened waveforms are not validated training examples. Summed segment time double-counts six seconds of timestamp overlap; unique coverage is reported separately.

All metadata columns are present, all cells are nonempty, and development filenames are unique and safe. There are 85 untrimmed training transcript cells. No normalized references become empty and all raw transcripts are NFC.

## 1. Fix or quarantine timestamp defects before training

There are **46 distinct training rows requiring timestamp review**:

- **38 zero-duration rows**.
- **3 reversed intervals**: TSV data rows 2,728 (session 63), 4,696 (session 139), and 5,283 (session 174).
- **5 intervals extending beyond decoded recording duration**.

| TSV data row | Session | Annotated interval, seconds | Beyond available audio |
| --- | ---: | ---: | ---: |
| 594 | 12 | 30–35 | 2.216 seconds |
| 1,373 | 30 | 743–761 | **16.160 seconds** |
| 2,165 | 45 | 230–235 | 2.798 seconds |
| 4,928 | 153 | 79–83 | 0.800 seconds |
| 6,274 | 201 | 452–459 | 0.384 seconds |

Two additional rows in session 63 overlap preceding annotation coverage by two and four seconds. Review that session as a unit; it also contains a reversed interval.

The existing `prepare_jember.py` skips nonpositive intervals and does not check the decoded recording boundary. Running it without this audit would silently omit 41 rows and could produce truncated segments for the other five.

Use [timestamp_quarantine_candidates.csv](../reports/private/timestamp_quarantine_candidates.csv) as a review list. Row numbers are one-based data rows, excluding the TSV header. Do not invent corrections or infer missing speech from text. The original archives and transcripts remain intact; this audit did not filter, repair, or overwrite them.

## 2. Development coverage is narrow and uneven

| Clip label | Clips | Clip share | Reference words | Audio hours |
| --- | ---: | ---: | ---: | ---: |
| Indonesian–Javanese (`javind`) | 352 | 94.62% | 16,624 | 2.435 |
| Indonesian only (`ind`) | 20 | 5.38% | 861 | 0.126 |
| Javanese only (`jav`) | **0** | 0% | 0 | 0 |

All labels are permitted values, but this development set cannot measure monolingual Javanese performance. The 20 Indonesian-only clips all come from the smaller conversation.

| Conversation | Speakers | Clips | Reference words | Word share | Audio hours |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | 5, 7 | 78 | 3,556 | 20.34% | 0.518 |
| 5 | 1, 6 | 294 | 13,929 | **79.66%** | 2.044 |

Speakers do not cross these conversations. Hold out whole conversations and their speakers; random clip splits would share speakers and conversational context. Report both directional holdouts, per-conversation WER, per-language WER where labels are represented, and pooled corpus WER. The two directional holdouts are not interchangeable: training on conversation 5 provides no development examples labeled `ind`.

A verified [development_folds.csv](../reports/private/development_folds.csv) assigns all 372 clips for each of the two holdouts, without copying transcripts. It is a suggested evaluation manifest, not a model evaluation. Repeated tuning on these same two conversations will still overfit them.

Jember provides session IDs but **no speaker or conversation annotations**. Session-disjoint training folds cannot be described as speaker-disjoint without additional evidence.

## 3. Training and development transcription conventions differ

Jember uses many `è`, `ê`, and `ě` characters, plus a few other marked letters. Development's only non-ASCII character is `é`, with 54 occurrences. Approximately **14.29%** of all training scoring words change if accents are stripped, versus **0.30%** in development.

| Vocabulary comparison against valid Jember rows | Development word occurrences unseen in training |
| --- | ---: |
| Official scoring tokens | 21.29% |
| Case-insensitive comparison | 21.01% |
| Accent- and case-insensitive comparison | 18.00% |

This is evidence of transcription and vocabulary mismatch. Accent/case folding narrows the gap by 3.29 percentage points but leaves substantial unseen vocabulary. These are reference-token coverage measurements, **not ASR WER improvements**.

Blindly lowercasing the development references changes **3.32%** of their scoring words. The existing scorer preserves meaningful mid-sentence case. Do not blanket-lowercase model output. Evaluate alternative training-label conventions on held-out conversations while preserving the original labels and keeping the official scoring normalizer fixed.

Other measured conventions:

- Development: 280 rows contain hyphens, 125 contain ellipses, and none contain digits.
- Full training metadata: 1,565 rows contain hyphens, 714 contain ellipses, and 49 contain digits.
- Full training metadata has 32 repeated normalized-text groups, representing 108 excess rows. Positive timestamp rows have 29 such groups and 86 excess rows. Repeated short speech is not automatically erroneous or duplicate audio.
- No exact normalized transcript duplicates occur within development.

The audit does not infer word-level code-switches from a small dictionary. The two languages share vocabulary, and the supplied labels are clip-level annotations.

## 4. Duration and recording quality differ

Development averages **24.79 seconds and 47.0 words per clip**; valid Jember utterances average **5.41 seconds and 9.89 words**. There are **37 development clips longer than 30 seconds** (9.95%). A decoder that silently processes only the first 30 seconds would lose reference-bearing audio.

There are 388 training utterances shorter than two seconds. These are source-corpus segments, so development's published approximate duration range is not a reason to reject them. Consider a held-out experiment with longer training windows formed only from adjacent, validated intervals in the same recording.

| Review flag | Development, 372 clips | Training, 6,638 measured segments |
| --- | ---: | ---: |
| Near-full-scale samples exceed 0.01% | 13 | 228 |
| More than half of analysis frames relatively silent | 1 | 181 |
| Frame-energy contrast below 15 dB | 0 | 396 |
| RMS below −35 dBFS | 0 | 798 |
| Fewer than 0.5 scoring words/second | 0 | 51 |
| More than 6 scoring words/second | 0 | 3 |

Flags overlap. The combined [review_queue.csv](../reports/private/review_queue.csv), including timing defects, contains 1,471 distinct training rows and 14 development clips. This is **not a discard list**. Prioritize timing errors, extreme token rates, and the most severely affected recordings for local listening.

Median RMS is −22.26 dBFS in development versus −25.87 dBFS in training. Median estimated effective bandwidth is 7.59 kHz versus 9.73 kHz. These are descriptive differences, not proof of different devices or an instruction to denoise.

The clipping flag measures decoded samples at or above 0.999 in absolute magnitude. MP3 reconstruction can overshoot full scale, so it does not by itself establish hard clipping in the original recording. “SNR” in the JSON is the 90th-minus-10th-percentile frame-level energy contrast; it is not a calibrated acoustic SNR, and digital silence can make it very large.

## 5. No overlap detected by the implemented screens

- No exact normalized train/development transcript matches.
- No shared five-word phrases after case folding, including a second pass that also strips accents.
- No exact source-file duplicates within either dataset or exact native-PCM duplicates among their measured utterances.
- No whole-development-clip acoustic candidate reaches the exploratory review threshold of 0.65; the highest score is **0.486**.

The acoustic screen resamples locally to 8 kHz and searches eight-band spectral changes across all 208 complete training recordings for each of the 372 development clips. It is independent of the native rates, filenames, and transcripts. Five training excerpts were cropped at a non-frame-aligned offset and re-encoded at 48 kHz as positive controls. **All five were located in the correct recording within half a second**, with scores 0.656–0.740.

This supports “no overlap detected,” not a guarantee of disjointness. The acoustic threshold is exploratory and tested on only five positive controls. Short shared excerpts, spliced or speed-altered audio, and shared speakers remain possible. Native-PCM hashes alone are particularly weak across different sample rates and MP3 encodings. Hidden test data was not accessed or audited.

## Recommended preparation order

1. Review the 46 timestamp quarantine candidates and the two overlapping intervals. Record explicit keep/fix/exclude decisions with source row IDs.
2. Resample from each file's actual native rate; preserve original audio and text. Make decoder handling of the 37 clips over 30 seconds explicit.
3. Establish conversation-held-out baseline WER and report the smaller conversation separately. Do not tune a `jav` router using this development set: it has no such examples.
4. Compare training transcript conventions using held-out WER. The accent mismatch deserves an experiment; measured token changes do not establish which transformation improves a model.
5. Review the quality queue before choosing filtering or augmentation. Jember session IDs must not be treated as verified speaker IDs.

No model was trained or scored in this audit, so it makes no claim about leaderboard performance.

## Outputs and reproduction

- [Main machine-readable summary](../reports/data_audit.json).
- [Supplement: complete metadata text, corrected coverage, and acoustic-screen summary](../reports/audit_supplement.json).
- [Private per-clip metrics](../reports/private/clip_metrics.json).
- [Private timestamp details](../reports/private/timestamp_anomalies.json).
- [Private acoustic matches and positive controls](../reports/private/acoustic_overlap.json).
- [Private text-overlap evidence](../reports/private/overlap_candidates.json).

From `mozilla_indonesia/`, using the existing parent environment:

```bash
../.venv/bin/python scripts/audit_data.py --workers 4
../.venv/bin/python scripts/audit_audio_overlap.py
../.venv/bin/python scripts/summarize_audit.py
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -v
```

Requirements: Python 3.12, NumPy (run used 2.5.2), and ffmpeg/ffprobe. The full workspace test run passed **52 tests**, including timestamp validation, interval unions, actual WAV/MP3 decoding, scorer-normalization comparisons, and acoustic match localization.

The language-independent `src/mdc_asr/audio_probe.py` and its tests were copied from the Spanish workspace so this nested repository does not depend on parent source imports. The probe source is unchanged, SHA-256 `4fbd7fb9de05be6d973a75784fb2992b6e829d9afd1d74b9cc844b0fac2ffa60`. The Indonesian audit uses its own scorer and excludes Nahuatl-specific orthography assumptions.

Downloads, extracted data, signal caches, and per-file reports are gitignored. No audio or transcript text was uploaded to an external service. No original dataset files were edited.

## Source provenance

Both supplied archives match the public catalog checksums verified in the initial access review:

| Archive | SHA-256 |
| --- | --- |
| Development | `effcb1b34991e3ba7a0f8abc2d24324d42a8e0304a6da366dffeb683da773cad` |
| Jember | `87ecd658248746239360ea800967458c9fcaa785d225913d49446ef3d2651fc1` |

- [Development source](https://mozilladatacollective.com/datasets/cmsxiwqdo000fny07e5fx0yi5): 72,772,516-byte archive; catalog license CC-BY-SA-4.0.
- [Jember source](https://mozilladatacollective.com/datasets/cmlgm5a94008kny07nz2intus): 284,847,670-byte archive; catalog license CC-BY-NC-SA-4.0.
- [Official task and data specification](https://competitions.mozilladatacollective.com/competitions/1/lost-in-transcription-in-jv/page/2/).

Keep the separate source licenses with any subsequent training artifacts. The documented development/test speaker separation does not establish Jember-training/development speaker separation.
