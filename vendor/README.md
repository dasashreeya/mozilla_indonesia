# Vendored training helper

`qwen3_asr_sft.py` is the official Qwen3-ASR supervised fine-tuning script fetched from:

- Repository: <https://github.com/QwenLM/Qwen3-ASR>
- Commit: `7c6daf77a2421100f5fb066495372c00129d39ff`
- Upstream path: `finetuning/qwen3_asr_sft.py`
- SHA-256: `6f3e5cc530d9da9ff405144fc94a5c4790018eaf956dbc623634818160418976`
- Upstream license: Apache-2.0 (the file retains its copyright/license header)

Do not silently update it during the competition. If updating, pin the new commit and hash, inspect the diff, rerun both validation folds, and update this receipt.

Local recovery patch (2026-09-05): the upstream hash above describes the original
source, not the patched file. `response_only_labels` now masks the first N active
prompt positions using the attention mask, supporting either padding side
and preserving real EOS when its ID equals the padding ID. Although the tokenizer
property reports right padding, the processor overrides it to left padding on
actual calls (verified with remote attention masks). The defect affects training,
not the earlier leaderboard submission that used base weights. Labelled forward calls
disable KV caching for gradient-checkpointed training. No upstream revision was
changed. Synthetic padding/EOS regression tests verify the masking logic; actual
in-domain validation and the opposite conversation fold remain promotion gates.
