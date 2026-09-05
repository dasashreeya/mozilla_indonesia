PYTHON ?= python3
export PYTHONPATH := src

.PHONY: test catalog score runtime openslr-download openslr-subset whisper-smoke qwen-smoke hf-backup-local modal-status modal-stage-qwen modal-stage-openslr modal-train modal-eval modal-package modal-download modal-backup-hf

test:
	$(PYTHON) -m unittest discover -s tests -v

catalog:
	$(PYTHON) scripts/mdc_fetch.py catalog --query Javanese --task ASR

score:
	@test -n "$(TRUTH)" -a -n "$(PRED)" || (echo "Usage: make score TRUTH=... PRED=..."; exit 2)
	$(PYTHON) -m mdc_asr.wer "$(TRUTH)" "$(PRED)" --group-by language

runtime:
	@test -d runtime/.git || git clone https://github.com/drivendataorg/lost-in-transcription-runtime.git runtime
	git -C runtime pull --ff-only
	@git -C runtime rev-parse HEAD

openslr-download:
	$(PYTHON) scripts/fetch_openslr35.py

openslr-subset: openslr-download
	@test -f data/subsets/openslr35/shard0-3000-seed-20260905/SUBSET_MANIFEST.json \
		&& echo "Using prepared OpenSLR subset" \
		|| $(PYTHON) scripts/prepare_openslr35.py

whisper-smoke:
	uv run --python 3.12 --with-requirements requirements-local-eval.txt \
		python scripts/evaluate_whisper.py --limit 10 \
		--output experiments/predictions/openslr35-whisper-jw-smoke.csv

qwen-smoke:
	uv run --python 3.12 --with-requirements requirements-local-qwen.txt \
		python scripts/evaluate_qwen.py --limit 10 \
		--output experiments/predictions/openslr35-qwen-auto-smoke.csv

hf-backup-local:
	uv run --python 3.12 --with huggingface_hub \
		python scripts/hf_backup.py

modal-status:
	uvx --from modal modal run modal_app.py::volume_status

modal-stage-qwen:
	uvx --from modal modal run modal_app.py::stage_qwen

modal-stage-openslr:
	uvx --from modal modal run modal_app.py::stage_openslr

modal-train:
	uvx --from modal modal run --detach modal_app.py::train_qwen

modal-eval:
	uvx --from modal modal run modal_app.py::evaluate_qwen

modal-package:
	uvx --from modal modal run modal_app.py::package_submission

modal-download:
	uvx --from modal modal volume get govind-llm-data \
		artifacts/modal-qwen-openslr35-stage1-submission.zip \
		artifacts/modal-qwen-openslr35-stage1-submission.zip

modal-backup-hf:
	uvx --from modal modal run modal_app.py::backup_to_hf
