PYTHON ?= python3
export PYTHONPATH := src

.PHONY: w2-upload w2-prep w2-stage0 w2-train w2-soup w2-final w2-report w2-plots test catalog score runtime openslr-download openslr-subset whisper-smoke qwen-smoke hf-backup-local modal-status modal-stage-qwen modal-stage-openslr modal-train modal-eval modal-package modal-download modal-backup-hf

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

# --- Wave 2: the depth path (docs/kb/12-runbook-wave2.md) -------------------
# MODAL_APP_NAME / MODAL_VOLUME_NAME override the generic defaults if needed.
MODAL ?= uvx --from modal modal
PREP_SHA ?=

w2-upload:
	python3 scripts/upload_external.py

w2-prep:
	@test -n "$(PREP_SHA)" || (echo "set PREP_SHA=<sha256 of uploads/prep-sources.tar>" && exit 1)
	$(MODAL) run scripts/modal_recovery.py::prep2 --sources-sha256 $(PREP_SHA)

w2-stage0:
	$(MODAL) run scripts/modal_recovery.py::stage0

w2-train:
	$(MODAL) run --detach scripts/modal_recovery.py::wave2

w2-soup:
	$(MODAL) run scripts/modal_recovery.py::soup_best

w2-final:
	@test -n "$(STEPS)" || (echo "set STEPS=<optimizer steps the arms selected>" && exit 1)
	$(MODAL) run --detach scripts/modal_recovery.py::final_fit --steps $(STEPS)

w2-plots:
	$(MODAL) run scripts/modal_recovery.py::plots

w2-report:
	$(MODAL) run scripts/modal_recovery.py::report
