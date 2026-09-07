"""Isolated, bounded Modal training in the connected Indonesian workspace."""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "mozilla-indonesian-recovery"
app = modal.App("mozilla-indonesian-recovery")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)

base = (modal.Image.debian_slim(python_version="3.12")
        .apt_install("ffmpeg", "libsndfile1")
        .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
              "PYTHONPATH": "/workspace/src:/workspace/vendor", "OMP_NUM_THREADS": "8"}))
stage_image = base.uv_pip_install("huggingface_hub==1.30.0").add_local_dir(ROOT / "configs", "/workspace/configs")
train_image = (base.uv_pip_install("torch==2.11.0", "torchaudio==2.11.0", index_url="https://download.pytorch.org/whl/cu130")
               .uv_pip_install("accelerate==1.12.0", "datasets==5.0.1", "librosa==1.0.0",
                               "qwen-asr==0.0.6", "soundfile==0.14.0", "transformers==4.57.6",
                               "numpy==2.5.2", "safetensors==0.8.0", "peft==0.18.0")
               .add_local_dir(ROOT / "src", "/workspace/src")
               .add_local_dir(ROOT / "vendor", "/workspace/vendor")
               .add_local_file(ROOT / "scripts/train_recovery.py", "/workspace/scripts/train_recovery.py")
               .add_local_file(ROOT / "scripts/train_whisper_recovery.py", "/workspace/scripts/train_whisper_recovery.py")
               .add_local_file(ROOT / "scripts/eval_checkpoint.py", "/workspace/scripts/eval_checkpoint.py")
               .add_local_file(ROOT / "scripts/model_soup.py", "/workspace/scripts/model_soup.py"))


SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def check_name(value, what):
    if not value or any(c not in SAFE for c in value):
        raise ValueError(f"Unsafe {what}")
    return value


prep_image = (base.uv_pip_install("numpy==2.5.2", "soundfile==0.14.0")
              .add_local_dir(ROOT / "src", "/workspace/src")
              .add_local_file(ROOT / "scripts/__init__.py", "/workspace/scripts/__init__.py")
              .add_local_file(ROOT / "scripts/audit_data.py", "/workspace/scripts/audit_data.py")
              .add_local_file(ROOT / "scripts/prepare_recovery.py", "/workspace/scripts/prepare_recovery.py")
              .add_local_file(ROOT / "scripts/prepare_external.py", "/workspace/scripts/prepare_external.py"))

# Archive basenames as uploaded to /vol/uploads/external/.
EXTERNAL_ARCHIVES = {
    "tts_lumajang": "1769864789237-TTS Javanese-Lumajang Dialect.tar.gz",
    "tts_bojonegoro": "1770350405122-Recordings.tar.gz",
    "cv_jv": "1781690344059-sps-corpus-4.0-2026-06-12-jv.tar.gz",
    "podcast_hari": "1762320200795-Hari Minggoean.tar.gz",
    "podcast_homostoria": "1763685862325-homostoria.tar.gz",
}

# Only these preparation flags may be set remotely. Building the variants on the
# volume avoids re-uploading a ~2 GB prepared dataset per arm; the ~350 MB of
# sources is uploaded once instead.
PREP_FLAGS = {"--heldout", "--style", "--dev-repeats", "--heldout-jember-sessions", "--seed",
              "--window-mode", "--window-min", "--window-max", "--window-stride",
              "--short-fraction", "--language-tags", "--prompt", "--filter-flags"}


@app.function(image=prep_image, volumes={"/vol": volume}, timeout=4 * 3600, cpu=8, memory=32768)
def prepare(name: str, options: list[str], sources_sha256: str):
    """Build one training-data variant on the volume from the uploaded sources."""
    import hashlib
    import json
    import subprocess
    import tarfile
    from pathlib import Path
    check_name(name, "dataset name")
    for option in options:
        if option.startswith("--") and option.split("=")[0] not in PREP_FLAGS:
            raise ValueError(f"Preparation flag not allowed: {option}")
    sources = Path("/vol/prep-sources")
    if not (sources / "reports/private/review_queue.csv").is_file():
        archive = Path("/vol/uploads/prep-sources.tar")
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != sources_sha256:
            raise ValueError("Preparation sources SHA-256 mismatch")
        sources.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(sources, filter="data")
        # An archive built on macOS can carry AppleDouble "._name.tsv" companions,
        # which match a "*.tsv" glob and make the corpus look like it holds two
        # timestamp tables. Drop them rather than trusting the archive.
        stray = [p for p in sources.rglob("._*") if p.is_file()]
        for path in stray:
            path.unlink()
        print(f"Extracted sources; removed {len(stray)} AppleDouble files", flush=True)
        volume.commit()
    # prepare_recovery.py resolves its inputs relative to its own parent's parent,
    # so the sources are linked into the image root rather than copied.
    for link in ("data", "reports"):
        target = Path("/workspace") / link
        if not target.exists():
            target.symlink_to(sources / link)
    output = Path("/vol/data") / name
    command = ["python", "/workspace/scripts/prepare_recovery.py", "--output", str(output), *options]
    print(f"Preparing {name}: {' '.join(options)}", flush=True)
    subprocess.run(command, check=True)
    volume.commit()
    receipt = json.loads((output / "PREPARATION_RECEIPT.json").read_text())
    return {k: v for k, v in receipt.items() if k != "excluded_source_rows"}


@app.function(image=prep_image, volumes={"/vol": volume}, timeout=6 * 3600, cpu=16, memory=32768)
def prepare_external(name: str, corpora: list[str], language_tags: bool = False,
                     prompt: str = "", min_seconds: float = 18.0, max_seconds: float = 30.0,
                     stride: float = 1.0):
    """Build the training-only external corpus from the uploaded MDC archives."""
    import json
    import subprocess
    from pathlib import Path
    check_name(name, "dataset name")
    output = Path("/vol/data") / name
    command = ["python", "/workspace/scripts/prepare_external.py", "--output", str(output),
               "--min-seconds", str(min_seconds), "--max-seconds", str(max_seconds),
               "--stride", str(stride)]
    for corpus in corpora:
        if corpus not in EXTERNAL_ARCHIVES:
            raise ValueError(f"Unknown corpus {corpus!r}")
        archive = Path("/vol/uploads/external") / EXTERNAL_ARCHIVES[corpus]
        if not archive.is_file():
            raise FileNotFoundError(f"Upload missing: {archive}")
        command += ["--archive", f"{corpus}={archive}"]
    if language_tags:
        command.append("--language-tags")
    if prompt:
        command += ["--prompt", prompt]
    subprocess.run(command, check=True)
    volume.commit()
    receipt = json.loads((output / "PREPARATION_RECEIPT.json").read_text())
    return receipt


@app.function(image=stage_image, volumes={"/vol": volume}, timeout=3600, cpu=2, memory=8192)
def stage_data(archive_sha256: str, dataset: str = "recovery-v1"):
    import hashlib
    import json
    import tarfile
    from pathlib import Path
    if not dataset or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in dataset):
        raise ValueError("Unsafe dataset name")
    archive = Path(f"/vol/uploads/{dataset}.tar")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != archive_sha256:
        raise ValueError("Training archive SHA-256 mismatch")
    destination = Path("/vol/data")
    receipt = destination / dataset / "PREPARATION_RECEIPT.json"
    if not receipt.exists():
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as handle:
            handle.extractall(destination, filter="data")
    volume.commit()
    result = json.loads(receipt.read_text())
    return {"training_rows": result["training_rows_with_repeats"], "validation_rows": result["validation_rows"],
            "archive_sha256": archive_sha256}


@app.function(image=stage_image, volumes={"/vol": volume}, timeout=3600, cpu=2, memory=4096)
def stage_base():
    import json
    from huggingface_hub import snapshot_download
    from pathlib import Path
    config = json.loads(Path("/workspace/configs/models.json").read_text())["models"]["qwen3-asr-1.7b"]
    destination = Path("/vol/models/qwen3-asr-1.7b")
    snapshot_download(config["repo_id"], revision=config["revision"], local_dir=str(destination))
    receipt = {"repo_id": config["repo_id"], "revision": config["revision"], "license": config["license"]}
    (destination / "MODEL_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    volume.commit()
    return receipt


@app.function(image=stage_image, volumes={"/vol": volume}, timeout=3600, cpu=2, memory=4096)
def stage_whisper():
    """Stage Whisper large-v3, the one candidate base with a native Javanese token."""
    import json
    from huggingface_hub import snapshot_download
    from pathlib import Path
    config = json.loads(Path("/workspace/configs/models.json").read_text())["models"]["whisper-large-v3"]
    destination = Path("/vol/models/whisper-large-v3")
    snapshot_download(config["repo_id"], revision=config["revision"], local_dir=str(destination),
                      allow_patterns=config["files"])
    receipt = {"repo_id": config["repo_id"], "revision": config["revision"], "license": config["license"]}
    (destination / "MODEL_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    volume.commit()
    return receipt


@app.function(image=train_image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=2 * 3600, retries=0, single_use_containers=True)
def evaluate(label: str, model: str, dataset: str = "recovery-jav1", beams: int = 1,
             force_language: bool = False, prompt: str = "", language_map: str = "",
             batch_size: int = 8, limit: int = 0, backend: str = "auto"):
    """Score an existing checkpoint without training it -- a decode pass, not a run."""
    import json
    import subprocess
    from pathlib import Path
    check_name(label, "label")
    check_name(dataset, "dataset name")
    model_path = Path("/vol") / model
    if ".." in model or not model_path.is_dir():
        raise ValueError(f"Model not found on the volume: {model}")
    output = Path("/vol/evals") / f"{label}.json"
    command = ["python", "/workspace/scripts/eval_checkpoint.py", "--model", str(model_path),
               "--data", f"/vol/data/{dataset}", "--output", str(output), "--label", label,
               "--beams", str(beams), "--batch-size", str(batch_size), "--limit", str(limit),
               "--backend", backend, "--prompt", prompt, "--volume-name", VOLUME]
    if force_language:
        command.append("--force-language")
    if language_map:
        command += ["--language-map", language_map]
    subprocess.run(command, check=True)
    volume.commit()
    return json.loads(output.read_text())


@app.function(image=train_image, volumes={"/vol": volume}, cpu=8, memory=65536, timeout=3600)
def soup(label: str, checkpoints: list[str]):
    """Average checkpoints into one model: ensembling that costs no inference time."""
    import json
    import subprocess
    from pathlib import Path
    check_name(label, "label")
    output = Path("/vol/soups") / label
    command = ["python", "/workspace/scripts/model_soup.py", "--output", str(output)]
    for checkpoint in checkpoints:
        path = Path("/vol") / checkpoint
        if ".." in checkpoint or not path.is_dir():
            raise ValueError(f"Checkpoint not found on the volume: {checkpoint}")
        command += ["--checkpoint", str(path)]
    subprocess.run(command, check=True)
    volume.commit()
    return json.loads((output / "SOUP_RECEIPT.json").read_text())


@app.function(image=train_image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=6 * 3600, retries=0, single_use_containers=True)
def train_whisper(run_id: str, dataset: str = "recovery-jav1", epochs: int = 1,
                  batch_size: int = 8, grad_acc: int = 2, lr: float = 1e-5,
                  force_language: bool = True):
    """Fine-tune Whisper large-v3 on the same data and holdout as the Qwen arms."""
    import json
    import subprocess
    from pathlib import Path
    check_name(run_id, "run ID")
    check_name(dataset, "dataset name")
    output = Path("/vol/runs") / run_id
    command = ["python", "/workspace/scripts/train_whisper_recovery.py",
               "--model", "/vol/models/whisper-large-v3", "--data", f"/vol/data/{dataset}",
               "--output", str(output), "--epochs", str(epochs), "--batch-size", str(batch_size),
               "--grad-acc", str(grad_acc), "--lr", str(lr), "--volume-name", VOLUME]
    if force_language:
        command.append("--force-language")
    subprocess.run(command, check=True)
    volume.commit()
    return json.loads((output / "TRAIN_RESULT.json").read_text())


@app.function(image=train_image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=6 * 3600, retries=0, single_use_containers=True)
def train(run_id: str = "qwen-recovery-v1-fold5", epochs: int = 3, max_steps: int = -1,
          dataset: str = "recovery-v1", lr: float = 1e-5, batch_size: int = 2, grad_acc: int = 8,
          lora_rank: int = 0, lora_alpha: int = 32, spec_augment: float = 0.0,
          force_language: bool = False, keep_epochs: bool = False,
          extra_data: str = "", extra_fraction: float = 1.0):
    import hashlib
    import json
    import subprocess
    from pathlib import Path
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("Unsafe run ID")
    if not dataset or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in dataset):
        raise ValueError("Unsafe dataset name")
    data = Path("/vol/data") / dataset
    receipt = json.loads((data / "PREPARATION_RECEIPT.json").read_text())
    for filename, key in [("metadata.csv", "metadata_sha256"), ("train.qwen.jsonl", "train_jsonl_sha256"),
                           ("validation.qwen.jsonl", "validation_jsonl_sha256")]:
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != receipt[key]:
            raise ValueError("Uploaded training input hash mismatch")
    output = Path("/vol/runs") / run_id
    # Modal restarts a preempted container with the same input, and
    # train_recovery.py refuses to write into a non-empty output directory. Left
    # alone the two combine to kill the run permanently on the first preemption,
    # which is exactly how arm-extra was lost at step 94 of 636. The partial
    # attempt is moved aside rather than deleted so its logs remain inspectable.
    if output.exists() and not (output / "TRAIN_RESULT.json").is_file():
        import time as _time
        stale = output.with_name(f"{run_id}-partial-{int(_time.time())}")
        output.rename(stale)
        print(f"Preempted attempt moved aside to {stale}", flush=True)
        volume.commit()
    command = ["python", "/workspace/scripts/train_recovery.py", "--model", "/vol/models/qwen3-asr-1.7b",
               "--data", str(data), "--output", str(output), "--epochs", str(epochs), "--max-steps", str(max_steps),
               "--lr", str(lr), "--batch-size", str(batch_size), "--grad-acc", str(grad_acc),
               "--lora-rank", str(lora_rank), "--lora-alpha", str(lora_alpha),
               "--spec-augment", str(spec_augment), "--volume-name", VOLUME]
    if force_language:
        command.append("--force-language")
    if keep_epochs:
        command.append("--keep-epochs")
    for extra in (e for e in extra_data.split(",") if e):
        check_name(extra, "extra dataset name")
        command += ["--extra-data", f"/vol/data/{extra}"]
    if extra_data:
        command += ["--extra-fraction", str(extra_fraction)]
    print(f"Run {run_id}; validation={receipt['validation_rows']}; training rows={receipt['training_rows_with_repeats']}", flush=True)
    subprocess.run(command, check=True)
    volume.commit()
    return json.loads((output / "TRAIN_RESULT.json").read_text())


# Wave 0 answers, for one decode pass each, the questions that would otherwise
# cost a fine-tune: is Whisper actually better in domain, does forcing the
# competition language label help, and does beam search buy anything. The two
# controls reproduce numbers already measured by the trainer, so a harness bug
# shows up as a disagreement rather than as a wrong conclusion.
WAVE0 = [
    {"label": "w0-qwen-ft-greedy", "model": "runs/qwen-jav-holdout-v1/best-epoch-1"},
    {"label": "w0-qwen-base", "model": "models/qwen3-asr-1.7b"},
    {"label": "w0-qwen-ft-beam4", "model": "runs/qwen-jav-holdout-v1/best-epoch-1", "beams": 4},
    {"label": "w0-whisper-auto", "model": "models/whisper-large-v3"},
    {"label": "w0-whisper-routed", "model": "models/whisper-large-v3", "force_language": True,
     "language_map": '{"unknown": "jw", "javind": "id"}'},
    {"label": "w0-whisper-jw", "model": "models/whisper-large-v3", "force_language": True,
     "language_map": '{"unknown": "jw", "javind": "jw"}'},
]


@app.local_entrypoint()
def wave0(dataset: str = "recovery-jav1"):
    """Run every Wave 0 evaluation in parallel and print a ranked table."""
    import json
    handles = [(arm["label"], evaluate.spawn(dataset=dataset, **arm)) for arm in WAVE0]
    results = []
    for label, handle in handles:
        try:
            results.append(handle.get())
        except Exception as exc:  # one failed arm must not discard the others
            print(f"{label}: FAILED {exc!r}", flush=True)
    results.sort(key=lambda r: r["wer"])
    print(f"\n{'label':<24} {'WER':>9} {'javanese':>9} {'codeswitch':>11}  model")
    for r in results:
        groups = r["groups"]
        jav = groups.get("language:unknown", groups.get("language:jav", {})).get("wer")
        mix = groups.get("language:javind", {}).get("wer")
        print(f"{r['label']:<24} {r['wer']:>9.4f} "
              f"{jav if jav is None else f'{jav:>9.4f}'} "
              f"{mix if mix is None else f'{mix:>11.4f}'}  {r['model']}")
    print(json.dumps([{k: v for k, v in r.items() if k != "groups"} for r in results], indent=2))
    return results


# Every variant holds out the same 20 Jember sessions and the same development
# conversation as the measured 0.2685 baseline, so an arm differs from it only by
# the single factor under test.
BASELINE_PREP = ["--heldout", "5", "--style", "dev_style", "--dev-repeats", "8",
                 "--heldout-jember-sessions", "20", "--seed", "20260905"]
DURATION_PREP = ["--window-mode", "sliding", "--window-min", "18", "--window-max", "30",
                 "--window-stride", "0.5", "--short-fraction", "0.35"]
FILTER_PREP = ["--filter-flags", "mostly_silent,low_token_rate,invalid_timestamp"]
CONTEXT_PROMPT = ("Transcribe this Javanese-Indonesian voice note. "
                  "Speakers code-switch between Javanese and Indonesian.")

VARIANTS = {
    # The baseline recipe rebuilt here rather than locally. mp3 decoding differs
    # between ffmpeg builds -- the same validation window is 531666 bytes from the
    # macOS build and 531667 from the Linux one -- which moves baseline WER by
    # 0.0075. Every variant below is built on this machine, so this control, not
    # the locally built recovery-jav1, is what they must be compared against.
    "jav-control": [],
    "jav-dur": DURATION_PREP,
    "jav-lang": ["--language-tags"],
    "jav-filter": FILTER_PREP,
    "jav-prompt": ["--prompt", CONTEXT_PROMPT],
    "jav-combo": DURATION_PREP + FILTER_PREP + ["--language-tags"],
}

# Wave 1 isolates one factor per arm against the same holdout, plus a combined arm
# that tests whether the individual gains actually stack.
WAVE1 = [
    {"run_id": "arm-control", "dataset": "jav-control"},
    {"run_id": "arm-dur", "dataset": "jav-dur"},
    {"run_id": "arm-lang", "dataset": "jav-lang", "force_language": True},
    {"run_id": "arm-filter", "dataset": "jav-filter"},
    {"run_id": "arm-prompt", "dataset": "jav-prompt"},
    {"run_id": "arm-specaug", "dataset": "recovery-jav1", "spec_augment": 0.5},
    # The 1st-place Pasketti recipe: LoRA r=16/alpha=32 at lr 1e-4 with SpecAugment.
    {"run_id": "arm-lora", "dataset": "recovery-jav1", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 3, "keep_epochs": True},
    {"run_id": "arm-combo", "dataset": "jav-combo", "force_language": True},
    # 17.31 h of previously unused permissive audio: CC0 spontaneous Javanese plus
    # two CC-BY-SA Indonesian conversation corpora, against the same control.
    {"run_id": "arm-extra", "dataset": "jav-control", "extra_data": "external-v1"},
    {"run_id": "arm-kitchen", "dataset": "jav-combo", "force_language": True,
     "extra_data": "external-v1"},
    # Wave 1 moved every arm except the Javanese half, which is ~49% of the test
    # set. These add East Javanese read speech from the two dialects closest to
    # Jember. Read speech transferred negatively on its own once before
    # (OpenSLR-35, standard Central Javanese), so it is only ever a supplement.
    {"run_id": "arm-javtts", "dataset": "jav-control", "extra_data": "external-jav1"},
    {"run_id": "arm-alldata", "dataset": "jav-control",
     "extra_data": "external-v1,external-jav1"},
]


@app.local_entrypoint()
def build_external(name: str = "external-v1", corpora: str = "cv_jv,podcast_hari,podcast_homostoria",
                   stride: float = 1.0):
    """Build the external training corpus. Wraps the remote call because the CLI
    cannot parse a list-typed parameter on a directly invoked function."""
    import json
    receipt = prepare_external.remote(name, [c for c in corpora.split(",") if c], stride=stride)
    print(json.dumps({k: v for k, v in receipt.items() if k != "sources"}, indent=2))
    for source, detail in receipt["sources"].items():
        print(f"{source:<20} rows={detail['rows']:>6} hours={detail['hours']:>6.2f} "
              f"language={detail['language']} licence={detail['licence']}")
    return receipt


@app.local_entrypoint()
def prep_wave1(sources_sha256: str, only: str = ""):
    """Build every Wave 1 dataset variant on the volume, in parallel."""
    import json
    wanted = set(only.split(",")) if only else set(VARIANTS)
    handles = [(name, prepare.spawn(name, BASELINE_PREP + options, sources_sha256))
               for name, options in VARIANTS.items() if name in wanted]
    receipts = {}
    for name, handle in handles:
        try:
            receipts[name] = handle.get()
            r = receipts[name]
            print(f"{name}: train={r['training_rows_with_repeats']} "
                  f"({r['training_audio_hours_with_views_and_repeats']:.1f} h) "
                  f"validation={r['validation_rows']} short={r['jember_short']} "
                  f"long={r['jember_long']}", flush=True)
        except Exception as exc:
            print(f"{name}: FAILED {exc!r}", flush=True)
    print(json.dumps(receipts, indent=2))
    return receipts


@app.local_entrypoint()
def wave1(only: str = "", whisper: bool = True):
    """Launch every Wave 1 training arm in parallel and print a ranked table."""
    import json
    wanted = set(only.split(",")) if only else {a["run_id"] for a in WAVE1}
    handles = [(arm["run_id"], train.spawn(epochs=arm.pop("epochs", 1), **arm))
               for arm in (dict(a) for a in WAVE1) if arm["run_id"] in wanted]
    if whisper and (not only or "arm-whisper" in wanted):
        handles.append(("arm-whisper", train_whisper.spawn(run_id="arm-whisper",
                                                           dataset="recovery-jav1")))
    results = []
    for run_id, handle in handles:
        try:
            results.append({"run_id": run_id, **handle.get()})
        except Exception as exc:
            print(f"{run_id}: FAILED {exc!r}", flush=True)
    results.sort(key=lambda r: r["best_wer"])
    print(f"\n{'arm':<16} {'best WER':>9} {'baseline':>9} {'delta':>8}")
    for r in results:
        print(f"{r['run_id']:<16} {r['best_wer']:>9.4f} {r['baseline_wer']:>9.4f} "
              f"{r['best_wer'] - r['baseline_wer']:>+8.4f}")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def run(archive_sha256: str, run_id: str = "qwen-recovery-v1-fold5", epochs: int = 3,
        max_steps: int = -1, dataset: str = "recovery-v1"):
    print("Verified dataset staging:", stage_data.remote(archive_sha256, dataset=dataset))
    print("Pinned model staging:", stage_base.remote())
    print(train.remote(run_id=run_id, epochs=epochs, max_steps=max_steps, dataset=dataset))
