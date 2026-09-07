"""Bounded speech-model training on Modal.

Naming note. This runs in a shared Modal account, and the app name, volume name
and dataset directory names are the parts visible to anyone else with dashboard
access. They are deliberately generic -- `govind-llm`, `ds-a`, `arm-*` -- and
carry nothing about the task, corpus or competition. Keep it that way when
adding anything here: put the descriptive name in a comment, not in a string
Modal will render. Override with MODAL_APP_NAME / MODAL_VOLUME_NAME if the
account needs different ones.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = os.environ.get("MODAL_APP_NAME", "govind-llm")
VOLUME = os.environ.get("MODAL_VOLUME_NAME", "govind-llm-data")
app = modal.App(APP_NAME)
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

# Archives are uploaded to /vol/uploads/external/<corpus>/ and resolved by
# directory rather than by filename: MDC renames its downloads between fetches
# (the Lumajang archive arrived once as "1769864789237-TTS Javanese-Lumajang
# Dialect.tar.gz" and later as "tts-javanese-lumajang-dialect-ec830842.tar.gz"),
# and a hard-coded basename turns that into a FileNotFoundError mid-pipeline.
EXTERNAL_CORPORA = ("tts_lumajang", "tts_bojonegoro", "cv_jv", "cv_id_scripted",
                    "podcast_hari", "podcast_homostoria")


def resolve_external_archive(corpus: str, root="/vol/uploads/external"):
    """The single archive uploaded for `corpus`, whatever MDC happened to call it."""
    from pathlib import Path
    if corpus not in EXTERNAL_CORPORA:
        raise ValueError(f"Unknown corpus {corpus!r}")
    directory = Path(root) / corpus
    if not directory.is_dir():
        raise FileNotFoundError(f"Upload missing: {directory}")
    archives = sorted(p for p in directory.iterdir()
                      if p.is_file() and p.name.endswith((".tar.gz", ".tgz", ".tar")))
    if not archives:
        raise FileNotFoundError(f"No archive uploaded for {corpus} in {directory}")
    if len(archives) > 1:
        raise ValueError(f"{corpus} has {len(archives)} archives; expected one: "
                         f"{[p.name for p in archives]}")
    return archives[0]

# Only these preparation flags may be set remotely. Building the variants on the
# volume avoids re-uploading a ~2 GB prepared dataset per arm; the ~350 MB of
# sources is uploaded once instead.
PREP_FLAGS = {"--heldout", "--style", "--dev-repeats", "--heldout-jember-sessions", "--seed",
              "--window-mode", "--window-min", "--window-max", "--window-stride",
              "--short-fraction", "--language-tags", "--prompt", "--prompt-preset",
              "--filter-flags"}


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
                     style_tags: bool = False, prompt: str = "",
                     min_seconds: float = 18.0, max_seconds: float = 30.0,
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
        command += ["--archive", f"{corpus}={resolve_external_archive(corpus)}"]
    if language_tags:
        command.append("--language-tags")
    if style_tags:
        command.append("--style-tags")
    if prompt:
        command += ["--prompt", prompt]
    subprocess.run(command, check=True)
    volume.commit()
    receipt = json.loads((output / "PREPARATION_RECEIPT.json").read_text())
    return receipt


@app.function(image=stage_image, volumes={"/vol": volume}, timeout=3600, cpu=2, memory=8192)
def stage_data(archive_sha256: str, dataset: str = "ds-v1"):
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
def evaluate(label: str, model: str, dataset: str = "ds-w0", beams: int = 1,
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
def train_whisper(run_id: str, dataset: str = "ds-w0", epochs: int = 1,
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
              memory=65536, timeout=11 * 3600, retries=0, single_use_containers=True)
def train(run_id: str = "run-v1", epochs: int = 3, max_steps: int = -1,
          dataset: str = "ds-v1", lr: float = 1e-5, batch_size: int = 2, grad_acc: int = 8,
          lora_rank: int = 0, lora_alpha: int = 32, spec_augment: float = 0.0,
          force_language: bool = False, keep_epochs: bool = False,
          extra_data: str = "", extra_fraction: float = 1.0,
          evals_per_epoch: int = 1, patience: int = 2, final_fit: bool = False):
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
               "--spec-augment", str(spec_augment), "--volume-name", VOLUME,
               "--evals-per-epoch", str(evals_per_epoch), "--patience", str(patience)]
    if force_language:
        command.append("--force-language")
    if keep_epochs:
        command.append("--keep-epochs")
    if final_fit:
        command.append("--final-fit")
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
    {"label": "w0-qwen-ft-greedy", "model": "runs/run-w0/best-epoch-1"},
    {"label": "w0-qwen-base", "model": "models/qwen3-asr-1.7b"},
    {"label": "w0-qwen-ft-beam4", "model": "runs/run-w0/best-epoch-1", "beams": 4},
    {"label": "w0-whisper-auto", "model": "models/whisper-large-v3"},
    {"label": "w0-whisper-routed", "model": "models/whisper-large-v3", "force_language": True,
     "language_map": '{"unknown": "jw", "javind": "id"}'},
    {"label": "w0-whisper-jw", "model": "models/whisper-large-v3", "force_language": True,
     "language_map": '{"unknown": "jw", "javind": "jw"}'},
]


@app.local_entrypoint()
def wave0(dataset: str = "ds-w0"):
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
    "ds-w1-base": [],
    "ds-w1-dur": DURATION_PREP,
    "ds-w1-tag": ["--language-tags"],
    "ds-w1-filter": FILTER_PREP,
    "ds-w1-prompt": ["--prompt", CONTEXT_PROMPT],
    "ds-w1-combo": DURATION_PREP + FILTER_PREP + ["--language-tags"],
}

# Wave 1 isolates one factor per arm against the same holdout, plus a combined arm
# that tests whether the individual gains actually stack.
WAVE1 = [
    {"run_id": "arm-control", "dataset": "ds-w1-base"},
    {"run_id": "arm-dur", "dataset": "ds-w1-dur"},
    {"run_id": "arm-lang", "dataset": "ds-w1-tag", "force_language": True},
    {"run_id": "arm-filter", "dataset": "ds-w1-filter"},
    {"run_id": "arm-prompt", "dataset": "ds-w1-prompt"},
    {"run_id": "arm-specaug", "dataset": "ds-w0", "spec_augment": 0.5},
    # The 1st-place Pasketti recipe: LoRA r=16/alpha=32 at lr 1e-4 with SpecAugment.
    {"run_id": "arm-lora", "dataset": "ds-w0", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 3, "keep_epochs": True},
    {"run_id": "arm-combo", "dataset": "ds-w1-combo", "force_language": True},
    # 17.31 h of previously unused permissive audio: CC0 spontaneous Javanese plus
    # two CC-BY-SA Indonesian conversation corpora, against the same control.
    {"run_id": "arm-extra", "dataset": "ds-w1-base", "extra_data": "ext-a"},
    {"run_id": "arm-kitchen", "dataset": "ds-w1-combo", "force_language": True,
     "extra_data": "ext-a"},
    # Wave 1 moved every arm except the Javanese half, which is ~49% of the test
    # set. These add East Javanese read speech from the two dialects closest to
    # Jember. Read speech transferred negatively on its own once before
    # (OpenSLR-35, standard Central Javanese), so it is only ever a supplement.
    {"run_id": "arm-ttsx", "dataset": "ds-w1-base", "extra_data": "ext-b"},
    {"run_id": "arm-allx", "dataset": "ds-w1-base",
     "extra_data": "ext-a,ext-b"},
]


@app.local_entrypoint()
def build_external(name: str = "ext-a", corpora: str = "cv_jv,podcast_hari,podcast_homostoria",
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
                                                           dataset="ds-w0")))
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
def run(archive_sha256: str, run_id: str = "run-v1", epochs: int = 3,
        max_steps: int = -1, dataset: str = "ds-v1"):
    print("Verified dataset staging:", stage_data.remote(archive_sha256, dataset=dataset))
    print("Pinned model staging:", stage_base.remote())
    print(train.remote(run_id=run_id, epochs=epochs, max_steps=max_steps, dataset=dataset))


# ---------------------------------------------------------------------------
# Wave 2: the depth path. One base model, trained properly, checkpoints
# averaged. See docs/kb/11-audit-and-methodology-v2.md section 0 for why this
# and not a second architecture.
#
# Names here are the ones a shared dashboard renders, so they say nothing about
# the task: `ds-*` for prepared data, `arm-*` for runs, preset names instead of
# prompt text.
# ---------------------------------------------------------------------------

# Held-out preparation, identical to every measured arm so the comparison holds.
W2_PREP = ["--heldout", "5", "--style", "dev_style", "--dev-repeats", "8",
           "--heldout-jember-sessions", "20", "--seed", "20260905"]
# The last run only: no holdout at all. Every labelled row trains, including the
# 294 development clips and 20 sessions each earlier arm had to withhold.
W2_FULL_PREP = ["--heldout", "none", "--style", "dev_style", "--dev-repeats", "8",
                "--heldout-jember-sessions", "0", "--seed", "20260905"]

W2_VARIANTS = {
    "ds-base": W2_PREP + ["--language-tags"],
    "ds-prompt": W2_PREP + ["--language-tags", "--prompt-preset", "codeswitch"],
    "ds-full": W2_FULL_PREP + ["--language-tags", "--prompt-preset", "codeswitch"],
}

# One external corpus, every Javanese source available, tagged by transcription
# convention rather than by language so its orthography trains the encoder
# without reaching the decoder.
#
# The two Indonesian podcast corpora are not here: they are not downloaded, and
# they were the source of the only positive signal the previous wave got on the
# code-switched half. If they are fetched later, add them as a second corpus
# rather than folding them into this one -- they need their own `Indpod` tag,
# and mixing registers under one name defeats the point of style conditioning.
W2_EXTERNAL = {
    "ext-b": ["tts_lumajang", "tts_bojonegoro", "cv_jv"],
}

# The 1st-place reference recipe and the variations around it that were never
# measured. `epochs` and `keep_epochs` matter as much as the learning rate: with
# one checkpoint there is nothing to average, and averaging is what separated
# 1st from 3rd in the reference competition.
WAVE2 = [
    {"run_id": "arm-lora", "dataset": "ds-base", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 3, "keep_epochs": True, "force_language": True},
    {"run_id": "arm-lora-long", "dataset": "ds-base", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 6, "keep_epochs": True, "force_language": True},
    {"run_id": "arm-lr3", "dataset": "ds-base", "lr": 3e-5, "epochs": 3,
     "keep_epochs": True, "force_language": True},
    {"run_id": "arm-prompt", "dataset": "ds-prompt", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 3, "keep_epochs": True, "force_language": True},
    {"run_id": "arm-extb", "dataset": "ds-base", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 3, "keep_epochs": True, "force_language": True,
     "extra_data": "ext-b"},
    # Same external audio at a third of the weight. Read speech has never helped
    # this task on its own, so the mixture ratio is itself under test.
    {"run_id": "arm-extb-lo", "dataset": "ds-base", "lora_rank": 16, "lr": 1e-4,
     "spec_augment": 0.5, "epochs": 3, "keep_epochs": True, "force_language": True,
     "extra_data": "ext-b", "extra_fraction": 0.33},
]


plot_image = (base.uv_pip_install("matplotlib==3.10.6", "numpy==2.5.2")
              .add_local_file(ROOT / "scripts/plot_runs.py", "/workspace/scripts/plot_runs.py"))

hf_image = (base.uv_pip_install("huggingface_hub==1.30.0")
            .add_local_dir(ROOT / "src", "/workspace/src"))


@app.function(image=plot_image, volumes={"/vol": volume}, timeout=1800, cpu=4, memory=16384)
def plots(minority_share: float = 0.49, session_label: str = ""):
    """Render the four decision figures from whatever the volume already holds."""
    import json
    import subprocess
    from pathlib import Path
    command = ["python", "/workspace/scripts/plot_runs.py", "--root", "/vol",
               "--output", "/vol/plots", "--test-jav-share", str(minority_share)]
    if session_label:
        check_name(session_label, "session label")
        command += ["--session-label", session_label]
    subprocess.run(command, check=True)
    volume.commit()
    return {"plots": sorted(p.name for p in Path("/vol/plots").iterdir())}


# Everything except model weights: receipts, metrics, predictions, figures. Small
# enough to push after every wave, which is the point -- the expensive thing to
# lose is the measurement, not the checkpoint.
RESULT_PATTERNS = ("plots", "evals", "experiments")
RUN_RESULT_SUFFIXES = (".json", ".csv")


@app.function(image=hf_image, volumes={"/vol": volume}, timeout=4 * 3600, cpu=4,
              memory=16384, secrets=[modal.Secret.from_name("hf-token")])
def push_hf(repo_id: str = "", weights: list[str] = None, private: bool = True,
            dry_run: bool = False):
    """Back the volume up to a private Hugging Face repo.

    Results always go: figures, evaluation JSONs, predictions, and every run's
    receipts. Weights go only when named in `weights`, because a run directory
    holds several multi-gigabyte checkpoints and pushing them all by default
    would turn a routine backup into an hour of upload.
    """
    import json
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, "/workspace/src")
    from mdc_asr.hf_backup import BackupItem, DEFAULT_REPO_NAME, upload_items

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise ValueError("HF_TOKEN missing; create the Modal secret with "
                         "`modal secret create hf-token HF_TOKEN=...`")
    root = Path("/vol")
    items = []
    for relative in RESULT_PATTERNS:
        path = root / relative
        if path.exists():
            items.append(BackupItem(path, relative))
    for run in sorted((root / "runs").glob("*")):
        if not run.is_dir():
            continue
        for path in sorted(run.iterdir()):
            if path.is_file() and path.suffix in RUN_RESULT_SUFFIXES:
                items.append(BackupItem(path, f"runs/{run.name}/{path.name}"))
    for relative in weights or []:
        if ".." in relative:
            raise ValueError(f"Unsafe weights path: {relative}")
        path = root / relative
        if not path.is_dir():
            raise FileNotFoundError(f"No such checkpoint on the volume: {relative}")
        items.append(BackupItem(path, relative))
    result = upload_items(token=token, repo_id=repo_id or DEFAULT_REPO_NAME,
                          items=items, private=private, commit_prefix="modal run",
                          dry_run=dry_run)
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, indent=2))
    return result


def _language_wers(metrics, test_jav_share=0.49):
    """Javanese WER, non-Javanese WER, and the two reweighted to the test set.

    The fold is 37.96% Javanese by reference words and the test set is ~49%, so
    ranking on the fold mean systematically under-weights the half that carries
    most of the error.
    """
    groups = metrics.get("groups") or {}
    jav = groups.get("language:jav")
    others = [v for k, v in groups.items()
              if k.startswith("language:") and k != "language:jav"]
    if not jav or not others:
        return None, None, metrics.get("wer")
    total = {"substitutions": 0, "deletions": 0, "insertions": 0, "reference_words": 0}
    for part in others:
        for key in total:
            total[key] += part.get(key, 0)
    other_wer = ((total["substitutions"] + total["deletions"] + total["insertions"])
                 / total["reference_words"]) if total["reference_words"] else None
    if other_wer is None:
        return jav["wer"], None, metrics.get("wer")
    return jav["wer"], other_wer, test_jav_share * jav["wer"] + (1 - test_jav_share) * other_wer


@app.local_entrypoint()
def stage0(model: str = "runs/arm-control/best-epoch-1", dataset: str = "ds-base"):
    """Fix the instrument before spending a GPU-hour on anything else.

    Scores the current best checkpoint on the fold with per-session grouping,
    which is what surfaces training sessions carrying session 35's timestamp
    misalignment, and re-anchors the local-to-public calibration against a model
    whose public score is actually known.
    """
    import json
    result = evaluate.remote(label="s0-current", model=model, dataset=dataset,
                             force_language=True)
    jav, other, weighted = _language_wers(result)
    sessions = sorted(((k.split(":", 1)[1], v["wer"], v["reference_words"])
                       for k, v in (result.get("groups") or {}).items()
                       if k.startswith("session:")), key=lambda t: -t[1])
    print(f"\nfold {result['wer']:.4f}  javanese {jav}  code-switched {other}")
    if weighted is not None:
        print(f"test-weighted {weighted:.4f}  forecast public {weighted + 0.052:.4f}")
    if sessions:
        values = sorted(v for _, v, _ in sessions)
        p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
        print(f"\n{len(sessions)} sessions, median {values[len(values) // 2]:.4f}, p90 {p90:.4f}")
        print("worst sessions -- inspect these for timestamp lag before training on them:")
        for name, value, words in sessions[:8]:
            flag = "  <-- suspect" if value > p90 else ""
            print(f"  session {name:<10} {value:.4f}  ({words} words){flag}")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))
    return result


@app.local_entrypoint()
def prep2(sources_sha256: str, only: str = ""):
    """Build every Wave 2 dataset variant and both external corpora."""
    import json
    wanted = set(only.split(",")) if only else set(W2_VARIANTS) | set(W2_EXTERNAL)
    handles = [(name, prepare.spawn(name, options, sources_sha256))
               for name, options in W2_VARIANTS.items() if name in wanted]
    handles += [(name, prepare_external.spawn(name, corpora, language_tags=True,
                                              style_tags=True, stride=1.0))
                for name, corpora in W2_EXTERNAL.items() if name in wanted]
    receipts = {}
    for name, handle in handles:
        try:
            receipts[name] = handle.get()
            rows = receipts[name].get("training_rows_with_repeats")
            hours = receipts[name].get("training_audio_hours_with_views_and_repeats")
            print(f"{name:<12} rows={rows} hours={hours}", flush=True)
        except Exception as exc:  # one failed variant must not discard the rest
            print(f"{name}: FAILED {exc!r}", flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sources"}
                      for k, v in receipts.items()}, indent=2))
    return receipts


@app.local_entrypoint()
def wave2(only: str = "", evals_per_epoch: int = 4, patience: int = 8):
    """Run the Wave 2 arms in parallel and rank them on the test weighting."""
    import json
    wanted = set(only.split(",")) if only else {a["run_id"] for a in WAVE2}
    handles = []
    for arm in (dict(a) for a in WAVE2):
        if arm["run_id"] not in wanted:
            continue
        handles.append((arm["run_id"], train.spawn(
            epochs=arm.pop("epochs", 3), evals_per_epoch=evals_per_epoch,
            patience=patience, **arm)))
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
def soup_best(label: str = "soup-v1", runs: str = "", top: int = 4,
              dataset: str = "ds-base"):
    """Average the best checkpoints across runs, then score the average.

    Selection is by held-out WER over every kept checkpoint of every named run,
    which is why the arms run with --keep-epochs and --evals-per-epoch: three
    epoch-end checkpoints from one run are too few and too correlated to make a
    useful soup.
    """
    import json
    names = [r for r in runs.split(",") if r] or [a["run_id"] for a in WAVE2]
    # Select first, merge second. Merging every adapter a run kept would be ~12
    # CPU merges per run of a 1.7B model -- hours of work to throw most of it
    # away. Scanning the recorded WERs is free, so only the checkpoints that
    # actually make the soup get merged.
    candidates = scan_checkpoints.remote(names)
    if not candidates:
        raise SystemExit("No scored checkpoints found; did the arms write BEST_MODEL.json?")
    chosen = candidates[:top]
    print(f"Selected {len(chosen)} of {len(candidates)} checkpoints:")
    for wer, run, label in chosen:
        print(f"  {wer:.4f}  {run}/{label}")
    by_run = {}
    for _, run, label in chosen:
        by_run.setdefault(run, []).append(label)
    merge_args = list(by_run.items())
    for (run, _), done in zip(merge_args, merge_adapters.starmap(merge_args)):
        if done.get("merged"):
            print(f"{run}: {len(done['merged'])} adapter(s) merged", flush=True)
    picked = collect_checkpoints.remote(names, top)
    if len(picked) < 2:
        raise SystemExit(f"Need at least two checkpoints to average; found {picked}")
    print("Averaging:\n  " + "\n  ".join(picked))
    receipt = soup.remote(label=label, checkpoints=picked)
    print(json.dumps(receipt, indent=2))
    result = evaluate.remote(label=f"eval-{label}", model=f"soups/{label}",
                             dataset=dataset, force_language=True)
    jav, other, weighted = _language_wers(result)
    print(f"\n{label}: fold {result['wer']:.4f}  javanese {jav}  code-switched {other}")
    if weighted is not None:
        print(f"test-weighted {weighted:.4f}  forecast public {weighted + 0.052:.4f}")
    return {"soup": receipt, "eval": result}


def _is_full_model(path):
    """A directory a Qwen3ASRModel can be loaded from, as opposed to an adapter.

    With --lora-rank, every intermediate checkpoint is a PEFT adapter -- an
    `adapter_model.safetensors` and no `config.json`. Averaging those together
    is a legitimate technique, but the result cannot be evaluated or packaged
    without a base model to merge into, and model_soup.py would happily average
    an adapter with a full model and produce nonsense. So the pool is restricted
    to real models, and `merge_adapters` below turns adapters into them.
    """
    return (path / "config.json").is_file() and any(path.glob("*.safetensors"))


@app.function(image=train_image, volumes={"/vol": volume}, timeout=2 * 3600,
              cpu=8, memory=131072)
def merge_adapters(run: str, labels: list[str] = None):
    """Merge a run's LoRA adapters into the base, so each becomes a full model.

    train_recovery.py merges only the single best epoch. Souping wants several
    checkpoints per run, so this merges the rest on CPU after the fact.
    """
    import json
    import sys
    import torch
    from pathlib import Path
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    from transformers import AutoProcessor
    sys.path.insert(0, "/workspace/scripts")
    from train_recovery import sanitize_generation_config
    check_name(run, "run ID")
    directory = Path("/vol/runs") / run
    adapters = [d for d in sorted(directory.iterdir())
                if d.is_dir() and (d / "adapter_config.json").is_file()
                and (not labels or d.name in labels)]
    if not adapters:
        return {"run": run, "merged": [], "note": "no adapters; nothing to merge"}
    processor = AutoProcessor.from_pretrained("/vol/models/qwen3-asr-1.7b",
                                              local_files_only=True)
    merged = []
    for adapter_dir in adapters:
        target = directory / f"merged-{adapter_dir.name.split('-', 1)[1]}"
        if _is_full_model(target):
            merged.append(str(target))
            continue
        base = Qwen3ASRModel.from_pretrained("/vol/models/qwen3-asr-1.7b",
               dtype=torch.bfloat16, device_map="cpu", local_files_only=True).model
        merged_model = sanitize_generation_config(
            PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload())
        merged_model.save_pretrained(target, safe_serialization=True)
        processor.save_pretrained(target)
        del base
        merged.append(str(target))
        print(f"Merged {adapter_dir.name} -> {target.name}", flush=True)
        volume.commit()
    volume.commit()
    return {"run": run, "merged": merged}


@app.function(image=stage_image, volumes={"/vol": volume}, timeout=900, cpu=2, memory=4096)
def scan_checkpoints(runs: list[str]):
    """Every recorded checkpoint as (wer, run, label), best first.

    Reads only BEST_MODEL.json, so it costs nothing and works whether the
    checkpoint on disk is a merged model or a bare LoRA adapter. Selection
    happens against this list; merging happens afterwards, for the winners only.
    """
    import json
    from pathlib import Path
    found = []
    for run in runs:
        check_name(run, "run ID")
        directory = Path("/vol/runs") / run
        best = directory / "BEST_MODEL.json"
        if not best.is_file():
            continue
        for entry in json.loads(best.read_text()).get("epochs") or []:
            wer, label = entry.get("wer"), entry.get("label")
            if wer is None or not label:
                continue
            if any((directory / f"{p}-{label}").is_dir()
                   for p in ("merged", "weights", "best")):
                found.append((wer, run, label))
    found.sort()
    return found


@app.function(image=stage_image, volumes={"/vol": volume}, timeout=900, cpu=2, memory=4096)
def collect_checkpoints(runs: list[str], top: int = 4):
    """Volume-relative paths of the `top` best-scoring loadable checkpoints.

    Scored by the WER recorded for that checkpoint's label, so the pool is the
    genuinely best points across every run rather than one per run.
    """
    import json
    from pathlib import Path
    scored = []
    for run in runs:
        check_name(run, "run ID")
        directory = Path("/vol/runs") / run
        best = directory / "BEST_MODEL.json"
        if not best.is_file():
            continue
        history = json.loads(best.read_text()).get("epochs") or []
        for entry in history:
            wer, label = entry.get("wer"), entry.get("label")
            if wer is None or not label:
                continue
            for name in (f"merged-{label}", f"weights-{label}", f"best-{label}"):
                candidate = directory / name
                if candidate.is_dir() and _is_full_model(candidate):
                    scored.append((wer, f"runs/{run}/{name}"))
                    break
    scored.sort()
    return [path for _, path in scored[:top]]


@app.local_entrypoint()
def final_fit(run_id: str = "arm-final", dataset: str = "ds-full", steps: int = 0,
              lora_rank: int = 16, lr: float = 1e-4, extra_data: str = "ext-b"):
    """The last run: every labelled row, no holdout, no checkpoint selection.

    `steps` is the optimizer-step count the validated arms chose. It is required
    because there is nothing to early-stop on here -- pass the step at which the
    best arm's WER bottomed out, scaled for the larger row count.
    """
    import json
    if steps <= 0:
        raise SystemExit("Pass --steps: the step count the validated arms selected. "
                         "Read it from the winning arm's BEST_MODEL.json history.")
    result = train.remote(run_id=run_id, dataset=dataset, max_steps=steps, epochs=99,
                          lora_rank=lora_rank, lr=lr, spec_augment=0.5,
                          force_language=True, keep_epochs=True, final_fit=True,
                          extra_data=extra_data, evals_per_epoch=0)
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def report(repo_id: str = "", weights: str = "", minority_share: float = 0.49,
           session_label: str = "", dry_run: bool = False):
    """Render the figures, then back results -- and any named weights -- to HF."""
    import json
    print(json.dumps(plots.remote(minority_share=minority_share,
                                  session_label=session_label), indent=2))
    pushed = push_hf.remote(repo_id=repo_id,
                            weights=[w for w in weights.split(",") if w],
                            dry_run=dry_run)
    print(f"Backed up to {pushed.get('repo_id')}")
    return pushed


package_image = (base.uv_pip_install("numpy==2.5.2")
                 .add_local_dir(ROOT / "submission_src", "/workspace/submission_src")
                 .add_local_file(ROOT / "scripts/build_submission.py",
                                 "/workspace/scripts/build_submission.py"))


@app.function(image=package_image, volumes={"/vol": volume}, timeout=2 * 3600,
              cpu=8, memory=65536)
def package(model: str, label: str = "submission", prompt: str = "",
            language_routing: bool = True, batch_size: int = 24):
    """Build the offline submission ZIP on the volume.

    Packaged here rather than locally so the weights never round-trip: the model
    is already on the volume, and only the finished archive comes down.

    `prompt` must match what the packaged model trained with, byte for byte. The
    fine-tune learns to emit `language <Tag><asr_text>...`, and qwen-asr builds
    the decode prefix from the same string, so a mismatch silently costs the
    conditioning the arm was trained for.
    """
    import json
    import subprocess
    from pathlib import Path
    check_name(label, "label")
    if ".." in model:
        raise ValueError(f"Unsafe model path: {model}")
    model_path = Path("/vol") / model
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Not a packageable model directory: {model}")

    # submission_src ships from the image read-only, so config.json is written
    # into a working copy rather than edited in place.
    source = Path("/tmp/submission_src")
    if source.exists():
        subprocess.run(["rm", "-rf", str(source)], check=True)
    subprocess.run(["cp", "-r", "/workspace/submission_src", str(source)], check=True)
    config = json.loads((source / "config.json").read_text())
    config.update({"batch_size": batch_size, "language_routing": language_routing,
                   "prompt": prompt})
    (source / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    output = Path("/vol/artifacts") / f"{label}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["python", "/workspace/scripts/build_submission.py",
                    "--source", str(source), "--model-dir", str(model_path),
                    "--output", str(output)], check=True)
    volume.commit()
    receipt = json.loads(output.with_suffix(".zip.json").read_text())
    receipt["config"] = config
    receipt["model"] = model
    print(json.dumps(receipt, indent=2))
    return receipt
