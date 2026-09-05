"""Modal pipeline for the MDC Indonesian-Javanese ASR track."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import modal


LOCAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_ROOT / "src"))
sys.path.insert(0, "/workspace/src")

from mdc_asr.hf_backup import DEFAULT_REPO_NAME, default_modal_items, upload_items  # noqa: E402
from mdc_asr.modal_utils import latest_checkpoint, rebase_qwen_jsonl, safe_volume_path  # noqa: E402


APP_NAME = "govind-llm"
VOLUME_NAME = "govind-llm-data"
VOLUME_ROOT = Path("/vol")
WORKSPACE = Path("/workspace")
DEFAULT_GPU = "A100-80GB"
DEFAULT_OPENSLR_SUBSET = "data/subsets/openslr35/shard0-3000-seed-20260905"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/openslr35-stage1"


app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


BASE_PACKAGES = [
    "build-essential",
    "cmake",
    "curl",
    "ffmpeg",
    "g++",
    "gcc",
    "git",
    "libsndfile1",
    "tzdata",
    "wget",
    "zip",
]


def _with_source(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir(LOCAL_ROOT / "configs", "/workspace/configs", copy=True)
        .add_local_dir(LOCAL_ROOT / "scripts", "/workspace/scripts", copy=True)
        .add_local_dir(LOCAL_ROOT / "src", "/workspace/src", copy=True)
        .add_local_dir(LOCAL_ROOT / "submission_src", "/workspace/submission_src", copy=True)
        .add_local_dir(LOCAL_ROOT / "vendor", "/workspace/vendor", copy=True)
    )


base_image = _with_source(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*BASE_PACKAGES)
    .env(
        {
            "PYTHONPATH": "/workspace/src:/workspace/scripts",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)

train_image = (
    base_image.uv_pip_install(
        "torch==2.11.0",
        "torchaudio==2.11.0",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .uv_pip_install(
        "accelerate==1.12.0",
        "datasets==5.0.1",
        "librosa==1.0.0",
        "peft==0.20.0",
        "qwen-asr==0.0.6",
        "safetensors==0.8.0",
        "scipy==1.18.0",
        "soundfile==0.14.0",
        "transformers==4.57.6",
    )
)

backup_image = base_image.uv_pip_install("huggingface_hub==1.30.0")
hf_backup_secret = modal.Secret.from_name("hf-backup-token", required_keys=["HF_TOKEN"])


def _run(command: List[str], env: Optional[Dict[str, str]] = None) -> None:
    printable = " ".join(command)
    print("running: {}".format(printable), flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    subprocess.run(command, cwd=str(WORKSPACE), env=merged_env, check=True)


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _receipt(path: Path) -> Dict[str, object]:
    return _read_json(path) if path.exists() else {"path": str(path), "exists": False}


@app.function(image=base_image, volumes={VOLUME_ROOT: volume}, timeout=4 * 60 * 60)
def stage_qwen(force: bool = False) -> Dict[str, object]:
    """Download and SHA-verify the pinned public Qwen3-ASR base model."""

    model_dir = VOLUME_ROOT / "models" / "qwen3-asr-1.7b"
    receipt = model_dir / "MODEL_RECEIPT.json"
    if receipt.exists() and not force:
        return {"status": "cached", "model_dir": str(model_dir), "receipt": _receipt(receipt)}
    _run(
        [
            "python",
            "/workspace/scripts/fetch_hf_model.py",
            "qwen3-asr-1.7b",
            "--output-dir",
            str(model_dir),
        ]
    )
    volume.commit()
    return {"status": "ready", "model_dir": str(model_dir), "receipt": _receipt(receipt)}


@app.function(image=base_image, volumes={VOLUME_ROOT: volume}, timeout=4 * 60 * 60)
def stage_openslr(
    max_utterances: int = 3000,
    validation_fraction: float = 0.1,
    seed: int = 20260905,
    force: bool = False,
) -> Dict[str, object]:
    """Download OpenSLR shard 0 and prepare a deterministic speaker-disjoint subset."""

    download_dir = VOLUME_ROOT / "data" / "downloads" / "openslr35"
    subset_dir = VOLUME_ROOT / "data" / "subsets" / "openslr35" / (
        "shard0-{}-seed-{}".format(max_utterances, seed)
    )
    manifest = subset_dir / "SUBSET_MANIFEST.json"
    if manifest.exists() and not force:
        return {"status": "cached", "subset_dir": str(subset_dir), "manifest": _receipt(manifest)}
    if subset_dir.exists() and force:
        shutil.rmtree(subset_dir)
    _run(["python", "/workspace/scripts/fetch_openslr35.py", "--output-dir", str(download_dir)])
    _run(
        [
            "python",
            "/workspace/scripts/prepare_openslr35.py",
            "--archive",
            str(download_dir / "asr_javanese_0.zip"),
            "--transcripts",
            str(download_dir / "utt_spk_text.tsv"),
            "--output",
            str(subset_dir),
            "--max-utterances",
            str(max_utterances),
            "--validation-fraction",
            str(validation_fraction),
            "--seed",
            str(seed),
        ]
    )
    volume.commit()
    return {"status": "ready", "subset_dir": str(subset_dir), "manifest": _receipt(manifest)}


@app.function(
    image=base_image,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=65536,
    timeout=12 * 60 * 60,
)
def stage_openslr_continuation(
    shards: str = "1-f",
    exclude_subset: str = "data/subsets/openslr35/shard0-11000-seed-20260905",
    output_dir: str = "data/subsets/openslr35/shards1-f-continuation",
    max_utterances: int = 0,
    force: bool = False,
) -> Dict[str, object]:
    """Stage additional SLR35 shards while excluding fixed validation speakers."""

    download_dir = VOLUME_ROOT / "data" / "downloads" / "openslr35"
    resolved_exclude = safe_volume_path(VOLUME_ROOT, exclude_subset)
    resolved_output = safe_volume_path(VOLUME_ROOT, output_dir)
    manifest = resolved_output / "SUBSET_MANIFEST.json"
    if manifest.exists() and not force:
        return {"status": "cached", "subset_dir": str(resolved_output), "manifest": _receipt(manifest)}
    if resolved_output.exists() and force:
        shutil.rmtree(resolved_output)
    _run(
        [
            "python",
            "/workspace/scripts/fetch_openslr35.py",
            "--output-dir",
            str(download_dir),
            "--shards",
            shards,
            "--workers",
            "4",
        ]
    )
    command = [
        "python",
        "/workspace/scripts/prepare_openslr35_continuation.py",
        "--archives-dir",
        str(download_dir),
        "--transcripts",
        str(download_dir / "utt_spk_text.tsv"),
        "--checksums",
        str(download_dir / "asr_javanese.sha256"),
        "--exclude-metadata",
        str(resolved_exclude / "metadata.csv"),
        "--output",
        str(resolved_output),
        "--shards",
        shards,
        "--max-utterances",
        str(max_utterances),
    ]
    _run(command)
    volume.commit()
    return {"status": "ready", "subset_dir": str(resolved_output), "manifest": _receipt(manifest)}


@app.function(
    image=train_image,
    gpu=DEFAULT_GPU,
    volumes={VOLUME_ROOT: volume},
    cpu=16,
    memory=131072,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=1, initial_delay=30.0),
    single_use_containers=True,
)
def train_qwen(
    model_path: str = "models/qwen3-asr-1.7b",
    subset_path: str = DEFAULT_OPENSLR_SUBSET,
    output_dir: str = DEFAULT_CHECKPOINT_ROOT,
    batch_size: int = 4,
    grad_acc: int = 8,
    epochs: float = 2.0,
    lr: float = 1e-5,
    save_strategy: str = "epoch",
    save_steps: int = 50,
    save_total_limit: int = 4,
    num_workers: int = 8,
    resume: bool = True,
) -> Dict[str, object]:
    """Fine-tune Qwen3-ASR on staged data and keep checkpoints on the Modal volume."""

    resolved_model = safe_volume_path(VOLUME_ROOT, model_path)
    resolved_subset = safe_volume_path(VOLUME_ROOT, subset_path)
    resolved_output = safe_volume_path(VOLUME_ROOT, output_dir)
    train_jsonl = resolved_subset / "train.qwen.jsonl"
    eval_jsonl = resolved_subset / "validation.qwen.jsonl"
    audio_dir = resolved_subset / "audio"
    if not (resolved_model / "config.json").is_file():
        raise FileNotFoundError("Stage Qwen first: {}".format(resolved_model))
    if not train_jsonl.is_file():
        raise FileNotFoundError("Stage/prepare training data first: {}".format(train_jsonl))
    if not audio_dir.is_dir():
        raise FileNotFoundError("Missing audio directory: {}".format(audio_dir))

    input_dir = resolved_output / "inputs"
    train_rebased = input_dir / "train.modal.jsonl"
    eval_rebased = input_dir / "validation.modal.jsonl"
    train_counts = rebase_qwen_jsonl(train_jsonl, train_rebased, audio_dir)
    eval_counts = rebase_qwen_jsonl(eval_jsonl, eval_rebased, audio_dir) if eval_jsonl.exists() else None
    resolved_output.mkdir(parents=True, exist_ok=True)
    with (resolved_output / "TRAIN_INPUT_RECEIPT.json").open("w", encoding="utf-8") as handle:
        json.dump({"train": train_counts, "eval": eval_counts}, handle, indent=2)
        handle.write("\n")

    command = [
        "python",
        "/workspace/vendor/qwen3_asr_sft.py",
        "--model_path",
        str(resolved_model),
        "--train_file",
        str(train_rebased),
        "--eval_file",
        str(eval_rebased if eval_counts else ""),
        "--output_dir",
        str(resolved_output),
        "--batch_size",
        str(batch_size),
        "--grad_acc",
        str(grad_acc),
        "--lr",
        str(lr),
        "--epochs",
        str(epochs),
        "--save_strategy",
        save_strategy,
        "--save_steps",
        str(save_steps),
        "--save_total_limit",
        str(save_total_limit),
        "--num_workers",
        str(num_workers),
        "--pin_memory",
        "1",
        "--persistent_workers",
        "1" if num_workers > 0 else "0",
        "--prefetch_factor",
        "2",
        "--resume",
        "1" if resume else "0",
    ]
    _run(command, env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    checkpoint = latest_checkpoint(resolved_output)
    if checkpoint is None:
        raise RuntimeError("Training finished but no checkpoint was saved under {}".format(resolved_output))
    receipt = {
        "status": "trained",
        "checkpoint": str(checkpoint),
        "checkpoint_root": str(resolved_output),
        "model_path": str(resolved_model),
        "train_counts": train_counts,
        "eval_counts": eval_counts,
        "gpu": DEFAULT_GPU,
    }
    with (resolved_output / "TRAIN_RECEIPT.json").open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    volume.commit()
    return receipt


@app.function(
    image=train_image,
    gpu=DEFAULT_GPU,
    volumes={VOLUME_ROOT: volume},
    cpu=12,
    memory=65536,
    timeout=4 * 60 * 60,
    single_use_containers=True,
)
def evaluate_qwen(
    model_path: str = "",
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    subset_path: str = DEFAULT_OPENSLR_SUBSET,
    split: str = "validation",
    language: str = "auto",
    batch_size: int = 24,
    limit: int = 0,
    output: str = "results/openslr35-stage1-validation.csv",
) -> Dict[str, object]:
    """Score a staged Qwen model on the staged OpenSLR validation split."""

    if model_path:
        resolved_model = safe_volume_path(VOLUME_ROOT, model_path)
    else:
        checkpoint = latest_checkpoint(safe_volume_path(VOLUME_ROOT, checkpoint_root))
        if checkpoint is None:
            raise FileNotFoundError("No checkpoint found under {}".format(checkpoint_root))
        resolved_model = checkpoint
    resolved_subset = safe_volume_path(VOLUME_ROOT, subset_path)
    resolved_output = safe_volume_path(VOLUME_ROOT, output)
    command = [
        "python",
        "/workspace/scripts/evaluate_qwen.py",
        "--model",
        str(resolved_model),
        "--metadata",
        str(resolved_subset / "metadata.csv"),
        "--audio-dir",
        str(resolved_subset / "audio"),
        "--output",
        str(resolved_output),
        "--split",
        split,
        "--language",
        language,
        "--device",
        "cuda:0",
        "--batch-size",
        str(batch_size),
    ]
    if limit > 0:
        command.extend(["--limit", str(limit)])
    _run(command, env={"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    volume.commit()
    return {
        "status": "evaluated",
        "model_path": str(resolved_model),
        "predictions": str(resolved_output),
        "split": split,
        "language": language,
    }


@app.function(image=base_image, volumes={VOLUME_ROOT: volume}, timeout=4 * 60 * 60)
def package_submission(
    model_path: str = "",
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    output: str = "artifacts/modal-qwen-openslr35-stage1-submission.zip",
) -> Dict[str, object]:
    """Build an offline MDC submission zip from a checkpoint or base model."""

    if model_path:
        resolved_model = safe_volume_path(VOLUME_ROOT, model_path)
    else:
        checkpoint = latest_checkpoint(safe_volume_path(VOLUME_ROOT, checkpoint_root))
        if checkpoint is None:
            raise FileNotFoundError("No checkpoint found under {}".format(checkpoint_root))
        resolved_model = checkpoint
    resolved_output = safe_volume_path(VOLUME_ROOT, output)
    _run(
        [
            "python",
            "/workspace/scripts/build_submission.py",
            "--source",
            "/workspace/submission_src",
            "--model-dir",
            str(resolved_model),
            "--output",
            str(resolved_output),
        ]
    )
    volume.commit()
    return {
        "status": "packaged",
        "model_path": str(resolved_model),
        "zip": str(resolved_output),
        "receipt": _receipt(Path(str(resolved_output) + ".json")),
    }


@app.function(image=base_image, volumes={VOLUME_ROOT: volume}, timeout=10 * 60)
def volume_status() -> Dict[str, object]:
    """Report what the Modal volume currently contains for this project."""

    paths = {
        "qwen_model": VOLUME_ROOT / "models" / "qwen3-asr-1.7b" / "MODEL_RECEIPT.json",
        "openslr_subset": VOLUME_ROOT / DEFAULT_OPENSLR_SUBSET / "SUBSET_MANIFEST.json",
        "train_receipt": VOLUME_ROOT / DEFAULT_CHECKPOINT_ROOT / "TRAIN_RECEIPT.json",
        "submission_zip": VOLUME_ROOT / "artifacts" / "modal-qwen-openslr35-stage1-submission.zip",
    }
    result: Dict[str, object] = {"volume": VOLUME_NAME}
    for name, path in paths.items():
        result[name] = {"path": str(path), "exists": path.exists()}
        if path.suffix == ".json" and path.exists():
            result[name]["receipt"] = _receipt(path)
        elif path.exists():
            result[name]["bytes"] = path.stat().st_size
    checkpoint = latest_checkpoint(VOLUME_ROOT / DEFAULT_CHECKPOINT_ROOT)
    result["latest_checkpoint"] = str(checkpoint) if checkpoint else None
    return result


@app.function(
    image=backup_image,
    volumes={VOLUME_ROOT: volume},
    secrets=[hf_backup_secret],
    timeout=6 * 60 * 60,
)
def backup_to_hf(
    repo_id: str = DEFAULT_REPO_NAME,
    checkpoint_root: str = DEFAULT_CHECKPOINT_ROOT,
    include_base_model: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Back up Modal checkpoints, zips, receipts, and results to private HF."""

    token = os.environ.get("HF_TOKEN", "")
    items = default_modal_items(
        VOLUME_ROOT,
        checkpoint_root=checkpoint_root,
        include_base_models=include_base_model,
    )
    return upload_items(
        token=token,
        repo_id=repo_id,
        items=items,
        private=True,
        commit_prefix="modal backup",
        dry_run=dry_run,
    )
