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
                               "numpy==2.5.2", "safetensors==0.8.0")
               .add_local_dir(ROOT / "src", "/workspace/src")
               .add_local_dir(ROOT / "vendor", "/workspace/vendor")
               .add_local_file(ROOT / "scripts/train_recovery.py", "/workspace/scripts/train_recovery.py"))


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


@app.function(image=train_image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=6 * 3600, retries=0, single_use_containers=True)
def train(run_id: str = "qwen-recovery-v1-fold5", epochs: int = 3, max_steps: int = -1,
          dataset: str = "recovery-v1"):
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
    command = ["python", "/workspace/scripts/train_recovery.py", "--model", "/vol/models/qwen3-asr-1.7b",
               "--data", str(data), "--output", str(output), "--epochs", str(epochs), "--max-steps", str(max_steps),
               "--volume-name", VOLUME]
    print(f"Run {run_id}; validation={receipt['validation_rows']}; training rows={receipt['training_rows_with_repeats']}", flush=True)
    subprocess.run(command, check=True)
    volume.commit()
    return json.loads((output / "TRAIN_RESULT.json").read_text())


@app.local_entrypoint()
def run(archive_sha256: str, run_id: str = "qwen-recovery-v1-fold5", epochs: int = 3,
        max_steps: int = -1, dataset: str = "recovery-v1"):
    print("Verified dataset staging:", stage_data.remote(archive_sha256, dataset=dataset))
    print("Pinned model staging:", stage_base.remote())
    print(train.remote(run_id=run_id, epochs=epochs, max_steps=max_steps, dataset=dataset))
