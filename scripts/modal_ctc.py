"""Bounded complementary acoustic-model experiment on the existing volume."""
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = os.environ.get("MODAL_VOLUME_NAME", "govind-llm-data")
app = modal.App(os.environ.get("MODAL_APP_NAME", "govind-llm"))
volume = modal.Volume.from_name(VOLUME)
image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .env({"PYTHONPATH": "/workspace/src", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8"})
    .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("transformers==4.57.6", "datasets==5.0.1", "accelerate==1.12.0",
                    "soundfile==0.14.0", "librosa==1.0.0", "numpy==2.3.5", "safetensors==0.8.0")
    .add_local_dir(ROOT / "src", "/workspace/src")
    .add_local_file(ROOT / "scripts/train_ctc.py", "/workspace/scripts/train_ctc.py"))


def check_name(value):
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
        raise ValueError("Unsafe name")
    return value


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=32768, timeout=2 * 3600)
def prepare(dataset: str = "ds-base", name: str = "ds-c1"):
    import subprocess
    from huggingface_hub import HfApi, snapshot_download

    check_name(dataset)
    check_name(name)
    base = Path("/vol/models/encoder-c1")
    receipt = base / "MODEL_RECEIPT.json"
    if not receipt.exists():
        info = HfApi().model_info("facebook/w2v-bert-2.0")
        revision = info.sha
        snapshot_download("facebook/w2v-bert-2.0", revision=revision, local_dir=str(base),
                          allow_patterns=["*.json", "*.safetensors", "README.md"])
        receipt.write_text(json.dumps({"repo_id": "facebook/w2v-bert-2.0", "revision": revision,
                                       "license": "MIT"}, indent=2))
        volume.commit()
    output = Path("/vol/data") / name
    subprocess.run(["python", "/workspace/scripts/train_ctc.py", "prepare", "--model", str(base),
                    "--data", f"/vol/data/{dataset}", "--output", str(output)], check=True)
    volume.commit()
    return json.loads((output / "PREPARATION_RECEIPT.json").read_text())


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=6 * 3600, retries=0, single_use_containers=True)
def train(run_id: str = "arm-c1", dataset: str = "ds-c1", epochs: int = 8,
          max_steps: int = -1, lr: float = 5e-5, resume: bool = False):
    import subprocess
    check_name(run_id)
    check_name(dataset)
    output = Path("/vol/runs") / run_id
    command = ["python", "/workspace/scripts/train_ctc.py", "train",
               "--model", "/vol/models/encoder-c1", "--data", f"/vol/data/{dataset}",
               "--output", str(output), "--epochs", str(epochs), "--lr", str(lr),
               "--max-steps", str(max_steps), "--volume-name", VOLUME]
    if resume:
        command.append("--resume")
    subprocess.run(command, check=True)
    volume.commit()
    return json.loads((output / "TRAIN_RESULT.json").read_text())


@app.local_entrypoint()
def run(run_id: str = "arm-c1", dataset: str = "ds-c1", epochs: int = 8,
        max_steps: int = -1, lr: float = 5e-5, build: bool = False, resume: bool = False):
    if build:
        print(json.dumps(prepare.remote(name=dataset), indent=2))
    print(json.dumps(train.remote(run_id=run_id, dataset=dataset, epochs=epochs,
                                  max_steps=max_steps, lr=lr, resume=resume), indent=2))
