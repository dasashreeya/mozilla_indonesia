"""Evaluate training-only KenLM decoding against saved held-out CTC emissions."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = os.environ.get("MODAL_VOLUME_NAME", "govind-llm-data")
app = modal.App(os.environ.get("MODAL_APP_NAME", "govind-llm"))
volume = modal.Volume.from_name(VOLUME)
KENLM_REVISION = "4cb443e60b7bf2c0ddf3c745378f76cb59e254e5"
image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential", "libboost-all-dev", "libeigen3-dev",
                 "zlib1g-dev", "libbz2-dev", "liblzma-dev")
    .uv_pip_install("numpy==1.26.4", "pyctcdecode==0.5.0", "kenlm==0.3.0")
    .run_commands("git clone https://github.com/kpu/kenlm.git /opt/kenlm",
                  f"git -C /opt/kenlm checkout --detach {KENLM_REVISION}",
                  "cmake -S /opt/kenlm -B /opt/kenlm/build -DCMAKE_BUILD_TYPE=Release",
                  "cmake --build /opt/kenlm/build --parallel 8")
    .env({"PYTHONPATH": "/workspace/src:/workspace/scripts"})
    .add_local_dir(ROOT / "src", "/workspace/src")
    .add_local_file(ROOT / "scripts/train_ctc.py", "/workspace/scripts/train_ctc.py"))


def safe_name(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in name):
        raise ValueError("Unsafe name")
    return name


@app.function(image=image, volumes={"/vol": volume}, cpu=8, memory=16384, timeout=1200)
def build(dataset: str = "ds-c1"):
    import hashlib
    import json
    import subprocess
    safe_name(dataset)
    directory = Path("/vol/data") / dataset
    text = directory / "lm_train.txt"
    output = directory / "lm"
    output.mkdir(exist_ok=True)
    arpa = output / "train-3gram.arpa"
    binary = output / "train-3gram.bin"
    subprocess.run(["/opt/kenlm/build/bin/lmplz", "--order", "3", "--text", str(text),
                    "--arpa", str(arpa), "--memory", "1G", "--discount_fallback"], check=True)
    subprocess.run(["/opt/kenlm/build/bin/build_binary", "trie", str(arpa), str(binary)], check=True)
    unigrams = sorted(set(text.read_text(encoding="utf-8").split()))
    (output / "unigrams.txt").write_text("\n".join(unigrams) + "\n", encoding="utf-8")
    receipt = {"training_text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
               "kenlm_revision": KENLM_REVISION, "order": 3, "words": len(unigrams),
               "uses_validation_text": False, "uses_test_text": False}
    (output / "RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    volume.commit()
    return receipt


@app.function(image=image, volumes={"/vol": volume}, cpu=16, memory=32768, timeout=2 * 3600)
def decode(run_id: str = "arm-c1", dataset: str = "ds-c1", alpha: float = 0.5,
           beta: float = 1.0, beam_width: int = 32, no_lm: bool = False):
    import csv
    import json
    import multiprocessing
    import time
    import numpy as np
    from pyctcdecode import build_ctcdecoder
    from train_ctc import summarize

    safe_name(run_id)
    safe_name(dataset)
    directory = Path("/vol/data") / dataset
    run = Path("/vol/runs") / run_id
    rows = json.loads((directory / "validation_rows.json").read_text())
    vocab = json.loads((run / "best/vocab.json").read_text())
    labels = [token for token, index in sorted(vocab.items(), key=lambda item: item[1])]
    labels = ["" if token == "[PAD]" else " " if token == "|" else
              "\u2047" if token == "[UNK]" else token for token in labels]
    with np.load(run / "best-emissions.npz", allow_pickle=False) as saved:
        logits = [saved[str(i)].astype(np.float32) for i in range(len(rows))]
    lm_path = None if no_lm else str(directory / "lm/train-3gram.bin")
    unigrams = None if no_lm else (directory / "lm/unigrams.txt").read_text(encoding="utf-8").splitlines()
    decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path, unigrams=unigrams,
                               alpha=alpha, beta=beta)
    started = time.monotonic()
    with multiprocessing.get_context("fork").Pool(8) as pool:
        texts = decoder.decode_batch(pool, logits, beam_width=beam_width)
    metrics = summarize(rows, texts)
    metrics.update({"run_id": run_id, "alpha": alpha, "beta": beta, "beam_width": beam_width,
                    "no_lm": no_lm, "seconds": time.monotonic() - started,
                    "checkpoint": json.loads((run / "BEST_MODEL.json").read_text())["step"]})
    label = f"ctc-{run_id}-{'plain' if no_lm else f'a{alpha}-b{beta}'}-beam{beam_width}"
    output = Path("/vol/evals") / label
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    with output.with_suffix(".predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_filename", "transcript"])
        writer.writeheader()
        writer.writerows({"audio_filename": row["audio_filename"], "transcript": text}
                         for row, text in zip(rows, texts))
    volume.commit()
    print(json.dumps({k: v for k, v in metrics.items() if k != "groups"}), flush=True)
    return metrics
