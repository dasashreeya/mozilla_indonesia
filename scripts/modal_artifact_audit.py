"""Measure the actual offline ZIP entrypoint on labelled validation audio."""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = os.environ.get("MODAL_VOLUME_NAME", "govind-llm-data")
app = modal.App(os.environ.get("MODAL_APP_NAME", "govind-llm"))
volume = modal.Volume.from_name(VOLUME)
train_image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .env({"PYTHONPATH": "/workspace/src", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8"})
    .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install("qwen-asr==0.0.6", "transformers==4.57.6", "accelerate==1.12.0",
                    "soundfile==0.14.0", "librosa==1.0.0", "numpy==2.3.5", "safetensors==0.8.0")
    .add_local_dir(ROOT / "src", "/workspace/src"))


@app.function(image=train_image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=2 * 3600, retries=0, single_use_containers=True)
def audit(archive: str = "submission-soup-div2", dataset: str = "ds-base",
          limit: int = 0, resubstitution: bool = False):
    import csv
    import hashlib
    import json
    import shutil
    import subprocess
    import time
    import zipfile
    from collections import defaultdict
    from dataclasses import asdict
    from mdc_asr.wer import ErrorCounts, word_errors

    for value in (archive, dataset):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value):
            raise ValueError("Unsafe artifact or dataset name")
    bundle = Path("/vol/artifacts") / f"{archive}.zip"
    package = Path("/tmp/artifact-audit/package")
    package.mkdir(parents=True)
    digest = hashlib.sha256()
    with bundle.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(bundle) as handle:
        for item in handle.infolist():
            if not (package / item.filename).resolve().is_relative_to(package):
                raise ValueError("Unsafe ZIP member")
        handle.extractall(package)
    config = json.loads((package / "config.json").read_text())
    source = Path("/vol/data") / dataset
    with (source / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "validation"]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No validation rows")
    data = package.parent / "data"
    (data / "clips").mkdir(parents=True)
    for row in rows:
        (data / "clips" / row["audio_filename"]).symlink_to(source / "audio" / row["audio_filename"])
    with (data / "test_metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_filename", "language", "file_duration_seconds"])
        writer.writeheader()
        writer.writerows({"audio_filename": r["audio_filename"], "language": r["language"],
                           "file_duration_seconds": r["duration_seconds"]} for r in rows)
    out = package.parent / "submission.csv"
    log = package.parent / "runtime.log"
    env = dict(os.environ, MDC_DATA_DIR=str(data), MDC_SUBMISSION_PATH=str(out),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    started = time.monotonic()
    with log.open("w") as handle:
        proc = subprocess.run(["python", "main.py"], cwd=package, env=env,
                              stdout=handle, stderr=subprocess.STDOUT, timeout=6900)
    seconds = time.monotonic() - started
    print(log.read_text()[-9000:], flush=True)
    if proc.returncode:
        raise RuntimeError(f"Packaged entrypoint failed: {proc.returncode}")
    with out.open(encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if [p["audio_filename"] for p in predictions] != [r["audio_filename"] for r in rows]:
        raise ValueError("Packaged output order/count mismatch")
    counts = ErrorCounts()
    groups = defaultdict(ErrorCounts)
    for row, prediction in zip(rows, predictions):
        error = word_errors(row["transcript"], prediction["transcript"])
        counts += error
        for key in ("language", "session", "source"):
            if row.get(key):
                groups[f"{key}:{row[key]}"] += error
    result = {"archive": archive, "archive_sha256": digest.hexdigest(), "config": config,
              "dataset": dataset, "clips": len(rows), "wer": counts.wer, **asdict(counts),
              "empty_predictions": sum(not p["transcript"].strip() for p in predictions),
              "seconds": seconds, "projected_2118_seconds": seconds * 2118 / len(rows),
              "measurement": "resubstitution; NOT held-out" if resubstitution else "held-out validation",
              "groups": {k: {"wer": v.wer, **asdict(v)} for k, v in sorted(groups.items())}}
    destination = Path("/vol/evals") / f"package-{archive}{'-smoke' if limit else ''}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2))
    shutil.copy2(out, destination.with_suffix(".predictions.csv"))
    shutil.copy2(log, destination.with_suffix(".log"))
    volume.commit()
    print(json.dumps(result, indent=2), flush=True)
    return result
