"""Run the real submission entrypoint on a GPU exactly as the runtime would.

Docker is not available locally, so this is the closest available stand-in for
the official offline container: same CUDA, same offline weight load, same
main.py, same environment variables, same CSV contract.

It deliberately includes one corrupt clip. The whole point of the hardening is
that a single unreadable file must not cost the other 2,117, so the test asserts
the run survives it, blanks only that row, and still emits a complete CSV.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "mozilla-indonesian-recovery"
app = modal.App("mozilla-indonesian-smoke")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg", "libsndfile1")
         .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
               "OMP_NUM_THREADS": "8"})
         .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0",
                         index_url="https://download.pytorch.org/whl/cu130")
         .uv_pip_install("accelerate==1.12.0", "librosa==1.0.0", "qwen-asr==0.0.6",
                         "soundfile==0.14.0", "transformers==4.57.6",
                         "numpy==2.5.2", "safetensors==0.8.0")
         .add_local_dir(ROOT / "submission_src", "/pkg"))


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=1800, retries=0, single_use_containers=True)
def smoke(checkpoint: str = "/vol/runs/qwen-recovery-v2-fold5/best-epoch-2",
          clips: int = 12):
    import csv
    import os
    import shutil
    import subprocess

    # The image mount is read-only; assemble the runnable package elsewhere.
    pkg = Path("/run/pkg")
    pkg.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "config.json"):
        shutil.copy(Path("/pkg") / name, pkg / name)
    if not (pkg / "model").exists():
        (pkg / "model").symlink_to(checkpoint)

    data = Path("/code_execution/data")
    (data / "clips").mkdir(parents=True, exist_ok=True)
    out = Path("/code_execution/submission/submission.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    source = Path("/vol/data/recovery-v1")
    with (source / "metadata.csv").open() as fh:
        validation = [r for r in csv.DictReader(fh) if r["split"] == "validation"]
    # Bias toward the longest clips: those are the ones a decoder is most
    # likely to truncate or choke on.
    validation.sort(key=lambda r: -float(r["duration_seconds"]))
    chosen = validation[:clips]
    for row in chosen:
        shutil.copy(source / "audio" / row["audio_filename"], data / "clips" / row["audio_filename"])

    corrupt = "zzz-corrupt-clip.mp3"
    (data / "clips" / corrupt).write_bytes(b"")  # 0 bytes: must not kill the run
    names = [r["audio_filename"] for r in chosen] + [corrupt]

    with (data / "test_metadata.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audio_filename"])
        writer.writeheader()
        for name in names:
            writer.writerow({"audio_filename": name})

    env = dict(os.environ, MDC_DATA_DIR=str(data), MDC_SUBMISSION_PATH=str(out))
    proc = subprocess.run(["python", "main.py"], cwd=str(pkg), env=env,
                          capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-4000:], flush=True)

    rows = list(csv.DictReader(out.open(encoding="utf-8-sig", newline="")))
    by_name = {r["audio_filename"]: r["transcript"] for r in rows}
    durations = {r["audio_filename"]: float(r["duration_seconds"]) for r in chosen}
    real = [n for n in names if n != corrupt]
    result = {
        "exit_code": proc.returncode,
        "rows": len(rows),
        "expected_rows": len(names),
        "columns_ok": list(rows[0].keys()) == ["audio_filename", "transcript"] if rows else False,
        "all_filenames_present": set(by_name) == set(names),
        "real_clips_non_empty": sum(1 for n in real if by_name.get(n, "").strip()),
        "real_clips_total": len(real),
        "corrupt_clip_blank": by_name.get(corrupt, None) == "",
        "longest_clip_seconds": max(durations.values()) if durations else None,
        "sample": {n: by_name.get(n, "")[:120] for n in real[:2]},
    }
    result["PASS"] = bool(
        result["exit_code"] == 0
        and result["rows"] == result["expected_rows"]
        and result["columns_ok"]
        and result["all_filenames_present"]
        and result["corrupt_clip_blank"]
        and result["real_clips_non_empty"] == result["real_clips_total"]
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(checkpoint: str = "/vol/runs/qwen-recovery-v2-fold5/best-epoch-2", clips: int = 12):
    print(json.dumps(smoke.remote(checkpoint=checkpoint, clips=clips), indent=2))
