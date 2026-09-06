"""Test whether Javanese-only speech is what the leaderboard gap is made of.

Every WER this project has measured came from dev clips, and dev is 94.6%
code-switched `javind` with ZERO `jav` clips. The test set carries a `language`
column with three values, so it almost certainly contains Javanese-only audio
that dev cannot measure at all.

Base Qwen has never seen Jember, and Jember is spontaneous Javanese in the
target Pandhalungan dialect, so scoring base Qwen there is uncontaminated. The
comparison is duration-matched on purpose: Jember `long` windows run 18-28s
against dev's 26.2s median, so duration cannot explain a difference.

Prediction if the hypothesis holds: base Qwen scores ~0.27 on dev javind and
far worse on Jember Javanese. The epoch-2 number is reported too but IS
contaminated -- that checkpoint trained on these segments -- so it is a floor,
not a measurement.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "mozilla-indonesian-recovery"
app = modal.App("mozilla-indonesian-javprobe")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("ffmpeg", "libsndfile1")
         .env({"TOKENIZERS_PARALLELISM": "false", "HF_HUB_DISABLE_TELEMETRY": "1",
               "PYTHONPATH": "/workspace/src", "OMP_NUM_THREADS": "8"})
         .uv_pip_install("torch==2.11.0", "torchaudio==2.11.0",
                         index_url="https://download.pytorch.org/whl/cu130")
         .uv_pip_install("accelerate==1.12.0", "librosa==1.0.0", "qwen-asr==0.0.6",
                         "soundfile==0.14.0", "transformers==4.57.6",
                         "numpy==2.5.2", "safetensors==0.8.0")
         .add_local_dir(ROOT / "src", "/workspace/src"))

MODELS = [
    ("base", "/vol/models/qwen3-asr-1.7b", "clean: base never saw Jember"),
    ("epoch2", "/vol/runs/qwen-recovery-v2-fold5/best-epoch-2", "CONTAMINATED: trained on these"),
]


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=2400, retries=0, single_use_containers=True)
def probe(sample: int = 350, seed: int = 20260905):
    import csv
    import random
    import time
    from dataclasses import asdict

    import torch
    from qwen_asr import Qwen3ASRModel
    from mdc_asr.wer import ErrorCounts, word_errors

    data = Path("/vol/data/recovery-v1")
    out = Path("/vol/runs/javanese-probe-v1")
    out.mkdir(parents=True, exist_ok=True)

    with (data / "metadata.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    jember = [r for r in rows if r["source"] == "jember" and r["view"] == "long"]
    random.Random(seed).shuffle(jember)
    jember = jember[:sample]
    durations = sorted(float(r["duration_seconds"]) for r in jember)
    print(f"Jember long windows: n={len(jember)} "
          f"duration min={durations[0]:.1f} median={durations[len(durations)//2]:.1f} "
          f"max={durations[-1]:.1f}", flush=True)

    results = {"sample": len(jember),
               "duration_median": durations[len(durations) // 2],
               "models": {}}
    for label, path, note in MODELS:
        wrapper = Qwen3ASRModel.from_pretrained(
            path, dtype=torch.bfloat16, device_map="cuda:0",
            max_inference_batch_size=24, max_new_tokens=512, local_files_only=True)
        model = wrapper.model
        for flag in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
            if not model.generation_config.do_sample and getattr(model.generation_config, flag, None) is not None:
                setattr(model.generation_config, flag, None)
        model.eval()

        aggregate = ErrorCounts()
        preds, empty = [], 0
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(jember), 24):
                batch = jember[start:start + 24]
                outputs = wrapper.transcribe(
                    audio=[str(data / "audio" / r["audio_filename"]) for r in batch],
                    language=None)
                for row, output in zip(batch, outputs):
                    prediction = " ".join((output.text or "").splitlines()).strip()
                    aggregate += word_errors(row["transcript"], prediction)
                    empty += not bool(prediction)
                    preds.append({"audio_filename": row["audio_filename"],
                                  "transcript": prediction})
                if (start // 24) % 5 == 0:
                    print(f"{label}: {min(start + 24, len(jember))}/{len(jember)}", flush=True)

        with (out / f"{label}.jember.predictions.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["audio_filename", "transcript"])
            writer.writeheader()
            writer.writerows(preds)
        results["models"][label] = {"wer": aggregate.wer, **asdict(aggregate),
                                    "empty_predictions": empty, "note": note,
                                    "seconds": time.monotonic() - started}
        print(f"{label}: JEMBER-JAVANESE WER={aggregate.wer:.6f} "
              f"S={aggregate.substitutions} D={aggregate.deletions} "
              f"I={aggregate.insertions} N={aggregate.reference_words}  [{note}]", flush=True)
        del wrapper, model
        torch.cuda.empty_cache()
        (out / "JAVANESE_PROBE.json").write_text(json.dumps(results, indent=2) + "\n")
        volume.commit()

    dev_javind = 0.26656615693876085
    base = results["models"]["base"]["wer"]
    results["comparison"] = {
        "base_on_dev_javind": dev_javind,
        "base_on_jember_javanese": base,
        "ratio": base / dev_javind,
        "implied_jav_share_explaining_0.47555": (
            (0.47555 - dev_javind) / (base - dev_javind) if base > dev_javind else None),
    }
    (out / "JAVANESE_PROBE.json").write_text(json.dumps(results, indent=2) + "\n")
    volume.commit()
    print(json.dumps(results["comparison"], indent=2), flush=True)
    return results


@app.local_entrypoint()
def main(sample: int = 350):
    print(json.dumps(probe.remote(sample=sample), indent=2))
