"""Isolate the dev->test gap: does the submitted decode config degrade WER?

Base Qwen scores 0.2659 on the 294-clip held-out fold under the training
script's decode settings (batch 8, max_new_tokens 512) but 0.47555 on the
2,118-clip public test set. The submitted ZIP decoded with batch 24 and
max_new_tokens 256. This runs the same clips and the same base weights under
all three settings in ONE container so the difference, if any, is attributable
to the decode config rather than to the test distribution.

Deletions are the truncation tell: a capped generation loses trailing words.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "mozilla-indonesian-recovery"
app = modal.App("mozilla-indonesian-diagnose")
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

# (label, inference batch size, max_new_tokens)
CONFIGS = [
    ("local_b8_t512", 8, 512),    # reproduces the 0.2659 baseline
    ("submitted_b24_t256", 24, 256),  # exactly what scored 0.47555
    ("b24_t512", 24, 512),        # separates batching from truncation
]


@app.function(image=image, volumes={"/vol": volume}, gpu="A100-80GB", cpu=8,
              memory=65536, timeout=3600, retries=0, single_use_containers=True)
def diagnose(run_id: str = "decode-diagnostic-v1"):
    import csv
    import json
    import time
    from collections import defaultdict
    from dataclasses import asdict

    import torch
    from qwen_asr import Qwen3ASRModel
    from mdc_asr.wer import ErrorCounts, word_errors

    data = Path("/vol/data/recovery-v1")
    out = Path("/vol/runs") / run_id
    out.mkdir(parents=True, exist_ok=True)

    with (data / "metadata.csv").open() as f:
        validation = [r for r in csv.DictReader(f) if r["split"] == "validation"]
    if len(validation) != 294:
        raise ValueError(f"Expected 294 validation clips, found {len(validation)}")
    paths = [str(data / "audio" / r["audio_filename"]) for r in validation]

    results = {}
    for label, batch, tokens in CONFIGS:
        print(f"=== {label}: batch={batch} max_new_tokens={tokens} ===", flush=True)
        wrapper = Qwen3ASRModel.from_pretrained(
            "/vol/models/qwen3-asr-1.7b", dtype=torch.bfloat16, device_map="cuda:0",
            max_inference_batch_size=batch, max_new_tokens=tokens, local_files_only=True)
        model = wrapper.model
        for flag in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
            if not model.generation_config.do_sample and getattr(model.generation_config, flag, None) is not None:
                setattr(model.generation_config, flag, None)
        model.eval()

        aggregate = ErrorCounts()
        groups = defaultdict(ErrorCounts)
        preds, empty, pred_words = [], 0, 0
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(validation), batch):
                rows = validation[start:start + batch]
                outputs = wrapper.transcribe(audio=paths[start:start + batch], language=None)
                if len(outputs) != len(rows):
                    raise ValueError("Prediction count mismatch")
                for row, output in zip(rows, outputs):
                    prediction = " ".join((output.text or "").splitlines()).strip()
                    counts = word_errors(row["transcript"], prediction)
                    aggregate += counts
                    empty += not bool(prediction)
                    pred_words += len(prediction.split())
                    groups["duration:over30" if float(row["duration_seconds"]) > 30
                           else "duration:up_to30"] += counts
                    preds.append({"audio_filename": row["audio_filename"], "transcript": prediction})
                if (start // batch) % 5 == 0:
                    print(f"{label}: {min(start + batch, len(validation))}/{len(validation)}", flush=True)

        with (out / f"{label}.predictions.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["audio_filename", "transcript"])
            w.writeheader()
            w.writerows(preds)
        results[label] = {
            "wer": aggregate.wer, **asdict(aggregate), "empty_predictions": empty,
            "batch_size": batch, "max_new_tokens": tokens, "clips": len(validation),
            "predicted_words": pred_words, "seconds": time.monotonic() - started,
            "groups": {g: {"wer": c.wer, **asdict(c)} for g, c in groups.items()},
        }
        print(f"{label}: WER={aggregate.wer:.6f} S={aggregate.substitutions} "
              f"D={aggregate.deletions} I={aggregate.insertions} "
              f"pred_words={pred_words} empty={empty}", flush=True)

        del wrapper, model
        torch.cuda.empty_cache()
        (out / "DIAGNOSTIC.json").write_text(json.dumps(results, indent=2) + "\n")
        volume.commit()

    print(json.dumps({k: v["wer"] for k, v in results.items()}, indent=2), flush=True)
    return results


@app.local_entrypoint()
def main(run_id: str = "decode-diagnostic-v1"):
    print(json.dumps(diagnose.remote(run_id=run_id), indent=2))
