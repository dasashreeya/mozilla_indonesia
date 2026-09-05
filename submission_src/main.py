"""Offline Qwen3-ASR inference entrypoint for the MDC code-execution runtime."""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


SOURCE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MDC_DATA_DIR", "/code_execution/data"))
OUTPUT_PATH = Path(
    os.environ.get("MDC_SUBMISSION_PATH", "/code_execution/submission/submission.csv")
)


def load_config() -> Dict[str, object]:
    with (SOURCE_DIR / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_manifest() -> List[Dict[str, str]]:
    path = DATA_DIR / "test_metadata.csv"
    if not path.exists():
        path = DATA_DIR / "submission_format.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "audio_filename" not in rows[0]:
        raise ValueError("Input manifest is empty or lacks audio_filename")
    for row in rows:
        filename = row["audio_filename"]
        if Path(filename).name != filename:
            raise ValueError("Unsafe audio filename")
    return rows


def batched(values: Sequence[str], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def transcribe_qwen(audio_paths: Sequence[str], config: Mapping[str, object]) -> List[str]:
    import torch
    from qwen_asr import Qwen3ASRModel

    model_path = SOURCE_DIR / str(config["model_path"])
    model = Qwen3ASRModel.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=int(config["batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
    )
    predictions: List[str] = []
    for batch in batched(audio_paths, int(config["batch_size"])):
        results = model.transcribe(audio=list(batch), language=config.get("force_language"))
        if len(results) != len(batch):
            raise RuntimeError("Model returned the wrong number of transcripts")
        predictions.extend(" ".join(str(result.text or "").splitlines()).strip() for result in results)
    return predictions


def transcribe_whisper(audio_paths: Sequence[str], config: Mapping[str, object]) -> List[str]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_path = SOURCE_DIR / str(config["model_path"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(str(model_path))
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.bfloat16,
        device=0,
        batch_size=int(config["whisper_batch_size"]),
    )
    generate_kwargs = {"task": "transcribe"}
    if config.get("force_language"):
        generate_kwargs["language"] = config["force_language"]
    results = recognizer(list(audio_paths), generate_kwargs=generate_kwargs)
    if len(results) != len(audio_paths):
        raise RuntimeError("Model returned the wrong number of transcripts")
    return [" ".join(str(result.get("text", "")).splitlines()).strip() for result in results]


def detect_backend(config: Mapping[str, object]) -> str:
    configured = str(config.get("backend", "auto"))
    if configured != "auto":
        return configured
    model_config_path = SOURCE_DIR / str(config["model_path"]) / "config.json"
    with model_config_path.open("r", encoding="utf-8") as handle:
        model_type = json.load(handle).get("model_type")
    if model_type == "qwen3_asr":
        return "qwen3_asr_transformers"
    if model_type == "whisper":
        return "whisper_transformers"
    raise ValueError("Could not detect a supported model backend")


def write_submission(rows: Sequence[Mapping[str, str]], predictions: Sequence[str]) -> None:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match input manifest")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("audio_filename", "transcript"))
        writer.writeheader()
        for row, transcript in zip(rows, predictions):
            writer.writerow({"audio_filename": row["audio_filename"], "transcript": transcript})


def main() -> None:
    config = load_config()
    backend = detect_backend(config)
    if backend not in {"qwen3_asr_transformers", "whisper_transformers"}:
        raise ValueError("Unsupported backend")
    rows = read_manifest()
    audio_paths = [str(DATA_DIR / "clips" / row["audio_filename"]) for row in rows]
    if backend == "qwen3_asr_transformers":
        predictions = transcribe_qwen(audio_paths, config)
    else:
        predictions = transcribe_whisper(audio_paths, config)
    write_submission(rows, predictions)
    print("Inference completed successfully")


if __name__ == "__main__":
    main()
