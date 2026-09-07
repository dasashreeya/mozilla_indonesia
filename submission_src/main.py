"""Offline Qwen3-ASR inference entrypoint for the MDC code-execution runtime.

The runtime gives one shot at 2,118 clips inside a wall-clock limit, and a run
that raises produces no CSV at all -- the submission is spent for nothing. So
this entrypoint is written to always emit a complete, valid submission:

* a full-length CSV of empty transcripts is written before decoding starts, so
  a hard crash still leaves a well-formed file behind;
* a failing batch is retried clip by clip, and only the individual clips that
  fail are left blank;
* decoding stops early if the wall-clock deadline approaches, keeping whatever
  has been decoded so far.

An empty transcript scores as deletions for that one clip. A missing file
scores nothing at all. The trade is always worth it.
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence


# Competition label -> the target-side tag the fine-tune was trained to emit.
# test_metadata.csv carries this column, and roughly half the test set is
# Javanese-only while development contains none, so routing on it is the one
# piece of per-clip information the manifest gives away for free.
LANGUAGE_TAGS = {"jav": "Jav", "ind": "Ind", "javind": "Javind"}

SOURCE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MDC_DATA_DIR", "/code_execution/data"))
OUTPUT_PATH = Path(
    os.environ.get("MDC_SUBMISSION_PATH", "/code_execution/submission/submission.csv")
)

# The published runtime limit is 2 hours. Stop early enough to still write out.
DEFAULT_TIME_LIMIT_SECONDS = 2 * 60 * 60
DEFAULT_TIME_MARGIN_SECONDS = 5 * 60


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


def clean(text: object) -> str:
    """Collapse a raw model output into one CSV-safe line."""
    return " ".join(str(text or "").splitlines()).strip()


def resolve_deadline(config: Mapping[str, object], started: float) -> float:
    limit = float(os.environ.get("MDC_TIME_LIMIT_SECONDS",
                                 config.get("time_limit_seconds", DEFAULT_TIME_LIMIT_SECONDS)))
    margin = float(os.environ.get("MDC_TIME_MARGIN_SECONDS",
                                  config.get("time_margin_seconds", DEFAULT_TIME_MARGIN_SECONDS)))
    return started + max(limit - margin, 0.0)


def decode_all(audio_paths: Sequence[str], batch_size: int,
               decode_batch: Callable[[Sequence[str]], List[str]],
               deadline: float) -> List[str]:
    """Decode every path, isolating per-clip failures and honouring the deadline.

    The broad ``except`` here is deliberate and is the point of the function:
    an unhandled decode error on any single clip would otherwise discard the
    other 2,117. Every failure is printed so it stays visible in the run log.
    """
    predictions = [""] * len(audio_paths)
    failures = 0
    for start in range(0, len(audio_paths), batch_size):
        if time.monotonic() >= deadline:
            print(f"Deadline reached; leaving {len(audio_paths) - start} clips blank", flush=True)
            break
        chunk = list(audio_paths[start : start + batch_size])
        try:
            texts = decode_batch(chunk)
            if len(texts) != len(chunk):
                raise RuntimeError(f"Model returned {len(texts)} transcripts for {len(chunk)} clips")
        except Exception as exc:  # noqa: BLE001 - isolate the batch, keep the run
            print(f"Batch at offset {start} failed ({exc!r}); retrying individually", flush=True)
            texts = []
            for path in chunk:
                if time.monotonic() >= deadline:
                    texts.append("")
                    continue
                try:
                    single = decode_batch([path])
                    texts.append(single[0] if single else "")
                except Exception as inner:  # noqa: BLE001 - one clip must not end the run
                    failures += 1
                    print(f"Clip failed, left blank: {path} ({inner!r})", flush=True)
                    texts.append("")
        for offset, text in enumerate(texts):
            predictions[start + offset] = text
    if failures:
        print(f"{failures} clip(s) could not be decoded", flush=True)
    return predictions


def register_language_tags() -> List[str]:
    """Allow the fine-tune's own tags to be forced at decode time.

    ``validate_language`` rejects anything outside a module-level list, and
    Javanese is not in it. The list is extended in place, so every module that
    imported the validator sees the addition.
    """
    from qwen_asr.inference import utils

    missing = [tag for tag in LANGUAGE_TAGS.values() if tag not in utils.SUPPORTED_LANGUAGES]
    utils.SUPPORTED_LANGUAGES.extend(missing)
    return missing


def qwen_decoder(config: Mapping[str, object],
                 languages: Optional[Mapping[str, str]] = None
                 ) -> Callable[[Sequence[str]], List[str]]:
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
    prompt = str(config.get("prompt") or "")
    routing = bool(config.get("language_routing")) and bool(languages)
    if routing:
        print(f"Language routing on; registered tags: {register_language_tags()}", flush=True)

    def decode(paths: Sequence[str]) -> List[str]:
        if routing:
            # An unknown or missing label decodes unforced rather than guessing:
            # qwen-asr accepts None per item inside the batch.
            forced = [LANGUAGE_TAGS.get(languages.get(path, ""), None) for path in paths]
        else:
            forced = config.get("force_language")
        results = model.transcribe(audio=list(paths), context=[prompt] * len(paths),
                                   language=forced)
        return [clean(result.text) for result in results]

    return decode


def whisper_decoder(config: Mapping[str, object]) -> Callable[[Sequence[str]], List[str]]:
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

    def decode(paths: Sequence[str]) -> List[str]:
        results = recognizer(list(paths), generate_kwargs=generate_kwargs)
        if isinstance(results, dict):
            results = [results]
        return [clean(result.get("text", "")) for result in results]

    return decode


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
    started = time.monotonic()
    config = load_config()
    backend = detect_backend(config)
    if backend not in {"qwen3_asr_transformers", "whisper_transformers"}:
        raise ValueError("Unsupported backend")
    rows = read_manifest()
    deadline = resolve_deadline(config, started)

    # Emit a complete, valid CSV up front so even a hard failure below -- an
    # OOM while loading weights, say -- still leaves a submittable file.
    write_submission(rows, [""] * len(rows))
    print(f"Wrote placeholder submission for {len(rows)} clips", flush=True)

    audio_paths = [str(DATA_DIR / "clips" / row["audio_filename"]) for row in rows]
    languages = {path: (row.get("language") or "").strip()
                 for path, row in zip(audio_paths, rows)}
    labelled = sum(1 for value in languages.values() if value in LANGUAGE_TAGS)
    print(f"Manifest carries {labelled}/{len(rows)} recognised language labels", flush=True)
    if backend == "qwen3_asr_transformers":
        decode_batch = qwen_decoder(config, languages)
        batch_size = int(config["batch_size"])
    else:
        decode_batch = whisper_decoder(config)
        batch_size = int(config["whisper_batch_size"])

    predictions = decode_all(audio_paths, batch_size, decode_batch, deadline)
    write_submission(rows, predictions)
    non_empty = sum(1 for text in predictions if text)
    print(f"Inference completed successfully: {non_empty}/{len(rows)} non-empty "
          f"in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
