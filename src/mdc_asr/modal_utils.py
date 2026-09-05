"""Helpers used by the Modal training and packaging entrypoints."""

import json
import re
from pathlib import Path, PurePosixPath
from typing import Dict, Optional


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def safe_volume_path(volume_root: Path, value: str) -> Path:
    """Resolve a relative or absolute path while keeping it under volume_root."""

    if not value:
        raise ValueError("Path value is empty")
    root = volume_root.resolve(strict=False)
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        pure = PurePosixPath(value)
        if any(part == ".." for part in pure.parts):
            raise ValueError("Path may not contain '..': {}".format(value))
        resolved = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path escapes Modal volume: {}".format(value)) from exc
    return resolved


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    best_step = -1
    best_path: Optional[Path] = None
    if not output_dir.exists():
        return None
    for child in output_dir.iterdir():
        match = _CHECKPOINT_RE.match(child.name)
        if not match or not child.is_dir():
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = child
    return best_path


def rebase_qwen_jsonl(source: Path, destination: Path, audio_dir: Path) -> Dict[str, int]:
    """Rewrite Qwen JSONL audio paths to point at the mounted Modal volume."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing = 0
    with source.open("r", encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            audio_name = Path(str(payload.get("audio", ""))).name
            if not audio_name:
                raise ValueError("Missing audio field in {} line {}".format(source, line_number))
            rebased_audio = audio_dir / audio_name
            if not rebased_audio.is_file():
                missing += 1
                continue
            payload["audio"] = str(rebased_audio)
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
    if written == 0:
        raise ValueError("No usable rows written from {}".format(source))
    return {"written": written, "missing_audio": missing}
