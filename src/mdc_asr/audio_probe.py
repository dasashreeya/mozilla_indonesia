"""CPU-only waveform decoding and signal-quality probes built on ffmpeg.

The existing dataset tooling reads container metadata only. That is enough for
duration and declared sample rate, but it cannot describe the signal: level,
noise floor, clipping, silence, or the lowpass cutoff left behind by lossy
encoding. This module decodes the actual samples so those questions can be
answered.

Everything here runs on CPU with numpy alone. ffmpeg is invoked as a
subprocess, which avoids libsndfile's inconsistent MP3 support.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-12

# A sample within this distance of full scale is treated as clipped. Lossy
# codecs can overshoot slightly, so this is deliberately not exactly 1.0.
CLIP_THRESHOLD = 0.999

# Frames quieter than (loudest frame - this) count as silence.
SILENCE_RELATIVE_DB = 35.0

# Bandwidth is the highest frequency still within this range of the spectral
# peak. Chosen to sit below typical MP3 lowpass rolloff so the cutoff is found.
BANDWIDTH_RELATIVE_DB = 40.0

FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010


class FfmpegMissingError(RuntimeError):
    """Raised when the ffmpeg binaries required for decoding are absent."""


class AudioDecodeError(RuntimeError):
    """Raised when ffmpeg cannot decode a file."""


def require_ffmpeg() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise FfmpegMissingError(
            f"Missing required binaries: {', '.join(missing)}. Install ffmpeg to run the audit."
        )


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def probe_container(path: Path) -> dict[str, Any]:
    """Return declared container/codec properties without decoding samples."""

    require_ffmpeg()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioDecodeError(f"ffprobe failed for {path.name}: {result.stderr.strip()[:200]}")

    payload = json.loads(result.stdout or "{}")
    streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise AudioDecodeError(f"No audio stream in {path.name}")
    stream = streams[0]
    fmt = payload.get("format", {})

    return {
        "codec_name": stream.get("codec_name"),
        "sample_fmt": stream.get("sample_fmt"),
        "declared_sample_rate_hz": _to_int(stream.get("sample_rate")),
        "channels": _to_int(stream.get("channels")),
        "stream_bit_rate_bps": _to_int(stream.get("bit_rate")),
        "format_bit_rate_bps": _to_int(fmt.get("bit_rate")),
        "declared_duration_seconds": _to_float(stream.get("duration"))
        or _to_float(fmt.get("duration")),
        "format_name": fmt.get("format_name"),
        "audio_stream_count": len(streams),
        "bytes": path.stat().st_size,
    }


def decode(
    path: Path, *, container: dict[str, Any] | None = None
) -> tuple[np.ndarray, int]:
    """Decode to float32 at the file's native rate.

    Returns a ``(channels, samples)`` array so per-channel statistics stay
    available; callers that want a single signal should use :func:`downmix`.
    """

    require_ffmpeg()
    container = container or probe_container(path)
    sample_rate = container["declared_sample_rate_hz"]
    channels = container["channels"] or 1
    if not sample_rate:
        raise AudioDecodeError(f"Unknown sample rate for {path.name}")

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AudioDecodeError(f"ffmpeg failed for {path.name}: {result.stderr.decode()[:200]}")

    flat = np.frombuffer(result.stdout, dtype="<f4")
    if flat.size == 0:
        raise AudioDecodeError(f"Decoded zero samples from {path.name}")

    usable = (flat.size // channels) * channels
    samples = flat[:usable].reshape(-1, channels).T.astype(np.float64, copy=False)
    return samples, sample_rate


def downmix(samples: np.ndarray) -> np.ndarray:
    """Average channels into one signal."""

    return samples.mean(axis=0) if samples.ndim == 2 else samples


def _db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), _EPS)))


def _frame_rms_db(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS level in dBFS for overlapping analysis frames."""

    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    if signal.size < frame:
        return np.array([_db(np.sqrt(np.mean(signal**2)) if signal.size else 0.0)])

    count = 1 + (signal.size - frame) // hop
    # A strided view avoids materializing every frame for long clips.
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(count, frame),
        strides=(signal.strides[0] * hop, signal.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return 20.0 * np.log10(np.maximum(rms, _EPS))


def _mean_power_spectrum(signal: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Average power spectrum across Hann-windowed frames."""

    size = 1
    while size < int(0.032 * sample_rate):
        size *= 2
    size = min(size, max(256, 1 << int(np.floor(np.log2(max(signal.size, 256))))))
    if signal.size < size:
        padded = np.zeros(size)
        padded[: signal.size] = signal
        signal = padded

    hop = size // 2
    count = max(1, 1 + (signal.size - size) // hop)
    window = np.hanning(size)
    accumulator = np.zeros(size // 2 + 1)
    for index in range(count):
        start = index * hop
        segment = signal[start : start + size]
        if segment.size < size:
            break
        accumulator += np.abs(np.fft.rfft(segment * window)) ** 2
    accumulator /= max(1, count)
    freqs = np.fft.rfftfreq(size, d=1.0 / sample_rate)
    return freqs, accumulator


def signal_metrics(signal: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Compute level, noise, silence, and spectral descriptors for one signal.

    The SNR figure is an energy-percentile estimate, not a reference-based
    measurement: loud frames stand in for speech and quiet frames for the noise
    floor. It is comparable across clips of the same corpus but is not an
    absolute acoustic SNR.
    """

    if signal.size == 0:
        raise ValueError("Cannot compute metrics for an empty signal")

    absolute = np.abs(signal)
    peak = float(absolute.max())
    rms = float(np.sqrt(np.mean(signal**2)))
    frame_db = _frame_rms_db(signal, sample_rate)

    loud_db = float(np.percentile(frame_db, 90))
    quiet_db = float(np.percentile(frame_db, 10))
    silence_cutoff = float(frame_db.max()) - SILENCE_RELATIVE_DB
    silent_frames = int(np.count_nonzero(frame_db < silence_cutoff))

    freqs, power = _mean_power_spectrum(signal, sample_rate)
    total_power = float(power.sum())
    if total_power > _EPS:
        normalized = power / total_power
        centroid = float(np.sum(freqs * normalized))
        cumulative = np.cumsum(normalized)
        rolloff85 = float(freqs[int(np.searchsorted(cumulative, 0.85))])
        rolloff99 = float(freqs[int(np.searchsorted(cumulative, 0.99))])
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * normalized)))
    else:
        centroid = rolloff85 = rolloff99 = spread = 0.0

    positive = power[power > _EPS]
    flatness = (
        float(np.exp(np.mean(np.log(positive))) / max(float(np.mean(positive)), _EPS))
        if positive.size
        else 0.0
    )

    power_db = 10.0 * np.log10(np.maximum(power, _EPS))
    above = np.flatnonzero(power_db > (power_db.max() - BANDWIDTH_RELATIVE_DB))
    effective_bandwidth = float(freqs[above[-1]]) if above.size else 0.0

    crossings = int(np.count_nonzero(np.diff(np.signbit(signal))))

    return {
        "samples": int(signal.size),
        "decoded_duration_seconds": round(signal.size / sample_rate, 6),
        "peak_dbfs": round(_db(peak), 3),
        "rms_dbfs": round(_db(rms), 3),
        "crest_factor_db": round(_db(peak) - _db(rms), 3),
        "dc_offset": round(float(signal.mean()), 8),
        "clipped_sample_ratio": round(
            float(np.count_nonzero(absolute >= CLIP_THRESHOLD)) / signal.size, 8
        ),
        "estimated_snr_db": round(loud_db - quiet_db, 3),
        "noise_floor_dbfs": round(quiet_db, 3),
        "active_level_dbfs": round(loud_db, 3),
        "silence_ratio": round(silent_frames / frame_db.size, 6),
        "spectral_centroid_hz": round(centroid, 3),
        "spectral_spread_hz": round(spread, 3),
        "spectral_rolloff85_hz": round(rolloff85, 3),
        "spectral_rolloff99_hz": round(rolloff99, 3),
        "spectral_flatness": round(flatness, 8),
        "effective_bandwidth_hz": round(effective_bandwidth, 3),
        "zero_crossing_rate": round(crossings / signal.size, 8),
    }


def probe_clip(path: Path) -> dict[str, Any]:
    """Full container + signal description for a single audio file."""

    container = probe_container(path)
    samples, sample_rate = decode(path, container=container)
    mono = downmix(samples)
    metrics = signal_metrics(mono, sample_rate)

    channel_correlation = None
    if samples.shape[0] == 2:
        first, second = samples[0], samples[1]
        if first.std() > _EPS and second.std() > _EPS:
            channel_correlation = round(float(np.corrcoef(first, second)[0, 1]), 6)
        else:
            channel_correlation = 1.0

    declared = container.get("declared_duration_seconds")
    decoded = metrics["decoded_duration_seconds"]
    return {
        **container,
        **metrics,
        "sample_rate_hz": sample_rate,
        "duration_mismatch_seconds": (
            round(abs(declared - decoded), 6) if declared is not None else None
        ),
        "channel_correlation": channel_correlation,
        # Nyquist headroom that carries no content: a 48 kHz file whose energy
        # stops at 8 kHz was upsampled or heavily band-limited upstream.
        "bandwidth_utilization": round(
            metrics["effective_bandwidth_hz"] / (sample_rate / 2.0), 6
        ),
    }
