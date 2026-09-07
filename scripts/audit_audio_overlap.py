#!/usr/bin/env python3
"""Screen whole development clips against every training recording acoustically.

Uses normalized log spectral changes at 5 frames/sec and FFT sliding cosine
similarity. Independent of transcripts, filenames, native rates, and gain.
This is a candidate screen, not proof of absence of partial/altered duplicates.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATE = 8000
HOP = 1600
BANDS = 8


def decode_mono(path):
    p = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                        "-ar", str(RATE), "-ac", "1", "-f", "f32le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype="<f4")


def features(signal, phase=0):
    signal = signal[phase:]
    count = (len(signal) - HOP) // HOP + 1
    if count < 3:
        raise ValueError("Audio too short for overlap screen")
    frames = np.lib.stride_tricks.sliding_window_view(signal, HOP)[::HOP][:count]
    power = np.abs(np.fft.rfft(frames * np.hanning(HOP), n=2048)) ** 2
    frequencies = np.fft.rfftfreq(2048, 1 / RATE)
    edges = np.geomspace(80, 3900, BANDS + 1)
    bands = np.column_stack([np.log(np.maximum(power[:, (frequencies >= a) & (frequencies < b)].mean(axis=1), 1e-12))
                             for a, b in zip(edges[:-1], edges[1:])])
    delta = np.diff(bands, axis=0)
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.divide(delta, norm, out=np.zeros_like(delta), where=norm > 1e-6).astype(np.float32)


def feature_file(path):
    cache = ROOT / "data/audit/acoustic_features"
    cache.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    cached = cache / (identity + "-v1.npz")
    if cached.exists():
        with np.load(cached) as f:
            return str(path), (f["phase0"], f["phase1"])
    signal = decode_mono(path)
    f0, f1 = features(signal), features(signal, HOP // 2)
    np.savez(cached, phase0=f0, phase1=f1)
    return str(path), (f0, f1)


class Matcher:
    def __init__(self, recordings):
        self.recordings = recordings
        self.starts = []
        cursor = 0
        chunks = []
        for name, f in recordings:
            self.starts.append(cursor)
            chunks.extend([f, np.zeros((220, BANDS), dtype=np.float32)])
            cursor += len(f) + 220
        corpus = np.concatenate(chunks)
        self.size = len(corpus)
        self.nfft = 1 << (self.size + 220 - 1).bit_length()
        self.spectrum = np.fft.rfft(corpus, n=self.nfft, axis=0)

    def match(self, query):
        m = len(query)
        if m > 220:
            raise ValueError("Query exceeds whole-clip screening window (44 seconds)")
        qfft = np.fft.rfft(query, n=self.nfft, axis=0)
        scores = np.fft.irfft((self.spectrum * np.conj(qfft)).sum(axis=1), n=self.nfft) / m
        best = {"score": -1.0, "recording": None, "start_seconds": None}
        for (name, f), start in zip(self.recordings, self.starts):
            if len(f) < m:
                continue
            local = scores[start:start + len(f) - m + 1]
            offset = int(np.argmax(local))
            if local[offset] > best["score"]:
                best = {"score": float(local[offset]), "recording": name,
                        "start_seconds": offset * HOP / RATE}
        return best


def main():
    dev = sorted((ROOT / "data/subsets/dev/all").rglob("*.mp3"))
    train = sorted((ROOT / "data/subsets/jember/all").rglob("*.mp3"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        cached = dict(pool.map(feature_file, train + dev))
    print("Spectral features ready", flush=True)
    matcher = Matcher([(p.name, cached[str(p)][0]) for p in train])
    # Known-positive controls: differently aligned crops, re-encoded at 48 kHz.
    controls = []
    for p in [p for p in train if len(cached[str(p)][0]) > 200][:5]:
        signal = decode_mono(p)[int(7.37 * RATE):int(27.37 * RATE)]
        encoded = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(RATE),
                                  "-ac", "1", "-i", "-", "-ar", "48000", "-f", "mp3", "-"],
                                 input=signal.astype("<f4").tobytes(), capture_output=True, check=True)
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-ar", str(RATE),
                                  "-ac", "1", "-f", "f32le", "-"], input=encoded.stdout,
                                 capture_output=True, check=True)
        x = np.frombuffer(decoded.stdout, dtype="<f4")
        result = max((matcher.match(features(x, phase)) for phase in [0, HOP // 2]), key=lambda r: r["score"])
        controls.append({"expected_recording": p.name, **result,
                         "correct_recording": result["recording"] == p.name,
                         "within_half_second": abs(result["start_seconds"] - 7.37) < 0.5})
    results = []
    for i, p in enumerate(dev, 1):
        best = max((matcher.match(f) for f in cached[str(p)]), key=lambda r: r["score"])
        results.append({"dev_id": p.name, **best})
        if i % 50 == 0:
            print(f"Screened {i}/{len(dev)} clips", flush=True)
    results.sort(key=lambda r: r["score"], reverse=True)
    output = {"method": "FFT sliding cosine of 8-band log spectral differences, 8 kHz mono, 200 ms frames; two query phases",
              "training_recordings": len(train), "development_clips": len(dev),
              "known_positive_controls": controls, "review_threshold": 0.65,
              "above_review_threshold": sum(r["score"] >= 0.65 for r in results),
              "best_match_score": max(r["score"] for r in results),
              "limitation": "Whole-clip candidate screen, uncalibrated on this corpus. Does not rule out shorter excerpts, changed speed, splices, or speaker overlap.",
              "matches": results}
    destination = ROOT / "reports/private/acoustic_overlap.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "matches"}, indent=2))


if __name__ == "__main__":
    main()
