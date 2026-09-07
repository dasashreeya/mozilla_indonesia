"""Measure prepared holdout audio on CPU; diagnostic associations, not ASR gains."""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mdc_asr.audio_probe import decode, downmix, signal_metrics
from mdc_asr.normalization import normalize_text


def rank(x):
    x = np.asarray(x)
    return np.asarray([sum(x < v) + (sum(x == v) - 1) / 2 for v in x])


def rho(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/recovery-jav1")
    parser.add_argument("--results", type=Path, default=ROOT / "reports/private/cpu-viability-20260906/RESULTS.json")
    args = parser.parse_args()
    started = time.monotonic()
    previous = json.loads(args.results.read_text())
    prov = json.loads((args.data / "private_provenance.json").read_text())
    vocab = {w for r in prov if r["split"] == "train"
             for w in normalize_text(r["transcript"]).split()}
    validation = {r["audio_filename"]: r for r in prov if r["split"] == "validation"}
    rows = [r for r in previous["per_clip"] if r["source"] == "jember"]

    def measure(row):
        path = args.data / "audio" / row["audio_filename"]
        channels, sr = decode(path)
        signal = downmix(channels)
        metrics = signal_metrics(signal, sr)
        ref = normalize_text(validation[row["audio_filename"]]["transcript"]).split()
        return {**row, "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "metrics": metrics, "oov": sum(w not in vocab for w in ref)}

    measured = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(measure, rows):
            measured.append(result)
            if len(measured) % 50 == 0:
                print(f"Measured prepared audio {len(measured)}/{len(rows)}.", flush=True)
    features = ["rms_dbfs", "estimated_snr_db", "silence_ratio", "effective_bandwidth_hz"]
    sessions = []
    for group in sorted({r["group"] for r in rows}):
        clips = [r for r in measured if r["group"] == group]
        words = sum(r["words"] for r in clips)
        record = {"group": group, "clips": len(clips), "words": words,
                  "errors": sum(r["errors"]["baseline"] for r in clips),
                  "oov_rate": sum(r["oov"] for r in clips) / words,
                  "mean_duration": float(np.mean([validation[r["audio_filename"]]["duration_seconds"] for r in clips]))}
        record["wer"] = record["errors"] / words
        record.update({k: float(np.median([r["metrics"][k] for r in clips])) for k in features})
        sessions.append(record)
    correlations = {k: rho([s[k] for s in sessions], [s["wer"] for s in sessions])
                    for k in features + ["oov_rate", "mean_duration"]}
    without35 = [s for s in sessions if s["group"] != "jember:35"]
    snr_without35 = rho([s["estimated_snr_db"] for s in without35], [s["wer"] for s in without35])
    jackknife = [rho([s["estimated_snr_db"] for j, s in enumerate(sessions) if j != i],
                     [s["wer"] for j, s in enumerate(sessions) if j != i]) for i in range(len(sessions))]
    controls = np.column_stack([np.ones(len(sessions))] +
                              [rank([s[k] for s in sessions]) for k in ["silence_ratio", "oov_rate", "mean_duration"]])
    residuals = []
    for key in ["estimated_snr_db", "wer"]:
        v = rank([s[key] for s in sessions])
        residuals.append(v - controls @ np.linalg.lstsq(controls, v, rcond=None)[0])
    partial = float(np.corrcoef(*residuals)[0, 1])

    # Demonstrate a limitation of the proxy on a fixed synthetic noisy signal:
    # level changes preserve energy contrast; inserted silence inflates it
    # without removing any noise from the original speech-bearing samples.
    sr = 16000
    t = np.arange(sr * 2) / sr
    rng = np.random.default_rng(20260906)
    signal = 0.05 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 0.02, len(t))
    proxy_check = {name: signal_metrics(x, sr)["estimated_snr_db"] for name, x in {
        "original_noisy_signal": signal,
        "same_signal_half_amplitude": signal * 0.5,
        "same_signal_with_half_second_zero_padding": np.r_[signal, np.zeros(sr // 2)],
    }.items()}
    assert abs(proxy_check["original_noisy_signal"] - proxy_check["same_signal_half_amplitude"]) < 1e-8
    assert proxy_check["same_signal_with_half_second_zero_padding"] > proxy_check["original_noisy_signal"] + 100
    low = sorted(sessions, key=lambda s: s["estimated_snr_db"])[:5]
    all_words = sum(r["words"] for r in previous["per_clip"])
    all_errors = previous["scores"]["baseline"]["all"]["errors"]
    result = {
        "prepared_clips_measured": len(measured), "sessions": sessions,
        "spearman_vs_wer": correlations,
        "snr_spearman_excluding_session35": snr_without35,
        "snr_spearman_leave_one_session_out_range": [min(jackknife), max(jackknife)],
        "snr_partial_rank_correlation_controlling_silence_oov_duration": partial,
        "snr_proxy_synthetic_checks": proxy_check,
        "perfect_lowest_snr_five_sessions_scenario": {
            "groups": [s["group"] for s in low],
            "errors_removed_if_perfect": sum(s["errors"] for s in low),
            "mixed_wer_if_perfect": (all_errors - sum(s["errors"] for s in low)) / all_words,
            "qualification": "Unrealistic perfect-transcription scenario, not a predicted augmentation effect."},
        "seconds": time.monotonic() - started,
        "limitations": [
            "Energy-percentile contrast is not a true noise measurement or proof of an acoustic cause.",
            "Only 20 sessions; shared speakers, labels and correlated predictors limit inference.",
            "Partial correlation is exploratory with several controls and a small sample.",
            "No ASR was run on altered audio; this does not measure robustness or augmentation WER gains.",
        ],
        "per_clip": measured,
    }
    output = args.results.parent / "PREPARED_AUDIO_DIAGNOSTIC.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ["sessions", "per_clip"]}, indent=2))


if __name__ == "__main__":
    main()
