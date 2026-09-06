from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mdc_asr.audio_probe import (
    AudioDecodeError,
    decode,
    downmix,
    probe_clip,
    probe_container,
    signal_metrics,
)

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def write_wav(path: Path, signal: np.ndarray, sample_rate: int, channels: int = 1) -> None:
    """Render a float signal to a real WAV file via ffmpeg."""

    data = np.clip(signal, -1.0, 1.0).astype("<f4").tobytes()
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-i", "-",
            str(path),
        ],
        input=data,
        check=True,
        capture_output=True,
    )


def tone(seconds: float, sample_rate: int, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return amp * np.sin(2 * np.pi * freq * t)


class SignalMetricsTests(unittest.TestCase):
    def test_peak_and_rms_of_known_tone(self) -> None:
        sample_rate = 16000
        metrics = signal_metrics(tone(1.0, sample_rate, amp=0.5), sample_rate)
        # A 0.5-amplitude sine peaks at -6.02 dBFS and has RMS 3.01 dB below that.
        self.assertAlmostEqual(metrics["peak_dbfs"], -6.02, delta=0.1)
        self.assertAlmostEqual(metrics["rms_dbfs"], -9.03, delta=0.1)
        self.assertAlmostEqual(metrics["crest_factor_db"], 3.01, delta=0.1)

    def test_spectral_centroid_tracks_tone_frequency(self) -> None:
        sample_rate = 16000
        metrics = signal_metrics(tone(1.0, sample_rate, freq=1000.0), sample_rate)
        self.assertAlmostEqual(metrics["spectral_centroid_hz"], 1000.0, delta=25.0)

    def test_clipping_is_detected(self) -> None:
        sample_rate = 16000
        clipped = np.clip(tone(1.0, sample_rate, amp=2.0), -1.0, 1.0)
        self.assertGreater(signal_metrics(clipped, sample_rate)["clipped_sample_ratio"], 0.3)

    def test_clean_tone_reports_no_clipping(self) -> None:
        sample_rate = 16000
        metrics = signal_metrics(tone(1.0, sample_rate, amp=0.5), sample_rate)
        self.assertEqual(metrics["clipped_sample_ratio"], 0.0)

    def test_dc_offset_is_detected(self) -> None:
        sample_rate = 16000
        metrics = signal_metrics(tone(1.0, sample_rate) + 0.25, sample_rate)
        self.assertAlmostEqual(metrics["dc_offset"], 0.25, delta=0.01)

    def test_silence_ratio_matches_constructed_gap(self) -> None:
        sample_rate = 16000
        loud = tone(1.0, sample_rate, amp=0.5)
        quiet = np.zeros(sample_rate)
        metrics = signal_metrics(np.concatenate([loud, quiet]), sample_rate)
        self.assertGreater(metrics["silence_ratio"], 0.4)
        self.assertLess(metrics["silence_ratio"], 0.6)

    def test_snr_separates_loud_and_quiet_frames(self) -> None:
        sample_rate = 16000
        rng = np.random.default_rng(0)
        noise = 0.001 * rng.standard_normal(sample_rate)
        speechlike = np.concatenate([tone(1.0, sample_rate, amp=0.5) + noise, noise])
        metrics = signal_metrics(speechlike, sample_rate)
        self.assertGreater(metrics["estimated_snr_db"], 30.0)

    def test_band_limited_signal_reports_low_bandwidth_utilization(self) -> None:
        sample_rate = 48000
        # Content only up to 1 kHz in a 48 kHz container: heavy band limiting.
        metrics = signal_metrics(tone(1.0, sample_rate, freq=1000.0), sample_rate)
        self.assertLess(metrics["effective_bandwidth_hz"], 4000.0)

    def test_empty_signal_raises(self) -> None:
        with self.assertRaises(ValueError):
            signal_metrics(np.array([]), 16000)


@unittest.skipUnless(_HAS_FFMPEG, "ffmpeg is required for decode tests")
class DecodeTests(unittest.TestCase):
    def test_roundtrip_wav_preserves_duration_and_rate(self) -> None:
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tone.wav"
            write_wav(path, tone(0.5, sample_rate), sample_rate)

            samples, rate = decode(path)
            self.assertEqual(rate, sample_rate)
            self.assertEqual(samples.shape[0], 1)
            self.assertAlmostEqual(samples.shape[1] / rate, 0.5, delta=0.01)

    def test_container_probe_reports_declared_properties(self) -> None:
        sample_rate = 48000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tone.wav"
            write_wav(path, tone(0.25, sample_rate), sample_rate)

            container = probe_container(path)
            self.assertEqual(container["declared_sample_rate_hz"], sample_rate)
            self.assertEqual(container["channels"], 1)
            self.assertGreater(container["bytes"], 0)

    def test_stereo_downmix_and_channel_correlation(self) -> None:
        sample_rate = 16000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "stereo.wav"
            mono = tone(0.5, sample_rate)
            interleaved = np.repeat(mono, 2)  # identical channels
            write_wav(path, interleaved, sample_rate, channels=2)

            result = probe_clip(path)
            self.assertEqual(result["channels"], 2)
            # Duplicated channels are perfectly correlated: not true stereo.
            self.assertAlmostEqual(result["channel_correlation"], 1.0, delta=0.01)

    def test_mp3_roundtrip_reports_lossy_codec(self) -> None:
        sample_rate = 48000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tone.mp3"
            write_wav(path, tone(1.0, sample_rate), sample_rate)

            result = probe_clip(path)
            self.assertEqual(result["codec_name"], "mp3")
            self.assertAlmostEqual(result["decoded_duration_seconds"], 1.0, delta=0.1)
            self.assertIsNotNone(result["bandwidth_utilization"])

    def test_decode_rejects_non_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.mp3"
            path.write_bytes(b"not-real-audio")
            with self.assertRaises(AudioDecodeError):
                decode(path)


class DownmixTests(unittest.TestCase):
    def test_downmix_averages_channels(self) -> None:
        samples = np.array([[1.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(downmix(samples), np.array([0.5, 0.5]))

    def test_downmix_passes_through_mono(self) -> None:
        samples = np.array([0.25, -0.25])
        np.testing.assert_allclose(downmix(samples), samples)


if __name__ == "__main__":
    unittest.main()
