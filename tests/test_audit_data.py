"""Audit safeguards verified on synthetic data; no competition fixtures."""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from scripts.audit_data import interval_audit, overlap_candidates, probe_source, text_summary, timestamp
from mdc_asr.wer import corpus_errors
from scripts.audit_audio_overlap import BANDS, Matcher


class AuditTests(unittest.TestCase):
    def test_acoustic_matcher_locates_excerpt_and_does_not_cross_recording_boundary(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(100, BANDS)).astype(np.float32)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = rng.normal(size=(100, BANDS)).astype(np.float32)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        matcher = Matcher([("a", a), ("b", b)])
        result = matcher.match(b[15:55])
        self.assertEqual(result["recording"], "b")
        self.assertAlmostEqual(result["start_seconds"], 3.0)
        self.assertAlmostEqual(result["score"], 1.0, places=5)
        cross = matcher.match(np.concatenate([a[-20:], b[:20]]))
        self.assertLess(cross["score"], 0.9)

    def test_timestamps_reject_nonfinite_negative_and_invalid_clock_values(self):
        self.assertEqual(timestamp("1:02:03"), 3723)
        self.assertEqual(timestamp("0:00:02,50"), 2.5)
        for bad in ["nan", "inf", "-1", "1:60", "1:00:60", "1:2:3:4"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                timestamp(bad)

    def test_interval_union_handles_nested_and_touching_intervals(self):
        result = interval_audit([{"id": "a", "start": 0, "end": 10},
                                 {"id": "b", "start": 2, "end": 3},
                                 {"id": "c", "start": 9, "end": 12},
                                 {"id": "d", "start": 12, "end": 15}])
        self.assertEqual(result["union_seconds"], 15)
        self.assertEqual([r["id"] for r in result["overlap_rows"]], ["b", "c"])

    def test_text_overlap_finds_phrase_across_unequal_utterance_boundaries(self):
        train = [{"id": "t", "text": "One two three four five six seven."}]
        dev = [{"id": "d", "text": "Well one two three four five six seven again."}]
        result = overlap_candidates(train, dev)
        self.assertEqual(result["dev_rows_with_exact_text_match"], 0)
        self.assertEqual(result["dev_rows_with_shared_5gram"], 1)
        self.assertNotIn("Well", str(result))

    def test_case_change_fraction_matches_actual_scorer_on_reference_transform(self):
        texts = ["We met Alex in Paris.", "NASA saw Émile there."]
        result = text_summary([{"text": t} for t in texts])
        errors = corpus_errors((t, t.lower()) for t in texts)
        self.assertAlmostEqual(result["reference_lowercase_token_change_fraction"], errors.wer)
        self.assertEqual(text_summary([{"text": "[noise]"}])["empty_after_normalization_rows"], 1)

    def test_audio_probe_flags_boundaries_and_decodes_at_native_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            sr = 8000
            signal = (16000 * np.sin(np.arange(sr) * 2 * np.pi * 440 / sr)).astype("<i2")
            with wave.open(str(path), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(signal.tobytes())
            source, clips, issues = probe_source(("train", str(path), [
                {"id": "a", "start": 0.5, "end": 1.2, "text": "Test"},
                {"id": "b", "start": 2, "end": 3, "text": "Test"}]))
            self.assertNotIn("decode_error", source)
            self.assertEqual(len(clips), 1)
            self.assertAlmostEqual(clips[0]["decoded_duration_seconds"], 0.5)
            self.assertEqual(clips[0]["sample_rate_hz"], sr)
            self.assertEqual(sum(x["reason"] == "start_beyond_decoded_audio" for x in issues), 1)


if __name__ == "__main__":
    unittest.main()
