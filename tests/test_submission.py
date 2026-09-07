import csv
import json
import os
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

from submission_src import main as submission


class SubmissionTests(unittest.TestCase):
    def test_backend_auto_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
            old_source = submission.SOURCE_DIR
            try:
                submission.SOURCE_DIR = root
                self.assertEqual(
                    submission.detect_backend({"backend": "auto", "model_path": "model"}),
                    "whisper_transformers",
                )
            finally:
                submission.SOURCE_DIR = old_source

    def test_manifest_and_quoted_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            with (data / "test_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("audio_filename", "language"))
                writer.writeheader()
                writer.writerow({"audio_filename": "a.mp3", "language": "javind"})
            old_data, old_output = submission.DATA_DIR, submission.OUTPUT_PATH
            try:
                submission.DATA_DIR = data
                submission.OUTPUT_PATH = root / "submission" / "submission.csv"
                rows = submission.read_manifest()
                submission.write_submission(rows, ['aku, "lunga"'])
                with submission.OUTPUT_PATH.open("r", newline="", encoding="utf-8") as handle:
                    output = list(csv.DictReader(handle))
                self.assertEqual(output[0]["audio_filename"], "a.mp3")
                self.assertEqual(output[0]["transcript"], 'aku, "lunga"')
            finally:
                submission.DATA_DIR, submission.OUTPUT_PATH = old_data, old_output


class HardeningTests(unittest.TestCase):
    """The submission gets one attempt; a single bad clip must not end the run."""

    def test_failing_batch_is_retried_clip_by_clip(self):
        calls = []

        def decode(paths):
            calls.append(list(paths))
            if "bad.mp3" in paths and len(paths) > 1:
                raise RuntimeError("batch blew up")
            if paths == ["bad.mp3"]:
                raise RuntimeError("this clip is broken")
            return ["text:" + path for path in paths]

        predictions = submission.decode_all(
            ["a.mp3", "bad.mp3", "c.mp3"], 3, decode, time.monotonic() + 60
        )
        self.assertEqual(predictions, ["text:a.mp3", "", "text:c.mp3"])
        # one failed batch of three, then three individual retries
        self.assertEqual(len(calls), 4)

    def test_wrong_transcript_count_falls_back_to_singles(self):
        def decode(paths):
            return ["only-one"] if len(paths) > 1 else ["text:" + paths[0]]

        predictions = submission.decode_all(["a.mp3", "b.mp3"], 2, decode, time.monotonic() + 60)
        self.assertEqual(predictions, ["text:a.mp3", "text:b.mp3"])

    def test_expired_deadline_leaves_clips_blank_without_decoding(self):
        def decode(paths):
            raise AssertionError("must not decode past the deadline")

        predictions = submission.decode_all(["a.mp3", "b.mp3"], 1, decode, time.monotonic() - 1)
        self.assertEqual(predictions, ["", ""])

    def test_deadline_keeps_work_already_done(self):
        """Expiring mid-run must keep decoded clips and blank only the rest."""
        clock = iter([10.0, 20.0, 999.0])  # 3rd check is past the deadline

        def decode(paths):
            return ["text:" + path for path in paths]

        with unittest.mock.patch.object(submission.time, "monotonic", lambda: next(clock)):
            predictions = submission.decode_all(["a.mp3", "b.mp3", "c.mp3"], 1, decode, 100.0)
        self.assertEqual(predictions, ["text:a.mp3", "text:b.mp3", ""])

    def test_clean_collapses_multiline_output(self):
        self.assertEqual(submission.clean("aku\nlunga\n"), "aku lunga")
        self.assertEqual(submission.clean(None), "")

    def test_resolve_deadline_prefers_env_over_config(self):
        started = 1000.0
        config = {"time_limit_seconds": 7200, "time_margin_seconds": 300}
        self.assertEqual(submission.resolve_deadline(config, started), 1000.0 + 6900)
        os.environ["MDC_TIME_LIMIT_SECONDS"] = "60"
        os.environ["MDC_TIME_MARGIN_SECONDS"] = "10"
        try:
            self.assertEqual(submission.resolve_deadline(config, started), 1050.0)
        finally:
            del os.environ["MDC_TIME_LIMIT_SECONDS"]
            del os.environ["MDC_TIME_MARGIN_SECONDS"]

    def test_margin_larger_than_limit_does_not_go_backwards(self):
        self.assertEqual(
            submission.resolve_deadline({"time_limit_seconds": 10, "time_margin_seconds": 99}, 500.0),
            500.0,
        )




class LanguageRoutingTests(unittest.TestCase):
    def test_tags_cover_every_competition_label_and_survive_the_normalizer(self):
        self.assertEqual(set(submission.LANGUAGE_TAGS), {"jav", "ind", "javind"})
        for tag in submission.LANGUAGE_TAGS.values():
            # qwen-asr normalizes a forced language to first-upper/rest-lower; a
            # tag that does not survive that would not match the trained prefix.
            self.assertEqual(tag[:1].upper() + tag[1:].lower(), tag)

    def test_unknown_or_missing_labels_decode_unforced_instead_of_guessing(self):
        languages = {"a.mp3": "jav", "b.mp3": "javind", "c.mp3": "", "d.mp3": "xx"}
        forced = [submission.LANGUAGE_TAGS.get(languages.get(p, ""), None)
                  for p in ["a.mp3", "b.mp3", "c.mp3", "d.mp3", "missing.mp3"]]
        self.assertEqual(forced, ["Jav", "Javind", None, None, None])


if __name__ == "__main__":
    unittest.main()
