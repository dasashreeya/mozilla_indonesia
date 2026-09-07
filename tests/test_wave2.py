"""Unit tests for the Wave 2 additions: style tags, archive resolution, plots."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_runs import language_split, merge, wer  # noqa: E402
from scripts.prepare_external import SOURCES, target_tag  # noqa: E402
from scripts.upload_external import find_archives  # noqa: E402


def counts(substitutions=0, deletions=0, insertions=0, reference_words=0):
    return {"substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "reference_words": reference_words}


class StyleTagTests(unittest.TestCase):
    def test_style_tag_replaces_the_language_tag(self):
        self.assertEqual(target_tag("tts_lumajang", "jav", True, True), "Javtts")
        self.assertEqual(target_tag("podcast_hari", "ind", True, True), "Indpod")

    def test_language_tag_is_used_without_the_style_flag(self):
        self.assertEqual(target_tag("tts_lumajang", "jav", True, False), "Jav")
        self.assertEqual(target_tag("cv_id_scripted", "ind", True, False), "Ind")

    def test_untagged_mode_reproduces_the_original_two_way_tag(self):
        self.assertEqual(target_tag("cv_jv", "jav", False, False), "None")
        self.assertEqual(target_tag("podcast_hari", "ind", False, False), "Indonesian")

    def test_every_style_tag_survives_qwen_normalisation(self):
        # qwen-asr uppercases the first character and lowercases the rest, so a
        # tag that does not already have that shape would not match the trained
        # prefix when forced at decode time.
        for name, spec in SOURCES.items():
            tag = spec["style_tag"]
            with self.subTest(source=name):
                self.assertEqual(tag, tag[:1].upper() + tag[1:].lower())

    def test_in_domain_and_external_javanese_get_different_tags(self):
        # The whole point: Jember writes Javanese as `Jav`; if the TTS corpora
        # shared that tag the decoder would learn a blend of two orthographies.
        self.assertNotEqual(target_tag("tts_lumajang", "jav", True, True), "Jav")
        self.assertNotEqual(target_tag("cv_jv", "jav", True, True),
                            target_tag("tts_lumajang", "jav", True, True))


class ArchiveResolutionTests(unittest.TestCase):
    def test_finds_one_archive_per_corpus_whatever_it_is_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tts_lumajang").mkdir()
            archive = root / "tts_lumajang" / "tts-javanese-lumajang-ec830842.tar.gz"
            archive.write_bytes(b"x")
            found, missing = find_archives(root, ["tts_lumajang"])
            self.assertEqual(found, {"tts_lumajang": archive})
            self.assertEqual(missing, [])

    def test_reports_a_missing_corpus_instead_of_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            found, missing = find_archives(Path(tmp), ["tts_bojonegoro"])
            self.assertEqual(found, {})
            self.assertEqual(missing, ["tts_bojonegoro"])

    def test_two_archives_is_an_error_rather_than_a_silent_pick(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cv_jv").mkdir()
            (root / "cv_jv" / "a.tar.gz").write_bytes(b"x")
            (root / "cv_jv" / "b.tar.gz").write_bytes(b"x")
            with self.assertRaises(ValueError):
                find_archives(root, ["cv_jv"])

    def test_non_archive_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cv_jv").mkdir()
            (root / "cv_jv" / "notes.txt").write_text("x")
            archive = root / "cv_jv" / "corpus.tar.gz"
            archive.write_bytes(b"x")
            found, _ = find_archives(root, ["cv_jv"])
            self.assertEqual(found, {"cv_jv": archive})


class WeightingTests(unittest.TestCase):
    def test_wer_is_the_sum_of_edits_over_reference_words(self):
        self.assertAlmostEqual(wer(counts(6, 2, 2, 100)), 0.10)

    def test_wer_of_an_empty_group_is_none_rather_than_a_crash(self):
        self.assertIsNone(wer(counts()))

    def test_merge_pools_counts_before_dividing(self):
        pooled = merge([counts(5, 0, 0, 50), counts(15, 0, 0, 150)])
        self.assertEqual(pooled["reference_words"], 200)
        self.assertAlmostEqual(wer(pooled), 0.10)

    def test_test_weighting_differs_from_the_fold_weighting(self):
        # The recorded case: the fold is 37.96% Javanese, the test set ~49%, and
        # Javanese WER is much the worse of the two, so the fold mean flatters.
        groups = {"language:jav": counts(2871, 0, 0, 8524),
                  "language:javind": counts(3156, 0, 0, 13929)}
        jav, other, fold, weighted = language_split(groups, 0.49)
        self.assertAlmostEqual(jav, 0.3368, places=3)
        self.assertAlmostEqual(other, 0.2266, places=3)
        self.assertLess(fold, weighted)
        self.assertAlmostEqual(weighted, 0.49 * jav + 0.51 * other)

    def test_a_single_language_group_falls_back_to_the_fold_number(self):
        groups = {"language:javind": counts(100, 0, 0, 1000)}
        jav, other, fold, weighted = language_split(groups, 0.49)
        self.assertIsNone(jav)
        self.assertEqual(fold, weighted)

    def test_non_language_groups_do_not_leak_into_the_split(self):
        groups = {"language:jav": counts(10, 0, 0, 100),
                  "language:ind": counts(10, 0, 0, 100),
                  "session:35": counts(400, 0, 0, 432),
                  "source:jember": counts(10, 0, 0, 100)}
        jav, other, fold, _ = language_split(groups, 0.49)
        self.assertAlmostEqual(jav, 0.10)
        self.assertAlmostEqual(other, 0.10)
        self.assertAlmostEqual(fold, 0.10)


class PromptPresetTests(unittest.TestCase):
    def test_preset_text_stays_out_of_the_command_line(self):
        from scripts.prepare_recovery import PROMPT_PRESETS
        self.assertEqual(PROMPT_PRESETS["none"], "")
        self.assertIn("code-switch", PROMPT_PRESETS["codeswitch"])


class ModalNamingTests(unittest.TestCase):
    def test_nothing_modal_renders_names_the_task(self):
        # The account is shared. App name, volume name, dataset directories and
        # run IDs all show in another user's dashboard, so none of them may carry
        # the corpus, language or competition.
        import re
        source = (ROOT / "scripts/modal_recovery.py").read_text(encoding="utf-8")
        rendered = set()
        for pattern in (r'modal\.App\("([^"]+)"\)',
                        r'MODAL_APP_NAME", "([^"]+)"',
                        r'MODAL_VOLUME_NAME", "([^"]+)"',
                        r'"run_id": "([^"]+)"',
                        r'dataset: str = "([^"]+)"',
                        r'"dataset": "([^"]+)"',
                        r'model: str = "([^"]+)"',
                        r'run_id: str = "([^"]+)"',
                        r'^\s{4}"([a-z0-9-]+)": \[' ):
            rendered.update(re.findall(pattern, source, flags=re.MULTILINE))
        self.assertTrue(rendered, "found no rendered names to check")
        forbidden = ("mozilla", "javanese", "jember", "indonesia", "jav", "mdc",
                     "competition", "asr")
        for name in rendered:
            for word in forbidden:
                with self.subTest(name=name, word=word):
                    self.assertNotIn(word, name.lower())


if __name__ == "__main__":
    unittest.main()


class UploadAliasTests(unittest.TestCase):
    def test_every_remote_alias_is_a_corpus_the_builder_knows(self):
        from scripts.upload_external import REMOTE_ALIAS
        for download_name, alias in REMOTE_ALIAS.items():
            with self.subTest(download=download_name):
                self.assertIn(alias, SOURCES)

    def test_the_download_directories_map_onto_known_corpora(self):
        # Guards the one mismatch that cost an upload: downloads land in
        # cv_jv_spontaneous/, the builder calls the corpus cv_jv.
        from scripts.upload_external import REMOTE_ALIAS
        downloads = ROOT / "data/downloads"
        if not downloads.is_dir():
            self.skipTest("no local downloads to check")
        for directory in sorted(d for d in downloads.iterdir() if d.is_dir()):
            alias = REMOTE_ALIAS.get(directory.name, directory.name)
            with self.subTest(directory=directory.name):
                self.assertIn(alias, SOURCES)
