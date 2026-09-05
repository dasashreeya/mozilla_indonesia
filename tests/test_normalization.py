import unittest

from mdc_asr.normalization import normalize_text


class NormalizeTextTests(unittest.TestCase):
    def test_official_cleanup_behaviors(self):
        self.assertEqual(normalize_text("Hello, [laugh] WORLD! Next."), "hello WORLD next ")
        self.assertEqual(normalize_text("(aku) [???] ora (???) ngerti..."), "aku ora ngerti...")
        self.assertEqual(normalize_text("I don... I don't know"), "i don... i don't know")

    def test_sentence_initial_acronym_is_preserved(self):
        self.assertEqual(normalize_text("USA works."), "USA works ")
        self.assertEqual(normalize_text("Name NASA."), "name NASA ")

    def test_mid_sentence_case_is_not_folded(self):
        self.assertNotEqual(normalize_text("aku Jawa"), normalize_text("aku jawa"))


if __name__ == "__main__":
    unittest.main()

