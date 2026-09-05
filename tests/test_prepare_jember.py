import unittest

from scripts.prepare_jember import canonical_row, parse_timestamp


class PrepareJemberTests(unittest.TestCase):
    def test_timestamp_formats(self):
        self.assertEqual(parse_timestamp("1:02:03.5"), 3723.5)
        self.assertEqual(parse_timestamp("02:03"), 123.0)
        self.assertEqual(parse_timestamp("7,25"), 7.25)

    def test_header_canonicalization(self):
        row = canonical_row({"Audio file name": "001", "Start-Time": "0:01"})
        self.assertEqual(row, {"audio_file_name": "001", "start_time": "0:01"})


if __name__ == "__main__":
    unittest.main()

