import unittest
from unittest.mock import patch

from modules import history


class ProcessingHistoryTests(unittest.TestCase):
    def test_record_extraction_replaces_duplicate_and_keeps_newest_first(self):
        entries = []
        history.record_extraction(entries, "first", "one.png", "dog", b"cutout")
        history.record_extraction(entries, "second", "two.png", "cat", b"cutout")
        history.record_extraction(entries, "first", "one.png", "dog", b"updated")

        self.assertEqual([entry["id"] for entry in entries], ["first", "second"])
        self.assertEqual(entries[0]["extracted_image"], b"updated")

    def test_record_background_attaches_final_image_to_matching_cutout(self):
        entries = []
        history.record_extraction(entries, "first", "one.png", "dog", b"cutout")

        recorded = history.record_background(
            entries,
            "first",
            b"composite",
            "White",
        )

        self.assertTrue(recorded)
        self.assertEqual(entries[0]["edited_image"], b"composite")
        self.assertEqual(entries[0]["background_mode"], "White")

    def test_history_respects_entry_and_storage_limits(self):
        entries = []
        with patch.object(history, "MAX_HISTORY_ITEMS", 1):
            history.record_extraction(entries, "first", "one.png", "dog", b"one")
            history.record_extraction(entries, "second", "two.png", "cat", b"two")

        self.assertEqual([entry["id"] for entry in entries], ["second"])

        with patch.object(history, "MAX_HISTORY_BYTES", 3):
            saved = history.record_extraction(
                entries,
                "oversized",
                "large.png",
                "object",
                b"too large",
            )

        self.assertFalse(saved)
        self.assertEqual([entry["id"] for entry in entries], ["second"])


if __name__ == "__main__":
    unittest.main()
