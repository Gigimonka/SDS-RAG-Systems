from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.convert_helpman_to_wikijs import parse_xml


class ParseXmlTest(unittest.TestCase):
    def test_invalid_utf8_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "truncated.xml"
            source.write_bytes(b'<?xml version="1.0" encoding="UTF-8"?><topic>\xd0')

            root, recovered, error = parse_xml(source)

        self.assertIsNone(root)
        self.assertFalse(recovered)
        self.assertIn("не удалось декодировать UTF-8", error or "")

    def test_valid_utf8_with_bom_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "topic.xml"
            source.write_text("<topic><title>Тест</title></topic>", encoding="utf-8-sig")

            root, recovered, error = parse_xml(source)

        self.assertIsNotNone(root)
        self.assertEqual(root.tag if root is not None else None, "topic")
        self.assertFalse(recovered)
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
