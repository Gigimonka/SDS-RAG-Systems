from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.convert_helpman_to_wikijs import parse_xml, table_to_md


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


class TableToMarkdownTest(unittest.TestCase):
    def test_single_cell_layout_table_is_unwrapped(self) -> None:
        table = ET.fromstring(
            """
            <table>
              <tr>
                <td>
                  <para><text>Полный список</text></para>
                  <list type="ul">
                    <li><text>Первый</text></li>
                    <li><text>Второй</text></li>
                  </list>
                </td>
              </tr>
            </table>
            """
        )

        markdown = table_to_md(table, {"variables": {}})

        self.assertEqual(markdown, "Полный список\n\n- Первый\n- Второй")
        self.assertNotIn("| --- |", markdown)

    def test_single_row_data_table_gets_safe_header(self) -> None:
        table = ET.fromstring(
            """
            <table>
              <tr>
                <td><para><text>Статус чека</text></para></td>
                <td><para><text>Требуется</text></para></td>
              </tr>
            </table>
            """
        )

        markdown = table_to_md(table, {"variables": {}})

        self.assertEqual(
            markdown,
            "| &nbsp; | &nbsp; |\n"
            "| --- | --- |\n"
            "| Статус чека | Требуется |",
        )


if __name__ == "__main__":
    unittest.main()
