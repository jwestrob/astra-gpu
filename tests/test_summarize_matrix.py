import tempfile
import unittest
from pathlib import Path

from baseline.summarize_matrix import scientific_table_digest


class SummarizeMatrixTests(unittest.TestCase):
    def test_scientific_digest_ignores_comments_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.tblout"
            second = root / "second.tblout"
            first.write_text("# command one\nrow  1\n\nrow 2   \n", encoding="utf-8")
            second.write_text("# command two\n# date changes\nrow  1\nrow 2\n", encoding="utf-8")
            self.assertEqual(
                scientific_table_digest(first), scientific_table_digest(second)
            )


if __name__ == "__main__":
    unittest.main()
