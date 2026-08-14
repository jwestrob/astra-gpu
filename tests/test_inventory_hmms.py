import tempfile
import unittest
from pathlib import Path

from scripts.inventory_hmms import records_in_file, summarize


class InventoryHmmsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "models.hmm"
        self.path.write_text(
            """HMMER3/f [3.4 | Aug 2023]
NAME  short
ACC   TEST0001.1
LENG  7
GA    1.0 2.0;
TC    3.0 4.0;
NC    5.0 6.0;
HMM          A C
             m->m m->i
//
HMMER3/f [3.3 | Nov 2019]
NAME  long
LENG  21
HMM          A C
             m->m m->i
//
""",
            encoding="ascii",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_records_capture_headers_without_model_body(self):
        records = list(records_in_file(self.path))
        self.assertEqual([record["NAME"] for record in records], ["short", "long"])
        self.assertEqual([record["LENG"] for record in records], [7, 21])
        self.assertEqual(records[0]["ACC"], "TEST0001.1")
        self.assertNotIn("GA", records[1])

    def test_summary_counts_formats_cutoffs_and_states(self):
        summary = summarize("fixture", str(self.path))
        self.assertEqual(summary["profiles"], 2)
        self.assertEqual(summary["total_profile_states"], 28)
        self.assertEqual(summary["length"]["min"], 7)
        self.assertEqual(summary["length"]["max"], 21)
        self.assertEqual(summary["cutoff_counts"], {"GA": 1, "TC": 1, "NC": 1})
        self.assertEqual(
            summary["formats"],
            {
                "HMMER3/f [3.3 | Nov 2019]": 1,
                "HMMER3/f [3.4 | Aug 2023]": 1,
            },
        )
        self.assertEqual(summary["quantile_profiles"]["p0"]["name"], "short")
        self.assertEqual(summary["quantile_profiles"]["p100"]["name"], "long")


if __name__ == "__main__":
    unittest.main()
