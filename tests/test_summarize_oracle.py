import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_oracle import OracleSummaryError, summarize


def comparison(index: int, agreement: bool = True) -> dict[str, object]:
    return {
        "record": "comparison",
        "model_name": "model",
        "sequence_index": index,
        "M": 17,
        "L": 10 + index,
        "profile_u8": {"tjb_b": 21 + index},
        "msv_path": "ssv_ok" if index == 0 else "ssv_fallback_msv_ok",
        "public_msv": {"status": "eslOK"},
        "scalar_ssv": {
            "status": "eslOK",
            "agreement": {
                "status": agreement,
                "score_bits": agreement,
                "xE_u8": agreement,
            },
        },
        "scalar_full_msv": {"status": "eslOK"},
        "pass_F1": index == 0,
        "agreement": {
            "reference": "ssv",
            "status": agreement,
            "score_bits": agreement,
            "public_vs_full_msv": {"status": True, "score_bits": index == 0},
        },
    }


class SummarizeOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "oracle.jsonl"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, records: list[dict[str, object]]) -> None:
        self.path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_summarizes_coverage_and_paths(self) -> None:
        self.write(
            [
                {"record": "metadata", "schema_version": 1},
                comparison(0),
                comparison(1),
                {"record": "summary", "comparisons": 2, "mismatches": 0},
            ]
        )
        result = summarize(self.path)
        self.assertEqual(result["comparisons"], 2)
        self.assertEqual(result["passes_F1"], 1)
        self.assertEqual(
            result["msv_paths"], {"ssv_fallback_msv_ok": 1, "ssv_ok": 1}
        )
        self.assertEqual(result["coverage"]["distinct_tjb_b"], 2)
        self.assertEqual(result["models"]["model"], {"M": 17, "comparisons": 2})
        self.assertEqual(result["agreement_references"], {"ssv": 2})
        self.assertEqual(result["public_vs_full_msv"]["score_bit_mismatches"], 1)
        self.assertEqual(result["scalar_ssv"]["statuses"], {"eslOK": 2})
        self.assertEqual(result["scalar_ssv"]["xE_u8_mismatches"], 0)

    def test_rejects_disagreement_with_terminal_summary(self) -> None:
        self.write(
            [
                {"record": "metadata"},
                comparison(0, agreement=False),
                {"record": "summary", "comparisons": 1, "mismatches": 0},
            ]
        )
        with self.assertRaisesRegex(OracleSummaryError, "mismatch count"):
            summarize(self.path)

    def test_accepts_pipeline_skipped_empty_sequence(self) -> None:
        self.write(
            [
                {"record": "metadata"},
                {
                    "record": "summary",
                    "comparisons": 0,
                    "skipped_empty": 1,
                    "mismatches": 0,
                },
            ]
        )
        result = summarize(self.path)
        self.assertEqual(result["comparisons"], 0)
        self.assertEqual(result["skipped"], {"pipeline_skips_empty_sequence": 1})


if __name__ == "__main__":
    unittest.main()
