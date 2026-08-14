import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "build" / "oracle" / "msv-oracle"
HMM_20AA = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
HMM_M1 = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "M1.hmm"

ROUTE_SEQUENCES = {
    "ssv_ok": "G",
    "fallback_ok": "ACDEX",
    "fallback_erange": "ACDEXACDEXACDEX",
    "ssv_erange": "ACDEFGHIKLMNPQRSTVWY",
}


@unittest.skipUnless(ORACLE.is_file(), "build the MSV oracle before running integration tests")
class MsvOracleBoundaryTests(unittest.TestCase):
    def run_oracle(self, fasta, *, hmm=HMM_20AA, f1=None, strict=True, text=True):
        command = [
            str(ORACLE),
            "--max-models",
            "1",
            "--max-seqs",
            "0",
        ]
        if f1 is not None:
            command.extend(("--F1", f1))
        if strict:
            command.append("--strict")
        command.extend((str(hmm), str(fasta)))
        return subprocess.run(
            command,
            cwd=ROOT,
            text=text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def comparison_records(self, result):
        records = [json.loads(line) for line in result.stdout.splitlines()]
        return records, [
            record for record in records if record["record"] == "comparison"
        ]

    @staticmethod
    def write_fasta(path, sequences):
        path.write_text(
            "".join(f">{name}\n{sequence}\n" for name, sequence in sequences.items()),
            encoding="ascii",
        )

    def test_short_long_short_reconfiguration_has_no_stale_length_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "short-long-short.fa"
            self.write_fasta(
                fasta,
                {
                    "short_before": "ACDEX",
                    "long": "G" * 340,
                    "short_after": "ACDEX",
                },
            )
            result = self.run_oracle(fasta)

        self.assertEqual(result.returncode, 0, result.stderr)
        records, comparisons = self.comparison_records(result)
        self.assertEqual([record["L"] for record in comparisons], [5, 340, 5])
        self.assertEqual(
            [record["profile_u8"]["tjb_b"] for record in comparisons],
            [4, 21, 4],
        )

        before, _, after = comparisons
        length_dependent_fields = (
            "profile_u8",
            "msv_path",
            "ssv",
            "public_msv",
            "scalar_full_msv",
            "null_score",
            "bit_score",
            "P",
            "pass_F1",
            "agreement",
        )
        for field in length_dependent_fields:
            with self.subTest(field=field):
                self.assertEqual(before[field], after[field])
        self.assertEqual(records[-1]["mismatches"], 0)

    def test_tjb_quantization_changes_only_at_adjacent_length_boundaries(self):
        lengths = (4, 5, 6, 7, 8)
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "tjb-neighbors.fa"
            self.write_fasta(fasta, {f"L{length}": "G" * length for length in lengths})
            result = self.run_oracle(fasta)

        self.assertEqual(result.returncode, 0, result.stderr)
        _, comparisons = self.comparison_records(result)
        self.assertEqual([record["L"] for record in comparisons], list(lengths))
        self.assertEqual(
            [record["profile_u8"]["tjb_b"] for record in comparisons],
            [4, 4, 5, 5, 6],
        )
        self.assertEqual(
            {record["profile_u8"]["tbm_b"] for record in comparisons},
            {23},
        )

        raw_scores = [record["public_msv"]["score"] for record in comparisons]
        self.assertEqual(raw_scores[0], raw_scores[1])
        self.assertNotEqual(raw_scores[1], raw_scores[2])
        self.assertEqual(raw_scores[2], raw_scores[3])
        self.assertNotEqual(raw_scores[3], raw_scores[4])
        self.assertTrue(
            all(
                record["agreement"]["status"]
                and record["agreement"]["score_bits"]
                for record in comparisons
            )
        )

    def test_tjb_neighbor_switches_ssv_overflow_to_full_msv_fallback(self):
        consensus = "ACDEFGHIKLMNPQRSTVWY"
        before_length = 27_582
        after_length = 27_583
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "overflow-transition.fa"
            self.write_fasta(
                fasta,
                {
                    "before": consensus + "G" * (before_length - len(consensus)),
                    "after": consensus + "G" * (after_length - len(consensus)),
                },
            )
            result = self.run_oracle(fasta)

        self.assertEqual(result.returncode, 0, result.stderr)
        _, comparisons = self.comparison_records(result)
        before, after = comparisons
        self.assertEqual([before["L"], after["L"]], [before_length, after_length])
        self.assertEqual(
            [before["profile_u8"]["tjb_b"], after["profile_u8"]["tjb_b"]],
            [39, 40],
        )
        self.assertEqual(
            [
                record["profile_u8"]["base_b"]
                - record["profile_u8"]["tjb_b"]
                - record["profile_u8"]["tbm_b"]
                for record in comparisons
            ],
            [128, 127],
        )

        self.assertEqual(before["msv_path"], "ssv_erange")
        self.assertEqual(before["ssv"]["status"], "eslERANGE")
        self.assertEqual(before["ssv"]["score"]["class"], "+inf")
        self.assertEqual(before["agreement"]["reference"], "ssv")

        self.assertEqual(after["msv_path"], "ssv_fallback_msv_erange")
        self.assertEqual(after["ssv"]["status"], "eslENORESULT")
        self.assertEqual(after["ssv"]["score"]["class"], "+inf")
        self.assertEqual(after["public_msv"]["status"], "eslERANGE")
        self.assertEqual(after["scalar_full_msv"]["status"], "eslERANGE")
        self.assertEqual(after["agreement"]["reference"], "scalar_full_msv")
        self.assertTrue(after["agreement"]["status"])
        self.assertTrue(after["agreement"]["score_bits"])

    def test_F1_equality_and_adjacent_double_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "one.fa"
            self.write_fasta(fasta, {"one": "A"})
            baseline = self.run_oracle(fasta, hmm=HMM_M1)
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            _, baseline_comparisons = self.comparison_records(baseline)
            probability_hex = baseline_comparisons[0]["P"]["hex"]
            probability = float.fromhex(probability_hex)
            lower = math.nextafter(probability, 0.0)
            upper = math.nextafter(probability, math.inf)

            self.assertTrue(0.0 < lower < probability < upper < 1.0)
            cases = (
                (lower.hex(), False),
                (probability_hex, True),
                (upper.hex(), True),
            )
            for threshold, expected_pass in cases:
                with self.subTest(threshold=threshold):
                    result = self.run_oracle(fasta, hmm=HMM_M1, f1=threshold)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    _, comparisons = self.comparison_records(result)
                    comparison = comparisons[0]
                    self.assertEqual(comparison["P"]["hex"], probability_hex)
                    self.assertEqual(comparison["pass_F1"], expected_pass)

    def test_output_is_byte_identical_across_repeated_runs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "routes.fa"
            self.write_fasta(fasta, ROUTE_SEQUENCES)
            first = self.run_oracle(fasta, text=False)
            second = self.run_oracle(fasta, text=False)

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stderr, b"")
        self.assertEqual(first.stdout, second.stdout)

    def test_controlled_sequences_cover_all_public_msv_routes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "routes.fa"
            self.write_fasta(fasta, ROUTE_SEQUENCES)
            result = self.run_oracle(fasta)

        self.assertEqual(result.returncode, 0, result.stderr)
        records, comparisons = self.comparison_records(result)
        by_name = {record["sequence_name"]: record for record in comparisons}
        expected = {
            "ssv_ok": ("ssv_ok", "eslOK", "eslOK", "ssv"),
            "fallback_ok": (
                "ssv_fallback_msv_ok",
                "eslENORESULT",
                "eslOK",
                "scalar_full_msv",
            ),
            "fallback_erange": (
                "ssv_fallback_msv_erange",
                "eslENORESULT",
                "eslERANGE",
                "scalar_full_msv",
            ),
            "ssv_erange": ("ssv_erange", "eslERANGE", "eslERANGE", "ssv"),
        }
        for name, (path, ssv_status, public_status, reference) in expected.items():
            with self.subTest(sequence=name):
                comparison = by_name[name]
                self.assertEqual(comparison["msv_path"], path)
                self.assertEqual(comparison["ssv"]["status"], ssv_status)
                self.assertEqual(comparison["public_msv"]["status"], public_status)
                self.assertEqual(comparison["agreement"]["reference"], reference)
                self.assertTrue(comparison["agreement"]["status"])
                self.assertTrue(comparison["agreement"]["score_bits"])

        fallback_overflow = by_name["fallback_erange"]
        self.assertEqual(fallback_overflow["ssv"]["score"]["class"], "nan")
        self.assertEqual(fallback_overflow["public_msv"]["score"]["class"], "+inf")
        self.assertEqual(fallback_overflow["scalar_full_msv"]["overflow_row"], 14)
        self.assertEqual(fallback_overflow["scalar_full_msv"]["overflow_xE_u8"], 238)

        direct_overflow = by_name["ssv_erange"]
        self.assertEqual(direct_overflow["ssv"]["score"]["class"], "+inf")
        self.assertEqual(direct_overflow["public_msv"]["score"]["class"], "+inf")
        self.assertEqual(direct_overflow["scalar_full_msv"]["overflow_row"], 7)
        self.assertEqual(direct_overflow["scalar_full_msv"]["overflow_xE_u8"], 246)
        self.assertEqual(records[-1]["comparisons"], 4)
        self.assertEqual(records[-1]["mismatches"], 0)

    def test_direct_ssv_safe_overestimate_is_not_a_full_msv_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "ssv-overestimate.fa"
            self.write_fasta(fasta, {"negative_emission": "G" * 10})
            result = self.run_oracle(fasta, hmm=HMM_M1)

        self.assertEqual(result.returncode, 0, result.stderr)
        records, comparisons = self.comparison_records(result)
        comparison = comparisons[0]
        self.assertEqual(comparison["msv_path"], "ssv_ok")
        self.assertEqual(comparison["agreement"]["reference"], "ssv")
        self.assertTrue(comparison["agreement"]["status"])
        self.assertTrue(comparison["agreement"]["score_bits"])
        self.assertTrue(
            comparison["agreement"]["public_vs_full_msv"]["status"]
        )
        self.assertFalse(
            comparison["agreement"]["public_vs_full_msv"]["score_bits"]
        )
        self.assertEqual(
            comparison["public_msv"]["score"], comparison["ssv"]["score"]
        )
        self.assertNotEqual(
            comparison["public_msv"]["score"],
            comparison["scalar_full_msv"]["score"],
        )
        self.assertEqual(records[-1]["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
