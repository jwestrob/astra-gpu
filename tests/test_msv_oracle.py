import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORACLE = ROOT / "build" / "oracle" / "msv-oracle"
HMM = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
HMM_20AA = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
FASTA = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins45.fa"


@unittest.skipUnless(ORACLE.is_file(), "build the MSV oracle before running integration tests")
class MsvOracleTests(unittest.TestCase):
    def run_oracle(self, *arguments, hmm=HMM, fasta=FASTA):
        return subprocess.run(
            [str(ORACLE), *map(str, arguments), str(hmm), str(fasta)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_rejects_nonfinite_F1(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                result = self.run_oracle("--F1", value, "--max-seqs", "1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid F1 probability", result.stderr)

    def test_rejects_negative_limits(self):
        for option in ("--max-models", "--max-seqs"):
            with self.subTest(option=option):
                result = self.run_oracle(f"{option}=-1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"invalid {option} value", result.stderr)

    def test_rejects_target_above_protein_pipeline_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "too-long.fa"
            fasta.write_text(">too_long\n" + "A" * 100_001 + "\n", encoding="ascii")
            result = self.run_oracle(
                "--max-models", "1", "--max-seqs", "0", fasta=fasta
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("outside HMMER's protein pipeline range", result.stderr)

    def test_empty_sequence_is_not_sent_to_msv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "empty.fa"
            fasta.write_text(">empty\n>one\nA\n", encoding="ascii")
            result = self.run_oracle(
                "--max-models", "1", "--max-seqs", "0", "--strict", fasta=fasta
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        comparisons = [record for record in records if record["record"] == "comparison"]
        summary = records[-1]

        self.assertEqual(
            {record["record"] for record in records},
            {"metadata", "comparison", "summary"},
        )
        self.assertEqual(len(comparisons), 1)
        self.assertEqual(summary["record"], "summary")
        self.assertEqual(summary["comparisons"], 1)
        self.assertEqual(summary["skipped_empty"], 1)
        self.assertEqual(summary["mismatches"], 0)

    def test_ambiguities_and_nonresidues_match_public_msv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fasta = Path(temporary_directory) / "ambiguities.fa"
            fasta.write_text(
                ">nonresidue_match\nACDEFGHIK*MNPQRSTVWY\n"
                ">nonresidue_insert\nACDEFGHIKL*MNPQRSTVWY\n"
                ">ambiguities\nBZJXUO\n",
                encoding="ascii",
            )
            result = self.run_oracle(
                "--max-models", "1", "--max-seqs", "0", "--strict",
                hmm=HMM_20AA,
                fasta=fasta,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        summary = records[-1]
        self.assertEqual(summary["comparisons"], 3)
        self.assertEqual(summary["skipped_empty"], 0)
        self.assertEqual(summary["mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
