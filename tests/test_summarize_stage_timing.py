import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from baseline.summarize_stage_timing import (
    amdahl_speedup,
    atomic_json,
    scientific_table_digest,
    summarize,
)
from baseline.patches.validate_stage_timing import COUNT_ONLY, FIELDS


STAGE_COUNTS = {
    "target_reconfig": 10,
    "pipeline_total": 10,
    "null1": 10,
    "msv_public": 10,
    "msv_ssv_attempt": 10,
    "msv_classic_fallback": 2,
    "bias_filter": 8,
    "viterbi_filter": 7,
    "forward_parser": 4,
    "backward_parser": 2,
    "domain_workflow": 2,
}
COUNTER_COUNTS = {metric: 0 for metric in COUNT_ONLY} | {
    "msv_ssv_status_ok": 7,
    "msv_ssv_status_erange": 1,
    "msv_ssv_status_noresult": 2,
    "msv_ssv_status_other": 0,
    "msv_fallback_status_ok": 1,
    "msv_fallback_status_erange": 1,
    "msv_fallback_status_other": 0,
    "target_sequences": 10,
    "target_residues": 1000,
    "passed_msv": 8,
    "passed_bias": 7,
    "passed_viterbi": 4,
    "passed_forward": 2,
}


def write_timing(path: Path, scale: int) -> None:
    elapsed = {
        "target_reconfig": 20_000_000,
        "pipeline_total": 1_000_000_000,
        "null1": 30_000_000,
        "msv_public": 200_000_000,
        "msv_ssv_attempt": 160_000_000,
        "msv_classic_fallback": 40_000_000,
        "bias_filter": 50_000_000,
        "viterbi_filter": 80_000_000,
        "forward_parser": 90_000_000,
        "backward_parser": 100_000_000,
        "domain_workflow": 120_000_000,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for metric, count in {**STAGE_COUNTS, **COUNTER_COUNTS}.items():
            writer.writerow(
                {
                    "schema_version": 1,
                    "clock": "CLOCK_MONOTONIC_RAW",
                    "query_index": 1,
                    "query": "test-profile",
                    "metric": metric,
                    "count": count,
                    "elapsed_ns": elapsed.get(metric, 0) * scale,
                }
            )


def write_run(
    path: Path,
    index: int,
    timing_tsv: Path,
    tblout: Path,
    domtblout: Path,
    wall_seconds: float,
) -> None:
    executable = path.parent / "timed-hmmsearch"
    query_hmm = path.parent / "query.hmm"
    target_fasta = path.parent / "targets.faa"
    if not executable.exists():
        executable.write_bytes(b"instrumented executable\n")
        query_hmm.write_bytes(b"HMMER3/f\n//\n")
        target_fasta.write_bytes(b">target\nAAAA\n")
    value = {
        "schema_version": 1,
        "label": f"replicate-{index}",
        "cwd": str(path.parent),
        "command": [
            "hmmsearch",
            "--cpu",
            "0",
            "--stage-timing",
            str(timing_tsv),
            "--tblout",
            str(tblout),
            "--domtblout",
            str(domtblout),
            str(query_hmm),
            str(target_fasta),
        ],
        "executable": {
            "path": str(executable),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        },
        "host": {"hostname": "shared-test-node"},
        "timing": {"wall_seconds": wall_seconds},
        "exit_code": 0,
        "metadata": {
            "dataset_id": "test-dataset",
            "hmmer_cpu_workers": 0,
            "run_kind": "replicate",
            "run_index": index,
            "run_status": "pilot",
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")


class SummarizeStageTimingTests(unittest.TestCase):
    def test_scientific_digest_normalizes_only_nonscientific_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.tblout"
            second = root / "second.tblout"
            first.write_text("# command one\nrow  1   \n\nrow 2\n", encoding="utf-8")
            second.write_text("# command two\nrow  1\nrow 2  \n", encoding="utf-8")
            self.assertEqual(
                scientific_table_digest(first), scientific_table_digest(second)
            )
            second.write_text("# command two\nrow 1\nrow 2\n", encoding="utf-8")
            self.assertNotEqual(
                scientific_table_digest(first), scientific_table_digest(second)
            )

    def test_summary_computes_fractions_rates_and_amdahl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine_tblout = root / "pristine.tblout"
            pristine_domtblout = root / "pristine.domtblout"
            pristine_tblout.write_text("# pristine\ntarget row\n", encoding="utf-8")
            pristine_domtblout.write_text("# pristine\ndomain row\n", encoding="utf-8")
            astra = root / "astra-summary.json"
            astra.write_text('{"engine":"Astra"}\n', encoding="utf-8")
            timing_patch = root / "stage-timing.patch"
            timing_patch.write_text("test timing patch\n", encoding="utf-8")

            pairs = []
            for index, scale in ((1, 1), (2, 2)):
                timing = root / f"replicate-{index}.timing.tsv"
                tblout = root / f"replicate-{index}.tblout"
                domtblout = root / f"replicate-{index}.domtblout"
                run_json = root / f"replicate-{index}.json"
                write_timing(timing, scale)
                tblout.write_text(f"# timed {index}\ntarget row   \n", encoding="utf-8")
                domtblout.write_text(f"# timed {index}\ndomain row\n", encoding="utf-8")
                write_run(run_json, index, timing, tblout, domtblout, 2.0 * scale)
                pairs.append((run_json, timing))

            summary = summarize(
                pairs, pristine_tblout, pristine_domtblout, astra, timing_patch
            )
            self.assertEqual(summary["role"], "reference_oracle_stage_timing")
            self.assertEqual(summary["application_baseline"]["name"], "Astra")
            self.assertTrue(summary["scientific_output"]["all_equal_to_pristine"])
            self.assertAlmostEqual(
                summary["medians"]["stage_seconds"]["msv_public"], 0.3
            )
            self.assertAlmostEqual(summary["medians"]["f_pipeline"], 0.2)
            self.assertAlmostEqual(summary["medians"]["f_e2e"], 0.1)
            self.assertFalse(
                summary["measurement"]["observer_overhead"]["calibration_applied"]
            )
            self.assertEqual(summary["counts_per_replicate"]["clock_errors"], 0)
            self.assertEqual(
                summary["immutable_provenance"]["timing_patch"]["sha256"],
                hashlib.sha256(timing_patch.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                summary["immutable_provenance"]["target_fasta"]["sha256"],
                hashlib.sha256((root / "targets.faa").read_bytes()).hexdigest(),
            )
            self.assertAlmostEqual(summary["rates"]["fallback"]["median"], 0.2)
            self.assertAlmostEqual(summary["rates"]["passes"]["forward"]["median"], 0.2)
            self.assertAlmostEqual(
                summary["amdahl_speedup_if_msv_accelerated"]["using_median_f_pipeline"][
                    "10x"
                ],
                1.0 / 0.82,
            )

            output = root / "nested" / "summary.json"
            atomic_json(output, summary)
            self.assertEqual(
                json.loads(output.read_text())["dataset_id"], "test-dataset"
            )

    def test_scientific_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine_tblout = root / "pristine.tblout"
            pristine_domtblout = root / "pristine.domtblout"
            pristine_tblout.write_text("same\n", encoding="utf-8")
            pristine_domtblout.write_text("same\n", encoding="utf-8")
            astra = root / "astra.json"
            astra.write_text("{}\n", encoding="utf-8")
            timing_patch = root / "stage-timing.patch"
            timing_patch.write_text("test timing patch\n", encoding="utf-8")
            timing = root / "run.tsv"
            tblout = root / "run.tblout"
            domtblout = root / "run.domtblout"
            run_json = root / "run.json"
            write_timing(timing, 1)
            tblout.write_text("different\n", encoding="utf-8")
            domtblout.write_text("same\n", encoding="utf-8")
            write_run(run_json, 1, timing, tblout, domtblout, 2.0)
            with self.assertRaisesRegex(ValueError, "differ from pristine"):
                summarize(
                    [(run_json, timing)],
                    pristine_tblout,
                    pristine_domtblout,
                    astra,
                    timing_patch,
                )

    def test_executable_provenance_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pristine_tblout = root / "pristine.tblout"
            pristine_domtblout = root / "pristine.domtblout"
            pristine_tblout.write_text("same\n", encoding="utf-8")
            pristine_domtblout.write_text("same\n", encoding="utf-8")
            astra = root / "astra.json"
            astra.write_text("{}\n", encoding="utf-8")
            timing_patch = root / "stage-timing.patch"
            timing_patch.write_text("test timing patch\n", encoding="utf-8")
            pairs = []
            for index in (1, 2):
                timing = root / f"run-{index}.tsv"
                tblout = root / f"run-{index}.tblout"
                domtblout = root / f"run-{index}.domtblout"
                run_json = root / f"run-{index}.json"
                write_timing(timing, 1)
                tblout.write_text("same\n", encoding="utf-8")
                domtblout.write_text("same\n", encoding="utf-8")
                write_run(run_json, index, timing, tblout, domtblout, 2.0)
                pairs.append((run_json, timing))
            second = json.loads(pairs[1][0].read_text(encoding="utf-8"))
            second["executable"]["sha256"] = "0" * 64
            pairs[1][0].write_text(json.dumps(second), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different instrumented"):
                summarize(
                    pairs, pristine_tblout, pristine_domtblout, astra, timing_patch
                )

    def test_amdahl_infinite_full_fraction_is_unbounded(self) -> None:
        self.assertIsNone(amdahl_speedup(1.0, None))


if __name__ == "__main__":
    unittest.main()
