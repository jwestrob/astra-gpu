import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from baseline.astra_probe.summarize_probe import (
    SummaryError,
    atomic_compact_json,
    build_summary,
)
from baseline.astra_probe.validate_probe import EXPECTED_STAGES


class SummarizeAstraProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.astra = self._write("astra", b"#!/bin/sh\nexit 0\n")
        self.fasta = self._write("real.faa", b">protein\nMPEPTIDE\n")
        self.hmm = self._write("profiles.hmm", b"HMMER3/f\nNAME test\n//\n")
        self.probe_source = self._write("probe.c", b"/* observer */\n")
        self.probe_binary = self._write("probe.so", b"ELF probe bytes\n")
        self.target_library = self._write("liblibhmmer.so", b"ELF HMMER bytes\n")
        self.hit_bytes = b"sequence_id\thmm_name\nprotein\ttest\n"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _run_record(
        self,
        name: str,
        wall_seconds: float,
        *,
        threads: int = 1,
        exit_code: int = 0,
        role: str = "astra_uninstrumented_control",
        probe: Path | None = None,
        hmm: Path | None = None,
        installed_hmms: str | None = None,
        run_index: int | None = None,
    ) -> Path:
        outdir = self.root / f"{name}-out"
        hmm = hmm or self.hmm
        hmm_selection = (
            ["--installed_hmms", installed_hmms]
            if installed_hmms is not None
            else ["--hmm_in", str(hmm)]
        )
        if run_index is None:
            suffix = name.rsplit("-", 1)[-1]
            run_index = int(suffix) if suffix.isdigit() else 1
        overrides = {}
        if probe is not None:
            overrides = {
                "LD_PRELOAD": str(self.probe_binary),
                "PLAN7_ASTRA_STAGE_PROBE": str(probe),
            }
        value = {
            "schema_version": 1,
            "cwd": str(self.root),
            "command": [
                str(self.astra),
                "search",
                "--prot_in",
                str(self.fasta),
                *hmm_selection,
                "--outdir",
                str(outdir),
                "--threads",
                str(threads),
            ],
            "executable": {
                "path": str(self.astra),
                "sha256": self._sha(self.astra),
            },
            "exit_code": exit_code,
            "host": {"hostname": "shared-node"},
            "environment": {"SLURM_JOB_ID": "12345"},
            "environment_overrides": overrides,
            "metadata": {
                "engine": "astra",
                "threads": threads,
                "run_status": "pilot",
                "run_kind": "replicate",
                "run_index": run_index,
                "dataset_id": "test-dataset",
                "role": role,
                "astra_version": "astra-hmm 0.1.0",
                "pyhmmer_version": "0.12.0",
                "astra_revision": "abc123",
            },
            "timing": {"wall_seconds": wall_seconds},
        }
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def _probe(
        self,
        name: str,
        *,
        pipeline_ns: int,
        msv_ns: int,
        ssv_noresult: int = 2,
    ) -> Path:
        calls = {
            "p7_Pipeline": 100,
            "p7_bg_NullOne": 100,
            "p7_MSVFilter": 100,
            "p7_SSVFilter": 100,
            "p7_bg_FilterScore": 20,
            "p7_ViterbiFilter": 10,
            "p7_ForwardParser": 5,
            "p7_BackwardParser": 4,
            "p7_domaindef_ByPosteriorHeuristics": 4,
            "p7_Forward": 8,
            "p7_Backward": 4,
            "p7_DomainDecoding": 4,
        }
        elapsed = {
            stage: max(1, pipeline_ns // (index + 1))
            for index, stage in enumerate(EXPECTED_STAGES)
        }
        elapsed["p7_Pipeline"] = pipeline_ns
        elapsed["p7_MSVFilter"] = msv_ns
        rows = [
            "#schema\tplan7_astra_stage_probe\t1",
            "#clock\tCLOCK_MONOTONIC",
            "#aggregation\tprocess_wide_inclusive",
            f"#target_library\t{self.target_library}",
            (
                "#observer_overhead\ttwo_clock_gettime_calls_and_relaxed_"
                "atomic_updates_per_observed_call"
            ),
            "#clock_errors\t0",
            (
                "stage\tcalls\telapsed_ns\tstatus_ok\tstatus_erange\t"
                "status_noresult\tstatus_other"
            ),
        ]
        for stage in EXPECTED_STAGES:
            noresult = ssv_noresult if stage == "p7_SSVFilter" else 0
            ok = calls[stage] - noresult
            rows.append(
                f"{stage}\t{calls[stage]}\t{elapsed[stage]}\t{ok}\t0\t{noresult}\t0"
            )
        path = self.root / f"{name}.tsv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    def _valid_inputs(self):
        hit_paths = [
            self._write(f"hits-{index}.tsv", self.hit_bytes) for index in range(4)
        ]
        controls = [
            (self._run_record("control-1", 18.0), hit_paths[0]),
            (self._run_record("control-2", 20.0), hit_paths[1]),
        ]
        probe_1 = self._probe(
            "probe-1", pipeline_ns=8_000_000_000, msv_ns=4_000_000_000
        )
        probe_2 = self._probe(
            "probe-2", pipeline_ns=10_000_000_000, msv_ns=5_000_000_000
        )
        observed = [
            (
                self._run_record(
                    "observed-1",
                    20.0,
                    role="astra_in_process_stage_probe",
                    probe=probe_1,
                ),
                probe_1,
                hit_paths[2],
            ),
            (
                self._run_record(
                    "observed-2",
                    22.0,
                    role="astra_in_process_stage_probe",
                    probe=probe_2,
                ),
                probe_2,
                hit_paths[3],
            ),
        ]
        return observed, controls

    def _summarize(self, observed, controls):
        return build_summary(
            observed,
            controls,
            fasta=self.fasta,
            hmm=self.hmm,
            probe_source=self.probe_source,
            probe_binary=self.probe_binary,
        )

    def test_summary_computes_strict_serial_probe_metrics(self) -> None:
        observed, controls = self._valid_inputs()
        summary = self._summarize(observed, controls)

        self.assertTrue(summary["scientific_output"]["byte_identical"])
        self.assertEqual(summary["scientific_output"]["files_compared"], 4)
        self.assertEqual(summary["engine"]["version"], "astra-hmm 0.1.0")
        self.assertEqual(summary["engine"]["pyhmmer_version"], "0.12.0")
        self.assertEqual(summary["engine"]["revision"], "abc123")
        self.assertEqual(summary["workload"]["comparisons_per_run"], 100)
        self.assertEqual(summary["probe"]["ssv"]["full_msv_fallback_calls_per_run"], 2)
        self.assertAlmostEqual(
            summary["probe"]["ssv"]["full_msv_fallback_fraction"], 0.02
        )

        timing = summary["timing"]
        self.assertEqual(timing["external_wall_seconds"]["control"]["median"], 19.0)
        self.assertEqual(timing["external_wall_seconds"]["observed"]["median"], 21.0)
        self.assertEqual(timing["observer_overhead_estimate"]["seconds"], 2.0)
        self.assertEqual(timing["inclusive_stage_seconds_median"]["p7_MSVFilter"], 4.5)
        self.assertAlmostEqual(timing["fractions"]["msv_of_pipeline"], 0.5)
        expected_wall_fraction = statistics_median((4.0 / 20.0, 5.0 / 22.0))
        self.assertAlmostEqual(
            timing["fractions"]["msv_of_observed_external_wall"],
            expected_wall_fraction,
        )
        self.assertAlmostEqual(
            timing["amdahl_end_to_end_speedup"]["10x"],
            1.0 / ((1.0 - expected_wall_fraction) + expected_wall_fraction / 10.0),
        )
        self.assertAlmostEqual(
            timing["amdahl_end_to_end_speedup"]["infinite"],
            1.0 / (1.0 - expected_wall_fraction),
        )
        self.assertEqual(
            summary["probe"]["target_library"]["sha256"],
            self._sha(self.target_library),
        )

    def test_atomic_output_is_compact_valid_json(self) -> None:
        observed, controls = self._valid_inputs()
        summary = self._summarize(observed, controls)
        output = self.root / "summary.json"
        atomic_compact_json(output, summary)
        text = output.read_text(encoding="utf-8")
        self.assertEqual(len(text.splitlines()), 1)
        self.assertEqual(json.loads(text), summary)

    def test_rejects_scientific_output_mismatch(self) -> None:
        observed, controls = self._valid_inputs()
        observed[1][2].write_bytes(self.hit_bytes + b"different\n")
        with self.assertRaisesRegex(SummaryError, "not byte-identical"):
            self._summarize(observed, controls)

    def test_rejects_different_probe_count_sets(self) -> None:
        observed, controls = self._valid_inputs()
        observed[1] = (
            observed[1][0],
            self._probe(
                "probe-2",
                pipeline_ns=10_000_000_000,
                msv_ns=5_000_000_000,
                ssv_noresult=3,
            ),
            observed[1][2],
        )
        with self.assertRaisesRegex(SummaryError, "count sets differ"):
            self._summarize(observed, controls)

    def test_rejects_failed_or_nonserial_astra_commands(self) -> None:
        observed, controls = self._valid_inputs()
        bad_serial = self._run_record("bad-serial", 18.0, threads=2)
        with self.assertRaisesRegex(SummaryError, "not serial"):
            self._summarize(observed, [(bad_serial, controls[0][1])])

        failed = self._run_record("failed", 18.0, exit_code=1)
        with self.assertRaisesRegex(SummaryError, "command failed"):
            self._summarize(observed, [(failed, controls[0][1])])

    def test_rejects_wrong_role_or_probe_environment_binding(self) -> None:
        observed, controls = self._valid_inputs()
        with self.assertRaisesRegex(SummaryError, "unexpected run role"):
            self._summarize(
                [(controls[0][0], observed[0][1], controls[0][1])],
                [controls[0]],
            )

        control_record = json.loads(controls[0][0].read_text(encoding="utf-8"))
        control_record["environment"]["LD_PRELOAD"] = str(self.probe_binary)
        controls[0][0].write_text(json.dumps(control_record), encoding="utf-8")
        with self.assertRaisesRegex(SummaryError, "control run preloads"):
            self._summarize(observed, controls)
        del control_record["environment"]["LD_PRELOAD"]
        controls[0][0].write_text(json.dumps(control_record), encoding="utf-8")

        record = json.loads(observed[0][0].read_text(encoding="utf-8"))
        original_report = record["environment_overrides"]["PLAN7_ASTRA_STAGE_PROBE"]
        record["environment_overrides"]["PLAN7_ASTRA_STAGE_PROBE"] = str(observed[1][1])
        observed[0][0].write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(SummaryError, "supplied probe TSV differ"):
            self._summarize(observed, controls)

        record["environment_overrides"]["PLAN7_ASTRA_STAGE_PROBE"] = original_report
        record["environment_overrides"]["LD_PRELOAD"] = str(self.target_library)
        observed[0][0].write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(SummaryError, "supplied probe exactly once"):
            self._summarize(observed, controls)

    def test_installed_database_binds_pressed_runtime_artifacts(self) -> None:
        database_dir = self.root / "PFAM"
        database_dir.mkdir()
        source_hmm = database_dir / "Pfam-A.hmm"
        source_hmm.write_bytes(b"HMMER3/f\nNAME installed\n//\n")
        pressed_paths = []
        for suffix in ("h3f", "h3i", "h3m", "h3p"):
            path = database_dir / f"PFAM.{suffix}"
            path.write_bytes(f"pressed-{suffix}\n".encode())
            pressed_paths.append(path)
        config = self.root / "hmm_databases.json"
        config.write_text(
            json.dumps(
                {
                    "db_urls": [
                        {
                            "name": "PFAM",
                            "installation_dir": str(database_dir),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        probe = self._probe(
            "installed-probe", pipeline_ns=8_000_000_000, msv_ns=4_000_000_000
        )
        control_hits = self._write("installed-control-hits.tsv", self.hit_bytes)
        observed_hits = self._write("installed-observed-hits.tsv", self.hit_bytes)
        control = self._run_record(
            "installed-control-1",
            18.0,
            hmm=source_hmm,
            installed_hmms="PFAM",
        )
        observed = self._run_record(
            "installed-observed-1",
            20.0,
            role="astra_in_process_stage_probe",
            probe=probe,
            hmm=source_hmm,
            installed_hmms="PFAM",
        )

        summary = build_summary(
            [(observed, probe, observed_hits)],
            [(control, control_hits)],
            fasta=self.fasta,
            hmm=source_hmm,
            probe_source=self.probe_source,
            probe_binary=self.probe_binary,
            astra_config=config,
        )
        runtime = summary["workload"]["hmm_runtime"]
        self.assertEqual(runtime["runtime_format"], "pressed_hmm")
        self.assertEqual(
            [item["sha256"] for item in runtime["runtime_artifacts"]],
            [self._sha(path) for path in pressed_paths],
        )

        with self.assertRaisesRegex(SummaryError, "not a source in installed"):
            build_summary(
                [(observed, probe, observed_hits)],
                [(control, control_hits)],
                fasta=self.fasta,
                hmm=self.hmm,
                probe_source=self.probe_source,
                probe_binary=self.probe_binary,
                astra_config=config,
            )


def statistics_median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


if __name__ == "__main__":
    unittest.main()
