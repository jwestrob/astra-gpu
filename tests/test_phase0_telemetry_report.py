import copy
from array import array
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import types
import unittest
from unittest import mock

from plan7_gpu import _telemetry
from plan7_gpu import astra_search
from plan7_gpu.adapter import _new_candidate_batch
from plan7_gpu.telemetry_report import REPORT_SCHEMA_VERSION, TelemetryCollector
from plan7_gpu.telemetry_report import TelemetryReportCommittedError


BATCH_IDENTITY = (101, 202, 303)


def generation_statistics():
    profiles = []
    for model_length in (5, 7):
        metrics = (
            model_length,
            2,
            10,
            2,
            0,
            model_length * 10,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        reasons = []
        cells = []
        for stage, facts in _telemetry._REASON_FACTS.items():
            row = [0] * len(facts)
            if stage == "postfilter":
                row[10] = 2
            elif stage == "f2":
                row[0] = 2
            reasons.append(tuple(row))
            cells.append((0,) * len(facts))
        profiles.append((metrics, tuple(reasons), tuple(cells)))
    return _telemetry.build_generation_statistics(
        _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
        2,
        2,
        10,
        tuple(profiles),
        0,
        128,
        {
            "postfilter": {
                "candidate_count": 4,
                "full_msv_execution_count": 0,
                "viterbi_execution_count": 0,
                "full_msv_work_cells": 0,
                "viterbi_work_cells": 0,
                "work_cells": 0,
            },
            "forward": {
                "candidate_count": 0,
                "survivor_count": 0,
                "work_cells": 0,
                "output_cap_fallback_count": 0,
            },
            "backward_domain": {
                "candidate_count": 0,
                "device_result_count": 0,
                "cpu_required_count": 0,
                "work_cells": 0,
            },
            "rescore": None,
        },
        BATCH_IDENTITY,
    )


def continuation_statistics(local_index, wall_ns):
    return _telemetry.build_continuation_statistics(
        _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
        "postfilter",
        wall_ns,
        2,
        2,
        (0, 0, 2, 0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 2, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (local_index, 0, 0),
        BATCH_IDENTITY,
    )


def journal_generation_statistics():
    from tests.test_phase0_route_telemetry import valid_transport

    records, native = valid_transport()
    doubled_native = {
        stage: (
            None
            if values is None
            else {name: value * 2 for name, value in values.items()}
        )
        for stage, values in native.items()
    }
    return _telemetry.build_generation_statistics(
        _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
        2,
        3,
        20,
        (records[0], records[0]),
        0,
        8192,
        doubled_native,
        BATCH_IDENTITY,
    )


def journal_continuation_statistics(local_index, row_start):
    return _telemetry.build_continuation_statistics(
        _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
        "journal",
        10,
        3,
        2,
        (1, 0, 1, 0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 0, row_start, local_index, 1, 1),
        (0, 1, 0, 0, 1, 0),
        (0,) * 12,
        (local_index, row_start, row_start + 1),
        BATCH_IDENTITY,
    )


class ContinuationValidationTests(unittest.TestCase):
    def test_round_trip_is_deep_and_exact(self):
        value = continuation_statistics(0, 17)
        observed = _telemetry.validate_continuation_statistics(value)
        self.assertEqual(observed, value)
        self.assertIsNot(observed, value)
        changed = copy.deepcopy(value)
        changed["routes"]["definite_reject_count"] = 1
        with self.assertRaises(ValueError):
            _telemetry.validate_continuation_statistics(changed)
        changed = copy.deepcopy(value)
        changed["compact"]["first_attempt"] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "first attempt"):
            _telemetry.validate_continuation_statistics(changed)


class TelemetryCollectorTests(unittest.TestCase):
    def collector(self):
        collector = TelemetryCollector()
        collector.bind_expected_profiles((10, 20))
        source = generation_statistics()
        collector.record_generation(
            source,
            (10, 20),
            profile_keys=("alpha", "beta"),
        )
        source["profiles"][0]["counts"]["f1_candidate_count"] = 0
        return collector

    def test_join_reason_summary_and_exact_cpu_wall_pareto(self):
        collector = self.collector()
        collector.record_continuation(10, continuation_statistics(0, 100))
        collector.record_continuation(20, continuation_statistics(1, 300))
        snapshot = collector.snapshot()
        self.assertTrue(snapshot["complete"])
        self.assertEqual(snapshot["schema_version"], REPORT_SCHEMA_VERSION)
        self.assertEqual(snapshot["continuation_cpu_wall_ns"], 400)
        self.assertEqual(
            [row["profile_ordinal"] for row in snapshot["pareto"]],
            [20, 10],
        )
        self.assertEqual(
            [row["cumulative_cpu_wall_ns"] for row in snapshot["pareto"]],
            [300, 400],
        )
        final_reject = next(
            row
            for row in snapshot["reason_totals"]
            if row["stage"] == "postfilter"
            and row["reason"] == "final_reject"
        )
        self.assertEqual(final_reject, {
            "stage": "postfilter",
            "reason": "final_reject",
            "rows": 4,
            "logical_cells": 0,
        })

    def test_missing_duplicate_and_identity_drift_fail_closed(self):
        collector = self.collector()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            collector.snapshot()
        with self.assertRaisesRegex(ValueError, "already recorded"):
            collector.record_generation(generation_statistics(), (10, 20))
        with self.assertRaisesRegex(ValueError, "local profile identity"):
            collector.record_continuation(10, continuation_statistics(1, 5))
        collector.record_continuation(10, continuation_statistics(0, 5))
        with self.assertRaisesRegex(ValueError, "already recorded"):
            collector.record_continuation(10, continuation_statistics(0, 5))

        duplicate_batch = TelemetryCollector()
        duplicate_batch.bind_expected_profiles((10, 20, 30, 40))
        duplicate_batch.record_generation(generation_statistics(), (10, 20))
        with self.assertRaisesRegex(ValueError, "batch identity"):
            duplicate_batch.record_generation(
                generation_statistics(), (30, 40)
            )

    def test_cross_side_counts_batch_and_journal_prefix_are_exact(self):
        collector = self.collector()
        changed = _telemetry.build_continuation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            "postfilter",
            1,
            3,
            2,
            (1, 0, 2, 0, 0, 0, 0),
            (0, 0, 0, 0),
            (0,) * 8,
            (0, 2, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0),
            BATCH_IDENTITY,
        )
        with self.assertRaisesRegex(ValueError, "target count"):
            collector.record_continuation(10, changed)

        changed = _telemetry.build_continuation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            "postfilter",
            1,
            2,
            1,
            (1, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0),
            (0,) * 8,
            (0, 1, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0),
            BATCH_IDENTITY,
        )
        with self.assertRaisesRegex(ValueError, "post-filter count"):
            collector.record_continuation(10, changed)

        changed = continuation_statistics(0, 1)
        changed["batch_identity"]["batch_generation"] += 1
        changed = _telemetry.validate_continuation_statistics(changed)
        with self.assertRaisesRegex(ValueError, "batch identity"):
            collector.record_continuation(10, changed)

        journal_collector = TelemetryCollector()
        journal_collector.bind_expected_profiles((10, 20))
        journal_collector.record_generation(
            journal_generation_statistics(), (10, 20)
        )
        journal_collector.record_continuation(
            10, journal_continuation_statistics(0, 0)
        )
        with self.assertRaisesRegex(ValueError, "journal attribution"):
            journal_collector.record_continuation(
                20, journal_continuation_statistics(1, 0)
            )
        journal_collector.record_continuation(
            20, journal_continuation_statistics(1, 1)
        )
        self.assertTrue(journal_collector.snapshot()["complete"])

    def test_unbound_or_incomplete_profile_universe_cannot_export(self):
        unbound = TelemetryCollector()
        with self.assertRaisesRegex(RuntimeError, "bound before generation"):
            unbound.record_generation(generation_statistics(), (10, 20))
        with self.assertRaisesRegex(RuntimeError, "universe is not bound"):
            unbound.snapshot()

        incomplete = TelemetryCollector()
        incomplete.bind_expected_profiles((10, 20, 30))
        incomplete.record_generation(generation_statistics(), (10, 20))
        incomplete.record_continuation(10, continuation_statistics(0, 1))
        incomplete.record_continuation(20, continuation_statistics(1, 1))
        with self.assertRaisesRegex(RuntimeError, "missing generation=30"):
            incomplete.snapshot()

        unbound_generation = generation_statistics()
        unbound_generation["batch_identity"] = None
        unbound_generation = _telemetry.validate_generation_statistics(
            unbound_generation
        )
        bound = TelemetryCollector()
        bound.bind_expected_profiles((10, 20))
        with self.assertRaisesRegex(ValueError, "sealed batch identity"):
            bound.record_generation(unbound_generation, (10, 20))

    def test_atomic_export_is_complete_hashed_and_read_only(self):
        collector = self.collector()
        collector.record_continuation(10, continuation_statistics(0, 100))
        collector.record_continuation(20, continuation_statistics(1, 300))
        with tempfile.TemporaryDirectory(prefix="phase0-report-test-") as root:
            target = Path(root) / "report"
            paths = collector.export(target)
            self.assertEqual(set(paths), {
                "phase0-raw.json",
                "phase0-profiles.tsv",
                "phase0-reasons.tsv",
                "phase0-reason-summary.tsv",
                "phase0-pareto.tsv",
                "artifact.sha256",
            })
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
            manifest = {}
            for line in paths["artifact.sha256"].read_text().splitlines():
                digest, name = line.split("  ", 1)
                manifest[name] = digest
            self.assertEqual(set(manifest), set(paths) - {"artifact.sha256"})
            for name, digest in manifest.items():
                payload = paths[name].read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
                self.assertEqual(stat.S_IMODE(paths[name].stat().st_mode), 0o444)
            raw = json.loads(paths["phase0-raw.json"].read_text())
            self.assertEqual(raw["continuation_cpu_wall_ns"], 400)
            self.assertFalse(any(Path(root).glob(".report.tmp-*")))
            os.chmod(target, 0o700)

    def test_parent_fsync_failure_reports_an_already_committed_target(self):
        collector = self.collector()
        collector.record_continuation(10, continuation_statistics(0, 100))
        collector.record_continuation(20, continuation_statistics(1, 300))
        with tempfile.TemporaryDirectory(prefix="phase0-report-fsync-") as root:
            parent = Path(root)
            target = parent / "report"
            original = __import__(
                "plan7_gpu.telemetry_report", fromlist=["_fsync_directory"]
            )._fsync_directory

            def fail_parent(path):
                if Path(path) == parent:
                    raise OSError("injected parent fsync failure")
                return original(path)

            with mock.patch(
                "plan7_gpu.telemetry_report._fsync_directory",
                side_effect=fail_parent,
            ):
                with self.assertRaises(TelemetryReportCommittedError) as caught:
                    collector.export(target)
            self.assertEqual(caught.exception.target, target)
            self.assertTrue(target.is_dir())
            self.assertTrue((target / "artifact.sha256").is_file())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o555)
            os.chmod(target, 0o700)


class LocalAstraTelemetryIntegrationTests(unittest.TestCase):
    def test_unsupported_batch_fails_before_candidate_preparation(self):
        plain = _new_candidate_batch(
            (), object(), b"shared", array("I"), array("Q", [0]), b"",
            array("I"), 0.02,
        )
        with mock.patch.object(astra_search, "_prepare_candidates") as prepare:
            with self.assertRaisesRegex(ValueError, "generated with telemetry"):
                astra_search.hmmsearch((), plain, telemetry=True)
        prepare.assert_not_called()

    def test_explicit_collector_preserves_hits_and_joins_ordinals(self):
        generation = generation_statistics()

        class Pair:
            def __init__(self, ordinal):
                self.ordinal = ordinal

        class Candidates:
            generation_statistics = generation
            F1 = 0.02

            def __len__(self):
                return 2

            def search(self, row, pipeline, *, return_telemetry=False):
                self.return_flags.append(return_telemetry)
                return f"hits-{row}", continuation_statistics(row, row + 1)

            return_flags = []

        class Pipeline:
            def clear(self):
                pass

        candidates = Candidates()
        pairs = (Pair(4), Pair(9))
        collector = TelemetryCollector()
        collector.bind_expected_profiles((40, 90))
        replacement = types.SimpleNamespace(
            plan7=types.SimpleNamespace(Pipeline=lambda **_options: Pipeline())
        )
        with (
            mock.patch.object(astra_search, "CandidateBatch", Candidates),
            mock.patch.object(astra_search, "PressedProfilePair", Pair),
            mock.patch.object(astra_search, "pyhmmer", replacement),
            mock.patch.object(
                astra_search,
                "_prepare_candidates",
                return_value=candidates,
            ),
        ):
            observed = list(
                astra_search.hmmsearch(
                    pairs,
                    candidates,
                    telemetry_collector=collector,
                    profile_ordinals=(40, 90),
                )
            )
        self.assertEqual(observed, ["hits-0", "hits-1"])
        self.assertEqual(candidates.return_flags, [True, True])
        snapshot = collector.snapshot()
        self.assertEqual(
            [row["profile_ordinal"] for row in snapshot["profiles"]],
            [40, 90],
        )

    def test_final_pipeline_clear_failure_cannot_complete_collector(self):
        generation = generation_statistics()

        class Pair:
            def __init__(self, ordinal):
                self.ordinal = ordinal

        class Candidates:
            generation_statistics = generation
            F1 = 0.02

            def __len__(self):
                return 2

            def search(self, row, pipeline, *, return_telemetry=False):
                return f"hits-{row}", continuation_statistics(row, row + 1)

        class Pipeline:
            clear_count = 0

            def clear(self):
                self.clear_count += 1
                if self.clear_count == 2:
                    raise RuntimeError("injected final clear failure")

        candidates = Candidates()
        pairs = (Pair(4), Pair(9))
        collector = TelemetryCollector()
        collector.bind_expected_profiles((40, 90))
        replacement = types.SimpleNamespace(
            plan7=types.SimpleNamespace(Pipeline=lambda **_options: Pipeline())
        )
        with (
            mock.patch.object(astra_search, "CandidateBatch", Candidates),
            mock.patch.object(astra_search, "PressedProfilePair", Pair),
            mock.patch.object(astra_search, "pyhmmer", replacement),
            mock.patch.object(
                astra_search,
                "_prepare_candidates",
                return_value=candidates,
            ),
        ):
            iterator = astra_search.hmmsearch(
                pairs,
                candidates,
                telemetry_collector=collector,
                profile_ordinals=(40, 90),
            )
            self.assertEqual(next(iterator), "hits-0")
            with self.assertRaisesRegex(RuntimeError, "final clear failure"):
                next(iterator)
        with self.assertRaisesRegex(RuntimeError, "continuation=90"):
            collector.snapshot()


if __name__ == "__main__":
    unittest.main()
