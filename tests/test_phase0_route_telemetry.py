import inspect
import sys
import threading
import time
import types
import unittest
from unittest import mock
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import _native, _telemetry
    from plan7_gpu import astra_search as _astra_search
    from plan7_gpu.adapter import CandidateBatch, _new_candidate_batch
    from plan7_gpu.astra_search import _search_row
except ImportError:
    _native = None
    _telemetry = None
    _astra_search = None
    CandidateBatch = None
    _new_candidate_batch = None
    _search_row = None


def valid_transport():
    metrics = (
        10,  # model length
        3,  # target count
        20,  # target residues
        2,  # F1 candidates
        1,  # F1 rejects outside the candidate CSR
        200,  # F1 logical cells
        150,  # postfilter logical cells
        1,  # F2 passes
        50,  # Forward logical cells
        0,  # Forward CPU
        0,  # Forward reject
        1,  # Forward pass
        50,  # Backward logical cells
        0,  # Backward CPU
        0,  # Backward no-region
        1,  # Backward simple
        1,  # journal rows
        1,  # journal regions
        20,  # rescore logical cells
        0,  # rescore CPU
        1,  # rescore device
        1,  # rescore regions
        0,  # rescore certified GA rejects
    )
    postfilter = [0] * 16
    postfilter[0] = 1
    postfilter[10] = 1
    postfilter[11] = 1
    postfilter[14] = 1
    postfilter[15] = 1
    f2 = [0] * 5
    f2[0] = 1
    f2[4] = 1
    forward = [0] * 10
    forward[8] = 1
    backward = [0] * 18
    backward[14] = 1
    rescore = [0] * len(_telemetry._REASON_FACTS["rescore"])
    rescore[21] = 1

    postfilter_cells = [0] * 16
    postfilter_cells[0] = 50
    postfilter_cells[10] = 50
    postfilter_cells[11] = 100
    postfilter_cells[14] = 50
    postfilter_cells[15] = 100
    f2_cells = [0] * 5
    forward_cells = [0] * 10
    forward_cells[8] = 50
    backward_cells = [0] * 18
    backward_cells[14] = 50
    rescore_cells = [0] * len(_telemetry._REASON_FACTS["rescore"])
    rescore_cells[21] = 20

    profile_records = (
        (
            metrics,
            tuple(
                tuple(values)
                for values in (postfilter, f2, forward, backward, rescore)
            ),
            tuple(
                tuple(values)
                for values in (
                    postfilter_cells,
                    f2_cells,
                    forward_cells,
                    backward_cells,
                    rescore_cells,
                )
            ),
        ),
    )
    native_totals = {
        "postfilter": {
            "candidate_count": 2,
            "full_msv_execution_count": 1,
            "viterbi_execution_count": 1,
            "full_msv_work_cells": 50,
            "viterbi_work_cells": 100,
            "work_cells": 150,
        },
        "forward": {
            "candidate_count": 1,
            "survivor_count": 1,
            "work_cells": 50,
            "output_cap_fallback_count": 0,
        },
        "backward_domain": {
            "candidate_count": 1,
            "device_result_count": 1,
            "cpu_required_count": 0,
            "work_cells": 50,
        },
        "rescore": {
            "region_count": 1,
            "device_result_count": 1,
            "cpu_required_count": 0,
            "certified_ga_region_count": 0,
            "certified_ga_row_count": 0,
            "certified_ga_skipped_work_cells": 0,
            "ga_classification_ms": 0.0,
            "work_cells": 20,
        },
    }
    return profile_records, native_totals


@unittest.skipIf(_telemetry is None, "telemetry module unavailable")
class GenerationTelemetryTests(unittest.TestCase):
    def build(self):
        records, native = valid_transport()
        return _telemetry.build_generation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            1, 3, 20, records, 0, 4096, native
        )

    def test_schema_preserves_exact_bits_aggregates_cells_and_journal(self):
        value = self.build()
        self.assertEqual(
            value["schema_version"],
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
        )
        self.assertEqual(value["scope"], "generation")
        self.assertEqual(value["journal"]["allocation_bytes"], 4096)
        profile = value["profiles"][0]
        self.assertEqual(profile["journal"], {"row_count": 1, "region_count": 1})
        self.assertEqual(profile["logical_cells"]["forward_logical_cells"], 50)
        self.assertEqual(
            dict(profile["reason_counts"]["postfilter"])["raw_f1_reject"],
            1,
        )
        self.assertEqual(
            dict(profile["reason_logical_cells"]["rescore"])["device_result"],
            20,
        )
        self.assertEqual(
            dict(value["reason_fact_bits"]["f2"]),
            {
                "postfilter_not_pass_or_host_environment_unattested": 0x01,
                "input_invalid": 0x02,
                "msv_threshold_exceeded": 0x04,
                "viterbi_threshold_exceeded": 0x08,
                "pass": 0x10,
            },
        )
        self.assertEqual(_telemetry.validate_generation_statistics(value), value)
        self.assertEqual(
            _native.GENERATION_REASON_FACT_LAYOUT,
            tuple(
                tuple(bit for _name, bit in value["reason_fact_bits"][stage])
                for stage in value["reason_fact_bits"]
            ),
        )

    def test_validation_is_deep_and_defensive(self):
        value = self.build()
        stored = _telemetry.validate_generation_statistics(value)
        returned = _telemetry.defensive_generation_statistics(stored)
        returned["profiles"][0]["counts"]["f2_pass_count"] = 999
        returned["native_totals"]["forward"]["work_cells"] = 999
        self.assertEqual(stored["profiles"][0]["counts"]["f2_pass_count"], 1)
        self.assertEqual(stored["native_totals"]["forward"]["work_cells"], 50)

    def test_sealed_batch_identity_binding_is_exact_and_one_way(self):
        unbound = self.build()
        self.assertIsNone(unbound["batch_identity"])
        bound = _telemetry.bind_generation_statistics_identity(
            unbound, (11, 22, 33)
        )
        self.assertEqual(bound["batch_identity"], {
            "session_id": 11,
            "selection_id": 22,
            "batch_generation": 33,
        })
        self.assertIsNone(unbound["batch_identity"])
        self.assertEqual(
            _telemetry.bind_generation_statistics_identity(
                bound, (11, 22, 33)
            ),
            bound,
        )
        with self.assertRaisesRegex(ValueError, "identity changed"):
            _telemetry.bind_generation_statistics_identity(
                bound, (11, 22, 34)
            )

    def test_route_native_and_reason_cell_mismatches_fail_closed(self):
        records, native = valid_transport()
        cases = []

        metrics = list(records[0][0])
        metrics[9] = 1
        cases.append(((tuple(metrics), records[0][1], records[0][2]), native))

        bad_native = {
            **native,
            "forward": {**native["forward"], "work_cells": 49},
        }
        cases.append((records[0], bad_native))

        bad_postfilter_native = {
            **native,
            "postfilter": {
                **native["postfilter"],
                "viterbi_execution_count": 0,
            },
        }
        cases.append((records[0], bad_postfilter_native))

        cells = [list(row) for row in records[0][2]]
        cells[2][8] = 51
        cases.append(
            (
                (records[0][0], records[0][1], tuple(tuple(row) for row in cells)),
                native,
            )
        )

        missing_backward_terminal = [list(row) for row in records[0][1]]
        missing_backward_terminal[3][14] = 0
        cases.append(
            (
                (
                    records[0][0],
                    tuple(tuple(row) for row in missing_backward_terminal),
                    records[0][2],
                ),
                native,
            )
        )

        missing_rescore_terminal = [list(row) for row in records[0][1]]
        missing_rescore_terminal[4][21] = 0
        cases.append(
            (
                (
                    records[0][0],
                    tuple(tuple(row) for row in missing_rescore_terminal),
                    records[0][2],
                ),
                native,
            )
        )

        for profile, native_totals in cases:
            with self.subTest(profile=profile, native=native_totals):
                with self.assertRaises(ValueError):
                    _telemetry.build_generation_statistics(
                        _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
                        1, 3, 20, (profile,), 0, 4096, native_totals
                    )

    def test_f2_msv_failure_then_viterbi_pass_is_valid_multilabel_history(self):
        records, native = valid_transport()
        reasons = [list(row) for row in records[0][1]]
        reasons[1][2] = 1  # MSV threshold exceeded before Viterbi rescued it.
        profile = (
            records[0][0],
            tuple(tuple(row) for row in reasons),
            records[0][2],
        )
        value = _telemetry.build_generation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            1, 3, 20, (profile,), 0, 4096, native
        )
        f2 = dict(value["profiles"][0]["reason_counts"]["f2"])
        self.assertEqual(f2["msv_threshold_exceeded"], 1)
        self.assertEqual(f2["pass"], 1)

        reasons[1][2] = 0
        reasons[1][3] = 1
        reasons[1][4] = 0
        invalid = (
            records[0][0],
            tuple(tuple(row) for row in reasons),
            records[0][2],
        )
        with self.assertRaisesRegex(
            ValueError, "MSV/Viterbi transition attribution"
        ):
            _telemetry.build_generation_statistics(
                _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
                1, 3, 20, (invalid,), 0, 4096, native
            )

    def test_forward_call_fallback_does_not_claim_unrun_row_transitions(self):
        records, native = valid_transport()
        metrics = list(records[0][0])
        metrics[8:22] = (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        reasons = list(records[0][1])
        reasons[2] = (0,) * 10
        reasons[3] = (0,) * 18
        reasons[4] = (0,) * len(_telemetry._REASON_FACTS["rescore"])
        reason_cells = list(records[0][2])
        reason_cells[2] = (0,) * 10
        reason_cells[3] = (0,) * 18
        reason_cells[4] = (0,) * len(_telemetry._REASON_FACTS["rescore"])
        native = {
            "postfilter": native["postfilter"],
            "forward": {
                "candidate_count": 1,
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
        }
        value = _telemetry.build_generation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            1,
            3,
            20,
            ((tuple(metrics), tuple(reasons), tuple(reason_cells)),),
            1,
            4096,
            native,
        )
        self.assertEqual(
            value["call_reason_facts"]["forward"],
            (("contract_fallback", 0x01),),
        )
        self.assertEqual(value["profiles"][0]["reason_counts"]["forward"], ())

    def test_active_rescore_own_scales_cells_reconcile_native_work(self):
        records, native = valid_transport()
        metrics = list(records[0][0])
        metrics[19] = 1
        metrics[20] = 0
        reasons = [list(row) for row in records[0][1]]
        reason_cells = [list(row) for row in records[0][2]]
        reasons[4][1] = 1  # isolated Backward discovered own scales
        reasons[4][17] = 1  # rejected by exact host result validation
        reasons[4][20] = 1  # row-atomic CPU propagation
        reasons[4][21] = 0  # no device result on the CPU route
        reasons[4][24] = 1  # exact final CPU route
        reason_cells[4][1] = 20
        reason_cells[4][17] = 20
        reason_cells[4][20] = 20
        reason_cells[4][21] = 0
        reason_cells[4][24] = 20
        native = {
            **native,
            "rescore": {
                **native["rescore"],
                "device_result_count": 0,
                "cpu_required_count": 1,
            },
        }
        value = _telemetry.build_generation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            1,
            3,
            20,
            ((
                tuple(metrics),
                tuple(tuple(row) for row in reasons),
                tuple(tuple(row) for row in reason_cells),
            ),),
            0,
            4096,
            native,
        )
        cells = dict(
            value["profiles"][0]["reason_logical_cells"]["rescore"]
        )
        self.assertEqual(cells["own_scales"], 20)

    def test_mutated_nested_sidecar_is_revalidated_not_merely_copied(self):
        value = self.build()
        value["profiles"][0]["journal"]["row_count"] = 0
        with self.assertRaises(ValueError):
            _telemetry.validate_generation_statistics(value)

    def test_continuation_routes_cover_omitted_f1_rows_and_exact_retries(self):
        value = _telemetry.build_continuation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            "journal",
            123,
            10,
            6,
            (4, 1, 1, 0, 2, 1, 1),
            (4, 1, 1, 2),
            (2, 1, 1, 0, 9, 2, 7, 1),
            (1, 1, 0, 1, 3, 1),
            (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            (2, 8, 12),
        )
        self.assertEqual(value["target_count"], 10)
        self.assertEqual(value["postfilter_record_count"], 6)
        self.assertEqual(value["routes"]["f1_reject_count"], 4)
        self.assertEqual(value["compact"]["first_attempt"], (9, 2, 7, 1))

    def test_continuation_partitions_fail_closed(self):
        valid = dict(
            schema_version=_telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            path="journal",
            wall_ns=1,
            target_count=4,
            postfilter_record_count=3,
            route_counts=(1, 0, 0, 0, 1, 0, 2),
            journal_counts=(3, 1, 0, 2),
            compact_counts=(2, 2, 0, 0, 1, 0, 1, 1),
            source_counts=(0, 0, 0, 1, 2, 0),
            decision_counts=(0,) * 12,
            identity=(0, 0, 3),
        )
        cases = (
            {**valid, "route_counts": (0, 0, 0, 0, 0, 1, 2)},
            {**valid, "journal_counts": (3, 0, 0, 2)},
            {**valid, "compact_counts": (3, 2, 0, 0, 1, 0, 1, 1)},
            {**valid, "path": "forward"},
            {**valid, "identity": (1, 0, 3)},
            {
                **valid,
                "compact_counts": (2, 2, 0, 0, 3, 0, 1, 1),
            },
            {
                **valid,
                "source_counts": (0, 0, 0, 1, 2, 1),
            },
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _telemetry.build_continuation_statistics(**values)

    def test_continuation_bypass_facts_are_path_specific_and_complete(self):
        value = _telemetry.build_continuation_statistics(
            _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
            "forward",
            1,
            2,
            1,
            (1, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0),
        )
        self.assertEqual(
            value["decision_facts"]["journal"]["storage_unavailable"], 1
        )
        with self.assertRaisesRegex(ValueError, "decision facts"):
            _telemetry.build_continuation_statistics(
                _telemetry.GENERATION_TELEMETRY_SCHEMA_VERSION,
                "forward",
                1,
                2,
                1,
                (1, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0),
                (0,) * 12,
                (0, 0, 0),
            )


@unittest.skipIf(_native is None, "native extension unavailable")
class ActiveSourceReasonRemapTests(unittest.TestCase):
    def test_postfilter_work_is_exactly_zero_one_or_two_l_times_m(self):
        full = 0x4000
        viterbi = 0x8000
        self.assertEqual(_native.postfilter_execution_cells_for_test(7, 11, 0), 0)
        self.assertEqual(
            _native.postfilter_execution_cells_for_test(7, 11, full), 77
        )
        self.assertEqual(
            _native.postfilter_execution_cells_for_test(7, 11, viterbi), 77
        )
        self.assertEqual(
            _native.postfilter_execution_cells_for_test(
                7, 11, full | viterbi
            ),
            154,
        )
        with self.assertRaises(OverflowError):
            _native.postfilter_execution_cells_for_test(1 << 63, 2, full)

    def test_backward_active_rows_restore_original_sparse_ordinals(self):
        result = _native.backward_domain_merge_reason_facts_for_test(
            array("Q", [1, 4]),
            array("I", [0x0100, 0x4000]),
            array("I", [1, 2, 3, 4, 5, 6]),
        )
        self.assertEqual(list(result), [1, 0x0100, 3, 4, 0x4000, 6])

    def test_rescore_active_regions_restore_original_sparse_ordinals(self):
        result = _native.domain_rescore_merge_reason_facts_for_test(
            array("I", [0, 3, 7]),
            array("I", [0x80, 0x200000, 0x400]),
            array("I", range(10)),
        )
        self.assertEqual(
            list(result),
            [0x80, 1, 2, 0x200000, 4, 5, 6, 0x400, 8, 9],
        )

    def test_rescore_active_own_scales_retains_work_attribution(self):
        self.assertTrue(
            _native.domain_rescore_reason_admitted_work_for_test(0x00000002)
        )
        self.assertFalse(
            _native.domain_rescore_reason_admitted_work_for_test(0x00800000)
        )
        self.assertFalse(
            _native.domain_rescore_reason_admitted_work_for_test(0x00000010)
        )

    def test_invalid_or_duplicate_source_ordinals_fail_closed(self):
        with self.assertRaises(ValueError):
            _native.backward_domain_merge_reason_facts_for_test(
                array("Q", [2, 2]), array("I", [1, 2]), array("I", [0, 0, 0])
            )
        with self.assertRaises(ValueError):
            _native.domain_rescore_merge_reason_facts_for_test(
                array("I", [3]), array("I", [1]), array("I", [0, 0, 0])
            )


@unittest.skipIf(_search_row is None, "Astra bridge unavailable")
class DisabledShapeAndSearchTelemetryTests(unittest.TestCase):
    class Candidates:
        def __init__(self):
            self.calls = []

        def search(self, row, pipeline, *, return_telemetry=False):
            self.calls.append((row, pipeline, return_telemetry))
            return ("hits", {"scope": "continuation"}) if return_telemetry else "hits"

    class Pipeline:
        def __init__(self):
            self.clear_count = 0

        def clear(self):
            self.clear_count += 1

    def test_search_row_default_and_opt_in_shapes_are_exact(self):
        candidates = self.Candidates()
        pipeline = self.Pipeline()
        self.assertEqual(_search_row(candidates, 7, pipeline, False), "hits")
        self.assertEqual(
            _search_row(candidates, 8, pipeline, True),
            ("hits", {"scope": "continuation"}),
        )
        self.assertEqual(
            candidates.calls,
            [(7, pipeline, False), (8, pipeline, True)],
        )
        self.assertEqual(pipeline.clear_count, 2)

    def test_threaded_astra_opt_in_preserves_query_order(self):
        class Pipeline(self.Pipeline):
            pass

        class Candidates:
            def __len__(self):
                return 10

            def search(self, row, pipeline, *, return_telemetry=False):
                time.sleep(0.001 * (row % 3))
                return (
                    (row, {"row": row, "thread": threading.get_ident()})
                    if return_telemetry
                    else row
                )

        replacement = types.SimpleNamespace(
            plan7=types.SimpleNamespace(Pipeline=lambda **_options: Pipeline())
        )
        with mock.patch.object(_astra_search, "pyhmmer", replacement):
            observed = list(
                _astra_search._threaded_hmmsearch(
                    Candidates(), 3, {}, True
                )
            )
        self.assertEqual([row for row, _telemetry in observed], list(range(10)))
        self.assertEqual(
            [record["row"] for _row, record in observed], list(range(10))
        )

    def test_public_defaults_and_private_capsule_default_are_preserved(self):
        self.assertFalse(
            inspect.signature(CandidateBatch.search)
            .parameters["return_telemetry"]
            .default
        )
        native_source = (ROOT / "python/plan7_gpu/_native.pyx").read_text()
        backward_source = (ROOT / "cuda/backward_domain_cuda.cu").read_text()
        rescore_source = (ROOT / "cuda/domain_rescore_cuda.cu").read_text()
        self.assertIn("bint _return_generation_statistics=False", native_source)
        self.assertIn("return result[0]", native_source)
        self.assertIn(
            "posterior_byte_budget, false, false,\n        output",
            backward_source,
        )
        self.assertIn(
            "matrix_byte_budget, trace_byte_budget, false,\n"
            "        nullptr, 0, nullptr, 0,\n        output",
            rescore_source,
        )

    def test_plain_candidate_has_no_generation_sidecar(self):
        candidates = _new_candidate_batch(
            (), object(), b"shared", array("I"), array("Q", [0]), b"",
            array("I"), 0.02,
        )
        self.assertIsNone(candidates.generation_statistics)

    def test_compiled_continuation_builder_uses_current_schema_constant(self):
        source = (ROOT / "python/plan7_gpu/_pipeline.pyx").read_text(
            encoding="utf-8"
        )
        marker = "_telemetry_module.build_continuation_statistics("
        start = source.index(marker)
        call = source[start:start + 256]
        self.assertIn(
            "_telemetry_module.GENERATION_TELEMETRY_SCHEMA_VERSION",
            call,
        )


if __name__ == "__main__":
    unittest.main()
