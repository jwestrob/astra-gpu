import math
import struct
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import pyhmmer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
    from plan7_gpu import _native
    from plan7_gpu.adapter import (
        _pair_state,
        _profile_selection_state,
        _sequence_state,
    )
except ImportError:
    ProfileSession = None
    SequenceBatch = None
    load_pressed_profiles = None
    _native = None
    _pair_state = None
    _profile_selection_state = None
    _sequence_state = None

try:
    from audit_cuda_domain_rescore_pfam import reconcile_final_domains
except ImportError:
    reconcile_final_domains = None


DATA = Path(pyhmmer.__file__).parent / "tests" / "data"
RESULT = struct.Struct("=9I5f4BI")
TRACE_STEP = struct.Struct("=IIfB3x")
UPSTREAM_RESULT = struct.Struct("=IIffIIIBBBB")


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


@unittest.skipIf(_native is None, "native extension unavailable")
class DomainRescoreHostPredicateTests(unittest.TestCase):
    def test_stock_boundary_predicates(self):
        boundary_bits = struct.unpack("=I", struct.pack("=f", 1.0e16))[0]
        below = struct.unpack("=f", struct.pack("=I", boundary_bits - 1))[0]
        boundary = struct.unpack("=f", struct.pack("=I", boundary_bits))[0]
        above = struct.unpack("=f", struct.pack("=I", boundary_bits + 1))[0]
        self.assertLessEqual(below, 1.0e16)
        self.assertGreater(boundary, 1.0e16)
        self.assertFalse(
            _native.domain_rescore_own_scale_required_for_test(below)
        )
        self.assertTrue(
            _native.domain_rescore_own_scale_required_for_test(boundary)
        )
        self.assertTrue(
            _native.domain_rescore_own_scale_required_for_test(above)
        )

        negative_infinity = -math.inf
        choose_j = _native.domain_rescore_oatrace_j_predecessor_for_test
        # ReconfigUnihit keeps J-loop and disables E->J. A reachable zero J
        # path therefore beats disabled -infinity; comparing it to zero is
        # the historical bug this case prevents.
        self.assertTrue(choose_j(0.0, 0.0, True, False))
        self.assertFalse(
            choose_j(negative_infinity, 0.0, True, False)
        )
        # Stock uses strict greater-than, so an enabled exact tie chooses E.
        self.assertFalse(choose_j(0.0, 0.0, True, True))

    @unittest.skipIf(
        reconcile_final_domains is None,
        "domain rescore audit helper unavailable",
    )
    def test_final_domain_accounting_is_row_based(self):
        simple = _native.BACKWARD_DOMAIN_SIMPLE
        cpu_route = _native.BACKWARD_DOMAIN_CPU_REQUIRED
        device = _native.DOMAIN_RESCORE_DEVICE_RESULT
        cpu = _native.DOMAIN_RESCORE_CPU_REQUIRED

        def upstream(route):
            values = [0] * 11
            values[8] = route
            return tuple(values)

        def result(row, action):
            values = [0] * 19
            values[0] = row
            values[15] = action
            return tuple(values)

        census = {
            "stage-cpu": {"ndom": "3"},
            "device": {"ndom": "1"},
            "front-cpu": {"ndom": "2"},
            "route-cpu": {"ndom": "4"},
        }
        upstream_keys = ["stage-cpu", "device", "route-cpu"]
        upstream_results = [
            upstream(simple),
            upstream(simple),
            upstream(cpu_route),
        ]
        results = [result(0, cpu), result(0, cpu), result(1, device)]
        statistics = {
            "global_cpu_fallback_count": 0,
            "device_result_count": 1,
        }
        accounting = reconcile_final_domains(
            census,
            upstream_keys,
            upstream_results,
            results,
            statistics,
            simple_route=simple,
            device_action=device,
            cpu_action=cpu,
        )
        self.assertEqual(accounting["front_half_cpu_retained_domains"], 2)
        self.assertEqual(
            accounting["upstream_route_cpu_retained_domains"], 4
        )
        # Two fallback envelopes retain all three final domains in the row.
        self.assertEqual(accounting["stage_cpu_retained_domains"], 3)
        self.assertEqual(accounting["device_final_domains"], 1)
        self.assertEqual(accounting["accounted_final_domains"], 10)
        self.assertEqual(accounting["stage_mixed_action_rows"], 0)
        self.assertEqual(accounting["device_domain_count_mismatches"], 0)
        self.assertEqual(
            accounting["device_result_statistic_mismatches"], 0
        )

        mixed = reconcile_final_domains(
            census,
            upstream_keys,
            upstream_results,
            [result(0, cpu), result(0, device), result(1, device)],
            {**statistics, "device_result_count": 2},
            simple_route=simple,
            device_action=device,
            cpu_action=cpu,
        )
        # A broken mixed row is retained wholly on CPU and its device record
        # is excluded from the disjoint D partition, avoiding double-counting.
        self.assertEqual(mixed["accounted_final_domains"], 10)
        self.assertEqual(mixed["stage_mixed_action_rows"], 1)
        self.assertEqual(mixed["device_result_statistic_mismatches"], 1)


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaDomainRescoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="plan7-domain-rescore-test-"
        )
        names = ("RREFam.hmm", "Thioesterase.hmm", "KR.hmm", "LuxC.hmm")
        hmms = []
        for name in names:
            with pyhmmer.plan7.HMMFile(DATA / "hmms" / "txt" / name) as source:
                hmms.append(source.read())
        cls.alphabet = hmms[0].alphabet
        cls.base = Path(cls.temporary.name) / "models"
        pyhmmer.hmmer.hmmpress(hmms, cls.base)
        cls.pairs = load_pressed_profiles(cls.base)
        sequences = []
        with pyhmmer.easel.SequenceFile(
            DATA / "seqs" / "938293.PRJEB85.HG003687.faa",
            digital=True,
            alphabet=cls.alphabet,
        ) as source:
            for sequence in source:
                sequences.append(sequence)
                if len(sequences) == 4:
                    break
        consensus = hmms[1].consensus
        if isinstance(consensus, bytes):
            consensus = consensus.decode()
        consensus = consensus.replace("-", "")
        sequences.append(
            pyhmmer.easel.TextSequence(
                name=b"two-domain-probe",
                sequence=consensus + "X" * 100 + consensus,
            ).digitize(cls.alphabet)
        )
        cls.sequences = tuple(sequences)
        cls.targets = pyhmmer.easel.DigitalSequenceBlock(
            cls.alphabet, cls.sequences
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_diagnostic(
        self,
        compact_budget=0,
        matrix_budget=0,
        trace_budget=0,
        test_fault=0,
    ):
        session = ProfileSession(self.pairs)
        selection = session.select(range(len(self.pairs)))
        batch = SequenceBatch(self.targets)
        try:
            selection_state = _profile_selection_state(selection)
            sequence_state = _sequence_state(batch)
            _journal, payload = (
                sequence_state.native
                ._postfilter_forward_domain_selection_sealed(
                    selection_state.native,
                    0.99,
                    1.0,
                    1.0,
                    2.0e-4,
                    _native.FORWARD_MAX_GATHERED_BYTES,
                    True,
                    matrix_budget,
                    trace_budget,
                    compact_budget,
                    test_fault,
                )
            )
            return payload
        finally:
            batch.close()
            selection.close()
            session.close()

    def run_pair_diagnostic(
        self, profile_index, sequence_index, matrix_budget=0
    ):
        session = ProfileSession(self.pairs)
        selection = session.select([profile_index])
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[sequence_index]]
        )
        batch = SequenceBatch(targets)
        try:
            selection_state = _profile_selection_state(selection)
            sequence_state = _sequence_state(batch)
            _journal, payload = (
                sequence_state.native
                ._postfilter_forward_domain_selection_sealed(
                    selection_state.native,
                    0.99,
                    1.0,
                    1.0,
                    2.0e-4,
                    _native.FORWARD_MAX_GATHERED_BYTES,
                    True,
                    matrix_budget,
                )
            )
            return payload
        finally:
            batch.close()
            selection.close()
            session.close()

    def test_real_simple_regions_match_pristine_hmmer(self):
        result_bytes, trace_offsets, trace_bytes, null2, statistics, _upstream = (
            self.run_diagnostic()
        )
        results = list(RESULT.iter_unpack(result_bytes))
        gpu_trace = list(TRACE_STEP.iter_unpack(trace_bytes))
        gpu_null2 = list(null2)

        self.assertGreater(len(results), 0)
        self.assertEqual(statistics["region_count"], len(results))
        self.assertEqual(
            statistics["simple_row_count"],
            len({result[0] for result in results}),
        )
        self.assertTrue(
            any(count >= 2 for count in Counter(
                result[0] for result in results
            ).values())
        )
        self.assertEqual(statistics["device_result_count"], len(results))
        self.assertEqual(statistics["cpu_required_count"], 0)
        self.assertEqual(statistics["result_count"], len(results))
        self.assertEqual(statistics["trace_count"], len(gpu_trace))
        self.assertEqual(statistics["null2_count"], len(gpu_null2))
        self.assertEqual(len(gpu_null2), len(results) * 29)
        self.assertEqual(len(trace_offsets), len(results) + 1)
        self.assertEqual(trace_offsets[0], 0)
        self.assertEqual(trace_offsets[-1], len(gpu_trace))
        self.assertGreater(statistics["work_cells"], 0)
        self.assertGreater(statistics["kernel_ms"], 0.0)
        self.assertLess(
            statistics["compact_output_bytes"],
            statistics["forward_matrix_bytes"]
            + statistics["posterior_matrix_bytes"],
        )

        for index, gpu_result in enumerate(results):
            with self.subTest(region=index):
                profile_index = gpu_result[1]
                sequence_index = gpu_result[2]
                envelope_begin = gpu_result[3]
                envelope_end = gpu_result[4]
                profile = _pair_state(
                    self.pairs[profile_index]
                ).optimized_profile
                residues = memoryview(
                    self.sequences[sequence_index].sequence
                ).cast("B")
                cpu_bytes, cpu_null2, cpu_trace_bytes = (
                    _native.domain_rescore_cpu_oracle_raw(
                        profile, residues, envelope_begin, envelope_end
                    )
                )
                cpu_result = RESULT.unpack(cpu_bytes)
                cpu_trace = list(TRACE_STEP.iter_unpack(cpu_trace_bytes))
                begin = trace_offsets[index]
                end = trace_offsets[index + 1]
                row_trace = gpu_trace[begin:end]
                row_null2 = gpu_null2[index * 29 : (index + 1) * 29]

                self.assertEqual(
                    gpu_result[14:17],
                    (
                        _native.DOMAIN_RESCORE_STATUS_OK,
                        _native.DOMAIN_RESCORE_DEVICE_RESULT,
                        cpu_result[16],
                    ),
                )
                self.assertEqual(gpu_result[3:9], cpu_result[3:9])
                for gpu_score, cpu_score in zip(
                    gpu_result[9:14], cpu_result[9:14], strict=True
                ):
                    self.assertTrue(math.isfinite(gpu_score))
                    self.assertLessEqual(abs(gpu_score - cpu_score), 5.0e-7)
                self.assertEqual(
                    [(step[0], step[1], step[3]) for step in row_trace],
                    [(step[0], step[1], step[3]) for step in cpu_trace],
                )
                self.assertEqual(len(row_trace), len(cpu_trace))
                self.assertLessEqual(
                    max(
                        (
                            abs(gpu[2] - cpu[2])
                            for gpu, cpu in zip(
                                row_trace, cpu_trace, strict=True
                            )
                        ),
                        default=0.0,
                    ),
                    1.0e-6,
                )
                self.assertLessEqual(
                    max(
                        abs(gpu - cpu)
                        for gpu, cpu in zip(
                            row_null2, cpu_null2, strict=True
                        )
                    ),
                    1.0e-6,
                )

    def test_tiny_matrix_budget_falls_back_row_atomically(self):
        discovery_bytes, _, _, _, _, _ = self.run_diagnostic()
        discovery = list(RESULT.iter_unpack(discovery_bytes))
        counts = Counter(result[0] for result in discovery)
        multi_row = next(
            (row for row, count in counts.items() if count >= 2), None
        )
        self.assertIsNotNone(multi_row)
        seed = next(result for result in discovery if result[0] == multi_row)
        profile_index = seed[1]
        sequence_index = seed[2]
        # A stock CPU census under the identical F1/F2/F3/domain thresholds
        # proves profiles 0, 1, and 2 each produce two guarded SIMPLE regions
        # for this final synthetic target. The four original targets produce
        # no multi-envelope row in this focused fixture.
        self.assertIn(profile_index, (0, 1, 2))
        self.assertEqual(sequence_index, len(self.sequences) - 1)

        baseline_bytes, _, _, _, baseline_statistics, upstream_payload = (
            self.run_pair_diagnostic(profile_index, sequence_index)
        )
        baseline = list(RESULT.iter_unpack(baseline_bytes))
        self.assertGreaterEqual(len(baseline), 2)
        self.assertEqual(len({result[0] for result in baseline}), 1)
        self.assertEqual(
            baseline_statistics["device_result_count"], len(baseline)
        )
        upstream_bytes, upstream_offsets, upstream_region_values = (
            upstream_payload
        )
        upstream_results = list(UPSTREAM_RESULT.iter_unpack(upstream_bytes))
        upstream_offsets = list(upstream_offsets)
        upstream_region_values = list(upstream_region_values)
        upstream_regions = list(zip(
            upstream_region_values[0::2],
            upstream_region_values[1::2],
            strict=True,
        ))
        self.assertEqual(len(upstream_results), 1)
        self.assertEqual(
            upstream_results[0][8], _native.BACKWARD_DOMAIN_SIMPLE
        )
        self.assertEqual(upstream_offsets, [0, len(baseline)])
        self.assertEqual(
            upstream_regions,
            [(result[3], result[4]) for result in baseline],
        )

        model_length = (
            _pair_state(self.pairs[profile_index]).optimized_profile.M
        )
        q = max(2, (model_length + 3) // 4)
        per_envelope_dense_bytes = [
            (result[4] - result[3] + 2) * (96 * q + 48)
            for result in baseline
        ]
        matrix_budget = max(per_envelope_dense_bytes)
        self.assertTrue(all(
            size <= matrix_budget for size in per_envelope_dense_bytes
        ))
        self.assertGreater(sum(per_envelope_dense_bytes), matrix_budget)

        result_bytes, trace_offsets, trace_bytes, null2, statistics, _ = (
            self.run_pair_diagnostic(
                profile_index, sequence_index, matrix_budget=matrix_budget
            )
        )
        results = list(RESULT.iter_unpack(result_bytes))

        self.assertEqual(len(results), len(baseline))
        row_counts = Counter(result[0] for result in results)
        self.assertTrue(any(count >= 2 for count in row_counts.values()))
        self.assertEqual(statistics["device_result_count"], 0)
        self.assertEqual(statistics["cpu_required_count"], len(results))
        self.assertEqual(statistics["cap_fallback_count"], len(results))
        self.assertEqual(statistics["numeric_fallback_count"], 0)
        self.assertEqual(trace_bytes, b"")
        self.assertEqual(list(trace_offsets), [0] * (len(results) + 1))
        self.assertEqual(len(null2), len(results) * 29)
        self.assertTrue(all(math.isnan(value) for value in null2))
        for result in results:
            self.assertEqual(
                result[14:16],
                (
                    _native.DOMAIN_RESCORE_STATUS_ECAP,
                    _native.DOMAIN_RESCORE_CPU_REQUIRED,
                ),
            )
        for row, count in row_counts.items():
            self.assertGreaterEqual(count, 1)
            self.assertTrue(all(
                result[15] == _native.DOMAIN_RESCORE_CPU_REQUIRED
                for result in results if result[0] == row
            ))

    def test_compact_output_cap_retains_every_region_on_cpu(self):
        result_bytes, trace_offsets, trace_bytes, null2, statistics, _ = (
            self.run_diagnostic(compact_budget=1)
        )
        self.assertEqual(result_bytes, b"")
        self.assertEqual(trace_bytes, b"")
        self.assertEqual(list(trace_offsets), [0])
        self.assertEqual(len(null2), 0)
        self.assertGreater(statistics["region_count"], 0)
        self.assertEqual(
            statistics["global_cpu_fallback_count"],
            statistics["region_count"],
        )
        self.assertEqual(
            statistics["cpu_required_count"], statistics["region_count"]
        )
        self.assertEqual(
            statistics["cap_fallback_count"], statistics["region_count"]
        )
        self.assertEqual(statistics["device_result_count"], 0)
        self.assertEqual(statistics["compact_output_byte_limit"], 8)
        self.assertEqual(statistics["compact_output_bytes"], 8)

    def test_sealed_own_scale_row_falls_back_atomically(self):
        result_bytes, trace_offsets, _, null2, statistics, _ = (
            self.run_diagnostic(
                test_fault=(
                    _native.BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE
                )
            )
        )
        results = list(RESULT.iter_unpack(result_bytes))
        forced = [result for result in results if result[16] == 1]
        self.assertGreater(len(forced), 0)
        forced_rows = {result[0] for result in forced}
        for index, result in enumerate(results):
            if result[0] not in forced_rows:
                continue
            self.assertEqual(
                result[14:16],
                (
                    _native.DOMAIN_RESCORE_STATUS_ERANGE,
                    _native.DOMAIN_RESCORE_CPU_REQUIRED,
                ),
            )
            self.assertEqual(trace_offsets[index], trace_offsets[index + 1])
            row_null2 = null2[index * 29 : (index + 1) * 29]
            self.assertTrue(all(math.isnan(value) for value in row_null2))
        self.assertEqual(statistics["cap_fallback_count"], 0)
        self.assertGreaterEqual(
            statistics["numeric_fallback_count"], len(forced)
        )

    def test_tampered_opaque_upstream_provenance_is_rejected(self):
        for fault in (
            _native.BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH,
            _native.BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH,
        ):
            with self.subTest(fault=fault):
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    self.run_diagnostic(test_fault=fault)


if __name__ == "__main__":
    unittest.main()
