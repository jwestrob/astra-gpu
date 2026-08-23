import math
import sys
import unittest
from array import array
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
try:
    from plan7_gpu import _native, _pipeline
    from plan7_gpu.adapter import (
        _ForwardAugmentation,
        _new_candidate_batch,
    )
except ImportError:
    _native = None
    _pipeline = None
    _ForwardAugmentation = None
    _new_candidate_batch = None


@unittest.skipUnless(_pipeline is not None, "pipeline extension unavailable")
class NativeStageTimingBoundaryTests(unittest.TestCase):
    @staticmethod
    def valid(rescore=True):
        return (
            1,
            (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
            (1.1, 1.2, 1.3, 1.4),
            (2.1, 2.2, 2.3, 2.4) if rescore else None,
        )

    def test_exact_schema_maps_native_fields_and_scopes(self):
        self.assertEqual(
            _native.SEALED_STAGE_TIMING_SCHEMA_VERSION,
            _pipeline.SEALED_STAGE_TIMING_SCHEMA_VERSION,
        )
        evidence = _pipeline._validate_native_stage_timings_bound(
            self.valid(), True
        )
        self.assertEqual(evidence["schema_version"], 1)
        self.assertEqual(evidence["units"], "milliseconds")
        self.assertEqual(
            evidence["forward"]["profile_staging_scope"],
            "per-selection database",
        )
        self.assertIn("provenance sealing", evidence["forward"]["call_total_scope"])
        self.assertEqual(evidence["forward"]["profile_pack_ms"], 0.1)
        self.assertEqual(evidence["forward"]["profile_upload_ms"], 0.2)
        self.assertEqual(evidence["forward"]["run_upload_ms"], 0.3)
        self.assertEqual(evidence["forward"]["kernel_ms"], 0.4)
        self.assertEqual(evidence["forward"]["download_ms"], 0.7)
        self.assertEqual(evidence["forward"]["timed_loop_ms"], 0.8)
        self.assertEqual(evidence["forward"]["call_total_ms"], 0.9)
        self.assertEqual(evidence["backward_domain"]["kernel_ms"], 1.1)
        self.assertNotIn("download_ms", evidence["backward_domain"])
        self.assertEqual(
            evidence["backward_domain"]["post_primary_materialization_ms"],
            1.3,
        )
        self.assertIn(
            "download_milliseconds",
            evidence["backward_domain"][
                "post_primary_materialization_native_field"
            ],
        )
        self.assertEqual(evidence["domain_rescore"]["total_ms"], 2.4)

    def test_schema_rejects_wrong_shape_type_range_and_rescore_binding(self):
        invalid = (
            (None, True, TypeError),
            ((1,), True, ValueError),
            ((True,) + self.valid()[1:], True, ValueError),
            ((2,) + self.valid()[1:], True, ValueError),
            ((1, list(self.valid()[1]), self.valid()[2], self.valid()[3]), True, TypeError),
            ((1, self.valid()[1][:-1], self.valid()[2], self.valid()[3]), True, ValueError),
            ((1, (0, *self.valid()[1][1:]), self.valid()[2], self.valid()[3]), True, TypeError),
            (
                (1, (math.nan, *self.valid()[1][1:]), self.valid()[2], self.valid()[3]),
                True,
                ValueError,
            ),
            ((1, (-0.1, *self.valid()[1][1:]), self.valid()[2], self.valid()[3]), True, ValueError),
            (
                (1, (math.inf, *self.valid()[1][1:]), self.valid()[2], self.valid()[3]),
                True,
                ValueError,
            ),
            (self.valid(False), True, TypeError),
            (self.valid(True), False, ValueError),
        )
        for values, rescore_expected, error in invalid:
            with self.subTest(values=values, rescore_expected=rescore_expected):
                with self.assertRaises(error):
                    _pipeline._validate_native_stage_timings_bound(
                        values, rescore_expected
                    )

    def test_evidence_is_a_defensive_copy(self):
        values = self.valid()
        first = _pipeline._validate_native_stage_timings_bound(values, True)
        first["forward"]["kernel_ms"] = -1.0
        second = _pipeline._validate_native_stage_timings_bound(values, True)
        self.assertEqual(second["forward"]["kernel_ms"], 0.4)


@unittest.skipUnless(
    _new_candidate_batch is not None,
    "candidate adapter unavailable",
)
class CandidateResidentMemoryTests(unittest.TestCase):
    def test_plain_candidate_exact_payload_is_stable_and_source_independent(self):
        indices = array("I", [1, 2])
        offsets = array("Q", [0])
        all_targets = array("I", [3])
        all_rows = bytearray(b"\x00")
        postfilter_records = bytearray(b"abc")
        candidates = _new_candidate_batch(
            (),
            object(),
            b"shared-residue-storage",
            indices,
            offsets,
            all_rows,
            all_targets,
            0.02,
            postfilter_records=postfilter_records,
        )
        expected = 8 + 8 + 1 + 4 + 3
        first = candidates.resident_memory
        self.assertEqual(
            first["accounting_scope"],
            "owned immutable backing-buffer payload",
        )
        self.assertIs(type(candidates.resident_bytes), int)
        self.assertEqual(candidates.resident_bytes, expected)
        self.assertEqual(first["resident_bytes"], expected)
        self.assertEqual(first["owned_host_bytes"], expected)
        self.assertEqual(first["owned_device_bytes"], 0)
        self.assertIsNone(first["sealed"])
        indices.append(99)
        offsets.append(99)
        all_targets.append(99)
        all_rows[0] = 1
        postfilter_records[:] = b"changed"
        first["state_buffers"]["indices_bytes"] = 0
        self.assertEqual(candidates.resident_bytes, expected)

    def test_legacy_forward_payload_counts_each_backing_buffer_once(self):
        records = bytearray(b"1234")
        row_offsets = array("Q", [0, 0])
        special_offsets = array("Q", [0, 0, 0])
        specials = array("f", [0.0] * 5)
        forward = _ForwardAugmentation(
            records,
            row_offsets,
            special_offsets,
            specials,
            1,
            2,
            True,
        )
        records[:] = b"changed"
        row_offsets.append(0)
        special_offsets.append(0)
        specials.append(0.0)
        candidates = _new_candidate_batch(
            (),
            object(),
            b"shared",
            array("I"),
            array("Q", [0]),
            b"",
            array("I"),
            0.02,
            forward=forward,
        )
        expected = 8 + 4 + 16 + 24 + 20
        memory = candidates.resident_memory
        self.assertEqual(memory["resident_bytes"], expected)
        self.assertEqual(
            memory["state_owned_host_bytes"],
            sum(memory["state_buffers"].values()),
        )
        self.assertEqual(memory["owned_device_bytes"], 0)


class PrivateTransportSourceCompatibilityTests(unittest.TestCase):
    def test_f1_device_candidate_cache_fails_closed_before_overwrite(self):
        source = (ROOT / "cuda/ssv_cuda.cu").read_text()
        getter = source[
            source.index("plan7_ssv_sequence_batch_get_f1_candidate_view") :
            source.index("plan7_ssv_sequence_batch_bias_candidates_many")
        ]
        bias = source[
            source.index("plan7_ssv_sequence_batch_bias_candidates_many") :
            source.index("sequence_batch_postfilter_candidates_many_impl")
        ]
        postfilter = source[
            source.index("sequence_batch_postfilter_candidates_many_impl") :
            source.index("plan7_ssv_sequence_batch_postfilter_candidates_many(")
        ]

        self.assertEqual(source.count("invalidate_f1_cache(batch);"), 3)
        self.assertIn(
            "!batch->f1_cache_valid || !batch->f1_device_candidates_valid",
            getter,
        )
        for implementation in (bias, postfilter):
            invalidation = implementation.index(
                "invalidate_f1_device_candidates(batch);"
            )
            self.assertLess(
                invalidation,
                implementation.index(
                    "realloc(batch->host_bias_candidates, candidate_bytes)"
                ),
            )
            self.assertLess(
                invalidation,
                implementation.index(
                    "grow_device_buffer(&batch->device_bias_candidates"
                ),
            )
            self.assertLess(
                invalidation,
                implementation.index(
                    "cudaMemcpy(batch->device_bias_candidates"
                ),
            )

    def test_direct_v3_uses_fused_plan_and_one_emission_scan(self):
        native = (ROOT / "python/plan7_gpu/_native.pyx").read_text()
        pipeline = (ROOT / "python/plan7_gpu/_pipeline.pyx").read_text()

        self.assertIn("candidate_postfilter_sources.push_back(cursor)", native)
        self.assertIn("_direct_v3_plan_initial(", native)
        self.assertIn("_direct_v3_plan_replace(", native)
        self.assertIn("planning_profile_count = 0", pipeline)
        self.assertIn("sealed._journal_v3_source_scan_count = 1", pipeline)
        self.assertIn("sealed._journal_v3_decision_scan_count = 0", pipeline)
        self.assertIn('"planner_source_scan_count": (', pipeline)
        self.assertIn('"separate_decision_scan_count": (', pipeline)

    def test_direct_v3_releases_identity_staging_before_zero_accounting(self):
        source = (ROOT / "python/plan7_gpu/_pipeline.pyx").read_text()
        drop = source[
            source.index("cdef void _v3_drop_direct_staging"):
            source.index("cdef plan7_continuation_journal_v3 *_v3_validate_capsule")
        ]
        resident = source[
            source.index("def _sealed_resident_memory_bound"):
            source.index("def _search_hmm_candidates")
        ]

        for field in (
            "_source_identity_tokens",
            "_source_profile_fingerprints",
            "_source_sequence_fingerprint",
            "_direct_v3_decisions",
        ):
            self.assertIn(f"sealed.{field} = ", drop)
        self.assertIn(
            'raise RuntimeError("direct v3 staging identity remains retained")',
            resident,
        )
        self.assertIn(
            '"direct_v3_staging_retained_bytes": (',
            resident,
        )

    def test_default_capsule_shape_and_all_direct_callers_are_deliberate(self):
        native = (ROOT / "python/plan7_gpu/_native.pyx").read_text()
        adapter = (ROOT / "python/plan7_gpu/adapter.py").read_text()
        profile_test = (ROOT / "tests/test_cuda_profile_session.py").read_text()
        domain_test = (ROOT / "tests/test_cuda_domain_rescore.py").read_text()
        domain_audit = (
            ROOT / "tests/audit_cuda_domain_rescore_pfam.py"
        ).read_text()

        self.assertIn("bint _return_stage_timings=False", native)
        self.assertIn("return result[0]", native)
        self.assertIn("native_stage_timings=None", (
            ROOT / "python/plan7_gpu/_pipeline.pyx"
        ).read_text())
        self.assertIn("_return_stage_timings=True", adapter)
        self.assertIn("_return_stage_timings=True", profile_test)
        self.assertNotIn("_return_stage_timings", domain_test)
        self.assertNotIn("_return_stage_timings", domain_audit)
        self.assertEqual(
            sum(
                source.count("._postfilter_forward_domain_selection_sealed(")
                for source in (adapter, profile_test, domain_test, domain_audit)
            ),
            5,
        )


if __name__ == "__main__":
    unittest.main()
