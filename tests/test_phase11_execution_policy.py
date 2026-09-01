import os
import sys
import unittest
from array import array
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pyhmmer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import SequenceBatch, _native
    from plan7_gpu.adapter import _normalize_execution_policy
except ImportError:
    SequenceBatch = None
    _native = None
    _normalize_execution_policy = None


HMM_20AA = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
POLICY_ENVIRONMENT = (
    "PLAN7_GPU_SSV_PROFILE_POLICY",
    "PLAN7_GPU_SSV_LENGTH_METADATA",
    "PLAN7_GPU_FULL_MSV_POLICY",
    "PLAN7_GPU_FULL_MSV_ARITHMETIC",
    "PLAN7_GPU_FORWARD_OWNERSHIP",
    "PLAN7_GPU_FORWARD_CPU_MIN_CELLS",
    "PLAN7_GPU_FORWARD_CPU_MIN_LENGTH",
    "PLAN7_GPU_FORWARD_CPU_MAX_CELLS",
    "ASTRA_GPU_CONTINUATION_POOL",
)


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


class ExecutionPolicyHostTests(unittest.TestCase):
    def test_policy_names_are_exact_and_stable(self):
        if _normalize_execution_policy is None:
            self.skipTest("plan7_gpu extension unavailable")
        self.assertEqual(
            _normalize_execution_policy("auto"),
            ("auto", _native.EXECUTION_POLICY_AUTO),
        )
        self.assertEqual(
            _normalize_execution_policy("simple"),
            ("simple", _native.EXECUTION_POLICY_SIMPLE),
        )
        self.assertEqual(
            _normalize_execution_policy("throughput"),
            ("throughput", _native.EXECUTION_POLICY_THROUGHPUT),
        )
        for invalid in (None, 0, True, "packed", "AUTO", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "execution_policy"):
                    _normalize_execution_policy(invalid)

    def test_invalid_public_policy_fails_before_cuda_or_alphabet_work(self):
        if SequenceBatch is None:
            self.skipTest("plan7_gpu extension unavailable")
        with self.assertRaisesRegex(ValueError, "execution_policy"):
            SequenceBatch([], execution_policy="experimental")


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class ExecutionPolicyCudaTests(unittest.TestCase):
    @staticmethod
    def _inputs(profile_count=40, sequence_count=320):
        with pyhmmer.plan7.HMMFile(HMM_20AA) as hmm_file:
            hmm = hmm_file.read()
        background = pyhmmer.plan7.Background(hmm.alphabet)
        profiles = [
            hmm.to_profile(background, L=64).to_optimized()
            for _ in range(profile_count)
        ]
        sequences = [
            pyhmmer.easel.TextSequence(
                name=f"target-{index}".encode(),
                sequence=("ACDEFGHIKLMNPQRSTVWY"[: 1 + index % 20]),
            ).digitize(hmm.alphabet)
            for index in range(sequence_count)
        ]
        residues = bytearray()
        offsets = array("Q", [0])
        for sequence in sequences:
            residues.extend(memoryview(sequence.sequence).cast("B"))
            offsets.append(len(residues))
        return background, profiles, residues, offsets

    def test_forceable_policies_preserve_complete_postfilter_rows(self):
        background, profiles, residues, offsets = self._inputs()
        results = {}
        clean_environment = {
            name: value
            for name, value in os.environ.items()
            if name not in POLICY_ENVIRONMENT
        }
        with mock.patch.dict(os.environ, clean_environment, clear=True):
            for policy_name, policy_code in (
                ("auto", _native.EXECUTION_POLICY_AUTO),
                ("simple", _native.EXECUTION_POLICY_SIMPLE),
                ("throughput", _native.EXECUTION_POLICY_THROUGHPUT),
            ):
                with ExitStack() as stack:
                    session = _native.ProfileSession(
                        profiles,
                        memoryview(background.residue_frequencies),
                        4,
                        4,
                    )
                    stack.callback(session.close)
                    selection = session.select(range(len(profiles)))
                    stack.callback(selection.close)
                    batch = stack.enter_context(
                        _native.SequenceBatch(
                            residues,
                            offsets,
                            profiles[0].alphabet.Kp,
                            policy_code,
                        )
                    )
                    records, row_offsets = (
                        batch.postfilter_profile_selection_csr_raw(
                            selection,
                            0.02,
                        )
                    )
                    results[policy_name] = (
                        bytes(records),
                        tuple(row_offsets),
                        dict(batch.workspace_statistics),
                    )

        self.assertEqual(results["auto"][:2], results["simple"][:2])
        self.assertEqual(results["auto"][:2], results["throughput"][:2])
        auto = results["auto"][2]
        simple = results["simple"][2]
        throughput = results["throughput"][2]
        self.assertEqual(auto["execution_policy_mode"], _native.EXECUTION_POLICY_AUTO)
        self.assertEqual(
            simple["execution_policy_mode"], _native.EXECUTION_POLICY_SIMPLE
        )
        self.assertEqual(
            throughput["execution_policy_mode"],
            _native.EXECUTION_POLICY_THROUGHPUT,
        )
        self.assertEqual(simple["f1_profile_packed_run_count"], 0)
        self.assertEqual(simple["f1_length_class_run_count"], 0)
        self.assertEqual(simple["full_msv_compaction_run_count"], 0)
        self.assertEqual(simple["full_msv_packed_run_count"], 0)
        self.assertEqual(simple["full_msv_legacy_run_count"], 1)
        self.assertEqual(auto["f1_profile_packed_run_count"], 1)
        self.assertEqual(auto["f1_length_class_run_count"], 1)
        self.assertEqual(throughput["f1_profile_packed_run_count"], 1)
        self.assertEqual(throughput["f1_length_class_run_count"], 1)
        self.assertEqual(throughput["full_msv_compaction_run_count"], 1)
        self.assertEqual(
            throughput["execution_policy_forward_candidates_per_warp"], 1
        )

    def test_request_local_forward_threshold_is_exact_and_override_safe(self):
        target_count = 65_537
        residues = bytearray(b"A" * target_count)
        offsets = array("Q", range(target_count + 1))
        clean_environment = {
            name: value
            for name, value in os.environ.items()
            if name not in POLICY_ENVIRONMENT
        }
        with mock.patch.dict(os.environ, clean_environment, clear=True):
            with _native.SequenceBatch(
                residues,
                offsets,
                29,
                _native.EXECUTION_POLICY_AUTO,
                200_000,
            ) as batch:
                statistics = batch.workspace_statistics
                self.assertEqual(statistics["cpu_forward_ownership_mode"], 4)
                self.assertTrue(statistics["cpu_forward_ownership_auto"])
                self.assertEqual(
                    statistics["cpu_forward_max_cells"], 200_000
                )
            with _native.SequenceBatch(
                residues,
                offsets,
                29,
                _native.EXECUTION_POLICY_AUTO,
            ) as batch:
                self.assertEqual(
                    batch.workspace_statistics["cpu_forward_ownership_mode"],
                    0,
                )

            for invalid in (True, 0, -1, 1 << 64):
                with self.subTest(invalid=invalid), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    _native.SequenceBatch(
                        residues,
                        offsets,
                        29,
                        _native.EXECUTION_POLICY_AUTO,
                        invalid,
                    )

        for environment in POLICY_ENVIRONMENT[-5:]:
            with self.subTest(environment=environment), mock.patch.dict(
                os.environ,
                {**clean_environment, environment: "1"},
                clear=True,
            ), self.assertRaisesRegex(ValueError, "conflicts"):
                _native.SequenceBatch(
                    residues,
                    offsets,
                    29,
                    _native.EXECUTION_POLICY_AUTO,
                    200_000,
                )
