import ctypes
import math
import platform
import struct
import sys
import unittest
from array import array
from pathlib import Path
from unittest import mock

import pyhmmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import _native, _pipeline
    from plan7_gpu.adapter import _pack_profiles
except ImportError:
    _native = None
    _pipeline = None
    _pack_profiles = None


HMM_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
HMM_STRIPES = ROOT / "results" / "datasets" / "pfam-stripe-boundaries.hmm"
FASTA_FIRST1000 = ROOT / "results" / "datasets" / "PLM2_5.first1000.faa"


def float32_bits(value):
    return struct.unpack("=I", struct.pack("=f", value))[0]


def float32_from_bits(bits):
    return struct.unpack("=f", struct.pack("=I", bits))[0]


def float32(value):
    return float32_from_bits(float32_bits(value))


def next_float32(value, upward):
    bits = float32_bits(value)
    if math.isnan(value):
        return value
    if value == 0.0:
        return float32_from_bits(1 if upward else 0x80000001)
    if (value > 0.0) == upward:
        bits += 1
    else:
        bits -= 1
    return float32_from_bits(bits)


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


def bias_attested():
    return cuda_available() and _native.bias_environment_attested()[0]


def load_optimized(path):
    with pyhmmer.plan7.HMMFile(path) as hmm_file:
        hmms = list(hmm_file)
    background = pyhmmer.plan7.Background(hmms[0].alphabet)
    profiles = [hmm.to_profile(background, L=100).to_optimized() for hmm in hmms]
    return background, profiles


def digitize(alphabet, values):
    return [
        pyhmmer.easel.TextSequence(
            name=f"target-{index}".encode(), sequence=value
        ).digitize(alphabet)
        for index, value in enumerate(values)
    ]


def native_batch(sequences):
    residues = bytearray()
    offsets = array("Q", [0])
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        offsets.append(len(residues))
    return _native.SequenceBatch(residues, offsets, sequences[0].alphabet.Kp)


def pack_bias_profiles(background, profiles, f1):
    packed = bytearray()
    m_mu = array("f")
    m_lambda = array("f")
    for profile in profiles:
        mu, lambda_ = profile.evalue_parameters.as_vector()[:2]
        mode, cutoff = _native.f1_cutoff(mu, lambda_, f1)
        m_mu.append(mu)
        m_lambda.append(lambda_)
        packed.extend(
            _native.pack_bias_profile_raw(
                memoryview(background.residue_frequencies),
                memoryview(profile.compositions),
                profile.M,
                profile.scale_b,
                mode,
                float("nan") if cutoff is None else cutoff,
            )
        )
    return packed, m_mu, m_lambda


@unittest.skipUnless(
    _native is not None and _pipeline is not None, "native modules unavailable"
)
class BiasHostTests(unittest.TestCase):
    def test_cuda_target_allowlist_has_host_unit_boundaries(self):
        common = {
            "multiprocessor_count": 132,
            "total_global_memory": 141 * 1024**3,
            "uuid": bytes(range(1, 17)),
            "pci_bus_address": "0000:01:00.0",
        }
        for name in ("NVIDIA H200", "NVIDIA H200 NVL"):
            target, reason = _native._bias_cuda_identity_target_raw(
                name, 9, 0, **common
            )
            self.assertEqual(target, _native.BIAS_CUDA_SM90_H200)
            self.assertEqual(reason, "")

        target, reason = _native._bias_cuda_identity_target_raw(
            "NVIDIA GeForce RTX 2080 Ti",
            7,
            5,
            68,
            11 * 1024**3,
            bytes(range(1, 17)),
            "0000:01:00.0",
        )
        self.assertEqual(target, _native.BIAS_CUDA_SM75_RTX2080_TI)
        self.assertEqual(reason, "")

        rejected = (
            {"name": "NVIDIA H100", "compute_major": 9, "compute_minor": 0},
            {"name": "NVIDIA H200", "compute_major": 8, "compute_minor": 9},
            {
                "name": "NVIDIA H200",
                "compute_major": 9,
                "compute_minor": 0,
                "runtime_version": 12040,
            },
            {
                "name": "NVIDIA H200",
                "compute_major": 9,
                "compute_minor": 0,
                "driver_version": 12040,
            },
            {
                "name": "NVIDIA H200",
                "compute_major": 9,
                "compute_minor": 0,
                "nvcc_build": 81,
            },
            {
                "name": "NVIDIA H200",
                "compute_major": 9,
                "compute_minor": 0,
                "uuid": bytes(16),
            },
        )
        for changes in rejected:
            arguments = {**common, **changes}
            with self.subTest(changes=changes):
                target, reason = _native._bias_cuda_identity_target_raw(
                    arguments.pop("name"),
                    arguments.pop("compute_major"),
                    arguments.pop("compute_minor"),
                    **arguments,
                )
                self.assertEqual(target, _native.BIAS_CUDA_UNATTESTED)
                self.assertTrue(reason)

    def test_cpu_allowlist_is_bound_to_accelerator_target(self):
        leaf1_ecx = (1 << 12) | (1 << 19) | (1 << 27) | (1 << 28)
        leaf1_edx = 1 << 26
        leaf7_ebx = (
            (1 << 5)
            | (1 << 16)
            | (1 << 17)
            | (1 << 28)
            | (1 << 30)
            | (1 << 31)
        )
        common = (leaf1_ecx, leaf1_edx, leaf7_ebx, 0xE6)
        accepted, reason = _native._bias_host_identity_attested_raw(
            _native.BIAS_CUDA_SM75_RTX2080_TI, 6, 85, 4, *common
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "")
        accepted, reason = _native._bias_host_identity_attested_raw(
            _native.BIAS_CUDA_SM90_H200, 6, 143, 8, *common
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "")

        for target, family, model, stepping in (
            (_native.BIAS_CUDA_SM90_H200, 6, 85, 4),
            (_native.BIAS_CUDA_SM75_RTX2080_TI, 6, 143, 8),
            (_native.BIAS_CUDA_SM90_H200, 6, 143, 7),
        ):
            with self.subTest(target=target, model=model, stepping=stepping):
                accepted, reason = _native._bias_host_identity_attested_raw(
                    target, family, model, stepping, *common
                )
                self.assertFalse(accepted)
                self.assertTrue(reason)

        accepted, reason = _native._bias_host_identity_attested_raw(
            _native.BIAS_CUDA_SM90_H200,
            6,
            143,
            8,
            leaf1_ecx,
            leaf1_edx,
            leaf7_ebx & ~(1 << 16),
            0xE6,
        )
        self.assertFalse(accepted)
        self.assertRegex(reason, "feature set")

    def test_libm_build_ids_are_bound_to_accelerator_target(self):
        sm75 = bytes.fromhex("e6f050696120aeb5134a14e38bc54e8ce1bdc0b5")
        h200 = bytes.fromhex("b2b4cd0f73a7c7a6da5ba4126bceb33ce68853d5")
        self.assertEqual(len(sm75), _native.BIAS_LIBM_BUILD_ID_SIZE)
        self.assertEqual(len(h200), _native.BIAS_LIBM_BUILD_ID_SIZE)
        self.assertTrue(
            _native._bias_libm_build_id_attested_raw(
                sm75, _native.BIAS_CUDA_SM75_RTX2080_TI
            )
        )
        self.assertTrue(
            _native._bias_libm_build_id_attested_raw(
                h200, _native.BIAS_CUDA_SM90_H200
            )
        )
        for build_id, target in (
            (sm75, _native.BIAS_CUDA_SM90_H200),
            (h200, _native.BIAS_CUDA_SM75_RTX2080_TI),
            (bytes(len(sm75)), _native.BIAS_CUDA_SM75_RTX2080_TI),
            (bytes(len(h200)), _native.BIAS_CUDA_SM90_H200),
            (sm75[:-1], _native.BIAS_CUDA_SM75_RTX2080_TI),
            (b"", _native.BIAS_CUDA_SM90_H200),
            (None, _native.BIAS_CUDA_SM90_H200),
            (sm75, _native.BIAS_CUDA_UNATTESTED),
        ):
            with self.subTest(
                build_id=None if build_id is None else build_id.hex(),
                target=target,
            ):
                self.assertFalse(
                    _native._bias_libm_build_id_attested_raw(build_id, target)
                )

    def test_nondefault_float_modes_fail_closed(self):
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original_rounding = libc.fegetround()
        try:
            self.assertEqual(libc.fesetround(0x400), 0)
            self.assertFalse(_native.bias_host_environment_attested())
            self.assertEqual(
                _native.f1_cutoff(0.0, 1.0, 0.02),
                (_native.F1_CUTOFF_INVALID, None),
            )
            with self.assertRaisesRegex(ValueError, "floating-point environment"):
                _native.pack_bias_profile_raw(
                    array("f", [0.05] * 20),
                    array("f", [0.05] * 20),
                    20,
                    4.328084945678711,
                    _native.BIAS_CUTOFF_SCORE,
                    0.0,
                )
        finally:
            self.assertEqual(libc.fesetround(original_rounding), 0)

        if platform.system() != "Linux" or platform.machine() != "x86_64":
            return
        environment_type = ctypes.c_ubyte * 32
        original_environment = environment_type()
        changed_environment = environment_type()
        self.assertEqual(libc.fegetenv(ctypes.byref(original_environment)), 0)
        ctypes.memmove(changed_environment, original_environment, 32)
        mxcsr = int.from_bytes(bytes(changed_environment[28:32]), "little")
        changed_mxcsr = (mxcsr | 0x8000).to_bytes(4, "little")
        for index, value in enumerate(changed_mxcsr, start=28):
            changed_environment[index] = value
        try:
            self.assertEqual(libc.fesetenv(ctypes.byref(changed_environment)), 0)
            self.assertFalse(_native.bias_host_environment_attested())
        finally:
            self.assertEqual(libc.fesetenv(ctypes.byref(original_environment)), 0)

    def test_rolling_oracle_matches_allocating_hmmer(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        profile = profiles[0]
        mode, cutoff = _native.f1_cutoff(
            *profile.evalue_parameters.as_vector()[:2], 0.02
        )
        packed = _native.pack_bias_profile_raw(
            memoryview(background.residue_frequencies),
            memoryview(profile.compositions),
            profile.M,
            profile.scale_b,
            mode,
            cutoff,
        )
        sequences = digitize(
            profile.alphabet,
            [
                "A",
                "ACDEFGHIKLMNPQRSTVWY",
                "BJZOUX",
                "-*~",
                "G" * 999,
                "G" * 100_000,
            ],
        )
        pipeline = pyhmmer.plan7.Pipeline(profile.alphabet)
        for sequence in sequences:
            observed = _native.bias_filter_score_host_raw(
                packed, memoryview(sequence.sequence).cast("B")
            )
            expected = _pipeline._bias_filter_score_bits(pipeline, profile, sequence)
            self.assertEqual(observed, expected)

    def test_profile_carries_filter_hmm_boundary_constants(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        profile = profiles[0]
        mode, cutoff = _native.f1_cutoff(
            *profile.evalue_parameters.as_vector()[:2], 0.02
        )
        packed = _native.pack_bias_profile_raw(
            memoryview(background.residue_frequencies),
            memoryview(profile.compositions),
            profile.M,
            profile.scale_b,
            mode,
            cutoff,
        )
        self.assertEqual(len(packed), _native.BIAS_PROFILE_SIZE)
        fields = struct.unpack_from("=ffffiIffff", packed)
        self.assertEqual(float32_bits(fields[6]), 0x3F7FBE77)
        self.assertEqual(float32_bits(fields[7]), 0x3A83126F)
        self.assertEqual(fields[8:], (1.0, 1.0))

    def test_rebias_boundary_is_strict(self):
        action, bit_bits = _native.bias_rebias_decision_raw(
            _native.STATUS_OK,
            -31,
            3.0,
            -7.0,
            _native.BIAS_CUTOFF_INVALID,
            float("nan"),
        )
        self.assertEqual(action, _native.BIAS_CPU_REQUIRED)
        bit_score = float32_from_bits(bit_bits)
        at_boundary, _ = _native.bias_rebias_decision_raw(
            _native.STATUS_OK,
            -31,
            3.0,
            -7.0,
            _native.BIAS_CUTOFF_SCORE,
            bit_score,
        )
        above_boundary, _ = _native.bias_rebias_decision_raw(
            _native.STATUS_OK,
            -31,
            3.0,
            -7.0,
            _native.BIAS_CUTOFF_SCORE,
            next_float32(bit_score, True),
        )
        below_boundary, _ = _native.bias_rebias_decision_raw(
            _native.STATUS_OK,
            -31,
            3.0,
            -7.0,
            _native.BIAS_CUTOFF_SCORE,
            next_float32(bit_score, False),
        )
        self.assertEqual(at_boundary, _native.BIAS_DEFINITE_PASS)
        self.assertEqual(above_boundary, _native.BIAS_DEFINITE_REJECT)
        self.assertEqual(below_boundary, _native.BIAS_DEFINITE_PASS)

    def test_invalid_profile_is_rejected_before_cuda(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        with self.assertRaisesRegex(ValueError, "invalid amino bias profile"):
            _native.pack_bias_profile_raw(
                array("f", [0.0] * 20),
                memoryview(profiles[0].compositions),
                profiles[0].M,
                profiles[0].scale_b,
                _native.BIAS_CUTOFF_SCORE,
                0.0,
            )


@unittest.skipUnless(bias_attested(), "exact bias CUDA environment not attested")
class CudaBiasTests(unittest.TestCase):
    def test_runtime_provenance_matches_the_attested_target(self):
        provenance = _native.bias_environment_provenance()
        self.assertTrue(provenance["attested"], provenance["reason"])
        cuda = provenance["cuda"]
        expected = {
            "sm75_rtx2080ti": (
                _native.BIAS_CUDA_SM75_RTX2080_TI,
                [7, 5],
                {"NVIDIA GeForce RTX 2080 Ti"},
                "e6f050696120aeb5134a14e38bc54e8ce1bdc0b5",
            ),
            "sm90_h200": (
                _native.BIAS_CUDA_SM90_H200,
                [9, 0],
                {"NVIDIA H200", "NVIDIA H200 NVL"},
                "b2b4cd0f73a7c7a6da5ba4126bceb33ce68853d5",
            ),
        }
        self.assertIn(provenance["target"], expected)
        target_code, capability, names, libm_build_id = expected[
            provenance["target"]
        ]
        self.assertEqual(provenance["target_code"], target_code)
        self.assertEqual(provenance["libm"]["gnu_build_id"], libm_build_id)
        self.assertTrue(provenance["libm"]["attested_for_target"])
        self.assertEqual(cuda["compute_capability"], capability)
        self.assertIn(cuda["name"], names)
        self.assertEqual(cuda["runtime_version"], 12050)
        self.assertGreaterEqual(cuda["driver_version"], 12050)
        self.assertEqual(cuda["nvcc"], {"major": 12, "minor": 5, "build": 82})
        self.assertRegex(cuda["uuid"], r"^GPU-[0-9a-f-]{36}$")
        self.assertRegex(cuda["pci_bus_address"], r"^[0-9A-Fa-f:.]+$")

    def test_bad_batch_creation_environment_routes_every_target_to_cpu(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        profile = profiles[0]
        sequences = digitize(profile.alphabet, ["A", "G" * 20])
        packed_ssv = _pack_profiles(profiles)
        packed_bias, m_mu, m_lambda = pack_bias_profiles(background, profiles, 0.02)
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original_rounding = libc.fegetround()
        batch = None
        try:
            self.assertEqual(libc.fesetround(0x400), 0)
            batch = native_batch(sequences)
        finally:
            self.assertEqual(libc.fesetround(original_rounding), 0)
        try:
            single_candidates = batch.cpu_candidates_raw(
                memoryview(profile.sbv).cast("B"),
                profile.sbv.shape[1],
                profile.M,
                profile.alphabet.Kp,
                profile.tbm,
                profile.tec,
                profile.base_b,
                profile.bias_b,
                profile.scale_b,
                *profile.evalue_parameters.as_vector()[:2],
                0.02,
            )
            rows = batch.bias_candidates_many_raw(
                *packed_ssv, m_mu, m_lambda, 0.02, packed_bias
            )
        finally:
            batch.close()
        self.assertEqual(single_candidates, list(range(len(sequences))))
        self.assertEqual(len(rows[0]), len(sequences))
        self.assertTrue(
            all(record[1] == _native.BIAS_CPU_REQUIRED for record in rows[0])
        )

    def test_candidate_kernel_matches_hmmer_and_preserves_ssv(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        profile = profiles[0]
        sequences = digitize(
            profile.alphabet,
            ["A", "ACDEFGHIKLMNPQRSTVWY", "BJZOUX", "-*~", "G" * 999],
        )
        packed_ssv = _pack_profiles(profiles)
        packed_bias, m_mu, m_lambda = pack_bias_profiles(background, profiles, 1.0)
        pipeline = pyhmmer.plan7.Pipeline(profile.alphabet)
        with native_batch(sequences) as batch:
            rows = batch.bias_candidates_many_raw(
                *packed_ssv, m_mu, m_lambda, 1.0, packed_bias
            )
            workspace = batch.workspace_statistics
        self.assertEqual(workspace["f1_device_compaction_run_count"], 1)
        self.assertEqual(workspace["f1_candidate_upload_avoided_count"], 1)
        self.assertEqual(workspace["f1_candidate_upload_count"], 0)
        with native_batch(sequences) as reference_batch:
            ssv_rows = reference_batch.filter_many_raw(*packed_ssv)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), len(sequences))
        for expected_index, (record, sequence) in enumerate(
            zip(rows[0], sequences, strict=True)
        ):
            sequence_index, action, status, numerator, filtersc_bits = record
            self.assertEqual(sequence_index, expected_index)
            self.assertEqual(action, _native.BIAS_DEFINITE_PASS)
            self.assertEqual(status, ssv_rows[0][expected_index][0])
            self.assertEqual(numerator, ssv_rows[0][expected_index][3])
            self.assertEqual(
                filtersc_bits,
                _pipeline._bias_filter_score_bits(pipeline, profile, sequence),
            )

    def test_raw_xe_compaction_reconstructs_exact_ssv_and_skips_replay(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        sequences = digitize(
            profiles[0].alphabet,
            ["", "A", "ACDEFGHIKLMNPQRSTVWY", "BJZOUX", "G" * 127],
        )
        packed_ssv = _pack_profiles(profiles)
        packed_bias, m_mu, m_lambda = pack_bias_profiles(
            background, profiles, 1.0
        )
        with mock.patch.dict(
            "os.environ", {"PLAN7_GPU_F1_RAW_XE": "1"}, clear=False
        ):
            with native_batch(sequences) as batch:
                reconstructed = batch.bias_candidates_many_raw(
                    *packed_ssv, m_mu, m_lambda, 1.0, packed_bias
                )
                statistics = dict(batch.workspace_statistics)
        with mock.patch.dict(
            "os.environ", {"PLAN7_GPU_F1_RAW_XE": "0"}, clear=False
        ):
            with native_batch(sequences) as batch:
                replayed = batch.bias_candidates_many_raw(
                    *packed_ssv, m_mu, m_lambda, 1.0, packed_bias
                )
        self.assertEqual(reconstructed, replayed)
        candidate_count = sum(map(len, reconstructed))
        self.assertEqual(statistics["f1_raw_xe_run_count"], 1)
        self.assertEqual(
            statistics["f1_raw_xe_logical_pair_count"],
            len(profiles) * len(sequences),
        )
        self.assertEqual(
            statistics["f1_raw_xe_candidate_gather_count"], candidate_count
        )
        self.assertEqual(statistics["f1_candidate_ssv_replay_count"], 0)
        self.assertEqual(
            statistics["f1_candidate_ssv_replay_avoided_count"],
            candidate_count,
        )

    def test_real_f1_rows_are_exact_and_keep_all_actions(self):
        background, profiles = load_optimized(HMM_STRIPES)
        with pyhmmer.easel.SequenceFile(
            FASTA_FIRST1000,
            digital=True,
            alphabet=profiles[0].alphabet,
        ) as sequence_file:
            sequences = [next(sequence_file) for _ in range(256)]
        packed_ssv = _pack_profiles(profiles)
        packed_bias, m_mu, m_lambda = pack_bias_profiles(background, profiles, 0.02)
        pipeline = pyhmmer.plan7.Pipeline(profiles[0].alphabet)
        with native_batch(sequences) as batch:
            rows = batch.bias_candidates_many_raw(
                *packed_ssv, m_mu, m_lambda, 0.02, packed_bias
            )
            records, offsets = batch.bias_candidates_many_csr_raw(
                *packed_ssv, m_mu, m_lambda, 0.02, packed_bias
            )
        self.assertEqual(len(records), offsets[-1] * _native.BIAS_RESULT_SIZE)
        self.assertEqual(
            [len(row) for row in rows],
            [offsets[index + 1] - offsets[index] for index in range(len(profiles))],
        )

        actions = set()
        direct_count = 0
        for profile, row in zip(profiles, rows, strict=True):
            mode, cutoff = _native.f1_cutoff(
                *profile.evalue_parameters.as_vector()[:2], 0.02
            )
            for sequence_index, action, status, numerator, filtersc_bits in row:
                actions.add(action)
                if status != _native.STATUS_OK:
                    self.assertEqual(action, _native.BIAS_CPU_REQUIRED)
                    self.assertIsNone(filtersc_bits)
                    continue
                direct_count += 1
                sequence = sequences[sequence_index]
                self.assertEqual(
                    filtersc_bits,
                    _pipeline._bias_filter_score_bits(pipeline, profile, sequence),
                )
                filtersc = float32_from_bits(filtersc_bits)
                expected_action, _ = _native.bias_rebias_decision_raw(
                    status,
                    numerator,
                    profile.scale_b,
                    filtersc,
                    mode,
                    cutoff,
                )
                self.assertEqual(action, expected_action)
        self.assertGreater(direct_count, 0)
        self.assertEqual(
            actions,
            {
                _native.BIAS_CPU_REQUIRED,
                _native.BIAS_DEFINITE_REJECT,
                _native.BIAS_DEFINITE_PASS,
            },
        )

    def test_empty_and_uncertain_scores_fail_closed(self):
        background, profiles = load_optimized(HMM_GLOBINS)
        profile = profiles[0]
        sequences = digitize(profile.alphabet, ["", "A"])
        packed_ssv = _pack_profiles(profiles)
        packed_bias, m_mu, m_lambda = pack_bias_profiles(background, profiles, 1.0)
        with _native.SequenceBatch(bytearray(), array("Q", [0]), 29) as empty:
            self.assertEqual(
                empty.bias_candidates_many_raw(
                    *packed_ssv, m_mu, m_lambda, 1.0, packed_bias
                ),
                [[]],
            )
        with native_batch(sequences) as batch:
            rows = batch.bias_candidates_many_raw(
                *packed_ssv, m_mu, m_lambda, 1.0, packed_bias
            )
            self.assertEqual(rows[0][0][1], _native.BIAS_CPU_REQUIRED)
            self.assertEqual(rows[0][0][2], _native.STATUS_EMPTY)
            self.assertIsNone(rows[0][0][4])

            corrupted = bytearray(packed_bias)
            struct.pack_into("=ffff", corrupted, 24, 0.0, 0.0, 1.0, 1.0)
            uncertain = batch.bias_candidates_many_raw(
                *packed_ssv, m_mu, m_lambda, 1.0, corrupted
            )
            self.assertEqual(uncertain[0][1][1], _native.BIAS_CPU_REQUIRED)
            self.assertIsNone(uncertain[0][1][4])


if __name__ == "__main__":
    unittest.main()
