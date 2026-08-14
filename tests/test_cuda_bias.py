import ctypes
import math
import platform
import struct
import sys
import unittest
from array import array
from pathlib import Path

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
