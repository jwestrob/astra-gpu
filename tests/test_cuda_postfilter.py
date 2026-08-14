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


DATA = Path(pyhmmer.__file__).parent / "tests" / "data"
HMM_20AA = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
PROTEINS = DATA / "seqs" / "938293.PRJEB85.HG003687.faa"
POSTFILTER_FORMAT = "=IfhBBf"
BIAS_FORMAT = "=IfhBB"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


def bits(value):
    return struct.unpack("=I", struct.pack("=f", value))[0]


def load_profile(name, length=400):
    with pyhmmer.plan7.HMMFile(DATA / "hmms" / "txt" / name) as hmm_file:
        hmm = hmm_file.read()
    background = pyhmmer.plan7.Background(hmm.alphabet)
    return hmm, background, hmm.to_profile(background, L=length).to_optimized()


def digitize(profile, values):
    return [
        pyhmmer.easel.TextSequence(
            name=f"target-{i}".encode(), sequence=value
        ).digitize(profile.alphabet)
        for i, value in enumerate(values)
    ]


def native_batch(sequences):
    residues = bytearray()
    offsets = array("Q", [0])
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        offsets.append(len(residues))
    return _native.SequenceBatch(residues, offsets, sequences[0].alphabet.Kp)


def bias_inputs(background, profiles, f1):
    packed = bytearray()
    mu = array("f")
    lambda_ = array("f")
    for profile in profiles:
        m_mu, m_lambda = profile.evalue_parameters.as_vector()[:2]
        mode, cutoff = _native.f1_cutoff(m_mu, m_lambda, f1)
        mu.append(m_mu)
        lambda_.append(m_lambda)
        packed.extend(
            _native.pack_bias_profile_raw(
                memoryview(background.residue_frequencies),
                memoryview(profile.compositions),
                profile.M,
                profile.scale_b,
                mode,
                math.nan if cutoff is None else cutoff,
            )
        )
    return packed, mu, lambda_


def scalar_full_msv_numerator(profile, sequence):
    q_count = max(2, (profile.M + 15) // 16)
    costs = memoryview(profile.rbv)
    previous = [0] * (profile.M + 1)
    xj = 0
    transition = (profile.tjb + profile.tbm) & 0xFF
    xb = max(0, profile.base_b - transition)
    for residue in sequence.sequence:
        diagonal = 0
        xe = 0
        for k in range(profile.M):
            old = previous[k + 1]
            value = min(255, max(diagonal, xb) + profile.bias_b)
            value = max(
                0,
                value - costs[residue, 16 * (k % q_count) + k // q_count],
            )
            previous[k + 1] = value
            diagonal = old
            xe = max(xe, value)
        if min(255, xe + profile.bias_b) == 255:
            return None
        xe = max(0, xe - profile.tec)
        xj = max(xj, xe)
        xb = max(0, max(profile.base_b, xj) - transition)
    return xj - profile.tjb - profile.base_b


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaPostfilterTests(unittest.TestCase):
    def records(self, background, profiles, sequences, f1):
        packed = _pack_profiles(profiles)
        packed_bias, mu, lambda_ = bias_inputs(background, profiles, f1)
        with _native.ViterbiProfiles(profiles) as resident:
            with native_batch(sequences) as batch:
                records, offsets = batch.postfilter_candidates_many_csr_raw(
                    *packed,
                    mu,
                    lambda_,
                    f1,
                    packed_bias,
                    profiles,
                    resident,
                )
        return records, offsets

    def test_versioned_abi_and_direct_prefix(self):
        self.assertEqual(_native.POSTFILTER_RECORD_VERSION, 1)
        self.assertEqual(_native.POSTFILTER_RESULT_SIZE, 16)
        self.assertEqual(_native.BIAS_RESULT_SIZE, 12)
        self.assertEqual(struct.calcsize(POSTFILTER_FORMAT), 16)

        _, background, profile = load_profile("Thioesterase.hmm")
        with pyhmmer.easel.SequenceFile(
            PROTEINS, digital=True, alphabet=profile.alphabet
        ) as sequence_file:
            sequence = sequence_file.read_block()[13]
        packed = _pack_profiles([profile])
        packed_bias, mu, lambda_ = bias_inputs(background, [profile], 0.02)
        with _native.ViterbiProfiles([profile]) as resident:
            with native_batch([sequence]) as batch:
                bias_records, _ = batch.bias_candidates_many_csr_raw(
                    *packed, mu, lambda_, 0.02, packed_bias
                )
                post_records, offsets = batch.postfilter_candidates_many_csr_raw(
                    *packed,
                    mu,
                    lambda_,
                    0.02,
                    packed_bias,
                    [profile],
                    resident,
                )
        self.assertEqual(list(offsets), [0, 1])
        self.assertEqual(post_records[:12], bias_records)
        sequence_index, filtersc, numerator, status, action, vfsc = struct.unpack(
            POSTFILTER_FORMAT, post_records
        )
        self.assertEqual((sequence_index, numerator, status, action), (0, -27, 0, 2))
        pipeline = pyhmmer.plan7.Pipeline(profile.alphabet)
        self.assertEqual(
            bits(filtersc),
            _pipeline._bias_filter_score_bits(pipeline, profile, sequence),
        )
        self.assertEqual(bits(vfsc), 0xC114CFB3)

    def test_mixed_empty_and_exact_full_msv_fallback(self):
        _, background, profile = load_profile("Thioesterase.hmm")
        with pyhmmer.easel.SequenceFile(
            PROTEINS, digital=True, alphabet=profile.alphabet
        ) as sequence_file:
            source = sequence_file.read_block()
        empty = digitize(profile, [""])[0]
        sequences = [empty, source[13], source[67], source[1992]]
        records, offsets = self.records(background, [profile], sequences, 0.02)
        observed = list(struct.iter_unpack(POSTFILTER_FORMAT, records))
        self.assertEqual(list(offsets), [0, 4])
        self.assertEqual(observed[0][0:1] + observed[0][2:5], (0, 0, 255, 0))
        self.assertTrue(math.isnan(observed[0][1]))
        self.assertTrue(math.isnan(observed[0][5]))
        expected = (
            (1, -27, 0, 0xC114CFB3),
            (2, -12, 0, 0xC0BFBB60),
            (3, 10, 0, 0xBF1A6494),
        )
        for record, (index, numerator, status, vfsc_bits) in zip(
            observed[1:], expected, strict=True
        ):
            self.assertEqual(
                (record[0], record[2], record[3]), (index, numerator, status)
            )
            self.assertIn(
                record[4], (_native.BIAS_DEFINITE_REJECT, _native.BIAS_DEFINITE_PASS)
            )
            self.assertEqual(bits(record[5]), vfsc_bits)

    def test_ambiguity_full_msv_erange_and_raw_f1_reject(self):
        with pyhmmer.plan7.HMMFile(HMM_20AA) as hmm_file:
            hmm = hmm_file.read()
        background = pyhmmer.plan7.Background(hmm.alphabet)
        profile = hmm.to_profile(background, L=100).to_optimized()
        sequences = digitize(
            profile,
            ["ACDEX", "ACDEX" * 3, "ACDEFGHIKLMNPQRSTVWY"],
        )
        records, _ = self.records(background, [profile], sequences, 1.0)
        rows = list(struct.iter_unpack(POSTFILTER_FORMAT, records))
        profile.L = len(sequences[0])
        self.assertEqual(rows[0][2], scalar_full_msv_numerator(profile, sequences[0]))
        self.assertEqual(rows[0][3], _native.STATUS_OK)
        self.assertEqual(rows[0][4], _native.BIAS_CPU_REQUIRED)
        self.assertTrue(math.isnan(rows[0][1]))
        self.assertTrue(math.isnan(rows[0][5]))
        for row in rows[1:]:
            self.assertEqual(row[3], _native.STATUS_ERANGE)
            self.assertEqual(row[4], _native.BIAS_CPU_REQUIRED)
            self.assertTrue(math.isnan(row[1]))
            self.assertTrue(math.isnan(row[5]))

        rejected, _ = self.records(background, [profile], sequences[:1], 0.0)
        row = struct.unpack(POSTFILTER_FORMAT, rejected)
        self.assertEqual(row[0], 0)
        self.assertEqual(row[3], _native.STATUS_OK)
        self.assertEqual(row[4], _native.BIAS_DEFINITE_REJECT)
        self.assertTrue(math.isnan(row[1]))
        self.assertTrue(math.isnan(row[5]))

    def test_source_identity_and_mutation_are_rejected(self):
        hmm, background, profile = load_profile("Thioesterase.hmm")
        with pyhmmer.easel.SequenceFile(
            PROTEINS, digital=True, alphabet=profile.alphabet
        ) as sequence_file:
            sequence = sequence_file.read_block()[13]
        packed = _pack_profiles([profile])
        packed_bias, mu, lambda_ = bias_inputs(background, [profile], 0.02)
        other = profile.copy()
        with _native.ViterbiProfiles([profile]) as resident:
            with native_batch([sequence]) as batch:
                with self.assertRaisesRegex(ValueError, "identity"):
                    batch.postfilter_candidates_many_csr_raw(
                        *packed,
                        mu,
                        lambda_,
                        0.02,
                        packed_bias,
                        [other],
                        resident,
                    )
                changed_hmm = hmm.copy()
                changed_hmm.match_emissions[1, 0] = 0.0
                changed_hmm.renormalize()
                profile.convert(changed_hmm.to_profile(background, L=400))
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    batch.postfilter_candidates_many_csr_raw(
                        *packed,
                        mu,
                        lambda_,
                        0.02,
                        packed_bias,
                        [profile],
                        resident,
                    )

    def test_lossy_sbv_cell_uses_exact_full_msv_scores(self):
        alphabet = pyhmmer.easel.Alphabet.amino()
        hmm = pyhmmer.plan7.HMM.sample(alphabet, 100_000, 42)
        background = pyhmmer.plan7.Background(alphabet)
        hmm.match_emissions[1, 0] = 0.0
        hmm.renormalize()
        profile = hmm.to_profile(background, L=100).to_optimized()
        sequence = digitize(profile, ["A" * 100])[0]

        q_count = max(2, (profile.M + 15) // 16)
        column = 16 * (0 % q_count) + 0 // q_count
        rbv = memoryview(profile.rbv)
        sbv = memoryview(profile.sbv)
        raw = sbv[0, column]
        signed = raw if raw < 128 else raw - 256
        self.assertEqual(rbv[0, column], 255)
        self.assertEqual(raw, 127)
        self.assertEqual(signed + profile.bias_b, profile.bias_b + 127)
        self.assertNotEqual(signed + profile.bias_b, rbv[0, column])

        packed = _pack_profiles([profile])
        packed_bias, mu, lambda_ = bias_inputs(background, [profile], 1.0)
        with native_batch([sequence]) as batch:
            diagnostic, _ = batch.bias_candidates_many_csr_raw(
                *packed, mu, lambda_, 1.0, packed_bias
            )
        self.assertEqual(
            struct.unpack(BIAS_FORMAT, diagnostic)[3],
            _native.STATUS_ENORESULT,
        )

        records, _ = self.records(background, [profile], [sequence], 1.0)
        row = struct.unpack(POSTFILTER_FORMAT, records)
        self.assertEqual(row[3], _native.STATUS_OK)
        profile.L = len(sequence)
        cpu_score = profile.msv_filter(sequence)
        gpu_score = row[2] / profile.scale_b - 3.0
        self.assertEqual(bits(gpu_score), bits(cpu_score))

    def test_bias_reject_still_carries_exact_external_viterbi(self):
        _, background, profile = load_profile("LuxC.hmm")
        with pyhmmer.easel.SequenceFile(
            PROTEINS, digital=True, alphabet=profile.alphabet
        ) as sequence_file:
            sequence = sequence_file.read_block()[6]
        records, offsets = self.records(background, [profile], [sequence], 0.2)
        self.assertEqual(list(offsets), [0, 1])
        row = struct.unpack(POSTFILTER_FORMAT, records)
        self.assertEqual(row[3], _native.STATUS_OK)
        self.assertEqual(row[4], _native.BIAS_DEFINITE_REJECT)
        self.assertTrue(math.isfinite(row[1]))
        self.assertEqual(bits(row[5]), 0xC164B4CC)
        pipeline = pyhmmer.plan7.Pipeline(profile.alphabet)
        self.assertEqual(
            bits(row[1]),
            _pipeline._bias_filter_score_bits(pipeline, profile, sequence),
        )

    @unittest.skipUnless(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "fenv layout is Linux x86_64 specific",
    )
    def test_unattested_float_environment_fails_closed(self):
        _, background, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEX"])[0]
        packed = _pack_profiles([profile])
        packed_bias, mu, lambda_ = bias_inputs(background, [profile], 0.02)
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original = libc.fegetround()
        with _native.ViterbiProfiles([profile]) as resident:
            try:
                self.assertEqual(libc.fesetround(0x400), 0)
                with native_batch([sequence]) as batch:
                    records, _ = batch.postfilter_candidates_many_csr_raw(
                        *packed,
                        mu,
                        lambda_,
                        0.02,
                        packed_bias,
                        [profile],
                        resident,
                    )
            finally:
                self.assertEqual(libc.fesetround(original), 0)
        row = struct.unpack(POSTFILTER_FORMAT, records)
        self.assertEqual(
            (row[0], row[3], row[4]),
            (0, _native.STATUS_ENORESULT, _native.BIAS_CPU_REQUIRED),
        )
        self.assertTrue(math.isnan(row[1]))
        self.assertTrue(math.isnan(row[5]))


if __name__ == "__main__":
    unittest.main()
