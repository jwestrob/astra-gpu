import math
import struct
import sys
import unittest
from array import array
from pathlib import Path

import pyhmmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
try:
    from plan7_gpu import SequenceBatch, _native, filter_ssv
except ImportError:
    SequenceBatch = None
    _native = None
    filter_ssv = None

HMM_20AA = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
HMM_M1 = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "M1.hmm"


def float32_bits(value):
    return struct.unpack("=I", struct.pack("=f", value))[0]


def scalar_xe(profile, sequence):
    M = profile.M
    Q = max(2, (M + 15) // 16)
    scores = memoryview(profile.sbv)
    previous = [-128] * (M + 1)
    maximum = 128

    for residue in sequence.sequence:
        diagonal = -128
        for k in range(M):
            old = previous[k + 1]
            raw_cost = scores[residue, 16 * (k % Q) + k // Q]
            cost = raw_cost if raw_cost < 128 else raw_cost - 256
            value = max(-128, min(127, diagonal - cost))
            previous[k + 1] = value
            maximum = max(maximum, value & 0xFF)
            diagonal = old
    return maximum


def scalar_full_msv_numerator(profile, sequence):
    M = profile.M
    Q = max(2, (M + 15) // 16)
    costs = memoryview(profile.rbv)
    previous = [0] * (M + 1)
    xJ = 0
    transition = (profile.tjb + profile.tbm) & 0xFF
    xB = max(0, profile.base_b - transition)

    for residue in sequence.sequence:
        diagonal = 0
        row_maximum = 0
        for k in range(M):
            old = previous[k + 1]
            value = max(diagonal, xB)
            value = min(255, value + profile.bias_b)
            value = max(0, value - costs[residue, 16 * (k % Q) + k // Q])
            previous[k + 1] = value
            row_maximum = max(row_maximum, value)
            diagonal = old
        if min(255, row_maximum + profile.bias_b) == 255:
            return None
        row_maximum = max(0, row_maximum - profile.tec)
        xJ = max(xJ, row_maximum)
        xB = max(0, max(xJ, profile.base_b) - transition)
    return xJ - profile.tjb - profile.base_b


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaSsvTests(unittest.TestCase):
    @staticmethod
    def optimized(path, length=100):
        with pyhmmer.plan7.HMMFile(path) as hmm_file:
            hmm = next(hmm_file)
        background = pyhmmer.plan7.Background(hmm.alphabet)
        return hmm.to_profile(background, L=length).to_optimized()

    @staticmethod
    def sequences(profile, values):
        return [
            pyhmmer.easel.TextSequence(
                name=f"sequence-{index}".encode(), sequence=value
            ).digitize(profile.alphabet)
            for index, value in enumerate(values)
        ]

    def assert_matches_hmmer(self, profile, sequences):
        original_length = profile.L
        observed = filter_ssv(profile, sequences)
        self.assertEqual(profile.L, original_length)
        self.assertEqual(len(observed), len(sequences))

        for sequence, result in zip(sequences, observed, strict=True):
            profile.L = len(sequence)
            reference = profile.ssv_filter(sequence)
            if math.isnan(reference):
                expected_status = "eslENORESULT"
                expected_bits = None
            elif math.isinf(reference):
                expected_status = "eslERANGE"
                expected_bits = None
            else:
                expected_status = "eslOK"
                expected_bits = float32_bits(reference)
            self.assertEqual(result["status"], expected_status)
            self.assertEqual(result["tjb_u8"], profile.tjb)
            self.assertEqual(result["score_bits"], expected_bits)
            self.assertEqual(result["xE_u8"], scalar_xe(profile, sequence))

    def test_all_public_ssv_routes(self):
        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(
            profile,
            ["G", "ACDEX", "ACDEX" * 3, "ACDEFGHIKLMNPQRSTVWY"],
        )
        self.assert_matches_hmmer(profile, sequences)
        self.assertEqual(
            [result["status"] for result in filter_ssv(profile, sequences)],
            ["eslOK", "eslENORESULT", "eslENORESULT", "eslERANGE"],
        )
        self.assertEqual(
            [result["action"] for result in filter_ssv(profile, sequences)],
            ["threshold_score", "cpu_msv", "cpu_msv", "promote"],
        )

    def test_short_long_short_and_ambiguity_codes(self):
        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(
            profile,
            ["ACDEX", "G" * 340, "ACDEX", "BZJXUO", "ACDEFGHIK*MNPQRSTVWY"],
        )
        self.assert_matches_hmmer(profile, sequences)
        results = filter_ssv(profile, sequences)
        self.assertEqual(results[0], results[2])

    def test_direct_ssv_need_not_equal_full_msv(self):
        profile = self.optimized(HMM_M1)
        sequence = self.sequences(profile, ["G" * 10])[0]
        self.assert_matches_hmmer(profile, [sequence])
        profile.L = len(sequence)
        result = filter_ssv(profile, [sequence])[0]
        self.assertNotEqual(
            result["numerator"],
            scalar_full_msv_numerator(profile, sequence),
        )

    def test_empty_sequence_is_marked_without_filtering(self):
        profile = self.optimized(HMM_M1)
        sequence = self.sequences(profile, [""])[0]
        result = filter_ssv(profile, [sequence])[0]
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["xE_u8"], 128)
        self.assertIsNone(result["score_bits"])

    def test_empty_batch_is_valid(self):
        profile = self.optimized(HMM_M1)
        self.assertEqual(filter_ssv(profile, []), [])

    def test_persistent_batch_matches_repeated_one_shot_calls(self):
        first_profile = self.optimized(HMM_20AA)
        second_profile = self.optimized(HMM_M1)
        sequences = self.sequences(
            first_profile,
            ["", "G", "ACDEX", "ACDEFGHIKLMNPQRSTVWY"],
        )
        expected_first = filter_ssv(first_profile, sequences)
        expected_second = filter_ssv(second_profile, sequences)

        with SequenceBatch(sequences) as batch:
            self.assertEqual(len(batch), len(sequences))
            self.assertEqual(batch.filter_ssv(first_profile), expected_first)
            self.assertEqual(filter_ssv(second_profile, batch), expected_second)
            self.assertEqual(batch.filter_ssv(first_profile), expected_first)

        self.assertTrue(batch.closed)

    def test_persistent_batch_invalidates_length_transition_cache(self):
        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(profile, ["G", "ACDEX", "G" * 100])
        lengths = array("Q", map(len, sequences))
        scores = memoryview(profile.sbv).cast("B")
        batch = SequenceBatch(sequences)

        def observed_tjb(scale):
            raw = batch._native.filter_raw(
                scores,
                profile.sbv.shape[1],
                profile.M,
                profile.alphabet.Kp,
                profile.tbm,
                profile.tec,
                profile.base_b,
                profile.bias_b,
                scale,
            )
            return bytearray(result[2] for result in raw)

        try:
            first_scale = profile.scale_b
            second_scale = first_scale * 1.25
            for scale in (first_scale, first_scale, second_scale, first_scale):
                self.assertEqual(
                    observed_tjb(scale),
                    _native.tjb_for_lengths(scale, lengths),
                )
        finally:
            batch.close()

    def test_persistent_batch_lifecycle_is_idempotent(self):
        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(profile, ["G", "ACDEX"])
        batch = SequenceBatch(iter(sequences))
        del sequences
        self.assertFalse(batch.closed)
        self.assertEqual(len(batch.filter_ssv(profile)), 2)

        batch.close()
        batch.close()
        self.assertTrue(batch.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            batch.filter_ssv(profile)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            batch.__enter__()

    def test_persistent_empty_and_independent_batches(self):
        profile = self.optimized(HMM_20AA)
        empty = SequenceBatch([], alphabet=profile.alphabet)
        first = SequenceBatch(self.sequences(profile, ["G"]))
        second = SequenceBatch(self.sequences(profile, ["ACDEX", ""]))
        try:
            self.assertEqual(empty.filter_ssv(profile), [])
            self.assertEqual(
                first.filter_ssv(profile),
                filter_ssv(profile, self.sequences(profile, ["G"])),
            )
            self.assertEqual(
                second.filter_ssv(profile),
                filter_ssv(profile, self.sequences(profile, ["ACDEX", ""])),
            )
            self.assertEqual(len(first.filter_ssv(profile)), 1)
        finally:
            empty.close()
            first.close()
            second.close()

    def test_all_length_transition_quantization_matches_hmmer(self):
        profile = self.optimized(HMM_20AA)
        lengths = array("Q", range(100_001))
        observed = _native.tjb_for_lengths(profile.scale_b, lengths)
        expected = bytearray()
        for length in lengths:
            profile.L = length
            expected.append(profile.tjb)
        self.assertEqual(observed, expected)

    def test_length_transition_rejects_invalid_inputs(self):
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                _native.tjb_for_lengths(scale, array("Q", [1]))
        with self.assertRaises(ValueError):
            _native.tjb_for_lengths(3.0, array("Q", [100_001]))


if __name__ == "__main__":
    unittest.main()
