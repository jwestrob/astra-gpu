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
    from plan7_gpu import SequenceBatch, _native, cpu_candidates, filter_ssv
except ImportError:
    SequenceBatch = None
    _native = None
    cpu_candidates = None
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

    def test_f1_candidates_retain_every_non_direct_route(self):
        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(
            profile,
            ["G", "ACDEX", "ACDEX" * 3, "ACDEFGHIKLMNPQRSTVWY", ""],
        )
        original_length = profile.L
        with SequenceBatch(sequences) as batch:
            self.assertEqual(batch.cpu_candidates(profile), [1, 2, 3, 4])
            self.assertEqual(cpu_candidates(profile, batch), [1, 2, 3, 4])
        self.assertEqual(profile.L, original_length)

    def test_f1_exact_p_and_strict_threshold_boundary(self):
        profile = self.optimized(HMM_20AA)
        sequence = self.sequences(profile, ["G"])[0]
        result = filter_ssv(profile, [sequence])[0]
        parameters = profile.evalue_parameters.as_vector()
        action, probability = _native.f1_decision(
            _native.STATUS_OK,
            result["numerator"],
            1,
            profile.scale_b,
            parameters[0],
            parameters[1],
            1.0,
        )
        self.assertEqual(action, _native.F1_CPU_REQUIRED)
        self.assertEqual(probability.hex(), "0x1.e726801175503p-1")

        thresholds = (
            math.nextafter(probability, 0.0),
            probability,
            math.nextafter(probability, math.inf),
        )
        with SequenceBatch([sequence]) as batch:
            self.assertEqual(batch.cpu_candidates(profile, thresholds[0]), [])
            self.assertEqual(batch.cpu_candidates(profile, thresholds[1]), [0])
            self.assertEqual(batch.cpu_candidates(profile, thresholds[2]), [0])

    def test_f1_cutoff_matches_scalar_for_all_reachable_numerators(self):
        profiles = [self.optimized(HMM_20AA), self.optimized(HMM_M1)]
        lengths = (
            1,
            2,
            3,
            7,
            15,
            16,
            17,
            31,
            32,
            33,
            37,
            127,
            128,
            129,
            255,
            256,
            257,
            1023,
            4095,
            9999,
            65_535,
            99_999,
            100_000,
        )
        for profile in profiles:
            mu, lambda_ = profile.evalue_parameters.as_vector()[:2]
            _, boundary = _native.f1_decision(
                _native.STATUS_OK, 0, 37, profile.scale_b, mu, lambda_, 0.02
            )
            thresholds = (
                0.0,
                1e-300,
                1e-15,
                math.nextafter(boundary, 0.0),
                boundary,
                math.nextafter(boundary, math.inf),
                0.02,
                0.5,
                math.nextafter(1.0, 0.0),
            )
            for threshold in thresholds:
                mode, cutoff = _native.f1_cutoff(mu, lambda_, threshold)
                self.assertNotEqual(mode, _native.F1_CUTOFF_INVALID)
                cutoff_value = 0.0 if cutoff is None else cutoff
                for length in lengths:
                    for numerator in range(-510, 256):
                        scalar, _ = _native.f1_decision(
                            _native.STATUS_OK,
                            numerator,
                            length,
                            profile.scale_b,
                            mu,
                            lambda_,
                            threshold,
                        )
                        fast = _native.f1_cutoff_decision(
                            _native.STATUS_OK,
                            numerator,
                            length,
                            profile.scale_b,
                            mode,
                            cutoff_value,
                        )
                        if fast != scalar:
                            self.fail(
                                f"cutoff mismatch: threshold={threshold!r}, "
                                f"length={length}, numerator={numerator}"
                            )

    def test_f1_cutoff_extremes_and_invalid_inputs_fail_closed(self):
        minimum_subnormal = struct.unpack("=f", struct.pack("=I", 1))[0]
        mode, cutoff = _native.f1_cutoff(0.0, minimum_subnormal, 0.0)
        self.assertEqual(mode, _native.F1_CUTOFF_ALWAYS_REJECT)
        self.assertIsNone(cutoff)
        mode, cutoff = _native.f1_cutoff(
            0.0, minimum_subnormal, math.nextafter(1.0, 0.0)
        )
        self.assertEqual(mode, _native.F1_CUTOFF_ALWAYS_CPU)
        self.assertIsNone(cutoff)

        invalid_parameters = (
            (math.nan, 0.7, 0.02),
            (math.inf, 0.7, 0.02),
            (-99999.0, 0.7, 0.02),
            (0.0, 0.0, 0.02),
            (0.0, -1.0, 0.02),
            (0.0, -99999.0, 0.02),
            (0.0, math.nan, 0.02),
            (0.0, 0.7, -1.0),
            (0.0, 0.7, math.nan),
            (0.0, 0.7, math.inf),
        )
        for arguments in invalid_parameters:
            mode, cutoff = _native.f1_cutoff(*arguments)
            self.assertEqual(mode, _native.F1_CUTOFF_INVALID)
            self.assertIsNone(cutoff)

        for arguments in (
            (_native.STATUS_ENORESULT, 0, 10, 1.0),
            (_native.STATUS_OK, 0, 0, 1.0),
            (_native.STATUS_OK, 0, 100_001, 1.0),
            (_native.STATUS_OK, 0, 10, 0.0),
            (_native.STATUS_OK, 255, 10, minimum_subnormal),
            (_native.STATUS_OK, 0, 10, math.nan),
        ):
            self.assertEqual(
                _native.f1_cutoff_decision(
                    *arguments, _native.F1_CUTOFF_SCORE, math.nan
                ),
                _native.F1_CPU_REQUIRED,
            )

    def test_f1_cutoff_falls_back_at_easel_smallx_jump(self):
        smallx = 5e-9
        jump_bottom = 1.0 - math.exp(-smallx)
        threshold = (jump_bottom + smallx) / 2.0
        mu = struct.unpack("=f", struct.pack("=f", math.log(smallx)))[0]

        mode, cutoff = _native.f1_cutoff(mu, 1.0, threshold)
        self.assertEqual(mode, _native.F1_CUTOFF_INVALID)
        self.assertIsNone(cutoff)
        self.assertNotEqual(
            _native.f1_cutoff(mu, 1.0, smallx)[0],
            _native.F1_CUTOFF_INVALID,
        )

        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(profile, ["G", "ACDEX", "G" * 100, ""])
        parameters = profile.evalue_parameters.as_vector()
        original_mu = parameters[0]
        original_lambda = parameters[1]
        try:
            parameters[0] = mu
            parameters[1] = 1.0
            with SequenceBatch(sequences) as batch:
                expected = batch.cpu_candidates(profile, threshold)
                self.assertEqual(
                    batch.cpu_candidates_many([profile], threshold), [expected]
                )
        finally:
            parameters[0] = original_mu
            parameters[1] = original_lambda

    def test_f1_invalid_inputs_are_conservative(self):
        profile = self.optimized(HMM_20AA)
        sequences = self.sequences(profile, ["G", "ACDEX", ""])
        all_indexes = list(range(len(sequences)))
        with SequenceBatch(sequences) as batch:
            for threshold in (1.0, -1.0, 1.1, math.nan, math.inf):
                self.assertEqual(batch.cpu_candidates(profile, threshold), all_indexes)

            parameters = profile.evalue_parameters.as_vector()
            original_mu = parameters[0]
            original_lambda = parameters[1]
            try:
                parameters[0] = math.nan
                self.assertEqual(batch.cpu_candidates(profile), all_indexes)
                parameters[0] = original_mu
                parameters[1] = -99999.0
                self.assertEqual(batch.cpu_candidates(profile), all_indexes)
            finally:
                parameters[0] = original_mu
                parameters[1] = original_lambda

        for status in (
            _native.STATUS_ERANGE,
            _native.STATUS_ENORESULT,
            _native.STATUS_EMPTY,
            254,
        ):
            action, probability = _native.f1_decision(
                status, 0, 10, profile.scale_b, -1.0, 0.7, 0.02
            )
            self.assertEqual(action, _native.F1_CPU_REQUIRED)
            self.assertTrue(math.isnan(probability))

        parameters = profile.evalue_parameters.as_vector()
        invalid_arguments = (
            (
                _native.STATUS_OK,
                0,
                0,
                profile.scale_b,
                parameters[0],
                parameters[1],
                0.02,
            ),
            (
                _native.STATUS_OK,
                0,
                100_001,
                profile.scale_b,
                parameters[0],
                parameters[1],
                0.02,
            ),
            (_native.STATUS_OK, 0, 10, 0.0, parameters[0], parameters[1], 0.02),
            (_native.STATUS_OK, 0, 10, profile.scale_b, -99999.0, parameters[1], 0.02),
            (_native.STATUS_OK, 0, 10, profile.scale_b, parameters[0], 0.0, 0.02),
            (_native.STATUS_OK, 0, 10, profile.scale_b, parameters[0], -99999.0, 0.02),
            (
                _native.STATUS_OK,
                0,
                10,
                profile.scale_b,
                parameters[0],
                parameters[1],
                math.nan,
            ),
        )
        for arguments in invalid_arguments:
            action, probability = _native.f1_decision(*arguments)
            self.assertEqual(action, _native.F1_CPU_REQUIRED)
            self.assertTrue(math.isnan(probability))

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

    def test_multi_profile_batch_matches_repeated_calls(self):
        profiles = [
            self.optimized(HMM_20AA),
            self.optimized(HMM_M1),
            self.optimized(HMM_20AA),
        ]
        sequences = self.sequences(
            profiles[0], ["", "G", "ACDEX", "ACDEFGHIKLMNPQRSTVWY"]
        )
        expected_results = [filter_ssv(profile, sequences) for profile in profiles]
        expected_candidates = [
            cpu_candidates(profile, sequences) for profile in profiles
        ]
        original_lengths = [profile.L for profile in profiles]

        with SequenceBatch(sequences) as batch:
            self.assertEqual(batch.filter_ssv_many(profiles), expected_results)
            self.assertEqual(batch.cpu_candidates_many(profiles), expected_candidates)
            self.assertEqual(batch.filter_ssv(profiles[0]), expected_results[0])
            self.assertEqual(batch.filter_ssv_many(profiles[:1]), expected_results[:1])
            self.assertEqual(
                batch.cpu_candidates_many(reversed(profiles)),
                list(reversed(expected_candidates)),
            )
        self.assertEqual([profile.L for profile in profiles], original_lengths)

    def test_multi_profile_empty_and_invalid_parameter_cases(self):
        profiles = [self.optimized(HMM_20AA), self.optimized(HMM_M1)]
        sequences = self.sequences(profiles[0], ["G", "ACDEX", ""])
        with SequenceBatch(sequences) as batch:
            self.assertEqual(batch.filter_ssv_many([]), [])
            self.assertEqual(batch.cpu_candidates_many([]), [])
            self.assertEqual(
                batch.cpu_candidates_many(profiles, F1=1.0),
                [list(range(len(sequences)))] * len(profiles),
            )

            parameters = profiles[1].evalue_parameters.as_vector()
            original_mu = parameters[0]
            try:
                parameters[0] = math.nan
                self.assertEqual(
                    batch.cpu_candidates_many(profiles),
                    [
                        cpu_candidates(profiles[0], sequences),
                        list(range(len(sequences))),
                    ],
                )
            finally:
                parameters[0] = original_mu

        with SequenceBatch([], alphabet=profiles[0].alphabet) as empty:
            self.assertEqual(empty.filter_ssv_many(profiles), [[], []])
            self.assertEqual(empty.cpu_candidates_many(profiles), [[], []])

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
