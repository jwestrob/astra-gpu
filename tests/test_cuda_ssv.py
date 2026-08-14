import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock
from array import array
from pathlib import Path

import pyhmmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
try:
    from plan7_gpu import (
        CandidateBatch,
        PressedProfilePair,
        SequenceBatch,
        _native,
        _pipeline,
        cpu_candidates,
        filter_ssv,
        load_pressed_profiles,
    )
    from plan7_gpu.adapter import _pack_profiles
    from plan7_gpu.adapter import (
        _PIPELINE_LEASES,
        _candidate_state,
        _pair_state,
        _sequence_native,
    )
    from plan7_gpu._abi import pyhmmer_abi_fingerprint
    import plan7_gpu.pressed_manifest as pressed_manifest
except ImportError:
    SequenceBatch = None
    _native = None
    _pipeline = None
    cpu_candidates = None
    filter_ssv = None
    _pack_profiles = None
    CandidateBatch = None
    PressedProfilePair = None
    load_pressed_profiles = None
    _PIPELINE_LEASES = None
    _candidate_state = None
    _pair_state = None
    _sequence_native = None
    pyhmmer_abi_fingerprint = None
    pressed_manifest = None

HMM_20AA = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
HMM_M1 = ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "M1.hmm"
HMM_STRIPE_BOUNDARIES = ROOT / "results" / "datasets" / "pfam-stripe-boundaries.hmm"
HMM_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
FASTA_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins45.fa"


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
            for threshold in thresholds:
                self.assertEqual(
                    batch.cpu_candidates_many([profile], threshold),
                    [batch.cpu_candidates(profile, threshold)],
                )

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

    def test_f1_cutoff_fails_closed_at_easel_smallx_jump(self):
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
                self.assertEqual(expected, [1, 3])
                self.assertEqual(
                    batch.cpu_candidates_many([profile], threshold),
                    [list(range(len(sequences)))],
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

    def test_fused_candidates_deduplicate_nonadjacent_scale_rows(self):
        profile = self.optimized(HMM_20AA)
        profiles = [profile.copy(), profile.copy(), profile.copy()]
        sequences = self.sequences(
            profile, ["", "G", "ACDEX", "ACDEFGHIKLMNPQRSTVWY", "G" * 100]
        )
        packed = _pack_profiles(profiles)
        scales = array("f", [profile.scale_b, profile.scale_b * 1.25, profile.scale_b])
        parameters = profile.evalue_parameters.as_vector()
        m_mu = array("f", [parameters[0]] * len(profiles))
        m_lambda = array("f", [parameters[1]] * len(profiles))
        threshold = 0.02

        with SequenceBatch(sequences) as batch:
            native = _sequence_native(batch)
            scores = memoryview(profile.sbv).cast("B")
            expected = []
            for scale in scales:
                raw = native.filter_raw(
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
                row = []
                for index, result in enumerate(raw):
                    action, _ = _native.f1_decision(
                        result[0],
                        result[3],
                        len(sequences[index]),
                        scale,
                        parameters[0],
                        parameters[1],
                        threshold,
                    )
                    if action == _native.F1_CPU_REQUIRED:
                        row.append(index)
                expected.append(row)

            indices, offsets = native.cpu_candidates_many_csr_raw(
                packed.scores,
                packed.score_offsets,
                packed.score_counts,
                packed.score_strides,
                packed.model_lengths,
                packed.constants,
                scales,
                m_mu,
                m_lambda,
                threshold,
            )
            observed = [
                list(indices[offsets[row] : offsets[row + 1]])
                for row in range(len(profiles))
            ]
            self.assertEqual(observed, expected)

    def test_fused_candidate_mask_word_boundaries_and_fail_closed_rows(self):
        profile = self.optimized(HMM_20AA)
        profiles = [profile.copy(), profile.copy()]
        packed = _pack_profiles(profiles)
        parameters = profile.evalue_parameters.as_vector()
        m_mu = array("f", [parameters[0], math.nan])
        m_lambda = array("f", [parameters[1], parameters[1]])
        values = ["G", "ACDEX", "ACDEX" * 3, "ACDEFGHIKLMNPQRSTVWY", ""]

        for target_count in (31, 32, 33, 63, 64, 65):
            with self.subTest(target_count=target_count):
                sequences = self.sequences(
                    profile,
                    [values[index % len(values)] for index in range(target_count)],
                )
                with SequenceBatch(sequences) as batch:
                    expected = batch.cpu_candidates(profile)
                    native = _sequence_native(batch)
                    indices, offsets = native.cpu_candidates_many_csr_raw(
                        *packed, m_mu, m_lambda, 0.02
                    )

                self.assertEqual(list(indices[offsets[0] : offsets[1]]), expected)
                self.assertEqual(
                    list(indices[offsets[1] : offsets[2]]),
                    list(range(target_count)),
                )

    def test_compact_candidate_csr_matches_diagnostic_lists(self):
        profiles = [
            self.optimized(HMM_20AA),
            self.optimized(HMM_M1),
            self.optimized(HMM_20AA),
        ]
        sequences = self.sequences(
            profiles[0], ["", "G", "ACDEX", "ACDEFGHIKLMNPQRSTVWY"]
        )
        packed = _pack_profiles(profiles)
        m_mu = array(
            "f", [profile.evalue_parameters.as_vector()[0] for profile in profiles]
        )
        m_lambda = array(
            "f", [profile.evalue_parameters.as_vector()[1] for profile in profiles]
        )

        with SequenceBatch(sequences) as batch:
            native = _sequence_native(batch)
            empty_indices, empty_offsets = native.cpu_candidates_many_csr_raw(
                *_pack_profiles([]), array("f"), array("f"), 0.02
            )
            self.assertEqual(empty_indices.tolist(), [])
            self.assertEqual(empty_offsets.tolist(), [0])

            expected = batch.cpu_candidates_many(profiles)
            indices, offsets = native.cpu_candidates_many_csr_raw(
                *packed, m_mu, m_lambda, 0.02
            )
            observed = [
                list(indices[offsets[row] : offsets[row + 1]])
                for row in range(len(profiles))
            ]
            self.assertEqual(observed, expected)
            self.assertEqual(indices.typecode, "I")
            self.assertEqual(indices.itemsize, 4)
            self.assertEqual(offsets.typecode, "Q")
            self.assertEqual(offsets.tolist()[0], 0)
            self.assertEqual(offsets.tolist()[-1], len(indices))

            m_mu[1] = math.nan
            invalid_indices, invalid_offsets = native.cpu_candidates_many_csr_raw(
                *packed, m_mu, m_lambda, 0.02
            )
            self.assertEqual(
                list(invalid_indices[invalid_offsets[1] : invalid_offsets[2]]),
                list(range(len(sequences))),
            )

    def test_compact_many_matches_striped_at_model_boundaries(self):
        with pyhmmer.plan7.HMMFile(HMM_STRIPE_BOUNDARIES) as hmm_file:
            hmms = list(hmm_file)
        self.assertEqual([hmm.M for hmm in hmms], [15, 16, 17, 31, 32, 33])
        profiles = [
            hmm.to_profile(pyhmmer.plan7.Background(hmm.alphabet), L=100).to_optimized()
            for hmm in hmms
        ]
        symbols = profiles[0].alphabet.symbols
        sequences = self.sequences(
            profiles[0],
            [symbols, symbols[::-1], symbols[::2] + symbols[1::2]],
        )
        self.assertEqual(
            sorted({code for sequence in sequences for code in sequence.sequence}),
            list(range(profiles[0].alphabet.Kp)),
        )
        expected_results = [filter_ssv(profile, sequences) for profile in profiles]
        expected_candidates = [
            cpu_candidates(profile, sequences) for profile in profiles
        ]

        with SequenceBatch(sequences) as batch:
            self.assertEqual(batch.filter_ssv_many(profiles), expected_results)
            self.assertEqual(batch.cpu_candidates_many(profiles), expected_candidates)

    def test_compact_profile_packing_and_range_validation(self):
        profiles = [
            self.optimized(HMM_M1),
            self.optimized(HMM_20AA),
            self.optimized(HMM_M1),
        ]
        packed = _pack_profiles(profiles)
        expected = bytearray()
        expected_offsets = []
        for profile in profiles:
            expected_offsets.append(len(expected))
            striped = memoryview(profile.sbv)
            q_count = max(2, (profile.M + 15) // 16)
            for k in range(profile.M):
                column = 16 * (k % q_count) + k // q_count
                expected.extend(
                    striped[residue, column] for residue in range(profile.alphabet.Kp)
                )

        self.assertEqual(packed.scores, expected)
        self.assertEqual(packed.score_offsets.tolist(), expected_offsets)
        self.assertEqual(
            packed.score_counts.tolist(),
            [profile.M * profile.alphabet.Kp for profile in profiles],
        )
        self.assertEqual(
            packed.score_strides.tolist(),
            [profile.alphabet.Kp for profile in profiles],
        )

        sequences = self.sequences(profiles[0], ["G", "ACDEX"])
        with SequenceBatch(sequences) as batch:
            native = _sequence_native(batch)
            bad_offsets = array("Q", packed.score_offsets)
            bad_offsets[0] = 1
            with self.assertRaisesRegex(
                RuntimeError, "invalid compact profile score range"
            ):
                native.filter_many_raw(
                    packed.scores,
                    bad_offsets,
                    packed.score_counts,
                    packed.score_strides,
                    packed.model_lengths,
                    packed.constants,
                    packed.scales,
                )

            bad_strides = array("i", packed.score_strides)
            bad_strides[0] += 1
            with self.assertRaisesRegex(
                RuntimeError, "invalid compact profile score range"
            ):
                native.filter_many_raw(
                    packed.scores,
                    packed.score_offsets,
                    packed.score_counts,
                    bad_strides,
                    packed.model_lengths,
                    packed.constants,
                    packed.scales,
                )

            with self.assertRaisesRegex(RuntimeError, "trailing bytes"):
                native.filter_many_raw(
                    packed.scores + b"\0",
                    packed.score_offsets,
                    packed.score_counts,
                    packed.score_strides,
                    packed.model_lengths,
                    packed.constants,
                    packed.scales,
                )

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
            raw = _sequence_native(batch).filter_raw(
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


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaCandidateBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(
            prefix="plan7-candidate-batch-test-"
        )
        cls.pressed_base = Path(cls._temporary.name) / "globins"
        hmms = []
        for path in (HMM_GLOBINS, HMM_M1):
            with pyhmmer.plan7.HMMFile(path) as hmm_file:
                hmms.extend(hmm_file)
        pyhmmer.hmmer.hmmpress(hmms, cls.pressed_base)
        cls.manifest_path = Path(cls._temporary.name) / "globins.manifest.json"
        pressed_manifest.create_pressed_manifest(cls.pressed_base, cls.manifest_path)
        cls.pairs = load_pressed_profiles(cls.pressed_base)
        cls.alphabet = cls.pairs[0].hmm.alphabet
        with pyhmmer.easel.SequenceFile(
            FASTA_GLOBINS, digital=True, alphabet=cls.alphabet
        ) as sequence_file:
            cls.targets = sequence_file.read_block()

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @classmethod
    def pipeline(cls, **options):
        defaults = {"E": 10.0, "domE": 10.0, "incE": 10.0, "incdomE": 10.0}
        defaults.update(options)
        return pyhmmer.plan7.Pipeline(cls.alphabet, **defaults)

    @staticmethod
    def table_bytes(hits, format):
        output = io.BytesIO()
        hits.write(output, format=format, header=True)
        return output.getvalue()

    @staticmethod
    def hmm_bytes(hmm):
        output = io.BytesIO()
        hmm.write(output, binary=True)
        return output.getvalue()

    def copy_pressed_database(self, label):
        directory = Path(self._temporary.name) / self._testMethodName / label
        directory.mkdir(parents=True)
        base = directory / self.pressed_base.name
        for suffix in pressed_manifest.PRESSED_SUFFIXES:
            shutil.copy2(f"{self.pressed_base}.{suffix}", f"{base}.{suffix}")
        return base

    def assert_exact_hits(self, expected, actual):
        fields = (
            "Z",
            "domZ",
            "searched_models",
            "searched_nodes",
            "searched_residues",
            "searched_sequences",
        )
        self.assertEqual(
            tuple(getattr(actual, field) for field in fields),
            tuple(getattr(expected, field) for field in fields),
        )
        self.assertEqual(
            self.table_bytes(actual, "targets"),
            self.table_bytes(expected, "targets"),
        )
        self.assertEqual(
            self.table_bytes(actual, "domains"),
            self.table_bytes(expected, "domains"),
        )

    def test_default_f1_end_to_end_query_copy_and_reuse(self):
        pair = self.pairs[0]
        owner = pair.hmm
        expected_pipeline = self.pipeline()
        actual_pipeline = self.pipeline()
        with SequenceBatch(self.targets) as sequences:
            candidates = sequences.candidate_batch([pair])
            self.assertFalse(hasattr(sequences, "_sequences"))
        self.assertIsInstance(candidates, CandidateBatch)
        self.assertEqual(candidates.F1, actual_pipeline.F1)
        self.assertLessEqual(candidates.candidate_count(0), len(self.targets))
        self.assertFalse(hasattr(candidates, "_targets"))
        self.assertFalse(hasattr(candidates, "_pairs"))
        self.assertIsInstance(_candidate_state(candidates).indices, bytes)
        self.assertIsInstance(_candidate_state(candidates).offsets, bytes)

        expected_first = expected_pipeline.search_hmm(owner, self.targets)
        actual_first = candidates.search(0, actual_pipeline)
        self.assertIsNot(actual_first.query, owner)
        self.assertEqual(self.hmm_bytes(actual_first.query), self.hmm_bytes(owner))
        self.assert_exact_hits(expected_first, actual_first)

        expected_second = expected_pipeline.search_hmm(pair.hmm, self.targets)
        actual_second = candidates.search(0, actual_pipeline)
        self.assert_exact_hits(expected_second, actual_second)

    def test_manifest_fast_path_skips_rebuild_and_is_exact(self):
        with mock.patch(
            "plan7_gpu.adapter._verify_pressed_profile",
            side_effect=AssertionError("manifest path rebuilt a profile"),
        ):
            fast_pairs = load_pressed_profiles(
                self.pressed_base, manifest=self.manifest_path
            )

        self.assertEqual(len(fast_pairs), len(self.pairs))
        for expected, observed in zip(self.pairs, fast_pairs, strict=True):
            self.assertEqual(expected.canonical_base, observed.canonical_base)
            self.assertEqual(expected.ordinal, observed.ordinal)
            self.assertEqual(expected.stat_token, observed.stat_token)
            self.assertIs(type(expected.stat_token), type(observed.stat_token))
            self.assertEqual(expected.cutoffs, observed.cutoffs)
            self.assertEqual(self.hmm_bytes(expected.hmm), self.hmm_bytes(observed.hmm))

        pair = fast_pairs[0]
        with SequenceBatch(self.targets) as sequences:
            full_candidates = sequences.candidate_batch([self.pairs[0]])
            fast_candidates = sequences.candidate_batch([pair])
        full_state = _candidate_state(full_candidates)
        fast_state = _candidate_state(fast_candidates)
        self.assertEqual(full_state.all_rows, fast_state.all_rows)
        self.assertEqual(full_state.indices, fast_state.indices)
        self.assertEqual(full_state.offsets, fast_state.offsets)
        self.assertEqual(full_state.all_targets, fast_state.all_targets)
        self.assertEqual(
            fast_candidates.candidate_count(0),
            full_candidates.candidate_count(0),
        )
        expected = self.pipeline().search_hmm(pair.hmm, self.targets)
        actual = fast_candidates.search(0, self.pipeline())
        self.assert_exact_hits(expected, actual)

    def test_manifest_corruption_wrong_database_and_count_fail_closed(self):
        directory = Path(self._temporary.name) / self._testMethodName
        directory.mkdir()

        corrupt_manifest = directory / "corrupt.json"
        corrupt_manifest.write_bytes(b"{not valid JSON")
        with self.assertRaises(pressed_manifest.PressedManifestError):
            load_pressed_profiles(self.pressed_base, manifest=corrupt_manifest)

        wrong_base = directory / "wrong" / self.pressed_base.name
        wrong_base.parent.mkdir()
        pyhmmer.hmmer.hmmpress([self.pairs[-1].hmm], wrong_base)
        wrong_manifest = directory / "wrong.manifest.json"
        pressed_manifest.create_pressed_manifest(wrong_base, wrong_manifest)
        with self.assertRaisesRegex(
            pressed_manifest.PressedManifestError, "(?:size|SHA256) mismatch"
        ):
            load_pressed_profiles(self.pressed_base, manifest=wrong_manifest)

        count_manifest = directory / "count.manifest.json"
        count_data = json.loads(self.manifest_path.read_text(encoding="ascii"))
        count_data["database"]["model_count"] += 1
        count_data["verification"]["models_compared"] += 1
        count_manifest.write_text(
            json.dumps(count_data, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(ValueError, "profile count"):
            load_pressed_profiles(self.pressed_base, manifest=count_manifest)

    def test_manifest_validation_to_load_token_drift_fails_before_open(self):
        copied_base = self.copy_pressed_database("drift")
        original_validate = pressed_manifest.validate_pressed_manifest

        def validate_then_drift(database, manifest, **options):
            validation = original_validate(database, manifest, **options)
            member = Path(f"{database}.h3m")
            metadata = member.stat()
            os.utime(
                member,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
            )
            return validation

        with (
            mock.patch.object(
                pressed_manifest,
                "validate_pressed_manifest",
                side_effect=validate_then_drift,
            ),
            mock.patch.object(
                pyhmmer.plan7,
                "HMMFile",
                side_effect=AssertionError("database opened after token drift"),
            ),
            self.assertRaisesRegex(RuntimeError, "changed after.*validated"),
        ):
            load_pressed_profiles(copied_base, manifest=self.manifest_path)

    def test_pinned_loader_ignores_parent_symlink_target_swap(self):
        perturbed_hmms = [pair.hmm for pair in self.pairs]
        perturbed = perturbed_hmms[0]
        row = perturbed.match_emissions[1]
        source, destination = sorted(range(len(row)), key=row.__getitem__)[:2]
        epsilon = min(float(row[source]) / 2.0, 1e-4)
        row[source] -= epsilon
        row[destination] += epsilon
        perturbed.renormalize()
        self.assertEqual(perturbed.consensus, self.pairs[0].hmm.consensus)
        self.assertNotEqual(
            self.hmm_bytes(perturbed), self.hmm_bytes(self.pairs[0].hmm)
        )

        original_hmm_file = pyhmmer.plan7.HMMFile
        original_pressed_file = pyhmmer.plan7.HMMPressedFile
        options = {
            "without-manifest": {},
            "with-manifest": {"manifest": self.manifest_path},
        }
        for label, load_options in options.items():
            with self.subTest(label):
                victim_base = self.copy_pressed_database(label)
                root = victim_base.parent.parent
                perturbed_directory = root / f"{label}-perturbed"
                perturbed_directory.mkdir()
                perturbed_base = perturbed_directory / victim_base.name
                pyhmmer.hmmer.hmmpress(perturbed_hmms, perturbed_base)
                parked_directory = root / f"{label}-parked"
                victim_directory = victim_base.parent
                swapped = False

                def swap_parent_then_open(path, *args, **kwargs):
                    nonlocal swapped
                    if not swapped:
                        os.rename(victim_directory, parked_directory)
                        os.symlink(
                            perturbed_directory,
                            victim_directory,
                            target_is_directory=True,
                        )
                        swapped = True
                    return original_hmm_file(path, *args, **kwargs)

                def restore_parent():
                    if victim_directory.is_symlink():
                        victim_directory.unlink()
                        os.rename(parked_directory, victim_directory)

                class RestoreAfterPressedRead:
                    def __init__(self, inner):
                        self.inner = inner

                    def __enter__(self):
                        return self.inner.__enter__()

                    def __exit__(self, *args):
                        try:
                            return self.inner.__exit__(*args)
                        finally:
                            restore_parent()

                def open_pressed(path, *args, **kwargs):
                    return RestoreAfterPressedRead(
                        original_pressed_file(path, *args, **kwargs)
                    )

                try:
                    with (
                        mock.patch.object(
                            pyhmmer.plan7,
                            "HMMFile",
                            side_effect=swap_parent_then_open,
                        ),
                        mock.patch.object(
                            pyhmmer.plan7,
                            "HMMPressedFile",
                            side_effect=open_pressed,
                        ),
                    ):
                        observed = load_pressed_profiles(victim_base, **load_options)
                finally:
                    restore_parent()

                self.assertTrue(swapped)
                self.assertEqual(len(observed), len(self.pairs))
                self.assertEqual(
                    tuple(self.hmm_bytes(pair.hmm) for pair in observed),
                    tuple(self.hmm_bytes(pair.hmm) for pair in self.pairs),
                )
                self.assertNotEqual(
                    self.hmm_bytes(observed[0].hmm),
                    self.hmm_bytes(perturbed),
                )
                self.assertTrue(
                    all(pair.canonical_base == victim_base for pair in observed)
                )

    def test_f1_and_row_mismatches_fail_before_pipeline_mutation(self):
        pair = self.pairs[0]
        with SequenceBatch(self.targets) as sequences:
            candidates = sequences.candidate_batch([pair])
        long_targets = pyhmmer.plan7.LongTargetsPipeline(pyhmmer.easel.Alphabet.dna())
        with self.assertRaisesRegex(TypeError, "exactly pyhmmer.plan7.Pipeline"):
            candidates.search(0, long_targets)
        self.assertEqual(_PIPELINE_LEASES, {})

        pipeline = self.pipeline(F1=0.03)
        with self.assertRaisesRegex(ValueError, "does not match"):
            candidates.search(0, pipeline)
        with self.assertRaises(IndexError):
            candidates.search(1, pipeline)
        with self.assertRaises(TypeError):
            candidates.candidate_count(True)

        pipeline.F1 = candidates.F1
        actual = candidates.search(0, pipeline)
        expected = self.pipeline().search_hmm(pair.hmm, self.targets)
        self.assert_exact_hits(expected, actual)

    def test_custom_background_is_rejected_before_pipeline_mutation(self):
        pair = self.pairs[0]
        with SequenceBatch(self.targets) as sequences:
            candidates = sequences.candidate_batch([pair])

        uniform = self.pipeline(
            background=pyhmmer.plan7.Background(self.alphabet, uniform=True)
        )
        with self.assertRaisesRegex(ValueError, "canonical hmmpress background"):
            candidates.search(0, uniform)

        pipeline = self.pipeline()
        frequencies = pipeline.background.residue_frequencies
        original = (frequencies[0], frequencies[1])
        frequencies[0] = original[0] + 0.001
        frequencies[1] = original[1] - 0.001
        with self.assertRaisesRegex(ValueError, "canonical hmmpress background"):
            candidates.search(0, pipeline)
        frequencies[0], frequencies[1] = original

        actual = candidates.search(0, pipeline)
        expected = self.pipeline().search_hmm(pair.hmm, self.targets)
        self.assert_exact_hits(expected, actual)

    def test_same_pipeline_calls_are_process_wide_serialized(self):
        with SequenceBatch(self.targets) as first_sequences:
            first = first_sequences.candidate_batch([self.pairs[0]])
        with SequenceBatch(self.targets) as second_sequences:
            second = second_sequences.candidate_batch([self.pairs[1]])

        pipeline = self.pipeline()
        first_entered = threading.Event()
        second_started = threading.Event()
        second_entered = threading.Event()
        release_first = threading.Event()
        state_lock = threading.Lock()
        outcomes = {}
        call_count = 0
        original_search = _pipeline._search_hmm_candidates

        def fake_search(pipeline, query, optimized, targets, row):
            nonlocal call_count
            with state_lock:
                call_count += 1
                call_number = call_count
            if call_number == 1:
                first_entered.set()
                release_first.wait(2.0)
            else:
                second_entered.set()
            return query

        def invoke(name, candidates, started=None):
            if started is not None:
                started.set()
            try:
                outcomes[name] = candidates.search(0, pipeline)
            except BaseException as error:
                outcomes[name] = error

        first_thread = threading.Thread(target=invoke, args=("first", first))
        second_thread = threading.Thread(
            target=invoke, args=("second", second, second_started)
        )
        _pipeline._search_hmm_candidates = fake_search
        try:
            first_thread.start()
            self.assertTrue(first_entered.wait(2.0))
            second_thread.start()
            self.assertTrue(second_started.wait(2.0))
            self.assertFalse(second_entered.wait(0.1))
            pipeline.F1 = 0.03
            release_first.set()
            first_thread.join(2.0)
            second_thread.join(2.0)
        finally:
            release_first.set()
            if first_thread.ident is not None:
                first_thread.join(2.0)
            if second_thread.ident is not None:
                second_thread.join(2.0)
            _pipeline._search_hmm_candidates = original_search
            pipeline.F1 = 0.02

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertNotIsInstance(outcomes["first"], BaseException)
        self.assertIsInstance(outcomes["second"], ValueError)
        self.assertFalse(second_entered.is_set())
        self.assertEqual(_PIPELINE_LEASES, {})

    def test_external_block_membership_and_order_are_isolated(self):
        source = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [sequence.copy() for sequence in self.targets]
        )
        snapshot = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [sequence.copy() for sequence in source]
        )
        with SequenceBatch(source) as sequences:
            self.assertFalse(hasattr(sequences, "_native"))
            self.assertFalse(hasattr(sequences, "_lock"))
            self.assertEqual(sequences.alphabet, self.alphabet)
            for attribute, replacement in (
                ("_native", object()),
                ("_lock", threading.Lock()),
                ("alphabet", pyhmmer.easel.Alphabet.dna()),
            ):
                with self.subTest(attribute=attribute):
                    with self.assertRaises(AttributeError):
                        setattr(sequences, attribute, replacement)
            candidates = sequences.candidate_batch(self.pairs[:1])
        source[0].sequence[0] = (source[0].sequence[0] + 1) % self.alphabet.K
        source[0].name = b"mutated-after-upload"
        last = source.pop()
        source.insert(0, last)
        source.pop()
        source.append(last)

        actual = candidates.search(0, self.pipeline())
        expected = self.pipeline().search_hmm(self.pairs[0].hmm, snapshot)
        self.assert_exact_hits(expected, actual)

    def test_f1_one_uses_one_immutable_shared_all_target_row(self):
        pair = self.pairs[0]
        with SequenceBatch(self.targets) as sequences:
            all_candidates = sequences.candidate_batch([pair, pair, pair], F1=1.0)
        state = _candidate_state(all_candidates)
        self.assertEqual(all_candidates.F1, 1.0)
        self.assertEqual(state.all_rows, b"\x01\x01\x01")
        self.assertEqual(len(state.indices), 0)
        self.assertIsInstance(state.all_targets, bytes)
        self.assertEqual(len(state.all_targets), 4 * len(self.targets))
        self.assertEqual(all_candidates.candidate_count(2), len(self.targets))
        self.assert_exact_hits(
            self.pipeline(F1=1.0).search_hmm(pair.hmm, self.targets),
            all_candidates.search(2, self.pipeline(F1=1.0)),
        )

    def test_invalid_profile_uses_shared_fail_closed_row(self):
        pair = self.pairs[0]
        profile_state = _pair_state(pair)
        parameters = profile_state.optimized_profile.evalue_parameters.as_vector()
        original_mu = parameters[0]
        try:
            parameters[0] = math.nan
            with SequenceBatch(self.targets) as sequences:
                candidates = sequences.candidate_batch([self.pairs[1], pair])
        finally:
            parameters[0] = original_mu

        candidate_state = _candidate_state(candidates)
        self.assertEqual(candidate_state.all_rows, b"\x00\x01")
        self.assertEqual(candidates.candidate_count(1), len(self.targets))
        self.assert_exact_hits(
            self.pipeline().search_hmm(pair.hmm, self.targets),
            candidates.search(1, self.pipeline()),
        )

    def test_unprovable_cutoff_uses_shared_fail_closed_row(self):
        pair = self.pairs[0]
        profile_state = _pair_state(pair)
        parameters = profile_state.optimized_profile.evalue_parameters.as_vector()
        original_mu = parameters[0]
        original_lambda = parameters[1]
        smallx = 5e-9
        threshold = ((1.0 - math.exp(-smallx)) + smallx) / 2.0
        try:
            parameters[0] = struct.unpack("=f", struct.pack("=f", math.log(smallx)))[0]
            parameters[1] = 1.0
            with SequenceBatch(self.targets) as sequences:
                candidates = sequences.candidate_batch([pair], F1=threshold)
        finally:
            parameters[0] = original_mu
            parameters[1] = original_lambda

        candidate_state = _candidate_state(candidates)
        self.assertEqual(candidate_state.all_rows, b"\x01")
        self.assertEqual(len(candidate_state.indices), 0)
        self.assertEqual(candidates.candidate_count(0), len(self.targets))
        self.assert_exact_hits(
            self.pipeline(F1=threshold).search_hmm(pair.hmm, self.targets),
            candidates.search(0, self.pipeline(F1=threshold)),
        )

    def test_only_lockstep_pressed_pairs_are_accepted(self):
        pair = self.pairs[0]
        self.assertFalse(hasattr(pair, "optimized_profile"))
        self.assertFalse(hasattr(pair, "_optimized_profile"))
        exposed = pair.hmm
        original_name = pair.hmm.name
        exposed.name = f"{original_name}-mutated-copy"
        self.assertEqual(pair.hmm.name, original_name)
        raw_optimized = pair.hmm.to_profile(
            pyhmmer.plan7.Background(self.alphabet), L=400
        ).to_optimized()
        with SequenceBatch(self.targets) as sequences:
            with self.assertRaisesRegex(TypeError, "load_pressed_profiles"):
                sequences.candidate_batch([(pair.hmm, raw_optimized)])

    def test_private_abi_fingerprint_matches_runtime(self):
        self.assertEqual(
            _pipeline.PYHMMER_PRIVATE_ABI_SHA256,
            pyhmmer_abi_fingerprint(),
        )

    def test_mixed_pressed_profile_bytes_are_rejected(self):
        original = self.pairs[0].hmm
        perturbed = original.copy()
        row = perturbed.match_emissions[1]
        ranked = sorted(range(len(row)), key=row.__getitem__)
        source, destination = ranked[:2]
        epsilon = min(float(row[source]) / 2.0, 1e-4)
        row[source] -= epsilon
        row[destination] += epsilon
        perturbed.renormalize()
        self.assertEqual(perturbed.consensus, original.consensus)

        directory = Path(self._temporary.name)
        original_base = directory / "original-single"
        perturbed_base = directory / "perturbed-single"
        mixed_base = directory / "mixed-single"
        pyhmmer.hmmer.hmmpress([original], original_base)
        pyhmmer.hmmer.hmmpress([perturbed], perturbed_base)
        for suffix in ("h3m", "h3i"):
            shutil.copyfile(f"{original_base}.{suffix}", f"{mixed_base}.{suffix}")
        for suffix in ("h3f", "h3p"):
            shutil.copyfile(f"{perturbed_base}.{suffix}", f"{mixed_base}.{suffix}")

        with self.assertRaisesRegex(ValueError, "score mismatch at ordinal 0"):
            load_pressed_profiles(mixed_base)

    def test_non_amino_pressed_profiles_are_rejected(self):
        alphabet = pyhmmer.easel.Alphabet.dna()
        sequence = pyhmmer.easel.TextSequence(
            name=b"dna-model", sequence="ACGTACGTACGT"
        ).digitize(alphabet)
        hmm, _, _ = pyhmmer.plan7.Builder(alphabet).build(
            sequence, pyhmmer.plan7.Background(alphabet)
        )
        base = Path(self._temporary.name) / "dna-model"
        pyhmmer.hmmer.hmmpress([hmm], base)

        with self.assertRaisesRegex(ValueError, "only supports amino-acid"):
            load_pressed_profiles(base)


if __name__ == "__main__":
    unittest.main()
