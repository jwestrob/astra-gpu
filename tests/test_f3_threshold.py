import math
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import _native
except ImportError:
    _native = None


def float_from_bits(bits):
    return struct.unpack("=f", struct.pack("=I", bits))[0]


def float_bits(value):
    return struct.unpack("=I", struct.pack("=f", value))[0]


def ordered_key(bits):
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits ^ 0x80000000


@unittest.skipUnless(_native is not None, "native extension unavailable")
class ExactF3ThresholdTests(unittest.TestCase):
    def assert_compiled_matches(
        self, compiled, tau, lambda_, f3, score_bits
    ):
        score = float_from_bits(score_bits)
        self.assertFalse(math.isnan(score))
        threshold = float_from_bits(compiled["threshold_bits"])
        observed = score >= threshold
        expected = _native.f3_oracle_pass_bits_for_test(
            score_bits, tau, lambda_, f3
        )
        self.assertEqual(observed, expected, hex(score_bits))

    def test_boundary_certificate_and_adversarial_binary32_scores(self):
        parameters = (
            (-4.25, 0.69, 1.0e-5),
            (0.0, 1.0, 0.0),
            (-0.0, float_from_bits(1), 0.5),
            (float_from_bits(0x7F7FFFFF), 0.5, 1.0e-300),
            (-float_from_bits(0x7F7FFFFF), float_from_bits(0x7F7FFFFF), 0.5),
        )
        for tau, lambda_, f3 in parameters:
            with self.subTest(tau=tau, lambda_=lambda_, f3=f3):
                compiled = _native.compile_f3_threshold_for_test(
                    tau, lambda_, f3
                )
                self.assertTrue(compiled["supported"])
                self.assertTrue(compiled["threshold_pass"])
                self.assertTrue(compiled["positive_infinity_pass"])
                self.assertTrue(compiled["quiet_nan_oracle_pass"])
                self.assertTrue(compiled["nan_requires_fallback"])
                if compiled["predecessor_bits"] is not None:
                    self.assertFalse(compiled["predecessor_pass"])
                if compiled["successor_bits"] is not None:
                    self.assertTrue(compiled["successor_pass"])

                threshold_key = ordered_key(compiled["threshold_bits"])
                raw_scores = {
                    0xFF800000,
                    0xFF7FFFFF,
                    0x80000001,
                    0x80000000,
                    0x00000000,
                    0x00000001,
                    0x7F7FFFFF,
                    0x7F800000,
                }
                first_key = ordered_key(0xFF800000)
                last_key = ordered_key(0x7F800000)
                for delta in range(-4096, 4097):
                    key = threshold_key + delta
                    if first_key <= key <= last_key:
                        raw_scores.add(
                            (key ^ 0x80000000)
                            if key & 0x80000000
                            else (~key & 0xFFFFFFFF)
                        )
                for sign in (0, 0x80000000):
                    for exponent in range(256):
                        for mantissa in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF):
                            bits = sign | (exponent << 23) | mantissa
                            if not math.isnan(float_from_bits(bits)):
                                raw_scores.add(bits)
                for bits in raw_scores:
                    self.assert_compiled_matches(
                        compiled, tau, lambda_, f3, bits
                    )

    def test_exhaustive_binary16_projection_matches_exact_oracle(self):
        tau, lambda_, f3 = -4.25, 0.69, 1.0e-5
        compiled = _native.compile_f3_threshold_for_test(tau, lambda_, f3)
        threshold = float_from_bits(compiled["threshold_bits"])
        for half_bits in range(1 << 16):
            score = struct.unpack(">e", half_bits.to_bytes(2, "big"))[0]
            if math.isnan(score):
                continue
            score_bits = float_bits(score)
            expected = _native.f3_oracle_pass_bits_for_test(
                score_bits, tau, lambda_, f3
            )
            self.assertEqual(score >= threshold, expected, hex(half_bits))

    def test_f3_one_includes_negative_infinity_and_f3_zero_is_bounded(self):
        all_pass = _native.compile_f3_threshold_for_test(-4.25, 0.69, 1.0)
        self.assertTrue(all_pass["supported"])
        self.assertEqual(all_pass["threshold_bits"], 0xFF800000)
        self.assertIsNone(all_pass["predecessor_bits"])
        self.assertTrue(all_pass["negative_infinity_pass"])

        zero = _native.compile_f3_threshold_for_test(-4.25, 0.69, 0.0)
        self.assertTrue(zero["supported"])
        self.assertNotEqual(zero["threshold_bits"], 0x7F800000)
        self.assertFalse(zero["predecessor_pass"])
        self.assertTrue(zero["threshold_pass"])

    def test_degenerate_parameters_fall_back_and_nan_score_is_explicit(self):
        invalid = (
            (math.nan, 0.69, 1.0e-5),
            (math.inf, 0.69, 1.0e-5),
            (0.0, 0.0, 1.0e-5),
            (0.0, -0.69, 1.0e-5),
            (0.0, math.inf, 1.0e-5),
            (0.0, 0.69, math.nan),
            (0.0, 0.69, -1.0),
            (0.0, 0.69, math.nextafter(1.0, math.inf)),
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                compiled = _native.compile_f3_threshold_for_test(*parameters)
                self.assertFalse(compiled["supported"])
                self.assertNotEqual(compiled["reason"], 0)
                self.assertIsNone(compiled["threshold_bits"])

        for nan_bits in (0x7FC00000, 0xFFC00000):
            self.assertTrue(
                _native.f3_oracle_pass_bits_for_test(
                    nan_bits, -4.25, 0.69, 1.0e-5
                )
            )


if __name__ == "__main__":
    unittest.main()
