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
class ExactF2ThresholdTests(unittest.TestCase):
    def test_boundary_and_finite_domain_projection_match_gumbel_oracle(self):
        parameters = (
            (13.0, 0.7, 1.0e-3),
            (0.0, 1.0, 0.0),
            (-0.0, float_from_bits(1), 0.5),
            (float_from_bits(0x7F7FFFFF), 0.5, 1.0e-300),
            (-float_from_bits(0x7F7FFFFF), float_from_bits(0x7F7FFFFF), 0.5),
        )
        for mu, lambda_, f2 in parameters:
            with self.subTest(mu=mu, lambda_=lambda_, f2=f2):
                compiled = _native.compile_f2_threshold_for_test(
                    mu, lambda_, f2
                )
                self.assertTrue(compiled["supported"])
                self.assertTrue(compiled["threshold_pass"])
                self.assertTrue(compiled["positive_infinity_pass"])
                if compiled["predecessor_bits"] is not None:
                    self.assertFalse(compiled["predecessor_pass"])

                threshold = float_from_bits(compiled["threshold_bits"])
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
                for delta in range(-4096, 4097):
                    key = threshold_key + delta
                    if ordered_key(0xFF800000) <= key <= ordered_key(0x7F800000):
                        raw_scores.add(
                            (key ^ 0x80000000)
                            if key & 0x80000000
                            else (~key & 0xFFFFFFFF)
                        )
                for bits in raw_scores:
                    score = float_from_bits(bits)
                    self.assertEqual(
                        score >= threshold,
                        _native.f2_oracle_pass_bits_for_test(
                            bits, mu, lambda_, f2
                        ),
                        hex(bits),
                    )

        compiled = _native.compile_f2_threshold_for_test(13.0, 0.7, 1.0e-3)
        threshold = float_from_bits(compiled["threshold_bits"])
        for half_bits in range(1 << 16):
            score = struct.unpack(">e", half_bits.to_bytes(2, "big"))[0]
            if math.isnan(score):
                continue
            bits = float_bits(score)
            self.assertEqual(
                score >= threshold,
                _native.f2_oracle_pass_bits_for_test(
                    bits, 13.0, 0.7, 1.0e-3
                ),
                hex(half_bits),
            )

    def test_invalid_parameters_fall_back_and_nan_remains_explicit(self):
        invalid = (
            (math.nan, 0.7, 1.0e-3),
            (math.inf, 0.7, 1.0e-3),
            (0.0, 0.0, 1.0e-3),
            (0.0, -0.7, 1.0e-3),
            (0.0, math.inf, 1.0e-3),
            (0.0, 0.7, math.nan),
            (0.0, 0.7, -1.0),
            (0.0, 0.7, math.nextafter(1.0, math.inf)),
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                compiled = _native.compile_f2_threshold_for_test(*parameters)
                self.assertFalse(compiled["supported"])
                self.assertNotEqual(compiled["reason"], 0)
                self.assertIsNone(compiled["threshold_bits"])

        self.assertTrue(
            _native.f2_oracle_pass_bits_for_test(
                0x7FC00000, 13.0, 0.7, 1.0e-3
            )
        )


if __name__ == "__main__":
    unittest.main()
