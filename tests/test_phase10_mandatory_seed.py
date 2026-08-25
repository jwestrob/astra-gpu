import itertools
import importlib.util
import random
import sys
import unittest
from pathlib import Path


WORKTREE = Path(__file__).resolve().parents[1]
MODULE_PATH = WORKTREE / "python" / "plan7_gpu" / "_mandatory_seed.py"
SPEC = importlib.util.spec_from_file_location("_phase10_mandatory_seed", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

from _phase10_mandatory_seed import (  # noqa: E402
    SSVSeedParameters,
    compile_required_gain,
    enumerate_mandatory_seeds,
    equal_integer_quotas,
    maximum_diagonal_gain,
    maximum_bounded_diagonal_gain,
    seed_false_rejects,
    window_seed_threshold,
)


class MandatorySeedTests(unittest.TestCase):
    def test_equal_quotas_close_the_integer_pigeonhole_bound(self):
        for gain in range(1, 128):
            for blocks in range(1, 12):
                quotas = equal_integer_quotas(gain, blocks)
                self.assertTrue(all(value >= 1 for value in quotas))
                self.assertEqual(sum(value - 1 for value in quotas), gain - 1)

    def test_required_gain_replays_ssv_status_and_exact_cutoff(self):
        parameters = SSVSeedParameters(
            tbm=7,
            tec=3,
            base=190,
            bias=12,
            scale=3.0,
            cutoff_mode=1,
            cutoff_bit_score=0.0,
        )

        def cutoff(_status, numerator, _length, _scale, _mode, _cutoff):
            return 0 if numerator >= -20 else 1

        # F1 crosses at gain = numerator + 2*tjb + tbm + tec.
        self.assertEqual(
            compile_required_gain(
                parameters,
                100,
                9,
                cutoff,
                status_ok=0,
                cpu_required=0,
            ),
            8,
        )

        unsafe = SSVSeedParameters(100, 3, 190, 24, 3.0, 1, 0.0)
        self.assertIsNone(
            compile_required_gain(
                unsafe, 100, 9, cutoff, status_ok=0, cpu_required=0
            )
        )

    def test_seed_theorem_is_exhaustive_on_small_integer_models(self):
        rng = random.Random(0x5100)
        for model_length in range(1, 8):
            for alphabet_size in (2, 3):
                for _ in range(20):
                    gains = tuple(
                        tuple(rng.randrange(-3, 5) for _ in range(alphabet_size))
                        for _ in range(model_length)
                    )
                    required_gain = rng.randrange(1, 10)
                    plan = enumerate_mandatory_seeds(
                        gains,
                        required_gain,
                        block_count=min(4, model_length),
                        alphabet_size=alphabet_size,
                        association_limit=1_000_000,
                        node_limit=10_000_000,
                    )
                    self.assertFalse(plan.truncated)
                    sequences = (
                        sequence
                        for length in range(1, 7)
                        for sequence in itertools.product(
                            range(alphabet_size), repeat=length
                        )
                    )
                    required, missed = seed_false_rejects(gains, sequences, plan)
                    self.assertEqual(missed, 0, (gains, required_gain, required))

    def test_truncated_plan_fails_open(self):
        gains = tuple((3, 3, 3, 3) for _ in range(12))
        plan = enumerate_mandatory_seeds(
            gains,
            1,
            alphabet_size=4,
            association_limit=2,
            node_limit=2,
        )
        self.assertTrue(plan.truncated)
        self.assertEqual(maximum_diagonal_gain(gains, (0, 0, 0)), 9)
        required, missed = seed_false_rejects(gains, [(0, 0, 0)], plan)
        self.assertEqual((required, missed), (1, 0))

    def test_bounded_word_threshold_is_mandatory(self):
        rng = random.Random(0x5101)
        for model_length in range(1, 9):
            for target_length in range(1, 9):
                for word_length in range(1, 5):
                    for _ in range(12):
                        gains = tuple(
                            tuple(rng.randrange(-3, 5) for _ in range(3))
                            for _ in range(model_length)
                        )
                        sequence = tuple(
                            rng.randrange(3) for _ in range(target_length)
                        )
                        required_gain = rng.randrange(1, 12)
                        threshold = window_seed_threshold(
                            required_gain,
                            model_length,
                            target_length,
                            word_length,
                        )
                        full = maximum_diagonal_gain(gains, sequence)
                        bounded = maximum_bounded_diagonal_gain(
                            gains, sequence, word_length
                        )
                        if full >= required_gain:
                            self.assertGreaterEqual(
                                bounded,
                                threshold,
                                (gains, sequence, required_gain, word_length),
                            )


if __name__ == "__main__":
    unittest.main()
