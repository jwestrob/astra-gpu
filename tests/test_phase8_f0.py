import math
import sys
import unittest
from array import array
from pathlib import Path

import pyhmmer


WORKTREE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKTREE if (WORKTREE / "refs" / "src").exists() else WORKTREE.parents[1]
sys.path.insert(0, str(WORKTREE / "python"))

try:
    from plan7_gpu import SequenceBatch, _native
    from plan7_gpu.adapter import _pack_profiles, _sequence_native
except ImportError:
    SequenceBatch = None
    _native = None
    _pack_profiles = None
    _sequence_native = None


HMM_20AA = SOURCE_ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "20aa.hmm"
HMM_M1 = SOURCE_ROOT / "refs" / "src" / "hmmer-3.4" / "testsuite" / "M1.hmm"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


def optimized(path):
    with pyhmmer.plan7.HMMFile(path) as stream:
        hmm = next(stream)
    background = pyhmmer.plan7.Background(hmm.alphabet)
    return hmm.to_profile(background, L=100).to_optimized()


def f1_parameters(profiles):
    values = [profile.evalue_parameters.as_vector() for profile in profiles]
    return array("f", (value[0] for value in values)), array(
        "f", (value[1] for value in values)
    )


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class Phase8F0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = [optimized(HMM_20AA), optimized(HMM_M1)]
        alphabet = cls.profiles[0].alphabet
        cls.targets = [
            pyhmmer.easel.TextSequence(
                name=f"f0-{index}".encode(), sequence=value
            ).digitize(alphabet)
            for index, value in enumerate(
                ("G", "ACDEX", "ACDEFGHIKLMNPQRSTVWY", "G" * 127, "X" * 31)
            )
        ]

    def evaluate(self, packed, classes, class_count):
        m_mu, m_lambda = f1_parameters(self.profiles)
        with SequenceBatch(self.targets) as batch:
            return _sequence_native(batch).evaluate_f0_many_raw(
                *packed, m_mu, m_lambda, 0.02, classes, class_count
            )

    def test_identity_matches_exact_and_reduced_is_a_superset(self):
        packed = _pack_profiles(self.profiles)
        alphabet_size = self.profiles[0].alphabet.Kp
        identity = bytearray(range(alphabet_size))
        exact = self.evaluate(packed, identity, alphabet_size)
        self.assertEqual(exact["false_reject_count"], 0)
        self.assertEqual(
            exact["coarse_candidate_count"], exact["exact_candidate_count"]
        )

        symbols = self.profiles[0].alphabet.symbols.encode()
        groups = tuple(map(set, (b"AG", b"P", b"ST", b"DENQ", b"HKR", b"ILMV", b"FWY", b"C")))
        aliases = {ord("B"): ord("D"), ord("J"): ord("I"), ord("Z"): ord("E"), ord("U"): ord("C"), ord("O"): ord("K")}
        reduced = bytearray(
            next(
                (
                    group_index
                    for group_index, group in enumerate(groups)
                    if aliases.get(symbol, symbol) in group
                ),
                0,
            )
            for symbol in symbols
        )
        coarse = self.evaluate(packed, reduced, len(groups))
        self.assertEqual(coarse["false_reject_count"], 0)
        self.assertGreaterEqual(
            coarse["coarse_candidate_count"], coarse["exact_candidate_count"]
        )
        self.assertEqual(
            coarse["logical_pair_count"], len(self.profiles) * len(self.targets)
        )
        self.assertEqual(
            coarse["certified_reject_count"] + coarse["coarse_candidate_count"],
            coarse["logical_pair_count"],
        )
        self.assertTrue(math.isfinite(coarse["coarse_kernel_milliseconds"]))

    def test_invalid_partition_and_non_hmmer_scores_fail_closed(self):
        packed = _pack_profiles(self.profiles)
        alphabet_size = self.profiles[0].alphabet.Kp
        with self.assertRaisesRegex(RuntimeError, "empty class"):
            self.evaluate(packed, bytearray(alphabet_size), 4)

        scores = bytearray(packed.scores)
        scores[0] = (-self.profiles[0].bias_b - 1) & 0xFF
        invalid = packed._replace(scores=scores)
        with self.assertRaisesRegex(RuntimeError, "bias bound"):
            self.evaluate(invalid, bytearray(range(alphabet_size)), alphabet_size)


if __name__ == "__main__":
    unittest.main()
