import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pyhmmer

from plan7_gpu import load_pressed_profiles
from plan7_gpu.adapter import (
    _iter_pressed_profile_chunks,
    _new_pressed_profile_pair,
    _pair_state,
)
from plan7_gpu.pressed_manifest import create_pressed_manifest


ROOT = Path(__file__).resolve().parents[1]
HMMER_ROOT = ROOT / "refs" / "src" / "hmmer-3.4"
if not HMMER_ROOT.is_dir():
    HMMER_ROOT = ROOT.parents[1] / "refs" / "src" / "hmmer-3.4"
HMM_SOURCES = (
    HMMER_ROOT / "tutorial" / "globins4.hmm",
    HMMER_ROOT / "tutorial" / "Pkinase.hmm",
    HMMER_ROOT / "tutorial" / "fn3.hmm",
    HMMER_ROOT / "testsuite" / "LuxC.hmm",
    HMMER_ROOT / "testsuite" / "M1.hmm",
)


def _hmm_bytes(hmm):
    output = io.BytesIO()
    hmm.write(output, binary=True)
    return output.getvalue()


class PressedProfileStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory(
            prefix="plan7-pressed-profile-stream-"
        )
        cls.base = Path(cls._temporary.name) / "profiles"
        hmms = []
        for source in HMM_SOURCES:
            with pyhmmer.plan7.HMMFile(source) as hmm_file:
                hmms.extend(hmm_file)
        pyhmmer.hmmer.hmmpress(hmms, cls.base)
        cls.manifest = Path(cls._temporary.name) / "profiles.manifest.json"
        create_pressed_manifest(cls.base, cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    def test_chunks_are_lazy_canonical_and_exact(self):
        eager = load_pressed_profiles(self.base, manifest=self.manifest)
        created = 0

        def observe(*args, **kwargs):
            nonlocal created
            created += 1
            return _new_pressed_profile_pair(*args, **kwargs)

        with mock.patch(
            "plan7_gpu.adapter._new_pressed_profile_pair", side_effect=observe
        ):
            streamed = _iter_pressed_profile_chunks(
                self.base, 2, manifest=self.manifest
            )
            first = next(streamed)
            self.assertEqual(created, 2)
            remaining = tuple(streamed)

        chunks = (first,) + remaining
        observed = tuple(pair for chunk in chunks for pair in chunk)
        self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 1])
        self.assertEqual(len(observed), len(eager))
        for expected, actual in zip(eager, observed, strict=True):
            self.assertEqual(actual.ordinal, expected.ordinal)
            self.assertEqual(actual.canonical_base, expected.canonical_base)
            self.assertEqual(actual.stat_token, expected.stat_token)
            self.assertEqual(actual.cutoffs, expected.cutoffs)
            self.assertEqual(_hmm_bytes(actual.hmm), _hmm_bytes(expected.hmm))
            self.assertEqual(
                _pair_state(actual).profile_fingerprint,
                _pair_state(expected).profile_fingerprint,
            )

    def test_chunk_size_fails_closed(self):
        for value in (True, 1.5, "2"):
            with self.assertRaises(TypeError):
                next(_iter_pressed_profile_chunks(self.base, value))
        for value in (0, -1):
            with self.assertRaises(ValueError):
                next(_iter_pressed_profile_chunks(self.base, value))

    def test_early_close_retires_the_pinned_pair_iterator(self):
        retired = False

        def pairs(*_args, **_kwargs):
            nonlocal retired
            try:
                yield object()
                yield object()
                yield object()
            finally:
                retired = True

        with mock.patch(
            "plan7_gpu.adapter._iter_pressed_profile_pairs",
            side_effect=pairs,
        ):
            chunks = _iter_pressed_profile_chunks(self.base, 2)
            self.assertEqual(len(next(chunks)), 2)
            self.assertFalse(retired)
            chunks.close()
        self.assertTrue(retired)


if __name__ == "__main__":
    unittest.main()
