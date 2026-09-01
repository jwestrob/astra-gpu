import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import _pipeline
except ImportError:
    _pipeline = None


class RequestLocalReleaseTests(unittest.TestCase):
    def test_avx_swap_restores_every_prior_state(self):
        if _pipeline is None:
            self.skipTest("plan7_gpu extension unavailable")
        try:
            for prior in (None, False, True):
                with self.subTest(prior=prior):
                    _pipeline._configure_avx512_tail_madvise_bound(prior)
                    observed = _pipeline._swap_avx512_tail_madvise_bound(True)
                    self.assertIs(observed, prior)
                    self.assertIs(
                        _pipeline._swap_avx512_tail_madvise_bound(observed),
                        True,
                    )
        finally:
            _pipeline._configure_avx512_tail_madvise_bound(None)

    def test_intrarow_swap_restores_prior_programmatic_threshold(self):
        if _pipeline is None:
            self.skipTest("plan7_gpu extension unavailable")
        try:
            _pipeline._configure_intrarow_page_release_bound(4096)
            self.assertEqual(
                _pipeline._intrarow_page_release_configuration_bound(), 4096
            )
            previous = _pipeline._swap_intrarow_page_release_bound(16 << 20)
            self.assertEqual(previous, 4096)
            self.assertEqual(
                _pipeline._intrarow_page_release_configuration_bound(),
                16 << 20,
            )
            self.assertEqual(
                _pipeline._swap_intrarow_page_release_bound(previous),
                16 << 20,
            )
            self.assertEqual(
                _pipeline._intrarow_page_release_configuration_bound(), 4096
            )
        finally:
            _pipeline._configure_intrarow_page_release_bound(0)

    def test_filter_tail_simd_swap_restores_every_prior_state(self):
        if _pipeline is None:
            self.skipTest("plan7_gpu extension unavailable")
        try:
            for prior in (None, False, True):
                with self.subTest(prior=prior):
                    _pipeline._configure_filter_tail_simd_bound(prior)
                    observed = _pipeline._swap_filter_tail_simd_bound(True)
                    self.assertIs(observed, prior)
                    self.assertIs(
                        _pipeline._swap_filter_tail_simd_bound(observed),
                        True,
                    )
        finally:
            _pipeline._configure_filter_tail_simd_bound(None)


if __name__ == "__main__":
    unittest.main()
