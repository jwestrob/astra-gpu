import io
import math
import struct
import sys
import tempfile
import threading
import time
import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pyhmmer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
try:
    from plan7_gpu import _native, _pipeline
    from plan7_gpu.adapter import (
        _background_fingerprint,
        _new_candidate_batch,
        _pair_state,
        load_pressed_profiles,
    )
except ImportError:
    _native = None
    _pipeline = None
    _background_fingerprint = None
    _new_candidate_batch = None
    _pair_state = None
    load_pressed_profiles = None


HMM_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
HMM_FN3 = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "fn3.hmm"
FASTA_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins45.fa"


@unittest.skipUnless(
    _pipeline is not None and hasattr(_pipeline, "_seal_postfilter_batch_bound"),
    "sealed candidate extension unavailable",
)
class SealedCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for fixture in (HMM_GLOBINS, HMM_FN3, FASTA_GLOBINS):
            if not fixture.is_file():
                raise unittest.SkipTest(f"missing HMMER fixture: {fixture}")
        if not _native.bias_host_environment_attested():
            raise unittest.SkipTest("host floating-point environment is unattested")

        cls.temporary = tempfile.TemporaryDirectory(prefix="plan7-sealed-test-")
        source_hmms = []
        for path in (HMM_GLOBINS, HMM_FN3):
            with pyhmmer.plan7.HMMFile(path) as hmm_file:
                source_hmms.append(next(hmm_file))
        pressed_base = Path(cls.temporary.name) / "models"
        pyhmmer.hmmer.hmmpress(source_hmms, pressed_base)
        cls.pairs = load_pressed_profiles(pressed_base)
        cls.alphabet = cls.pairs[0].hmm.alphabet
        with pyhmmer.easel.SequenceFile(
            FASTA_GLOBINS, digital=True, alphabet=cls.alphabet
        ) as sequence_file:
            targets = list(sequence_file)[:12]
        cls.targets = pyhmmer.easel.DigitalSequenceBlock(cls.alphabet, targets)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def pipeline(**options):
        defaults = {"E": 10.0, "domE": 10.0, "incE": 10.0, "incdomE": 10.0}
        defaults.update(options)
        return pyhmmer.plan7.Pipeline(SealedCandidateTests.alphabet, **defaults)

    @staticmethod
    def table_bytes(hits, format):
        output = io.BytesIO()
        hits.write(output, format=format, header=True)
        return output.getvalue()

    def assert_exact_hits(self, expected, actual):
        self.assertEqual(
            self.table_bytes(expected, "targets"),
            self.table_bytes(actual, "targets"),
        )
        self.assertEqual(
            self.table_bytes(expected, "domains"),
            self.table_bytes(actual, "domains"),
        )
        self.assertEqual(
            struct.pack(
                "=QQQQdd",
                expected.searched_models,
                expected.searched_nodes,
                expected.searched_sequences,
                expected.searched_residues,
                expected.Z,
                expected.domZ,
            ),
            struct.pack(
                "=QQQQdd",
                actual.searched_models,
                actual.searched_nodes,
                actual.searched_sequences,
                actual.searched_residues,
                actual.Z,
                actual.domZ,
            ),
        )

    def candidates(self, pairs=None):
        if pairs is None:
            pairs = self.pairs
        pairs = tuple(pairs)
        states = tuple(_pair_state(pair) for pair in pairs)
        target_count = len(self.targets)
        one_row = b"".join(
            struct.pack("=IfhBBf", index, math.nan, 0, 255, 0, math.nan)
            for index in range(target_count)
        )
        records = one_row * len(pairs)
        row_offsets = array("Q", (row * target_count for row in range(len(pairs) + 1)))
        residue_offsets = array("Q", [0])
        for target in self.targets:
            residue_offsets.append(residue_offsets[-1] + len(target))
        background = pyhmmer.plan7.Background(self.alphabet)
        sealed = _pipeline._seal_postfilter_batch_bound(
            tuple(state.hmm for state in states),
            tuple(state.optimized_profile for state in states),
            self.targets,
            records,
            row_offsets,
            residue_offsets,
            0.02,
            _background_fingerprint(background),
        )
        return _new_candidate_batch(
            pairs,
            self.targets,
            residue_offsets.tobytes(),
            array("I"),
            row_offsets,
            bytes(len(pairs)),
            array("I"),
            0.02,
            postfilter_records=records,
            sealed_postfilter=sealed,
        )

    def test_candidate_reuse_has_exact_independent_queries(self):
        candidates = self.candidates()
        for row, pair in enumerate(self.pairs):
            expected = self.pipeline().search_hmm(pair.hmm, self.targets)
            first = candidates.search(row, self.pipeline())
            second = candidates.search(row, self.pipeline())
            self.assert_exact_hits(expected, first)
            self.assert_exact_hits(expected, second)
            self.assertIsNot(first.query, second.query)
        self.assertFalse(hasattr(candidates, "_sealed_postfilter"))
        with self.assertRaises((AttributeError, TypeError)):
            candidates.records = b""

    def test_pipeline_lease_and_duplicate_pair_lock_remain_exclusive(self):
        cases = (
            (self.candidates(), self.pipeline(), "pipeline lease"),
            (self.candidates((self.pairs[0], self.pairs[0])), None, "pair lock"),
        )
        original = _pipeline._search_hmm_sealed_postfilter_bound
        for candidates, shared_pipeline, label in cases:
            active = 0
            maximum = 0
            execution_order = []
            counter_lock = threading.Lock()

            def observed(*arguments):
                nonlocal active, maximum
                with counter_lock:
                    active += 1
                    maximum = max(maximum, active)
                    execution_order.append(arguments[1])
                try:
                    time.sleep(0.02)
                    return original(*arguments)
                finally:
                    with counter_lock:
                        active -= 1

            pipelines = (
                (shared_pipeline, shared_pipeline)
                if shared_pipeline is not None
                else (self.pipeline(), self.pipeline())
            )
            with mock.patch.object(
                _pipeline, "_search_hmm_sealed_postfilter_bound", new=observed
            ):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(candidates.search, row, pipelines[row])
                        for row in range(2)
                    ]
                    hits = [future.result(timeout=10) for future in futures]
            with self.subTest(label=label):
                self.assertEqual(maximum, 1)
                if shared_pipeline is not None:
                    reference_pipeline = self.pipeline()
                    expected_by_row = {}
                    for row in execution_order:
                        expected_by_row[row] = reference_pipeline.search_hmm(
                            candidates_pair(candidates, row).hmm, self.targets
                        )
                    for row, actual in enumerate(hits):
                        self.assert_exact_hits(expected_by_row[row], actual)
                else:
                    for row, actual in enumerate(hits):
                        expected = self.pipeline().search_hmm(
                            candidates_pair(candidates, row).hmm, self.targets
                        )
                        self.assert_exact_hits(expected, actual)

    def test_live_errors_release_locks_and_leave_pipeline_reusable(self):
        candidates = self.candidates()
        attestation_pipeline = self.pipeline()
        with (
            mock.patch.object(
                _native, "bias_host_environment_attested", return_value=False
            ),
            self.assertRaisesRegex(RuntimeError, "attested host"),
        ):
            candidates.search(0, attestation_pipeline)

        pipeline = self.pipeline(F1=0.5)
        with self.assertRaisesRegex(ValueError, "does not match candidate F1"):
            candidates.search(0, pipeline)
        expected = self.pipeline(F1=0.5).search_hmm(self.pairs[0].hmm, self.targets)
        actual = pipeline.search_hmm(self.pairs[0].hmm, self.targets)
        self.assert_exact_hits(expected, actual)


def candidates_pair(candidates, row):
    # Tests deliberately avoid exposing this through the public CandidateBatch.
    from plan7_gpu.adapter import _candidate_state

    return _candidate_state(candidates).pairs[row]


if __name__ == "__main__":
    unittest.main()
