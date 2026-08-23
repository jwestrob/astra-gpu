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

    def setUp(self):
        self.identity_sources = []

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

    def candidates(self, pairs=None, *, identity_mode="owned"):
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
        residue_payload = residue_offsets.tobytes()
        background_payload = _background_fingerprint(background)
        ownership_options = {}
        if identity_mode == "owned":
            residue_input = memoryview(residue_payload).cast("Q")
            background_input = background_payload
        elif identity_mode == "shared":
            residue_input = memoryview(residue_payload).cast("Q")
            background_input = memoryview(background_payload)
            self.identity_sources.extend((residue_payload, background_payload))
            ownership_options = {
                "_residue_offsets_shared": True,
                "_background_fingerprint_shared": True,
            }
        elif identity_mode == "shared_slices":
            residue_owner = bytes(8) + residue_payload + bytes(8)
            background_owner = b"x" + background_payload + b"y"
            residue_input = memoryview(residue_owner)[8:-8].cast("Q")
            background_input = memoryview(background_owner)[1:-1]
            self.identity_sources.extend(
                (residue_payload, residue_owner, background_owner)
            )
            ownership_options = {
                "_residue_offsets_shared": True,
                "_background_fingerprint_shared": True,
            }
        else:
            raise ValueError("unknown identity ownership test mode")
        sealed = _pipeline._seal_postfilter_batch_bound(
            tuple(state.hmm for state in states),
            tuple(state.optimized_profile for state in states),
            self.targets,
            records,
            row_offsets,
            residue_input,
            0.02,
            background_input,
            **ownership_options,
        )
        return _new_candidate_batch(
            pairs,
            self.targets,
            residue_payload,
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

    def test_sealed_resident_memory_counts_backings_not_subviews(self):
        candidates = self.candidates()
        memory = candidates.resident_memory
        sealed = memory["sealed"]
        self.assertIs(type(candidates.resident_bytes), int)
        self.assertGreater(candidates.resident_bytes, 0)
        self.assertEqual(memory["owned_device_bytes"], 0)
        self.assertEqual(memory["state_buffers"]["postfilter_records_bytes"], 0)
        self.assertEqual(sealed["journal_allocation_bytes"], 0)
        charged_sealed_fields = (
            "journal_allocation_bytes",
            "postfilter_records_bytes",
            "postfilter_offsets_bytes",
            "forward_records_bytes",
            "forward_offsets_bytes",
            "special_offsets_bytes",
            "specials_bytes",
            "row_markers_bytes",
            "journal_sentinel_bytes",
            "owned_residue_offsets_bytes",
            "owned_background_fingerprint_bytes",
        )
        self.assertEqual(
            sealed["owned_host_bytes"],
            sum(sealed[field] for field in charged_sealed_fields),
        )
        self.assertEqual(
            memory["resident_bytes"],
            memory["state_owned_host_bytes"] + sealed["owned_host_bytes"],
        )
        self.assertGreater(sealed["journal_subview_bytes"], 0)
        self.assertGreater(sealed["owned_residue_offsets_bytes"], 0)
        self.assertGreater(sealed["owned_background_fingerprint_bytes"], 0)
        self.assertEqual(sealed["excluded_shared_identity_bytes"], 0)
        original = candidates.resident_bytes
        sealed["postfilter_records_bytes"] = -1
        memory["state_buffers"]["offsets_bytes"] = -1
        self.assertEqual(candidates.resident_bytes, original)

    def test_identity_ownership_distinguishes_full_shared_views_and_slices(self):
        shared = self.candidates(identity_mode="shared").resident_memory["sealed"]
        copied = self.candidates(
            identity_mode="shared_slices"
        ).resident_memory["sealed"]

        self.assertEqual(shared["owned_residue_offsets_bytes"], 0)
        self.assertEqual(shared["owned_background_fingerprint_bytes"], 0)
        self.assertGreater(shared["excluded_residue_offsets_bytes"], 0)
        self.assertGreater(shared["excluded_background_fingerprint_bytes"], 0)
        self.assertEqual(
            shared["excluded_shared_identity_bytes"],
            shared["excluded_residue_offsets_bytes"]
            + shared["excluded_background_fingerprint_bytes"],
        )
        self.assertGreater(copied["owned_residue_offsets_bytes"], 0)
        self.assertGreater(copied["owned_background_fingerprint_bytes"], 0)
        self.assertEqual(copied["excluded_shared_identity_bytes"], 0)

    def test_sparse_v3_dispatch_is_batch_scoped_and_default_stays_dense(self):
        candidates = self.candidates()
        dense_marker = object()
        sparse_marker = object()
        with mock.patch.object(
            _pipeline,
            "_sealed_sparse_journal_v3_enabled_bound",
            return_value=False,
        ), mock.patch.object(
            _pipeline,
            "_search_hmm_sealed_postfilter_bound",
            return_value=dense_marker,
        ) as dense_search, mock.patch.object(
            _pipeline,
            "_search_hmm_sealed_sparse_journal_v3_bound",
            return_value=sparse_marker,
        ) as sparse_search:
            self.assertIs(candidates.search(0, self.pipeline()), dense_marker)
            dense_search.assert_called_once()
            sparse_search.assert_not_called()

        with mock.patch.object(
            _pipeline,
            "_sealed_sparse_journal_v3_enabled_bound",
            return_value=True,
        ), mock.patch.object(
            _pipeline,
            "_search_hmm_sealed_sparse_journal_v3_bound",
            return_value=sparse_marker,
        ) as sparse_search:
            self.assertIs(candidates.search(0, self.pipeline()), sparse_marker)
            sparse_search.assert_called_once()

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
