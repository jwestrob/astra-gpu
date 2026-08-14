import io
import random
import struct
import sys
import tempfile
import threading
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import pyhmmer
from pyhmmer.errors import MissingCutoffs


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
try:
    from plan7_gpu import _native
    from plan7_gpu.adapter import CandidateBatch, SequenceBatch, load_pressed_profiles
    from plan7_gpu.astra_search import hmmsearch
except ImportError:
    _native = None
    CandidateBatch = None
    SequenceBatch = None
    load_pressed_profiles = None
    hmmsearch = None


HMM_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
HMM_FN3 = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "fn3.hmm"
FASTA_GLOBINS = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins45.fa"


def bridge_available():
    if _native is None or hmmsearch is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


@unittest.skipUnless(bridge_available(), "Astra bridge CUDA backend unavailable")
class AstraSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for fixture in (HMM_GLOBINS, HMM_FN3, FASTA_GLOBINS):
            if not fixture.is_file():
                raise unittest.SkipTest(f"missing HMMER fixture: {fixture}")

        cls.temporary = tempfile.TemporaryDirectory(prefix="plan7-astra-search-test-")
        cls.alphabet = pyhmmer.easel.Alphabet.amino()
        cls.background = pyhmmer.plan7.Background(cls.alphabet)

        real_hmms = []
        for path in (HMM_GLOBINS, HMM_FN3):
            with pyhmmer.plan7.HMMFile(path) as hmm_file:
                real_hmms.append(next(hmm_file))

        pattern = "ACDEFGHIKLMNPQRSTVWY" * 3
        no_match = "AAAAAAAAAAVVVVVVVVVVLLLLLLLLLL"
        builder = pyhmmer.plan7.Builder(cls.alphabet, seed=42)
        synthetic_hmms = []
        for name, sequence in (
            (b"synthetic-pattern", pattern),
            (b"synthetic-no-match", no_match),
        ):
            query = pyhmmer.easel.TextSequence(name=name, sequence=sequence).digitize(
                cls.alphabet
            )
            hmm, _, _ = builder.build(query, cls.background)
            hmm.cutoffs.gathering = (0.0, 0.0)
            synthetic_hmms.append(hmm)

        # Extra distinct rows make it possible to prove that each threaded
        # worker recycles one Pipeline instead of creating one per query.
        extra_hmms = []
        for index, source in enumerate(synthetic_hmms):
            for copy_index in range(2):
                hmm = source.copy()
                hmm.name = f"thread-{index}-{copy_index}".encode()
                hmm.accession = None
                extra_hmms.append(hmm)

        cls.source_hmms = tuple(real_hmms + synthetic_hmms + extra_hmms)
        cls.pressed_base = Path(cls.temporary.name) / "models"
        pyhmmer.hmmer.hmmpress(cls.source_hmms, cls.pressed_base)
        cls.pairs = load_pressed_profiles(cls.pressed_base)

        with pyhmmer.easel.SequenceFile(
            FASTA_GLOBINS, digital=True, alphabet=cls.alphabet
        ) as sequence_file:
            cls.real_targets = sequence_file.read_block()

        rng = random.Random(1)
        letters = "ACDEFGHIKLMNPQRSTVWY"
        synthetic_targets = [
            pyhmmer.easel.TextSequence(
                name=f"random-{index}".encode(),
                sequence="".join(rng.choice(letters) for _ in range(60)),
            ).digitize(cls.alphabet)
            for index in range(30)
        ]
        synthetic_targets.append(
            pyhmmer.easel.TextSequence(
                name=b"pattern-match", sequence=pattern
            ).digitize(cls.alphabet)
        )
        cls.synthetic_targets = pyhmmer.easel.DigitalSequenceBlock(
            cls.alphabet, synthetic_targets
        )
        cls.real_batch = SequenceBatch(cls.real_targets)
        cls.synthetic_batch = SequenceBatch(cls.synthetic_targets)

    @classmethod
    def tearDownClass(cls):
        cls.synthetic_batch.close()
        cls.real_batch.close()
        cls.temporary.cleanup()

    @staticmethod
    def _table_bytes(hits, table_format):
        output = io.BytesIO()
        hits.write(output, format=table_format, header=True)
        return output.getvalue()

    @staticmethod
    def _accounting_bytes(hits):
        return struct.pack(
            "=QQQQdd",
            hits.searched_models,
            hits.searched_nodes,
            hits.searched_sequences,
            hits.searched_residues,
            hits.Z,
            hits.domZ,
        )

    def assert_exact_hits(self, expected, actual):
        self.assertEqual(actual.query.name, expected.query.name)
        self.assertEqual(actual.query.accession, expected.query.accession)
        self.assertEqual(actual.query.M, expected.query.M)
        self.assertEqual(len(actual), len(expected))
        self.assertEqual(len(actual.reported), len(expected.reported))
        self.assertEqual(len(actual.included), len(expected.included))
        self.assertEqual(
            self._table_bytes(actual, "targets"),
            self._table_bytes(expected, "targets"),
        )
        self.assertEqual(
            self._table_bytes(actual, "domains"),
            self._table_bytes(expected, "domains"),
        )
        self.assertEqual(
            self._accounting_bytes(actual), self._accounting_bytes(expected)
        )

    def reference(self, pairs, targets, *, cpus=1, **options):
        queries = [pair.hmm for pair in pairs]
        return list(pyhmmer.hmmsearch(queries, targets, cpus=cpus, **options))

    def assert_search_matches(self, pairs, batch, targets, *, cpus=1, **options):
        expected = self.reference(pairs, targets, cpus=cpus, **options)
        actual = list(hmmsearch(pairs, batch, cpus=cpus, **options))
        self.assertEqual(len(actual), len(expected))
        for expected_hits, actual_hits in zip(expected, actual, strict=True):
            self.assert_exact_hits(expected_hits, actual_hits)
        return actual

    def test_serial_sequence_batch_forwards_astra_pipeline_kwargs(self):
        pairs = self.pairs[2:4]
        options = {
            "bit_cutoffs": "gathering",
            "Z": 123.0,
            "domZ": 17.0,
            "F1": 0.02,
        }
        actual = self.assert_search_matches(
            pairs,
            self.synthetic_batch,
            self.synthetic_targets,
            cpus=1,
            **options,
        )
        for hits in actual:
            self.assertEqual(hits.Z, 123.0)
            self.assertEqual(hits.domZ, 17.0)
            self.assertEqual(hits.bit_cutoffs, "gathering")

    def test_precomputed_none_sparse_and_all_rows_match_pyhmmer(self):
        cases = (
            ("none", self.pairs[3:4], 0.0),
            ("sparse", self.pairs[2:3], 0.02),
            ("all", self.pairs[2:4], 1.0),
        )
        for label, pairs, f1 in cases:
            with self.subTest(label=label):
                candidates = self.synthetic_batch.candidate_batch(pairs, F1=f1)
                counts = [
                    candidates.candidate_count(row) for row in range(len(candidates))
                ]
                if label == "none":
                    self.assertEqual(counts, [0])
                elif label == "sparse":
                    self.assertTrue(
                        all(0 < count < len(self.synthetic_batch) for count in counts)
                    )
                else:
                    self.assertEqual(counts, [len(self.synthetic_batch)] * len(pairs))

                # Omit F1 from the bridge call: a precomputed batch carries
                # the exact normalized value that must configure each worker.
                expected = self.reference(pairs, self.synthetic_targets, cpus=1, F1=f1)
                actual = list(hmmsearch(pairs, candidates, cpus=1))
                for expected_hits, actual_hits in zip(expected, actual, strict=True):
                    self.assert_exact_hits(expected_hits, actual_hits)

                if label == "none":
                    hits = actual[0]
                    self.assertEqual(
                        hits.searched_sequences, len(self.synthetic_targets)
                    )
                    self.assertEqual(
                        hits.searched_residues,
                        self.synthetic_targets.total_length(),
                    )
                    self.assertEqual(hits.Z, float(len(self.synthetic_targets)))

    def test_real_tutorial_profiles_match_serial_and_threaded(self):
        pairs = self.pairs[:2]
        options = {"E": 10.0, "domE": 10.0, "incE": 10.0, "incdomE": 10.0}
        for cpus in (1, 2):
            with self.subTest(cpus=cpus):
                self.assert_search_matches(
                    pairs,
                    self.real_batch,
                    self.real_targets,
                    cpus=cpus,
                    **options,
                )

    def test_threaded_order_and_worker_local_pipeline_reuse(self):
        pairs = self.pairs
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        expected = self.reference(pairs, self.synthetic_targets, cpus=2, F1=1.0)

        original_search = CandidateBatch.search
        row_two_started = threading.Event()
        row_three_finished = threading.Event()
        row_four_started = threading.Event()
        release_row_zero = threading.Event()
        row_zero_released = threading.Event()
        lock = threading.Lock()
        active_pipelines = set()
        rows_started_while_zero_blocked = set()
        ownership_errors = []
        pipelines_by_thread = defaultdict(set)
        threads_by_pipeline = defaultdict(set)
        pipeline_calls = Counter()
        completion_order = []

        def observed_search(candidate_batch, row, pipeline):
            thread_id = threading.get_ident()
            pipeline_id = id(pipeline)
            with lock:
                if not release_row_zero.is_set():
                    rows_started_while_zero_blocked.add(row)
                if type(pipeline) is not pyhmmer.plan7.Pipeline:
                    ownership_errors.append("pipeline is not exact base Pipeline")
                if pipeline_id in active_pipelines:
                    ownership_errors.append("pipeline used concurrently")
                active_pipelines.add(pipeline_id)
                pipelines_by_thread[thread_id].add(pipeline_id)
                threads_by_pipeline[pipeline_id].add(thread_id)
                pipeline_calls[pipeline_id] += 1
            try:
                # Keep row 0 blocked while the other worker drains the rest of
                # the four-row reorder window. Row 4 must remain unsubmitted.
                if row == 0:
                    if not release_row_zero.wait(10):
                        raise RuntimeError("test did not release row 0")
                    row_zero_released.set()
                elif row == 2:
                    row_two_started.set()
                elif row == 4:
                    row_four_started.set()
                return original_search(candidate_batch, row, pipeline)
            finally:
                with lock:
                    completion_order.append(row)
                    active_pipelines.remove(pipeline_id)
                if row == 3:
                    row_three_finished.set()

        actual = []
        consumer_errors = []

        def consume():
            try:
                actual.extend(hmmsearch(pairs, candidates, cpus=2))
            except BaseException as error:
                consumer_errors.append(error)

        with mock.patch.object(CandidateBatch, "search", new=observed_search):
            consumer = threading.Thread(target=consume, name="astra-test-consumer")
            consumer.start()
            try:
                self.assertTrue(row_two_started.wait(10))
                self.assertTrue(row_three_finished.wait(10))
                self.assertFalse(
                    row_four_started.wait(0.5),
                    "row beyond the reorder window started while row 0 was held",
                )
                with lock:
                    self.assertEqual(rows_started_while_zero_blocked, {0, 1, 2, 3})
            finally:
                release_row_zero.set()
                consumer.join(10)

        self.assertTrue(row_two_started.is_set())
        self.assertTrue(row_zero_released.is_set())
        self.assertTrue(row_four_started.is_set())
        self.assertFalse(consumer.is_alive())
        self.assertFalse(consumer_errors)
        self.assertEqual(completion_order[0], 1)
        self.assertEqual(
            [hits.query.name for hits in actual],
            [pair.hmm.name for pair in pairs],
        )
        self.assertFalse(ownership_errors)
        self.assertEqual(len(pipelines_by_thread), 2)
        self.assertEqual(len(threads_by_pipeline), 2)
        self.assertTrue(all(len(ids) == 1 for ids in pipelines_by_thread.values()))
        self.assertTrue(all(len(ids) == 1 for ids in threads_by_pipeline.values()))
        self.assertTrue(any(count > 1 for count in pipeline_calls.values()))
        for expected_hits, actual_hits in zip(expected, actual, strict=True):
            self.assert_exact_hits(expected_hits, actual_hits)

    def test_threaded_yields_prior_row_then_propagates_pipeline_error(self):
        pairs = (self.pairs[2], self.pairs[0])
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        iterator = hmmsearch(
            pairs,
            candidates,
            cpus=2,
            bit_cutoffs="gathering",
        )

        expected_first = self.reference(
            pairs[:1],
            self.synthetic_targets,
            cpus=1,
            F1=1.0,
            bit_cutoffs="gathering",
        )[0]
        self.assert_exact_hits(expected_first, next(iterator))
        with self.assertRaises(MissingCutoffs) as actual_error:
            next(iterator)

        with self.assertRaises(MissingCutoffs) as reference_error:
            pyhmmer.plan7.Pipeline(
                self.alphabet, F1=1.0, bit_cutoffs="gathering"
            ).search_hmm(self.pairs[0].hmm, self.synthetic_targets)
        self.assertEqual(str(actual_error.exception), str(reference_error.exception))
        self.assertEqual(
            actual_error.exception.model_name,
            reference_error.exception.model_name,
        )

    def test_empty_inputs_and_bridge_contract_validation(self):
        self.assertEqual(list(hmmsearch((), self.synthetic_batch, cpus=1)), [])
        empty_candidates = self.synthetic_batch.candidate_batch((), F1=0.25)
        self.assertEqual(list(hmmsearch((), empty_candidates, cpus=8)), [])

        for cpus in (0, -1):
            with self.subTest(cpus=cpus):
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    hmmsearch((), self.synthetic_batch, cpus=cpus)
        for cpus in (True, 1.5):
            with self.subTest(cpus=cpus):
                with self.assertRaises(TypeError):
                    hmmsearch((), self.synthetic_batch, cpus=cpus)

        pairs = self.pairs[2:4]
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=0.02)
        with self.assertRaisesRegex(ValueError, "same order"):
            hmmsearch(tuple(reversed(pairs)), candidates, cpus=1)
        with self.assertRaisesRegex(ValueError, "does not match"):
            hmmsearch(pairs, candidates, cpus=1, F1=0.5)
        with self.assertRaisesRegex(ValueError, "bound target alphabet"):
            hmmsearch(
                pairs,
                candidates,
                cpus=1,
                alphabet=pyhmmer.easel.Alphabet.dna(),
            )
        self.assertEqual(
            len(
                list(
                    hmmsearch(
                        pairs,
                        candidates,
                        cpus=1,
                        alphabet=pyhmmer.easel.Alphabet.amino(),
                    )
                )
            ),
            len(pairs),
        )
        with self.assertRaises(TypeError):
            hmmsearch((pairs[0].hmm,), self.synthetic_batch, cpus=1)
        with self.assertRaisesRegex(TypeError, "SequenceBatch or CandidateBatch"):
            hmmsearch(pairs, self.synthetic_targets, cpus=1)

    def test_threaded_pipeline_construction_errors_propagate_directly(self):
        pairs = self.pairs[2:4]
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        iterator = hmmsearch(
            pairs,
            candidates,
            cpus=2,
            not_a_pipeline_option=True,
        )
        with self.assertRaises(TypeError):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
