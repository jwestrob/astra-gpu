import io
import os
import random
import struct
import sys
import tempfile
import threading
import unittest
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pyhmmer
from pyhmmer.errors import MissingCutoffs


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
try:
    from plan7_gpu import _native, _pipeline
    from plan7_gpu.adapter import (
        CandidateBatch,
        ProfileSession,
        SequenceBatch,
        _candidate_state,
        load_pressed_profiles,
    )
    import plan7_gpu.astra_search as astra_search_module
    from plan7_gpu.astra_search import (
        _AstraTSVRows,
        _ContinuationPool,
        _hmmsearch_astra_tsv,
        _hmmsearch_astra_tsv_with_continuation_pool,
        _hmmsearch_with_continuation_pool,
        hmmsearch,
    )
except ImportError:
    _native = None
    _pipeline = None
    CandidateBatch = None
    ProfileSession = None
    SequenceBatch = None
    _candidate_state = None
    load_pressed_profiles = None
    astra_search_module = None
    hmmsearch = None
    _ContinuationPool = None
    _AstraTSVRows = None
    _hmmsearch_astra_tsv = None
    _hmmsearch_astra_tsv_with_continuation_pool = None
    _hmmsearch_with_continuation_pool = None


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

    def test_postfilter_matches_pyhmmer_when_private_seam_is_available(self):
        if _pipeline is None or not _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is unavailable")
        pairs = self.pairs[:4]
        option_sets = (
            {"F1": 0.02},
            {"F1": 0.02, "bias_filter": False},
            {"F1": 0.02, "F2": 1.0, "F3": 1.0},
        )
        for options in option_sets:
            for cpus in (1, 2):
                with self.subTest(options=options, cpus=cpus):
                    expected = self.reference(
                        pairs, self.real_targets, cpus=cpus, **options
                    )
                    actual = list(
                        hmmsearch(
                            pairs,
                            self.real_batch,
                            cpus=cpus,
                            postfilter=True,
                            **options,
                        )
                    )
                    for expected_hits, actual_hits in zip(
                        expected, actual, strict=True
                    ):
                        self.assert_exact_hits(expected_hits, actual_hits)

    def test_forward_augmentation_tiny_budget_falls_back_exactly(self):
        if (
            _pipeline is None
            or not _pipeline._filter_scores_seam_available()
            or not _pipeline._filter_and_forward_scores_seam_available()
            or not hasattr(_native, "ForwardProfiles")
            or not hasattr(_native.SequenceBatch, "forward_candidates_many_raw")
        ):
            self.skipTest("exact Forward augmentation is unavailable")

        pairs = self.pairs[2:3]
        options = {"F1": 0.5, "F2": 1.0, "F3": 1.0}
        expected = self.reference(pairs, self.synthetic_targets, **options)
        original = SequenceBatch._postfilter_forward_batch
        calls = []
        built = []

        def observed(batch, profile_pairs, f1, f2, f3, bias_filter):
            calls.append((tuple(profile_pairs), f1, f2, f3, bias_filter))
            candidates = original(batch, profile_pairs, f1, f2, f3, bias_filter)
            built.append(candidates)
            return candidates

        with (
            mock.patch.object(SequenceBatch, "_postfilter_forward_batch", new=observed),
            mock.patch("plan7_gpu.adapter._FORWARD_SPECIAL_BYTE_BUDGET", 1),
        ):
            actual = list(
                hmmsearch(
                    pairs,
                    self.synthetic_batch,
                    cpus=1,
                    postfilter=True,
                    **options,
                )
            )

        self.assertEqual(calls, [(pairs, 0.5, 1.0, 1.0, True)])
        state = _candidate_state(built[0])
        self.assertIsNotNone(state.sealed_postfilter)
        self.assertIsNone(state.forward)
        self.assert_exact_hits(expected[0], actual[0])

    def test_missing_seal_factory_keeps_validated_fallback(self):
        if _pipeline is None or not _pipeline._filter_scores_seam_available():
            self.skipTest("exact post-filter continuation is unavailable")
        pairs = self.pairs[2:3]
        options = {"F1": 0.5, "F2": 1.0, "F3": 1.0}
        expected = self.reference(pairs, self.synthetic_targets, **options)
        with mock.patch.object(_pipeline, "_seal_postfilter_batch_bound", new=None):
            actual = list(
                hmmsearch(
                    pairs,
                    self.synthetic_batch,
                    cpus=1,
                    postfilter=True,
                    **options,
                )
            )
        self.assert_exact_hits(expected[0], actual[0])

    def test_candidate_search_consumes_real_forward_augmentation(self):
        if (
            _pipeline is None
            or not _pipeline._filter_scores_seam_available()
            or not _pipeline._filter_and_forward_scores_seam_available()
            or not hasattr(_native, "ForwardProfiles")
            or not hasattr(_native.SequenceBatch, "forward_candidates_many_raw")
        ):
            self.skipTest("exact Forward augmentation is unavailable")

        pairs = self.pairs[2:4]
        options = {"F1": 0.5, "F2": 1.0, "F3": 1.0}
        with SequenceBatch(self.synthetic_targets) as batch:
            candidates = batch._postfilter_forward_batch(pairs, 0.5, 1.0, 1.0, True)
        state = _candidate_state(candidates)
        self.assertIsNotNone(state.sealed_postfilter)
        self.assertIsNone(state.forward)
        expected = self.reference(pairs, self.synthetic_targets, **options)
        original = _pipeline._search_hmm_sealed_postfilter_bound
        calls = []

        def observed(*args):
            calls.append(threading.get_ident())
            return original(*args)

        with mock.patch.object(
            _pipeline,
            "_search_hmm_sealed_postfilter_bound",
            new=observed,
        ):
            actual = list(hmmsearch(pairs, candidates, cpus=2, **options))

        self.assertGreaterEqual(len(calls), 1)
        for expected_hits, actual_hits in zip(expected, actual, strict=True):
            self.assert_exact_hits(expected_hits, actual_hits)

    def test_sharded_sparse_continuation_matches_unsharded(self):
        if (
            _pipeline is None
            or not hasattr(
                _pipeline,
                "_search_hmm_sealed_sparse_journal_v3_shard_bound",
            )
        ):
            self.skipTest("sparse continuation sharding is unavailable")
        pairs = self.pairs
        options = {
            "F1": 0.5,
            "F2": 1.0,
            "F3": 1.0,
            "E": 10.0,
            "domE": 10.0,
            "incE": 10.0,
            "incdomE": 10.0,
        }
        with ProfileSession(pairs, pack_workers=1) as session:
            with session.select(range(len(pairs))) as selection:
                candidates = self.synthetic_batch._postfilter_forward_selection(
                    selection,
                    options["F1"],
                    options["F2"],
                    options["F3"],
                    True,
                    pipeline=pyhmmer.plan7.Pipeline(
                        self.alphabet, **options
                    ),
                    sparse_journal_v3=True,
                )

        state = _candidate_state(candidates)
        shard_hints = tuple(
            _pipeline._sealed_continuation_shard_work_hints_bound(
                state.sealed_postfilter
            )
        )
        try:
            heavy_row = next(
                row for row, hints in enumerate(shard_hints) if len(hints) >= 2
            )
        except StopIteration:
            self.skipTest("fixture produced no shardable sparse profile")
        expected = [
            candidates.search(
                row,
                pyhmmer.plan7.Pipeline(self.alphabet, **options),
            )
            for row in range(len(candidates))
        ]

        forced_profile_hints = [1] * len(candidates)
        forced_profile_hints[heavy_row] = max(
            1_000_000,
            sum(shard_hints[heavy_row]),
        )
        astra_search_module._reset_continuation_scheduler_statistics()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "PLAN7_GPU_CONTINUATION_SCHEDULER": "completion",
                    "PLAN7_GPU_CONTINUATION_TASK_POLICY": "sharded",
                    "PLAN7_GPU_CONTINUATION_PROFILE": "1",
                },
            ),
            mock.patch.object(
                _pipeline,
                "_sealed_continuation_work_hints_bound",
                return_value=tuple(forced_profile_hints),
            ),
        ):
            actual = list(
                hmmsearch(pairs, candidates, cpus=4, **options)
            )

        self.assertEqual(len(actual), len(expected))
        for expected_hits, actual_hits in zip(expected, actual, strict=True):
            self.assert_exact_hits(expected_hits, actual_hits)
        statistics = astra_search_module._continuation_scheduler_statistics()
        self.assertEqual(statistics["sharded_task_call_count"], 1)
        self.assertGreater(statistics["shard_task_count"], 1)

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

    def test_private_worker_tsv_sink_is_exact_and_ordered(self):
        pairs = self.pairs[:2]
        options = {"E": 10.0, "domE": 10.0, "incE": 10.0, "incdomE": 10.0}
        expected = list(hmmsearch(pairs, self.real_batch, cpus=2, **options))
        for cpus in (1, 2):
            with self.subTest(cpus=cpus):
                actual = list(
                    _hmmsearch_astra_tsv(
                        pairs, self.real_batch, cpus=cpus, **options
                    )
                )
                self.assertTrue(
                    all(type(item) is _AstraTSVRows for item in actual)
                )
                self.assertEqual(
                    [item.rows for item in actual],
                    [
                        _pipeline._astra_tsv_rows_bound(hits)
                        for hits in expected
                    ],
                )
                self.assertEqual(
                    [item.row_count for item in actual],
                    [item.rows.count("\n") for item in actual],
                )
                self.assertEqual(
                    [item.byte_count for item in actual],
                    [len(item.rows.encode("utf-8")) for item in actual],
                )

        pool = _ContinuationPool(2)
        try:
            pooled = list(
                _hmmsearch_astra_tsv_with_continuation_pool(
                    pairs,
                    self.real_batch,
                    cpus=2,
                    continuation_pool=pool,
                    **options,
                )
            )
        finally:
            pool.close()
        self.assertEqual(
            [item.rows for item in pooled],
            [_pipeline._astra_tsv_rows_bound(hits) for hits in expected],
        )

    def test_private_worker_tsv_sink_preserves_canonical_failure_order(self):
        pairs = self.pairs[:3]
        options = {"E": 10.0, "domE": 10.0, "incE": 10.0, "incdomE": 10.0}

        class RenderFailure(RuntimeError):
            pass

        def renderer(hits):
            if hits.query.name == pairs[1].hmm.name:
                raise RenderFailure("row-one-render-failure")
            return hits.query.name

        iterator = astra_search_module._hmmsearch_impl(
            pairs,
            self.real_batch,
            cpus=2,
            _result_renderer=renderer,
            **options,
        )
        first = next(iterator)
        self.assertIs(type(first), _AstraTSVRows)
        self.assertEqual(first.rows, pairs[0].hmm.name)
        with self.assertRaisesRegex(RenderFailure, "row-one-render-failure"):
            next(iterator)
        iterator.close()

    def test_private_continuation_pool_reuses_worker_pipelines_across_calls(self):
        pairs = self.pairs * 2
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        expected = self.reference(pairs, self.synthetic_targets, cpus=2, F1=1.0)
        pool = _ContinuationPool(2)
        try:
            first = list(
                _hmmsearch_with_continuation_pool(
                    pairs,
                    candidates,
                    cpus=2,
                    F1=1.0,
                    continuation_pool=pool,
                )
            )
            first_statistics = pool.statistics
            second = list(
                _hmmsearch_with_continuation_pool(
                    pairs,
                    candidates,
                    cpus=2,
                    F1=1.0,
                    continuation_pool=pool,
                )
            )
            second_statistics = pool.statistics
        finally:
            pool.close()

        self.assertEqual(first_statistics["call_count"], 1)
        self.assertEqual(second_statistics["call_count"], 2)
        self.assertGreater(first_statistics["pipeline_count"], 0)
        self.assertLessEqual(first_statistics["pipeline_count"], 2)
        self.assertEqual(
            second_statistics["pipeline_count"],
            first_statistics["pipeline_count"],
        )
        for observed in (first, second):
            for expected_hits, actual_hits in zip(
                expected, observed, strict=True
            ):
                self.assert_exact_hits(expected_hits, actual_hits)

    def test_threaded_order_and_worker_local_pipeline_reuse(self):
        # Forty rows select the maximum eight-row task size and leave one task
        # beyond the strict four-task reorder window for two workers.
        pairs = self.pairs * 5
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        expected = self.reference(pairs, self.synthetic_targets, cpus=2, F1=1.0)

        original_search = CandidateBatch.search
        row_eight_started = threading.Event()
        row_thirty_one_finished = threading.Event()
        row_thirty_two_started = threading.Event()
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
        submitted_ranges = []
        shutdown_calls = []

        class ObservedExecutor:
            def __init__(self, *args, **kwargs):
                self._executor = RealThreadPoolExecutor(*args, **kwargs)

            def submit(self, function, *args, **kwargs):
                with lock:
                    submitted_ranges.append(args[1:3])
                return self._executor.submit(function, *args, **kwargs)

            def shutdown(self, *, wait=True, cancel_futures=False):
                shutdown_calls.append((wait, cancel_futures))
                return self._executor.shutdown(
                    wait=wait,
                    cancel_futures=cancel_futures,
                )

        def observed_search(candidate_batch, row, pipeline, **kwargs):
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
                # Keep the first eight-row task blocked while the other worker
                # drains the other three tasks in the reorder window. Row 32
                # must remain unsubmitted.
                if row == 0:
                    if not release_row_zero.wait(10):
                        raise RuntimeError("test did not release row 0")
                    row_zero_released.set()
                elif row == 8:
                    row_eight_started.set()
                elif row == 32:
                    row_thirty_two_started.set()
                return original_search(candidate_batch, row, pipeline, **kwargs)
            finally:
                with lock:
                    completion_order.append(row)
                    active_pipelines.remove(pipeline_id)
                if row == 31:
                    row_thirty_one_finished.set()

        actual = []
        consumer_errors = []

        def consume():
            try:
                actual.extend(hmmsearch(pairs, candidates, cpus=2))
            except BaseException as error:
                consumer_errors.append(error)

        with (
            mock.patch(
                "plan7_gpu.astra_search.ThreadPoolExecutor",
                new=ObservedExecutor,
            ),
            mock.patch.dict(
                os.environ,
                {
                    "PLAN7_GPU_CONTINUATION_SCHEDULER": "oldest",
                    "PLAN7_GPU_CONTINUATION_TASK_POLICY": "fixed",
                },
            ),
            mock.patch.object(CandidateBatch, "search", new=observed_search),
        ):
            consumer = threading.Thread(target=consume, name="astra-test-consumer")
            consumer.start()
            try:
                self.assertTrue(row_eight_started.wait(10))
                self.assertTrue(row_thirty_one_finished.wait(10))
                self.assertFalse(
                    row_thirty_two_started.wait(0.5),
                    "row beyond the reorder window started while row 0 was held",
                )
                with lock:
                    self.assertEqual(
                        rows_started_while_zero_blocked,
                        {0, *range(8, 32)},
                    )
                    self.assertEqual(
                        submitted_ranges,
                        [(0, 8), (8, 16), (16, 24), (24, 32)],
                    )
            finally:
                release_row_zero.set()
                consumer.join(10)

        self.assertTrue(row_eight_started.is_set())
        self.assertTrue(row_zero_released.is_set())
        self.assertTrue(row_thirty_two_started.is_set())
        self.assertFalse(consumer.is_alive())
        self.assertFalse(consumer_errors)
        self.assertEqual(completion_order[0], 8)
        self.assertEqual(
            submitted_ranges,
            [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)],
        )
        self.assertEqual(shutdown_calls, [(True, True)])
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

    def test_retained_scheduler_defaults_to_completion_and_balanced(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                astra_search_module._continuation_scheduler_mode(),
                "completion",
            )
            self.assertEqual(
                astra_search_module._continuation_task_policy(),
                "balanced",
            )

    def test_completion_scheduler_refills_behind_blocked_oldest_task(self):
        pairs = self.pairs * 5
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        expected = self.reference(pairs, self.synthetic_targets, cpus=2, F1=1.0)
        original_search = CandidateBatch.search
        release_row_zero = threading.Event()
        row_thirty_two_started = threading.Event()

        def observed_search(candidate_batch, row, pipeline, **kwargs):
            if row == 0:
                if not release_row_zero.wait(10):
                    raise RuntimeError("test did not release row 0")
            elif row == 32:
                row_thirty_two_started.set()
            return original_search(candidate_batch, row, pipeline, **kwargs)

        actual = []
        errors = []

        def consume():
            try:
                actual.extend(hmmsearch(pairs, candidates, cpus=2))
            except BaseException as error:
                errors.append(error)

        astra_search_module._reset_continuation_scheduler_statistics()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "PLAN7_GPU_CONTINUATION_SCHEDULER": "completion",
                    "PLAN7_GPU_CONTINUATION_PROFILE": "1",
                },
            ),
            mock.patch.object(CandidateBatch, "search", new=observed_search),
        ):
            consumer = threading.Thread(target=consume, name="astra-test-consumer")
            consumer.start()
            try:
                self.assertTrue(
                    row_thirty_two_started.wait(10),
                    "completion-driven scheduler did not refill behind row 0",
                )
                self.assertEqual(actual, [])
            finally:
                release_row_zero.set()
                consumer.join(10)

        self.assertFalse(consumer.is_alive())
        self.assertFalse(errors)
        self.assertEqual(
            [hits.query.name for hits in actual],
            [pair.hmm.name for pair in pairs],
        )
        for expected_hits, actual_hits in zip(expected, actual, strict=True):
            self.assert_exact_hits(expected_hits, actual_hits)
        statistics = astra_search_module._continuation_scheduler_statistics()
        self.assertEqual(statistics["completion_call_count"], 1)
        self.assertEqual(statistics["oldest_call_count"], 0)
        self.assertEqual(statistics["task_count"], 5)
        self.assertEqual(statistics["maximum_active_workers"], 2)
        self.assertGreaterEqual(statistics["maximum_completed_tasks"], 1)

    def test_threaded_yields_prior_row_then_propagates_pipeline_error(self):
        # Ten rows produce three-row tasks for two workers. The missing-cutoff
        # failure is inside the first task, so row 0 must still be yielded and
        # row 2 in that task must never run.
        pairs = (self.pairs[2], self.pairs[0]) + (self.pairs[2],) * 8
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        searched_rows = []
        original_search = CandidateBatch.search

        def observed_search(candidate_batch, row, pipeline, **kwargs):
            searched_rows.append(row)
            return original_search(candidate_batch, row, pipeline, **kwargs)

        expected_first = self.reference(
            pairs[:1],
            self.synthetic_targets,
            cpus=1,
            F1=1.0,
            bit_cutoffs="gathering",
        )[0]
        with (
            mock.patch.dict(
                os.environ,
                {"PLAN7_GPU_CONTINUATION_SCHEDULER": "completion"},
            ),
            mock.patch.object(CandidateBatch, "search", new=observed_search),
        ):
            iterator = hmmsearch(
                pairs,
                candidates,
                cpus=2,
                bit_cutoffs="gathering",
            )
            self.assert_exact_hits(expected_first, next(iterator))
            with self.assertRaises(MissingCutoffs) as actual_error:
                next(iterator)

        self.assertNotIn(2, searched_rows)

        with self.assertRaises(MissingCutoffs) as reference_error:
            pyhmmer.plan7.Pipeline(
                self.alphabet, F1=1.0, bit_cutoffs="gathering"
            ).search_hmm(self.pairs[0].hmm, self.synthetic_targets)
        self.assertEqual(str(actual_error.exception), str(reference_error.exception))
        self.assertEqual(
            actual_error.exception.model_name,
            reference_error.exception.model_name,
        )
        self.assertFalse(
            any(
                thread.name.startswith("plan7-gpu-astra")
                for thread in threading.enumerate()
            )
        )

    def test_threaded_close_cancels_queued_chunks_and_joins_workers(self):
        pairs = self.pairs * 5
        candidates = self.synthetic_batch.candidate_batch(pairs, F1=1.0)
        original_search = CandidateBatch.search
        blocked_rows_started = {8: threading.Event(), 16: threading.Event()}
        release_workers = threading.Event()
        searched_rows = []
        lock = threading.Lock()

        def observed_search(candidate_batch, row, pipeline, **kwargs):
            with lock:
                searched_rows.append(row)
            if row in blocked_rows_started:
                blocked_rows_started[row].set()
                if not release_workers.wait(10):
                    raise RuntimeError("test did not release worker")
            return original_search(candidate_batch, row, pipeline, **kwargs)

        with mock.patch.object(CandidateBatch, "search", new=observed_search):
            iterator = hmmsearch(pairs, candidates, cpus=2)
            first = next(iterator)
            self.assertEqual(first.query.name, pairs[0].hmm.name)
            for event in blocked_rows_started.values():
                self.assertTrue(event.wait(10))

            closer = threading.Thread(target=iterator.close, name="astra-test-close")
            closer.start()
            try:
                closer.join(0.2)
                self.assertTrue(closer.is_alive(), "close did not wait for workers")
            finally:
                release_workers.set()
                closer.join(10)

        self.assertFalse(closer.is_alive())
        with lock:
            self.assertFalse(any(row >= 24 for row in searched_rows))
        self.assertFalse(
            any(
                thread.name.startswith("plan7-gpu-astra")
                for thread in threading.enumerate()
            )
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
