import ctypes
import io
import math
import os
import platform
import subprocess
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import pyhmmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import (
        ProfileSelection,
        ProfileSession,
        SequenceBatch,
        load_pressed_profiles,
    )
    from plan7_gpu import _native
    from plan7_gpu.adapter import (
        _candidate_state,
        _pair_state,
        _profile_selection_state,
        _sequence_state,
    )
except ImportError:
    ProfileSelection = None
    ProfileSession = None
    SequenceBatch = None
    load_pressed_profiles = None
    _native = None
    _candidate_state = None
    _pair_state = None
    _profile_selection_state = None
    _sequence_state = None

try:
    from plan7_gpu import _pipeline
except ImportError:
    _pipeline = None


DATA = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


def postfilter_seam_available():
    return _pipeline is not None and _pipeline._filter_scores_seam_available()


def forward_seam_available():
    return (
        _pipeline is not None
        and _pipeline._filter_and_forward_scores_seam_available()
    )


class ProfileSessionFixture:
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="plan7-profile-session-test-"
        )
        hmms = []
        for name in ("RREFam.hmm", "LuxC.hmm", "Thioesterase.hmm"):
            with pyhmmer.plan7.HMMFile(DATA / name) as hmm_file:
                hmms.append(hmm_file.read())
        cls.base = Path(cls.temporary.name) / "models"
        pyhmmer.hmmer.hmmpress(hmms, cls.base)
        cls.pairs = load_pressed_profiles(cls.base)
        cls.alphabet = cls.pairs[0].hmm.alphabet
        sequences = [
            pyhmmer.easel.TextSequence(
                name=b"mixed",
                sequence="ACDEFGHIKLMNPQRSTVWY" * 5,
            ).digitize(cls.alphabet),
            pyhmmer.easel.TextSequence(
                name=b"low-complexity",
                sequence="A" * 91,
            ).digitize(cls.alphabet),
            pyhmmer.easel.TextSequence(
                name=b"short",
                sequence="MTEYKLVVVGAGGVGKSALTIQLIQ",
            ).digitize(cls.alphabet),
        ]
        cls.targets = pyhmmer.easel.DigitalSequenceBlock(cls.alphabet, sequences)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def pipeline(cls, f1=0.02, **options):
        arguments = {
            "F1": f1,
            "E": 10.0,
            "domE": 10.0,
            "incE": 10.0,
            "incdomE": 10.0,
        }
        arguments.update(options)
        return pyhmmer.plan7.Pipeline(
            cls.alphabet,
            **arguments,
        )

    @staticmethod
    def hits_bytes(hits):
        output = io.BytesIO()
        hits.write(output, format="targets", header=True)
        return output.getvalue()


@unittest.skipUnless(_native is not None, "CUDA extension unavailable")
class HostProfileSessionTests(ProfileSessionFixture, unittest.TestCase):
    def test_snapshot_statistics_and_ordered_noncontiguous_selection(self):
        session = ProfileSession(self.pairs)
        try:
            statistics = session.statistics
            lengths = [pair.hmm.M for pair in self.pairs]
            q_counts = [(length + 31) // 32 for length in lengths]
            forward_q_counts = [(length + 3) // 4 for length in lengths]
            self.assertEqual(statistics["profile_count"], 3)
            self.assertEqual(statistics["worker_count"], 3)
            self.assertEqual(statistics["build_worker_count"], 3)
            self.assertEqual(statistics["selection_worker_count"], 3)
            self.assertEqual(statistics["parallel_run_count"], 1)
            self.assertEqual(statistics["build_parallel_run_count"], 1)
            self.assertEqual(statistics["selection_parallel_run_count"], 0)
            self.assertEqual(statistics["ssv_score_bytes"], sum(lengths) * 29)
            self.assertEqual(statistics["bias_profile_bytes"], 3 * 272)
            self.assertEqual(statistics["viterbi_descriptor_bytes"], 3 * 96)
            self.assertEqual(
                statistics["viterbi_emission_bytes"],
                sum(q_counts) * 29 * 32 * 2,
            )
            self.assertEqual(
                statistics["viterbi_transition_bytes"],
                sum(q_counts) * 8 * 32 * 2,
            )
            self.assertEqual(statistics["forward_descriptor_bytes"], 3 * 48)
            self.assertEqual(
                statistics["forward_emission_bytes"],
                sum(forward_q_counts) * 29 * 4 * 4,
            )
            self.assertEqual(
                statistics["forward_transition_bytes"],
                sum(forward_q_counts) * 8 * 4 * 4,
            )

            first = session.select([2, 0])
            empty = session.select([])
            try:
                self.assertEqual(first.indices, (2, 0))
                self.assertEqual(len(first), 2)
                self.assertEqual(first.identity[0], statistics["session_id"])
                self.assertGreater(first.host_bytes, 0)
                self.assertEqual(empty.indices, ())
                self.assertEqual(empty.host_bytes, 0)
                updated = session.statistics
                self.assertEqual(updated["selection_count"], 2)
                self.assertEqual(updated["parallel_run_count"], 2)
                self.assertEqual(updated["build_parallel_run_count"], 1)
                self.assertEqual(updated["selection_parallel_run_count"], 1)
            finally:
                empty.close()
                first.close()
        finally:
            session.close()

    def test_pack_worker_budget_is_explicit_and_bounded(self):
        for requested, expected in (
            (None, 3),
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (99, 3),
        ):
            with self.subTest(requested=requested):
                options = {} if requested is None else {"pack_workers": requested}
                with ProfileSession(self.pairs, **options) as session:
                    self.assertEqual(
                        session.statistics["worker_count"], expected
                    )
                    self.assertEqual(
                        session.statistics["build_worker_count"], expected
                    )
                    self.assertEqual(
                        session.statistics["selection_worker_count"], expected
                    )
                    with session.select([2, 0]):
                        self.assertEqual(
                            session.statistics["parallel_run_count"], 2
                        )
        for value in (True, 1.5, "1"):
            with self.subTest(invalid=value):
                with self.assertRaisesRegex(TypeError, "pack_workers"):
                    ProfileSession(self.pairs, pack_workers=value)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ProfileSession(self.pairs, pack_workers=-1)

    def test_build_and_selection_worker_budgets_are_independent(self):
        with ProfileSession(
            self.pairs,
            build_workers=2,
            selection_workers=0,
        ) as session:
            statistics = session.statistics
            self.assertEqual(statistics["worker_count"], 0)
            self.assertEqual(statistics["build_worker_count"], 2)
            self.assertEqual(statistics["selection_worker_count"], 0)
            self.assertEqual(statistics["build_parallel_run_count"], 1)
            self.assertEqual(statistics["selection_parallel_run_count"], 0)
            with session.select([2, 0]):
                statistics = session.statistics
                self.assertEqual(statistics["parallel_run_count"], 2)
                self.assertEqual(
                    statistics["selection_parallel_run_count"], 1
                )

        with ProfileSession(
            self.pairs,
            pack_workers=0,
            build_workers=2,
        ) as session:
            self.assertEqual(session.statistics["build_worker_count"], 2)
            self.assertEqual(session.statistics["selection_worker_count"], 0)

        with ProfileSession(
            self.pairs,
            pack_workers=1,
            build_workers=99,
            selection_workers=2,
        ) as session:
            self.assertEqual(session.statistics["build_worker_count"], 3)
            self.assertEqual(session.statistics["selection_worker_count"], 2)

        for name in ("build_workers", "selection_workers"):
            for value in (True, 1.5, "1"):
                with self.subTest(name=name, invalid=value):
                    with self.assertRaisesRegex(TypeError, name):
                        ProfileSession(self.pairs, **{name: value})
            with self.subTest(name=name, invalid=-1):
                with self.assertRaisesRegex(ValueError, name):
                    ProfileSession(self.pairs, **{name: -1})

    @unittest.skipUnless(
        platform.system() == "Linux" and Path("/proc/self/task").is_dir(),
        "worker thread lifecycle probe requires Linux procfs",
    )
    def test_build_workers_retire_and_selection_workers_persist(self):
        tasks = Path("/proc/self/task")
        baseline = {entry.name for entry in tasks.iterdir()}
        with ProfileSession(
            self.pairs,
            build_workers=2,
            selection_workers=0,
        ):
            self.assertEqual(
                {entry.name for entry in tasks.iterdir()}, baseline
            )

        session = ProfileSession(
            self.pairs,
            build_workers=0,
            selection_workers=2,
        )
        persistent = {entry.name for entry in tasks.iterdir()} - baseline
        try:
            self.assertEqual(len(persistent), 2)
        finally:
            session.close()
        self.assertTrue(
            persistent.isdisjoint(entry.name for entry in tasks.iterdir())
        )

    @unittest.skipUnless(
        platform.system() == "Linux" and Path("/proc/self/task").is_dir(),
        "worker thread lifecycle probe requires Linux procfs",
    )
    def test_build_workers_retire_after_snapshot_failure(self):
        tasks = Path("/proc/self/task")
        baseline = {entry.name for entry in tasks.iterdir()}
        invalid = pyhmmer.plan7.OptimizedProfile.__new__(
            pyhmmer.plan7.OptimizedProfile
        )
        background = pyhmmer.plan7.Background(self.alphabet)
        with self.assertRaisesRegex(
            ValueError, "invalid optimized Viterbi profile"
        ):
            _native.ProfileSession(
                [invalid],
                memoryview(background.residue_frequencies),
                1,
                0,
            )
        self.assertEqual({entry.name for entry in tasks.iterdir()}, baseline)

    def test_selection_validation_and_lifecycle(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ProfileSession([])
        with self.assertRaisesRegex(ValueError, "unique"):
            ProfileSession([self.pairs[0], self.pairs[0]])
        reloaded = load_pressed_profiles(self.base)
        with self.assertRaisesRegex(ValueError, "unique"):
            ProfileSession([self.pairs[0], reloaded[0]])
        other_base = Path(self.temporary.name) / "other-models"
        pyhmmer.hmmer.hmmpress(
            [pair.hmm for pair in self.pairs], other_base
        )
        other_pairs = load_pressed_profiles(other_base)
        with self.assertRaisesRegex(ValueError, "different databases"):
            ProfileSession([self.pairs[0], other_pairs[1]])
        with self.assertRaisesRegex(TypeError, "load_pressed_profiles"):
            ProfileSession([object()])
        with self.assertRaisesRegex(TypeError, "ProfileSession.select"):
            ProfileSelection()

        session = ProfileSession(self.pairs)
        for values, exception in (
            ([True], TypeError),
            ([1.0], TypeError),
            ([-1], IndexError),
            ([3], IndexError),
            ([1, 1], ValueError),
        ):
            with self.subTest(values=values):
                with self.assertRaises(exception):
                    session.select(values)
        selection = session.select([1])
        session.close()
        self.assertTrue(session.closed)
        self.assertFalse(selection.closed)
        self.assertEqual(selection.indices, (1,))
        selection.close()
        self.assertTrue(selection.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.select([0])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            selection.identity

    def test_session_creation_waits_for_every_pair_lock(self):
        pair_state = _pair_state(self.pairs[1])
        started = threading.Event()
        result = []
        failures = []

        def create():
            started.set()
            try:
                result.append(ProfileSession(self.pairs))
            except BaseException as error:
                failures.append(error)

        pair_state.lock.acquire()
        try:
            worker = threading.Thread(target=create)
            worker.start()
            self.assertTrue(started.wait(2))
            time.sleep(0.05)
            self.assertTrue(worker.is_alive())
        finally:
            pair_state.lock.release()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        if failures:
            raise failures[0]
        self.assertEqual(len(result), 1)
        result[0].close()

    def test_selection_creation_and_close_are_serialized(self):
        session = ProfileSession(self.pairs)
        barrier = threading.Barrier(8)
        failures = []

        def select_and_close(worker_index):
            try:
                barrier.wait()
                order = [2, 0] if worker_index % 2 else [0, 2]
                selection = session.select(order)
                self.assertEqual(selection.indices, tuple(order))
                selection.close()
            except BaseException as error:
                failures.append(error)

        workers = [
            threading.Thread(target=select_and_close, args=(index,))
            for index in range(8)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        if failures:
            raise failures[0]
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertEqual(session.statistics["selection_count"], 8)

        barrier = threading.Barrier(8)
        failures.clear()

        def close():
            try:
                barrier.wait()
                session.close()
            except BaseException as error:
                failures.append(error)

        workers = [threading.Thread(target=close) for _ in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        if failures:
            raise failures[0]
        self.assertTrue(session.closed)

    def test_host_snapshot_needs_no_visible_cuda_device(self):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONPATH"] = str(ROOT / "python")
        command = (
            "from plan7_gpu import ProfileSession, load_pressed_profiles\n"
            f"pairs = load_pressed_profiles({str(self.base)!r})\n"
            "session = ProfileSession(pairs)\n"
            "selection = session.select([2, 0])\n"
            "assert selection.indices == (2, 0)\n"
            "assert session.statistics['parallel_run_count'] == 2\n"
            "selection.close(); session.close()\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", command],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    @unittest.skipUnless(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "floating-point environment probe is Linux x86_64 specific",
    )
    def test_snapshot_rejects_hostile_float_environment(self):
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original = libc.fegetround()
        try:
            self.assertEqual(libc.fesetround(0x400), 0)
            with self.assertRaisesRegex(RuntimeError, "floating-point"):
                ProfileSession(self.pairs)
        finally:
            self.assertEqual(libc.fesetround(original), 0)


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
@unittest.skipUnless(_pipeline is not None, "pipeline extension unavailable")
@unittest.skipIf(
    postfilter_seam_available(), "private filter-score seam is available"
)
class StockCudaProfileSessionTests(ProfileSessionFixture, unittest.TestCase):
    def test_postfilter_apis_require_private_continuation_seam(self):
        with ProfileSession(self.pairs) as session:
            with session.select([2]) as selection:
                with SequenceBatch(self.targets) as batch:
                    with self.assertRaisesRegex(
                        RuntimeError, "p7_PipelineFromFilterScores"
                    ):
                        batch.postfilter_batch([self.pairs[2]])
                    with self.assertRaisesRegex(
                        RuntimeError, "p7_PipelineFromFilterScores"
                    ):
                        batch.postfilter_selection(selection)
                    with self.assertRaisesRegex(
                        RuntimeError, "p7_PipelineFromFilterScores"
                    ):
                        batch._postfilter_forward_selection(
                            selection, 0.02, 1.0, 1.0, True
                        )


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
@unittest.skipUnless(
    postfilter_seam_available(), "private filter-score seam is unavailable"
)
class CudaProfileSessionTests(ProfileSessionFixture, unittest.TestCase):
    def test_noncontiguous_selection_matches_live_path_exactly(self):
        with ProfileSession(self.pairs) as session:
            with session.select([2, 0]) as selection:
                with SequenceBatch(self.targets) as batch:
                    expected = batch.postfilter_batch(
                        [self.pairs[2], self.pairs[0]], F1=0.02
                    )
                    actual = batch.postfilter_selection(selection, F1=0.02)
        expected_state = _candidate_state(expected)
        actual_state = _candidate_state(actual)
        self.assertEqual(actual_state.pairs, (self.pairs[2], self.pairs[0]))
        self.assertEqual(actual_state.offsets, expected_state.offsets)
        self.assertEqual(
            actual_state.postfilter_records,
            expected_state.postfilter_records,
        )
        self.assertEqual(
            [actual.candidate_count(row) for row in range(2)],
            [expected.candidate_count(row) for row in range(2)],
        )
        self.assertIsNotNone(actual_state.sealed_postfilter)

    @unittest.skipUnless(
        forward_seam_available(), "private Forward-score seam is unavailable"
    )
    def test_noncontiguous_forward_selection_matches_live_path_exactly(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        with ProfileSession(self.pairs, pack_workers=1) as session:
            with session.select([2, 0]) as selection:
                with SequenceBatch(self.targets) as batch:
                    expected = batch._postfilter_forward_batch(
                        [self.pairs[2], self.pairs[0]],
                        options["F1"],
                        options["F2"],
                        options["F3"],
                        True,
                    )
                    actual = batch._postfilter_forward_selection(
                        selection,
                        options["F1"],
                        options["F2"],
                        options["F3"],
                        True,
                    )
                    actual_state = _candidate_state(actual)
                    live_rows, live_indices, _live_filters = (
                        _pipeline._select_forward_inputs_bound(
                            [
                                _pair_state(pair).optimized_profile
                                for pair in (self.pairs[2], self.pairs[0])
                            ],
                            memoryview(actual_state.postfilter_records),
                            memoryview(actual_state.offsets).cast("Q"),
                            memoryview(
                                _sequence_state(batch).residue_offsets
                            ).cast("Q"),
                            options["F2"],
                        )
                    )
                    (
                        _,
                        snapshot_rows,
                        snapshot_indices,
                        _,
                        _,
                        _,
                    ) = _sequence_state(
                        batch
                    ).native.forward_profile_selection_raw(
                        _profile_selection_state(selection).native,
                        memoryview(actual_state.postfilter_records),
                        memoryview(actual_state.offsets).cast("Q"),
                        memoryview(
                            _sequence_state(batch).residue_offsets
                        ).cast("Q"),
                        options["F2"],
                        options["F3"],
                    )
                    self.assertEqual(list(snapshot_rows), list(live_rows))
                    self.assertEqual(
                        list(snapshot_indices), list(live_indices)
                    )
                    self.assertGreaterEqual(
                        _sequence_state(batch).native.workspace_statistics[
                            "forward_run_count"
                        ],
                        2,
                    )
        expected_state = _candidate_state(expected)
        actual_state = _candidate_state(actual)
        self.assertEqual(actual_state.pairs, (self.pairs[2], self.pairs[0]))
        self.assertEqual(actual_state.offsets, expected_state.offsets)
        self.assertEqual(
            actual_state.postfilter_records,
            expected_state.postfilter_records,
        )
        self.assertIsNotNone(actual_state.sealed_postfilter)
        self.assertIsNone(actual_state.forward)
        for row in range(2):
            with self.subTest(row=row):
                expected_hits = expected.search(row, self.pipeline(**options))
                actual_hits = actual.search(row, self.pipeline(**options))
                self.assertEqual(
                    self.hits_bytes(actual_hits), self.hits_bytes(expected_hits)
                )

    @unittest.skipUnless(
        forward_seam_available(), "private Forward-score seam is unavailable"
    )
    def test_forward_batch_outlives_inputs_and_searches_concurrently(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        session = ProfileSession(self.pairs, pack_workers=0)
        selection = session.select([2, 0])
        session.close()
        batch = SequenceBatch(self.targets)
        candidates = batch._postfilter_forward_selection(
            selection,
            options["F1"],
            options["F2"],
            options["F3"],
            True,
        )
        batch.close()
        selection.close()

        expected = [
            self.hits_bytes(
                self.pipeline(**options).search_hmm(pair.hmm, self.targets)
            )
            for pair in (self.pairs[2], self.pairs[0])
        ]
        barrier = threading.Barrier(8)
        failures = []

        def search(worker_index):
            try:
                barrier.wait()
                row = worker_index % 2
                hits = candidates.search(row, self.pipeline(**options))
                self.assertEqual(self.hits_bytes(hits), expected[row])
            except BaseException as error:
                failures.append(error)

        workers = [
            threading.Thread(target=search, args=(index,)) for index in range(8)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(20)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        if failures:
            raise failures[0]

    @unittest.skipUnless(
        forward_seam_available(), "private Forward-score seam is unavailable"
    )
    def test_forward_provenance_mismatch_falls_back_exactly(self):
        generation = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        with ProfileSession(self.pairs) as session:
            with session.select([2]) as selection:
                with SequenceBatch(self.targets) as batch:
                    candidates = batch._postfilter_forward_selection(
                        selection,
                        generation["F1"],
                        generation["F2"],
                        generation["F3"],
                        True,
                    )
        mismatches = (
            {"F2": math.nextafter(1.0, 0.0), "F3": 1.0},
            {"F2": 1.0, "F3": math.nextafter(1.0, 0.0)},
            {"F2": 1.0, "F3": 1.0, "bias_filter": False},
        )
        for mismatch in mismatches:
            with self.subTest(mismatch=mismatch):
                options = dict(generation)
                options.update(mismatch)
                actual = candidates.search(0, self.pipeline(**options))
                expected = self.pipeline(**options).search_hmm(
                    self.pairs[2].hmm, self.targets
                )
                self.assertEqual(
                    self.hits_bytes(actual), self.hits_bytes(expected)
                )

    @unittest.skipUnless(
        forward_seam_available(), "private Forward-score seam is unavailable"
    )
    def test_forward_generation_uses_snapshot_after_live_arrays_change(self):
        options = {"F1": 0.99, "F2": 0.99, "F3": 0.99}
        session = ProfileSession(self.pairs)
        selection = session.select([2])
        profile = _pair_state(self.pairs[2]).optimized_profile
        emissions = memoryview(profile.rfv).cast("B")
        transitions = memoryview(profile.tfv).cast("B")
        parameters = profile.evalue_parameters.as_vector()
        original_emissions = emissions.tobytes()
        original_transitions = transitions.tobytes()
        original_parameters = tuple(parameters)
        try:
            with SequenceBatch(self.targets) as batch:
                baseline = batch._postfilter_forward_selection(
                    selection,
                    options["F1"],
                    options["F2"],
                    options["F3"],
                    True,
                )
                emissions[:] = bytes(len(emissions))
                transitions[:] = bytes(len(transitions))
                for index in range(2, 6):
                    parameters[index] = 1000.0
                changed = batch._postfilter_forward_selection(
                    selection,
                    options["F1"],
                    options["F2"],
                    options["F3"],
                    True,
                )
        finally:
            emissions[:] = original_emissions
            transitions[:] = original_transitions
            for index, value in enumerate(original_parameters):
                parameters[index] = value
            selection.close()
            session.close()

        baseline_hits = baseline.search(0, self.pipeline(**options))
        changed_hits = changed.search(0, self.pipeline(**options))
        expected_hits = self.pipeline(**options).search_hmm(
            self.pairs[2].hmm, self.targets
        )
        self.assertEqual(
            self.hits_bytes(changed_hits), self.hits_bytes(baseline_hits)
        )
        self.assertEqual(
            self.hits_bytes(changed_hits), self.hits_bytes(expected_hits)
        )

    @unittest.skipUnless(
        forward_seam_available(), "private Forward-score seam is unavailable"
    )
    def test_forward_native_stage_and_run_release_gil(self):
        targets = []
        for index in range(1024):
            target = self.targets[2].copy()
            target.name = f"target-{index}".encode()
            targets.append(target)
        target_block = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, targets
        )
        with ProfileSession(self.pairs, pack_workers=1) as session:
            with session.select([2]) as selection:
                with SequenceBatch(target_block) as batch:
                    candidates = batch.postfilter_selection(
                        selection, F1=0.99
                    )
                    state = _candidate_state(candidates)
                    self.assertEqual(candidates.candidate_count(0), len(targets))
                    ready = threading.Event()
                    stop = threading.Event()
                    counter = [0]

                    def spin():
                        ready.set()
                        while not stop.is_set():
                            counter[0] += 1

                    worker = threading.Thread(target=spin)
                    worker.start()
                    self.assertTrue(ready.wait(2))
                    before = counter[0]
                    try:
                        _sequence_state(
                            batch
                        ).native.forward_profile_selection_raw(
                            _profile_selection_state(selection).native,
                            memoryview(state.postfilter_records),
                            memoryview(state.offsets).cast("Q"),
                            memoryview(
                                _sequence_state(batch).residue_offsets
                            ).cast("Q"),
                            1.0,
                            1.0,
                        )
                        after = counter[0]
                    finally:
                        stop.set()
                        worker.join(5)
                    self.assertFalse(worker.is_alive())
                    self.assertGreater(after, before)

    def test_exact_rbv_snapshot_matches_live_path(self):
        profile = _pair_state(self.pairs[2]).optimized_profile
        scores = memoryview(profile.rbv)
        original = scores[0, 0]
        try:
            scores[0, 0] = (original + 1) % 256
            with ProfileSession(self.pairs) as session:
                self.assertEqual(
                    session.statistics["viterbi_exact_rbv_bytes"],
                    self.pairs[2].hmm.M * 29,
                )
                with session.select([2]) as selection:
                    with SequenceBatch(self.targets) as batch:
                        expected = batch.postfilter_batch([self.pairs[2]])
                        actual = batch.postfilter_selection(selection)
            self.assertEqual(
                _candidate_state(actual).postfilter_records,
                _candidate_state(expected).postfilter_records,
            )
            self.assertEqual(
                _candidate_state(actual).offsets,
                _candidate_state(expected).offsets,
            )
        finally:
            scores[0, 0] = original

    def test_generation_uses_snapshot_after_live_arrays_change(self):
        session = ProfileSession(self.pairs)
        selection = session.select([2])
        state = _pair_state(self.pairs[2])
        try:
            with SequenceBatch(self.targets) as batch:
                baseline = batch.postfilter_selection(selection)
                self.assertGreater(baseline.candidate_count(0), 0)
                parameters = state.optimized_profile.evalue_parameters.as_vector()
                scores = memoryview(state.optimized_profile.sbv)
                viterbi_scores = memoryview(state.optimized_profile.rbv)
                compositions = memoryview(state.optimized_profile.compositions)
                original_mu = parameters[0]
                original_score = scores[0, 0]
                original_viterbi_score = viterbi_scores[0, 0]
                original_composition = compositions[0]
                try:
                    parameters[0] = math.nan
                    scores[0, 0] = (original_score + 1) % 256
                    viterbi_scores[0, 0] = (original_viterbi_score + 1) % 256
                    compositions[0] = original_composition * 2.0
                    changed = batch.postfilter_selection(selection)
                finally:
                    parameters[0] = original_mu
                    scores[0, 0] = original_score
                    viterbi_scores[0, 0] = original_viterbi_score
                    compositions[0] = original_composition
            self.assertEqual(
                _candidate_state(changed).postfilter_records,
                _candidate_state(baseline).postfilter_records,
            )
            self.assertEqual(
                _candidate_state(changed).offsets,
                _candidate_state(baseline).offsets,
            )
        finally:
            selection.close()
            session.close()

    def test_candidate_batch_outlives_batch_selection_and_session(self):
        session = ProfileSession(self.pairs)
        selection = session.select([2])
        session.close()
        batch = SequenceBatch(self.targets)
        candidates = batch.postfilter_selection(selection)
        selection.close()
        batch.close()

        actual = candidates.search(0, self.pipeline())
        expected = self.pipeline().search_hmm(self.pairs[2].hmm, self.targets)
        self.assertEqual(self.hits_bytes(actual), self.hits_bytes(expected))

    def test_f1_one_preserves_shared_all_target_semantics(self):
        with ProfileSession(self.pairs) as session:
            with session.select([2, 0]) as selection:
                with SequenceBatch(self.targets) as batch:
                    candidates = batch.postfilter_selection(selection, F1=1.0)
                    with session.select([]) as empty:
                        empty_candidates = batch.postfilter_selection(
                            empty, F1=1.0
                        )
                        empty_filtered = batch.postfilter_selection(empty)
        state = _candidate_state(candidates)
        self.assertIsNone(state.postfilter_records)
        self.assertEqual(state.all_rows, b"\x01\x01")
        self.assertEqual(candidates.candidate_count(0), len(self.targets))
        self.assertEqual(candidates.candidate_count(1), len(self.targets))
        self.assertEqual(len(empty_candidates), 0)
        self.assertEqual(_candidate_state(empty_candidates).all_targets, b"")
        self.assertEqual(len(empty_filtered), 0)
        self.assertEqual(
            _candidate_state(empty_filtered).postfilter_records, b""
        )

    @unittest.skipUnless(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "floating-point environment probe is Linux x86_64 specific",
    )
    def test_generation_after_float_environment_change_fails_closed(self):
        session = ProfileSession(self.pairs)
        selection = session.select([0])
        batch = SequenceBatch(self.targets)
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original = libc.fegetround()
        try:
            self.assertEqual(libc.fesetround(0x400), 0)
            if forward_seam_available():
                candidates = batch._postfilter_forward_selection(
                    selection, 0.02, 1.0, 1.0, True
                )
            else:
                candidates = batch.postfilter_selection(selection)
        finally:
            self.assertEqual(libc.fesetround(original), 0)
            batch.close()
            selection.close()
            session.close()

        state = _candidate_state(candidates)
        self.assertEqual(candidates.candidate_count(0), len(self.targets))
        actions = [
            struct.unpack_from("=IfhBBf", state.postfilter_records, offset)[4]
            for offset in range(
                0, len(state.postfilter_records), _native.POSTFILTER_RESULT_SIZE
            )
        ]
        self.assertEqual(actions, [_native.BIAS_CPU_REQUIRED] * len(self.targets))
        actual = candidates.search(0, self.pipeline())
        expected = self.pipeline().search_hmm(self.pairs[0].hmm, self.targets)
        self.assertEqual(self.hits_bytes(actual), self.hits_bytes(expected))
