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
    from plan7_gpu.adapter import _candidate_state, _pair_state
except ImportError:
    ProfileSelection = None
    ProfileSession = None
    SequenceBatch = None
    load_pressed_profiles = None
    _native = None
    _candidate_state = None
    _pair_state = None


DATA = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


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
    def pipeline(cls, f1=0.02):
        return pyhmmer.plan7.Pipeline(
            cls.alphabet,
            F1=f1,
            E=10.0,
            domE=10.0,
            incE=10.0,
            incdomE=10.0,
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
            self.assertEqual(statistics["profile_count"], 3)
            self.assertEqual(statistics["worker_count"], 3)
            self.assertEqual(statistics["parallel_run_count"], 1)
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
            finally:
                empty.close()
                first.close()
        finally:
            session.close()

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
