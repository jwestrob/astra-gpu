import ctypes
import gc
import hashlib
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
        _profile_session_state,
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
    _profile_session_state = None
    _profile_selection_state = None
    _sequence_state = None

try:
    from plan7_gpu import _pipeline
except ImportError:
    _pipeline = None


DATA = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
JOURNAL_CAPSULE_NAME = b"plan7_gpu._native._continuation_journal_v2"
REAL_PFAM_SOURCE = Path(
    os.environ.get(
        "PLAN7_GPU_TEST_PFAM_SOURCE",
        Path.home() / ".config" / "Astra" / "PFAM" / "Pfam-A.hmm",
    )
)
REAL_PFAM_FIRST1000 = Path(
    os.environ.get(
        "PLAN7_GPU_TEST_PFAM_FIRST1000",
        ROOT / "results" / "datasets" / "PLM2_5.first1000.faa",
    )
)


class ContinuationJournalPrefix(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint16),
        ("header_size", ctypes.c_uint16),
        ("row_size", ctypes.c_uint32),
        ("region_size", ctypes.c_uint32),
        ("compact_result_size", ctypes.c_uint32),
        ("compact_trace_step_size", ctypes.c_uint32),
        ("compact_null2_stride", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint64),
        ("session_id", ctypes.c_uint64),
        ("selection_id", ctypes.c_uint64),
        ("profile_count", ctypes.c_uint64),
        ("postfilter_count", ctypes.c_uint64),
        ("forward_count", ctypes.c_uint64),
        ("row_count", ctypes.c_uint64),
        ("special_count", ctypes.c_uint64),
        ("region_count", ctypes.c_uint64),
        ("compact_result_count", ctypes.c_uint64),
        ("compact_trace_offset_count", ctypes.c_uint64),
        ("compact_trace_count", ctypes.c_uint64),
        ("compact_null2_count", ctypes.c_uint64),
        ("generation_tail_fingerprint", ctypes.c_uint64),
        ("rescore_simple_row_count", ctypes.c_uint64),
        ("rescore_device_result_count", ctypes.c_uint64),
        ("rescore_cpu_required_count", ctypes.c_uint64),
        ("rescore_numeric_fallback_count", ctypes.c_uint64),
        ("rescore_cap_fallback_count", ctypes.c_uint64),
        ("rescore_global_cpu_fallback_count", ctypes.c_uint64),
        ("rescore_compact_output_byte_limit", ctypes.c_uint64),
        ("rescore_compact_output_bytes", ctypes.c_uint64),
        ("generation_f1_bits", ctypes.c_uint64),
        ("generation_f2_bits", ctypes.c_uint64),
        ("generation_f3_bits", ctypes.c_uint64),
        ("rt1_bits", ctypes.c_uint32),
        ("rt2_bits", ctypes.c_uint32),
        ("rt3_bits", ctypes.c_uint32),
        ("guard_band_bits", ctypes.c_uint32),
        ("generation_bias_filter", ctypes.c_uint8),
        ("generation_compact_domains", ctypes.c_uint8),
        ("compact_global_fallback", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 5),
        ("sequence_content_fingerprint", ctypes.c_uint8 * 32),
        ("postfilter_offsets_offset", ctypes.c_uint64),
        ("postfilter_records_offset", ctypes.c_uint64),
        ("forward_offsets_offset", ctypes.c_uint64),
        ("forward_records_offset", ctypes.c_uint64),
        ("forward_special_offsets_offset", ctypes.c_uint64),
        ("profile_offsets_offset", ctypes.c_uint64),
        ("identity_tokens_offset", ctypes.c_uint64),
        ("profile_fingerprints_offset", ctypes.c_uint64),
        ("rows_offset", ctypes.c_uint64),
        ("special_offsets_offset", ctypes.c_uint64),
        ("specials_offset", ctypes.c_uint64),
        ("region_offsets_offset", ctypes.c_uint64),
        ("regions_offset", ctypes.c_uint64),
        ("compact_row_offsets_offset", ctypes.c_uint64),
        ("compact_results_offset", ctypes.c_uint64),
        ("compact_trace_offsets_offset", ctypes.c_uint64),
        ("compact_traces_offset", ctypes.c_uint64),
        ("compact_null2_offset", ctypes.c_uint64),
    ]


def continuation_journal_address(capsule):
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    get_pointer.restype = ctypes.c_void_p
    address = get_pointer(capsule, JOURNAL_CAPSULE_NAME)
    if not address:
        raise RuntimeError("test journal capsule has no storage")
    return address


def refresh_journal_integrity(address, header):
    integrity_offset = header.header_size - ctypes.sizeof(ctypes.c_uint64)
    storage = (ctypes.c_uint8 * header.total_bytes).from_address(address)
    value = 1469598103934665603
    for index in range(integrity_offset):
        value = ((value ^ storage[index]) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for index in range(header.header_size, header.total_bytes):
        value = ((value ^ storage[index]) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    ctypes.c_uint64.from_address(address + integrity_offset).value = value


def tamper_first_journal_sequence_index(capsule):
    address = continuation_journal_address(capsule)
    header = ContinuationJournalPrefix.from_address(address)
    if header.row_count == 0:
        raise RuntimeError("test journal has no rows")
    ctypes.c_uint32.from_address(address + header.rows_offset + 4).value = 0xFFFFFFFF
    refresh_journal_integrity(address, header)


def tamper_first_compact_forward_score(capsule):
    address = continuation_journal_address(capsule)
    header = ContinuationJournalPrefix.from_address(address)
    if header.compact_result_count == 0:
        raise RuntimeError("test journal has no compact results")
    score = ctypes.c_uint32.from_address(
        address + header.compact_results_offset + 36
    )
    score.value ^= 1
    refresh_journal_integrity(address, header)


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


def simple_region_seam_available():
    return _pipeline is not None and _pipeline._simple_regions_seam_available()


def compact_domain_seam_available():
    return _pipeline is not None and _pipeline._compact_domains_seam_available()


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
        hits.write(output, format="domains", header=True)
        return output.getvalue()

    @staticmethod
    def semantic_pipeline_state(hits):
        fields = (
            "Z_setby",
            "domZ_setby",
            "n_past_msv",
            "n_past_bias",
            "n_past_vit",
            "n_past_fwd",
            "pos_past_msv",
            "pos_past_bias",
            "pos_past_vit",
            "pos_past_fwd",
            "mode",
            "W",
        )
        top_fields = (
            "Z",
            "domZ",
            "searched_models",
            "searched_nodes",
            "searched_residues",
            "searched_sequences",
        )
        pipeline = hits.__getstate__()["pipeline"]
        return (
            tuple(pipeline[field] for field in fields),
            tuple(getattr(hits, field) for field in top_fields),
        )


@unittest.skipUnless(_native is not None, "CUDA extension unavailable")
class HostProfileSessionTests(ProfileSessionFixture, unittest.TestCase):
    def test_session_retains_original_profiles_without_database_copies(self):
        with ProfileSession(self.pairs, pack_workers=0) as session:
            state = _profile_session_state(session)
            pair_states = tuple(_pair_state(pair) for pair in self.pairs)
            self.assertTrue(
                all(
                    query is pair_state.hmm
                    for query, pair_state in zip(
                        state.queries, pair_states, strict=True
                    )
                )
            )
            self.assertTrue(
                all(
                    profile is pair_state.optimized_profile
                    for profile, pair_state in zip(
                        state.profiles, pair_states, strict=True
                    )
                )
            )
            self.assertEqual(
                sum(map(len, state.profile_fingerprints)), 32 * len(self.pairs)
            )

    def test_sequence_batch_rejects_virtual_copy_sources(self):
        target = self.targets[0]

        class AliasingSequence:
            alphabet = target.alphabet

            @staticmethod
            def copy():
                return target

        with self.assertRaisesRegex(TypeError, "exact DigitalSequence"):
            SequenceBatch([AliasingSequence()], alphabet=self.alphabet)

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

    def test_chunk_local_pack_matches_eager_snapshot_and_accounts_lifetime(self):
        eager = ProfileSession(self.pairs, pack_workers=0)
        lazy = ProfileSession(
            self.pairs,
            build_workers=3,
            selection_workers=0,
            _chunk_local_pack=True,
        )
        eager_selection = None
        lazy_selection = None
        eager_empty = None
        lazy_empty = None
        try:
            eager_statistics = eager.statistics
            lazy_statistics = lazy.statistics
            self.assertFalse(eager_statistics["chunk_local_pack"])
            self.assertEqual(eager_statistics["profile_pointer_bytes"], 0)
            self.assertEqual(eager_statistics["identity_token_bytes"], 0)
            self.assertEqual(eager_statistics["background_bytes"], 0)
            self.assertTrue(lazy_statistics["chunk_local_pack"])
            self.assertEqual(lazy_statistics["profile_count"], len(self.pairs))
            self.assertEqual(lazy_statistics["build_worker_count"], 0)
            self.assertEqual(lazy_statistics["build_parallel_run_count"], 0)
            self.assertEqual(
                lazy_statistics["profile_pointer_bytes"],
                len(self.pairs) * struct.calcsize("P"),
            )
            self.assertEqual(
                lazy_statistics["identity_token_bytes"],
                len(self.pairs) * struct.calcsize("P"),
            )
            self.assertEqual(lazy_statistics["background_bytes"], 20 * 4)
            self.assertEqual(
                lazy_statistics["host_bytes"],
                lazy_statistics["profile_pointer_bytes"]
                + lazy_statistics["identity_token_bytes"]
                + lazy_statistics["background_bytes"],
            )
            self.assertLess(
                lazy_statistics["host_bytes"], eager_statistics["host_bytes"]
            )
            for field in (
                "ssv_score_bytes",
                "bias_profile_bytes",
                "viterbi_descriptor_bytes",
                "viterbi_emission_bytes",
                "viterbi_transition_bytes",
                "viterbi_exact_rbv_bytes",
                "forward_descriptor_bytes",
                "forward_emission_bytes",
                "forward_transition_bytes",
            ):
                self.assertEqual(lazy_statistics[field], 0, field)

            eager_selection = eager.select([2, 0])
            lazy_selection = lazy.select([2, 0])
            eager_empty = eager.select([])
            lazy_empty = lazy.select([])
            eager_native = _profile_selection_state(eager_selection).native
            lazy_native = _profile_selection_state(lazy_selection).native
            self.assertTrue(
                eager_native._snapshot_equal_for_test(lazy_native)
            )
            self.assertEqual(
                eager_selection.host_bytes, lazy_selection.host_bytes
            )
            self.assertTrue(
                _profile_selection_state(eager_empty).native
                ._snapshot_equal_for_test(
                    _profile_selection_state(lazy_empty).native
                )
            )
            self.assertEqual(lazy.statistics["selection_count"], 2)
            self.assertEqual(
                lazy.statistics["selection_parallel_run_count"], 1
            )

            # Every selection is pointer-free and remains valid after the
            # native session source-pointer table is released.
            lazy.close()
            self.assertTrue(lazy.closed)
            self.assertTrue(
                eager_native._snapshot_equal_for_test(lazy_native)
            )
        finally:
            for selection in (
                lazy_empty,
                eager_empty,
                lazy_selection,
                eager_selection,
            ):
                if selection is not None:
                    selection.close()
            lazy.close()
            eager.close()

    def test_chunk_local_pack_opt_in_is_exact_bool(self):
        for value in (0, 1, None, "yes"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "_chunk_local_pack"):
                    ProfileSession(self.pairs, _chunk_local_pack=value)

    def test_native_chunk_local_session_owns_sources_until_close(self):
        background = pyhmmer.plan7.Background(self.alphabet)
        sources = [
            _pair_state(pair).optimized_profile.copy() for pair in self.pairs
        ]
        native = _native.ProfileSession(
            sources,
            memoryview(background.residue_frequencies),
            0,
            0,
            None,
            True,
        )
        eager = ProfileSession(self.pairs, pack_workers=0)
        native_selection = None
        eager_selection = None
        try:
            del sources
            gc.collect()
            native_selection = native.select([2, 0])
            eager_selection = eager.select([2, 0])
            eager_native = _profile_selection_state(eager_selection).native
            self.assertTrue(
                eager_native._snapshot_equal_for_test(native_selection)
            )
            native.close()
            self.assertTrue(
                eager_native._snapshot_equal_for_test(native_selection)
            )
        finally:
            if native_selection is not None:
                native_selection.close()
            if eager_selection is not None:
                eager_selection.close()
            native.close()
            eager.close()

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
    def test_compact_pair_is_absent_and_frozen_on_stock_abi(self):
        self.assertFalse(compact_domain_seam_available())
        self.assertFalse(compact_domain_seam_available())
        info = _pipeline._continuation_seam_cache_info()["compact_domains"]
        self.assertTrue(info["resolved"])
        self.assertFalse(info["available"])
        self.assertFalse(info["same_dso"])
        self.assertEqual(info["resolutions"], 1)
        self.assertEqual(info["dlopen_calls"], info["dlclose_calls"])
        with self.assertRaisesRegex(RuntimeError, "compact-domain seam"):
            _pipeline._compact_tail_fingerprint_bound(self.pipeline())

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
                    with self.assertRaisesRegex(
                        RuntimeError, "project-private HMMER seams"
                    ):
                        batch._postfilter_forward_selection(
                            selection,
                            0.02,
                            1.0,
                            1.0,
                            True,
                            pipeline=self.pipeline(F1=0.02, F2=1.0, F3=1.0),
                        )


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
@unittest.skipUnless(
    postfilter_seam_available(), "private filter-score seam is unavailable"
)
class CudaProfileSessionTests(ProfileSessionFixture, unittest.TestCase):
    @staticmethod
    def mint_continuation_capsule(
        selection,
        batch,
        options,
        guard=2.0e-4,
        *,
        pipeline=None,
        compact_budget=0,
        matrix_budget=0,
        trace_budget=0,
        test_fault=0,
    ):
        selection_state = _profile_selection_state(selection)
        sequence_state = _sequence_state(batch)
        tail_fingerprint = (
            _pipeline._compact_tail_fingerprint_bound(pipeline)
            if pipeline is not None
            else 0
        )
        return sequence_state.native._postfilter_forward_domain_selection_sealed(
            selection_state.native,
            options["F1"],
            options["F2"],
            options["F3"],
            guard,
            _native.FORWARD_MAX_GATHERED_BYTES,
            False,
            matrix_budget,
            trace_budget,
            compact_budget,
            test_fault,
            tail_fingerprint,
            _return_stage_timings=True,
        )

    @staticmethod
    def seal_continuation_capsule(
        transport, selection, batch, pipeline, options, guard=2.0e-4
    ):
        selection_state = _profile_selection_state(selection)
        sequence_state = _sequence_state(batch)
        capsule, native_stage_timings = transport
        return _pipeline._seal_profile_selection_continuation_bound(
            selection_state.queries,
            selection_state.profiles,
            sequence_state.targets,
            memoryview(sequence_state.residue_offsets).cast("Q"),
            options["F1"],
            memoryview(selection_state.background_fingerprint),
            capsule,
            selection_state.native.identity,
            memoryview(
                selection_state.native._identity_tokens_for_seal()
            ).cast("Q"),
            memoryview(b"".join(selection_state.profile_fingerprints)),
            sequence_state.native_generation,
            memoryview(sequence_state.content_fingerprint),
            pipeline,
            guard,
            native_stage_timings,
        )

    def test_sequence_batch_owns_an_independent_exact_target_copy(self):
        source = self.targets[0].copy()
        original = memoryview(source.sequence).cast("B").tobytes()
        with SequenceBatch([source]) as batch:
            state = _sequence_state(batch)
            source.sequence[0] = (source.sequence[0] + 1) % self.alphabet.K
            self.assertEqual(
                memoryview(state.targets[0].sequence).cast("B").tobytes(),
                original,
            )
            generation, fingerprint = (
                state.native._generation_and_content_for_seal()
            )
            self.assertEqual(generation, state.native_generation)
            self.assertEqual(fingerprint, state.content_fingerprint)

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
        simple_region_seam_available(),
        "private simple-region seam is unavailable",
    )
    def test_fused_domain_selection_is_exact_opaque_and_lifetime_safe(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        before = _native._sealed_journal_transport_statistics()
        session = ProfileSession(self.pairs, pack_workers=1)
        selection = session.select([2, 0])
        batch = SequenceBatch(self.targets)
        legacy = batch._postfilter_forward_selection(
            selection,
            options["F1"],
            options["F2"],
            options["F3"],
            True,
        )
        generation_pipeline = self.pipeline(**options)
        fused = batch._postfilter_forward_selection(
            selection,
            options["F1"],
            options["F2"],
            options["F3"],
            True,
            pipeline=generation_pipeline,
        )
        route_statistics = _pipeline._sealed_continuation_statistics_bound(
            _candidate_state(fused).sealed_postfilter
        )
        native_timings = route_statistics["native_stage_timings"]
        self.assertEqual(native_timings["schema_version"], 1)
        self.assertEqual(native_timings["units"], "milliseconds")
        self.assertEqual(
            native_timings["forward"]["profile_staging_scope"],
            "per-selection database",
        )
        self.assertIn(
            "provenance sealing",
            native_timings["forward"]["call_total_scope"],
        )
        self.assertIn(
            "download_milliseconds",
            native_timings["backward_domain"][
                "post_primary_materialization_native_field"
            ],
        )
        for stage in ("forward", "backward_domain", "domain_rescore"):
            values = native_timings[stage]
            if stage == "domain_rescore" and not route_statistics["compact_enabled"]:
                self.assertIsNone(values)
                continue
            self.assertIsNotNone(values)
            for name, value in values.items():
                if name.endswith("_ms"):
                    self.assertIs(type(value), float)
                    self.assertTrue(math.isfinite(value))
                    self.assertGreaterEqual(value, 0.0)
        forward = native_timings["forward"]
        for name in (
            "run_upload_ms",
            "kernel_ms",
            "classification_ms",
            "gather_ms",
            "download_ms",
            "timed_loop_ms",
        ):
            self.assertGreaterEqual(forward["call_total_ms"] + 0.1, forward[name])
        backward = native_timings["backward_domain"]
        for name in (
            "upload_ms",
            "kernel_ms",
            "post_primary_materialization_ms",
        ):
            self.assertGreaterEqual(backward["total_ms"] + 0.1, backward[name])
        rescore = native_timings["domain_rescore"]
        if rescore is not None:
            for name in ("upload_ms", "kernel_ms", "download_ms"):
                self.assertGreaterEqual(rescore["total_ms"] + 0.1, rescore[name])
        resident_memory = fused.resident_memory
        self.assertIs(type(fused.resident_bytes), int)
        self.assertEqual(
            fused.resident_bytes, resident_memory["resident_bytes"]
        )
        self.assertEqual(resident_memory["owned_device_bytes"], 0)
        self.assertEqual(
            resident_memory["sealed"]["journal_allocation_bytes"],
            route_statistics["journal_bytes"],
        )
        self.assertEqual(
            resident_memory["sealed"]["owned_host_bytes"],
            resident_memory["sealed"]["journal_allocation_bytes"]
            + resident_memory["sealed"]["row_markers_bytes"],
        )
        # Returned records are defensive snapshots, not mutable accounting.
        resident_memory["state_buffers"]["offsets_bytes"] = -1
        native_timings["forward"]["kernel_ms"] = -1.0
        self.assertGreaterEqual(fused.resident_bytes, 0)
        self.assertGreaterEqual(
            _pipeline._sealed_continuation_statistics_bound(
                _candidate_state(fused).sealed_postfilter
            )["native_stage_timings"]["forward"]["kernel_ms"],
            0.0,
        )
        after = _native._sealed_journal_transport_statistics()
        self.assertEqual(after["build_count"], before["build_count"] + 1)
        self.assertGreater(after["payload_bytes"], before["payload_bytes"])
        self.assertEqual(after["duplicate_python_bytes"], 0)
        self.assertEqual(
            route_statistics["row_count"],
            route_statistics["cpu_required_count"]
            + route_statistics["no_region_count"]
            + route_statistics["simple_count"],
        )
        self.assertGreater(route_statistics["cpu_required_count"], 0)
        self.assertGreater(
            route_statistics["no_region_count"]
            + route_statistics["simple_count"],
            0,
        )
        self.assertEqual(
            [fused.candidate_count(row) for row in range(2)],
            [legacy.candidate_count(row) for row in range(2)],
        )
        batch.close()
        selection.close()
        session.close()

        actual_pipeline = self.pipeline(**options)
        expected_pipeline = self.pipeline(**options)
        for reuse in range(2):
            for row, pair in enumerate((self.pairs[2], self.pairs[0])):
                with self.subTest(row=row, reuse=reuse):
                    actual = fused.search(row, actual_pipeline)
                    expected = expected_pipeline.search_hmm(pair.hmm, self.targets)
                    self.assertEqual(
                        self.hits_bytes(actual), self.hits_bytes(expected)
                    )
                    self.assertEqual(
                        self.semantic_pipeline_state(actual),
                        self.semantic_pipeline_state(expected),
                    )

        actual_pipeline = self.pipeline(**options, null2=False)
        expected_pipeline = self.pipeline(**options, null2=False)
        for row, pair in enumerate((self.pairs[2], self.pairs[0])):
            with self.subTest(tail_mismatch="null2", row=row):
                actual = fused.search(row, actual_pipeline)
                expected = expected_pipeline.search_hmm(pair.hmm, self.targets)
                self.assertEqual(self.hits_bytes(actual), self.hits_bytes(expected))
                self.assertEqual(
                    self.semantic_pipeline_state(actual),
                    self.semantic_pipeline_state(expected),
                )

        expected = [
            self.hits_bytes(self.pipeline(**options).search_hmm(pair.hmm, self.targets))
            for pair in (self.pairs[2], self.pairs[0])
        ]
        barrier = threading.Barrier(8)
        failures = []

        def search(worker_index):
            try:
                barrier.wait()
                row = worker_index % 2
                hits = fused.search(row, self.pipeline(**options))
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
        simple_region_seam_available(),
        "private simple-region seam is unavailable",
    )
    def test_generation_ledger_is_opt_in_and_covers_fused_boundaries(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        previous = os.environ.get("PLAN7_GPU_GENERATION_LEDGER")
        os.environ["PLAN7_GPU_GENERATION_LEDGER"] = "1"
        try:
            with ProfileSession(self.pairs, pack_workers=1) as session:
                with session.select([2, 0]) as selection:
                    with SequenceBatch(self.targets) as batch:
                        candidates = batch._postfilter_forward_selection(
                            selection,
                            options["F1"],
                            options["F2"],
                            options["F3"],
                            True,
                            pipeline=self.pipeline(**options),
                        )
                        ledger = _sequence_state(
                            batch
                        ).native.workspace_statistics["generation_ledger"]
                        self.assertEqual(ledger["schema_version"], 1)
                        self.assertTrue(ledger["enabled"])
                        self.assertEqual(ledger["fused_call_count"], 1)
                        for name in (
                            "fused_total_ns",
                            "f1_native_ns",
                            "f1_candidate_mirror_ns",
                            "postfilter_host_prepare_ns",
                            "viterbi_stage_ns",
                            "postfilter_native_ns",
                            "postfilter_materialize_ns",
                            "f2_control_ns",
                            "forward_stage_ns",
                            "forward_native_ns",
                            "backward_native_ns",
                        ):
                            self.assertGreater(ledger[name], 0, name)
                        self.assertGreaterEqual(
                            ledger["fused_total_ns"],
                            sum(
                                value
                                for name, value in ledger.items()
                                if name.endswith("_ns")
                                and name != "fused_total_ns"
                            ),
                        )
                        self.assertIsNotNone(
                            _candidate_state(candidates).sealed_postfilter
                        )
        finally:
            if previous is None:
                os.environ.pop("PLAN7_GPU_GENERATION_LEDGER", None)
            else:
                os.environ["PLAN7_GPU_GENERATION_LEDGER"] = previous

    @unittest.skipUnless(
        simple_region_seam_available(),
        "private simple-region seam is unavailable",
    )
    def test_cpu_domain_ownership_matches_exact_hmmer_continuation(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        previous = os.environ.get("PLAN7_GPU_DOMAIN_OWNERSHIP")
        os.environ["PLAN7_GPU_DOMAIN_OWNERSHIP"] = "cpu"
        try:
            with ProfileSession(self.pairs, pack_workers=1) as session:
                with session.select([2, 0]) as selection:
                    with SequenceBatch(self.targets) as batch:
                        fused = batch._postfilter_forward_selection(
                            selection,
                            options["F1"],
                            options["F2"],
                            options["F3"],
                            True,
                            pipeline=self.pipeline(**options),
                            sparse_journal_v3=True,
                        )
                        routes = _pipeline._sealed_continuation_statistics_bound(
                            _candidate_state(fused).sealed_postfilter
                        )
                        self.assertTrue(
                            _sequence_state(batch).native.workspace_statistics[
                                "cpu_domain_ownership"
                            ]
                        )
                        self.assertGreater(routes["cpu_required_count"], 0)
                        self.assertEqual(routes["no_region_count"], 0)
                        self.assertEqual(routes["simple_count"], 0)
                        for row, pair in enumerate((self.pairs[2], self.pairs[0])):
                            actual_pipeline = self.pipeline(**options)
                            expected_pipeline = self.pipeline(**options)
                            actual = fused.search(row, actual_pipeline)
                            expected = expected_pipeline.search_hmm(
                                pair.hmm, self.targets
                            )
                            self.assertEqual(
                                self.hits_bytes(actual), self.hits_bytes(expected)
                            )
                            self.assertEqual(
                                self.semantic_pipeline_state(actual),
                                self.semantic_pipeline_state(expected),
                            )
        finally:
            if previous is None:
                os.environ.pop("PLAN7_GPU_DOMAIN_OWNERSHIP", None)
            else:
                os.environ["PLAN7_GPU_DOMAIN_OWNERSHIP"] = previous

    @unittest.skipUnless(
        simple_region_seam_available(),
        "private simple-region seam is unavailable",
    )
    def test_cpu_rescore_ownership_matches_exact_hmmer_continuation(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        previous = os.environ.get("PLAN7_GPU_DOMAIN_OWNERSHIP")
        os.environ["PLAN7_GPU_DOMAIN_OWNERSHIP"] = "cpu_rescore"
        try:
            with ProfileSession(self.pairs, pack_workers=1) as session:
                with session.select([2, 0]) as selection:
                    with SequenceBatch(self.targets) as batch:
                        fused = batch._postfilter_forward_selection(
                            selection,
                            options["F1"],
                            options["F2"],
                            options["F3"],
                            True,
                            pipeline=self.pipeline(**options),
                            sparse_journal_v3=True,
                        )
                        workspace = (
                            _sequence_state(batch).native.workspace_statistics
                        )
                        self.assertTrue(workspace["cpu_rescore_ownership"])
                        self.assertFalse(workspace["cpu_domain_ownership"])
                        for row, pair in enumerate((self.pairs[2], self.pairs[0])):
                            actual_pipeline = self.pipeline(**options)
                            expected_pipeline = self.pipeline(**options)
                            actual = fused.search(row, actual_pipeline)
                            expected = expected_pipeline.search_hmm(
                                pair.hmm, self.targets
                            )
                            self.assertEqual(
                                self.hits_bytes(actual), self.hits_bytes(expected)
                            )
                            self.assertEqual(
                                self.semantic_pipeline_state(actual),
                                self.semantic_pipeline_state(expected),
                            )
        finally:
            if previous is None:
                os.environ.pop("PLAN7_GPU_DOMAIN_OWNERSHIP", None)
            else:
                os.environ["PLAN7_GPU_DOMAIN_OWNERSHIP"] = previous

    @unittest.skipUnless(
        simple_region_seam_available(),
        "private simple-region seam is unavailable",
    )
    def test_fused_domain_selection_rejects_pair_and_mode_drift(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        with ProfileSession(self.pairs, pack_workers=0) as session:
            with session.select([0]) as selection:
                with SequenceBatch(self.targets) as batch:
                    selection_state = _profile_selection_state(selection)
                    pair_state = _pair_state(self.pairs[0])
                    original_profiles = selection_state.profiles
                    impostor = pair_state.optimized_profile.copy()
                    impostor.rfv[0, 1] += 0.25
                    selection_state.profiles = (impostor,)
                    pipeline = self.pipeline(**options)
                    try:
                        with self.assertRaisesRegex(
                            ValueError, "selected pair objects differ"
                        ):
                            batch._postfilter_forward_selection(
                                selection,
                                options["F1"],
                                options["F2"],
                                options["F3"],
                                True,
                                pipeline=pipeline,
                            )
                    finally:
                        selection_state.profiles = original_profiles

                    pair_state.optimized_profile.multihit = False
                    try:
                        with self.assertRaisesRegex(ValueError, "local multihit"):
                            batch._postfilter_forward_selection(
                                selection,
                                options["F1"],
                                options["F2"],
                                options["F3"],
                                True,
                                pipeline=pipeline,
                            )
                    finally:
                        pair_state.optimized_profile.multihit = True

                    actual = pipeline.search_hmm(self.pairs[0].hmm, self.targets)
                    expected = self.pipeline(**options).search_hmm(
                        self.pairs[0].hmm, self.targets
                    )
                    self.assertEqual(self.hits_bytes(actual), self.hits_bytes(expected))
                    self.assertEqual(
                        self.semantic_pipeline_state(actual),
                        self.semantic_pipeline_state(expected),
                    )

    @unittest.skipUnless(
        simple_region_seam_available(),
        "private simple-region seam is unavailable",
    )
    def test_fused_capsule_rejects_target_rebinding_and_bad_index(self):
        options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        altered_targets = [sequence.copy() for sequence in self.targets]
        altered_targets[0].sequence[0] = (
            altered_targets[0].sequence[0] + 1
        ) % self.alphabet.K

        with ProfileSession(self.pairs, pack_workers=0) as session:
            with session.select([0]) as selection:
                with SequenceBatch(self.targets) as original_batch:
                    with SequenceBatch(altered_targets) as altered_batch:
                        pipeline = self.pipeline(**options)
                        capsule = self.mint_continuation_capsule(
                            selection, original_batch, options
                        )
                        with self.assertRaisesRegex(
                            ValueError, "target content differs"
                        ):
                            self.seal_continuation_capsule(
                                capsule,
                                selection,
                                altered_batch,
                                pipeline,
                                options,
                            )

                        capsule = self.mint_continuation_capsule(
                            selection, original_batch, options
                        )
                        tamper_first_journal_sequence_index(capsule[0])
                        with self.assertRaisesRegex(
                            ValueError, "Forward identity differs"
                        ):
                            self.seal_continuation_capsule(
                                capsule,
                                selection,
                                original_batch,
                                pipeline,
                                options,
                            )

                        actual = pipeline.search_hmm(
                            self.pairs[0].hmm, self.targets
                        )
                        expected = self.pipeline(**options).search_hmm(
                            self.pairs[0].hmm, self.targets
                        )
                        self.assertEqual(
                            self.hits_bytes(actual), self.hits_bytes(expected)
                        )
                        self.assertEqual(
                            self.semantic_pipeline_state(actual),
                            self.semantic_pipeline_state(expected),
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
        fused_pipeline = None
        try:
            self.assertEqual(libc.fesetround(0x400), 0)
            if simple_region_seam_available():
                fused_pipeline = self.pipeline(F1=0.02, F2=1.0, F3=1.0)
                with self.assertRaisesRegex(RuntimeError, "floating-point"):
                    batch._postfilter_forward_selection(
                        selection,
                        0.02,
                        1.0,
                        1.0,
                        True,
                        pipeline=fused_pipeline,
                    )
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
        if fused_pipeline is not None:
            actual = fused_pipeline.search_hmm(self.pairs[0].hmm, self.targets)
            expected = self.pipeline(F1=0.02, F2=1.0, F3=1.0).search_hmm(
                self.pairs[0].hmm, self.targets
            )
            self.assertEqual(self.hits_bytes(actual), self.hits_bytes(expected))
            self.assertEqual(
                self.semantic_pipeline_state(actual),
                self.semantic_pipeline_state(expected),
            )


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
@unittest.skipUnless(
    compact_domain_seam_available(),
    "private compact-domain seam is unavailable",
)
class CompactDomainProfileSessionTests(ProfileSessionFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="plan7-compact-session-test-"
        )
        data = Path(pyhmmer.__file__).parent / "tests" / "data"
        hmms = []
        for name in ("RREFam.hmm", "Thioesterase.hmm", "KR.hmm", "LuxC.hmm"):
            with pyhmmer.plan7.HMMFile(data / "hmms" / "txt" / name) as source:
                hmms.append(source.read())
        cls.base = Path(cls.temporary.name) / "models"
        pyhmmer.hmmer.hmmpress(hmms, cls.base)
        cls.pairs = load_pressed_profiles(cls.base)
        cls.alphabet = hmms[0].alphabet
        sequences = []
        with pyhmmer.easel.SequenceFile(
            data / "seqs" / "938293.PRJEB85.HG003687.faa",
            digital=True,
            alphabet=cls.alphabet,
        ) as source:
            for sequence in source:
                sequences.append(sequence)
                if len(sequences) == 4:
                    break
        consensus = hmms[1].consensus
        if isinstance(consensus, bytes):
            consensus = consensus.decode()
        consensus = consensus.replace("-", "")
        sequences.append(
            pyhmmer.easel.TextSequence(
                name=b"two-domain-probe",
                sequence=consensus + "X" * 100 + consensus,
            ).digitize(cls.alphabet)
        )
        cls.targets = pyhmmer.easel.DigitalSequenceBlock(
            cls.alphabet, sequences
        )
        cls.options = {"F1": 0.99, "F2": 1.0, "F3": 1.0}
        cls.selection_indices = (3, 1, 0)

    def assert_hits_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_hit, expected_hit in zip(actual, expected, strict=True):
            self.assertEqual(actual_hit.name, expected_hit.name)
            self.assertEqual(actual_hit.accession, expected_hit.accession)
            self.assertEqual(actual_hit.reported, expected_hit.reported)
            self.assertEqual(actual_hit.included, expected_hit.included)
            self.assertAlmostEqual(
                actual_hit.score, expected_hit.score, delta=2.0e-4
            )
            self.assertAlmostEqual(
                actual_hit.bias, expected_hit.bias, delta=2.0e-4
            )
            self.assertEqual(len(actual_hit.domains), len(expected_hit.domains))
            for actual_domain, expected_domain in zip(
                actual_hit.domains, expected_hit.domains, strict=True
            ):
                self.assertEqual(
                    (actual_domain.env_from, actual_domain.env_to),
                    (expected_domain.env_from, expected_domain.env_to),
                )
                self.assertEqual(
                    (
                        actual_domain.alignment.target_from,
                        actual_domain.alignment.target_to,
                        actual_domain.alignment.hmm_from,
                        actual_domain.alignment.hmm_to,
                    ),
                    (
                        expected_domain.alignment.target_from,
                        expected_domain.alignment.target_to,
                        expected_domain.alignment.hmm_from,
                        expected_domain.alignment.hmm_to,
                    ),
                )
                self.assertAlmostEqual(
                    actual_domain.score,
                    expected_domain.score,
                    delta=2.0e-4,
                )
                self.assertAlmostEqual(
                    actual_domain.correction,
                    expected_domain.correction,
                    delta=2.0e-4,
                )

    def assert_batch_matches_cpu(self, candidates, pairs, reuse_count=2):
        actual_pipeline = self.pipeline(**self.options)
        expected_pipeline = self.pipeline(**self.options)
        for _ in range(reuse_count):
            for row, pair in enumerate(pairs):
                actual = candidates.search(row, actual_pipeline)
                expected = expected_pipeline.search_hmm(pair.hmm, self.targets)
                self.assert_hits_close(actual, expected)
                self.assertEqual(
                    self.semantic_pipeline_state(actual),
                    self.semantic_pipeline_state(expected),
                )

    @unittest.skipUnless(
        forward_seam_available() and simple_region_seam_available(),
        "fused continuation seams are unavailable",
    )
    def test_phase0_telemetry_preserves_default_results_and_source_order(self):
        indices = (2, 0)
        pairs = tuple(self.pairs[index] for index in indices)
        with ProfileSession(self.pairs) as session:
            with session.select(indices) as selection:
                with SequenceBatch(self.targets) as batch:
                    ordinary = batch._postfilter_forward_selection(
                        selection,
                        **self.options,
                        bias_filter=True,
                        pipeline=self.pipeline(**self.options),
                    )
                    instrumented = batch._postfilter_forward_selection(
                        selection,
                        **self.options,
                        bias_filter=True,
                        pipeline=self.pipeline(**self.options),
                        telemetry=True,
                    )

        self.assertIsNone(ordinary.generation_statistics)
        generation = instrumented.generation_statistics
        self.assertIs(type(generation), dict)
        self.assertEqual(generation["profile_count"], len(indices))
        self.assertEqual(generation["target_count"], len(self.targets))
        self.assertEqual(
            tuple(profile["profile_index"] for profile in generation["profiles"]),
            tuple(range(len(indices))),
        )
        self.assertEqual(
            [ordinary.candidate_count(row) for row in range(len(indices))],
            [instrumented.candidate_count(row) for row in range(len(indices))],
        )
        for row, pair in enumerate(pairs):
            ordinary_pipeline = self.pipeline(**self.options)
            instrumented_pipeline = self.pipeline(**self.options)
            ordinary_hits = ordinary.search(row, ordinary_pipeline)
            instrumented_hits, continuation = instrumented.search(
                row, instrumented_pipeline, return_telemetry=True
            )
            self.assert_hits_close(instrumented_hits, ordinary_hits)
            self.assertEqual(
                self.semantic_pipeline_state(instrumented_hits),
                self.semantic_pipeline_state(ordinary_hits),
            )
            self.assertEqual(continuation["target_count"], len(self.targets))
            self.assertEqual(
                sum(continuation["routes"].values()), len(self.targets)
            )
            self.assertEqual(instrumented_hits.query.name, pair.hmm.name)

    def test_production_compact_path_is_near_exact_multi_domain_and_reusable(self):
        pairs = tuple(self.pairs[index] for index in self.selection_indices)
        with ProfileSession(self.pairs) as session:
            with session.select(self.selection_indices) as selection:
                with SequenceBatch(self.targets) as batch:
                    legacy = batch._postfilter_forward_selection(
                        selection, **self.options, bias_filter=True
                    )
                    fused = batch._postfilter_forward_selection(
                        selection,
                        **self.options,
                        bias_filter=True,
                        pipeline=self.pipeline(**self.options),
                    )
                    statistics = (
                        _pipeline._sealed_continuation_statistics_bound(
                            _candidate_state(fused).sealed_postfilter
                        )
                    )
                    self.assertEqual(_candidate_state(fused).pairs, pairs)
                    self.assertEqual(
                        [fused.candidate_count(row) for row in range(len(pairs))],
                        [legacy.candidate_count(row) for row in range(len(pairs))],
                    )

        self.assertTrue(statistics["compact_enabled"])
        self.assertGreater(statistics["compact_simple_row_count"], 0)
        self.assertGreater(statistics["compact_device_result_count"], 0)
        self.assertEqual(
            statistics["compact_device_result_count"]
            + statistics["compact_cpu_required_count"],
            statistics["compact_cap_fallback_count"]
            + statistics["compact_numeric_fallback_count"]
            + statistics["compact_device_result_count"],
        )
        consumption = {
            "attempt_count": 0,
            "accepted_count": 0,
            "invalid_retry_count": 0,
            "threshold_retry_count": 0,
        }
        diagnostic_pipeline = self.pipeline(**self.options)
        sealed = _candidate_state(fused).sealed_postfilter
        for row in range(len(pairs)):
            _, row_consumption = (
                _pipeline._search_hmm_sealed_postfilter_bound(
                    sealed,
                    row,
                    diagnostic_pipeline,
                    True,
                )
            )
            for name in consumption:
                consumption[name] += row_consumption[name]
        self.assertGreater(consumption["attempt_count"], 0)
        self.assertEqual(
            consumption["accepted_count"], consumption["attempt_count"]
        )
        self.assertEqual(consumption["invalid_retry_count"], 0)
        self.assertEqual(consumption["threshold_retry_count"], 0)
        self.assert_batch_matches_cpu(fused, pairs)

        thioesterase_row = self.selection_indices.index(1)
        multi = fused.search(
            thioesterase_row, self.pipeline(**self.options)
        )
        probe = next(
            hit for hit in multi if hit.name == "two-domain-probe"
        )
        self.assertGreaterEqual(len(probe.domains), 2)

        cache = _pipeline._continuation_seam_cache_info()["compact_domains"]
        self.assertTrue(cache["available"])
        self.assertTrue(cache["same_dso"])
        self.assertEqual(cache["resolutions"], 1)
        self.assertEqual(cache["dlopen_calls"], cache["dlclose_calls"])

    @unittest.skipUnless(
        REAL_PFAM_SOURCE.is_file() and REAL_PFAM_FIRST1000.is_file(),
        "real PFAM/first1000 regression inputs are unavailable",
    )
    def test_real_pfam_first_rejected_row_is_compact_accepted(self):
        with tempfile.TemporaryDirectory(
            prefix="plan7-real-pfam-compact-test-"
        ) as temporary:
            with REAL_PFAM_FIRST1000.open("rb") as source:
                self.assertEqual(
                    hashlib.file_digest(source, "sha256").hexdigest(),
                    "b835fa20310971a507f5067f7136291cf2a4f5671e7ad64cf503c258cf24db2b",
                )
            with pyhmmer.plan7.HMMFile(REAL_PFAM_SOURCE) as source:
                hmms = [source.read() for _ in range(13)]
            self.assertTrue(all(hmm is not None for hmm in hmms))
            self.assertEqual(hmms[12].name, "2-Hacid_dh_C")

            base = Path(temporary) / "first13"
            pyhmmer.hmmer.hmmpress(hmms, base)
            pairs = load_pressed_profiles(base)
            alphabet = hmms[0].alphabet
            with pyhmmer.easel.SequenceFile(
                REAL_PFAM_FIRST1000,
                digital=True,
                alphabet=alphabet,
            ) as source:
                targets = source.read_block()
            self.assertEqual(len(targets), 1000)
            self.assertEqual(
                targets[56].name,
                "PLM2_5_b1_jun17_scaffold_0_57",
            )

            options = {
                "F1": 0.02,
                "F2": 0.001,
                "F3": 0.00001,
                "bit_cutoffs": "gathering",
            }
            with ProfileSession(pairs, pack_workers=1) as session:
                with session.select(range(13)) as selection:
                    with SequenceBatch(targets) as batch:
                        candidates = batch._postfilter_forward_selection(
                            selection,
                            options["F1"],
                            options["F2"],
                            options["F3"],
                            True,
                            pipeline=pyhmmer.plan7.Pipeline(
                                alphabet, **options
                            ),
                        )
                        sealed = _candidate_state(candidates).sealed_postfilter
                        actual, consumption = (
                            _pipeline._search_hmm_sealed_postfilter_bound(
                                sealed,
                                12,
                                pyhmmer.plan7.Pipeline(alphabet, **options),
                                True,
                            )
                        )

            self.assertGreater(consumption["attempt_count"], 0)
            self.assertEqual(
                consumption["accepted_count"],
                consumption["attempt_count"],
            )
            self.assertEqual(consumption["invalid_retry_count"], 0)
            self.assertEqual(consumption["threshold_retry_count"], 0)
            self.assertEqual(consumption["first_attempt"], (0, 12, 56, 1))
            expected = pyhmmer.plan7.Pipeline(
                alphabet, **options
            ).search_hmm(pairs[12].hmm, targets)
            self.assert_hits_close(actual, expected)
            self.assertEqual(
                self.semantic_pipeline_state(actual),
                self.semantic_pipeline_state(expected),
            )
            hit = next(
                hit
                for hit in actual
                if hit.name == "PLM2_5_b1_jun17_scaffold_0_57"
            )
            self.assertEqual(len(hit.domains), 1)

    def test_threshold_adjacent_device_row_retries_full_pipeline_exactly(self):
        pair = self.pairs[0]
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.targets[-1]]
        )
        baseline = self.pipeline(**self.options).search_hmm(pair.hmm, targets)
        self.assertEqual([hit.name for hit in baseline], ["two-domain-probe"])
        self.assertGreaterEqual(len(baseline[0].domains), 2)

        cases = (
            ("target_score", {"T": baseline[0].score}),
            (
                "dynamic_domain_evalue",
                {"domE": baseline[0].domains[0].i_evalue},
            ),
        )
        for label, threshold in cases:
            with self.subTest(label=label):
                options = {**self.options, **threshold}
                with ProfileSession(self.pairs) as session:
                    with session.select([0]) as selection:
                        with SequenceBatch(targets) as batch:
                            candidates = batch._postfilter_forward_selection(
                                selection,
                                **self.options,
                                bias_filter=True,
                                pipeline=self.pipeline(**options),
                            )
                            statistics = (
                                _pipeline._sealed_continuation_statistics_bound(
                                    _candidate_state(candidates).sealed_postfilter
                                )
                            )

                self.assertTrue(statistics["compact_enabled"])
                self.assertGreater(
                    statistics["compact_device_result_count"], 0
                )
                actual = candidates.search(0, self.pipeline(**options))
                expected = self.pipeline(**options).search_hmm(pair.hmm, targets)
                self.assertEqual(self.hits_bytes(actual), self.hits_bytes(expected))
                self.assertEqual(
                    self.semantic_pipeline_state(actual),
                    self.semantic_pipeline_state(expected),
                )

    def test_global_cap_and_numeric_rows_fall_back_without_missed_calls(self):
        pairs = tuple(self.pairs[index] for index in self.selection_indices)
        with ProfileSession(self.pairs) as session:
            with session.select(self.selection_indices) as selection:
                with SequenceBatch(self.targets) as batch:
                    capped = batch._postfilter_forward_selection(
                        selection,
                        **self.options,
                        bias_filter=True,
                        pipeline=self.pipeline(**self.options),
                        _rescore_compact_byte_budget=1,
                    )
                    capped_statistics = (
                        _pipeline._sealed_continuation_statistics_bound(
                            _candidate_state(capped).sealed_postfilter
                        )
                    )
                    numeric = batch._postfilter_forward_selection(
                        selection,
                        **self.options,
                        bias_filter=True,
                        pipeline=self.pipeline(**self.options),
                        _rescore_test_fault=(
                            _native.BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE
                        ),
                    )
                    numeric_statistics = (
                        _pipeline._sealed_continuation_statistics_bound(
                            _candidate_state(numeric).sealed_postfilter
                        )
                    )

        self.assertGreater(
            capped_statistics["compact_global_cpu_fallback_count"], 0
        )
        self.assertEqual(capped_statistics["compact_device_result_count"], 0)
        self.assertEqual(
            capped_statistics["compact_cap_fallback_count"],
            capped_statistics["compact_global_cpu_fallback_count"],
        )
        self.assertGreater(
            numeric_statistics["compact_numeric_fallback_count"], 0
        )
        self.assertGreater(numeric_statistics["cpu_required_count"], 0)
        self.assert_batch_matches_cpu(capped, pairs, reuse_count=1)
        self.assert_batch_matches_cpu(numeric, pairs, reuse_count=1)

    def test_compact_hash_tail_and_selection_cross_binding_fail_closed(self):
        with ProfileSession(self.pairs) as session:
            with session.select(self.selection_indices) as selection:
                with session.select((1, 3, 0)) as other_selection:
                    with SequenceBatch(self.targets) as batch:
                        generation_pipeline = self.pipeline(**self.options)
                        capsule = CudaProfileSessionTests.mint_continuation_capsule(
                            selection,
                            batch,
                            self.options,
                            pipeline=generation_pipeline,
                        )
                        tamper_first_compact_forward_score(capsule[0])
                        with self.assertRaisesRegex(
                            ValueError, "compact hashes differ"
                        ):
                            CudaProfileSessionTests.seal_continuation_capsule(
                                capsule,
                                selection,
                                batch,
                                generation_pipeline,
                                self.options,
                            )

                        capsule = CudaProfileSessionTests.mint_continuation_capsule(
                            selection,
                            batch,
                            self.options,
                            pipeline=generation_pipeline,
                        )
                        tail_drift = self.pipeline(
                            **self.options, null2=False
                        )
                        with self.assertRaisesRegex(
                            ValueError, "compact provenance differs"
                        ):
                            CudaProfileSessionTests.seal_continuation_capsule(
                                capsule,
                                selection,
                                batch,
                                tail_drift,
                                self.options,
                            )

                        capsule = CudaProfileSessionTests.mint_continuation_capsule(
                            selection,
                            batch,
                            self.options,
                            pipeline=generation_pipeline,
                        )
                        with self.assertRaisesRegex(
                            ValueError, "selection identity differs"
                        ):
                            CudaProfileSessionTests.seal_continuation_capsule(
                                capsule,
                                other_selection,
                                batch,
                                generation_pipeline,
                                self.options,
                            )

                        actual = generation_pipeline.search_hmm(
                            self.pairs[1].hmm, self.targets
                        )
                        expected = self.pipeline(**self.options).search_hmm(
                            self.pairs[1].hmm, self.targets
                        )
                        self.assertEqual(
                            self.hits_bytes(actual), self.hits_bytes(expected)
                        )
                        self.assertEqual(
                            self.semantic_pipeline_state(actual),
                            self.semantic_pipeline_state(expected),
                        )
