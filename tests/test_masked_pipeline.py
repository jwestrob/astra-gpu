import importlib.util
import ctypes
import gc
import io
import json
import math
import pickle
import subprocess
import struct
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

import pyhmmer
from pyhmmer.errors import AlphabetMismatch, MissingCutoffs


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "python" / "plan7_gpu"
HMM_FIXTURE = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
HMM_MISMATCH_FIXTURE = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "fn3.hmm"
FASTA_FIXTURE = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins45.fa"


def _load_pipeline_extension():
    """Load the private extension without importing the CUDA package facade."""
    paths = sorted(PACKAGE_DIR.glob("_pipeline*.so"))
    if not paths:
        return None
    spec = importlib.util.spec_from_file_location("_pipeline", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pipeline = _load_pipeline_extension()


@unittest.skipUnless(_pipeline is not None, "masked pipeline extension unavailable")
class MaskedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if array("I").itemsize != 4:
            raise unittest.SkipTest("native unsigned int is not uint32")

        cls._temporary = tempfile.TemporaryDirectory(prefix="plan7-pipeline-test-")
        pressed_base = Path(cls._temporary.name) / "globins"

        source_hmms = []
        for path in (HMM_FIXTURE, HMM_MISMATCH_FIXTURE):
            with pyhmmer.plan7.HMMFile(path) as hmm_file:
                source_hmms.extend(hmm_file)
        pyhmmer.hmmer.hmmpress(source_hmms, pressed_base)

        with pyhmmer.plan7.HMMFile(pressed_base) as hmm_file:
            cls.hmms = list(hmm_file)
        with pyhmmer.plan7.HMMPressedFile(pressed_base) as pressed_file:
            cls.optimized_profiles = list(pressed_file)

        cls.alphabet = cls.hmms[0].alphabet
        with pyhmmer.easel.SequenceFile(
            FASTA_FIXTURE, digital=True, alphabet=cls.alphabet
        ) as sequence_file:
            cls.sequences = sequence_file.read_block()

    @classmethod
    def tearDownClass(cls):
        cls._temporary.cleanup()

    @staticmethod
    def pipeline(**options):
        defaults = {"E": 10.0, "domE": 10.0, "incE": 10.0, "incdomE": 10.0}
        defaults.update(options)
        return pyhmmer.plan7.Pipeline(MaskedPipelineTests.alphabet, **defaults)

    @staticmethod
    def candidates(*indexes):
        return array("I", indexes)

    @staticmethod
    def residue_offsets(sequences):
        offsets = array("Q", [0])
        for sequence in sequences:
            offsets.append(offsets[-1] + len(sequence))
        return offsets

    @staticmethod
    def bias_record(index, filtersc, numerator, status, action):
        return struct.pack("=IfhBB", index, filtersc, numerator, status, action)

    @staticmethod
    def postfilter_record(index, filtersc, numerator, status, action, vfsc):
        return struct.pack("=IfhBBf", index, filtersc, numerator, status, action, vfsc)

    @staticmethod
    def forward_record(index, fwdsc, status, action, reserved=0):
        return struct.pack("=IfBBH", index, fwdsc, status, action, reserved)

    @staticmethod
    def continuation_row(
        profile,
        index,
        *,
        domain_status,
        domain_route,
        uncertain=0,
        region_count=0,
        multidomain=0,
        compact_count=0,
        compact_route=0,
        usc=-3.0,
        filtersc=0.0,
        vfsc=0.0,
        fwdsc=-7.0,
    ):
        return struct.pack(
            "=IIffffffIII8BIB3sI",
            profile,
            index,
            usc,
            filtersc,
            vfsc,
            fwdsc,
            0.0,
            0.0,
            uncertain,
            region_count,
            multidomain,
            0,
            2,
            0,
            2,
            domain_status,
            domain_route,
            0,
            0,
            compact_count,
            compact_route,
            b"\0\0\0",
            0,
        )

    @staticmethod
    def compact_result(
        row,
        profile,
        index,
        begin,
        end,
        *,
        status,
        action,
        forward_score=None,
        oa_score=None,
    ):
        score = 0.0 if action == 1 else math.nan
        if forward_score is not None:
            score = forward_score
        oa = score if oa_score is None else oa_score
        return struct.pack(
            "=9I5f4BI",
            row,
            profile,
            index,
            begin,
            end,
            begin,
            end,
            1,
            1,
            score,
            score,
            oa,
            0.0 if action == 1 else math.nan,
            0.0,
            status,
            action,
            0,
            0,
            0,
        )

    @staticmethod
    def double_bits(value):
        return struct.unpack("=Q", struct.pack("=d", value))[0]

    @staticmethod
    def background_fingerprint(background):
        return memoryview(background.residue_frequencies).cast("B").tobytes() + (
            struct.pack("=f", background.omega)
        )

    def seal_v3_fixture(
        self,
        sequences,
        postfilter_records=b"",
        *,
        f1=1.0,
        f2=1.0,
        f3=1.0,
        forward_records=None,
        special_offsets=None,
        specials=None,
    ):
        postfilter_count = len(postfilter_records) // 16
        residue_offsets = self.residue_offsets(sequences)
        background = pyhmmer.plan7.Background(self.alphabet)
        options = {
            "generation_f2_bits": self.double_bits(f2),
            "generation_f3_bits": self.double_bits(f3),
            "generation_bias_filter": True,
        }
        if forward_records is not None:
            forward_count = len(forward_records) // 12
            options.update(
                forward_records=forward_records,
                forward_offsets=array("Q", [0, forward_count]),
                special_offsets=special_offsets,
                specials=specials,
                expected_forward_indices=array(
                    "I",
                    (
                        struct.unpack_from("=I", forward_records, offset)[0]
                        for offset in range(0, len(forward_records), 12)
                    ),
                ),
            )
        return _pipeline._seal_postfilter_batch_bound(
            (self.hmms[0],),
            (self.optimized_profiles[0].copy(),),
            sequences,
            postfilter_records,
            array("Q", [0, postfilter_count]),
            residue_offsets,
            f1,
            self.background_fingerprint(background),
            **options,
        )

    def seal_v2_terminal_fixture(
        self,
        sequences,
        postfilter_records=b"",
        *,
        pipeline=None,
        f1=0.02,
        sparse_journal_v3=False,
    ):
        """Mint a complete v2-backed seal with no Forward/domain rows."""
        if pipeline is None:
            pipeline = self.pipeline(F1=f1, F2=0.0, F3=0.0)
        postfilter_count = len(postfilter_records) // 16
        return _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            (self.hmms[0],),
            (self.optimized_profiles[0].copy(),),
            sequences,
            self.residue_offsets(sequences),
            f1,
            self.background_fingerprint(pipeline.background),
            postfilter_records,
            array("Q", [0, postfilter_count]),
            b"",
            array("Q", [0, 0]),
            array("Q", [0]),
            array("f"),
            array("Q", [0, 0]),
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("f"),
            pipeline,
            2.0e-4,
            sparse_journal_v3,
        )

    def seal_v2_f3_reject_fixture(self, sequences, target, *, pipeline):
        """Mint a v2-backed row whose exact Forward score rejects at F3."""
        postfilter = self.postfilter_record(
            target, 0.0, 0, 0, 2, math.inf
        )
        forward = self.forward_record(target, -1.0e30, 0, 1)
        return _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            (self.hmms[0],),
            (self.optimized_profiles[0].copy(),),
            sequences,
            self.residue_offsets(sequences),
            1.0,
            self.background_fingerprint(pipeline.background),
            postfilter,
            array("Q", [0, 1]),
            forward,
            array("Q", [0, 1]),
            array("Q", [0, 0]),
            array("f"),
            array("Q", [0, 0]),
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("f"),
            pipeline,
            2.0e-4,
        )

    @staticmethod
    def table_bytes(hits, format):
        output = io.BytesIO()
        hits.write(output, format=format, header=True)
        return output.getvalue()

    @staticmethod
    def semantic_state(hits):
        alignment_fields = (
            "hmm_accession",
            "hmm_from",
            "hmm_length",
            "hmm_name",
            "hmm_sequence",
            "hmm_to",
            "identity_sequence",
            "posterior_probabilities",
            "target_from",
            "target_length",
            "target_name",
            "target_sequence",
            "target_to",
        )
        domain_fields = (
            "bias",
            "c_evalue",
            "correction",
            "env_from",
            "env_to",
            "envelope_score",
            "i_evalue",
            "included",
            "pvalue",
            "reported",
            "score",
            "strand",
        )
        hit_fields = (
            "accession",
            "bias",
            "description",
            "dropped",
            "duplicate",
            "evalue",
            "included",
            "length",
            "name",
            "new",
            "pre_score",
            "pvalue",
            "reported",
            "score",
            "sum_score",
        )
        top_fields = (
            "E",
            "T",
            "Z",
            "bit_cutoffs",
            "block_length",
            "domE",
            "domT",
            "domZ",
            "incE",
            "incT",
            "incdomE",
            "incdomT",
            "long_targets",
            "mode",
            "searched_models",
            "searched_nodes",
            "searched_residues",
            "searched_sequences",
            "strand",
        )
        pipeline_fields = (
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
        pipeline_state = hits.__getstate__()["pipeline"]

        hit_states = []
        for hit in hits:
            domains = []
            for domain in hit.domains:
                alignment = domain.alignment
                domains.append(
                    (
                        tuple(getattr(domain, name) for name in domain_fields),
                        None
                        if alignment is None
                        else tuple(
                            getattr(alignment, name) for name in alignment_fields
                        ),
                    )
                )
            hit_states.append(
                (tuple(getattr(hit, name) for name in hit_fields), tuple(domains))
            )
        return (
            tuple(getattr(hits, name) for name in top_fields),
            tuple(pipeline_state[name] for name in pipeline_fields),
            hits.is_sorted(by="key"),
            len(hits.reported),
            len(hits.included),
            tuple(hit_states),
        )

    def assert_exact_hits(self, expected, actual):
        self.assertEqual(self.semantic_state(actual), self.semantic_state(expected))
        self.assertEqual(
            self.table_bytes(actual, "targets"),
            self.table_bytes(expected, "targets"),
        )
        self.assertEqual(
            self.table_bytes(actual, "domains"),
            self.table_bytes(expected, "domains"),
        )

    def test_pressed_hmm_and_optimized_profiles_are_lockstep(self):
        self.assertEqual(len(self.hmms), len(self.optimized_profiles))
        self.assertGreater(len(self.hmms), 1)
        for hmm, optimized in zip(self.hmms, self.optimized_profiles, strict=True):
            self.assertEqual(hmm.alphabet, optimized.alphabet)
            self.assertEqual(hmm.M, optimized.M)
            self.assertEqual(hmm.name, optimized.name)
            self.assertEqual(hmm.accession, optimized.accession)

    def test_all_candidates_match_complete_pyhmmer_tophits_and_tables(self):
        hmm = self.hmms[0]
        optimized = self.optimized_profiles[0].copy()
        expected = self.pipeline().search_hmm(hmm, self.sequences)
        actual = _pipeline._search_hmm_candidates(
            self.pipeline(),
            hmm,
            optimized,
            self.sequences,
            self.candidates(*range(len(self.sequences))),
        )

        self.assertIs(actual.query, hmm)
        self.assert_exact_hits(expected, actual)

    def test_all_rejects_keep_database_accounting_and_automatic_z(self):
        hmm = self.hmms[0]
        hits = _pipeline._search_hmm_candidates(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(),
        )

        self.assertIs(hits.query, hmm)
        self.assertEqual(len(hits), 0)
        self.assertEqual(hits.searched_models, 1)
        self.assertEqual(hits.searched_nodes, hmm.M)
        self.assertEqual(hits.searched_sequences, len(self.sequences))
        self.assertEqual(hits.searched_residues, self.sequences.total_length())
        self.assertEqual(hits.Z, float(len(self.sequences)))
        self.assertEqual(hits.domZ, 0.0)

    def test_sparse_candidates_match_cpu_subset_but_count_full_database(self):
        hmm = self.hmms[0]
        indexes = (1, 7, len(self.sequences) - 1)
        subset = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[index] for index in indexes]
        )
        database_size = len(self.sequences)

        expected = self.pipeline(Z=database_size).search_hmm(hmm, subset)
        actual = _pipeline._search_hmm_candidates(
            self.pipeline(Z=database_size),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(*indexes),
        )

        self.assertEqual(
            self.table_bytes(actual, "targets"),
            self.table_bytes(expected, "targets"),
        )
        self.assertEqual(
            self.table_bytes(actual, "domains"),
            self.table_bytes(expected, "domains"),
        )
        self.assertEqual(actual.searched_sequences, database_size)
        self.assertEqual(actual.searched_residues, self.sequences.total_length())
        self.assertEqual(actual.Z, float(database_size))

    def test_bound_residue_prefix_matches_raw_path_and_validates_inputs(self):
        hmm = self.hmms[0]
        offsets = self.residue_offsets(self.sequences)
        masks = (
            (),
            (1, 7, len(self.sequences) - 1),
            tuple(range(len(self.sequences))),
        )

        for indexes in masks:
            with self.subTest(indexes=indexes):
                expected = _pipeline._search_hmm_candidates(
                    self.pipeline(),
                    hmm,
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    self.candidates(*indexes),
                )
                actual = _pipeline._search_hmm_candidates_bound(
                    self.pipeline(),
                    hmm,
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    self.candidates(*indexes),
                    offsets,
                )
                self.assert_exact_hits(expected, actual)

        invalid = array("Q", offsets)
        invalid[2] += 1
        pipeline = self.pipeline()
        with self.assertRaisesRegex(ValueError, "differs from target length"):
            _pipeline._search_hmm_candidates_bound(
                pipeline,
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                self.candidates(1),
                invalid,
            )
        hits = _pipeline._search_hmm_candidates_bound(
            pipeline,
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(),
            offsets,
        )
        self.assertEqual(hits.searched_models, 1)

    def test_all_cpu_bias_records_match_sparse_candidate_pipeline(self):
        hmm = self.hmms[0]
        offsets = self.residue_offsets(self.sequences)
        indexes = (1, 7, len(self.sequences) - 1)
        records = b"".join(
            self.bias_record(index, math.nan, 0, 173, 0) for index in indexes
        )
        expected = _pipeline._search_hmm_candidates_bound(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(*indexes),
            offsets,
        )
        actual = _pipeline._search_hmm_bias_bound(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            records,
            offsets,
        )
        self.assert_exact_hits(expected, actual)

    def test_all_cpu_postfilter_records_match_sparse_candidate_pipeline(self):
        hmm = self.hmms[0]
        offsets = self.residue_offsets(self.sequences)
        indexes = (1, 7, len(self.sequences) - 1)
        records = b"".join(
            self.postfilter_record(index, math.nan, 0, 173, 0, math.nan)
            for index in indexes
        )
        expected = _pipeline._search_hmm_candidates_bound(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(*indexes),
            offsets,
        )
        actual = _pipeline._search_hmm_postfilter_bound(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            records,
            offsets,
        )
        self.assert_exact_hits(expected, actual)

    def test_full_msv_reject_postfilter_row_is_safely_accounted(self):
        offsets = self.residue_offsets(self.sequences)
        record = self.postfilter_record(7, math.nan, -12, 0, 1, math.nan)
        expected = _pipeline._search_hmm_candidates_bound(
            self.pipeline(),
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(),
            offsets,
        )
        actual = _pipeline._search_hmm_postfilter_bound(
            self.pipeline(),
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            record,
            offsets,
        )
        self.assert_exact_hits(expected, actual)

    def test_direct_postfilter_symbol_failure_precedes_pipeline_mutation(self):
        if _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is available")
        offsets = self.residue_offsets(self.sequences)
        pipeline = self.pipeline()
        direct = self.postfilter_record(0, 0.0, 0, 0, 1, 0.0)

        with self.assertRaisesRegex(RuntimeError, "p7_PipelineFromFilterScores"):
            _pipeline._search_hmm_postfilter_bound(
                pipeline,
                self.hmms[0],
                self.optimized_profiles[0].copy(),
                self.sequences,
                direct,
                offsets,
            )

        hits = _pipeline._search_hmm_postfilter_bound(
            pipeline,
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            b"",
            offsets,
        )
        self.assertEqual(hits.searched_models, 1)

    def test_forward_score_seam_tracks_private_hmmer(self):
        if _pipeline._filter_scores_seam_available():
            self.assertTrue(_pipeline._filter_and_forward_scores_seam_available())
        else:
            self.assertFalse(_pipeline._filter_and_forward_scores_seam_available())

    def test_compact_resolver_rejects_legacy_abi_pair(self):
        library = (
            Path(pyhmmer.__file__).resolve().parent.parent
            / "pyhmmer.libs"
            / "liblibhmmer.so"
        )
        hmmer = ctypes.CDLL(str(library))
        legacy = all(
            hasattr(hmmer, symbol)
            for symbol in (
                "p7_PipelineFromFilterForwardAndCompactDomains",
                "p7_pipeline_CompactTailFingerprint",
            )
        )
        current = all(
            hasattr(hmmer, symbol)
            for symbol in (
                "p7_PipelineFromFilterForwardAndCompactDomainsV2",
                "p7_pipeline_CompactTailFingerprintV2",
            )
        )
        if not legacy or current:
            self.skipTest("loaded HMMER is not the legacy-only compact ABI")
        self.assertFalse(_pipeline._compact_domains_seam_available())
        state = _pipeline._continuation_seam_cache_info()["compact_domains"]
        self.assertTrue(state["resolved"])
        self.assertFalse(state["available"])
        self.assertFalse(state["same_dso"])

    def test_continuation_seam_cache_handles_concurrent_first_use(self):
        extension = sorted(PACKAGE_DIR.glob("_pipeline*.so"))[0]
        code = r"""
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

spec = importlib.util.spec_from_file_location("_pipeline", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
before = module._continuation_seam_cache_info()
assert all(not state["resolved"] for state in before.values()), before

barrier = Barrier(32)
def first_use(index):
    barrier.wait()
    if index % 4 == 0:
        return "filter", module._filter_scores_seam_available()
    if index % 4 == 1:
        return "forward", module._filter_and_forward_scores_seam_available()
    if index % 4 == 2:
        return "simple_regions", module._simple_regions_seam_available()
    return "compact_domains", module._compact_domains_seam_available()

with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(first_use, range(32)))
after_first = module._continuation_seam_cache_info()
for _ in range(1000):
    assert module._filter_scores_seam_available() == after_first["filter"]["available"]
    assert module._filter_and_forward_scores_seam_available() == after_first["forward"]["available"]
    assert module._simple_regions_seam_available() == after_first["simple_regions"]["available"]
    assert module._compact_domains_seam_available() == after_first["compact_domains"]["available"]
after_repeat = module._continuation_seam_cache_info()
assert after_repeat == after_first, (after_first, after_repeat)
assert all(value == after_first[name]["available"] for name, value in results)
for state in after_repeat.values():
    assert state["resolved"]
    assert state["resolutions"] == 1
    assert state["dlopen_calls"] == 1
    assert state["dlclose_calls"] == 1
    assert state["same_dso"] == state["available"]
print(json.dumps(after_repeat, sort_keys=True))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code, str(extension)],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        child = json.loads(completed.stdout)
        self.assertEqual(
            child["filter"]["available"],
            _pipeline._filter_scores_seam_available(),
        )
        self.assertEqual(
            child["forward"]["available"],
            _pipeline._filter_and_forward_scores_seam_available(),
        )
        self.assertEqual(
            child["simple_regions"]["available"],
            _pipeline._simple_regions_seam_available(),
        )
        self.assertEqual(
            child["compact_domains"]["available"],
            _pipeline._compact_domains_seam_available(),
        )

    def test_forward_selector_keeps_exact_f2_survivors(self):
        target = 1
        records = self.postfilter_record(target, 0.0, 0, 0, 2, 0.0)
        postfilter_offsets = array("Q", [0, 1])
        residue_offsets = self.residue_offsets(self.sequences)

        selected = _pipeline._select_forward_inputs_bound(
            [self.optimized_profiles[0]],
            records,
            postfilter_offsets,
            residue_offsets,
            1.0,
        )
        self.assertEqual(list(selected[0]), [0, 1])
        self.assertEqual(list(selected[1]), [target])
        self.assertEqual(len(selected[2]), 1)

    def test_journal_v3_certifies_exact_terminal_stages(self):
        if not _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is unavailable")
        if not _pipeline._filter_and_forward_scores_seam_available():
            self.skipTest("private Forward-score seam is unavailable")

        records = b"".join(
            (
                # Target 0 is absent: rejected before the retained F1 stream.
                self.postfilter_record(1, math.nan, -12, 0, 1, math.nan),
                self.postfilter_record(2, 10.0, 0, 0, 1, 0.0),
                self.postfilter_record(3, 0.0, 0, 0, 2, -1.0e30),
                self.postfilter_record(4, 0.0, 0, 0, 2, math.inf),
                self.postfilter_record(5, math.nan, 0, 173, 0, math.nan),
            )
        )
        forward = self.forward_record(4, -1.0e30, 0, 1)
        sealed = self.seal_v3_fixture(
            self.sequences,
            records,
            f1=0.02,
            f2=0.0,
            f3=0.0,
            forward_records=forward,
            special_offsets=array("Q", [0, 0]),
            specials=array("f"),
        )
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, include_details=True
        )

        self.assertEqual(summary["schema_version"], 3)
        self.assertEqual(summary["exception_count"], 1)
        self.assertEqual(
            summary["stage_counts"],
            {
                "before_f1": len(self.sequences) - 5,
                "raw_f1_reject": 1,
                "bias_reject": 1,
                "f2_reject": 1,
                "f3_reject": 1,
                "domain_no_regions": 0,
            },
        )
        self.assertEqual(
            summary["promotion_deltas"],
            {
                "n_past_msv": 3,
                "n_past_bias": 2,
                "n_past_vit": 1,
                "n_past_fwd": 0,
            },
        )
        profile = summary["profiles"][0]
        self.assertEqual(
            [(item["begin"], item["end"]) for item in profile["certificates"]],
            [(0, 5), (6, len(self.sequences))],
        )
        self.assertEqual(
            [item["sequence_index"] for item in profile["exceptions"]], [5]
        )
        self.assertEqual(summary["exception_routes"]["full_pipeline"], 1)

    def test_journal_v3_exception_routes_copy_only_required_payloads(self):
        if not _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is unavailable")
        if not _pipeline._filter_and_forward_scores_seam_available():
            self.skipTest("private Forward-score seam is unavailable")

        matrix_target = 3
        matrix_count = 6 * (len(self.sequences[matrix_target]) + 1)
        records = b"".join(
            (
                self.postfilter_record(1, math.nan, 0, 173, 0, math.nan),
                self.postfilter_record(2, 0.0, 0, 0, 2, 0.0),
                self.postfilter_record(matrix_target, 0.0, 0, 0, 2, 0.0),
            )
        )
        forward = self.forward_record(matrix_target, -7.0, 0, 2)
        sealed = self.seal_v3_fixture(
            self.sequences,
            records,
            f2=1.0,
            f3=1.0,
            forward_records=forward,
            special_offsets=array("Q", [0, matrix_count]),
            specials=array("f", [0.0]) * matrix_count,
        )
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, include_details=True
        )
        self.assertEqual(
            summary["exception_routes"],
            {
                "full_pipeline": 1,
                "filter_scores": 1,
                "forward_scores": 1,
                "simple_regions": 0,
                "compact_domains": 0,
            },
        )
        self.assertEqual(summary["payload_counts"]["specials"], matrix_count)
        exceptions = summary["profiles"][0]["exceptions"]
        self.assertEqual([item["sequence_index"] for item in exceptions], [1, 2, 3])
        self.assertEqual([item["special_count"] for item in exceptions], [0, 0, matrix_count])

    def test_journal_v3_rejects_forward_rows_that_fail_exact_f2(self):
        target = 2
        records = self.postfilter_record(
            target, 0.0, 0, 0, 2, -1.0e30
        )
        forward = self.forward_record(target, -7.0, 0, 1)
        sealed = self.seal_v3_fixture(
            self.sequences,
            records,
            f2=0.0,
            f3=1.0,
            forward_records=forward,
            special_offsets=array("Q", [0, 0]),
            specials=array("f"),
        )
        with self.assertRaisesRegex(ValueError, "exact F2 survivor"):
            _pipeline._plan_continuation_journal_v3_bound(sealed)

    def test_journal_v3_host_source_requires_bias_enabled_generation(self):
        background = pyhmmer.plan7.Background(self.alphabet)
        sealed = _pipeline._seal_postfilter_batch_bound(
            (self.hmms[0],),
            (self.optimized_profiles[0].copy(),),
            self.sequences,
            b"",
            array("Q", [0, 0]),
            self.residue_offsets(self.sequences),
            1.0,
            self.background_fingerprint(background),
            generation_f2_bits=self.double_bits(1.0),
            generation_f3_bits=self.double_bits(1.0),
            generation_bias_filter=False,
        )
        with self.assertRaisesRegex(ValueError, "bias-enabled"):
            _pipeline._plan_continuation_journal_v3_bound(sealed)

    def test_journal_v3_serializes_validated_v2_domain_bundle(self):
        if not _pipeline._simple_regions_seam_available():
            self.skipTest("private simple-region seam is unavailable")
        if not _pipeline._compact_domains_seam_available():
            self.skipTest("private compact-domain seam is unavailable")

        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet,
            [self.sequences[index % len(self.sequences)] for index in range(80)],
        )
        profile_sequences = ((0, 13, 31, 32, 33, 67), (1, 7))
        postfilter = b"".join(
            self.postfilter_record(index, 0.0, 0, 0, 2, 0.0)
            for indexes in profile_sequences
            for index in indexes
        )
        forward = b"".join(
            self.forward_record(index, -7.0, 0, 2)
            for indexes in profile_sequences
            for index in indexes
        )
        row_offsets = array("Q", [0, 6, 8])
        special_offsets = array("Q", [0])
        specials = array("f")
        for indexes in profile_sequences:
            for index in indexes:
                count = 6 * (len(targets[index]) + 1)
                specials.extend([0.0] * count)
                special_offsets.append(len(specials))

        domain_rows = b"".join(
            (
                self.continuation_row(
                    0, 0, domain_status=0, domain_route=1
                ),
                self.continuation_row(
                    0,
                    13,
                    domain_status=0,
                    domain_route=2,
                    region_count=1,
                    compact_count=1,
                    compact_route=1,
                ),
                self.continuation_row(
                    0, 31, domain_status=16, domain_route=0
                ),
                self.continuation_row(
                    0,
                    32,
                    domain_status=0,
                    domain_route=0,
                    uncertain=1,
                ),
                self.continuation_row(
                    0,
                    33,
                    domain_status=0,
                    domain_route=0,
                    multidomain=1,
                ),
                self.continuation_row(
                    0,
                    67,
                    domain_status=0,
                    domain_route=2,
                    region_count=1,
                    compact_count=1,
                    compact_route=2,
                ),
                self.continuation_row(
                    1, 1, domain_status=0, domain_route=1
                ),
                self.continuation_row(
                    1,
                    7,
                    domain_status=0,
                    domain_route=2,
                    region_count=1,
                    compact_count=1,
                    compact_route=2,
                ),
            )
        )
        region_offsets = array("Q", [0, 0, 1, 1, 1, 1, 2, 2, 3])
        regions = b"".join(
            struct.pack("=II", begin, end)
            for begin, end in ((1, 5), (2, 6), (3, 7))
        )
        compact_results = b"".join(
            (
                self.compact_result(
                    1, 0, 13, 1, 5, status=75, action=0
                ),
                self.compact_result(
                    5, 0, 67, 2, 6, status=0, action=1
                ),
                self.compact_result(
                    7, 1, 7, 3, 7, status=0, action=1
                ),
            )
        )
        compact_trace_offsets = array("Q", [0, 0, 1, 2])
        compact_traces = b"".join(
            (
                struct.pack("=IIfB3x", 2, 1, 1.0, 1),
                struct.pack("=IIfB3x", 3, 1, 1.0, 1),
            )
        )
        compact_null2 = array(
            "f", [math.nan] * 29 + [1.0] * 29 + [1.0] * 29
        )
        pipeline = self.pipeline(F1=1.0, F2=1.0, F3=1.0)
        profiles = tuple(
            self.optimized_profiles[index].copy() for index in range(2)
        )
        sealed = _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            tuple(self.hmms[:2]),
            profiles,
            targets,
            self.residue_offsets(targets),
            1.0,
            self.background_fingerprint(pipeline.background),
            postfilter,
            row_offsets,
            forward,
            row_offsets,
            special_offsets,
            specials,
            row_offsets,
            domain_rows,
            region_offsets,
            regions,
            region_offsets,
            compact_results,
            compact_trace_offsets,
            compact_traces,
            compact_null2,
            pipeline,
            2.0e-4,
        )
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, include_details=True
        )

        self.assertEqual(summary["source_kind"], "v2_journal")
        self.assertTrue(summary["options_complete"])
        self.assertGreater(summary["source_v2_bytes"], 0)
        self.assertGreater(summary["source_v2_integrity_tag"], 0)
        self.assertGreater(summary["session_id"], 0)
        self.assertGreater(summary["selection_id"], 0)
        self.assertGreater(summary["batch_generation"], 0)
        self.assertGreater(summary["generation_tail_fingerprint"], 0)
        self.assertEqual(summary["dense_postfilter_count"], 8)
        self.assertEqual(summary["dense_forward_count"], 8)
        self.assertEqual(summary["dense_domain_count"], 8)
        self.assertEqual(summary["stage_counts"]["domain_no_regions"], 2)
        self.assertEqual(summary["exception_count"], 6)
        self.assertEqual(
            summary["exception_routes"],
            {
                "full_pipeline": 0,
                "filter_scores": 0,
                "forward_scores": 3,
                "simple_regions": 1,
                "compact_domains": 2,
            },
        )
        self.assertEqual(summary["payload_counts"]["regions"], 3)
        self.assertEqual(summary["payload_counts"]["compact_results"], 2)
        self.assertEqual(summary["payload_counts"]["compact_traces"], 2)
        self.assertEqual(summary["payload_counts"]["compact_null2"], 58)
        self.assertEqual(summary["compact_trace_offsets"], (0, 1, 2))

        first, second = summary["profiles"]
        self.assertEqual((first["flags"], second["flags"]), (3, 3))
        self.assertNotEqual(first["identity_token"], second["identity_token"])
        self.assertEqual(
            tuple(
                sum(item["no_region"] for item in profile["certificates"])
                for profile in (first, second)
            ),
            (1, 1),
        )
        self.assertEqual(first["source_postfilter_span"], (0, 6))
        self.assertEqual(first["source_forward_span"], (0, 6))
        self.assertEqual(first["source_domain_span"], (0, 6))
        self.assertEqual(second["source_postfilter_span"], (6, 8))
        self.assertEqual(second["source_forward_span"], (6, 8))
        self.assertEqual(second["source_domain_span"], (6, 8))
        self.assertEqual(
            [item["sequence_index"] for item in first["exceptions"]],
            [13, 31, 32, 33, 67],
        )
        self.assertEqual(
            [item["sequence_index"] for item in second["exceptions"]], [7]
        )
        all_exceptions = first["exceptions"] + second["exceptions"]
        self.assertTrue(
            all(item["preconditions"] & 0x01 for item in all_exceptions)
        )
        by_sequence = {
            (item["source_domain_index"], item["sequence_index"]): item
            for item in all_exceptions
        }
        for source_row, sequence in ((2, 31), (3, 32), (4, 33)):
            item = by_sequence[(source_row, sequence)]
            self.assertEqual((item["source_stage"], item["route"]), (9, 3))
            self.assertEqual(item["payload_flags"], 0x0F)
        simple = by_sequence[(1, 13)]
        self.assertEqual((simple["source_stage"], simple["route"]), (10, 4))
        self.assertEqual(simple["payload_flags"], 0x17)
        compact = [
            item for item in all_exceptions if item["compact_result_count"]
        ]
        self.assertEqual(
            [item["compact_result_begin"] for item in compact], [0, 1]
        )
        self.assertEqual(
            [item["compact_trace_begin"] for item in compact], [0, 1]
        )
        self.assertEqual(
            [item["compact_null2_begin"] for item in compact], [0, 29]
        )
        self.assertTrue(all(item["special_count"] > 0 for item in compact))
        self.assertTrue(all(item["region_count"] == 1 for item in compact))
        self.assertTrue(all(item["payload_flags"] == 0x3F for item in compact))
        self.assertTrue(
            all(item["compact_trace_count"] == 1 for item in compact)
        )
        self.assertTrue(
            all(item["compact_null2_count"] == 29 for item in compact)
        )

    def test_ga_census_certifies_only_complete_compact_target_bounds(self):
        if not _pipeline._simple_regions_seam_available():
            self.skipTest("private simple-region seam is unavailable")
        if not _pipeline._compact_domains_seam_available():
            self.skipTest("private compact-domain seam is unavailable")

        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[0], self.sequences[1]]
        )
        query = self.hmms[0].copy()
        profile = self.optimized_profiles[0].copy()
        query.cutoffs.gathering = (50.0, 20.0)
        profile.cutoffs.gathering = (50.0, 20.0)
        options = {
            "F1": 1.0,
            "F2": 1.0,
            "F3": 1.0,
            "bit_cutoffs": "gathering",
        }
        generation = self.pipeline(**options)
        row_offsets = array("Q", [0, 2])
        postfilter = b"".join(
            self.postfilter_record(index, 0.0, 0, 0, 2, 0.0)
            for index in range(2)
        )
        forward = b"".join(
            self.forward_record(index, -100.0, 0, 2)
            for index in range(2)
        )
        special_offsets = array("Q", [0])
        specials = array("f")
        for target in targets:
            specials.extend([0.0] * (6 * (len(target) + 1)))
            special_offsets.append(len(specials))
        domain_rows = b"".join(
            self.continuation_row(
                0,
                index,
                domain_status=0,
                domain_route=2,
                region_count=1,
                compact_count=1,
                compact_route=2,
                fwdsc=-100.0,
            )
            for index in range(2)
        )
        region_offsets = array("Q", [0, 1, 2])
        regions = b"".join(struct.pack("=II", 1, 1) for _ in range(2))
        compact_results = b"".join(
            (
                self.compact_result(
                    0,
                    0,
                    0,
                    1,
                    1,
                    status=0,
                    action=1,
                    forward_score=-100.0,
                    oa_score=1.0,
                ),
                self.compact_result(
                    1,
                    0,
                    1,
                    1,
                    1,
                    status=0,
                    action=1,
                    forward_score=100.0,
                    oa_score=1.0,
                ),
            )
        )
        trace_offsets = array("Q", [0, 1, 2])
        traces = b"".join(
            struct.pack("=IIfB3x", 1, 1, 1.0, 1) for _ in range(2)
        )
        null2 = array("f", [1.0] * 58)
        sealed = _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            (query,),
            (profile,),
            targets,
            self.residue_offsets(targets),
            1.0,
            self.background_fingerprint(generation.background),
            postfilter,
            row_offsets,
            forward,
            row_offsets,
            special_offsets,
            specials,
            row_offsets,
            domain_rows,
            region_offsets,
            regions,
            region_offsets,
            compact_results,
            trace_offsets,
            traces,
            null2,
            generation,
            2.0e-4,
            True,
        )
        census_pipeline = self.pipeline(**options)
        before = _pipeline._semantic_pipeline_state_fingerprint_bound(
            census_pipeline
        )
        census = _pipeline._sealed_ga_cutoff_census_bound(
            sealed, 0, census_pipeline, include_indices=True
        )

        self.assertEqual(census["evaluable_target_rows"], 2)
        self.assertEqual(census["whole_forward_upper_below_ga_count"], 2)
        self.assertEqual(census["reconstruction_upper_below_ga_count"], 1)
        self.assertEqual(census["certified_target_reject_count"], 1)
        self.assertEqual(census["certified_target_indices"], (0,))
        self.assertEqual(census["domain_upper_below_ga_count"], 1)
        self.assertEqual(census["compact_region_count"], 2)
        self.assertEqual(census["native_temporary_bytes"], 0)
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(
                census_pipeline
            ),
            before,
        )

    def test_journal_v3_partitions_first_last_consecutive_and_large_gaps(self):
        def cpu(index):
            return self.postfilter_record(index, math.nan, 0, 173, 0, math.nan)

        cases = (
            ((0,), [(0, 0), (1, len(self.sequences))]),
            (
                (len(self.sequences) - 1,),
                [(0, len(self.sequences) - 1), (len(self.sequences), len(self.sequences))],
            ),
            ((5, 6), [(0, 5), (6, 6), (7, len(self.sequences))]),
            ((), [(0, len(self.sequences))]),
        )
        for indexes, expected_spans in cases:
            with self.subTest(indexes=indexes):
                sealed = self.seal_v3_fixture(
                    self.sequences, b"".join(cpu(index) for index in indexes)
                )
                capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
                summary = _pipeline._validate_continuation_journal_v3_bound(
                    capsule, sealed, include_details=True
                )
                certificates = summary["profiles"][0]["certificates"]
                self.assertEqual(
                    [(item["begin"], item["end"]) for item in certificates],
                    expected_spans,
                )
                self.assertEqual(summary["exception_count"], len(indexes))

        large_count = 4098
        large = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[0]] * large_count
        )
        sealed = self.seal_v3_fixture(large, cpu(2048) + cpu(2049))
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, include_details=True
        )
        certificates = summary["profiles"][0]["certificates"]
        self.assertEqual(
            [(item["begin"], item["end"]) for item in certificates],
            [(0, 2048), (2049, 2049), (2050, large_count)],
        )
        self.assertEqual(
            sum(item["target_delta"] for item in certificates),
            large_count - 2,
        )

    def test_journal_v3_handles_empty_targets_and_empty_database(self):
        empty = pyhmmer.easel.TextSequence(name=b"empty", sequence="").digitize(
            self.alphabet
        )
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [empty, self.sequences[0], empty]
        )
        sealed = self.seal_v3_fixture(targets)
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, include_details=True
        )
        certificate = summary["profiles"][0]["certificates"][0]
        self.assertEqual((certificate["begin"], certificate["end"]), (0, 3))
        self.assertEqual(certificate["residue_delta"], len(self.sequences[0]))

        empty_database = pyhmmer.easel.DigitalSequenceBlock(self.alphabet)
        sealed = self.seal_v3_fixture(empty_database)
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, include_details=True
        )
        certificate = summary["profiles"][0]["certificates"][0]
        self.assertEqual(
            (certificate["begin"], certificate["end"], certificate["residue_delta"]),
            (0, 0, 0),
        )

    def test_journal_v3_corruption_overflow_and_one_shot_fail_safely(self):
        sealed = self.seal_v3_fixture(self.sequences)
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        summary = _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.restype = ctypes.c_void_p
        get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
        address = get_pointer(
            capsule, b"plan7_gpu._pipeline._continuation_journal_v3"
        )
        self.assertTrue(address)

        total_bytes = ctypes.c_uint64.from_address(
            address + summary["total_bytes_offset"]
        )
        original_total = total_bytes.value
        total_bytes.value = (1 << 64) - 1
        with self.assertRaisesRegex(ValueError, "ABI header"):
            _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)
        total_bytes.value = original_total

        integrity = ctypes.c_uint64.from_address(
            address + summary["integrity_offset"]
        )
        original_integrity = integrity.value
        integrity.value ^= 1
        with self.assertRaisesRegex(ValueError, "integrity"):
            _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)
        integrity.value = original_integrity

        # Both failures precede any source/pipeline mutation; the original seal
        # still produces and validates an independent packet.
        second = _pipeline._plan_continuation_journal_v3_bound(sealed)
        _pipeline._validate_continuation_journal_v3_bound(second, sealed)
        _pipeline._validate_continuation_journal_v3_bound(
            capsule, sealed, consume=True
        )
        with self.assertRaisesRegex(TypeError, "invalid or consumed"):
            _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

    def test_journal_v3_sparse_dual_matches_dense_gap_and_tail_accounting(self):
        terminal = b"".join(
            (
                self.postfilter_record(1, math.nan, -12, 0, 1, math.nan),
                self.postfilter_record(2, 10.0, 0, 0, 1, 0.0),
                self.postfilter_record(3, 0.0, 0, 0, 2, -1.0e30),
            )
        )
        last = len(self.sequences) - 1
        cpu = lambda index: self.postfilter_record(
            index, math.nan, 0, 173, 0, math.nan
        )
        empty = pyhmmer.easel.TextSequence(
            name=b"empty-final", sequence=""
        ).digitize(self.alphabet)
        final_empty = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[0], empty]
        )
        large = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[0]] * 4098
        )
        cases = (
            ("no-exceptions", self.sequences, terminal),
            ("first", self.sequences, cpu(0)),
            ("last", self.sequences, cpu(last)),
            ("consecutive", self.sequences, cpu(5) + cpu(6)),
            ("first-last", self.sequences, cpu(0) + cpu(last)),
            ("tail-config-empty-target", final_empty, cpu(0)),
            ("large-prefix-tail", large, cpu(2048) + cpu(2049)),
            (
                "empty-database",
                pyhmmer.easel.DigitalSequenceBlock(self.alphabet),
                b"",
            ),
        )

        for name, targets, records in cases:
            with self.subTest(name=name):
                generation = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
                sealed = self.seal_v2_terminal_fixture(
                    targets, records, pipeline=generation
                )
                capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
                dense = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
                sparse = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
                audit = _pipeline._audit_continuation_journal_v3_bound(
                    capsule, sealed, dense, sparse
                )
                self.assertTrue(audit["equal"])
                self.assertEqual(len(audit["rows"]), 1)
                row = audit["rows"][0]
                self.assertTrue(row["semantic"]["pipeline"]["equal"])
                self.assertTrue(row["semantic"]["tophits"]["equal"])
                self.assertEqual(len(row["targets_sha256"]), 64)
                self.assertTrue(row["route_reconciliation"]["equal"])
                self.assertEqual(
                    row["certificate"]["omitted_targets"]
                    + row["certificate"]["exception_count"],
                    len(targets),
                )
                with self.assertRaisesRegex(TypeError, "invalid or consumed"):
                    _pipeline._validate_continuation_journal_v3_bound(
                        capsule, sealed
                    )

    def test_journal_v3_sparse_preflight_failures_do_not_mutate_or_claim(self):
        generation = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
        sealed = self.seal_v2_terminal_fixture(
            self.sequences, pipeline=generation
        )

        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        mismatched = self.pipeline(F1=0.03, F2=0.0, F3=0.0)
        before = _pipeline._semantic_pipeline_state_fingerprint_bound(mismatched)
        with self.assertRaisesRegex(ValueError, "options differ"):
            _pipeline._consume_continuation_journal_v3_bound(
                capsule, sealed, mismatched
            )
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(mismatched),
            before,
        )
        _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        overflow = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
        previous = _pipeline._v3_test_set_pipeline_counter_bound(
            overflow, "nres", (1 << 64) - 1
        )
        before = _pipeline._semantic_pipeline_state_fingerprint_bound(overflow)
        with self.assertRaisesRegex(OverflowError, "counter prestate"):
            _pipeline._consume_continuation_journal_v3_bound(
                capsule, sealed, overflow
            )
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(overflow),
            before,
        )
        _pipeline._v3_test_set_pipeline_counter_bound(
            overflow, "nres", previous
        )
        _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        shared = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
        before = _pipeline._semantic_pipeline_state_fingerprint_bound(shared)
        with self.assertRaisesRegex(ValueError, "must be independent"):
            _pipeline._audit_continuation_journal_v3_bound(
                capsule, sealed, shared, shared
            )
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(shared),
            before,
        )
        _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        altered_rounding = self.pipeline(F1=0.02, F2=0.0, F3=0.0)
        before = _pipeline._semantic_pipeline_state_fingerprint_bound(
            altered_rounding
        )
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original_rounding = libc.fegetround()
        try:
            self.assertEqual(libc.fesetround(0x400), 0)
            self.assertFalse(
                _pipeline._v3_host_float_environment_attested_bound()
            )
            with self.assertRaisesRegex(RuntimeError, "floating-point"):
                _pipeline._consume_continuation_journal_v3_bound(
                    capsule, sealed, altered_rounding
                )
        finally:
            self.assertEqual(libc.fesetround(original_rounding), 0)
        self.assertTrue(_pipeline._v3_host_float_environment_attested_bound())
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(
                altered_rounding
            ),
            before,
        )
        _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

    def test_journal_v3_sparse_direct_consumer_returns_exact_evidence(self):
        records = b"".join(
            (
                self.postfilter_record(1, math.nan, -12, 0, 1, math.nan),
                self.postfilter_record(5, math.nan, 0, 173, 0, math.nan),
            )
        )
        options = {"F1": 0.02, "F2": 0.0, "F3": 0.0}
        generation = self.pipeline(**options)
        sealed = self.seal_v2_terminal_fixture(
            self.sequences, records, pipeline=generation
        )
        expected = _pipeline._search_hmm_sealed_postfilter_bound(
            sealed, 0, self.pipeline(**options)
        )
        sparse_pipeline = self.pipeline(**options)
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        result = _pipeline._consume_continuation_journal_v3_bound(
            capsule, sealed, sparse_pipeline
        )

        self.assertEqual(result["schema_version"], 3)
        self.assertGreater(result["packet_bytes"], 0)
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(len(result["profiles"]), 1)
        self.assertEqual(len(result["rows"]), 1)
        self.assert_exact_hits(expected, result["hits"][0])
        row = result["rows"][0]
        self.assertEqual(row["cpu_pipeline_count"], 1)
        self.assertEqual(row["certificate"]["stages"]["raw_f1"], 1)
        self.assertEqual(
            row["certificate"]["omitted_targets"]
            + row["certificate"]["exception_count"],
            len(self.sequences),
        )
        with self.assertRaisesRegex(TypeError, "invalid or consumed"):
            _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

    def test_seal_owned_sparse_v3_is_opt_in_reusable_and_fail_safe(self):
        records = b"".join(
            (
                self.postfilter_record(1, math.nan, -12, 0, 1, math.nan),
                self.postfilter_record(5, math.nan, 0, 173, 0, math.nan),
            )
        )
        options = {"F1": 0.02, "F2": 0.0, "F3": 0.0}
        dense = self.seal_v2_terminal_fixture(
            self.sequences, records, pipeline=self.pipeline(**options)
        )
        sparse = self.seal_v2_terminal_fixture(
            self.sequences,
            records,
            pipeline=self.pipeline(**options),
            sparse_journal_v3=True,
        )
        self.assertFalse(
            _pipeline._sealed_sparse_journal_v3_enabled_bound(dense)
        )
        self.assertTrue(
            _pipeline._sealed_sparse_journal_v3_enabled_bound(sparse)
        )
        work_hints = _pipeline._sealed_continuation_work_hints_bound(sparse)
        self.assertEqual(len(work_hints), 1)
        self.assertGreater(work_hints[0], 0)

        expected = _pipeline._search_hmm_sealed_postfilter_bound(
            dense, 0, self.pipeline(**options)
        )
        first, telemetry = (
            _pipeline._search_hmm_sealed_sparse_journal_v3_bound(
                sparse,
                0,
                self.pipeline(**options),
                _return_route_statistics=True,
            )
        )
        second = _pipeline._search_hmm_sealed_sparse_journal_v3_bound(
            sparse, 0, self.pipeline(**options)
        )
        self.assert_exact_hits(expected, first)
        self.assert_exact_hits(expected, second)
        self.assertEqual(telemetry["path"], "journal")
        self.assertGreaterEqual(telemetry["wall_ns"], 0)
        preparation = _pipeline._sealed_continuation_statistics_bound(
            sparse
        )["sparse_journal_v3"]
        self.assertTrue(preparation["enabled"])
        self.assertGreater(preparation["packet_bytes"], 0)
        self.assertGreaterEqual(preparation["planning_ns"], 0)
        self.assertGreaterEqual(preparation["validation_ns"], 0)
        self.assertEqual(preparation["planner_source_scan_count"], 2)
        self.assertEqual(preparation["separate_decision_scan_count"], 1)
        self.assertEqual(
            _pipeline._sealed_resident_memory_bound(sparse)[
                "sparse_journal_v3_bytes"
            ],
            preparation["packet_bytes"],
        )

        row_offsets = array("Q", [0, 0, 0])
        multi = _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            tuple(self.hmms[:2]),
            tuple(profile.copy() for profile in self.optimized_profiles[:2]),
            self.sequences,
            self.residue_offsets(self.sequences),
            options["F1"],
            self.background_fingerprint(self.pipeline(**options).background),
            b"",
            row_offsets,
            b"",
            row_offsets,
            array("Q", [0]),
            array("f"),
            row_offsets,
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("f"),
            self.pipeline(**options),
            2.0e-4,
            True,
        )
        expected_row = _pipeline._search_hmm_sealed_postfilter_bound(
            multi, 1, self.pipeline(**options)
        )
        actual_row = _pipeline._search_hmm_sealed_sparse_journal_v3_bound(
            multi, 1, self.pipeline(**options)
        )
        self.assert_exact_hits(expected_row, actual_row)
        self.assertEqual(actual_row.searched_models, 1)

        mismatch = self.pipeline(F1=0.5, F2=0.0, F3=0.0)
        before = _pipeline._semantic_pipeline_state_fingerprint_bound(mismatch)
        with self.assertRaisesRegex(ValueError, "options differ"):
            _pipeline._search_hmm_sealed_sparse_journal_v3_bound(
                sparse, 0, mismatch
            )
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(mismatch),
            before,
        )

    def test_sparse_v3_profile_shards_merge_exactly(self):
        indexes = (1, 5, 6, len(self.sequences) - 2)
        records = b"".join(
            self.postfilter_record(
                index, math.nan, 0, 173, 0, math.nan
            )
            for index in indexes
        )
        options = {"F1": 1.0, "F2": 1.0, "F3": 1.0}
        sealed = self.seal_v2_terminal_fixture(
            self.sequences,
            records,
            pipeline=self.pipeline(**options),
            f1=1.0,
            sparse_journal_v3=True,
        )

        expected = _pipeline._search_hmm_sealed_sparse_journal_v3_bound(
            sealed, 0, self.pipeline(**options)
        )
        self.assertGreater(len(expected), 0)
        first = _pipeline._search_hmm_sealed_sparse_journal_v3_shard_bound(
            sealed, 0, 0, 2, self.pipeline(**options)
        )
        last = _pipeline._search_hmm_sealed_sparse_journal_v3_shard_bound(
            sealed, 0, 2, 4, self.pipeline(**options)
        )
        actual = last.merge(first)

        self.assert_exact_hits(expected, actual)
        self.assertEqual(actual.searched_models, 1)
        self.assertEqual(actual.searched_nodes, self.hmms[0].M)
        self.assertEqual(actual.searched_sequences, len(self.sequences))
        self.assertEqual(actual.searched_residues, self.sequences.total_length())
        self.assertEqual(actual.Z, float(len(self.sequences)))

        with self.assertRaisesRegex(ValueError, "deterministic reseeding"):
            _pipeline._search_hmm_sealed_sparse_journal_v3_shard_bound(
                sealed,
                0,
                0,
                2,
                self.pipeline(seed=0, **options),
            )

    def test_journal_v3_postclaim_failure_stays_consumed(self):
        options = {
            "F1": 0.02,
            "F2": 0.0,
            "F3": 0.0,
            "bit_cutoffs": "gathering",
        }
        generation = self.pipeline(**options)
        sealed = self.seal_v2_terminal_fixture(
            self.sequences, pipeline=generation
        )
        capsule = _pipeline._plan_continuation_journal_v3_bound(sealed)
        with self.assertRaises(MissingCutoffs):
            _pipeline._consume_continuation_journal_v3_bound(
                capsule, sealed, self.pipeline(**options)
            )
        with self.assertRaisesRegex(TypeError, "invalid or consumed"):
            _pipeline._validate_continuation_journal_v3_bound(capsule, sealed)

    def test_journal_v3_sparse_dual_certifies_exact_f3_reject(self):
        generation = self.pipeline(F1=1.0, F2=1.0, F3=0.0)
        sealed = self.seal_v2_f3_reject_fixture(
            self.sequences, 3, pipeline=generation
        )
        audit = _pipeline._audit_continuation_journal_v3_bound(
            _pipeline._plan_continuation_journal_v3_bound(sealed),
            sealed,
            self.pipeline(F1=1.0, F2=1.0, F3=0.0),
            self.pipeline(F1=1.0, F2=1.0, F3=0.0),
        )
        row = audit["rows"][0]
        self.assertTrue(audit["equal"])
        self.assertEqual(row["certificate"]["stages"]["f3"], 1)
        self.assertEqual(row["dense"]["forward_continuation_count"], 1)
        self.assertEqual(row["sparse"]["forward_continuation_count"], 0)
        self.assertTrue(row["route_reconciliation"]["equal"])

    def test_journal_v3_sparse_dual_certifies_domain_no_region(self):
        target = 3
        row_offsets = array("Q", [0, 1])
        special_count = 6 * (len(self.sequences[target]) + 1)
        options = {"F1": 1.0, "F2": 1.0, "F3": 1.0}
        generation = self.pipeline(**options)
        sealed = _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            (self.hmms[0],),
            (self.optimized_profiles[0].copy(),),
            self.sequences,
            self.residue_offsets(self.sequences),
            1.0,
            self.background_fingerprint(generation.background),
            self.postfilter_record(target, 0.0, 0, 0, 2, 0.0),
            row_offsets,
            self.forward_record(target, -7.0, 0, 2),
            row_offsets,
            array("Q", [0, special_count]),
            array("f", [0.0]) * special_count,
            row_offsets,
            self.continuation_row(
                0, target, domain_status=0, domain_route=1
            ),
            array("Q", [0, 0]),
            b"",
            array("Q", [0, 0]),
            b"",
            array("Q", [0]),
            b"",
            array("f"),
            generation,
            2.0e-4,
        )
        audit = _pipeline._audit_continuation_journal_v3_bound(
            _pipeline._plan_continuation_journal_v3_bound(sealed),
            sealed,
            self.pipeline(**options),
            self.pipeline(**options),
        )
        row = audit["rows"][0]
        self.assertTrue(audit["equal"])
        self.assertEqual(row["certificate"]["stages"]["no_region"], 1)
        self.assertEqual(row["dense"]["simple_continuation_count"], 1)
        self.assertEqual(row["sparse"]["simple_continuation_count"], 0)
        self.assertTrue(row["route_reconciliation"]["equal"])

    def test_journal_v3_sparse_dual_handles_multiple_profiles(self):
        options = {"F1": 0.02, "F2": 0.0, "F3": 0.0}
        generation = self.pipeline(**options)
        row_offsets = array("Q", [0, 0, 0])
        sealed = _pipeline._seal_continuation_journal_v2_test_fixture_bound(
            tuple(self.hmms[:2]),
            tuple(profile.copy() for profile in self.optimized_profiles[:2]),
            self.sequences,
            self.residue_offsets(self.sequences),
            options["F1"],
            self.background_fingerprint(generation.background),
            b"",
            row_offsets,
            b"",
            row_offsets,
            array("Q", [0]),
            array("f"),
            row_offsets,
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("Q", [0]),
            b"",
            array("f"),
            generation,
            2.0e-4,
        )
        audit = _pipeline._audit_continuation_journal_v3_bound(
            _pipeline._plan_continuation_journal_v3_bound(sealed),
            sealed,
            self.pipeline(**options),
            self.pipeline(**options),
        )
        self.assertTrue(audit["equal"])
        self.assertEqual(len(audit["rows"]), 2)
        self.assertTrue(all(row["semantic"]["pipeline"]["equal"] for row in audit["rows"]))
        self.assertTrue(all(row["semantic"]["tophits"]["equal"] for row in audit["rows"]))

    def test_journal_v3_sparse_preserves_fixed_spaces_and_cumulative_reuse(self):
        options = {"F1": 0.02, "F2": 0.0, "F3": 0.0, "Z": 97, "domZ": 11}
        generation = self.pipeline(**options)
        sealed = self.seal_v2_terminal_fixture(
            self.sequences, pipeline=generation
        )
        dense = self.pipeline(**options)
        sparse = self.pipeline(**options)

        first = _pipeline._audit_continuation_journal_v3_bound(
            _pipeline._plan_continuation_journal_v3_bound(sealed),
            sealed,
            dense,
            sparse,
        )
        self.assertTrue(first["equal"])
        self.assertEqual(first["dense_hits"][0].Z, 97.0)
        self.assertEqual(first["dense_hits"][0].domZ, 11.0)
        self.assertEqual(first["dense_hits"][0].searched_models, 1)

        second = _pipeline._audit_continuation_journal_v3_bound(
            _pipeline._plan_continuation_journal_v3_bound(sealed),
            sealed,
            dense,
            sparse,
        )
        self.assertTrue(second["equal"])
        self.assertEqual(second["dense_hits"][0].Z, 97.0)
        self.assertEqual(second["dense_hits"][0].domZ, 11.0)
        self.assertEqual(second["dense_hits"][0].searched_models, 2)

        dense.clear()
        sparse.clear()
        third = _pipeline._audit_continuation_journal_v3_bound(
            _pipeline._plan_continuation_journal_v3_bound(sealed),
            sealed,
            dense,
            sparse,
        )
        self.assertTrue(third["equal"])
        self.assertEqual(third["dense_hits"][0].searched_models, 1)

    def test_forward_batch_validation_preserves_global_row_mapping(self):
        matrix_target = 3
        matrix_count = 6 * (len(self.sequences[matrix_target]) + 1)
        postfilter = b"".join(
            (
                self.postfilter_record(1, 0.0, 0, 0, 2, 0.0),
                self.postfilter_record(2, 0.0, 0, 0, 2, 0.0),
                self.postfilter_record(matrix_target, 0.0, 0, 0, 2, 0.0),
                self.postfilter_record(4, 0.0, 0, 0, 2, 0.0),
            )
        )
        forward = b"".join(
            (
                self.forward_record(1, -7.0, 0, 1),
                self.forward_record(2, -7.0, 0, 0),
                self.forward_record(matrix_target, -7.0, 0, 2),
            )
        )
        postfilter_offsets = array("Q", [0, 1, 4, 4])
        forward_offsets = array("Q", [0, 1, 3, 3])
        expected_indices = array("I", [1, 2, matrix_target])
        special_offsets = array("Q", [0, 0, 0, matrix_count])
        specials = array("f", [0.0]) * matrix_count

        _pipeline._validate_forward_batch_bound(
            self.sequences,
            postfilter,
            postfilter_offsets,
            forward,
            forward_offsets,
            expected_indices,
            special_offsets,
            specials,
        )
        with self.assertRaisesRegex(ValueError, "order differs"):
            _pipeline._validate_forward_batch_bound(
                self.sequences,
                postfilter,
                postfilter_offsets,
                forward,
                forward_offsets,
                array("I", [2, 1, matrix_target]),
                special_offsets,
                specials,
            )
        with self.assertRaisesRegex(ValueError, "subset"):
            _pipeline._validate_forward_batch_bound(
                self.sequences,
                postfilter,
                postfilter_offsets,
                forward,
                array("Q", [0, 2, 3, 3]),
                expected_indices,
                special_offsets,
                specials,
            )

    def test_sealed_cpu_batch_is_exact_reusable_opaque_and_lifetime_pinned(self):
        hmm = self.hmms[0]
        optimized = self.optimized_profiles[0].copy()
        residue_offsets = self.residue_offsets(self.sequences)
        records = b"".join(
            self.postfilter_record(index, math.nan, 0, 255, 0, math.nan)
            for index in range(len(self.sequences))
        )
        row_offsets = array("Q", [0, len(self.sequences)])
        background = pyhmmer.plan7.Background(self.alphabet)
        fingerprint = self.background_fingerprint(background)
        sealed = _pipeline._seal_postfilter_batch_bound(
            (hmm,),
            (optimized,),
            self.sequences,
            records,
            row_offsets,
            residue_offsets,
            0.02,
            fingerprint,
        )

        # The seal owns all search inputs; dropping the construction views does
        # not affect later reuse.
        del optimized, records, row_offsets, residue_offsets, background, fingerprint
        gc.collect()

        expected = self.pipeline().search_hmm(hmm, self.sequences)
        first = _pipeline._search_hmm_sealed_postfilter_bound(
            sealed, 0, self.pipeline()
        )
        second = _pipeline._search_hmm_sealed_postfilter_bound(
            sealed, 0, self.pipeline()
        )
        self.assert_exact_hits(expected, first)
        self.assert_exact_hits(expected, second)
        self.assertIsNot(first.query, second.query)
        self.assertIsNot(first.query, hmm)
        original_name = hmm.name
        first.query.name = b"mutated-result-query"
        self.assertEqual(second.query.name, original_name)
        self.assertEqual(hmm.name, original_name)
        self.assertFalse(hasattr(sealed, "_queries"))
        with self.assertRaises(AttributeError):
            sealed.records = b""
        with self.assertRaisesRegex(TypeError, "cannot be pickled"):
            pickle.dumps(sealed)

    def test_sealed_live_mismatch_fails_before_pipeline_mutation(self):
        hmm = self.hmms[0]
        residue_offsets = self.residue_offsets(self.sequences)
        records = b"".join(
            self.postfilter_record(index, math.nan, 0, 255, 0, math.nan)
            for index in range(len(self.sequences))
        )
        background = pyhmmer.plan7.Background(self.alphabet)
        sealed = _pipeline._seal_postfilter_batch_bound(
            (hmm,),
            (self.optimized_profiles[0].copy(),),
            self.sequences,
            records,
            array("Q", [0, len(self.sequences)]),
            residue_offsets,
            0.02,
            self.background_fingerprint(background),
        )
        pipeline = self.pipeline(F1=0.5)
        with self.assertRaisesRegex(ValueError, "does not match candidate F1"):
            _pipeline._search_hmm_sealed_postfilter_bound(sealed, 0, pipeline)

        expected = self.pipeline(F1=0.5).search_hmm(hmm, self.sequences)
        actual = pipeline.search_hmm(hmm, self.sequences)
        self.assert_exact_hits(expected, actual)

        background_mismatch = self.pipeline()
        background_mismatch.background.residue_frequencies[0] += 0.01
        with self.assertRaisesRegex(ValueError, "canonical hmmpress background"):
            _pipeline._search_hmm_sealed_postfilter_bound(
                sealed, 0, background_mismatch
            )

    def test_seal_freezes_all_retained_caller_buffers(self):
        hmm = self.hmms[0]
        records = bytearray(
            b"".join(
                self.postfilter_record(index, math.nan, 0, 255, 0, math.nan)
                for index in range(len(self.sequences))
            )
        )
        row_offsets = array("Q", [0, len(self.sequences)])
        residue_offsets = self.residue_offsets(self.sequences)
        background = pyhmmer.plan7.Background(self.alphabet)
        fingerprint = bytearray(self.background_fingerprint(background))
        sealed = _pipeline._seal_postfilter_batch_bound(
            (hmm,),
            (self.optimized_profiles[0].copy(),),
            self.sequences,
            records,
            row_offsets,
            residue_offsets,
            0.02,
            fingerprint,
        )

        records[:4] = struct.pack("=I", len(self.sequences) - 1)
        row_offsets[1] = 0
        residue_offsets[1] = 0
        fingerprint[0] ^= 0xFF

        expected = self.pipeline().search_hmm(hmm, self.sequences)
        actual = _pipeline._search_hmm_sealed_postfilter_bound(
            sealed, 0, self.pipeline()
        )
        self.assert_exact_hits(expected, actual)

    def test_sealed_forward_row_matches_validated_continuation(self):
        if not _pipeline._filter_and_forward_scores_seam_available():
            self.skipTest("private Forward-score seam is unavailable")
        target = 1
        hmm = self.hmms[0]
        postfilter = self.postfilter_record(target, 0.0, 0, 0, 2, 0.0)
        forward = self.forward_record(target, -7.0, 0, 1)
        mutable_forward = bytearray(forward)
        forward_offsets = array("Q", [0, 1])
        special_offsets = array("Q", [0, 0])
        specials = array("f")
        residue_offsets = self.residue_offsets(self.sequences)
        generation_f2_bits = self.double_bits(1.0)
        generation_f3_bits = self.double_bits(1e-5)
        background = pyhmmer.plan7.Background(self.alphabet)
        sealed = _pipeline._seal_postfilter_batch_bound(
            (hmm,),
            (self.optimized_profiles[0].copy(),),
            self.sequences,
            postfilter,
            array("Q", [0, 1]),
            residue_offsets,
            1.0,
            self.background_fingerprint(background),
            forward_records=mutable_forward,
            forward_offsets=forward_offsets,
            special_offsets=special_offsets,
            specials=specials,
            expected_forward_indices=array("I", [target]),
            generation_f2_bits=generation_f2_bits,
            generation_f3_bits=generation_f3_bits,
            generation_bias_filter=True,
        )
        mutable_forward[9] = 0
        forward_offsets[1] = 0
        options = {"F1": 1.0, "F2": 1.0, "F3": 1e-5}
        expected = _pipeline._search_hmm_postfilter_forward_bound(
            self.pipeline(**options),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            postfilter,
            forward,
            array("Q", [0, 0]),
            array("f"),
            residue_offsets,
            generation_f2_bits,
            generation_f3_bits,
            True,
        )
        actual = _pipeline._search_hmm_sealed_postfilter_bound(
            sealed, 0, self.pipeline(**options)
        )
        self.assert_exact_hits(expected, actual)

        fallback_options = dict(options, F3=math.nextafter(1e-5, 0.0))
        expected_fallback = _pipeline._search_hmm_postfilter_bound(
            self.pipeline(**fallback_options),
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            postfilter,
            residue_offsets,
        )
        actual_fallback = _pipeline._search_hmm_sealed_postfilter_bound(
            sealed, 0, self.pipeline(**fallback_options)
        )
        self.assert_exact_hits(expected_fallback, actual_fallback)

    def test_seal_rejects_malformed_global_row_maps(self):
        background = pyhmmer.plan7.Background(self.alphabet)
        arguments = (
            (self.hmms[0],),
            (self.optimized_profiles[0].copy(),),
            self.sequences,
            self.postfilter_record(1, 0.0, 0, 0, 2, 0.0),
            array("Q", [0, 1]),
            self.residue_offsets(self.sequences),
            1.0,
            self.background_fingerprint(background),
        )
        with self.assertRaisesRegex(ValueError, "order differs"):
            _pipeline._seal_postfilter_batch_bound(
                *arguments,
                forward_records=self.forward_record(1, -7.0, 0, 1),
                forward_offsets=array("Q", [0, 1]),
                special_offsets=array("Q", [0, 0]),
                specials=array("f"),
                expected_forward_indices=array("I", [2]),
                generation_bias_filter=True,
            )

    def test_ok_cpu_forward_cap_fallback_uses_original_continuation(self):
        if not _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is unavailable")
        target = 1
        offsets = self.residue_offsets(self.sequences)
        postfilter = self.postfilter_record(target, 0.0, 0, 0, 2, 0.0)
        # Native output-budget fallback preserves its exact finite fwdsc but
        # deliberately supplies no special matrix.
        forward = self.forward_record(target, -7.0, 0, 0)
        options = {"F1": 1.0, "F2": 1.0, "F3": 1.0}
        expected = _pipeline._search_hmm_postfilter_bound(
            self.pipeline(**options),
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            postfilter,
            offsets,
        )
        actual = _pipeline._search_hmm_postfilter_forward_bound(
            self.pipeline(**options),
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            postfilter,
            forward,
            array("Q", [0, 0]),
            array("f"),
            offsets,
            self.double_bits(1.0),
            self.double_bits(1.0),
            True,
        )
        self.assert_exact_hits(expected, actual)

    def test_forward_option_mismatch_uses_original_continuation(self):
        if not _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is unavailable")
        target = 1
        offsets = self.residue_offsets(self.sequences)
        postfilter = self.postfilter_record(target, 0.0, 0, 0, 2, 0.0)
        forward = self.forward_record(target, -7.0, 0, 1)
        ordinary = {"F1": 1.0, "F2": 1.0, "F3": 1.0}
        mismatches = (
            ("F2 nextafter", ordinary, math.nextafter(1.0, 0.0), 1.0, True),
            ("F3 nextafter", ordinary, 1.0, math.nextafter(1.0, 0.0), True),
            ("F2 NaN", ordinary, math.nan, 1.0, True),
            ("F3 NaN", ordinary, 1.0, math.nan, True),
            ("F2 infinity", ordinary, math.inf, 1.0, True),
            ("F3 infinity", ordinary, 1.0, math.inf, True),
            ("F2 negative", ordinary, -1.0, 1.0, True),
            ("F3 negative", ordinary, 1.0, -1.0, True),
            (
                "F2 signed zero",
                {"F1": 1.0, "F2": 0.0, "F3": 1.0},
                -0.0,
                1.0,
                True,
            ),
            (
                "F3 signed zero",
                {"F1": 1.0, "F2": 1.0, "F3": 0.0},
                1.0,
                -0.0,
                True,
            ),
            ("bias", ordinary, 1.0, 1.0, False),
        )
        for label, options, generation_f2, generation_f3, bias_filter in mismatches:
            with self.subTest(label=label):
                expected = _pipeline._search_hmm_postfilter_bound(
                    self.pipeline(**options),
                    self.hmms[0],
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    postfilter,
                    offsets,
                )
                actual = _pipeline._search_hmm_postfilter_forward_bound(
                    self.pipeline(**options),
                    self.hmms[0],
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    postfilter,
                    forward,
                    array("Q", [0, 0]),
                    array("f"),
                    offsets,
                    self.double_bits(generation_f2),
                    self.double_bits(generation_f3),
                    bias_filter,
                )
                self.assert_exact_hits(expected, actual)

    def test_invalid_forward_augmentation_precedes_pipeline_mutation(self):
        target = 1
        offsets = self.residue_offsets(self.sequences)
        postfilter = self.postfilter_record(target, 0.0, 0, 0, 2, 0.0)
        invalid = (
            (b"\0", array("Q", [0]), array("f"), "trailing bytes"),
            (
                self.forward_record(target, math.nan, 19, 0, 1),
                array("Q", [0, 0]),
                array("f"),
                "reserved field",
            ),
            (
                self.forward_record(target + 1, math.nan, 19, 0),
                array("Q", [0, 0]),
                array("f"),
                "subset",
            ),
            (
                self.forward_record(target, 0.0, 0, 3),
                array("Q", [0, 0]),
                array("f"),
                "unknown Forward.*action",
            ),
            (
                self.forward_record(target, 0.0, 0, 2),
                array("Q", [0, 0]),
                array("f"),
                "wrong special-state span",
            ),
            (
                self.forward_record(target, 0.0, 0, 1),
                array("Q", [0, 1]),
                array("f", [0.0]),
                "must not have special states",
            ),
        )
        for records, special_offsets, specials, message in invalid:
            with self.subTest(message=message):
                pipeline = self.pipeline(F1=1.0, F2=1.0, F3=1.0)
                with self.assertRaisesRegex((IndexError, ValueError), message):
                    _pipeline._search_hmm_postfilter_forward_bound(
                        pipeline,
                        self.hmms[0],
                        self.optimized_profiles[0].copy(),
                        self.sequences,
                        postfilter,
                        records,
                        special_offsets,
                        specials,
                        offsets,
                        self.double_bits(1.0),
                        self.double_bits(1.0),
                        True,
                    )
                hits = _pipeline._search_hmm_postfilter_bound(
                    pipeline,
                    self.hmms[0],
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    b"",
                    offsets,
                )
                self.assertEqual(hits.searched_models, 1)

    def test_invalid_postfilter_rows_fail_before_pipeline_mutation(self):
        offsets = self.residue_offsets(self.sequences)

        def cpu(index):
            return self.postfilter_record(index, math.nan, 0, 255, 0, math.nan)

        invalid_rows = (
            (b"\0", "trailing bytes"),
            (cpu(0) + cpu(0), "strictly increasing"),
            (cpu(1) + cpu(0), "strictly increasing"),
            (cpu(len(self.sequences)), "out of range"),
            (self.postfilter_record(0, 0.0, 0, 0, 3, 0.0), "unknown.*action"),
            (
                self.postfilter_record(0, 0.0, 0, 19, 1, 0.0),
                "requires eslOK",
            ),
            (
                self.postfilter_record(0, math.nan, 0, 0, 2, math.nan),
                "finite filter",
            ),
            (
                self.postfilter_record(0, 0.0, 0, 0, 1, math.nan),
                r"finite or \+infinity",
            ),
            (
                self.postfilter_record(0, 0.0, 0, 16, 1, math.inf),
                "requires eslOK",
            ),
        )
        for records, message in invalid_rows:
            with self.subTest(message=message):
                pipeline = self.pipeline()
                with self.assertRaisesRegex((IndexError, ValueError), message):
                    _pipeline._search_hmm_postfilter_bound(
                        pipeline,
                        self.hmms[0],
                        self.optimized_profiles[0].copy(),
                        self.sequences,
                        records,
                        offsets,
                    )
                hits = _pipeline._search_hmm_postfilter_bound(
                    pipeline,
                    self.hmms[0],
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    b"",
                    offsets,
                )
                self.assertEqual(hits.searched_models, 1)

    def test_direct_bias_symbol_failure_precedes_pipeline_mutation(self):
        if _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is available")
        hmm = self.hmms[0]
        offsets = self.residue_offsets(self.sequences)
        pipeline = self.pipeline()
        direct = self.bias_record(0, 0.0, 0, 0, 1)

        with self.assertRaisesRegex(RuntimeError, "p7_PipelineFromFilterScores"):
            _pipeline._search_hmm_bias_bound(
                pipeline,
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                direct,
                offsets,
            )

        hits = _pipeline._search_hmm_bias_bound(
            pipeline,
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            b"",
            offsets,
        )
        self.assertEqual(hits.searched_models, 1)

    def test_invalid_bias_rows_fail_before_pipeline_mutation(self):
        hmm = self.hmms[0]
        offsets = self.residue_offsets(self.sequences)

        def cpu(index):
            return self.bias_record(index, math.nan, 0, 255, 0)

        invalid_rows = (
            (b"\0", "trailing bytes"),
            (cpu(0) + cpu(0), "strictly increasing"),
            (cpu(1) + cpu(0), "strictly increasing"),
            (cpu(len(self.sequences)), "out of range"),
            (self.bias_record(0, 0.0, 0, 0, 3), "unknown.*action"),
            (self.bias_record(0, 0.0, 0, 16, 1), "eslOK"),
            (self.bias_record(0, math.nan, 0, 0, 2), "finite filter"),
        )
        for records, message in invalid_rows:
            with self.subTest(message=message):
                pipeline = self.pipeline()
                with self.assertRaisesRegex((IndexError, ValueError), message):
                    _pipeline._search_hmm_bias_bound(
                        pipeline,
                        hmm,
                        self.optimized_profiles[0].copy(),
                        self.sequences,
                        records,
                        offsets,
                    )
                hits = _pipeline._search_hmm_bias_bound(
                    pipeline,
                    hmm,
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    b"",
                    offsets,
                )
                self.assertEqual(hits.searched_models, 1)

        invalid_offsets = array("Q", offsets)
        invalid_offsets[2] += 1
        pipeline = self.pipeline()
        with self.assertRaisesRegex(ValueError, "differs from target length"):
            _pipeline._search_hmm_bias_bound(
                pipeline,
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                cpu(1),
                invalid_offsets,
            )
        hits = _pipeline._search_hmm_bias_bound(
            pipeline,
            hmm,
            self.optimized_profiles[0].copy(),
            self.sequences,
            b"",
            offsets,
        )
        self.assertEqual(hits.searched_models, 1)

    def test_empty_target_cannot_claim_a_direct_bias_result(self):
        hmm = self.hmms[0]
        empty = pyhmmer.easel.TextSequence(name=b"empty", sequence="").digitize(
            self.alphabet
        )
        targets = pyhmmer.easel.DigitalSequenceBlock(self.alphabet, [empty])
        with self.assertRaisesRegex(ValueError, "empty target"):
            _pipeline._search_hmm_bias_bound(
                self.pipeline(),
                hmm,
                self.optimized_profiles[0].copy(),
                targets,
                self.bias_record(0, 0.0, 0, 0, 1),
                array("Q", [0, 0]),
            )

    def test_bias_bridge_requires_exact_pipeline_and_contiguous_bytes(self):
        hmm = self.hmms[0]
        offsets = self.residue_offsets(self.sequences)
        record = self.bias_record(0, math.nan, 0, 255, 0)

        with self.assertRaisesRegex(TypeError, "exactly pyhmmer.plan7.Pipeline"):
            _pipeline._search_hmm_bias_bound(
                pyhmmer.plan7.LongTargetsPipeline(pyhmmer.easel.Alphabet.dna()),
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                record,
                offsets,
            )
        with self.assertRaises((BufferError, TypeError, ValueError)):
            _pipeline._search_hmm_bias_bound(
                self.pipeline(),
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                list(record),
                offsets,
            )
        with self.assertRaises((BufferError, TypeError, ValueError)):
            _pipeline._search_hmm_bias_bound(
                self.pipeline(),
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                memoryview(record * 2)[::2],
                offsets,
            )

    def test_patched_bias_continuation_matches_ordinary_pipeline(self):
        if not _pipeline._filter_scores_seam_available():
            self.skipTest("private filter-score seam is unavailable")
        hmm = self.hmms[1]
        optimized = self.optimized_profiles[1].copy()
        oracle_pipeline = self.pipeline(F1=1.0)
        offsets = self.residue_offsets(self.sequences)
        records = []
        direct_actions = set()

        for index, sequence in enumerate(self.sequences):
            optimized.L = len(sequence)
            score = optimized.ssv_filter(sequence)
            if score is None or not math.isfinite(score):
                records.append(self.bias_record(index, math.nan, 0, 255, 0))
                continue
            numerator = round((score + 3.0) * optimized.scale_b)
            reconstructed = struct.unpack(
                "=f",
                struct.pack(
                    "=f",
                    struct.unpack("=f", struct.pack("=f", float(numerator)))[0]
                    / optimized.scale_b
                    - 3.0,
                ),
            )[0]
            if struct.pack("=f", reconstructed) != struct.pack("=f", score):
                records.append(self.bias_record(index, math.nan, 0, 255, 0))
                continue
            filtersc_bits = _pipeline._bias_filter_score_bits(
                oracle_pipeline, optimized, sequence
            )
            filtersc = struct.unpack("=f", struct.pack("=I", filtersc_bits))[0]
            action = 1 + index % 2
            direct_actions.add(action)
            records.append(self.bias_record(index, filtersc, numerator, 0, action))

        self.assertEqual(direct_actions, {1, 2})
        records = b"".join(records)
        option_sets = (
            {"F1": 1.0, "bias_filter": True},
            {"F1": 1.0, "bias_filter": False},
            {"F1": 0.02, "bias_filter": True},
            {"F1": 1.0, "F2": 1.0, "F3": 1.0, "bias_filter": False},
        )
        saw_post_f2 = False
        for options in option_sets:
            with self.subTest(**options):
                expected = self.pipeline(**options).search_hmm(hmm, self.sequences)
                actual = _pipeline._search_hmm_bias_bound(
                    self.pipeline(**options),
                    hmm,
                    self.optimized_profiles[1].copy(),
                    self.sequences,
                    records,
                    offsets,
                )
                self.assert_exact_hits(expected, actual)
                state = actual.__getstate__()["pipeline"]
                saw_post_f2 = saw_post_f2 or state["n_past_vit"] > 0
        self.assertTrue(saw_post_f2)
        cache = _pipeline._continuation_seam_cache_info()["filter"]
        self.assertEqual(cache["resolutions"], 1)
        self.assertEqual(cache["dlopen_calls"], cache["dlclose_calls"])

    def test_candidate_masks_preserve_auto_and_explicit_search_spaces(self):
        hmm = self.hmms[0]
        database_size = len(self.sequences)
        masks = (
            (),
            (1, 7, database_size // 2),
            tuple(range(database_size)),
        )
        search_spaces = (
            {},
            {"Z": 97},
            {"domZ": 11},
            {"Z": 97, "domZ": 11},
        )

        for indexes in masks:
            subset = pyhmmer.easel.DigitalSequenceBlock(
                self.alphabet, [self.sequences[index] for index in indexes]
            )
            for options in search_spaces:
                with self.subTest(indexes=indexes, options=options):
                    reference_options = dict(options)
                    reference_options.setdefault("Z", database_size)
                    expected = self.pipeline(**reference_options).search_hmm(
                        hmm, subset
                    )
                    actual = _pipeline._search_hmm_candidates(
                        self.pipeline(**options),
                        hmm,
                        self.optimized_profiles[0].copy(),
                        self.sequences,
                        self.candidates(*indexes),
                    )

                    self.assertEqual(
                        self.table_bytes(actual, "targets"),
                        self.table_bytes(expected, "targets"),
                    )
                    self.assertEqual(
                        self.table_bytes(actual, "domains"),
                        self.table_bytes(expected, "domains"),
                    )
                    self.assertEqual(actual.Z, float(options.get("Z", database_size)))
                    self.assertEqual(actual.domZ, expected.domZ)
                    self.assertEqual(actual.searched_sequences, database_size)
                    self.assertEqual(
                        actual.searched_residues, self.sequences.total_length()
                    )

    def test_skipped_empty_rejects_match_complete_pipeline_state(self):
        hmm = self.hmms[0]
        empties = [
            pyhmmer.easel.TextSequence(
                name=f"empty-{index}".encode(), sequence=""
            ).digitize(self.alphabet)
            for index in range(3)
        ]
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet,
            [empties[0], self.sequences[0], empties[1], self.sequences[1], empties[2]],
        )

        for options in ({}, {"Z": 97}, {"domZ": 11}, {"Z": 97, "domZ": 11}):
            with self.subTest(options=options):
                expected = self.pipeline(**options).search_hmm(hmm, targets)
                actual = _pipeline._search_hmm_candidates(
                    self.pipeline(**options),
                    hmm,
                    self.optimized_profiles[0].copy(),
                    targets,
                    self.candidates(1, 3),
                )
                self.assert_exact_hits(expected, actual)

    def test_rejected_tail_preserves_final_model_lengths(self):
        hmm = self.hmms[0]
        last_length = len(self.sequences[-1])
        expected_transition = array("f", [last_length / (last_length + 1)])[0]
        expected_optimized = self.optimized_profiles[0].copy()
        expected_optimized.L = last_length

        for indexes in ((), (0,), tuple(range(len(self.sequences)))):
            with self.subTest(indexes=indexes):
                pipeline = self.pipeline()
                optimized = self.optimized_profiles[0].copy()
                _pipeline._search_hmm_candidates(
                    pipeline,
                    hmm,
                    optimized,
                    self.sequences,
                    self.candidates(*indexes),
                )
                self.assertEqual(optimized.L, last_length)
                self.assertTrue(
                    _pipeline._oprofiles_equal_hmmer(expected_optimized, optimized)
                )
                self.assertEqual(
                    pipeline.background.transition_probability,
                    expected_transition,
                )

    def test_final_empty_member_preserves_zero_model_lengths(self):
        hmm = self.hmms[0]
        empty = pyhmmer.easel.TextSequence(name=b"empty", sequence="").digitize(
            self.alphabet
        )
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[0], empty]
        )
        expected_optimized = self.optimized_profiles[0].copy()
        expected_optimized.L = 0

        for indexes in ((), (0,), (0, 1)):
            with self.subTest(indexes=indexes):
                pipeline = self.pipeline()
                optimized = self.optimized_profiles[0].copy()
                _pipeline._search_hmm_candidates(
                    pipeline,
                    hmm,
                    optimized,
                    targets,
                    self.candidates(*indexes),
                )
                self.assertEqual(optimized.L, 0)
                self.assertTrue(
                    _pipeline._oprofiles_equal_hmmer(expected_optimized, optimized)
                )
                self.assertEqual(pipeline.background.transition_probability, 0.0)

    def test_empty_target_block_still_registers_the_model(self):
        hmm = self.hmms[0]
        empty = pyhmmer.easel.DigitalSequenceBlock(self.alphabet)
        pipeline = self.pipeline()
        optimized = self.optimized_profiles[0].copy()
        initial_length = optimized.L
        initial_transition = pipeline.background.transition_probability
        hits = _pipeline._search_hmm_candidates(
            pipeline,
            hmm,
            optimized,
            empty,
            self.candidates(),
        )

        self.assertEqual(hits.searched_models, 1)
        self.assertEqual(hits.searched_nodes, hmm.M)
        self.assertEqual(hits.searched_sequences, 0)
        self.assertEqual(hits.searched_residues, 0)
        self.assertEqual(hits.Z, 0.0)
        self.assertEqual(optimized.L, initial_length)
        self.assertEqual(pipeline.background.transition_probability, initial_transition)

    def test_explicit_z_and_domz_survive_all_rejects(self):
        hits = _pipeline._search_hmm_candidates(
            self.pipeline(Z=123, domZ=17),
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(),
        )
        self.assertEqual(hits.Z, 123.0)
        self.assertEqual(hits.domZ, 17.0)

    def test_rejected_pass_can_be_cleared_and_reused_for_exact_search(self):
        hmm = self.hmms[0]
        optimized = self.optimized_profiles[0].copy()
        pipeline = self.pipeline()

        rejected = _pipeline._search_hmm_candidates(
            pipeline, hmm, optimized, self.sequences, self.candidates()
        )
        self.assertEqual(len(rejected), 0)

        pipeline.clear()
        actual = _pipeline._search_hmm_candidates(
            pipeline,
            hmm,
            optimized,
            self.sequences,
            self.candidates(*range(len(self.sequences))),
        )
        expected = self.pipeline().search_hmm(hmm, self.sequences)
        self.assert_exact_hits(expected, actual)

    def test_pipeline_reuse_without_clear_matches_pyhmmer_and_keeps_snapshots(self):
        hmm = self.hmms[0]
        optimized = self.optimized_profiles[0].copy()
        candidates = self.candidates(*range(len(self.sequences)))
        expected_pipeline = self.pipeline()
        actual_pipeline = self.pipeline()

        expected_first = expected_pipeline.search_hmm(hmm, self.sequences)
        actual_first = _pipeline._search_hmm_candidates(
            actual_pipeline, hmm, optimized, self.sequences, candidates
        )
        self.assert_exact_hits(expected_first, actual_first)
        first_state = self.semantic_state(actual_first)
        first_targets = self.table_bytes(actual_first, "targets")
        first_domains = self.table_bytes(actual_first, "domains")

        expected_second = expected_pipeline.search_hmm(hmm, self.sequences)
        actual_second = _pipeline._search_hmm_candidates(
            actual_pipeline, hmm, optimized, self.sequences, candidates
        )
        self.assert_exact_hits(expected_second, actual_second)
        self.assertEqual(self.semantic_state(actual_first), first_state)
        self.assertEqual(self.table_bytes(actual_first, "targets"), first_targets)
        self.assertEqual(self.table_bytes(actual_first, "domains"), first_domains)

    def test_rejected_then_sparse_pipeline_reuse_without_clear(self):
        hmm = self.hmms[0]
        optimized = self.optimized_profiles[0].copy()
        pipeline = self.pipeline()
        database_size = len(self.sequences)
        total_length = self.sequences.total_length()

        rejected = _pipeline._search_hmm_candidates(
            pipeline, hmm, optimized, self.sequences, self.candidates()
        )
        rejected_state = self.semantic_state(rejected)

        indexes = (1, 7, database_size // 2)
        subset = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [self.sequences[index] for index in indexes]
        )
        expected = self.pipeline(Z=database_size).search_hmm(hmm, subset)
        actual = _pipeline._search_hmm_candidates(
            pipeline,
            hmm,
            optimized,
            self.sequences,
            self.candidates(*indexes),
        )

        self.assertEqual(
            self.table_bytes(actual, "targets"),
            self.table_bytes(expected, "targets"),
        )
        self.assertEqual(
            self.table_bytes(actual, "domains"),
            self.table_bytes(expected, "domains"),
        )
        self.assertEqual(actual.searched_models, 2)
        self.assertEqual(actual.searched_nodes, 2 * hmm.M)
        self.assertEqual(actual.searched_sequences, database_size)
        self.assertEqual(actual.searched_residues, 2 * total_length)
        self.assertEqual(self.semantic_state(rejected), rejected_state)

    def test_embedded_empty_target_matches_pyhmmer_or_is_accounted_when_rejected(self):
        hmm = self.hmms[0]
        empty = pyhmmer.easel.TextSequence(name=b"empty", sequence="").digitize(
            self.alphabet
        )
        targets = pyhmmer.easel.DigitalSequenceBlock(
            self.alphabet, [empty, self.sequences[0]]
        )
        expected = self.pipeline().search_hmm(hmm, targets)
        actual = _pipeline._search_hmm_candidates(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            targets,
            self.candidates(0, 1),
        )
        self.assert_exact_hits(expected, actual)

        rejected = _pipeline._search_hmm_candidates(
            self.pipeline(),
            hmm,
            self.optimized_profiles[0].copy(),
            targets,
            self.candidates(),
        )
        self.assertEqual(rejected.searched_sequences, 2)
        self.assertEqual(rejected.searched_residues, len(self.sequences[0]))

    def test_invalid_candidate_rows_fail_before_pipeline_mutation(self):
        invalid_rows = (
            self.candidates(0, 0),
            self.candidates(1, 0),
            self.candidates(len(self.sequences)),
            self.candidates(2**32 - 1),
        )
        for candidates in invalid_rows:
            with self.subTest(candidates=list(candidates)):
                pipeline = self.pipeline()
                with self.assertRaises((IndexError, ValueError)):
                    _pipeline._search_hmm_candidates(
                        pipeline,
                        self.hmms[0],
                        self.optimized_profiles[0].copy(),
                        self.sequences,
                        candidates,
                    )
                hits = _pipeline._search_hmm_candidates(
                    pipeline,
                    self.hmms[0],
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    self.candidates(),
                )
                self.assertEqual(hits.searched_models, 1)
                self.assertEqual(hits.searched_sequences, len(self.sequences))

    def test_candidate_row_requires_contiguous_native_uint32_buffer(self):
        with self.assertRaises((BufferError, TypeError, ValueError)):
            _pipeline._search_hmm_candidates(
                self.pipeline(),
                self.hmms[0],
                self.optimized_profiles[0].copy(),
                self.sequences,
                list(range(len(self.sequences))),
            )

        storage = self.candidates(0, 1, 2, 3)
        with self.assertRaises((BufferError, TypeError, ValueError)):
            _pipeline._search_hmm_candidates(
                self.pipeline(),
                self.hmms[0],
                self.optimized_profiles[0].copy(),
                self.sequences,
                memoryview(storage)[::2],
            )

    def test_model_length_and_metadata_mismatches_fail_before_mutation(self):
        hmm = self.hmms[0]
        mismatches = [self.optimized_profiles[1].copy()]
        renamed = hmm.copy()
        renamed.name = f"{hmm.name}-wrong-owner"

        for query, optimized in (
            (hmm, mismatches[0]),
            (renamed, self.optimized_profiles[0].copy()),
        ):
            with self.subTest(query=query.name, optimized=optimized.name):
                pipeline = self.pipeline()
                with self.assertRaises(ValueError):
                    _pipeline._search_hmm_candidates(
                        pipeline,
                        query,
                        optimized,
                        self.sequences,
                        self.candidates(),
                    )
                hits = _pipeline._search_hmm_candidates(
                    pipeline,
                    hmm,
                    self.optimized_profiles[0].copy(),
                    self.sequences,
                    self.candidates(),
                )
                self.assertEqual(hits.searched_models, 1)

    def test_alphabet_mismatches_use_pyhmmer_exception(self):
        dna = pyhmmer.easel.Alphabet.dna()
        dna_sequence = pyhmmer.easel.TextSequence(
            name=b"dna", sequence="ACGT"
        ).digitize(dna)
        dna_sequences = pyhmmer.easel.DigitalSequenceBlock(dna, [dna_sequence])

        with self.assertRaises(AlphabetMismatch):
            _pipeline._search_hmm_candidates(
                self.pipeline(),
                self.hmms[0],
                self.optimized_profiles[0].copy(),
                dna_sequences,
                self.candidates(),
            )
        with self.assertRaises(AlphabetMismatch):
            _pipeline._search_hmm_candidates(
                pyhmmer.plan7.Pipeline(dna),
                self.hmms[0],
                self.optimized_profiles[0].copy(),
                self.sequences,
                self.candidates(),
            )

    def test_overlong_reject_is_checked_before_pipeline_mutation(self):
        long_sequence = pyhmmer.easel.TextSequence(
            name=b"too-long", sequence="A" * 100_001
        ).digitize(self.alphabet)
        long_block = pyhmmer.easel.DigitalSequenceBlock(self.alphabet, [long_sequence])
        pipeline = self.pipeline()

        with self.assertRaisesRegex(
            ValueError,
            r"sequence length over comparison pipeline limit \(100000\)",
        ):
            _pipeline._search_hmm_candidates(
                pipeline,
                self.hmms[0],
                self.optimized_profiles[0].copy(),
                long_block,
                self.candidates(),
            )

        hits = _pipeline._search_hmm_candidates(
            pipeline,
            self.hmms[0],
            self.optimized_profiles[0].copy(),
            self.sequences,
            self.candidates(),
        )
        self.assertEqual(hits.searched_models, 1)

    def test_missing_cutoff_exception_matches_pyhmmer(self):
        hmm = self.hmms[0]
        with self.assertRaises(MissingCutoffs) as reference_error:
            pyhmmer.plan7.Pipeline(self.alphabet, bit_cutoffs="gathering").search_hmm(
                hmm, self.sequences
            )

        with self.assertRaises(MissingCutoffs) as actual_error:
            _pipeline._search_hmm_candidates(
                pyhmmer.plan7.Pipeline(self.alphabet, bit_cutoffs="gathering"),
                hmm,
                self.optimized_profiles[0].copy(),
                self.sequences,
                self.candidates(),
            )

        self.assertEqual(str(actual_error.exception), str(reference_error.exception))
        self.assertEqual(
            actual_error.exception.model_name,
            reference_error.exception.model_name,
        )
        self.assertEqual(
            actual_error.exception.bit_cutoffs,
            reference_error.exception.bit_cutoffs,
        )

    def test_valid_model_cutoffs_match_pyhmmer(self):
        hmm = self.hmms[0].copy()
        hmm.cutoffs.gathering = (0.0, 0.0)
        hmm.cutoffs.trusted = (0.0, 0.0)
        hmm.cutoffs.noise = (0.0, 0.0)

        with tempfile.TemporaryDirectory(prefix="plan7-cutoff-test-") as directory:
            pressed_base = Path(directory) / "one"
            pyhmmer.hmmer.hmmpress([hmm], pressed_base)
            with pyhmmer.plan7.HMMFile(pressed_base) as hmm_file:
                query = next(hmm_file)
            with pyhmmer.plan7.HMMPressedFile(pressed_base) as pressed_file:
                optimized = next(pressed_file)

            for cutoff in ("gathering", "trusted", "noise"):
                with self.subTest(cutoff=cutoff):
                    expected = self.pipeline(bit_cutoffs=cutoff).search_hmm(
                        query, self.sequences
                    )
                    actual = _pipeline._search_hmm_candidates(
                        self.pipeline(bit_cutoffs=cutoff),
                        query,
                        optimized.copy(),
                        self.sequences,
                        self.candidates(*range(len(self.sequences))),
                    )
                    self.assert_exact_hits(expected, actual)


if __name__ == "__main__":
    unittest.main()
