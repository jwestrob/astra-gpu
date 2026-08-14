import importlib.util
import io
import math
import struct
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
