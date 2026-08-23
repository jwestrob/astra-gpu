import importlib.util
import io
import struct
import unittest
from pathlib import Path

import pyhmmer


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "python" / "plan7_gpu"
HMM_FIXTURE = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins4.hmm"
FASTA_FIXTURE = ROOT / "refs" / "src" / "hmmer-3.4" / "tutorial" / "globins45.fa"


def _load_pipeline_extension():
    paths = sorted(PACKAGE_DIR.glob("_pipeline*.so"))
    if not paths:
        return None
    spec = importlib.util.spec_from_file_location("_pipeline", paths[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pipeline = _load_pipeline_extension()


@unittest.skipUnless(_pipeline is not None, "pipeline extension unavailable")
class SemanticFingerprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with pyhmmer.plan7.HMMFile(HMM_FIXTURE) as hmm_file:
            cls.hmm = hmm_file.read()
        cls.alphabet = cls.hmm.alphabet
        with pyhmmer.easel.SequenceFile(
            FASTA_FIXTURE,
            digital=True,
            alphabet=cls.alphabet,
        ) as sequence_file:
            cls.sequences = sequence_file.read_block()

    @classmethod
    def optimized_profile(cls):
        background = pyhmmer.plan7.Background(cls.alphabet)
        return cls.hmm.to_profile(background, L=400).to_optimized()

    @classmethod
    def distinct_optimized_profile(cls):
        changed_hmm = cls.hmm.copy()
        changed_hmm.match_emissions[1, 0] = 0.0
        changed_hmm.renormalize()
        background = pyhmmer.plan7.Background(cls.alphabet)
        return changed_hmm.to_profile(background, L=400).to_optimized()

    @staticmethod
    def pipeline(**options):
        defaults = {
            "E": 10.0,
            "domE": 10.0,
            "incE": 10.0,
            "incdomE": 10.0,
        }
        defaults.update(options)
        return pyhmmer.plan7.Pipeline(
            SemanticFingerprintTests.alphabet,
            **defaults,
        )

    @classmethod
    def search(cls, **options):
        pipeline = cls.pipeline(**options)
        profile = cls.optimized_profile()
        hits = pipeline.search_hmm(profile, cls.sequences)
        return pipeline, profile, hits

    def test_independent_equivalent_states_are_byte_stable(self):
        left_pipeline, left_profile, left_hits = self.search()
        right_pipeline, right_profile, right_hits = self.search()

        left_pipeline_bytes = _pipeline._semantic_pipeline_state_encoding_bound(
            left_pipeline, left_profile
        )
        right_pipeline_bytes = _pipeline._semantic_pipeline_state_encoding_bound(
            right_pipeline, right_profile
        )
        self.assertEqual(left_pipeline_bytes, right_pipeline_bytes)
        self.assertEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(
                left_pipeline, left_profile
            ),
            _pipeline._semantic_pipeline_state_fingerprint_bound(
                right_pipeline, right_profile
            ),
        )

        left_hits_bytes = _pipeline._semantic_tophits_encoding_bound(left_hits)
        right_hits_bytes = _pipeline._semantic_tophits_encoding_bound(right_hits)
        self.assertEqual(left_hits_bytes, right_hits_bytes)
        self.assertEqual(
            _pipeline._semantic_tophits_fingerprint_bound(left_hits),
            _pipeline._semantic_tophits_fingerprint_bound(right_hits),
        )

        comparison = _pipeline._semantic_dual_state_compare_bound(
            left_pipeline,
            right_pipeline,
            left_hits,
            right_hits,
            left_profile,
            right_profile,
        )
        self.assertEqual(comparison["schema_version"], 1)
        self.assertTrue(comparison["pipeline"]["equal"])
        self.assertTrue(comparison["tophits"]["equal"])
        self.assertIsNone(comparison["pipeline"]["first_difference"])
        self.assertIsNone(comparison["tophits"]["first_difference"])

    def test_bias_disabled_equivalent_states_never_read_filter_hmm(self):
        encodings = []
        hit_encodings = []
        for _ in range(10):
            pipeline, profile, hits = self.search(bias_filter=False)
            encodings.append(
                _pipeline._semantic_pipeline_state_encoding_bound(
                    pipeline, profile
                )
            )
            hit_encodings.append(
                _pipeline._semantic_tophits_encoding_bound(hits)
            )
        self.assertEqual(len(set(encodings)), 1)
        self.assertEqual(len(set(hit_encodings)), 1)
        self.assertIn(b"background.fhmm.state", encodings[0])
        self.assertIn(b"excluded-unproven", encodings[0])
        for numeric_field in (
            b"background.fhmm.M",
            b"background.fhmm.K",
            b"background.fhmm.pi[",
            b"background.fhmm.t[",
            b"background.fhmm.e[",
            b"background.fhmm.eo[",
        ):
            self.assertNotIn(numeric_field, encodings[0])

    def test_bias_setter_after_disabled_search_never_exposes_filter_hmm(self):
        encodings = []
        for _ in range(10):
            pipeline, profile, _ = self.search(bias_filter=False)
            pipeline.bias_filter = True
            encoding = _pipeline._semantic_pipeline_state_encoding_bound(
                pipeline, profile
            )
            self.assertIn(b"excluded-unproven", encoding)
            self.assertNotIn(b"background.fhmm.pi[", encoding)
            self.assertNotIn(b"background.fhmm.t[", encoding)
            self.assertNotIn(b"background.fhmm.e[", encoding)
            self.assertNotIn(b"background.fhmm.eo[", encoding)
            encodings.append(encoding)
        self.assertEqual(len(set(encodings)), 1)

    def test_raw_ieee_threshold_bits_are_sensitive(self):
        positive_zero = self.pipeline(T=0.0)
        negative_zero = self.pipeline(T=-0.0)
        positive = _pipeline._semantic_pipeline_state_encoding_bound(positive_zero)
        negative = _pipeline._semantic_pipeline_state_encoding_bound(negative_zero)
        self.assertNotEqual(positive, negative)
        self.assertIn(struct.pack("<Q", 0x8000000000000000), negative)

        comparison = _pipeline._semantic_dual_state_compare_bound(
            positive_zero,
            negative_zero,
            left_optimized_profile=self.optimized_profile(),
            right_optimized_profile=self.optimized_profile(),
        )
        self.assertFalse(comparison["pipeline"]["equal"])
        self.assertIsNotNone(comparison["pipeline"]["first_difference"])

    def test_profile_target_configuration_is_sensitive(self):
        pipeline = self.pipeline()
        short = self.optimized_profile()
        long = short.copy()
        short.L = 31
        long.L = 311
        self.assertNotEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(pipeline, short),
            _pipeline._semantic_pipeline_state_fingerprint_bound(pipeline, long),
        )

    def test_empty_database_and_uninitialized_empty_tophits(self):
        empty = pyhmmer.easel.DigitalSequenceBlock(self.alphabet)
        left_pipeline = self.pipeline()
        right_pipeline = self.pipeline()
        self.assertEqual(
            _pipeline._semantic_pipeline_state_encoding_bound(left_pipeline),
            _pipeline._semantic_pipeline_state_encoding_bound(right_pipeline),
        )
        left_profile = self.optimized_profile()
        right_profile = self.optimized_profile()
        left_hits = left_pipeline.search_hmm(left_profile, empty)
        right_hits = right_pipeline.search_hmm(right_profile, empty)

        self.assertEqual(len(left_hits), 0)
        comparison = _pipeline._semantic_dual_state_compare_bound(
            left_pipeline,
            right_pipeline,
            left_hits,
            right_hits,
            left_profile,
            right_profile,
        )
        self.assertTrue(comparison["pipeline"]["equal"])
        self.assertTrue(comparison["tophits"]["equal"])

        for format_name in ("targets", "domains"):
            left_table = io.BytesIO()
            right_table = io.BytesIO()
            left_hits.write(left_table, format=format_name, header=True)
            right_hits.write(right_table, format=format_name, header=True)
            self.assertEqual(left_table.getvalue(), right_table.getvalue())

        standalone_left = pyhmmer.plan7.TopHits(self.hmm)
        standalone_right = pyhmmer.plan7.TopHits(self.hmm)
        self.assertEqual(
            _pipeline._semantic_tophits_encoding_bound(standalone_left),
            _pipeline._semantic_tophits_encoding_bound(standalone_right),
        )

    def test_dynamic_and_fixed_z_domz_are_distinguished(self):
        dynamic_pipeline, dynamic_profile, dynamic_hits = self.search()
        fixed_pipeline, fixed_profile, fixed_hits = self.search(Z=123, domZ=17)
        self.assertEqual(dynamic_hits.Z, float(len(self.sequences)))
        self.assertEqual(fixed_hits.Z, 123.0)
        self.assertEqual(fixed_hits.domZ, 17.0)
        self.assertNotEqual(
            _pipeline._semantic_pipeline_state_fingerprint_bound(
                dynamic_pipeline, dynamic_profile
            ),
            _pipeline._semantic_pipeline_state_fingerprint_bound(
                fixed_pipeline, fixed_profile
            ),
        )
        self.assertNotEqual(
            _pipeline._semantic_tophits_fingerprint_bound(dynamic_hits),
            _pipeline._semantic_tophits_fingerprint_bound(fixed_hits),
        )

    def test_reuse_and_clear_remain_stable_and_sensitive(self):
        left_pipeline = self.pipeline()
        right_pipeline = self.pipeline()
        left_profile = self.optimized_profile()
        right_profile = self.optimized_profile()

        left_pipeline.search_hmm(left_profile, self.sequences)
        right_pipeline.search_hmm(right_profile, self.sequences)
        first = _pipeline._semantic_pipeline_state_encoding_bound(
            left_pipeline, left_profile
        )
        self.assertEqual(
            first,
            _pipeline._semantic_pipeline_state_encoding_bound(
                right_pipeline, right_profile
            ),
        )

        left_pipeline.search_hmm(left_profile, self.sequences)
        right_pipeline.search_hmm(right_profile, self.sequences)
        second = _pipeline._semantic_pipeline_state_encoding_bound(
            left_pipeline, left_profile
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            second,
            _pipeline._semantic_pipeline_state_encoding_bound(
                right_pipeline, right_profile
            ),
        )

        left_pipeline.clear()
        right_pipeline.clear()
        cleared = _pipeline._semantic_pipeline_state_encoding_bound(
            left_pipeline, left_profile
        )
        self.assertNotEqual(second, cleared)
        self.assertEqual(
            cleared,
            _pipeline._semantic_pipeline_state_encoding_bound(
                right_pipeline, right_profile
            ),
        )

        left_hits = left_pipeline.search_hmm(left_profile, self.sequences)
        right_hits = right_pipeline.search_hmm(right_profile, self.sequences)
        comparison = _pipeline._semantic_dual_state_compare_bound(
            left_pipeline,
            right_pipeline,
            left_hits,
            right_hits,
            left_profile,
            right_profile,
        )
        self.assertTrue(comparison["pipeline"]["equal"])
        self.assertTrue(comparison["tophits"]["equal"])

    def test_tophits_content_and_order_are_sensitive(self):
        _, _, full_hits = self.search()
        pipeline = self.pipeline()
        profile = self.optimized_profile()
        partial_hits = pipeline.search_hmm(profile, self.sequences[:10])
        self.assertNotEqual(
            _pipeline._semantic_tophits_fingerprint_bound(full_hits),
            _pipeline._semantic_tophits_fingerprint_bound(partial_hits),
        )

        copied = full_hits.copy()
        self.assertEqual(
            _pipeline._semantic_tophits_fingerprint_bound(full_hits),
            _pipeline._semantic_tophits_fingerprint_bound(copied),
        )
        if len(copied) > 1:
            before = _pipeline._semantic_tophits_encoding_bound(copied)
            swapped = (
                _pipeline._semantic_test_swapped_tophits_order_encoding_bound(
                    copied
                )
            )
            self.assertNotEqual(before, swapped)
            # The private audit helper must restore the input exactly.
            self.assertNotEqual(
                swapped,
                _pipeline._semantic_tophits_encoding_bound(copied),
            )
            self.assertEqual(
                before,
                _pipeline._semantic_tophits_encoding_bound(copied),
            )

    def test_reusable_domain_child_state_is_required(self):
        pipeline, profile, _ = self.search()
        before = _pipeline._semantic_pipeline_state_encoding_bound(
            pipeline, profile
        )
        for field in (
            "sp.nsamples",
            "sp.n",
            "sp.nc",
            "sp.nsigc",
            "tr.N",
            "tr.M",
            "tr.L",
            "tr.ndom",
            "gtr.N",
            "gtr.M",
            "gtr.L",
            "gtr.ndom",
        ):
            with self.subTest(field=field):
                self.assertTrue(
                    _pipeline._semantic_test_dirty_reusable_state_rejected_bound(
                        pipeline, field
                    )
                )
                self.assertEqual(
                    before,
                    _pipeline._semantic_pipeline_state_encoding_bound(
                        pipeline, profile
                    ),
                )

    def test_dual_oracle_rejects_distinct_immutable_profile_identity(self):
        empty = pyhmmer.easel.DigitalSequenceBlock(self.alphabet)
        changed_profile = self.distinct_optimized_profile()
        original_profile = self.optimized_profile()
        self.assertEqual(original_profile.name, changed_profile.name)
        self.assertEqual(original_profile.accession, changed_profile.accession)
        self.assertEqual(original_profile.M, changed_profile.M)
        left_pipeline = self.pipeline(bias_filter=False)
        right_pipeline = self.pipeline(bias_filter=False)
        left_hits = left_pipeline.search_hmm(original_profile, empty)
        right_hits = right_pipeline.search_hmm(changed_profile, empty)
        with self.assertRaisesRegex(ValueError, "identities differ"):
            _pipeline._semantic_dual_state_compare_bound(
                left_pipeline,
                right_pipeline,
                left_hits,
                right_hits,
                left_optimized_profile=original_profile,
                right_optimized_profile=changed_profile,
            )

    def test_dual_oracle_requires_profiles_even_for_empty_results(self):
        empty = pyhmmer.easel.DigitalSequenceBlock(self.alphabet)
        left_profile = self.optimized_profile()
        right_profile = self.distinct_optimized_profile()
        left_pipeline = self.pipeline(bias_filter=False)
        right_pipeline = self.pipeline(bias_filter=False)
        left_hits = left_pipeline.search_hmm(left_profile, empty)
        right_hits = right_pipeline.search_hmm(right_profile, empty)
        with self.assertRaisesRegex(ValueError, "profiles are required"):
            _pipeline._semantic_dual_state_compare_bound(
                left_pipeline,
                right_pipeline,
                left_hits,
                right_hits,
            )

    def test_dual_oracle_binds_optimized_tophits_queries_to_profiles(self):
        empty = pyhmmer.easel.DigitalSequenceBlock(self.alphabet)
        left_profile = self.optimized_profile()
        right_profile = self.optimized_profile()
        left_pipeline = self.pipeline(bias_filter=False)
        right_pipeline = self.pipeline(bias_filter=False)
        left_hits = left_pipeline.search_hmm(left_profile, empty)
        right_hits = right_pipeline.search_hmm(right_profile, empty)
        unrelated_left = self.distinct_optimized_profile()
        unrelated_right = unrelated_left.copy()
        with self.assertRaisesRegex(ValueError, "TopHits query identity differs"):
            _pipeline._semantic_dual_state_compare_bound(
                left_pipeline,
                right_pipeline,
                left_hits,
                right_hits,
                unrelated_left,
                unrelated_right,
            )

    def test_canonical_encoding_excludes_dormant_and_pointer_fields(self):
        pipeline, profile, hits = self.search()
        pipeline_encoding = _pipeline._semantic_pipeline_state_encoding_bound(
            pipeline, profile
        )
        hits_encoding = _pipeline._semantic_tophits_encoding_bound(hits)

        for forbidden in (
            b"n_output",
            b"pos_output",
            b"strands",
            b"block_length",
            b"window_length",
            b"].seqidx",
            b"subseq_start",
            b"iorf",
            b"jorf",
            b"memsize",
            b"Nalloc",
        ):
            self.assertNotIn(forbidden, pipeline_encoding)
            self.assertNotIn(forbidden, hits_encoding)

        before = _pipeline._semantic_tophits_fingerprint_bound(hits)
        pickle_projection = hits.__getstate__()
        pickle_projection["pipeline"]["n_output"] = 2**63
        pickle_projection["pipeline"]["pos_output"] = 2**63 - 1
        self.assertEqual(
            before,
            _pipeline._semantic_tophits_fingerprint_bound(hits),
        )

        targets = io.BytesIO()
        domains = io.BytesIO()
        hits.write(targets, format="targets", header=True)
        hits.write(domains, format="domains", header=True)
        self.assertTrue(targets.getvalue())
        self.assertTrue(domains.getvalue())

    def test_private_entry_points_reject_nonexact_types_and_half_pairs(self):
        pipeline = self.pipeline()
        left_profile = self.optimized_profile()
        right_profile = self.optimized_profile()
        with self.assertRaises(TypeError):
            _pipeline._semantic_pipeline_state_encoding_bound(object())
        with self.assertRaises(TypeError):
            _pipeline._semantic_tophits_encoding_bound(object())
        with self.assertRaises(ValueError):
            _pipeline._semantic_dual_state_compare_bound(
                pipeline,
                self.pipeline(),
                pyhmmer.plan7.TopHits(self.hmm),
                left_optimized_profile=left_profile,
                right_optimized_profile=right_profile,
            )
        with self.assertRaisesRegex(ValueError, "profiles are required"):
            _pipeline._semantic_dual_state_compare_bound(
                pipeline,
                self.pipeline(),
            )
        with self.assertRaisesRegex(ValueError, "profiles are required"):
            _pipeline._semantic_dual_state_compare_bound(
                pipeline,
                self.pipeline(),
                left_optimized_profile=left_profile,
            )
        with self.assertRaises(TypeError):
            _pipeline._semantic_dual_state_compare_bound(
                pipeline,
                self.pipeline(),
                left_optimized_profile=object(),
                right_optimized_profile=right_profile,
            )


if __name__ == "__main__":
    unittest.main()
