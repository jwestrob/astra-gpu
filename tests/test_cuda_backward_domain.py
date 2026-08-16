import math
import struct
import sys
import unittest
from array import array
from pathlib import Path

import pyhmmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import _native
except ImportError:
    _native = None


DATA = Path(pyhmmer.__file__).parent / "tests" / "data"
PROTEINS = DATA / "seqs" / "938293.PRJEB85.HG003687.faa"
RESULT_FORMAT = "=IIffIIIBBBB"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


def load_profile(name, length):
    with pyhmmer.plan7.HMMFile(DATA / "hmms" / "txt" / name) as hmm_file:
        hmm = hmm_file.read()
    background = pyhmmer.plan7.Background(hmm.alphabet)
    return hmm.to_profile(background, L=length).to_optimized()


def load_real_sequences(alphabet, count=4):
    sequences = []
    with pyhmmer.easel.SequenceFile(PROTEINS, digital=True, alphabet=alphabet) as f:
        for sequence in f:
            sequences.append(sequence)
            if len(sequences) == count:
                break
    return sequences


def native_batch(sequences, alphabet):
    residues = bytearray()
    offsets = array("Q", [0])
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        offsets.append(len(residues))
    return _native.SequenceBatch(residues, offsets, alphabet.Kp), residues, offsets


def unpack_results(records):
    return list(struct.iter_unpack(RESULT_FORMAT, records))


def posterior_rows(flat, begin, end):
    values = flat[begin * 3 : end * 3]
    return list(zip(values[0::3], values[1::3], values[2::3]))


def quantile(values, probability):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(probability * len(ordered)))]


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaBackwardDomainTests(unittest.TestCase):
    def forward_rows(self, profiles, sequences, rows):
        candidate_offsets = [0]
        candidate_indices = []
        for profile_index in range(len(profiles)):
            selected = [
                sequence_index
                for row_profile, sequence_index in rows
                if row_profile == profile_index
            ]
            candidate_indices.extend(selected)
            candidate_offsets.append(len(candidate_indices))
        batch, residues, residue_offsets = native_batch(
            sequences, profiles[0].alphabet
        )
        resident = _native.ForwardProfiles(profiles)
        records, offsets, specials, forward_statistics = (
            batch.forward_candidates_many_raw(
            array("Q", candidate_offsets),
            array("I", candidate_indices),
            array("f", [0.0] * len(candidate_indices)),
            1.0,
            profiles,
            resident,
            )
        )
        forward_results = list(struct.iter_unpack("=IfBBH", records))
        self.assertTrue(
            all(row[3] == _native.FORWARD_DEFINITE_PASS for row in forward_results)
        )
        provenance = forward_statistics["_provenance"]
        return (
            batch, resident, residues, residue_offsets,
            offsets, specials, provenance,
        )

    def test_real_fixture_error_distribution_and_region_decisions(self):
        profile_names = ["RREFam.hmm", "Thioesterase.hmm", "KR.hmm", "LuxC.hmm"]
        provisional = load_profile(profile_names[0], 400)
        sequences = load_real_sequences(provisional.alphabet, 4)
        profiles = [load_profile(name, 400) for name in profile_names]
        rows = [
            (profile_index, sequence_index)
            for profile_index in range(len(profiles))
            for sequence_index in range(len(sequences))
        ]
        batch, resident, residues, residue_offsets, fwd_offsets, specials, provenance = (
            self.forward_rows(profiles, sequences, rows)
        )
        try:
            records, offsets, gpu_flat, region_offsets, regions, statistics = (
                batch.backward_domain_many_raw(
                    array("I", [row[0] for row in rows]),
                    array("I", [row[1] for row in rows]),
                    fwd_offsets,
                    specials,
                    resident,
                    provenance,
                    guard_band=0.0,
                )
            )
            gpu_results = unpack_results(records)
            errors = []
            decision_mismatches = 0
            region_mismatches = 0
            decision_guard = 2.0e-4
            for candidate, (profile_index, sequence_index) in enumerate(rows):
                residue_begin = residue_offsets[sequence_index]
                residue_end = residue_offsets[sequence_index + 1]
                row_residues = memoryview(residues)[residue_begin:residue_end]
                cpu_record, cpu_flat, cpu_regions = (
                    _native.backward_domain_cpu_oracle_raw(
                    profiles[profile_index],
                    row_residues,
                    specials[fwd_offsets[candidate] : fwd_offsets[candidate + 1]],
                    guard_band=0.0,
                    )
                )
                cpu_result = struct.unpack(RESULT_FORMAT, cpu_record)
                gpu_result = gpu_results[candidate]
                cpu_rows = posterior_rows(cpu_flat, 0, len(cpu_flat) // 3)
                gpu_rows = posterior_rows(
                    gpu_flat, offsets[candidate], offsets[candidate + 1]
                )
                self.assertEqual(len(gpu_rows), len(cpu_rows))
                self.assertEqual(gpu_result[9], cpu_result[9])
                self.assertEqual(gpu_result[8], cpu_result[8])
                self.assertEqual(gpu_result[3], cpu_result[3])
                if gpu_result[5] != cpu_result[5] or gpu_result[6] != cpu_result[6]:
                    region_mismatches += 1
                if gpu_result[8] == _native.BACKWARD_DOMAIN_SIMPLE:
                    self.assertEqual(
                        list(regions[
                            region_offsets[candidate] * 2 :
                            region_offsets[candidate + 1] * 2
                        ]),
                        list(cpu_regions),
                    )
                for gpu_values, cpu_values in zip(gpu_rows, cpu_rows):
                    errors.extend(
                        abs(float(gpu) - float(cpu))
                        for gpu, cpu in zip(gpu_values, cpu_values)
                    )
                    gpu_bocc = gpu_values[0]
                    cpu_bocc = cpu_values[0]
                    # The cumulative values themselves feed rt3; rt1 is the
                    # most sensitive direct region-trigger comparison.
                    if (
                        min(abs(gpu_values[2] - 0.25), abs(cpu_values[2] - 0.25))
                        > decision_guard
                        and (gpu_values[2] >= 0.25) != (cpu_values[2] >= 0.25)
                    ):
                        decision_mismatches += 1
                    self.assertTrue(math.isfinite(gpu_bocc + cpu_bocc))
            self.assertEqual(decision_mismatches, 0)
            self.assertEqual(region_mismatches, 0)
            self.assertLessEqual(max(errors), 1.0e-6)
            self.assertLessEqual(quantile(errors, 0.99), 1.0e-7)
            self.assertLessEqual(quantile(errors, 0.50), 1.0e-7)
            self.assertEqual(statistics["candidate_count"], len(rows))
            self.assertEqual(
                statistics["device_result_count"]
                + statistics["cpu_required_count"],
                len(rows),
            )
            self.assertEqual(
                statistics["multidomain_fallback_count"],
                sum(row[6] != 0 for row in gpu_results),
            )
        finally:
            resident.close()
            batch.close()

    def test_model_stripe_boundaries_match_oracle(self):
        alphabet = pyhmmer.easel.Alphabet.amino()
        sequences = load_real_sequences(alphabet, 1)
        length = len(sequences[0])
        background = pyhmmer.plan7.Background(alphabet)
        lengths = (1, 3, 4, 5, 7, 8, 9, 31, 32, 33)
        profiles = [
            pyhmmer.plan7.HMM.sample(alphabet, model_length, 1000 + model_length)
            for model_length in lengths
        ]
        for hmm in profiles:
            hmm.evalue_parameters.f_tau = 0.0
            hmm.evalue_parameters.f_lambda = 1.0
        profiles = [
            hmm.to_profile(background, L=length).to_optimized()
            for hmm in profiles
        ]
        rows = [(profile_index, 0) for profile_index in range(len(profiles))]
        batch, resident, residues, residue_offsets, fwd_offsets, specials, provenance = (
            self.forward_rows(profiles, sequences, rows)
        )
        try:
            records, offsets, gpu_flat, _, _, _ = batch.backward_domain_many_raw(
                array("I", range(len(profiles))),
                array("I", [0] * len(profiles)),
                fwd_offsets,
                specials,
                resident,
                provenance,
                guard_band=0.0,
            )
            for candidate, profile in enumerate(profiles):
                cpu_record, cpu_flat, _ = _native.backward_domain_cpu_oracle_raw(
                    profile,
                    memoryview(residues),
                    specials[fwd_offsets[candidate] : fwd_offsets[candidate + 1]],
                    guard_band=0.0,
                )
                gpu_record = records[
                    candidate * _native.BACKWARD_DOMAIN_RESULT_SIZE :
                    (candidate + 1) * _native.BACKWARD_DOMAIN_RESULT_SIZE
                ]
                self.assertEqual(
                    struct.unpack(RESULT_FORMAT, gpu_record)[9],
                    struct.unpack(RESULT_FORMAT, cpu_record)[9],
                )
                gpu_rows = posterior_rows(
                    gpu_flat, offsets[candidate], offsets[candidate + 1]
                )
                cpu_rows = posterior_rows(cpu_flat, 0, len(cpu_flat) // 3)
                maximum = max(
                    abs(float(gpu) - float(cpu))
                    for gpu_row, cpu_row in zip(gpu_rows, cpu_rows)
                    for gpu, cpu in zip(gpu_row, cpu_row)
                )
                self.assertLessEqual(maximum, 1.0e-6, f"M={profile.M}")
        finally:
            resident.close()
            batch.close()

    def test_target_boundaries_empty_and_closed_lifetimes(self):
        profile = load_profile("Thioesterase.hmm", 64)
        source = load_real_sequences(profile.alphabet, 1)[0].textize().sequence
        lengths = (1, 2, 3, 4, 5, 31, 32, 33)
        sequences = [
            pyhmmer.easel.TextSequence(
                name=f"real-prefix-{length}".encode(),
                sequence=source[:length],
            ).digitize(profile.alphabet)
            for length in lengths
        ]
        rows = [(0, sequence_index) for sequence_index in range(len(sequences))]
        batch, resident, residues, residue_offsets, fwd_offsets, specials, provenance = (
            self.forward_rows([profile], sequences, rows)
        )
        try:
            records, offsets, gpu_flat, _, _, _ = batch.backward_domain_many_raw(
                array("I", [0] * len(sequences)),
                array("I", range(len(sequences))),
                fwd_offsets,
                specials,
                resident,
                provenance,
                guard_band=0.0,
            )
            gpu_results = unpack_results(records)
            for candidate, sequence in enumerate(sequences):
                begin = residue_offsets[candidate]
                end = residue_offsets[candidate + 1]
                cpu_record, cpu_flat, _ = _native.backward_domain_cpu_oracle_raw(
                    profile,
                    memoryview(residues)[begin:end],
                    specials[fwd_offsets[candidate] : fwd_offsets[candidate + 1]],
                    guard_band=0.0,
                )
                self.assertEqual(
                    gpu_flat[offsets[candidate] * 3 : offsets[candidate + 1] * 3],
                    cpu_flat,
                )
                self.assertEqual(
                    gpu_results[candidate][8],
                    struct.unpack(RESULT_FORMAT, cpu_record)[8],
                )
        finally:
            resident.close()
            batch.close()

        empty = pyhmmer.easel.TextSequence(
            name=b"empty", sequence=""
        ).digitize(profile.alphabet)
        empty_batch, _, _ = native_batch([empty], profile.alphabet)
        empty_resident = _native.ForwardProfiles([profile])
        try:
            f_records, _, f_specials, empty_statistics = (
                empty_batch.forward_candidates_many_raw(
                    array("Q", [0, 1]), array("I", [0]),
                    array("f", [0.0]), 1.0, [profile], empty_resident,
                )
            )
            empty_provenance = empty_statistics["_provenance"]
            self.assertEqual(
                struct.unpack("=IfBBH", f_records)[3],
                _native.FORWARD_CPU_REQUIRED,
            )
            records, offsets, posteriors, region_offsets, regions, statistics = (
                empty_batch.backward_domain_many_raw(
                    array("I"), array("I"), array("Q", [0]), f_specials,
                    empty_resident, empty_provenance,
                )
            )
            self.assertEqual(records, b"")
            self.assertEqual(list(offsets), [0])
            self.assertEqual(list(region_offsets), [0])
            self.assertEqual((len(posteriors), len(regions)), (0, 0))
            self.assertEqual(statistics["candidate_count"], 0)
        finally:
            empty_resident.close()
            empty_batch.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            empty_batch.backward_domain_many_raw(
                array("I"), array("I"), array("Q", [0]), array("f"),
                empty_resident, empty_provenance,
            )

    def test_maximum_model_and_target_bounds_match_oracle(self):
        alphabet = pyhmmer.easel.Alphabet.amino()
        background = pyhmmer.plan7.Background(alphabet)

        def check(profile, sequence, label):
            batch, resident, residues, _, fwd_offsets, specials, provenance = (
                self.forward_rows([profile], [sequence], [(0, 0)])
            )
            try:
                record, offsets, gpu, _, _, _ = (
                    batch.backward_domain_many_raw(
                        array("I", [0]), array("I", [0]),
                        fwd_offsets, specials, resident, provenance,
                        guard_band=0.0,
                    )
                )
                cpu_record, cpu, _ = _native.backward_domain_cpu_oracle_raw(
                    profile, memoryview(residues), specials, guard_band=0.0,
                )
                self.assertEqual(gpu, cpu, label)
                self.assertEqual(
                    struct.unpack(RESULT_FORMAT, record)[7:10],
                    struct.unpack(RESULT_FORMAT, cpu_record)[7:10],
                    label,
                )
                self.assertEqual(list(offsets), [0, len(sequence) + 1])
            finally:
                resident.close()
                batch.close()

        one = pyhmmer.easel.TextSequence(
            name=b"one", sequence="A"
        ).digitize(alphabet)
        for model_length in (99_999, 100_000):
            hmm = pyhmmer.plan7.HMM.sample(
                alphabet, model_length, 90_000 + model_length
            )
            hmm.evalue_parameters.f_tau = 0.0
            hmm.evalue_parameters.f_lambda = 1.0
            profile = hmm.to_profile(background, L=1).to_optimized()
            check(profile, one, f"M={model_length},L=1")

        hmm = pyhmmer.plan7.HMM.sample(alphabet, 3, 100_003)
        hmm.evalue_parameters.f_tau = 0.0
        hmm.evalue_parameters.f_lambda = 1.0
        profile = hmm.to_profile(background, L=100_000).to_optimized()
        long_sequence = pyhmmer.easel.TextSequence(
            name=b"maximum-target", sequence="A" * 100_000
        ).digitize(alphabet)
        check(profile, long_sequence, "M=3,L=100000")

    def test_row_and_run_work_caps_fall_back_without_launching(self):
        alphabet = pyhmmer.easel.Alphabet.amino()
        background = pyhmmer.plan7.Background(alphabet)

        def sampled_profile(model_length, target_length, seed):
            hmm = pyhmmer.plan7.HMM.sample(alphabet, model_length, seed)
            hmm.evalue_parameters.f_tau = 0.0
            hmm.evalue_parameters.f_lambda = 1.0
            return hmm.to_profile(
                background, L=target_length
            ).to_optimized()

        profile = sampled_profile(1_000, 10_001, 210_001)
        sequences = [
            pyhmmer.easel.TextSequence(
                name=f"row-cap-{length}".encode(), sequence="A" * length
            ).digitize(alphabet)
            for length in (10_000, 10_001)
        ]
        batch, resident, _, _, fwd_offsets, specials, provenance = (
            self.forward_rows([profile], sequences, [(0, 0), (0, 1)])
        )
        try:
            records, offsets, _, _, _, statistics = (
                batch.backward_domain_many_raw(
                    array("I", [0, 0]), array("I", [0, 1]),
                    fwd_offsets, specials, resident, provenance,
                    guard_band=0.0,
                )
            )
            results = unpack_results(records)
            self.assertEqual(
                statistics["work_cells"],
                _native.BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS,
            )
            self.assertEqual(statistics["work_cap_fallback_count"], 1)
            self.assertGreater(offsets[1], offsets[0])
            self.assertEqual(offsets[2], offsets[1])
            self.assertEqual(results[1][8], _native.BACKWARD_DOMAIN_CPU_REQUIRED)
            self.assertTrue(math.isnan(results[1][2]))
        finally:
            resident.close()
            batch.close()

        sequences = [
            pyhmmer.easel.TextSequence(
                name=f"run-cap-{index}".encode(), sequence="A" * 1_000
            ).digitize(alphabet)
            for index in range(269)
        ]
        batch, resident, _, _, fwd_offsets, specials, provenance = (
            self.forward_rows(
                [profile], sequences,
                [(0, sequence_index) for sequence_index in range(269)],
            )
        )
        try:
            records, offsets, _, _, _, statistics = (
                batch.backward_domain_many_raw(
                    array("I", [0] * 269), array("I", range(269)),
                    fwd_offsets, specials, resident, provenance,
                    guard_band=0.0,
                )
            )
            results = unpack_results(records)
            self.assertEqual(statistics["work_cells"], 268_000_000)
            self.assertLessEqual(
                statistics["work_cells"],
                _native.BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS,
            )
            self.assertEqual(statistics["work_cap_fallback_count"], 1)
            self.assertGreater(offsets[268], offsets[267])
            self.assertEqual(offsets[269], offsets[268])
            self.assertEqual(results[268][8], _native.BACKWARD_DOMAIN_CPU_REQUIRED)
            self.assertTrue(math.isnan(results[268][2]))
        finally:
            resident.close()
            batch.close()

    def test_guard_band_bad_numeric_output_cap_and_order_fail_closed(self):
        profile = load_profile("Thioesterase.hmm", 400)
        sequences = load_real_sequences(profile.alphabet, 2)
        rows = [(0, 0), (0, 1)]
        batch, resident, _, _, fwd_offsets, specials, provenance = self.forward_rows(
            [profile], sequences, rows
        )
        try:
            records, offsets, flat, _, _, _ = batch.backward_domain_many_raw(
                array("I", [0, 0]),
                array("I", [0, 1]),
                fwd_offsets,
                specials,
                resident,
                provenance,
                guard_band=0.0,
            )
            base_rows = unpack_results(records)
            first_mocc = flat[(offsets[0] + 1) * 3 + 2]
            guarded, guarded_offsets, guarded_flat, _, _, guarded_stats = (
                batch.backward_domain_many_raw(
                    array("I", [0, 0]),
                    array("I", [0, 1]),
                    fwd_offsets,
                    specials,
                    resident,
                    provenance,
                    rt1=first_mocc,
                    guard_band=0.0,
                )
            )
            guarded_rows = unpack_results(guarded)
            self.assertEqual(
                guarded_rows[0][8], _native.BACKWARD_DOMAIN_CPU_REQUIRED
            )
            self.assertGreater(guarded_rows[0][4], 0)
            self.assertGreater(guarded_offsets[1], guarded_offsets[0])
            self.assertGreaterEqual(
                guarded_stats["threshold_uncertain_count"], 1
            )

            first_bytes = (len(sequences[0]) + 1) * 12
            capped, capped_offsets, _, _, _, capped_stats = (
                batch.backward_domain_many_raw(
                    array("I", [0, 0]),
                    array("I", [0, 1]),
                    fwd_offsets,
                    specials,
                    resident,
                    provenance,
                    guard_band=0.0,
                    posterior_byte_budget=first_bytes,
                )
            )
            capped_rows = unpack_results(capped)
            self.assertEqual(capped_rows[0][8], base_rows[0][8])
            self.assertEqual(capped_rows[1][8], base_rows[1][8])
            self.assertEqual(capped_offsets[1], capped_offsets[2])
            self.assertEqual(capped_stats["posterior_omitted_count"], 1)

            corrupted = array("f", specials)
            corrupted[5] = math.inf
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                batch.backward_domain_many_raw(
                    array("I", [0, 0]),
                    array("I", [0, 1]),
                    fwd_offsets,
                    corrupted,
                    resident,
                    provenance,
                    guard_band=0.0,
                )

            first_cells = fwd_offsets[1] - fwd_offsets[0]
            duplicate_specials = array(
                "f", specials[fwd_offsets[0] : fwd_offsets[1]]
            )
            duplicate_specials.extend(duplicate_specials)
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                batch.backward_domain_many_raw(
                    array("I", [0, 0]),
                    array("I", [0, 0]),
                    array("Q", [0, first_cells, first_cells * 2]),
                    duplicate_specials,
                    resident,
                    provenance,
                )

            other_resident = _native.ForwardProfiles([profile])
            try:
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    batch.backward_domain_many_raw(
                        array("I", [0, 0]), array("I", [0, 1]),
                        fwd_offsets, specials, other_resident, provenance,
                    )
            finally:
                other_resident.close()

            other_batch, _, _ = native_batch(sequences, profile.alphabet)
            try:
                with self.assertRaisesRegex(RuntimeError, "provenance"):
                    other_batch.backward_domain_many_raw(
                        array("I", [0, 0]), array("I", [0, 1]),
                        fwd_offsets, specials, resident, provenance,
                    )
            finally:
                other_batch.close()

            for field in (
                "continuation_hash",
                "generation_f3_bits",
                "integrity_tag",
            ):
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeError, "provenance"):
                        batch.backward_domain_many_raw(
                            array("I", [0, 0]), array("I", [0, 1]),
                            fwd_offsets, specials, resident,
                            provenance._tampered_for_test(field),
                        )
        finally:
            resident.close()
            batch.close()

    def test_unihit_profile_is_not_routed_to_simple_continuation(self):
        with pyhmmer.plan7.HMMFile(
            DATA / "hmms" / "txt" / "Thioesterase.hmm"
        ) as hmm_file:
            hmm = hmm_file.read()
        background = pyhmmer.plan7.Background(hmm.alphabet)
        generic = pyhmmer.plan7.Profile(hmm.M, hmm.alphabet)
        generic.configure(
            hmm, background, L=400, multihit=False, local=True
        )
        profile = generic.to_optimized()
        sequences = load_real_sequences(profile.alphabet, 1)
        batch, resident, _, _, fwd_offsets, specials, provenance = (
            self.forward_rows([profile], sequences, [(0, 0)])
        )
        try:
            records, _, _, region_offsets, regions, statistics = (
                batch.backward_domain_many_raw(
                    array("I", [0]), array("I", [0]),
                    fwd_offsets, specials, resident, provenance,
                )
            )
            result = unpack_results(records)[0]
            self.assertEqual(
                result[8], _native.BACKWARD_DOMAIN_CPU_REQUIRED
            )
            self.assertEqual(list(region_offsets), [0, 0])
            self.assertEqual(len(regions), 0)
            self.assertEqual(statistics["cpu_required_count"], 1)
        finally:
            resident.close()
            batch.close()

    def test_profile_session_snapshot_stages_without_live_profile_pointers(self):
        with pyhmmer.plan7.HMMFile(
            DATA / "hmms" / "txt" / "Thioesterase.hmm"
        ) as hmm_file:
            hmm = hmm_file.read()
        background = pyhmmer.plan7.Background(hmm.alphabet)
        profile = hmm.to_profile(background, L=400).to_optimized()
        sequences = load_real_sequences(profile.alphabet, 1)
        batch, residues, _ = native_batch(sequences, profile.alphabet)
        session = _native.ProfileSession(
            [profile], memoryview(background.residue_frequencies), 1, 1
        )
        selection = session.select([0])
        staged = _native.ForwardProfiles(selection)
        selection.close()
        session.close()
        try:
            _, fwd_offsets, specials, forward_statistics = (
                batch.forward_candidates_many_raw(
                    array("Q", [0, 1]), array("I", [0]),
                    array("f", [0.0]), 1.0, None, staged,
                )
            )
            provenance = forward_statistics["_provenance"]
            records, offsets, gpu_flat, _, _, _ = batch.backward_domain_many_raw(
                array("I", [0]),
                array("I", [0]),
                fwd_offsets,
                specials,
                staged,
                provenance,
                guard_band=0.0,
            )
            self.assertEqual(
                unpack_results(records)[0][8] in (
                    _native.BACKWARD_DOMAIN_NO_REGIONS,
                    _native.BACKWARD_DOMAIN_SIMPLE,
                ),
                True,
            )
            _, cpu_flat, _ = _native.backward_domain_cpu_oracle_raw(
                profile, memoryview(residues), specials, guard_band=0.0
            )
            self.assertEqual(list(offsets), [0, len(sequences[0]) + 1])
            self.assertEqual(gpu_flat, cpu_flat)
        finally:
            staged.close()
            batch.close()

    def test_rescaling_and_backward_own_scale_route(self):
        profile = load_profile("Thioesterase.hmm", 4000)
        consensus = profile.consensus.replace("-", "")
        sequence = pyhmmer.easel.TextSequence(
            name=b"real-model-consensus-repeat",
            sequence=(consensus * 12)[:4000],
        ).digitize(profile.alphabet)
        batch, resident, residues, _, fwd_offsets, specials, provenance = self.forward_rows(
            [profile], [sequence], [(0, 0)]
        )
        try:
            self.assertTrue(any(specials[i] > 1.0 for i in range(5, len(specials), 6)))
            modified = array("f", specials)
            for i in range(5, len(modified), 6):
                modified[i] = 1.0
            # Modified Forward state deliberately invalidates its provenance.
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                batch.backward_domain_many_raw(
                    array("I", [0]),
                    array("I", [0]),
                    fwd_offsets,
                    modified,
                    resident,
                    provenance,
                    guard_band=0.0,
                )
            # The pristine CPU seam still exercises HMMER's own-scale route.
            cpu_record, cpu_flat, _ = _native.backward_domain_cpu_oracle_raw(
                profile,
                memoryview(residues),
                modified,
                guard_band=0.0,
            )
            cpu_result = struct.unpack(RESULT_FORMAT, cpu_record)
            self.assertEqual(cpu_result[9], 1)

            # A deliberately unsealed test seam exercises the CUDA own-scale
            # branch without allowing the synthetic state to produce a
            # consumable route or provenance seal.
            (
                synthetic_records,
                synthetic_offsets,
                synthetic_gpu,
                synthetic_region_offsets,
                synthetic_regions,
                synthetic_statistics,
            ) = batch.backward_domain_many_raw(
                array("I", [0]),
                array("I", [0]),
                fwd_offsets,
                modified,
                resident,
                provenance,
                guard_band=0.0,
                _unsealed_test=True,
            )
            synthetic_result = unpack_results(synthetic_records)[0]
            self.assertEqual(synthetic_result[9], 1)
            self.assertEqual(synthetic_result[7], cpu_result[7])
            self.assertEqual(
                synthetic_result[8], _native.BACKWARD_DOMAIN_CPU_REQUIRED
            )
            self.assertEqual(synthetic_gpu, cpu_flat)
            self.assertEqual(
                list(synthetic_offsets), [0, len(sequence) + 1]
            )
            self.assertEqual(list(synthetic_region_offsets), [0, 0])
            self.assertEqual(len(synthetic_regions), 0)
            self.assertTrue(synthetic_statistics["unsealed_test"])
            self.assertEqual(synthetic_statistics["own_scale_count"], 1)

            records, _, _, _, _, _ = batch.backward_domain_many_raw(
                array("I", [0]),
                array("I", [0]),
                fwd_offsets,
                specials,
                resident,
                provenance,
                guard_band=0.0,
            )
            gpu_result = unpack_results(records)[0]
            cpu_record, _, _ = _native.backward_domain_cpu_oracle_raw(
                profile,
                memoryview(residues),
                specials,
                guard_band=0.0,
            )
            cpu_result = struct.unpack(RESULT_FORMAT, cpu_record)
            self.assertEqual(gpu_result[9], cpu_result[9])
            self.assertEqual(gpu_result[7], cpu_result[7])
        finally:
            resident.close()
            batch.close()


if __name__ == "__main__":
    unittest.main()
