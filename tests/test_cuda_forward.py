import ctypes
import hashlib
import math
import platform
import struct
import sys
import threading
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
FORWARD_FORMAT = "=IfBBH"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


def float32(value):
    return struct.unpack("=f", struct.pack("=f", value))[0]


def bits(value):
    return struct.unpack("=I", struct.pack("=f", value))[0]


def load_profile(name, length=400):
    with pyhmmer.plan7.HMMFile(DATA / "hmms" / "txt" / name) as hmm_file:
        hmm = hmm_file.read()
    background = pyhmmer.plan7.Background(hmm.alphabet)
    return hmm, background, hmm.to_profile(background, L=length).to_optimized()


def digitize(profile, values):
    return [
        pyhmmer.easel.TextSequence(
            name=f"target-{index}".encode(), sequence=value
        ).digitize(profile.alphabet)
        for index, value in enumerate(values)
    ]


def native_batch(sequences, alphabet):
    residues = bytearray()
    offsets = array("Q", [0])
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        offsets.append(len(residues))
    return _native.SequenceBatch(residues, offsets, alphabet.Kp)


def run_forward(
    profiles,
    sequences,
    candidate_offsets,
    indices,
    filters,
    f3,
    gathered_byte_budget=None,
    f3_audit=False,
):
    with _native.ForwardProfiles(profiles) as resident:
        cold = resident.statistics
        with native_batch(sequences, profiles[0].alphabet) as batch:
            arguments = (
                array("Q", candidate_offsets),
                array("I", indices),
                array("f", filters),
                f3,
                profiles,
                resident,
            )
            if gathered_byte_budget is None:
                output = batch.forward_candidates_many_raw(
                    *arguments, _f3_audit=f3_audit
                )
            else:
                output = batch.forward_candidates_many_raw(
                    *arguments, gathered_byte_budget, _f3_audit=f3_audit
                )
    return (*output, cold)


def close_concurrently(value, worker_count=8):
    barrier = threading.Barrier(worker_count)
    failures = []

    def worker():
        try:
            barrier.wait()
            value.close()
        except BaseException as error:
            failures.append(error)

    threads = [threading.Thread(target=worker) for _ in range(worker_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise failures[0]


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaForwardTests(unittest.TestCase):
    def test_workspace_reuses_high_water_and_closes_concurrently(self):
        _, _, profile = load_profile("Thioesterase.hmm")
        sequences = digitize(
            profile,
            ["ACDEX", "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY" * 3],
        )
        batch = native_batch(sequences, profile.alphabet)
        with _native.ForwardProfiles([profile]) as resident:
            initial = batch.workspace_statistics
            self.assertEqual(initial["forward_device_bytes"], 0)
            smaller_arguments = (
                array("Q", [0, 1]),
                array("I", [0]),
                array("f", [0.0]),
                1.0,
                [profile],
                resident,
            )
            smaller = batch.forward_candidates_many_raw(*smaller_arguments)
            self.assertEqual(len(smaller[0]), _native.FORWARD_RESULT_SIZE)
            self.assertIs(type(smaller[0]), bytes)
            for buffer in smaller[1:3]:
                self.assertIs(type(buffer), memoryview)
                self.assertTrue(buffer.readonly)
                self.assertIs(type(buffer.obj), bytes)
            with self.assertRaises(TypeError):
                smaller[1][0] = 0
            if len(smaller[2]):
                with self.assertRaises(TypeError):
                    smaller[2][0] = 0.0
            low_water = batch.workspace_statistics
            self.assertEqual(low_water["forward_growth_count"], 11)
            self.assertEqual(low_water["forward_event_create_count"], 2)
            self.assertEqual(low_water["forward_run_count"], 1)
            arguments = (
                array("Q", [0, 3]),
                array("I", [0, 1, 2]),
                array("f", [0.0, 0.0, 0.0]),
                1.0,
                [profile],
                resident,
            )
            first = batch.forward_candidates_many_raw(*arguments)
            high_water = batch.workspace_statistics
            self.assertGreater(high_water["forward_device_bytes"], 0)
            self.assertGreater(high_water["forward_dp_capacity_bytes"], 0)
            self.assertGreater(high_water["forward_xmx_capacity_bytes"], 0)
            self.assertGreater(
                high_water["forward_growth_count"],
                low_water["forward_growth_count"],
            )
            self.assertEqual(high_water["forward_event_create_count"], 2)
            self.assertEqual(high_water["forward_run_count"], 2)
            snapshot = batch.memory_snapshot
            forward_capacities = [
                value
                for name, value in snapshot["capacity_bytes"].items()
                if name.startswith("forward_")
            ]
            self.assertEqual(
                sum(forward_capacities), high_water["forward_device_bytes"]
            )
            self.assertEqual(
                snapshot["capacity_bytes"]["forward_dp"],
                high_water["forward_dp_capacity_bytes"],
            )
            self.assertEqual(
                snapshot["persistent_device_bytes"],
                sum(snapshot["capacity_bytes"].values()),
            )

            second = batch.forward_candidates_many_raw(*arguments)
            reused = batch.workspace_statistics
            self.assertEqual(second[0], first[0])
            self.assertEqual(list(second[1]), list(first[1]))
            self.assertEqual(second[2].tobytes(), first[2].tobytes())
            for key in (
                "forward_device_bytes",
                "forward_dp_capacity_bytes",
                "forward_xmx_capacity_bytes",
                "forward_gather_capacity_bytes",
                "forward_growth_count",
                "forward_event_create_count",
            ):
                self.assertEqual(reused[key], high_water[key])
            self.assertEqual(reused["forward_run_count"], 3)

            smaller = batch.forward_candidates_many_raw(*smaller_arguments)
            self.assertEqual(len(smaller[0]), _native.FORWARD_RESULT_SIZE)
            shrunk = batch.workspace_statistics
            for key in (
                "forward_device_bytes",
                "forward_dp_capacity_bytes",
                "forward_xmx_capacity_bytes",
                "forward_gather_capacity_bytes",
                "forward_growth_count",
                "forward_event_create_count",
            ):
                self.assertEqual(shrunk[key], high_water[key])
            self.assertEqual(shrunk["forward_run_count"], 4)

        close_concurrently(batch)
        self.assertTrue(batch.closed)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = batch.workspace_statistics

    def test_native_owners_detach_before_concurrent_close(self):
        _, _, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEX"])[0]
        owners = (
            _native.ForwardProfiles([profile]),
            _native.ViterbiProfiles([profile]),
            native_batch([sequence], profile.alphabet),
        )
        for owner in owners:
            with self.subTest(owner=type(owner).__name__):
                close_concurrently(owner)
                self.assertTrue(owner.closed)
                owner.close()

    def test_versioned_abi_and_exact_known_rows(self):
        self.assertEqual(_native.FORWARD_RECORD_VERSION, 1)
        self.assertEqual(_native.FORWARD_RESULT_SIZE, 12)
        self.assertEqual(_native.FORWARD_MAX_GATHERED_BYTES, 384 << 20)
        self.assertEqual(struct.calcsize(FORWARD_FORMAT), 12)

        _, _, first = load_profile("Thioesterase.hmm")
        _, _, second = load_profile("LuxC.hmm")
        with pyhmmer.easel.SequenceFile(
            PROTEINS, digital=True, alphabet=first.alphabet
        ) as sequence_file:
            source = sequence_file.read_block()
        sequences = [source[13], source[67], source[1992]]
        records, offsets, specials, statistics, cold = run_forward(
            [first, second],
            sequences,
            [0, 2, 4],
            [0, 2, 1, 2],
            [0.0] * 4,
            1.0,
        )
        rows = list(struct.iter_unpack(FORWARD_FORMAT, records))
        self.assertEqual([row[0] for row in rows], [0, 2, 1, 2])
        self.assertTrue(all(row[2] == _native.FORWARD_STATUS_OK for row in rows))
        self.assertTrue(all(row[3] == _native.FORWARD_DEFINITE_PASS for row in rows))
        expected_cells = [6 * (len(sequences[row[0]]) + 1) for row in rows]
        self.assertEqual(
            [offsets[index + 1] - offsets[index] for index in range(4)],
            expected_cells,
        )
        self.assertEqual(len(specials), sum(expected_cells))
        self.assertEqual(statistics["survivor_count"], 4)
        self.assertEqual(
            statistics["generation_f3_bits"],
            struct.unpack("=Q", struct.pack("=d", 1.0))[0],
        )
        self.assertEqual(statistics["gathered_xmx_bytes"], len(specials) * 4)
        self.assertEqual(
            statistics["output_byte_limit"],
            _native.FORWARD_MAX_GATHERED_BYTES,
        )
        self.assertLessEqual(statistics["dp_workspace_bytes"], 256 << 20)
        self.assertLessEqual(statistics["xmx_workspace_bytes"], 512 << 20)
        self.assertGreater(cold["device_bytes"], 0)
        self.assertGreaterEqual(cold["pack_ms"], 0.0)

        observed = []
        raw_specials = memoryview(specials).cast("B")
        for index, row in enumerate(rows):
            begin = offsets[index] * 4
            end = offsets[index + 1] * 4
            observed.append(
                (
                    bits(row[1]),
                    hashlib.sha256(raw_specials[begin:end]).hexdigest(),
                )
            )
        # Filled from an independent HMMER 3.4 ForwardParser oracle.
        expected = [
            (
                0xC0E563EA,
                "088b0d48dd1be0227194d600224aff45d141860efbe623231f4fe932eba9c2a6",
            ),
            (
                0x3ED6B927,
                "809661b369520de580bb24de6296545d63d0bb08309505d74a74e8aeaa351cef",
            ),
            (
                0xC10167B2,
                "8a965cc875fec25385ba511876cff22b96c763329fb1fef75c1896993063e3b5",
            ),
            (
                0xC104C717,
                "ced1a0f4487ae9b9c68df3af6a3ba3085cd1f1633ed29f8f598c7e992e2b6327",
            ),
        ]
        self.assertEqual(observed, expected)

    def test_exact_f3_boundary_and_device_gather(self):
        _, _, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEFGHIKLMNPQRSTVWY" * 3])[0]
        records, offsets, specials, statistics, _ = run_forward(
            [profile], [sequence], [0, 1], [0], [0.0], 1.0
        )
        row = struct.unpack(FORWARD_FORMAT, records)
        self.assertEqual(row[3], _native.FORWARD_DEFINITE_PASS)
        self.assertEqual(list(offsets), [0, 6 * (len(sequence) + 1)])
        self.assertEqual(len(specials), offsets[1])
        self.assertEqual(statistics["f3_compiled_profile_count"], 1)
        self.assertEqual(statistics["f3_host_audit_count"], 0)
        self.assertEqual(statistics["f3_host_decision_avoided_count"], 1)
        self.assertEqual(statistics["f3_device_decision_count"], 1)
        self.assertEqual(statistics["f3_device_pass_count"], 1)
        self.assertEqual(statistics["f3_host_fallback_count"], 0)
        self.assertEqual(statistics["f3_decision_mismatch_count"], 0)
        self.assertEqual(statistics["f3_device_compaction_run_count"], 1)
        self.assertEqual(statistics["f3_device_compaction_candidate_count"], 1)
        self.assertEqual(statistics["f3_device_compacted_survivor_count"], 1)
        self.assertEqual(statistics["f3_survivor_upload_avoided_bytes"], 20)

        audited = run_forward(
            [profile], [sequence], [0, 1], [0], [0.0], 1.0,
            f3_audit=True,
        )
        self.assertEqual(audited[0], records)
        self.assertEqual(list(audited[1]), list(offsets))
        self.assertEqual(audited[2].tobytes(), specials.tobytes())
        for key in (
            "row_hash",
            "special_hash",
            "continuation_hash",
            "pass_count",
            "special_count",
        ):
            self.assertEqual(audited[3][key], statistics[key])
        self.assertEqual(audited[3]["f3_host_audit_count"], 1)
        self.assertEqual(audited[3]["f3_host_decision_avoided_count"], 0)
        self.assertEqual(audited[3]["f3_decision_mismatch_count"], 0)

        tau, lambda_ = profile.evalue_parameters.as_vector()[4:6]
        difference = float32(row[1] - 0.0)
        bit_score = float32(float(difference) / 0.69314718055994529)
        probability = 1.0 if bit_score < tau else math.exp(-lambda_ * (bit_score - tau))
        at, at_offsets, at_specials, at_statistics, _ = run_forward(
            [profile], [sequence], [0, 1], [0], [0.0], probability
        )
        below, below_offsets, below_specials, below_statistics, _ = run_forward(
            [profile],
            [sequence],
            [0, 1],
            [0],
            [0.0],
            math.nextafter(probability, -math.inf),
        )
        above, above_offsets, above_specials, above_statistics, _ = run_forward(
            [profile],
            [sequence],
            [0, 1],
            [0],
            [0.0],
            math.nextafter(probability, math.inf),
        )
        self.assertEqual(
            struct.unpack(FORWARD_FORMAT, at)[3],
            _native.FORWARD_DEFINITE_PASS,
        )
        self.assertEqual(
            struct.unpack(FORWARD_FORMAT, below)[3],
            _native.FORWARD_DEFINITE_REJECT,
        )
        self.assertEqual(
            struct.unpack(FORWARD_FORMAT, above)[3],
            _native.FORWARD_DEFINITE_PASS,
        )
        self.assertGreater(at_offsets[1], 0)
        self.assertEqual(list(below_offsets), [0, 0])
        self.assertEqual(len(below_specials), 0)
        self.assertGreater(above_offsets[1], 0)
        self.assertEqual(len(at_specials), len(above_specials))
        self.assertEqual(at_statistics["f3_device_pass_count"], 1)
        self.assertEqual(below_statistics["f3_device_reject_count"], 1)
        for observed in (at_statistics, below_statistics):
            self.assertEqual(observed["f3_host_audit_count"], 0)
            self.assertEqual(observed["f3_host_decision_avoided_count"], 1)
            self.assertEqual(observed["f3_device_decision_count"], 1)
            self.assertEqual(observed["f3_host_fallback_count"], 0)
            self.assertEqual(observed["f3_decision_mismatch_count"], 0)
        self.assertEqual(above_statistics["f3_compiled_profile_count"], 0)
        self.assertEqual(above_statistics["f3_unsupported_profile_count"], 1)
        self.assertEqual(above_statistics["f3_device_decision_count"], 0)
        self.assertEqual(above_statistics["f3_host_fallback_count"], 1)

    def test_unsupported_f3_values_fail_closed_with_exact_provenance(self):
        _, _, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEX"])[0]
        for f3 in (math.nan, math.inf, -math.inf, -1.0):
            with self.subTest(f3=f3):
                records, offsets, specials, statistics, _ = run_forward(
                    [profile], [sequence], [0, 1], [0], [0.0], f3
                )
                row = struct.unpack(FORWARD_FORMAT, records)
                self.assertEqual(
                    (row[2], row[3]),
                    (
                        _native.FORWARD_STATUS_ENORESULT,
                        _native.FORWARD_CPU_REQUIRED,
                    ),
                )
                self.assertTrue(math.isnan(row[1]))
                self.assertEqual(list(offsets), [0, 0])
                self.assertEqual(len(specials), 0)
                self.assertEqual(
                    statistics["generation_f3_bits"],
                    struct.unpack("=Q", struct.pack("=d", f3))[0],
                )

    def test_gather_budget_fails_later_passes_closed(self):
        _, _, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEFGHIKLMNPQRSTVWY"])[0]
        cells = 6 * (len(sequence) + 1)
        records, offsets, specials, statistics, _ = run_forward(
            [profile],
            [sequence],
            [0, 3],
            [0, 0, 0],
            [0.0, 0.0, 0.0],
            1.0,
            cells * 4,
        )
        rows = list(struct.iter_unpack(FORWARD_FORMAT, records))
        self.assertEqual(
            [row[3] for row in rows],
            [
                _native.FORWARD_DEFINITE_PASS,
                _native.FORWARD_CPU_REQUIRED,
                _native.FORWARD_CPU_REQUIRED,
            ],
        )
        self.assertTrue(all(row[2] == _native.FORWARD_STATUS_OK for row in rows))
        self.assertTrue(all(math.isfinite(row[1]) for row in rows))
        self.assertEqual(list(offsets), [0, cells, cells, cells])
        self.assertEqual(len(specials), cells)
        self.assertEqual(statistics["output_byte_limit"], cells * 4)
        self.assertEqual(statistics["output_cap_fallback_count"], 2)

    def test_empty_and_forward_range_rows_fail_closed(self):
        alphabet = pyhmmer.easel.Alphabet.amino()
        hmm = pyhmmer.plan7.HMM.sample(alphabet, 1, 42)
        for residue in range(alphabet.K):
            hmm.match_emissions[1, residue] = 0.0
        hmm.match_emissions[1, 0] = 1.0
        hmm.renormalize()
        background = pyhmmer.plan7.Background(alphabet)
        profile = hmm.to_profile(background, L=100).to_optimized()
        sequences = digitize(profile, ["", "C" * 20])
        records, offsets, specials, _, _ = run_forward(
            [profile], sequences, [0, 2], [0, 1], [0.0, 0.0], 1.0
        )
        empty, underflow = struct.iter_unpack(FORWARD_FORMAT, records)
        self.assertEqual(
            (empty[2], empty[3]),
            (_native.FORWARD_STATUS_EMPTY, _native.FORWARD_CPU_REQUIRED),
        )
        self.assertTrue(math.isnan(empty[1]))
        self.assertEqual(
            (underflow[2], underflow[3]),
            (_native.FORWARD_STATUS_ERANGE, _native.FORWARD_CPU_REQUIRED),
        )
        self.assertTrue(math.isnan(underflow[1]))
        self.assertEqual(list(offsets), [0, 0, 0])
        self.assertEqual(len(specials), 0)

    def test_source_identity_mutation_and_malformed_csr_are_rejected(self):
        hmm, background, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEX"])[0]
        other = profile.copy()
        with _native.ForwardProfiles([profile]) as resident:
            with native_batch([sequence], profile.alphabet) as batch:
                with self.assertRaisesRegex(ValueError, "identity"):
                    batch.forward_candidates_many_raw(
                        array("Q", [0, 1]),
                        array("I", [0]),
                        array("f", [0.0]),
                        1.0,
                        [other],
                        resident,
                    )
                with self.assertRaisesRegex(RuntimeError, "offsets"):
                    batch.forward_candidates_many_raw(
                        array("Q", [0, 2]),
                        array("I", [0]),
                        array("f", [0.0]),
                        1.0,
                        [profile],
                        resident,
                    )
                with self.assertRaisesRegex(RuntimeError, "out of range"):
                    batch.forward_candidates_many_raw(
                        array("Q", [0, 1]),
                        array("I", [1]),
                        array("f", [0.0]),
                        1.0,
                        [profile],
                        resident,
                    )
                changed = hmm.copy()
                changed.match_emissions[1, 0] = 0.0
                changed.renormalize()
                profile.convert(changed.to_profile(background, L=400))
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    batch.forward_candidates_many_raw(
                        array("Q", [0, 1]),
                        array("I", [0]),
                        array("f", [0.0]),
                        1.0,
                        [profile],
                        resident,
                    )

    @unittest.skipUnless(
        platform.system() == "Linux" and platform.machine() == "x86_64",
        "fenv layout is Linux x86_64 specific",
    )
    def test_unattested_consumption_environment_fails_closed(self):
        _, _, profile = load_profile("Thioesterase.hmm")
        sequence = digitize(profile, ["ACDEX"])[0]
        libc = ctypes.CDLL(None)
        libc.fegetround.restype = ctypes.c_int
        libc.fesetround.argtypes = [ctypes.c_int]
        original = libc.fegetround()
        with _native.ForwardProfiles([profile]) as resident:
            with native_batch([sequence], profile.alphabet) as batch:
                try:
                    self.assertEqual(libc.fesetround(0x400), 0)
                    records, offsets, specials, _ = batch.forward_candidates_many_raw(
                        array("Q", [0, 1]),
                        array("I", [0]),
                        array("f", [0.0]),
                        1.0,
                        [profile],
                        resident,
                    )
                finally:
                    self.assertEqual(libc.fesetround(original), 0)
        row = struct.unpack(FORWARD_FORMAT, records)
        self.assertEqual(
            (row[2], row[3]),
            (_native.FORWARD_STATUS_ENORESULT, _native.FORWARD_CPU_REQUIRED),
        )
        self.assertTrue(math.isnan(row[1]))
        self.assertEqual(list(offsets), [0, 0])
        self.assertEqual(len(specials), 0)


if __name__ == "__main__":
    unittest.main()
