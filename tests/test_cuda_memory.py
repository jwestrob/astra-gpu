import os
import subprocess
import sys
import unittest
from array import array
from pathlib import Path

import pyhmmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

try:
    from plan7_gpu import SequenceBatch, _native
    from plan7_gpu.adapter import _pack_profiles, _sequence_native
except ImportError:
    SequenceBatch = None
    _native = None
    _pack_profiles = None
    _sequence_native = None


DATA = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
HMM_LARGE = DATA / "LuxC.hmm"
HMM_SMALL = DATA / "RREFam.hmm"


def cuda_available():
    if _native is None:
        return False
    try:
        return _native.device_count() > 0
    except RuntimeError:
        return False


@unittest.skipUnless(_native is not None, "CUDA extension unavailable")
class ProfileMemoryPlanningTests(unittest.TestCase):
    def test_real_pfam_first_slice_footprint(self):
        # PFAM SHA-256 a78a42d6faf265b6bfca59e8f062d06fae6083ce2c6e335d7b381f20b82b7903,
        # profile ordinals [0, 55).
        lengths = (
            121, 41, 345, 102, 235, 474, 430, 223, 115, 30, 57,
            134, 178, 232, 41, 226, 140, 191, 146, 106, 353, 142,
            202, 120, 141, 35, 94, 34, 145, 116, 124, 416, 173,
            101, 194, 98, 98, 101, 211, 192, 79, 70, 152, 116, 151,
            132, 138, 259, 237, 280, 72, 97, 100, 688, 49,
        )
        self.assertEqual(
            _native.profile_footprint(lengths),
            {
                "profile_count": 55,
                "ssv_device_bytes": 269_033,
                "viterbi_device_bytes": 763_040,
                "viterbi_exact_rbv_upper_bytes": 269_033,
                "forward_device_bytes": 1_387_632,
                "bias_device_bytes": 14_960,
                "minimum_device_bytes": 2_434_665,
                "maximum_device_bytes": 2_703_698,
            },
        )

    def test_profile_cell_boundary_for_full_target_corpus(self):
        self.assertEqual(
            _native.profile_slice_cell_count(55, 1_802_439),
            99_134_145,
        )
        with self.assertRaisesRegex(ValueError, "exceeds cell limit"):
            _native.profile_slice_cell_count(56, 1_802_439)
        self.assertEqual(_native.profile_slice_cell_count(10, 10, 100), 100)
        with self.assertRaisesRegex(ValueError, "overflow"):
            _native.profile_slice_cell_count(1 << 63, 2, (1 << 64) - 1)

    def test_model_length_boundaries(self):
        self.assertEqual(
            _native.profile_footprint([100_000]),
            {
                "profile_count": 1,
                "ssv_device_bytes": 2_900_000,
                "viterbi_device_bytes": 7_400_096,
                "viterbi_exact_rbv_upper_bytes": 2_900_000,
                "forward_device_bytes": 14_800_032,
                "bias_device_bytes": 272,
                "minimum_device_bytes": 25_100_400,
                "maximum_device_bytes": 28_000_400,
            },
        )
        for model_length in (0, 100_001):
            with self.assertRaisesRegex(ValueError, r"\[1, 100000\]"):
                _native.profile_footprint([model_length])

    def test_allocate_before_free_exact_boundary(self):
        fitting = _native.simulate_allocate_before_free([100], [150], 150)
        self.assertEqual(
            fitting,
            {
                "fits": True,
                "peak_additional_bytes": 150,
                "final_additional_bytes": 50,
                "final_free_bytes": 100,
                "growth_count": 1,
                "first_unfit_index": None,
                "capacities": (150,),
            },
        )
        failing = _native.simulate_allocate_before_free([100], [150], 149)
        self.assertFalse(failing["fits"])
        self.assertEqual(failing["peak_additional_bytes"], 150)
        self.assertEqual(failing["first_unfit_index"], 0)
        self.assertEqual(
            _native.simulate_allocate_before_free([], [], 7)["final_free_bytes"],
            7,
        )
        with self.assertRaisesRegex(ValueError, "overflow"):
            _native.simulate_allocate_before_free(
                [0, 0], [(1 << 64) - 1, 1], (1 << 64) - 1
            )

    def test_allocate_before_free_persists_growth_across_groups(self):
        first = _native.simulate_allocate_before_free(
            [64, 0, 128], [64, 256, 256], 2_048
        )
        self.assertEqual(first["capacities"], (64, 256, 256))
        self.assertEqual(first["peak_additional_bytes"], 512)
        self.assertEqual(first["final_free_bytes"], 1_664)

        second = _native.simulate_allocate_before_free(
            first["capacities"], [512, 128, 1_024], first["final_free_bytes"]
        )
        self.assertTrue(second["fits"])
        self.assertEqual(second["capacities"], (512, 256, 1_024))
        self.assertEqual(second["peak_additional_bytes"], 1_472)
        self.assertEqual(second["final_free_bytes"], 448)

        reused = _native.simulate_allocate_before_free(
            second["capacities"], [1, 2, 3], second["final_free_bytes"]
        )
        self.assertEqual(reused["capacities"], second["capacities"])
        self.assertEqual(reused["peak_additional_bytes"], 0)
        self.assertEqual(reused["growth_count"], 0)
        self.assertEqual(reused["final_free_bytes"], 448)

    def test_device_ordinal_validation_seam(self):
        self.assertIsNone(_native._validate_device_ordinal(3, 3))
        with self.assertRaisesRegex(RuntimeError, "different device"):
            _native._validate_device_ordinal(3, 4)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            _native._validate_device_ordinal(3, -1)

    def test_memory_info_rejects_no_visible_device(self):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        python_path = str(ROOT / "python")
        if environment.get("PYTHONPATH"):
            python_path += os.pathsep + environment["PYTHONPATH"]
        environment["PYTHONPATH"] = python_path
        command = (
            "from plan7_gpu import _native\n"
            "assert _native.profile_footprint([1])['minimum_device_bytes'] == 3981\n"
            "try:\n"
            "    _native.device_memory_info()\n"
            "except RuntimeError as error:\n"
            "    assert 'cudaGetDevice' in str(error), error\n"
            "else:\n"
            "    raise SystemExit('memory info accepted no visible device')\n"
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


@unittest.skipUnless(cuda_available(), "CUDA backend or device unavailable")
class CudaMemorySnapshotTests(unittest.TestCase):
    @staticmethod
    def optimized(path):
        with pyhmmer.plan7.HMMFile(path) as hmm_file:
            hmm = next(hmm_file)
        background = pyhmmer.plan7.Background(hmm.alphabet)
        return hmm.to_profile(background, L=100).to_optimized()

    @staticmethod
    def sequences(profile):
        return [
            pyhmmer.easel.TextSequence(name=b"a", sequence="G").digitize(
                profile.alphabet
            ),
            pyhmmer.easel.TextSequence(name=b"b", sequence="ACD").digitize(
                profile.alphabet
            ),
        ]

    def test_snapshot_tracks_component_high_water_and_close(self):
        small = self.optimized(HMM_SMALL)
        large = self.optimized(HMM_LARGE)
        batch = SequenceBatch(self.sequences(small))
        native = _sequence_native(batch)
        try:
            initial = batch.memory_snapshot
            capacities = initial["capacity_bytes"]
            self.assertLessEqual(initial["cuda_free_bytes"], initial["cuda_total_bytes"])
            self.assertEqual(initial["persistent_device_bytes"], sum(capacities.values()))
            self.assertEqual(len(capacities), 35)
            self.assertEqual(capacities["input_residues"], 4)
            self.assertEqual(capacities["input_offsets"], 24)
            self.assertEqual(capacities["input_null_scores"], 8)
            self.assertEqual(capacities["length_tjb"], 2)
            self.assertEqual(capacities["results"], 12)

            first_pack = _pack_profiles([small])
            native.filter_many_raw(*first_pack)
            first = batch.memory_snapshot
            self.assertEqual(
                first["capacity_bytes"]["compact_scores"], len(first_pack.scores)
            )
            self.assertEqual(first["capacity_bytes"]["profiles"], 32)

            larger_pack = _pack_profiles([small, large, small])
            native.filter_many_raw(*larger_pack)
            larger = batch.memory_snapshot
            self.assertEqual(
                larger["capacity_bytes"]["compact_scores"],
                len(larger_pack.scores),
            )
            self.assertEqual(larger["capacity_bytes"]["profiles"], 3 * 32)
            self.assertEqual(larger["capacity_bytes"]["length_tjb"], 3 * 2)
            self.assertEqual(larger["capacity_bytes"]["results"], 3 * 2 * 6)
            self.assertEqual(
                larger["persistent_device_bytes"],
                sum(larger["capacity_bytes"].values()),
            )

            native.filter_many_raw(*first_pack)
            reused = batch.memory_snapshot
            self.assertEqual(reused["capacity_bytes"], larger["capacity_bytes"])
        finally:
            batch.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            _ = batch.memory_snapshot

    def test_global_memory_info_matches_batch_device(self):
        profile = self.optimized(HMM_SMALL)
        with SequenceBatch(self.sequences(profile)) as batch:
            global_info = _native.device_memory_info()
            snapshot = batch.memory_snapshot
            self.assertEqual(
                global_info["device_ordinal"], snapshot["device_ordinal"]
            )
            self.assertLessEqual(
                global_info["cuda_free_bytes"], global_info["cuda_total_bytes"]
            )


if __name__ == "__main__":
    unittest.main()
