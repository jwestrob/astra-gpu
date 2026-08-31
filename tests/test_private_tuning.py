import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "python" / "plan7_gpu" / "_private_tuning.py"
SPEC = importlib.util.spec_from_file_location("_plan7_private_tuning", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load private tuning module")
TUNING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TUNING)


def load_astra_search_for_host_test():
    package = ModuleType("plan7_gpu")
    package.__path__ = [str(ROOT / "python" / "plan7_gpu")]
    adapter = ModuleType("plan7_gpu.adapter")
    adapter.CandidateBatch = type("CandidateBatch", (), {})
    adapter.PressedProfilePair = type("PressedProfilePair", (), {})
    adapter.SequenceBatch = type("SequenceBatch", (), {})
    adapter._candidate_state = lambda candidates: candidates
    module_name = "plan7_gpu.astra_search_host_test"
    source = ROOT / "python" / "plan7_gpu" / "astra_search.py"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load Astra search module")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "plan7_gpu": package,
            "plan7_gpu.adapter": adapter,
            "plan7_gpu._private_tuning": TUNING,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class PrivateTuningHostTests(unittest.TestCase):
    def test_unset_environment_preserves_retained_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(TUNING.continuation_shard_trigger(), (2, 1))
            self.assertEqual(TUNING.auto_forward_cpu_max_cells(), 200_000)

    def test_shard_trigger_values_use_exact_strict_boundaries(self):
        cases = (
            ("2.0", (2, 1), 200),
            ("1.5", (3, 2), 150),
            ("1.0", (1, 1), 100),
        )
        for raw, expected_ratio, boundary in cases:
            with self.subTest(raw=raw), mock.patch.dict(
                os.environ,
                {TUNING.CONTINUATION_SHARD_TRIGGER_ENV: raw},
                clear=True,
            ):
                ratio = TUNING.continuation_shard_trigger()
                self.assertEqual(ratio, expected_ratio)
                self.assertFalse(TUNING.exceeds_ratio(boundary, 100, ratio))
                self.assertTrue(TUNING.exceeds_ratio(boundary + 1, 100, ratio))

    def test_shard_planner_changes_only_at_selected_trigger(self):
        astra_search = load_astra_search_for_host_test()
        profile_hints = (180, 20)
        exception_hints = ((60, 60, 60), (20,))
        retained = astra_search._sharded_task_bounds(
            profile_hints, exception_hints, 1
        )
        lower = astra_search._sharded_task_bounds(
            profile_hints, exception_hints, 1, (3, 2)
        )
        self.assertEqual(len(retained), 2)
        self.assertEqual(len(lower), 4)
        self.assertEqual(
            [(task[3], task[4], task[5]) for task in lower[:3]],
            [(0, 1, 0), (1, 2, 0), (2, 3, 1)],
        )

    def test_forward_auto_ownership_accepts_sweep_cutoffs(self):
        for cells in (150_000, 100_000):
            with self.subTest(cells=cells), mock.patch.dict(
                os.environ,
                {TUNING.FORWARD_CPU_MAX_CELLS_ENV: str(cells)},
                clear=True,
            ):
                self.assertEqual(TUNING.auto_forward_cpu_max_cells(), cells)

    def test_invalid_private_values_fail_closed(self):
        for raw in ("", "nan", "0.5", "2.5", "1/0"):
            with self.subTest(shard_trigger=raw), mock.patch.dict(
                os.environ,
                {TUNING.CONTINUATION_SHARD_TRIGGER_ENV: raw},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "from 1.0 through 2.0"):
                    TUNING.continuation_shard_trigger()
        for raw in ("", "0", "-1", str(1 << 64)):
            with self.subTest(forward_cells=raw), mock.patch.dict(
                os.environ,
                {TUNING.FORWARD_CPU_MAX_CELLS_ENV: raw},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "positive"):
                    TUNING.auto_forward_cpu_max_cells()


if __name__ == "__main__":
    unittest.main()
