import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import audit_h200_pfam_first1000 as audit


class H200First1000HarnessTests(unittest.TestCase):
    def test_identifier_bytes_accepts_both_pyhmmer_representations(self):
        self.assertEqual(audit.identifier_bytes("PF00001.1"), b"PF00001.1")
        self.assertEqual(audit.identifier_bytes(b"PF00001.1"), b"PF00001.1")
        self.assertIsNone(audit.identifier_bytes(None))
        with self.assertRaisesRegex(TypeError, "identifier type"):
            audit.identifier_bytes(1)

    def test_pci_parser_normalizes_cuda_and_nvidia_smi_widths(self):
        self.assertEqual(
            audit.pci_tuple("0000:5B:00.0"),
            audit.pci_tuple("00000000:5b:00.0"),
        )

    def test_nvidia_smi_identity_is_joined_by_uuid(self):
        cuda = {
            "uuid": "GPU-00112233-4455-6677-8899-aabbccddeeff",
            "name": "NVIDIA H200",
            "compute_capability": [9, 0],
            "pci_bus_address": "0000:1B:00.0",
        }
        output = (
            "0, NVIDIA H100, GPU-11111111-1111-1111-1111-111111111111, "
            "9.0, 81559, 570.00, 00000000:0A:00.0\n"
            "3, NVIDIA H200, GPU-00112233-4455-6677-8899-aabbccddeeff, "
            "9.0, 143771, 570.00, 00000000:1B:00.0"
        )
        with mock.patch.object(audit, "command_output", return_value=output):
            record = audit.visible_gpu_record(cuda)
        self.assertEqual(record["physical_index"], "3")
        self.assertEqual(record["driver_package_version"], "570.00")

    def test_nvidia_smi_cross_check_rejects_product_disagreement(self):
        cuda = {
            "uuid": "GPU-00112233-4455-6677-8899-aabbccddeeff",
            "name": "NVIDIA H200",
            "compute_capability": [9, 0],
            "pci_bus_address": "0000:1B:00.0",
        }
        output = (
            "3, NVIDIA H100, GPU-00112233-4455-6677-8899-aabbccddeeff, "
            "9.0, 143771, 570.00, 00000000:1B:00.0"
        )
        with (
            mock.patch.object(audit, "command_output", return_value=output),
            self.assertRaisesRegex(RuntimeError, "product names differ"),
        ):
            audit.visible_gpu_record(cuda)

    def test_fasta_boundary_is_exactly_one_thousand(self):
        with tempfile.TemporaryDirectory() as temporary:
            valid = Path(temporary) / "valid.faa"
            invalid = Path(temporary) / "invalid.faa"
            valid.write_text("".join(f">target-{index}\nA\n" for index in range(1000)))
            invalid.write_text("".join(f">target-{index}\nA\n" for index in range(999)))
            targets = audit.load_exact_first1000(valid)
            self.assertEqual(len(targets), 1000)
            self.assertEqual(targets.total_length(), 1000)
            with self.assertRaisesRegex(RuntimeError, "exactly 1000"):
                audit.load_exact_first1000(invalid)

    def test_nonvacuous_gpu_work_requires_every_native_boundary(self):
        workspace = {field: 1 for field in audit._POSITIVE_WORKSPACE_FIELDS}
        candidates = {
            "model_rows": 2,
            "total_retained_candidates": 3,
            "nonempty_model_rows": 2,
            "sealed_postfilter": True,
            "out_of_range_model_rows": 0,
            "continuation_routes": {
                "row_count": 3,
                "cpu_required_count": 1,
                "no_region_count": 1,
                "simple_count": 1,
            },
        }
        self.assertEqual(
            audit.nonvacuous_gpu_work_failures(workspace, candidates), []
        )
        for field in audit._POSITIVE_WORKSPACE_FIELDS:
            with self.subTest(field=field):
                bad_workspace = {**workspace, field: 0}
                self.assertIn(
                    f"workspace.{field} is not positive",
                    audit.nonvacuous_gpu_work_failures(
                        bad_workspace, candidates
                    ),
                )
        for field in (
            "model_rows",
            "total_retained_candidates",
            "nonempty_model_rows",
        ):
            with self.subTest(field=field):
                bad_candidates = {**candidates, field: 0}
                self.assertIn(
                    f"candidates.{field} is not positive",
                    audit.nonvacuous_gpu_work_failures(
                        workspace, bad_candidates
                    ),
                )
        self.assertIn(
            "candidates.sealed_postfilter is not true",
            audit.nonvacuous_gpu_work_failures(
                workspace, {**candidates, "sealed_postfilter": False}
            ),
        )
        self.assertIn(
            "candidate row counts fall outside [0, 1000]",
            audit.nonvacuous_gpu_work_failures(
                workspace, {**candidates, "out_of_range_model_rows": 1}
            ),
        )

    def test_nonvacuous_gpu_work_reconciles_optional_routes(self):
        workspace = {field: 1 for field in audit._POSITIVE_WORKSPACE_FIELDS}
        candidates = {
            "model_rows": 2,
            "total_retained_candidates": 3,
            "nonempty_model_rows": 2,
            "sealed_postfilter": True,
            "out_of_range_model_rows": 0,
            "continuation_routes": {
                "row_count": 4,
                "cpu_required_count": 1,
                "no_region_count": 1,
                "simple_count": 1,
            },
        }
        failures = audit.nonvacuous_gpu_work_failures(workspace, candidates)
        self.assertIn(
            "continuation route counts do not sum to row_count", failures
        )
        self.assertIn(
            "continuation rows differ from retained candidates", failures
        )

    def test_gpu_capability_record_requires_all_same_dso_boundaries(self):
        cache = {
            name: {
                "resolved": True,
                "available": True,
                "same_dso": True,
                "resolutions": 1,
                "dlopen_calls": 1,
                "dlclose_calls": 1,
            }
            for name in audit._SEAM_PROBES
        }
        with ExitStack() as stack:
            for attribute in audit._SEAM_PROBES.values():
                stack.enter_context(
                    mock.patch.object(
                        audit._pipeline, attribute, return_value=True
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    audit._pipeline,
                    "_continuation_seam_cache_info",
                    return_value=cache,
                )
            )
            record = audit.gpu_capability_record()
        self.assertTrue(record["attested"])
        self.assertEqual(record["failures"], [])

        broken = {
            **cache,
            "forward": {**cache["forward"], "same_dso": False},
        }
        with ExitStack() as stack:
            for attribute in audit._SEAM_PROBES.values():
                stack.enter_context(
                    mock.patch.object(
                        audit._pipeline, attribute, return_value=True
                    )
                )
            stack.enter_context(
                mock.patch.object(
                    audit._pipeline,
                    "_continuation_seam_cache_info",
                    return_value=broken,
                )
            )
            record = audit.gpu_capability_record()
        self.assertFalse(record["attested"])
        self.assertIn("forward.same_dso", record["failures"])


if __name__ == "__main__":
    unittest.main()
