import tempfile
import unittest
from pathlib import Path
from unittest import mock

import audit_h200_pfam_first1000 as audit


class H200First1000HarnessTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
