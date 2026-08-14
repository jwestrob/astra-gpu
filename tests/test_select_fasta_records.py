import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.inventory_fastas import build_inventory
from scripts.select_fasta_records import FastaSelectionError, select_records


class SelectFastaRecordsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def manifest_item(dataset_id: str, path: Path, sequences: int, residues: int) -> dict[str, object]:
        data = path.read_bytes()
        return {
            "id": dataset_id,
            "path": str(path),
            "size_bytes": len(data),
            "sequences": sequences,
            "residues": residues,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def make_inventory(self) -> tuple[Path, dict[str, bytes]]:
        records = {
            "dup1": b">dup first\r\nAA\r\n",
            "mid": b">mid\nA A A\n",
            "long": b">long\nAAAAA\n",
            "dup2": b">dup second\nAAAA\n",
            "tiny": b">tiny\nA\n",
        }
        first = self.root / "first.faa"
        second = self.root / "second.faa"
        first.write_bytes(records["dup1"] + records["mid"] + records["long"])
        second.write_bytes(records["dup2"] + records["tiny"])
        manifest = self.root / "datasets.json"
        manifest.write_text(
            json.dumps(
                {
                    "protein_fastas": [
                        self.manifest_item("first", first, 3, 10),
                        self.manifest_item("second", second, 2, 5),
                    ]
                }
            ),
            encoding="utf-8",
        )
        inventory = self.root / "inventory.json"
        build_inventory(manifest, inventory)
        return inventory, records

    def test_preserves_bytes_and_does_not_collapse_duplicate_ids(self) -> None:
        inventory, records = self.make_inventory()
        output = self.root / "selected" / "quantiles.faa"

        summary = select_records(inventory, output)

        self.assertEqual(summary.requested, 27)
        self.assertEqual(summary.written, 5)
        self.assertEqual(summary.duplicate_identifier_records, 1)
        self.assertEqual(summary.residues, 15)
        expected = (
            records["dup1"]
            + records["mid"]
            + records["long"]
            + records["tiny"]
            + records["dup2"]
        )
        self.assertEqual(output.read_bytes(), expected)
        self.assertEqual(summary.sha256, hashlib.sha256(expected).hexdigest())

    def test_length_mismatch_does_not_replace_existing_output(self) -> None:
        inventory, _ = self.make_inventory()
        data = json.loads(inventory.read_text(encoding="utf-8"))
        data["files"][0]["quantile_records"]["p0"]["length"] = 999
        inventory.write_text(json.dumps(data), encoding="utf-8")
        output = self.root / "selected.faa"
        output.write_bytes(b"keep me\n")

        with self.assertRaisesRegex(FastaSelectionError, "identity mismatch"):
            select_records(
                inventory, output, quantiles=["p0"], include_aggregate=False
            )

        self.assertEqual(output.read_bytes(), b"keep me\n")


if __name__ == "__main__":
    unittest.main()
