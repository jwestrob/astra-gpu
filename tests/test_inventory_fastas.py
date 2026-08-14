import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.inventory_fastas import FastaInventoryError, build_inventory


class InventoryFastasTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest_item(self, dataset_id: str, path: Path) -> dict[str, object]:
        data = path.read_bytes()
        lengths = []
        current = None
        for line in data.splitlines():
            if line.startswith(b">"):
                if current is not None:
                    lengths.append(current)
                current = 0
            elif current is not None:
                current += len(line.translate(None, b" \t\r\n\v\f"))
        if current is not None:
            lengths.append(current)
        return {
            "id": dataset_id,
            "path": str(path),
            "size_bytes": len(data),
            "sequences": len(lengths),
            "residues": sum(lengths),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def test_inventory_has_file_and_aggregate_exact_quantile_records(self) -> None:
        first = self.root / "first.faa"
        second = self.root / "second.faa"
        first.write_bytes(
            b">dup first\r\nAA\r\n>mid\nA A A\n>long\nAAAAA\n"
        )
        second.write_bytes(b">dup second\nAAAA\n>tiny\nA\n")
        manifest = self.root / "datasets.json"
        manifest.write_text(
            json.dumps(
                {
                    "protein_fastas": [
                        self.manifest_item("first", first),
                        self.manifest_item("second", second),
                    ]
                }
            ),
            encoding="utf-8",
        )
        output = self.root / "inventory.json"

        result = build_inventory(manifest, output)

        self.assertEqual(result["aggregate"]["sequences"], 5)
        self.assertEqual(result["aggregate"]["residues"], 15)
        self.assertEqual(result["aggregate"]["length"]["min"], 1)
        self.assertEqual(result["aggregate"]["length"]["max"], 5)
        picks = result["aggregate"]["quantile_records"]
        self.assertEqual((picks["p0"]["identifier"], picks["p0"]["length"]), ("tiny", 1))
        self.assertEqual((picks["p50"]["identifier"], picks["p50"]["length"]), ("mid", 3))
        self.assertEqual((picks["p100"]["identifier"], picks["p100"]["length"]), ("long", 5))
        self.assertEqual(picks["p0"]["ordinal_in_file"], 1)
        self.assertEqual(len(picks["p0"]["header_sha256"]), 64)
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["schema_version"], 1)

    def test_frozen_mismatch_does_not_replace_output(self) -> None:
        fasta = self.root / "input.faa"
        fasta.write_bytes(b">one\nAA\n")
        item = self.manifest_item("input", fasta)
        item["residues"] = 999
        manifest = self.root / "datasets.json"
        manifest.write_text(
            json.dumps({"protein_fastas": [item]}), encoding="utf-8"
        )
        output = self.root / "inventory.json"
        output.write_text("keep\n", encoding="utf-8")

        with self.assertRaisesRegex(FastaInventoryError, "frozen residues mismatch"):
            build_inventory(manifest, output)

        self.assertEqual(output.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
