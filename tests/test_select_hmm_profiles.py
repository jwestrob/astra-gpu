import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_hmm_profiles import SelectionError, select_profiles


def hmm_record(
    name: str, accession: str | None = None, newline: bytes = b"\n"
) -> bytes:
    lines = [
        b"HMMER3/f [3.4 | Aug 2023]",
        f"NAME  {name}".encode("ascii"),
    ]
    if accession is not None:
        lines.append(f"ACC   {accession}".encode("ascii"))
    lines.extend(
        [
            b"LENG  1",
            b"ALPH  amino",
            b"HMM          A",
            b"             m->m",
            b"      1      0.0",
            b"//",
        ]
    )
    return newline.join(lines) + newline


def profile(
    source: str, ordinal: int, name: str, accession: str | None
) -> dict[str, object]:
    return {
        "length": 1,
        "name": name,
        "accession": accession,
        "source": source,
        "ordinal_in_file": ordinal,
    }


class SelectHmmProfilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_inventory(self, datasets: dict[str, object]) -> Path:
        path = self.root / "inventory.json"
        path.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
        return path

    def test_extracts_multiple_sources_in_numeric_quantile_order(self) -> None:
        first = hmm_record("short", "PF00001.1")
        second = hmm_record("long", "PF00002.1")
        third = hmm_record("solo", None, newline=b"\r\n")
        source_one = self.root / "one.hmm"
        source_two = self.root / "two.hmm"
        source_one.write_bytes(first + second)
        source_two.write_bytes(third)
        inventory = self.write_inventory(
            {
                "alpha": {
                    "quantile_profiles": {
                        "p10": profile(str(source_one), 1, "long", "PF00002.1"),
                        "p1": profile(str(source_one), 0, "short", "PF00001.1"),
                    }
                },
                "beta": {
                    "quantile_profiles": {
                        "p0": profile("two.hmm", 0, "solo", None),
                        # The same physical record may represent several quantiles.
                        "p100": profile(str(source_one), 0, "short", "PF00001.1"),
                    }
                },
            }
        )

        output = self.root / "selected" / "oracle.hmm"
        summary = select_profiles(inventory, output)

        self.assertEqual(summary.requested, 4)
        self.assertEqual(summary.written, 3)
        self.assertEqual(summary.sources, 2)
        self.assertEqual(output.read_bytes(), first + second + third)

    def test_dataset_and_quantile_filters(self) -> None:
        first = hmm_record("one", "A.1")
        second = hmm_record("two", "A.2")
        source = self.root / "input.hmm"
        source.write_bytes(first + second)
        inventory = self.write_inventory(
            {
                "keep": {
                    "quantile_profiles": {
                        "p0": profile(str(source), 0, "one", "A.1"),
                        "p100": profile(str(source), 1, "two", "A.2"),
                    }
                },
                "skip": {
                    "quantile_profiles": {
                        "p0": profile(str(source), 1, "two", "A.2")
                    }
                },
            }
        )
        output = self.root / "one.hmm"

        summary = select_profiles(
            inventory, output, datasets=["keep"], quantiles=["p100"]
        )

        self.assertEqual(summary.requested, 1)
        self.assertEqual(output.read_bytes(), second)

    def test_identity_mismatch_does_not_replace_existing_output(self) -> None:
        source = self.root / "input.hmm"
        source.write_bytes(hmm_record("actual", "ACC.1"))
        inventory = self.write_inventory(
            {
                "db": {
                    "quantile_profiles": {
                        "p50": profile(str(source), 0, "wrong", "ACC.1")
                    }
                }
            }
        )
        output = self.root / "output.hmm"
        output.write_bytes(b"keep me\n")

        with self.assertRaisesRegex(SelectionError, "identity mismatch"):
            select_profiles(inventory, output)

        self.assertEqual(output.read_bytes(), b"keep me\n")

    def test_rejects_unterminated_selected_record(self) -> None:
        source = self.root / "broken.hmm"
        source.write_bytes(hmm_record("complete") + hmm_record("broken")[:-3])
        inventory = self.write_inventory(
            {
                "db": {
                    "quantile_profiles": {
                        "p100": profile(str(source), 1, "broken", None)
                    }
                }
            }
        )

        with self.assertRaisesRegex(SelectionError, "unterminated"):
            select_profiles(inventory, self.root / "output.hmm")

    def test_rejects_conflicting_duplicate_identity(self) -> None:
        source = self.root / "input.hmm"
        source.write_bytes(hmm_record("actual"))
        inventory = self.write_inventory(
            {
                "a": {
                    "quantile_profiles": {
                        "p0": profile(str(source), 0, "actual", None)
                    }
                },
                "b": {
                    "quantile_profiles": {
                        "p0": profile(str(source), 0, "different", None)
                    }
                },
            }
        )

        with self.assertRaisesRegex(SelectionError, "conflicting identities"):
            select_profiles(inventory, self.root / "output.hmm")


if __name__ == "__main__":
    unittest.main()
