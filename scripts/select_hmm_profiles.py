#!/usr/bin/env python3
"""Extract inventory-selected HMMER records into a small text database.

The inventory produced by :mod:`scripts.inventory_hmms` identifies each
quantile profile by source file, zero-based ordinal, NAME, and accession.  This
tool uses all four fields: source and ordinal locate a record, while NAME and
accession protect against silently extracting the wrong record from a changed
database.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence


class SelectionError(ValueError):
    """Raised when an inventory or HMM source cannot be selected safely."""


@dataclass(frozen=True)
class Selection:
    dataset: str
    quantile: str
    source: Path
    ordinal: int
    name: str
    accession: str | None

    @property
    def record_key(self) -> tuple[Path, int]:
        return self.source, self.ordinal


@dataclass(frozen=True)
class SelectionSummary:
    requested: int
    written: int
    sources: int
    output: Path


@dataclass(frozen=True)
class _Record:
    data: bytes
    name: str | None
    accession: str | None


_QUANTILE_RE = re.compile(r"^p([0-9]+(?:\.[0-9]+)?)$")


def _quantile_sort_key(value: str) -> tuple[int, float | str, str]:
    match = _QUANTILE_RE.fullmatch(value)
    if match is None:
        return (1, value, value)
    return (0, float(match.group(1)), value)


def _unique(values: Sequence[str], description: str) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise SelectionError(f"duplicate {description}: {value}")
        seen.add(value)
        result.append(value)
    return result


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SelectionError(f"{context} must be a JSON object")
    return value


def _optional_text(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SelectionError(f"{context} must be a nonempty string or null")
    return value


def load_selections(
    inventory_path: Path,
    datasets: Sequence[str] | None = None,
    quantiles: Sequence[str] | None = None,
) -> list[Selection]:
    """Load and validate quantile selections from an HMM inventory."""

    inventory_path = inventory_path.expanduser().resolve()
    try:
        with inventory_path.open("rt", encoding="utf-8") as handle:
            root = _mapping(json.load(handle), "inventory")
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionError(f"cannot read inventory {inventory_path}: {error}") from error

    inventory_datasets = _mapping(root.get("datasets"), "inventory.datasets")
    if datasets is None:
        selected_datasets = sorted(inventory_datasets)
    else:
        selected_datasets = _unique(list(datasets), "dataset")
    if not selected_datasets:
        raise SelectionError("no datasets were selected")

    selected_quantiles = None
    if quantiles is not None:
        selected_quantiles = _unique(list(quantiles), "quantile")
        if not selected_quantiles:
            raise SelectionError("no quantiles were selected")

    selections: list[Selection] = []
    for dataset in selected_datasets:
        if dataset not in inventory_datasets:
            raise SelectionError(f"dataset not found in inventory: {dataset}")
        dataset_value = _mapping(
            inventory_datasets[dataset], f"inventory.datasets.{dataset}"
        )
        profile_map = _mapping(
            dataset_value.get("quantile_profiles"),
            f"inventory.datasets.{dataset}.quantile_profiles",
        )
        profile_quantiles = (
            sorted(profile_map, key=_quantile_sort_key)
            if selected_quantiles is None
            else selected_quantiles
        )
        if not profile_quantiles:
            raise SelectionError(f"dataset {dataset} has no quantile profiles")

        for quantile in profile_quantiles:
            if quantile not in profile_map:
                raise SelectionError(
                    f"quantile {quantile} not found for dataset {dataset}"
                )
            context = f"inventory.datasets.{dataset}.quantile_profiles.{quantile}"
            item = _mapping(profile_map[quantile], context)

            source_value = item.get("source")
            if not isinstance(source_value, str) or not source_value:
                raise SelectionError(f"{context}.source must be a nonempty string")
            source = Path(source_value).expanduser()
            if not source.is_absolute():
                source = inventory_path.parent / source
            source = source.resolve()

            ordinal = item.get("ordinal_in_file")
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                raise SelectionError(
                    f"{context}.ordinal_in_file must be a nonnegative integer"
                )
            name = _optional_text(item.get("name"), f"{context}.name")
            if name is None:
                raise SelectionError(f"{context}.name must not be null")
            if "accession" not in item:
                raise SelectionError(
                    f"{context}.accession must be present (a string or null)"
                )
            accession = _optional_text(item.get("accession"), f"{context}.accession")
            selections.append(
                Selection(dataset, quantile, source, ordinal, name, accession)
            )

    return selections


def _deduplicate(selections: Sequence[Selection]) -> list[Selection]:
    unique: list[Selection] = []
    seen: dict[tuple[Path, int], Selection] = {}
    for selection in selections:
        previous = seen.get(selection.record_key)
        if previous is None:
            seen[selection.record_key] = selection
            unique.append(selection)
            continue
        if (previous.name, previous.accession) != (
            selection.name,
            selection.accession,
        ):
            raise SelectionError(
                "conflicting identities for "
                f"{selection.source} record {selection.ordinal}: "
                f"{previous.name!r}/{previous.accession!r} versus "
                f"{selection.name!r}/{selection.accession!r}"
            )
    return unique


def _header_field(line: bytes) -> tuple[bytes, str] | None:
    parts = line.rstrip(b"\r\n").split(None, 1)
    if not parts or parts[0] not in (b"NAME", b"ACC"):
        return None
    raw_value = parts[1].strip() if len(parts) == 2 else b""
    if not raw_value:
        raise SelectionError(f"malformed {parts[0].decode('ascii')} header")
    try:
        value = raw_value.decode("ascii")
    except UnicodeDecodeError as error:
        raise SelectionError("non-ASCII NAME or ACC header") from error
    return parts[0], value


def _finish_record(data: bytearray, headers: dict[bytes, str]) -> _Record:
    return _Record(bytes(data), headers.get(b"NAME"), headers.get(b"ACC"))


def _read_ordinals(source: Path, ordinals: set[int]) -> dict[int, _Record]:
    """Read requested zero-based records from one HMMER text source."""

    if not ordinals:
        return {}
    found: dict[int, _Record] = {}
    current_ordinal = -1
    selected_ordinal: int | None = None
    selected_data: bytearray | None = None
    selected_headers: dict[bytes, str] = {}
    in_header = False

    try:
        handle: BinaryIO
        with source.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                stripped = line.rstrip(b"\r\n")
                if selected_ordinal is None:
                    if line.startswith(b"HMMER3/"):
                        current_ordinal += 1
                        if current_ordinal in ordinals:
                            selected_ordinal = current_ordinal
                            selected_data = bytearray(line)
                            selected_headers = {}
                            in_header = True
                    continue

                if line.startswith(b"HMMER3/"):
                    raise SelectionError(
                        f"{source}:{line_number}: record {selected_ordinal} starts "
                        "another record before //"
                    )
                assert selected_data is not None
                selected_data.extend(line)

                if in_header:
                    token = stripped.split(None, 1)[0] if stripped else b""
                    if token == b"HMM":
                        in_header = False
                    else:
                        field = _header_field(line)
                        if field is not None:
                            key, value = field
                            if key in selected_headers:
                                raise SelectionError(
                                    f"{source}:{line_number}: duplicate "
                                    f"{key.decode('ascii')} header in record "
                                    f"{selected_ordinal}"
                                )
                            selected_headers[key] = value

                if stripped == b"//":
                    found[selected_ordinal] = _finish_record(
                        selected_data, selected_headers
                    )
                    selected_ordinal = None
                    selected_data = None
                    selected_headers = {}
                    in_header = False
                    if len(found) == len(ordinals):
                        break
    except OSError as error:
        raise SelectionError(f"cannot read HMM source {source}: {error}") from error

    if selected_ordinal is not None:
        raise SelectionError(
            f"{source}: selected record {selected_ordinal} is unterminated (missing //)"
        )
    missing = sorted(ordinals.difference(found))
    if missing:
        missing_text = ", ".join(str(value) for value in missing)
        raise SelectionError(f"{source}: record ordinal(s) not found: {missing_text}")
    return found


def _validate_identity(selection: Selection, record: _Record) -> None:
    if record.name != selection.name:
        raise SelectionError(
            f"identity mismatch for {selection.source} record {selection.ordinal}: "
            f"inventory NAME {selection.name!r}, source NAME {record.name!r}"
        )
    if record.accession != selection.accession:
        raise SelectionError(
            f"identity mismatch for {selection.source} record {selection.ordinal}: "
            f"inventory accession {selection.accession!r}, "
            f"source accession {record.accession!r}"
        )


def _atomic_write(output: Path, records: Sequence[_Record]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for record in records:
                handle.write(record.data)
                if not record.data.endswith(b"\n"):
                    handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def select_profiles(
    inventory_path: Path,
    output: Path,
    datasets: Sequence[str] | None = None,
    quantiles: Sequence[str] | None = None,
) -> SelectionSummary:
    """Extract selected profiles and atomically replace ``output``."""

    selections = load_selections(inventory_path, datasets, quantiles)
    unique = _deduplicate(selections)
    output = output.expanduser().resolve()
    source_paths = {selection.source for selection in unique}
    if output in source_paths:
        raise SelectionError(f"output would overwrite an HMM source: {output}")

    by_source: dict[Path, set[int]] = defaultdict(set)
    for selection in unique:
        by_source[selection.source].add(selection.ordinal)

    extracted: dict[tuple[Path, int], _Record] = {}
    for source in sorted(by_source, key=str):
        records = _read_ordinals(source, by_source[source])
        for ordinal, record in records.items():
            extracted[(source, ordinal)] = record

    for selection in unique:
        _validate_identity(selection, extracted[selection.record_key])

    _atomic_write(output, [extracted[item.record_key] for item in unique])
    return SelectionSummary(len(selections), len(unique), len(by_source), output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract quantile-selected HMM records from an inventory into one "
            "combined HMMER text database."
        )
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        action="append",
        help="inventory dataset label (repeatable; default: all datasets)",
    )
    parser.add_argument(
        "--quantile",
        action="append",
        help="quantile key such as p0 or p99 (repeatable; default: all)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = select_profiles(
            args.inventory, args.output, args.dataset, args.quantile
        )
    except SelectionError as error:
        parser.error(str(error))
    print(
        f"wrote {summary.written} profile(s) from {summary.sources} source "
        f"file(s) to {summary.output} "
        f"({summary.requested} inventory selection(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
