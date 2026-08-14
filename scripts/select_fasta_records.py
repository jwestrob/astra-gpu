#!/usr/bin/env python3
"""Extract FASTA quantile records by source and ordinal with identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


_FASTA_WHITESPACE = b" \t\r\n\v\f"
_QUANTILE_RE = re.compile(r"^p([0-9]+(?:\.[0-9]+)?)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FastaSelectionError(ValueError):
    """Raised when inventory-selected FASTA records cannot be verified."""


@dataclass(frozen=True)
class Selection:
    origin: str
    quantile: str
    source: Path
    ordinal: int
    identifier: str
    length: int
    header_sha256: str

    @property
    def record_key(self) -> tuple[Path, int]:
        return self.source, self.ordinal


@dataclass(frozen=True)
class _SourceIdentity:
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class _Record:
    data: bytes
    identifier: str
    length: int
    header_sha256: str


@dataclass(frozen=True)
class SelectionSummary:
    requested: int
    written: int
    sources: int
    residues: int
    size_bytes: int
    sha256: str
    duplicate_identifier_records: int
    output: Path


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FastaSelectionError(f"{context} must be a JSON object")
    return value


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
            raise FastaSelectionError(f"duplicate {description}: {value}")
        seen.add(value)
        result.append(value)
    return result


def _nonnegative_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FastaSelectionError(f"{context} must be a nonnegative integer")
    return value


def _source_path(raw: object, inventory_path: Path, context: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise FastaSelectionError(f"{context} must be a nonempty string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = inventory_path.parent / path
    return path.resolve()


def _selection(
    raw: object, origin: str, quantile: str, inventory_path: Path
) -> Selection:
    context = f"{origin}.quantile_records.{quantile}"
    item = _mapping(raw, context)
    identifier = item.get("identifier")
    if not isinstance(identifier, str) or not identifier:
        raise FastaSelectionError(f"{context}.identifier must be nonempty")
    header_sha256 = item.get("header_sha256")
    if not isinstance(header_sha256, str) or not _SHA256_RE.fullmatch(header_sha256):
        raise FastaSelectionError(f"{context}.header_sha256 is not a SHA-256 digest")
    return Selection(
        origin,
        quantile,
        _source_path(item.get("source"), inventory_path, f"{context}.source"),
        _nonnegative_int(item.get("ordinal_in_file"), f"{context}.ordinal_in_file"),
        identifier,
        _nonnegative_int(item.get("length"), f"{context}.length"),
        header_sha256,
    )


def _selected_quantiles(
    record_map: Mapping[str, object], quantiles: Sequence[str] | None, origin: str
) -> list[str]:
    selected = (
        sorted(record_map, key=_quantile_sort_key)
        if quantiles is None
        else list(quantiles)
    )
    if not selected:
        raise FastaSelectionError(f"{origin} has no selected quantiles")
    for quantile in selected:
        if quantile not in record_map:
            raise FastaSelectionError(f"quantile {quantile} not found for {origin}")
    return selected


def load_selections(
    inventory_path: Path,
    file_ids: Sequence[str] | None = None,
    quantiles: Sequence[str] | None = None,
    include_aggregate: bool = True,
) -> tuple[list[Selection], dict[Path, _SourceIdentity]]:
    """Load quantile selections and source stat identities from an inventory."""

    inventory_path = inventory_path.expanduser().resolve()
    try:
        with inventory_path.open("rt", encoding="utf-8") as handle:
            root = _mapping(json.load(handle), "inventory")
    except (OSError, json.JSONDecodeError) as error:
        raise FastaSelectionError(f"cannot read inventory {inventory_path}: {error}") from error

    raw_files = root.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise FastaSelectionError("inventory.files must be a nonempty array")
    requested_ids = None if file_ids is None else _unique(list(file_ids), "file id")
    requested_quantiles = (
        None if quantiles is None else _unique(list(quantiles), "quantile")
    )

    file_map: dict[str, Mapping[str, object]] = {}
    file_order: list[str] = []
    source_identities: dict[Path, _SourceIdentity] = {}
    for index, raw_file in enumerate(raw_files):
        context = f"inventory.files[{index}]"
        item = _mapping(raw_file, context)
        dataset_id = item.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise FastaSelectionError(f"{context}.dataset_id must be nonempty")
        if dataset_id in file_map:
            raise FastaSelectionError(f"duplicate inventory file id: {dataset_id}")
        file_map[dataset_id] = item
        file_order.append(dataset_id)

        source = _source_path(item.get("source"), inventory_path, f"{context}.source")
        identity = _SourceIdentity(
            _nonnegative_int(item.get("source_size_bytes"), f"{context}.source_size_bytes"),
            _nonnegative_int(item.get("source_mtime_ns"), f"{context}.source_mtime_ns"),
        )
        previous = source_identities.get(source)
        if previous is not None and previous != identity:
            raise FastaSelectionError(f"conflicting source identity for {source}")
        source_identities[source] = identity

    selected_ids = file_order if requested_ids is None else requested_ids
    selections: list[Selection] = []
    for dataset_id in selected_ids:
        if dataset_id not in file_map:
            raise FastaSelectionError(f"file id not found in inventory: {dataset_id}")
        origin = f"file[{dataset_id}]"
        record_map = _mapping(file_map[dataset_id].get("quantile_records"), origin)
        for quantile in _selected_quantiles(
            record_map, requested_quantiles, origin
        ):
            selections.append(
                _selection(record_map[quantile], origin, quantile, inventory_path)
            )

    if include_aggregate:
        aggregate = _mapping(root.get("aggregate"), "aggregate")
        record_map = _mapping(
            aggregate.get("quantile_records"), "aggregate.quantile_records"
        )
        for quantile in _selected_quantiles(
            record_map, requested_quantiles, "aggregate"
        ):
            selections.append(
                _selection(record_map[quantile], "aggregate", quantile, inventory_path)
            )
    if not selections:
        raise FastaSelectionError("no FASTA records were selected")
    return selections, source_identities


def _deduplicate(selections: Sequence[Selection]) -> list[Selection]:
    result: list[Selection] = []
    seen: dict[tuple[Path, int], Selection] = {}
    for selection in selections:
        previous = seen.get(selection.record_key)
        if previous is None:
            seen[selection.record_key] = selection
            result.append(selection)
            continue
        if (
            previous.identifier,
            previous.length,
            previous.header_sha256,
        ) != (
            selection.identifier,
            selection.length,
            selection.header_sha256,
        ):
            raise FastaSelectionError(
                f"conflicting identities for {selection.source} record "
                f"{selection.ordinal}"
            )
    return result


def _signature(stat: os.stat_result) -> tuple[int, int]:
    return stat.st_size, stat.st_mtime_ns


def _residue_count(line: bytes) -> int:
    return len(line.translate(None, _FASTA_WHITESPACE))


def _identifier(header: bytes, source: Path, line_number: int) -> str:
    tokens = header[1:].strip().split(None, 1)
    if not tokens:
        raise FastaSelectionError(f"{source}:{line_number}: empty FASTA identifier")
    try:
        return tokens[0].decode("ascii")
    except UnicodeDecodeError as error:
        raise FastaSelectionError(
            f"{source}:{line_number}: non-ASCII FASTA identifier"
        ) from error


def _finish_record(
    data: bytearray,
    identifier: str,
    length: int,
    header_sha256: str,
) -> _Record:
    return _Record(bytes(data), identifier, length, header_sha256)


def _read_ordinals(
    source: Path, ordinals: set[int], identity: _SourceIdentity
) -> dict[int, _Record]:
    try:
        before = source.stat()
    except OSError as error:
        raise FastaSelectionError(f"cannot stat FASTA {source}: {error}") from error
    if _signature(before) != (identity.size_bytes, identity.mtime_ns):
        raise FastaSelectionError(f"FASTA stat identity differs from inventory: {source}")

    found: dict[int, _Record] = {}
    current_ordinal = -1
    selected_ordinal: int | None = None
    selected_data: bytearray | None = None
    selected_identifier = ""
    selected_length = 0
    selected_header_sha256 = ""
    saw_header = False
    try:
        with source.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.startswith(b">"):
                    if selected_ordinal is not None:
                        assert selected_data is not None
                        found[selected_ordinal] = _finish_record(
                            selected_data,
                            selected_identifier,
                            selected_length,
                            selected_header_sha256,
                        )
                        selected_ordinal = None
                        selected_data = None
                        if len(found) == len(ordinals):
                            break

                    saw_header = True
                    current_ordinal += 1
                    identifier = _identifier(line, source, line_number)
                    if current_ordinal in ordinals:
                        selected_ordinal = current_ordinal
                        selected_data = bytearray(line)
                        selected_identifier = identifier
                        selected_length = 0
                        selected_header_sha256 = hashlib.sha256(line).hexdigest()
                    continue

                residues = _residue_count(line)
                if not saw_header and residues:
                    raise FastaSelectionError(
                        f"{source}:{line_number}: sequence data before first header"
                    )
                if selected_ordinal is not None:
                    assert selected_data is not None
                    selected_data.extend(line)
                    selected_length += residues
    except OSError as error:
        raise FastaSelectionError(f"cannot read FASTA {source}: {error}") from error

    if selected_ordinal is not None:
        assert selected_data is not None
        found[selected_ordinal] = _finish_record(
            selected_data,
            selected_identifier,
            selected_length,
            selected_header_sha256,
        )
    missing = sorted(ordinals.difference(found))
    if missing:
        raise FastaSelectionError(
            f"{source}: record ordinal(s) not found: "
            + ", ".join(str(value) for value in missing)
        )

    try:
        after = source.stat()
    except OSError as error:
        raise FastaSelectionError(f"cannot restat FASTA {source}: {error}") from error
    if _signature(after) != (identity.size_bytes, identity.mtime_ns):
        raise FastaSelectionError(f"FASTA changed while being read: {source}")
    return found


def _validate(selection: Selection, record: _Record) -> None:
    fields = (
        ("identifier", selection.identifier, record.identifier),
        ("length", selection.length, record.length),
        ("header_sha256", selection.header_sha256, record.header_sha256),
    )
    for field, expected, actual in fields:
        if expected != actual:
            raise FastaSelectionError(
                f"identity mismatch for {selection.source} record "
                f"{selection.ordinal}: inventory {field} {expected!r}, "
                f"source {field} {actual!r}"
            )


def _atomic_write(output: Path, records: Sequence[_Record]) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
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
                digest.update(record.data)
                size_bytes += len(record.data)
                if not record.data.endswith(b"\n"):
                    handle.write(b"\n")
                    digest.update(b"\n")
                    size_bytes += 1
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return size_bytes, digest.hexdigest()


def select_records(
    inventory_path: Path,
    output: Path,
    file_ids: Sequence[str] | None = None,
    quantiles: Sequence[str] | None = None,
    include_aggregate: bool = True,
) -> SelectionSummary:
    """Extract inventory-selected records and atomically replace ``output``."""

    selections, identities = load_selections(
        inventory_path, file_ids, quantiles, include_aggregate
    )
    unique = _deduplicate(selections)
    output = output.expanduser().resolve()
    if output in {selection.source for selection in unique}:
        raise FastaSelectionError(f"output would overwrite a FASTA source: {output}")

    by_source: dict[Path, set[int]] = defaultdict(set)
    for selection in unique:
        by_source[selection.source].add(selection.ordinal)
    extracted: dict[tuple[Path, int], _Record] = {}
    for source in sorted(by_source, key=str):
        if source not in identities:
            raise FastaSelectionError(f"source has no inventory identity: {source}")
        for ordinal, record in _read_ordinals(
            source, by_source[source], identities[source]
        ).items():
            extracted[(source, ordinal)] = record

    records: list[_Record] = []
    for selection in unique:
        record = extracted[selection.record_key]
        _validate(selection, record)
        records.append(record)
    size_bytes, sha256 = _atomic_write(output, records)
    unique_identifiers = len({record.identifier for record in records})
    return SelectionSummary(
        len(selections),
        len(records),
        len(by_source),
        sum(record.length for record in records),
        size_bytes,
        sha256,
        len(records) - unique_identifiers,
        output,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract real FASTA records selected by a length inventory."
    )
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--file",
        action="append",
        help="inventory dataset id (repeatable; default: all files)",
    )
    parser.add_argument(
        "--quantile",
        action="append",
        help="quantile key such as p0 or p99 (repeatable; default: all)",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="exclude aggregate-corpus quantile records",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = select_records(
            args.inventory,
            args.output,
            args.file,
            args.quantile,
            not args.no_aggregate,
        )
    except FastaSelectionError as error:
        parser.error(str(error))
    print(
        f"wrote {summary.written} unique record(s), {summary.residues} residue(s), "
        f"sha256={summary.sha256} to {summary.output} "
        f"({summary.requested} inventory selection(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
