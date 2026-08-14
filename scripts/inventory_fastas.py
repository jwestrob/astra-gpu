#!/usr/bin/env python3
"""Build an exact, bounded-memory length inventory for frozen FASTA files."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Protocol, Sequence


QUANTILES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
_FASTA_WHITESPACE = b" \t\r\n\v\f"


class FastaInventoryError(ValueError):
    """Raised when a FASTA or its frozen manifest identity is invalid."""


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...


@dataclass(frozen=True)
class FastaRecord:
    identifier: str
    length: int
    ordinal: int
    header_sha256: str


@dataclass(frozen=True)
class _FrozenFasta:
    dataset_id: str
    path: Path
    expected_size: int | None
    expected_sha256: str | None
    expected_sequences: int | None
    expected_residues: int | None


@dataclass(frozen=True)
class _FileScan:
    frozen: _FrozenFasta
    size_bytes: int
    mtime_ns: int
    signature: tuple[int, int, int, int]
    sha256: str
    sequences: int
    residues: int
    histogram: Counter[int]


@dataclass(frozen=True)
class _RankTarget:
    length: int
    offset_within_length: int


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FastaInventoryError(f"{context} must be a JSON object")
    return value


def _optional_nonnegative_int(
    item: Mapping[str, object], key: str, context: str
) -> int | None:
    if key not in item:
        return None
    value = item[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FastaInventoryError(f"{context}.{key} must be a nonnegative integer")
    return value


def _file_signature(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _residue_count(line: bytes) -> int:
    return len(line.translate(None, _FASTA_WHITESPACE))


def _identifier(header: bytes, path: Path, line_number: int) -> str:
    token = header[1:].strip().split(None, 1)
    if not token:
        raise FastaInventoryError(f"{path}:{line_number}: empty FASTA identifier")
    try:
        return token[0].decode("ascii")
    except UnicodeDecodeError as error:
        raise FastaInventoryError(
            f"{path}:{line_number}: non-ASCII FASTA identifier"
        ) from error


def records_in_file(
    path: Path, file_digest: _Digest | None = None
) -> Iterator[FastaRecord]:
    """Yield record metadata without retaining sequences in memory."""

    current_identifier: str | None = None
    current_length = 0
    current_header_sha256 = ""
    current_ordinal = -1
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                if file_digest is not None:
                    file_digest.update(line)
                if line.startswith(b">"):
                    if current_identifier is not None:
                        yield FastaRecord(
                            current_identifier,
                            current_length,
                            current_ordinal,
                            current_header_sha256,
                        )
                    current_ordinal += 1
                    current_identifier = _identifier(line, path, line_number)
                    current_length = 0
                    current_header_sha256 = hashlib.sha256(line).hexdigest()
                    continue

                residues = _residue_count(line)
                if current_identifier is None:
                    if residues:
                        raise FastaInventoryError(
                            f"{path}:{line_number}: sequence data before first header"
                        )
                else:
                    current_length += residues
    except OSError as error:
        raise FastaInventoryError(f"cannot read FASTA {path}: {error}") from error

    if current_identifier is not None:
        yield FastaRecord(
            current_identifier,
            current_length,
            current_ordinal,
            current_header_sha256,
        )


def _load_manifest(manifest_path: Path) -> list[_FrozenFasta]:
    try:
        with manifest_path.open("rt", encoding="utf-8") as handle:
            root = _mapping(json.load(handle), "manifest")
    except (OSError, json.JSONDecodeError) as error:
        raise FastaInventoryError(
            f"cannot read dataset manifest {manifest_path}: {error}"
        ) from error

    raw_fastas = root.get("protein_fastas")
    if not isinstance(raw_fastas, list) or not raw_fastas:
        raise FastaInventoryError("manifest.protein_fastas must be a nonempty array")

    result: list[_FrozenFasta] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for index, raw_item in enumerate(raw_fastas):
        context = f"manifest.protein_fastas[{index}]"
        item = _mapping(raw_item, context)
        dataset_id = item.get("id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise FastaInventoryError(f"{context}.id must be a nonempty string")
        if dataset_id in seen_ids:
            raise FastaInventoryError(f"duplicate FASTA dataset id: {dataset_id}")
        seen_ids.add(dataset_id)

        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise FastaInventoryError(f"{context}.path must be a nonempty string")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if path in seen_paths:
            raise FastaInventoryError(f"duplicate FASTA source path: {path}")
        seen_paths.add(path)

        expected_sha256 = item.get("sha256")
        if expected_sha256 is not None:
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
            ):
                raise FastaInventoryError(
                    f"{context}.sha256 must be a lowercase SHA-256 digest"
                )
        result.append(
            _FrozenFasta(
                dataset_id,
                path,
                _optional_nonnegative_int(item, "size_bytes", context),
                expected_sha256,
                _optional_nonnegative_int(item, "sequences", context),
                _optional_nonnegative_int(item, "residues", context),
            )
        )
    return result


def _verify_expected(
    frozen: _FrozenFasta,
    size_bytes: int,
    sha256: str,
    sequences: int,
    residues: int,
) -> None:
    checks = (
        ("size_bytes", frozen.expected_size, size_bytes),
        ("sha256", frozen.expected_sha256, sha256),
        ("sequences", frozen.expected_sequences, sequences),
        ("residues", frozen.expected_residues, residues),
    )
    for field, expected, actual in checks:
        if expected is not None and expected != actual:
            raise FastaInventoryError(
                f"{frozen.dataset_id}: frozen {field} mismatch: "
                f"expected {expected}, observed {actual}"
            )


def _scan_file(frozen: _FrozenFasta) -> _FileScan:
    try:
        before = frozen.path.stat()
    except OSError as error:
        raise FastaInventoryError(f"cannot stat FASTA {frozen.path}: {error}") from error
    if not frozen.path.is_file():
        raise FastaInventoryError(f"FASTA source is not a regular file: {frozen.path}")

    digest = hashlib.sha256()
    histogram: Counter[int] = Counter()
    sequences = 0
    residues = 0
    for record in records_in_file(frozen.path, digest):
        sequences += 1
        residues += record.length
        histogram[record.length] += 1
    if not sequences:
        raise FastaInventoryError(f"FASTA contains no records: {frozen.path}")

    try:
        after = frozen.path.stat()
    except OSError as error:
        raise FastaInventoryError(f"cannot restat FASTA {frozen.path}: {error}") from error
    if _file_signature(before) != _file_signature(after):
        raise FastaInventoryError(f"FASTA changed while being read: {frozen.path}")

    sha256 = digest.hexdigest()
    _verify_expected(frozen, after.st_size, sha256, sequences, residues)
    return _FileScan(
        frozen,
        after.st_size,
        after.st_mtime_ns,
        _file_signature(after),
        sha256,
        sequences,
        residues,
        histogram,
    )


def _quantile_key(quantile: float) -> str:
    return f"p{quantile * 100:g}"


def _rank_targets(histogram: Counter[int], count: int) -> dict[str, _RankTarget]:
    if count <= 0:
        raise FastaInventoryError("cannot compute quantiles for an empty FASTA")
    ordered = sorted(histogram.items())
    targets: dict[str, _RankTarget] = {}
    for quantile in QUANTILES:
        rank = round(quantile * (count - 1))
        before = 0
        for length, frequency in ordered:
            if rank < before + frequency:
                targets[_quantile_key(quantile)] = _RankTarget(
                    length, rank - before
                )
                break
            before += frequency
        else:  # pragma: no cover - guarded by count/histogram consistency
            raise FastaInventoryError(f"failed to resolve quantile rank {rank}")
    return targets


def _target_lookup(
    targets: Mapping[str, _RankTarget],
) -> dict[tuple[int, int], list[str]]:
    lookup: dict[tuple[int, int], list[str]] = defaultdict(list)
    for quantile, target in targets.items():
        lookup[(target.length, target.offset_within_length)].append(quantile)
    return lookup


def _selection(record: FastaRecord, scan: _FileScan) -> dict[str, object]:
    return {
        "dataset_id": scan.frozen.dataset_id,
        "header_sha256": record.header_sha256,
        "identifier": record.identifier,
        "length": record.length,
        "ordinal_in_file": record.ordinal,
        "source": str(scan.frozen.path),
    }


def _resolve_quantile_records(
    scans: Sequence[_FileScan],
    file_targets: Sequence[Mapping[str, _RankTarget]],
    aggregate_targets: Mapping[str, _RankTarget],
) -> tuple[list[dict[str, dict[str, object]]], dict[str, dict[str, object]]]:
    per_file: list[dict[str, dict[str, object]]] = []
    aggregate: dict[str, dict[str, object]] = {}
    aggregate_seen: Counter[int] = Counter()
    aggregate_lookup = _target_lookup(aggregate_targets)

    for scan, targets in zip(scans, file_targets, strict=True):
        try:
            before = scan.frozen.path.stat()
        except OSError as error:
            raise FastaInventoryError(
                f"cannot stat FASTA {scan.frozen.path}: {error}"
            ) from error
        if _file_signature(before) != scan.signature:
            raise FastaInventoryError(
                f"FASTA changed between inventory passes: {scan.frozen.path}"
            )

        file_seen: Counter[int] = Counter()
        file_lookup = _target_lookup(targets)
        picks: dict[str, dict[str, object]] = {}
        observed = 0
        for record in records_in_file(scan.frozen.path):
            observed += 1
            file_offset = file_seen[record.length]
            file_seen[record.length] += 1
            file_quantiles = file_lookup.get((record.length, file_offset), ())

            aggregate_offset = aggregate_seen[record.length]
            aggregate_seen[record.length] += 1
            aggregate_quantiles = aggregate_lookup.get(
                (record.length, aggregate_offset), ()
            )
            if file_quantiles or aggregate_quantiles:
                selected = _selection(record, scan)
                for quantile in file_quantiles:
                    picks[quantile] = dict(selected)
                for quantile in aggregate_quantiles:
                    aggregate[quantile] = dict(selected)

        try:
            after = scan.frozen.path.stat()
        except OSError as error:
            raise FastaInventoryError(
                f"cannot restat FASTA {scan.frozen.path}: {error}"
            ) from error
        if _file_signature(after) != scan.signature:
            raise FastaInventoryError(
                f"FASTA changed during quantile pass: {scan.frozen.path}"
            )
        if observed != scan.sequences or len(picks) != len(targets):
            raise FastaInventoryError(
                f"failed to resolve every quantile for {scan.frozen.dataset_id}"
            )
        per_file.append(picks)

    if len(aggregate) != len(aggregate_targets):
        raise FastaInventoryError("failed to resolve every aggregate quantile")
    return per_file, aggregate


def _length_summary(
    histogram: Counter[int],
    sequences: int,
    residues: int,
    targets: Mapping[str, _RankTarget],
) -> dict[str, object]:
    return {
        "max": max(histogram),
        "mean": residues / sequences,
        "min": min(histogram),
        "quantiles": {
            quantile: target.length for quantile, target in targets.items()
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise FastaInventoryError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def build_inventory(manifest_path: Path, output: Path) -> dict[str, object]:
    """Verify frozen FASTAs, compute exact quantiles, and write inventory JSON."""

    manifest_path = manifest_path.expanduser().resolve()
    output = output.expanduser().resolve()
    frozen_fastas = _load_manifest(manifest_path)
    source_paths = {item.path for item in frozen_fastas}
    if output == manifest_path or output in source_paths:
        raise FastaInventoryError(f"output would overwrite an input: {output}")

    scans = [_scan_file(item) for item in frozen_fastas]
    aggregate_histogram: Counter[int] = Counter()
    for scan in scans:
        aggregate_histogram.update(scan.histogram)
    aggregate_sequences = sum(scan.sequences for scan in scans)
    aggregate_residues = sum(scan.residues for scan in scans)

    file_targets = [
        _rank_targets(scan.histogram, scan.sequences) for scan in scans
    ]
    aggregate_targets = _rank_targets(aggregate_histogram, aggregate_sequences)
    file_picks, aggregate_picks = _resolve_quantile_records(
        scans, file_targets, aggregate_targets
    )

    files: list[dict[str, object]] = []
    for scan, targets, picks in zip(scans, file_targets, file_picks, strict=True):
        files.append(
            {
                "dataset_id": scan.frozen.dataset_id,
                "length": _length_summary(
                    scan.histogram, scan.sequences, scan.residues, targets
                ),
                "quantile_records": picks,
                "residues": scan.residues,
                "sequences": scan.sequences,
                "source": str(scan.frozen.path),
                "source_mtime_ns": scan.mtime_ns,
                "source_sha256": scan.sha256,
                "source_size_bytes": scan.size_bytes,
            }
        )

    result: dict[str, object] = {
        "aggregate": {
            "files": len(scans),
            "length": _length_summary(
                aggregate_histogram,
                aggregate_sequences,
                aggregate_residues,
                aggregate_targets,
            ),
            "quantile_records": aggregate_picks,
            "residues": aggregate_residues,
            "sequences": aggregate_sequences,
        },
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": files,
        "quantile_method": (
            "index=round(q*(n-1)); records ordered by length, then manifest "
            "file order, then zero-based ordinal"
        ),
        "quantiles": [_quantile_key(value) for value in QUANTILES],
        "schema_version": 1,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory frozen protein FASTAs and select exact length quantiles."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = build_inventory(args.manifest, args.output)
    except FastaInventoryError as error:
        parser.error(str(error))
    aggregate = _mapping(result["aggregate"], "aggregate")
    print(
        f"inventoried {aggregate['sequences']} sequence(s), "
        f"{aggregate['residues']} residue(s) in {aggregate['files']} file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
