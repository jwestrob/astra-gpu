#!/usr/bin/env python3
"""Inventory HMMER text profiles without materializing DP matrices."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import statistics
import tempfile
from collections import Counter
from pathlib import Path


HEADER_FIELDS = {"NAME", "ACC", "LENG", "GA", "TC", "NC"}
QUANTILES = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def parse_spec(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be LABEL=PATH_OR_GLOB")
    label, pattern = value.split("=", 1)
    if not label or not pattern:
        raise argparse.ArgumentTypeError("dataset must be LABEL=PATH_OR_GLOB")
    return label, pattern


def records_in_file(path: Path):
    current: dict[str, object] = {"source": str(path)}
    in_model_body = False
    with path.open("rt", encoding="ascii", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line.startswith("HMMER3/") and "format" not in current:
                current["format"] = line
                continue
            if line == "//":
                if "LENG" in current:
                    yield current
                current = {"source": str(path)}
                in_model_body = False
                continue
            if in_model_body or not line:
                continue
            key = line.split(maxsplit=1)[0]
            if key == "HMM":
                in_model_body = True
                continue
            if key in HEADER_FIELDS:
                value = line[len(key) :].strip()
                current[key] = int(value) if key == "LENG" else value
    if "LENG" in current:
        yield current


def summarize(label: str, pattern: str) -> dict[str, object]:
    paths = sorted(Path(path).resolve() for path in glob.glob(pattern))
    if not paths:
        raise ValueError(f"{label}: pattern matched no files: {pattern}")

    records: list[dict[str, object]] = []
    metadata_digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        stat = path.stat()
        total_bytes += stat.st_size
        metadata_digest.update(
            f"FILE\t{path}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode("utf-8")
        )
        for ordinal_in_file, record in enumerate(records_in_file(path)):
            record["ordinal_in_file"] = ordinal_in_file
            records.append(record)
            metadata_digest.update(
                (
                    "RECORD\t"
                    + "\t".join(
                        str(record.get(key, ""))
                        for key in ("source", "ordinal_in_file", "format", "NAME", "ACC", "LENG", "GA", "TC", "NC")
                    )
                    + "\n"
                ).encode("utf-8")
            )
    if not records:
        raise ValueError(f"{label}: no HMM records found")

    ordered = sorted(records, key=lambda record: (int(record["LENG"]), str(record.get("NAME", ""))))
    lengths = [int(record["LENG"]) for record in ordered]
    quantiles: dict[str, dict[str, object]] = {}
    for quantile in QUANTILES:
        index = round(quantile * (len(ordered) - 1))
        record = ordered[index]
        key = f"p{quantile * 100:g}"
        quantiles[key] = {
            "length": record["LENG"],
            "name": record.get("NAME"),
            "accession": record.get("ACC"),
            "source": record["source"],
            "ordinal_in_file": record["ordinal_in_file"],
        }

    return {
        "label": label,
        "input_pattern": pattern,
        "files": len(paths),
        "size_bytes": total_bytes,
        "profiles": len(records),
        "total_profile_states": sum(lengths),
        "length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
        },
        "formats": dict(sorted(Counter(str(record.get("format", "unknown")) for record in records).items())),
        "cutoff_counts": {
            key: sum(key in record for record in records) for key in ("GA", "TC", "NC")
        },
        "metadata_sha256": metadata_digest.hexdigest(),
        "quantile_profiles": quantiles,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=parse_spec, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "datasets": {label: summarize(label, pattern) for label, pattern in args.dataset},
        "digest_scope": "File path/size/mtime plus selected HMM header fields; not a content digest.",
    }
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
