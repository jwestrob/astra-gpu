#!/usr/bin/env python3
"""Reduce verbose byte-MSV oracle JSONL to a reviewable result record."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path


class OracleSummaryError(ValueError):
    pass


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def summarize(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    metadata: dict[str, object] | None = None
    reported_summary: dict[str, object] | None = None
    paths: Counter[str] = Counter()
    public_statuses: Counter[str] = Counter()
    scalar_statuses: Counter[str] = Counter()
    agreement_references: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    models: dict[str, dict[str, int]] = {}
    sequence_indexes: set[int] = set()
    sequence_lengths: set[int] = set()
    tjb_values: set[int] = set()
    comparisons = 0
    mismatches = 0
    public_vs_full_status_mismatches = 0
    public_vs_full_score_mismatches = 0
    passes = 0

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise OracleSummaryError(f"{path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise OracleSummaryError(f"{path}:{line_number}: record is not an object")
            kind = record.get("record")
            if reported_summary is not None:
                raise OracleSummaryError(f"{path}:{line_number}: data follows summary")
            if kind == "metadata":
                if metadata is not None or comparisons:
                    raise OracleSummaryError(f"{path}:{line_number}: misplaced metadata")
                metadata = record
            elif kind == "comparison":
                if metadata is None:
                    raise OracleSummaryError(f"{path}:{line_number}: comparison precedes metadata")
                comparisons += 1
                paths[str(record["msv_path"])] += 1
                public_statuses[str(record["public_msv"]["status"])] += 1
                scalar_statuses[str(record["scalar_full_msv"]["status"])] += 1
                agreement = record["agreement"]
                agreement_references[str(agreement.get("reference", "legacy"))] += 1
                if not agreement["status"] or not agreement["score_bits"]:
                    mismatches += 1
                full_agreement = agreement.get("public_vs_full_msv")
                if full_agreement is not None:
                    public_vs_full_status_mismatches += not full_agreement["status"]
                    public_vs_full_score_mismatches += not full_agreement["score_bits"]
                passes += bool(record["pass_F1"])
                name = str(record["model_name"])
                model = models.setdefault(name, {"M": int(record["M"]), "comparisons": 0})
                if model["M"] != int(record["M"]):
                    raise OracleSummaryError(f"{path}:{line_number}: inconsistent M for {name}")
                model["comparisons"] += 1
                sequence_indexes.add(int(record["sequence_index"]))
                sequence_lengths.add(int(record["L"]))
                tjb_values.add(int(record["profile_u8"]["tjb_b"]))
            elif kind == "skipped":
                if metadata is None:
                    raise OracleSummaryError(f"{path}:{line_number}: skipped record precedes metadata")
                skipped[str(record["reason"])] += 1
            elif kind == "summary":
                reported_summary = record
            else:
                raise OracleSummaryError(
                    f"{path}:{line_number}: unknown record kind {kind!r}"
                )

    if metadata is None or reported_summary is None:
        raise OracleSummaryError(f"{path}: metadata or terminal summary is missing")
    if int(reported_summary.get("comparisons", -1)) != comparisons:
        raise OracleSummaryError(f"{path}: comparison count disagrees with terminal summary")
    if int(reported_summary.get("mismatches", -1)) != mismatches:
        raise OracleSummaryError(f"{path}: mismatch count disagrees with terminal summary")
    reported_skipped = int(reported_summary.get("skipped_empty", 0))
    observed_skipped = skipped.get("pipeline_skips_empty_sequence", 0)
    if observed_skipped and reported_skipped != observed_skipped:
        raise OracleSummaryError(f"{path}: skipped-empty count disagrees with terminal summary")
    if reported_skipped and not observed_skipped:
        skipped["pipeline_skips_empty_sequence"] = reported_skipped

    return {
        "schema_version": 1,
        "source": {"path": str(path.resolve()), "sha256": digest.hexdigest()},
        "oracle_metadata": metadata,
        "comparisons": comparisons,
        "mismatches": mismatches,
        "agreement_references": dict(sorted(agreement_references.items())),
        "public_vs_full_msv": {
            "status_mismatches": public_vs_full_status_mismatches,
            "score_bit_mismatches": public_vs_full_score_mismatches,
        },
        "passes_F1": passes,
        "msv_paths": dict(sorted(paths.items())),
        "public_statuses": dict(sorted(public_statuses.items())),
        "scalar_statuses": dict(sorted(scalar_statuses.items())),
        "skipped": dict(sorted(skipped.items())),
        "models": dict(sorted(models.items())),
        "coverage": {
            "distinct_sequence_indexes": len(sequence_indexes),
            "distinct_sequence_lengths": len(sequence_lengths),
            "sequence_length_min": min(sequence_lengths) if sequence_lengths else None,
            "sequence_length_max": max(sequence_lengths) if sequence_lengths else None,
            "distinct_tjb_b": len(tjb_values),
            "tjb_b_values": sorted(tjb_values),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        result = summarize(args.input)
    except (OSError, KeyError, TypeError, OracleSummaryError) as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
