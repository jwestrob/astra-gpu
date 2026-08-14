#!/usr/bin/env python3
"""Validate the Astra HMMER preload probe's aggregate TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_STAGES = (
    "p7_Pipeline",
    "p7_bg_NullOne",
    "p7_MSVFilter",
    "p7_SSVFilter",
    "p7_bg_FilterScore",
    "p7_ViterbiFilter",
    "p7_ForwardParser",
    "p7_BackwardParser",
    "p7_domaindef_ByPosteriorHeuristics",
    "p7_Forward",
    "p7_Backward",
    "p7_DomainDecoding",
)
NUMERIC_COLUMNS = (
    "calls",
    "elapsed_ns",
    "status_ok",
    "status_erange",
    "status_noresult",
    "status_other",
)
STATUS_COLUMNS = (
    "status_ok",
    "status_erange",
    "status_noresult",
    "status_other",
)


def parse_probe(path: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    metadata: dict[str, list[str]] = {}
    data_lines: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                fields = line[1:].split("\t")
                if not fields[0] or fields[0] in metadata:
                    raise ValueError(f"invalid or duplicate metadata row: {line!r}")
                metadata[fields[0]] = fields[1:]
            elif line:
                data_lines.append(line)

    if metadata.get("schema") != ["plan7_astra_stage_probe", "1"]:
        raise ValueError("unsupported or missing probe schema")
    if metadata.get("clock") != ["CLOCK_MONOTONIC"]:
        raise ValueError("probe did not use CLOCK_MONOTONIC")
    if metadata.get("aggregation") != ["process_wide_inclusive"]:
        raise ValueError("unexpected aggregation mode")
    if not data_lines:
        raise ValueError("missing TSV data")

    reader = csv.DictReader(data_lines, delimiter="\t")
    expected_header = ["stage", *NUMERIC_COLUMNS]
    if reader.fieldnames != expected_header:
        raise ValueError(f"unexpected TSV header: {reader.fieldnames!r}")

    stages: dict[str, dict[str, int]] = {}
    for row in reader:
        stage = row.pop("stage")
        if stage in stages:
            raise ValueError(f"duplicate stage: {stage}")
        try:
            values = {key: int(row[key]) for key in NUMERIC_COLUMNS}
        except (TypeError, ValueError) as error:
            raise ValueError(f"non-integer metric for {stage}") from error
        if any(value < 0 for value in values.values()):
            raise ValueError(f"negative metric for {stage}")
        if sum(values[key] for key in STATUS_COLUMNS) != values["calls"]:
            raise ValueError(f"status counts do not sum to calls for {stage}")
        if values["calls"] == 0 and values["elapsed_ns"] != 0:
            raise ValueError(f"elapsed time without calls for {stage}")
        stages[stage] = values

    if tuple(stages) != EXPECTED_STAGES:
        raise ValueError("missing, extra, or reordered stages")
    if metadata.get("clock_errors") != ["0"]:
        raise ValueError(f"probe clock error: {metadata.get('clock_errors')!r}")
    for stage, values in stages.items():
        if values["calls"] and values["elapsed_ns"] == 0:
            raise ValueError(f"zero aggregate duration for called stage {stage}")

    if stages["p7_MSVFilter"]["calls"] > stages["p7_Pipeline"]["calls"]:
        raise ValueError("more MSV calls than pipeline calls")
    if stages["p7_SSVFilter"]["calls"] != stages["p7_MSVFilter"]["calls"]:
        raise ValueError("public MSV did not make exactly one observed SSV attempt")

    return metadata, stages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("--expect-pipeline-calls", type=int)
    args = parser.parse_args()

    metadata, stages = parse_probe(args.probe)
    pipeline_calls = stages["p7_Pipeline"]["calls"]
    if pipeline_calls == 0:
        raise SystemExit("probe saw no p7_Pipeline calls")
    if (
        args.expect_pipeline_calls is not None
        and pipeline_calls != args.expect_pipeline_calls
    ):
        raise SystemExit(
            f"expected {args.expect_pipeline_calls} pipeline calls, saw {pipeline_calls}"
        )

    summary = {
        "schema_version": 1,
        "target_library": metadata["target_library"][0],
        "pipeline_calls": pipeline_calls,
        "msv_calls": stages["p7_MSVFilter"]["calls"],
        "ssv_calls": stages["p7_SSVFilter"]["calls"],
        "ssv_status": {
            key.removeprefix("status_"): stages["p7_SSVFilter"][key]
            for key in STATUS_COLUMNS
        },
        "bias_calls": stages["p7_bg_FilterScore"]["calls"],
        "viterbi_calls": stages["p7_ViterbiFilter"]["calls"],
        "forward_parser_calls": stages["p7_ForwardParser"]["calls"],
        "domain_workflow_calls": stages["p7_domaindef_ByPosteriorHeuristics"]["calls"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
