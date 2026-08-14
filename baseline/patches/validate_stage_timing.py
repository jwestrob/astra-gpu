#!/usr/bin/env python3
"""Validate the invariants in plan7-gpu's HMMER stage-timing TSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "schema_version",
    "clock",
    "query_index",
    "query",
    "metric",
    "count",
    "elapsed_ns",
]
STAGES = {
    "target_reconfig",
    "pipeline_total",
    "null1",
    "msv_public",
    "msv_ssv_attempt",
    "msv_classic_fallback",
    "bias_filter",
    "viterbi_filter",
    "forward_parser",
    "backward_parser",
    "domain_workflow",
}
COUNT_ONLY = {
    "clock_errors",
    "msv_ssv_status_ok",
    "msv_ssv_status_erange",
    "msv_ssv_status_noresult",
    "msv_ssv_status_other",
    "msv_fallback_status_ok",
    "msv_fallback_status_erange",
    "msv_fallback_status_other",
    "target_sequences",
    "target_residues",
    "passed_msv",
    "passed_bias",
    "passed_viterbi",
    "passed_forward",
}
EXPECTED = STAGES | COUNT_ONLY


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(path: Path) -> tuple[int, int, int]:
    queries: dict[int, tuple[str, dict[str, tuple[int, int]]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == FIELDS, f"{path}: unexpected header")
        for line_number, row in enumerate(reader, start=2):
            where = f"{path}:{line_number}"
            require(row["schema_version"] == "1", f"{where}: unsupported schema")
            require(bool(row["clock"]), f"{where}: empty clock")
            query_index = int(row["query_index"])
            count = int(row["count"])
            elapsed_ns = int(row["elapsed_ns"])
            metric = row["metric"]
            require(query_index > 0, f"{where}: query_index must be positive")
            require(count >= 0 and elapsed_ns >= 0, f"{where}: negative value")
            require(metric in EXPECTED, f"{where}: unknown metric {metric!r}")
            name, metrics = queries.setdefault(query_index, (row["query"], {}))
            require(name == row["query"], f"{where}: query name changed within index")
            require(metric not in metrics, f"{where}: duplicate metric {metric!r}")
            metrics[metric] = (count, elapsed_ns)

    require(bool(queries), f"{path}: no timing rows")
    require(
        sorted(queries) == list(range(1, len(queries) + 1)),
        f"{path}: query indices are not contiguous",
    )

    total_targets = 0
    total_fallbacks = 0
    for query_index, (name, rows) in queries.items():
        where = f"{path}: query {query_index} ({name})"
        require(set(rows) == EXPECTED, f"{where}: missing or extra metrics")
        value = {metric: rows[metric][0] for metric in EXPECTED}
        require(value["clock_errors"] == 0, f"{where}: monotonic clock read failed")
        for metric in STAGES:
            count, elapsed_ns = rows[metric]
            require(
                (count == 0) == (elapsed_ns == 0),
                f"{where}: invalid timing for {metric}",
            )
        for metric in COUNT_ONLY:
            require(
                rows[metric][1] == 0,
                f"{where}: count-only metric {metric} has elapsed time",
            )

        targets = value["target_sequences"]
        require(
            value["target_reconfig"] == targets,
            f"{where}: reconfiguration count mismatch",
        )
        require(value["pipeline_total"] == targets, f"{where}: pipeline count mismatch")
        require(
            value["null1"] == value["msv_public"], f"{where}: null/MSV count mismatch"
        )

        ssv_status = sum(
            value[f"msv_ssv_status_{status}"]
            for status in ("ok", "erange", "noresult", "other")
        )
        fallback_status = sum(
            value[f"msv_fallback_status_{status}"]
            for status in ("ok", "erange", "other")
        )
        require(
            ssv_status == value["msv_ssv_attempt"],
            f"{where}: SSV status count mismatch",
        )
        require(
            fallback_status == value["msv_classic_fallback"],
            f"{where}: fallback status count mismatch",
        )
        if value["msv_ssv_attempt"]:
            require(
                value["msv_ssv_attempt"] == value["msv_public"],
                f"{where}: SSV/public count mismatch",
            )
            require(
                value["msv_classic_fallback"] == value["msv_ssv_status_noresult"],
                f"{where}: fallback trigger mismatch",
            )
        else:  # VMX has no SSV prefilter.
            require(
                value["msv_classic_fallback"] == value["msv_public"],
                f"{where}: classic/public count mismatch",
            )

        require(
            value["forward_parser"] == value["passed_viterbi"],
            f"{where}: Forward count mismatch",
        )
        require(
            value["backward_parser"] == value["passed_forward"],
            f"{where}: Backward count mismatch",
        )
        require(
            value["domain_workflow"] == value["passed_forward"],
            f"{where}: domain count mismatch",
        )
        require(
            value["passed_forward"]
            <= value["passed_viterbi"]
            <= value["passed_bias"]
            <= value["passed_msv"]
            <= targets,
            f"{where}: filter survivor counts are not monotone",
        )
        require(
            value["bias_filter"] <= value["passed_msv"],
            f"{where}: bias call count too large",
        )
        require(
            value["viterbi_filter"] <= value["passed_bias"],
            f"{where}: Viterbi call count too large",
        )
        total_targets += targets
        total_fallbacks += value["msv_classic_fallback"]

    return len(queries), total_targets, total_fallbacks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.tsv:
        query_count, target_count, fallback_count = validate(path)
        print(
            f"ok\t{path}\tqueries={query_count}\ttargets={target_count}\tfallbacks={fallback_count}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
