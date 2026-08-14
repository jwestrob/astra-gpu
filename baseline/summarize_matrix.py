#!/usr/bin/env python3
"""Summarize a run_hmmer_matrix.py result directory."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def normalized_science(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: value for key, value in summary.items() if key not in {"reported_timing", "mc_per_second"}}
        for summary in summaries
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = []
    for path in sorted(args.result_dir.glob("*.json")):
        if path.name.endswith("-stats.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = path
        records.append(record)
    if not records:
        parser.error(f"no benchmark JSON records in {args.result_dir}")
    if any(record["exit_code"] != 0 for record in records):
        parser.error("at least one benchmark command failed")

    science_reference = None
    mismatches: list[str] = []
    stats_reference: list[dict[str, object]] | None = None
    for record in records:
        record_path: Path = record["_path"]
        stats_path = record_path.with_name(record_path.stem + "-stats.json")
        summaries = json.loads(stats_path.read_text(encoding="utf-8"))["summaries"]
        normalized = normalized_science(summaries)
        if science_reference is None:
            science_reference = normalized
            stats_reference = summaries
        elif normalized != science_reference:
            mismatches.append(stats_path.name)

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["metadata"]["run_kind"] == "replicate":
            grouped[int(record["metadata"]["hmmer_cpu_workers"])].append(record)

    scaling: dict[str, dict[str, object]] = {}
    for workers, group in sorted(grouped.items()):
        walls = [float(record["timing"]["wall_seconds"]) for record in group]
        cpus = [
            float(record["timing"]["user_seconds"]) + float(record["timing"]["system_seconds"])
            for record in group
        ]
        rss = [int(record["timing"]["max_rss_kib"]) for record in group]
        scaling[str(workers)] = {
            "replicates": len(group),
            "wall_seconds": {
                "median": statistics.median(walls),
                "min": min(walls),
                "max": max(walls),
                "mean": statistics.fmean(walls),
                "stdev": statistics.stdev(walls) if len(walls) > 1 else 0.0,
            },
            "cpu_seconds_median": statistics.median(cpus),
            "max_rss_kib_median": statistics.median(rss),
            "cpu_utilization_median": statistics.median(
                cpu / wall for cpu, wall in zip(cpus, walls, strict=True)
            ),
        }

    baseline_workers = min(grouped)
    baseline_wall = float(scaling[str(baseline_workers)]["wall_seconds"]["median"])
    for result in scaling.values():
        result["speedup_vs_cpu0"] = baseline_wall / float(result["wall_seconds"]["median"])

    aggregate_stages: dict[str, dict[str, object]] = {}
    if stats_reference:
        comparisons = sum(
            int(summary["inputs"]["target_sequences"]["count"]) for summary in stats_reference
        )
        for stage in ("msv", "bias", "viterbi", "forward"):
            passed = sum(int(summary["stages"][stage]["passed"]) for summary in stats_reference)
            aggregate_stages[stage] = {
                "passed": passed,
                "comparisons": comparisons,
                "fraction": passed / comparisons,
            }

    output = {
        "schema_version": 1,
        "source_dir": str(args.result_dir),
        "dataset_id": records[0]["metadata"]["dataset_id"],
        "run_status": sorted({record["metadata"]["run_status"] for record in records}),
        "hosts": sorted({record["host"]["hostname"] for record in records}),
        "scientific_summary_equivalent": not mismatches,
        "scientific_summary_mismatches": mismatches,
        "aggregate_stage_promotions": aggregate_stages,
        "scaling": scaling,
        "warning": "Pilot timings from a shared allocation are not reportable performance results.",
    }
    atomic_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
