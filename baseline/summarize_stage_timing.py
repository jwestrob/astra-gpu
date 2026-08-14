#!/usr/bin/env python3
"""Summarize serial HMMER stage timings without confusing them with Astra."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from baseline.patches.validate_stage_timing import COUNT_ONLY, STAGES, validate
else:
    try:  # Package import in tests.
        from .patches.validate_stage_timing import COUNT_ONLY, STAGES, validate
    except ImportError:  # Direct execution from the repository root.
        from patches.validate_stage_timing import COUNT_ONLY, STAGES, validate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
JsonDict = dict[str, Any]
DEFAULT_ASTRA_SUMMARY = (
    PROJECT_ROOT / "results/baseline/pilot-astra-hyddb-plm2_5-summary.json"
)
DEFAULT_TIMING_PATCH = PROJECT_ROOT / "baseline/patches/hmmer-3.4-stage-timing.patch"
STAGE_ORDER = (
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
)
PASS_METRICS = {
    "msv": "passed_msv",
    "bias": "passed_bias",
    "viterbi": "passed_viterbi",
    "forward": "passed_forward",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    require(path.is_file(), f"missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_table_digest(path: Path) -> JsonDict:
    """Hash HMMER scientific rows after removing comments and trailing space."""
    require(path.is_file(), f"missing scientific table: {path}")
    digest = hashlib.sha256()
    rows = 0
    with path.open("rt", encoding="utf-8", errors="replace", newline=None) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            digest.update(line.rstrip().encode("utf-8"))
            digest.update(b"\n")
            rows += 1
    return {"sha256": digest.hexdigest(), "rows": rows}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def command_option(command: list[object], option: str) -> str:
    values: list[str] = []
    for index, raw in enumerate(command):
        value = str(raw)
        if value == option:
            require(index + 1 < len(command), f"{option} has no value in command")
            values.append(str(command[index + 1]))
        elif value.startswith(option + "="):
            values.append(value.split("=", 1)[1])
    require(len(values) == 1, f"command must contain exactly one {option}")
    return values[0]


def command_path(record: JsonDict, option: str) -> Path:
    command = record.get("command")
    if not isinstance(command, list):
        raise ValueError("run JSON command must be a list")
    value = Path(command_option(command, option))
    cwd = Path(str(record.get("cwd", "")))
    require(
        value.is_absolute() or cwd.is_absolute(), "relative command path has no cwd"
    )
    return value.resolve() if value.is_absolute() else (cwd / value).resolve()


def positional_input_paths(record: JsonDict) -> tuple[Path, Path]:
    command = record.get("command")
    if not isinstance(command, list) or len(command) < 3:
        raise ValueError("run command is too short")
    cwd = Path(str(record.get("cwd", "")))
    result = []
    for raw in command[-2:]:
        value = Path(str(raw))
        require(
            value.is_absolute() or cwd.is_absolute(), "relative input path has no cwd"
        )
        result.append(
            value.resolve() if value.is_absolute() else (cwd / value).resolve()
        )
    return result[0], result[1]


def load_timing(path: Path) -> JsonDict:
    query_count, target_count, fallback_count = validate(path)
    totals = {metric: {"count": 0, "elapsed_ns": 0} for metric in STAGES | COUNT_ONLY}
    clocks: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            clocks.add(row["clock"])
            metric = row["metric"]
            totals[metric]["count"] += int(row["count"])
            totals[metric]["elapsed_ns"] += int(row["elapsed_ns"])
    require(len(clocks) == 1, f"{path}: timing clock changed between rows")
    require(
        totals["target_sequences"]["count"] == target_count,
        f"{path}: validator target total disagrees with parsed rows",
    )
    require(
        totals["msv_classic_fallback"]["count"] == fallback_count,
        f"{path}: validator fallback total disagrees with parsed rows",
    )
    return {
        "query_count": query_count,
        "clock": next(iter(clocks)),
        "metrics": totals,
    }


def load_run(run_json: Path, timing_tsv: Path) -> JsonDict:
    require(run_json.is_file(), f"missing run JSON: {run_json}")
    record = json.loads(run_json.read_text(encoding="utf-8"))
    require(isinstance(record, dict), f"{run_json}: top level must be an object")
    require(record.get("schema_version") == 1, f"{run_json}: unsupported schema")
    require(record.get("exit_code") == 0, f"{run_json}: benchmark command failed")

    metadata = record.get("metadata")
    require(isinstance(metadata, dict), f"{run_json}: metadata must be an object")
    require(
        metadata.get("run_kind") == "replicate",
        f"{run_json}: only replicate runs may be summarized",
    )
    require(bool(metadata.get("dataset_id")), f"{run_json}: missing dataset_id")
    run_index = int(metadata.get("run_index", 0))
    require(run_index > 0, f"{run_json}: run_index must be positive")

    command = record.get("command")
    require(isinstance(command, list), f"{run_json}: command must be a list")
    cpu_workers = int(command_option(command, "--cpu"))
    require(
        cpu_workers == 0,
        f"{run_json}: Amdahl fractions require serial --cpu 0 timings; "
        "threaded TSV elapsed values are summed worker time",
    )
    if "hmmer_cpu_workers" in metadata:
        require(
            int(metadata["hmmer_cpu_workers"]) == cpu_workers,
            f"{run_json}: metadata and command CPU counts disagree",
        )

    expected_timing = command_path(record, "--stage-timing")
    require(
        timing_tsv.resolve() == expected_timing,
        f"{run_json}: supplied TSV is not the command's --stage-timing output",
    )
    tables = {
        "targets": command_path(record, "--tblout"),
        "domains": command_path(record, "--domtblout"),
    }

    timing = record.get("timing")
    require(isinstance(timing, dict), f"{run_json}: missing external timing")
    wall_seconds = float(timing.get("wall_seconds", 0.0))
    require(
        math.isfinite(wall_seconds) and wall_seconds > 0.0,
        f"{run_json}: external wall time must be positive and finite",
    )
    timing_data = load_timing(timing_tsv)
    executable = record.get("executable")
    require(isinstance(executable, dict), f"{run_json}: missing executable provenance")
    executable_path = executable.get("path")
    executable_sha256 = executable.get("sha256")
    require(bool(executable_path), f"{run_json}: missing executable path")
    require(
        isinstance(executable_sha256, str)
        and len(executable_sha256) == 64
        and all(character in "0123456789abcdef" for character in executable_sha256),
        f"{run_json}: invalid executable SHA-256",
    )
    query_hmm, target_fasta = positional_input_paths(record)
    return {
        "run_json": run_json.resolve(),
        "timing_tsv": timing_tsv.resolve(),
        "record": record,
        "metadata": metadata,
        "label": str(record.get("label", run_json.stem)),
        "run_index": run_index,
        "wall_seconds": wall_seconds,
        "tables": tables,
        "timing": timing_data,
        "executable": {
            "path": Path(str(executable_path)).resolve(),
            "sha256": executable_sha256,
        },
        "query_hmm": query_hmm,
        "target_fasta": target_fasta,
    }


def amdahl_speedup(fraction: float, acceleration: float | None) -> float | None:
    require(0.0 <= fraction <= 1.0, "Amdahl fraction is outside [0, 1]")
    denominator = 1.0 - fraction
    if acceleration is not None:
        require(acceleration > 0.0, "acceleration must be positive")
        denominator += fraction / acceleration
    return None if denominator == 0.0 else 1.0 / denominator


def summarize(
    run_pairs: list[tuple[Path, Path]],
    pristine_tblout: Path,
    pristine_domtblout: Path,
    astra_summary: Path,
    timing_patch: Path = DEFAULT_TIMING_PATCH,
) -> JsonDict:
    require(bool(run_pairs), "at least one run JSON/TSV pair is required")
    runs = [load_run(run_json, timing_tsv) for run_json, timing_tsv in run_pairs]
    runs.sort(key=lambda run: (int(run["run_index"]), str(run["label"])))

    labels = [str(run["label"]) for run in runs]
    require(len(set(labels)) == len(labels), "replicate labels must be unique")
    dataset_ids = {str(run["metadata"]["dataset_id"]) for run in runs}  # type: ignore[index]
    require(len(dataset_ids) == 1, "replicates use different datasets")
    run_statuses = {str(run["metadata"].get("run_status", "unknown")) for run in runs}  # type: ignore[union-attr]
    require(run_statuses == {"pilot"}, "stage summary is reserved for pilot runs")
    clocks = {str(run["timing"]["clock"]) for run in runs}  # type: ignore[index]
    require(len(clocks) == 1, "replicates use different timing clocks")
    executable_provenance = {
        (str(run["executable"]["path"]), str(run["executable"]["sha256"]))  # type: ignore[index]
        for run in runs
    }
    require(
        len(executable_provenance) == 1,
        "replicates use different instrumented executables or executable hashes",
    )
    query_paths = {Path(run["query_hmm"]) for run in runs}  # type: ignore[arg-type]
    target_paths = {Path(run["target_fasta"]) for run in runs}  # type: ignore[arg-type]
    require(len(query_paths) == 1, "replicates use different query HMM paths")
    require(len(target_paths) == 1, "replicates use different target FASTA paths")
    executable_path, executable_sha256 = next(iter(executable_provenance))
    query_hmm = next(iter(query_paths))
    target_fasta = next(iter(target_paths))

    count_sets: list[dict[str, int]] = []
    for run in runs:
        metrics = run["timing"]["metrics"]  # type: ignore[index]
        count_sets.append(
            {metric: int(value["count"]) for metric, value in metrics.items()}
        )
    require(
        all(counts == count_sets[0] for counts in count_sets[1:]),
        "stage/counter call counts differ between replicates",
    )
    counts = count_sets[0]
    comparisons = counts["target_sequences"]
    require(comparisons > 0, "timing inputs contain no target comparisons")
    require(counts["msv_ssv_attempt"] > 0, "SSV fallback rate has no attempts")

    pristine_paths = {
        "targets": pristine_tblout.resolve(),
        "domains": pristine_domtblout.resolve(),
    }
    reference_tables = {
        name: scientific_table_digest(path) for name, path in pristine_paths.items()
    }
    timed_tables: list[JsonDict] = []
    for run in runs:
        digests = {
            name: scientific_table_digest(path)
            for name, path in run["tables"].items()  # type: ignore[union-attr]
        }
        require(
            digests == reference_tables,
            f"{run['label']}: timed scientific tables differ from pristine HMMER",
        )
        timed_tables.append({"label": run["label"], **digests})

    astra_summary = astra_summary.resolve()
    require(
        astra_summary.is_file(), f"missing Astra application summary: {astra_summary}"
    )
    timing_patch = timing_patch.resolve()

    replicate_output: list[JsonDict] = []
    stage_replicates: dict[str, list[float]] = {stage: [] for stage in STAGE_ORDER}
    f_pipeline_values: list[float] = []
    f_e2e_values: list[float] = []
    for run in runs:
        metrics = run["timing"]["metrics"]  # type: ignore[index]
        stage_seconds = {
            stage: int(metrics[stage]["elapsed_ns"]) / 1_000_000_000
            for stage in STAGE_ORDER
        }
        for stage, seconds in stage_seconds.items():
            stage_replicates[stage].append(seconds)
        require(stage_seconds["pipeline_total"] > 0.0, "pipeline elapsed time is zero")
        f_pipeline = stage_seconds["msv_public"] / stage_seconds["pipeline_total"]
        f_e2e = stage_seconds["msv_public"] / float(run["wall_seconds"])
        require(
            0.0 <= f_pipeline <= 1.0,
            f"{run['label']}: MSV time exceeds inclusive pipeline time",
        )
        require(
            0.0 <= f_e2e <= 1.0,
            f"{run['label']}: MSV time exceeds external serial wall time",
        )
        f_pipeline_values.append(f_pipeline)
        f_e2e_values.append(f_e2e)
        record = run["record"]
        host = record.get("host", {})  # type: ignore[union-attr]
        replicate_output.append(
            {
                "label": run["label"],
                "run_index": run["run_index"],
                "host": host.get("hostname") if isinstance(host, dict) else None,
                "external_wall_seconds": run["wall_seconds"],
                "stage_seconds": stage_seconds,
                "f_pipeline": f_pipeline,
                "f_e2e": f_e2e,
            }
        )

    median_f_pipeline = statistics.median(f_pipeline_values)
    median_f_e2e = statistics.median(f_e2e_values)
    medians = {
        "external_wall_seconds": statistics.median(
            float(run["wall_seconds"]) for run in runs
        ),
        "stage_seconds": {
            stage: statistics.median(values)
            for stage, values in stage_replicates.items()
        },
        "f_pipeline": median_f_pipeline,
        "f_e2e": median_f_e2e,
    }

    fallback_rate = counts["msv_classic_fallback"] / counts["msv_ssv_attempt"]
    pass_rates = {
        name: counts[metric] / comparisons for name, metric in PASS_METRICS.items()
    }

    def projections(fraction: float) -> dict[str, float | None]:
        return {
            "10x": amdahl_speedup(fraction, 10.0),
            "100x": amdahl_speedup(fraction, 100.0),
            "infinite": amdahl_speedup(fraction, None),
        }

    hosts = sorted(
        {
            str(run["record"].get("host", {}).get("hostname"))  # type: ignore[union-attr]
            for run in runs
        }
    )
    return {
        "schema_version": 1,
        "role": "reference_oracle_stage_timing",
        "dataset_id": next(iter(dataset_ids)),
        "reference_engine": {
            "name": "pristine HMMER",
            "version": "3.4",
            "role": "semantic_and_stage_timing_oracle",
        },
        "application_baseline": {
            "name": "Astra",
            "role": "application_baseline",
            "summary": display_path(astra_summary),
            "summary_sha256": sha256_file(astra_summary),
        },
        "immutable_provenance": {
            "timing_patch": {
                "path": display_path(timing_patch),
                "sha256": sha256_file(timing_patch),
            },
            "timed_executable_from_run_json": {
                "path": display_path(Path(executable_path)),
                "sha256": executable_sha256,
            },
            "query_hmm": {
                "path": display_path(query_hmm),
                "sha256": sha256_file(query_hmm),
            },
            "target_fasta": {
                "path": display_path(target_fasta),
                "sha256": sha256_file(target_fasta),
            },
        },
        "measurement": {
            "run_status": "pilot",
            "hosts": hosts,
            "clock": next(iter(clocks)),
            "replicates": len(runs),
            "cpu_workers": 0,
            "accelerated_stage": "msv_public",
            "f_pipeline_definition": "msv_public / pipeline_total",
            "f_e2e_definition": "msv_public / external process wall time",
            "timing_semantics": "Serial elapsed time; nested stages are not additive.",
            "observer_overhead": {
                "calibration_applied": False,
                "pilot_observed_fraction_range": [0.03, 0.05],
                "implication": (
                    "Instrumented f_e2e is diagnostic and likely biased high; do not "
                    "treat it as a calibrated production fraction."
                ),
            },
        },
        "replicate_measurements": replicate_output,
        "medians": medians,
        "counts_per_replicate": {
            "queries": int(runs[0]["timing"]["query_count"]),  # type: ignore[index]
            "target_comparisons": comparisons,
            "target_residues": counts["target_residues"],
            "clock_errors": counts["clock_errors"],
            "ssv_attempts": counts["msv_ssv_attempt"],
            "classic_fallbacks": counts["msv_classic_fallback"],
            "ssv_status": {
                status: counts[f"msv_ssv_status_{status}"]
                for status in ("ok", "erange", "noresult", "other")
            },
            "fallback_status": {
                status: counts[f"msv_fallback_status_{status}"]
                for status in ("ok", "erange", "other")
            },
            "passed": {name: counts[metric] for name, metric in PASS_METRICS.items()},
        },
        "rates": {
            "fallback": {
                "definition": "msv_classic_fallback / msv_ssv_attempt",
                "per_replicate": [fallback_rate] * len(runs),
                "median": fallback_rate,
            },
            "passes": {
                name: {
                    "definition": f"{PASS_METRICS[name]} / target_sequences",
                    "per_replicate": [rate] * len(runs),
                    "median": rate,
                }
                for name, rate in pass_rates.items()
            },
        },
        "amdahl_speedup_if_msv_accelerated": {
            "formula": "1 / ((1 - f) + f / acceleration)",
            "using_median_f_pipeline": projections(median_f_pipeline),
            "using_median_f_e2e": projections(median_f_e2e),
        },
        "scientific_output": {
            "normalization": "Exclude comment/blank lines; strip trailing whitespace.",
            "required_equal": True,
            "all_equal_to_pristine": True,
            "pristine": {
                name: {"path": display_path(pristine_paths[name]), **digest}
                for name, digest in reference_tables.items()
            },
            "timed_replicates": timed_tables,
        },
        "input_provenance": [
            {
                "label": run["label"],
                "run_json": display_path(run["run_json"]),  # type: ignore[arg-type]
                "run_json_sha256": sha256_file(run["run_json"]),  # type: ignore[arg-type]
                "timing_tsv": display_path(run["timing_tsv"]),  # type: ignore[arg-type]
                "timing_tsv_sha256": sha256_file(run["timing_tsv"]),  # type: ignore[arg-type]
            }
            for run in runs
        ],
        "warning": (
            "Pilot/reference timing from a shared-node allocation; not reportable "
            "performance data. The timing observer added roughly 3-5% in pilot checks, "
            "so f_e2e is diagnostic and likely biased high. Astra, not pristine HMMER, "
            "is the application baseline."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        required=True,
        type=Path,
        metavar=("RUN_JSON", "TIMING_TSV"),
        help="repeat once for each serial replicate",
    )
    parser.add_argument(
        "--pristine-tblout",
        "--reference-tblout",
        dest="pristine_tblout",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--pristine-domtblout",
        "--reference-domtblout",
        dest="pristine_domtblout",
        required=True,
        type=Path,
    )
    parser.add_argument("--astra-summary", type=Path, default=DEFAULT_ASTRA_SUMMARY)
    parser.add_argument("--timing-patch", type=Path, default=DEFAULT_TIMING_PATCH)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = summarize(
            [(pair[0], pair[1]) for pair in args.run],
            args.pristine_tblout,
            args.pristine_domtblout,
            args.astra_summary,
            args.timing_patch,
        )
        atomic_json(args.output, output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
