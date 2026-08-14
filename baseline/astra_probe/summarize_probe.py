#!/usr/bin/env python3
"""Strictly summarize serial control and observed Astra stage-probe runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Sequence, TypedDict

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baseline.astra_probe.validate_probe import (  # noqa: E402
    EXPECTED_STAGES,
    STATUS_COLUMNS,
    parse_probe,
)


PRESSED_HMM_SUFFIXES = ("h3f", "h3i", "h3m", "h3p")


class SummaryError(ValueError):
    """An input set cannot support a strict probe summary."""


class Artifact(TypedDict):
    """A content-addressed regular file."""

    path: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> Artifact:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SummaryError(f"not a regular file: {path}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_compact_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_record(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot read benchmark record {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryError(f"benchmark record is not an object: {path}")
    return value


def _option_values(command: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == option:
            if index + 1 == len(command):
                raise SummaryError(f"{option} has no value")
            values.append(command[index + 1])
            index += 2
            continue
        prefix = option + "="
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def _canonical_command(command: Sequence[str]) -> tuple[str, ...]:
    canonical = list(command)
    outdirs = _option_values(command, "--outdir")
    if len(outdirs) != 1:
        raise SummaryError("each Astra command must have exactly one --outdir")
    for index, token in enumerate(canonical):
        if token == "--outdir":
            canonical[index + 1] = "<outdir>"
        elif token.startswith("--outdir="):
            canonical[index] = "--outdir=<outdir>"
    return tuple(canonical)


def _record_path(
    record: dict[str, object], value: str, *, strict: bool = False
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        cwd = record.get("cwd")
        if not isinstance(cwd, str) or not Path(cwd).is_absolute():
            raise SummaryError(f"relative path {value!r} lacks an absolute run cwd")
        path = Path(cwd) / path
    return path.resolve(strict=strict)


def _validate_probe_binding(
    path: Path,
    record: dict[str, object],
    *,
    expected_role: str,
    probe_tsv: Path | None,
    probe_binary: Path | None,
) -> None:
    metadata = record["metadata"]
    assert isinstance(metadata, dict)
    if metadata.get("role") != expected_role:
        raise SummaryError(f"unexpected run role in {path}: {metadata.get('role')!r}")
    if metadata.get("run_kind") != "replicate":
        raise SummaryError(f"run is not a measured replicate: {path}")

    overrides = record.get("environment_overrides")
    if not isinstance(overrides, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in overrides.items()
    ):
        raise SummaryError(f"missing or malformed environment overrides: {path}")

    if probe_tsv is None:
        environment = record.get("environment")
        if not isinstance(environment, dict):
            raise SummaryError(f"missing or malformed recorded environment: {path}")
        if any(
            isinstance(values.get("PLAN7_ASTRA_STAGE_PROBE"), str)
            and bool(values["PLAN7_ASTRA_STAGE_PROBE"])
            for values in (overrides, environment)
        ):
            raise SummaryError(f"control run enables the Astra stage probe: {path}")
        if probe_binary is not None:
            expected_binary = probe_binary.resolve(strict=True)
            for values in (overrides, environment):
                preload = values.get("LD_PRELOAD")
                if not isinstance(preload, str):
                    continue
                for entry in re.split(r"[\s:]+", preload):
                    if not entry or "/" not in entry:
                        continue
                    try:
                        candidate = _record_path(record, entry, strict=True)
                    except OSError:
                        continue
                    if candidate == expected_binary:
                        raise SummaryError(
                            f"control run preloads the Astra stage probe: {path}"
                        )
        return
    if probe_binary is None:
        raise AssertionError("an observed probe TSV requires a probe binary")

    report_value = overrides.get("PLAN7_ASTRA_STAGE_PROBE")
    if not isinstance(report_value, str):
        raise SummaryError(f"observed run does not set the probe report path: {path}")
    try:
        recorded_report = _record_path(record, report_value, strict=True)
        supplied_report = probe_tsv.resolve(strict=True)
    except OSError as error:
        raise SummaryError(f"probe report path is unavailable for {path}") from error
    if recorded_report != supplied_report:
        raise SummaryError(f"observed run and supplied probe TSV differ: {path}")

    preload = overrides.get("LD_PRELOAD")
    if not isinstance(preload, str) or not preload:
        raise SummaryError(f"observed run does not set LD_PRELOAD: {path}")
    expected_binary = probe_binary.resolve(strict=True)
    matching_entries = 0
    for entry in re.split(r"[\s:]+", preload):
        if not entry or "/" not in entry:
            continue
        try:
            candidate = _record_path(record, entry, strict=True)
        except OSError:
            continue
        if candidate == expected_binary:
            matching_entries += 1
    if matching_entries != 1:
        raise SummaryError(
            f"observed LD_PRELOAD does not name the supplied probe exactly once: {path}"
        )


def _default_astra_config() -> Path:
    config_root = os.environ.get("XDG_CONFIG_HOME")
    if config_root:
        return Path(config_root) / "Astra" / "hmm_databases.json"
    return Path.home() / ".config" / "Astra" / "hmm_databases.json"


def _hmm_runtime_provenance(
    command: Sequence[str],
    record: dict[str, object],
    hmm_artifact: Artifact,
    astra_config: Path | None,
) -> dict[str, object]:
    hmm_values = _option_values(command, "--hmm_in")
    installed_values = _option_values(command, "--installed_hmms")
    if hmm_values:
        return {
            "selection": "hmm_in",
            "runtime_format": "text_hmm",
            "runtime_artifacts": [hmm_artifact],
        }

    if len(installed_values) != 1 or not installed_values[0]:
        raise SummaryError("installed Astra command must select exactly one database")
    database_name = installed_values[0]
    if "," in database_name:
        raise SummaryError(
            "paired probe summaries require one installed Astra database"
        )

    config_path = astra_config or _default_astra_config()
    config_artifact = artifact(config_path)
    try:
        config = json.loads(Path(config_artifact["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryError(
            f"cannot read Astra database config {config_path}: {error}"
        ) from error
    entries = config.get("db_urls") if isinstance(config, dict) else None
    if not isinstance(entries, list):
        raise SummaryError("Astra database config has no db_urls list")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("name") == database_name
    ]
    if len(matches) != 1:
        raise SummaryError(
            f"Astra config must contain exactly one database named {database_name!r}"
        )
    configured_value = matches[0].get("installation_dir")
    if not isinstance(configured_value, str) or not configured_value:
        raise SummaryError(f"Astra database {database_name!r} has no installation_dir")
    configured_directory = Path(configured_value)
    configured_basename = configured_directory.name
    if not configured_directory.is_absolute():
        configured_directory = _record_path(record, configured_value)
    try:
        installation_directory = configured_directory.resolve(strict=True)
    except OSError as error:
        raise SummaryError(
            f"Astra database directory is unavailable: {configured_value}"
        ) from error
    if not installation_directory.is_dir():
        raise SummaryError(
            f"Astra installation_dir is not a directory: {configured_value}"
        )

    source_paths = sorted(
        (
            child.resolve(strict=True)
            for child in installation_directory.iterdir()
            if child.is_file() and child.name.endswith((".hmm", ".HMM"))
        ),
        key=str,
    )
    if not source_paths:
        raise SummaryError(
            f"installed Astra database has no HMM sources: {database_name}"
        )
    supplied_hmm = Path(hmm_artifact["path"])
    if supplied_hmm not in source_paths:
        raise SummaryError(
            f"supplied HMM is not a source in installed Astra database {database_name!r}"
        )

    # Mirror Astra search._find_pressed_db(): the pressed basename is the
    # installation directory's basename, and all four files must exist.
    pressed_base = installation_directory / configured_basename
    pressed_paths = [
        Path(f"{pressed_base}.{suffix}") for suffix in PRESSED_HMM_SUFFIXES
    ]
    if all(path.exists() for path in pressed_paths):
        runtime_format = "pressed_hmm"
        runtime_paths = pressed_paths
    else:
        runtime_format = "text_hmm"
        runtime_paths = source_paths

    source_artifacts = [
        hmm_artifact if path == supplied_hmm else artifact(path)
        for path in source_paths
    ]
    source_artifacts_by_path = {Path(item["path"]): item for item in source_artifacts}

    return {
        "selection": "installed_hmms",
        "installed_database": database_name,
        "astra_config": config_artifact,
        "installation_directory": str(installation_directory),
        "source_artifacts": source_artifacts,
        "runtime_format": runtime_format,
        "runtime_artifacts": [
            source_artifacts_by_path.get(path) or artifact(path)
            for path in runtime_paths
        ],
    }


def _validate_record(
    path: Path,
    record: dict[str, object],
    fasta: Path,
    hmm: Path,
    *,
    expected_role: str,
    probe_tsv: Path | None = None,
    probe_binary: Path | None = None,
) -> tuple[float, tuple[str, ...], str, int]:
    if record.get("schema_version") != 1:
        raise SummaryError(f"unsupported run schema in {path}")
    if record.get("exit_code") != 0:
        raise SummaryError(f"Astra command failed in {path}")

    command = record.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) for token in command)
    ):
        raise SummaryError(f"invalid command in {path}")
    if Path(command[0]).name != "astra" or command[1:2] != ["search"]:
        raise SummaryError(f"record is not an existing-Astra search command: {path}")

    threads = _option_values(command, "--threads")
    if threads != ["1"]:
        raise SummaryError(f"Astra command is not serial (--threads 1): {path}")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("engine") != "astra":
        raise SummaryError(f"run is not marked as the Astra engine: {path}")
    if "threads" in metadata and metadata["threads"] != 1:
        raise SummaryError(f"run metadata is not serial: {path}")
    if metadata.get("run_status") != "pilot":
        raise SummaryError(f"run is not explicitly marked as a pilot: {path}")
    _validate_probe_binding(
        path,
        record,
        expected_role=expected_role,
        probe_tsv=probe_tsv,
        probe_binary=probe_binary,
    )
    dataset_id = metadata.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise SummaryError(f"run has no dataset_id: {path}")
    run_index = metadata.get("run_index")
    if isinstance(run_index, bool) or not isinstance(run_index, int) or run_index < 1:
        raise SummaryError(f"run has no positive integer run_index: {path}")

    fasta_values = _option_values(command, "--prot_in")
    if (
        len(fasta_values) != 1
        or _record_path(record, fasta_values[0]) != fasta.resolve()
    ):
        raise SummaryError(f"Astra command does not use the supplied FASTA: {path}")
    hmm_values = _option_values(command, "--hmm_in")
    installed_values = _option_values(command, "--installed_hmms")
    if bool(hmm_values) == bool(installed_values):
        raise SummaryError(f"Astra command must select one HMM source: {path}")
    if hmm_values and (
        len(hmm_values) != 1 or _record_path(record, hmm_values[0]) != hmm.resolve()
    ):
        raise SummaryError(f"Astra command does not use the supplied HMM file: {path}")
    if installed_values and (
        len(installed_values) != 1
        or not installed_values[0]
        or "," in installed_values[0]
    ):
        raise SummaryError(f"Astra command must select one installed database: {path}")

    executable = record.get("executable")
    if not isinstance(executable, dict):
        raise SummaryError(f"missing Astra executable provenance: {path}")
    executable_path = executable.get("path")
    executable_sha = executable.get("sha256")
    if not isinstance(executable_path, str) or Path(executable_path).name != "astra":
        raise SummaryError(f"invalid Astra executable path: {path}")
    try:
        recorded_executable = Path(executable_path).resolve(strict=True)
        current_sha = sha256_file(recorded_executable)
    except OSError as error:
        raise SummaryError(
            f"recorded Astra executable is unavailable: {path}"
        ) from error
    if executable_sha != current_sha:
        raise SummaryError(f"recorded Astra executable SHA is stale: {path}")
    command_executable = Path(command[0])
    if command_executable.is_absolute():
        if command_executable.resolve() != recorded_executable:
            raise SummaryError(f"command and recorded Astra executable differ: {path}")
    elif "/" in command[0]:
        cwd = record.get("cwd")
        if not isinstance(cwd, str):
            raise SummaryError(f"relative Astra command lacks a recorded cwd: {path}")
        if (Path(cwd) / command_executable).resolve() != recorded_executable:
            raise SummaryError(f"command and recorded Astra executable differ: {path}")

    timing = record.get("timing")
    wall = timing.get("wall_seconds") if isinstance(timing, dict) else None
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        raise SummaryError(f"missing wall time in {path}")
    wall = float(wall)
    if not math.isfinite(wall) or wall <= 0:
        raise SummaryError(f"invalid wall time in {path}")

    return wall, _canonical_command(command), dataset_id, run_index


def _identity(
    records: Sequence[dict[str, object]],
    keys: Sequence[str],
    supplied: str | None,
    label: str,
) -> str:
    observed: set[str] = set()
    for record in records:
        metadata = record["metadata"]
        assert isinstance(metadata, dict)
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                observed.add(str(value))
                break
    if len(observed) > 1:
        raise SummaryError(f"inconsistent {label} metadata")
    if supplied is not None:
        if observed and observed != {supplied}:
            raise SummaryError(f"supplied {label} disagrees with run metadata")
        return supplied
    if not observed:
        raise SummaryError(f"{label} is absent; supply it explicitly")
    return next(iter(observed))


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "replicates": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _amdahl(fraction: float, acceleration: float | None) -> float:
    accelerated = 0.0 if acceleration is None else fraction / acceleration
    denominator = (1.0 - fraction) + accelerated
    return math.inf if denominator == 0.0 else 1.0 / denominator


def build_summary(
    observed_triplets: Sequence[tuple[Path, Path, Path]],
    control_pairs: Sequence[tuple[Path, Path]],
    *,
    fasta: Path,
    hmm: Path,
    probe_source: Path,
    probe_binary: Path,
    astra_config: Path | None = None,
    astra_version: str | None = None,
    pyhmmer_version: str | None = None,
    astra_revision: str | None = None,
) -> dict[str, object]:
    if not observed_triplets:
        raise SummaryError("at least one observed triplet is required")
    if not control_pairs:
        raise SummaryError("at least one control pair is required")

    fasta_artifact = artifact(fasta)
    hmm_artifact = artifact(hmm)
    source_artifact = artifact(probe_source)
    binary_artifact = artifact(probe_binary)

    observed_records = [_load_record(run) for run, _, _ in observed_triplets]
    control_records = [_load_record(run) for run, _ in control_pairs]
    all_records = [*control_records, *observed_records]

    control_walls: list[float] = []
    observed_walls: list[float] = []
    canonical_commands: set[tuple[str, ...]] = set()
    dataset_ids: set[str] = set()
    control_indices: set[int] = set()
    observed_indices: set[int] = set()
    for (run_path, _), record in zip(control_pairs, control_records, strict=True):
        wall, command, dataset_id, run_index = _validate_record(
            run_path,
            record,
            fasta,
            hmm,
            expected_role="astra_uninstrumented_control",
            probe_binary=probe_binary,
        )
        if run_index in control_indices:
            raise SummaryError("duplicate control run_index")
        control_walls.append(wall)
        canonical_commands.add(command)
        dataset_ids.add(dataset_id)
        control_indices.add(run_index)
    for (run_path, probe_tsv, _), record in zip(
        observed_triplets, observed_records, strict=True
    ):
        wall, command, dataset_id, run_index = _validate_record(
            run_path,
            record,
            fasta,
            hmm,
            expected_role="astra_in_process_stage_probe",
            probe_tsv=probe_tsv,
            probe_binary=probe_binary,
        )
        if run_index in observed_indices:
            raise SummaryError("duplicate observed run_index")
        observed_walls.append(wall)
        canonical_commands.add(command)
        dataset_ids.add(dataset_id)
        observed_indices.add(run_index)
    if len(canonical_commands) != 1:
        raise SummaryError("Astra commands differ by more than --outdir")
    if len(dataset_ids) != 1:
        raise SummaryError("Astra runs use different dataset_id values")
    if control_indices != observed_indices:
        raise SummaryError("control and observed replicate indices differ")
    canonical_command = list(next(iter(canonical_commands)))
    hmm_runtime = _hmm_runtime_provenance(
        canonical_command, all_records[0], hmm_artifact, astra_config
    )

    executable_paths: set[str] = set()
    executable_shas: set[str] = set()
    for record in all_records:
        executable = record.get("executable")
        if not isinstance(executable, dict):
            raise SummaryError("Astra executable provenance disappeared")
        executable_path = executable.get("path")
        executable_sha = executable.get("sha256")
        if not isinstance(executable_path, str) or not isinstance(executable_sha, str):
            raise SummaryError("Astra executable provenance is malformed")
        executable_paths.add(str(Path(executable_path).resolve()))
        executable_shas.add(executable_sha)
    if len(executable_paths) != 1 or len(executable_shas) != 1:
        raise SummaryError("Astra executable provenance differs across runs")

    astra_version = _identity(
        all_records, ("astra_version",), astra_version, "Astra version"
    )
    pyhmmer_version = _identity(
        all_records,
        ("pyhmmer_version", "pyhmmer"),
        pyhmmer_version,
        "PyHMMER version",
    )
    astra_revision = _identity(
        all_records, ("astra_revision",), astra_revision, "Astra revision"
    )

    hit_paths = [hits for _, hits in control_pairs] + [
        hits for _, _, hits in observed_triplets
    ]
    hit_artifacts = [artifact(path) for path in hit_paths]
    hit_shas = {str(item["sha256"]) for item in hit_artifacts}
    hit_sizes = {int(item["size_bytes"]) for item in hit_artifacts}
    if len(hit_shas) != 1 or len(hit_sizes) != 1:
        raise SummaryError("Astra scientific hit TSVs are not byte-identical")

    probe_metadata: list[dict[str, list[str]]] = []
    stage_runs: list[dict[str, dict[str, int]]] = []
    for _, probe_tsv, _ in observed_triplets:
        try:
            metadata, stages = parse_probe(probe_tsv)
        except (OSError, ValueError) as error:
            raise SummaryError(f"invalid probe TSV {probe_tsv}: {error}") from error
        probe_metadata.append(metadata)
        stage_runs.append(stages)

    count_keys = ("calls", *STATUS_COLUMNS)
    count_signatures = [
        {
            stage: {key: values[key] for key in count_keys}
            for stage, values in stages.items()
        }
        for stages in stage_runs
    ]
    if any(signature != count_signatures[0] for signature in count_signatures[1:]):
        raise SummaryError("probe call/status count sets differ across observed runs")
    count_signature = count_signatures[0]
    if count_signature["p7_Pipeline"]["calls"] == 0:
        raise SummaryError("probe observed no pipeline calls")

    target_paths: set[Path] = set()
    overhead_models: set[str] = set()
    for metadata in probe_metadata:
        target = metadata.get("target_library")
        overhead = metadata.get("observer_overhead")
        if target is None or len(target) != 1:
            raise SummaryError("probe TSV lacks one target library")
        if overhead is None or len(overhead) != 1:
            raise SummaryError("probe TSV lacks its observer-overhead label")
        target_paths.add(Path(target[0]).resolve(strict=True))
        overhead_models.add(overhead[0])
    if len(target_paths) != 1 or len(overhead_models) != 1:
        raise SummaryError("probe target or overhead model differs across runs")
    target_artifact = artifact(next(iter(target_paths)))

    inclusive_stage_seconds = {
        stage: statistics.median(
            [run[stage]["elapsed_ns"] / 1_000_000_000 for run in stage_runs]
        )
        for stage in EXPECTED_STAGES
    }
    msv_pipeline_fractions = [
        run["p7_MSVFilter"]["elapsed_ns"] / run["p7_Pipeline"]["elapsed_ns"]
        for run in stage_runs
    ]
    msv_wall_fractions = [
        run["p7_MSVFilter"]["elapsed_ns"] / 1_000_000_000 / wall
        for run, wall in zip(stage_runs, observed_walls, strict=True)
    ]
    if any(not 0.0 <= value <= 1.0 for value in msv_pipeline_fractions):
        raise SummaryError("invalid inclusive MSV/pipeline fraction")
    if any(not 0.0 <= value < 1.0 for value in msv_wall_fractions):
        raise SummaryError("invalid inclusive MSV/external-wall fraction")

    msv_pipeline_fraction = statistics.median(msv_pipeline_fractions)
    msv_wall_fraction = statistics.median(msv_wall_fractions)
    control_median = statistics.median(control_walls)
    observed_median = statistics.median(observed_walls)
    overhead_seconds = observed_median - control_median

    ssv = count_signature["p7_SSVFilter"]
    ssv_calls = ssv["calls"]
    fallback_calls = ssv["status_noresult"]

    host_values: set[str] = set()
    job_id_values: set[str] = set()
    for record in all_records:
        host = record.get("host")
        if isinstance(host, dict):
            hostname = host.get("hostname")
            if hostname is not None:
                host_values.add(str(hostname))
        environment = record.get("environment")
        if isinstance(environment, dict):
            job_id = environment.get("SLURM_JOB_ID")
            if job_id is not None:
                job_id_values.add(str(job_id))
    hosts = sorted(host_values)
    job_ids = sorted(job_id_values)

    return {
        "schema_version": 1,
        "run_status": "pilot",
        "dataset_id": next(iter(dataset_ids)),
        "warning": (
            "Shared-allocation pilot timings, including observer overhead, are "
            "not reportable performance results."
        ),
        "engine": {
            "name": "Astra",
            "version": astra_version,
            "revision": astra_revision,
            "pyhmmer_version": pyhmmer_version,
            "executable": {
                "path": next(iter(executable_paths)),
                "sha256": next(iter(executable_shas)),
            },
        },
        "workload": {
            "fasta": fasta_artifact,
            "hmm": hmm_artifact,
            "hmm_runtime": hmm_runtime,
            "command": canonical_command,
            "comparisons_per_run": count_signature["p7_Pipeline"]["calls"],
        },
        "scientific_output": {
            "byte_identical": True,
            "files_compared": len(hit_artifacts),
            "sha256": next(iter(hit_shas)),
            "size_bytes": next(iter(hit_sizes)),
        },
        "probe": {
            "source": source_artifact,
            "binary": binary_artifact,
            "target_library": target_artifact,
            "clock": "CLOCK_MONOTONIC",
            "aggregation": "process_wide_inclusive",
            "observer_overhead_model": next(iter(overhead_models)),
            "observed_replicates": len(observed_triplets),
            "stage_calls_per_run": {
                stage: values["calls"] for stage, values in count_signature.items()
            },
            "ssv": {
                "calls_per_run": ssv_calls,
                "statuses_per_run": {
                    key.removeprefix("status_"): ssv[key] for key in STATUS_COLUMNS
                },
                "full_msv_fallback_calls_per_run": fallback_calls,
                "full_msv_fallback_fraction": (
                    fallback_calls / ssv_calls if ssv_calls else 0.0
                ),
            },
        },
        "timing": {
            "external_wall_seconds": {
                "control": _distribution(control_walls),
                "observed": _distribution(observed_walls),
            },
            "observer_overhead_estimate": {
                "seconds": overhead_seconds,
                "fraction_of_control": overhead_seconds / control_median,
                "slowdown": observed_median / control_median,
            },
            "inclusive_stage_seconds_median": inclusive_stage_seconds,
            "fractions": {
                "aggregation": "median_of_per_run_ratios",
                "msv_of_pipeline": msv_pipeline_fraction,
                "msv_of_observed_external_wall": msv_wall_fraction,
            },
            "amdahl_end_to_end_speedup": {
                "basis": "median_per_run_msv_fraction_of_observed_external_wall",
                "10x": _amdahl(msv_wall_fraction, 10.0),
                "100x": _amdahl(msv_wall_fraction, 100.0),
                "infinite": _amdahl(msv_wall_fraction, None),
            },
        },
        "allocation": {
            "hosts": hosts,
            "slurm_job_ids": job_ids,
            "control_replicates": len(control_pairs),
            "observed_replicates": len(observed_triplets),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--observed",
        nargs=3,
        action="append",
        metavar=("RUN_JSON", "PROBE_TSV", "HIT_TSV"),
        required=True,
    )
    parser.add_argument(
        "--control",
        nargs=2,
        action="append",
        metavar=("RUN_JSON", "HIT_TSV"),
        required=True,
    )
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--hmm", type=Path, required=True)
    parser.add_argument(
        "--astra-config",
        type=Path,
        help=(
            "Astra hmm_databases.json used by --installed_hmms "
            "(defaults to the current user's Astra config)"
        ),
    )
    parser.add_argument(
        "--probe-source",
        type=Path,
        default=Path(__file__).with_name("astra_stage_probe.c"),
    )
    parser.add_argument("--probe-binary", type=Path, required=True)
    parser.add_argument("--astra-version")
    parser.add_argument("--pyhmmer-version")
    parser.add_argument("--astra-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        observed_triplets: list[tuple[Path, Path, Path]] = [
            (Path(item[0]), Path(item[1]), Path(item[2])) for item in args.observed
        ]
        control_pairs: list[tuple[Path, Path]] = [
            (Path(item[0]), Path(item[1])) for item in args.control
        ]
        summary = build_summary(
            observed_triplets,
            control_pairs,
            fasta=args.fasta,
            hmm=args.hmm,
            probe_source=args.probe_source,
            probe_binary=args.probe_binary,
            astra_config=args.astra_config,
            astra_version=args.astra_version,
            pyhmmer_version=args.pyhmmer_version,
            astra_revision=args.astra_revision,
        )
        atomic_compact_json(args.output, summary)
    except (OSError, SummaryError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
