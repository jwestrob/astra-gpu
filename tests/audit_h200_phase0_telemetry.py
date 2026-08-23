"""Focused H200 execution oracle for real Phase 0 telemetry integration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import stat
import sys
import tempfile
import time

if (
    os.environ.get("PYTHONDONTWRITEBYTECODE") != "1"
    or not sys.dont_write_bytecode
):
    raise RuntimeError("the H200 oracle requires PYTHONDONTWRITEBYTECODE=1")

import pyhmmer

from audit_h200_phase1a_sparse_v3 import (
    OPTIONS,
    SELECTION,
    fixture,
    git_record,
    pipeline,
    require_environment,
    sha256_file,
)
from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native, _pipeline, astra_search
from plan7_gpu._telemetry import GENERATION_TELEMETRY_SCHEMA_VERSION
from plan7_gpu.adapter import _sequence_state
from plan7_gpu.telemetry_report import REPORT_SCHEMA_VERSION, TelemetryCollector


PIPELINE_OPTIONS = {
    "E": 10.0,
    "domE": 10.0,
    "incE": 10.0,
    "incdomE": 10.0,
    "bias_filter": True,
    **OPTIONS,
}
EXPECTED_REPORT_FILES = {
    "artifact.sha256",
    "phase0-pareto.tsv",
    "phase0-profiles.tsv",
    "phase0-raw.json",
    "phase0-reason-summary.tsv",
    "phase0-reasons.tsv",
}
EXPECTED_NATIVE_SHA256 = (
    "8067fb336e3ef7412a0b01972177d94a867b01a8307c70396ad9b8bc6fd35b03"
)
EXPECTED_PIPELINE_SHA256 = (
    "1042c2ff578d8426b54f4eaa7a5d5d98cacf3d7e01bee3cc55cc0286499b8b3d"
)


def _exact_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise AssertionError(f"{field} is not an exact positive integer")
    return value


def _profile_key(pair: object) -> str:
    hmm = pair.hmm
    value = hmm.accession or hmm.name
    if type(value) is bytes:
        return value.decode("utf-8", "strict")
    if type(value) is str:
        return value
    raise AssertionError("fixture profile lacks a stable name or accession")


def _audit_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    if (
        snapshot["schema_version"] != REPORT_SCHEMA_VERSION
        or snapshot["scope"] != "phase0_route_telemetry_report"
        or snapshot["complete"] is not True
        or tuple(snapshot["expected_profile_ordinals"]) != SELECTION
        or snapshot["chunk_count"] != 1
        or snapshot["profile_count"] != len(SELECTION)
        or snapshot["expected_profile_count"] != len(SELECTION)
        or snapshot["missing_generation_profile_ordinals"] != ()
        or snapshot["missing_continuation_profile_ordinals"] != ()
    ):
        raise AssertionError("canonical Phase 0 report is incomplete or misbound")

    profiles = snapshot["profiles"]
    observed_ordinals = {row["profile_ordinal"] for row in profiles}
    if observed_ordinals != set(SELECTION):
        raise AssertionError("collector changed the global profile ordinals")

    stage_populations = {
        "postfilter": 0,
        "f2": 0,
        "forward": 0,
        "backward_domain": 0,
        "rescore": 0,
    }
    stage_cells = {stage: 0 for stage in stage_populations}
    positive_reason_rows = 0
    positive_reason_cells = 0
    continuation_routes = 0
    journal_matches = 0
    compact_attempts = 0
    compact_accepted = 0

    cell_names = {
        "postfilter": "postfilter_logical_cells",
        "f2": None,
        "forward": "forward_logical_cells",
        "backward_domain": "backward_logical_cells",
        "rescore": "rescore_logical_cells",
    }
    for profile in profiles:
        generation = profile["generation"]
        continuation = profile["continuation"]
        counts = generation["counts"]
        cells = generation["logical_cells"]
        populations = {
            "postfilter": counts["f1_candidate_count"],
            "f2": counts["f1_candidate_count"],
            "forward": counts["f2_pass_count"],
            "backward_domain": counts["forward_pass_count"],
            "rescore": counts["rescore_region_count"],
        }
        if continuation["path"] != "journal":
            raise AssertionError("real fused continuation did not use the journal")
        if continuation["batch_identity"] != profile["batch_identity"]:
            raise AssertionError("generation/continuation batch identity changed")
        if continuation["target_count"] != profile["target_count"]:
            raise AssertionError("generation/continuation target count changed")
        if (
            continuation["postfilter_record_count"]
            != counts["f1_candidate_count"]
        ):
            raise AssertionError("generation/continuation F1 census changed")
        if sum(continuation["routes"].values()) != profile["target_count"]:
            raise AssertionError("continuation routes do not partition targets")

        for stage, population in populations.items():
            stage_populations[stage] += population
            reasons = dict(generation["reason_counts"][stage])
            reason_cells = dict(generation["reason_logical_cells"][stage])
            if population and not any(value > 0 for value in reasons.values()):
                raise AssertionError(f"{stage} source facts are vacuous")
            positive_reason_rows += sum(reasons.values())
            positive_reason_cells += sum(reason_cells.values())
            cell_name = cell_names[stage]
            work_cells = 0 if cell_name is None else cells[cell_name]
            stage_cells[stage] += work_cells
            if work_cells and not any(value > 0 for value in reason_cells.values()):
                raise AssertionError(f"{stage} work lacks source attribution")

        continuation_routes += sum(continuation["source_routes"].values())
        journal_matches += continuation["journal"]["match_count"]
        compact_attempts += continuation["compact"]["attempt_count"]
        compact_accepted += continuation["compact"]["accepted_count"]

    missing_populations = [
        stage for stage, value in stage_populations.items() if value == 0
    ]
    # Full-MSV/Viterbi post-filter DP is conditional: a valid fixture can
    # traverse the source branches without executing either fallback kernel.
    # Forward, Backward/domain, and compact rescore are intentionally exercised
    # by this fixture and must carry nonzero source-attributed work.
    missing_cells = [
        stage
        for stage in ("forward", "backward_domain", "rescore")
        if stage_cells[stage] == 0
    ]
    if missing_populations or missing_cells:
        raise AssertionError(
            "fixture missed instrumented generation stages: "
            f"population={missing_populations}, cells={missing_cells}"
        )
    for field, value in (
        ("positive_reason_rows", positive_reason_rows),
        ("positive_reason_cells", positive_reason_cells),
        ("continuation_source_routes", continuation_routes),
        ("journal_matches", journal_matches),
        ("compact_attempts", compact_attempts),
        ("compact_accepted", compact_accepted),
        ("continuation_cpu_wall_ns", snapshot["continuation_cpu_wall_ns"]),
    ):
        _exact_positive_int(value, field)

    if len(snapshot["pareto"]) != len(SELECTION):
        raise AssertionError("CPU-wall Pareto omitted a profile")
    generation_totals = snapshot["chunks"][0]["generation"]["totals"]
    native_totals = snapshot["chunks"][0]["generation"]["native_totals"]
    _exact_positive_int(generation_totals["f1_logical_cells"], "F1 cells")
    for stage in ("forward", "backward_domain", "rescore"):
        _exact_positive_int(
            native_totals[stage]["work_cells"], f"native {stage} work cells"
        )
    return {
        "stage_populations": stage_populations,
        "stage_logical_cells": stage_cells,
        "positive_reason_rows": positive_reason_rows,
        "positive_reason_logical_cells": positive_reason_cells,
        "continuation_source_routes": continuation_routes,
        "journal_matches": journal_matches,
        "compact_attempts": compact_attempts,
        "compact_accepted": compact_accepted,
        "continuation_cpu_wall_ns": snapshot["continuation_cpu_wall_ns"],
        "generation_totals": generation_totals,
        "native_totals": native_totals,
    }


def _audit_report(directory: Path) -> dict[str, object]:
    if directory.is_symlink() or not directory.is_dir():
        raise AssertionError("canonical report is not a real directory")
    if stat.S_IMODE(directory.stat().st_mode) != 0o555:
        raise AssertionError("canonical report directory is not immutable")
    observed = {path.name for path in directory.iterdir()}
    if observed != EXPECTED_REPORT_FILES:
        raise AssertionError("canonical report membership changed")

    manifest_path = directory / "artifact.sha256"
    manifest: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if name in manifest or "/" in name or name == "artifact.sha256":
            raise AssertionError("canonical report manifest is invalid")
        manifest[name] = digest
    if set(manifest) != EXPECTED_REPORT_FILES - {"artifact.sha256"}:
        raise AssertionError("canonical report manifest membership changed")

    files: dict[str, dict[str, object]] = {}
    for name in sorted(EXPECTED_REPORT_FILES):
        path = directory / name
        mode = path.lstat().st_mode
        if path.is_symlink() or not stat.S_ISREG(mode):
            raise AssertionError(f"canonical report member is not regular: {name}")
        if stat.S_IMODE(mode) != 0o444:
            raise AssertionError(f"canonical report member is writable: {name}")
        digest = sha256_file(path)
        if name in manifest and digest != manifest[name]:
            raise AssertionError(f"canonical report hash changed: {name}")
        files[name] = {"bytes": path.stat().st_size, "sha256": digest}

    raw = json.loads((directory / "phase0-raw.json").read_text())
    if (
        raw.get("schema_version") != REPORT_SCHEMA_VERSION
        or raw.get("complete") is not True
        or raw.get("expected_profile_ordinals") != list(SELECTION)
        or raw.get("profile_count") != len(SELECTION)
    ):
        raise AssertionError("exported raw Phase 0 report changed")
    return {
        "path": str(directory),
        "artifact_manifest_sha256": files["artifact.sha256"]["sha256"],
        "files": files,
    }


def _atomic_json_noreplace(path: Path, value: object) -> None:
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("output parent must be an existing real directory")
    if os.path.lexists(path):
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            temporary.chmod(0o600)
            temporary.unlink()
        raise


def _audit_fixture(report_directory: Path) -> dict[str, object]:
    hmms, targets = fixture()
    alphabet = hmms[0].alphabet
    collector = TelemetryCollector()
    collector.bind_expected_profiles(SELECTION)
    with tempfile.TemporaryDirectory(prefix="phase0-h200-telemetry-") as temporary:
        base = Path(temporary) / "models"
        pyhmmer.hmmer.hmmpress(hmms, base)
        pairs = load_pressed_profiles(base)
        selected_pairs = tuple(pairs[index] for index in SELECTION)
        profile_keys = tuple(_profile_key(pair) for pair in selected_pairs)
        with (
            ProfileSession(pairs, pack_workers=1) as session,
            session.select(SELECTION) as selection,
            SequenceBatch(targets) as batch,
        ):
            candidates = batch._postfilter_forward_selection(
                selection,
                **OPTIONS,
                bias_filter=True,
                pipeline=pipeline(alphabet),
                telemetry=True,
            )
            generation = candidates.generation_statistics
            if (
                type(generation) is not dict
                or generation["schema_version"]
                != GENERATION_TELEMETRY_SCHEMA_VERSION
                or generation["profile_count"] != len(SELECTION)
                or generation["target_count"] != len(targets)
                or generation["batch_identity"] is None
            ):
                raise AssertionError("fused generation telemetry is absent or invalid")
            hits = list(
                astra_search.hmmsearch(
                    selected_pairs,
                    candidates,
                    cpus=1,
                    postfilter=True,
                    telemetry_collector=collector,
                    profile_ordinals=SELECTION,
                    profile_keys=profile_keys,
                    **PIPELINE_OPTIONS,
                )
            )
            if len(hits) != len(SELECTION) or any(
                type(item) is not pyhmmer.plan7.TopHits for item in hits
            ):
                raise AssertionError("telemetry changed the ordinary result shape")
            workspace = dict(_sequence_state(batch).native.workspace_statistics)

    if (
        workspace["postfilter_run_count"] <= 0
        or workspace["forward_run_count"] <= 0
    ):
        raise AssertionError("oracle did not execute real CUDA generation work")
    snapshot = collector.snapshot(require_complete=True)
    reconciliation = _audit_snapshot(snapshot)
    collector.export(report_directory)
    report = _audit_report(report_directory)
    return {
        "targets": len(targets),
        "profile_ordinals": SELECTION,
        "profile_keys": profile_keys,
        "workspace": workspace,
        "reconciliation": reconciliation,
        "report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = git_record(root)
    if source["dirty"] or source["revision"] != args.expected_revision:
        raise RuntimeError(
            "the H200 oracle requires the exact clean expected source revision"
        )
    report_directory = args.report_dir.absolute()
    output = args.output.absolute()
    if report_directory.parent.resolve(strict=True) != output.parent.resolve(
        strict=True
    ):
        raise ValueError("canonical report and summary must be sibling artifacts")
    if os.path.lexists(report_directory) or os.path.lexists(output):
        raise FileExistsError("oracle output already exists")

    native_path = Path(_native.__file__).resolve()
    pipeline_path = Path(_pipeline.__file__).resolve()
    native_sha256 = sha256_file(native_path)
    pipeline_sha256 = sha256_file(pipeline_path)
    if (
        native_sha256 != EXPECTED_NATIVE_SHA256
        or pipeline_sha256 != EXPECTED_PIPELINE_SHA256
    ):
        raise RuntimeError("the H200 oracle loaded an unexpected native artifact")
    environment = require_environment()
    started = time.monotonic()
    result = _audit_fixture(report_directory)
    record = {
        "schema_version": 1,
        "verdict": "PASS",
        "scope": "real fused CUDA Phase 0 generation/continuation telemetry",
        "source": {
            **source,
            "native_extension": {
                "path": str(native_path),
                "sha256": native_sha256,
            },
            "pipeline_extension": {
                "path": str(pipeline_path),
                "sha256": pipeline_sha256,
            },
        },
        "host": {
            "hostname": platform.node(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
        },
        "environment": environment,
        "fixture": {
            "selection": SELECTION,
            "pipeline_options": PIPELINE_OPTIONS,
        },
        "result": result,
        "wall_seconds": time.monotonic() - started,
    }
    _atomic_json_noreplace(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
