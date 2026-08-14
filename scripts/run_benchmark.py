#!/usr/bin/env python3
"""Run exactly one command and write an atomic provenance/timing record."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECORDED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "LD_PRELOAD",
    "OMP_NUM_THREADS",
    "PLAN7_ASTRA_STAGE_PROBE",
    "SLURM_CLUSTER_NAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_MEM_BIND",
    "SLURM_PROCID",
    "SLURM_TASKS_PER_NODE",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path, hash_file: bool) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path)}
    if not path.exists() or not path.is_file():
        result["exists"] = False
        return result
    stat = path.stat()
    result.update({"exists": True, "size_bytes": stat.st_size})
    if hash_file:
        result["sha256"] = sha256_file(path)
    return result


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def environment_override(value: str) -> tuple[str, str]:
    name, separator, setting = value.partition("=")
    if not separator or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise argparse.ArgumentTypeError("environment overrides must be NAME=VALUE")
    return name, setting


def resolve_executable(command: str, environment: dict[str, str]) -> Path | None:
    candidate = Path(command)
    if "/" in command:
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate.resolve() if candidate.is_file() else None
    resolved = shutil.which(command, path=environment.get("PATH"))
    return Path(resolved).resolve() if resolved else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one benchmark command; pass the command after --."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--label", required=True)
    metadata_group = parser.add_mutually_exclusive_group()
    metadata_group.add_argument("--metadata", type=Path)
    metadata_group.add_argument("--metadata-json")
    parser.add_argument("--hash-outputs", action="store_true")
    parser.add_argument(
        "--set-env",
        action="append",
        type=environment_override,
        default=[],
        metavar="NAME=VALUE",
        help="set and record an environment variable for the child command only",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    override_names = [name for name, _ in args.set_env]
    if len(override_names) != len(set(override_names)):
        parser.error("an environment variable may only be overridden once")
    child_environment = os.environ.copy()
    child_environment.update(args.set_env)

    metadata: object = {}
    if args.metadata_json:
        metadata = json.loads(args.metadata_json)
    elif args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))

    args.stdout.parent.mkdir(parents=True, exist_ok=True)
    args.stderr.parent.mkdir(parents=True, exist_ok=True)
    executable_path = resolve_executable(command[0], child_environment)
    executable_record = (
        {
            "path": str(executable_path),
            "sha256": sha256_file(executable_path),
        }
        if executable_path and executable_path.is_file()
        else {"path": None, "sha256": None}
    )

    started_wall = dt.datetime.now(dt.timezone.utc)
    started_monotonic = time.perf_counter_ns()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (
        args.stdout.open("wb") as stdout_handle,
        args.stderr.open("wb") as stderr_handle,
    ):
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=child_environment,
        )
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    ended_monotonic = time.perf_counter_ns()
    ended_wall = dt.datetime.now(dt.timezone.utc)

    record = {
        "schema_version": 1,
        "label": args.label,
        "command": command,
        "cwd": str(PROJECT_ROOT),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
        },
        "environment": {
            name: child_environment[name]
            for name in RECORDED_ENV
            if name in child_environment
        },
        "environment_overrides": dict(args.set_env),
        "executable": executable_record,
        "timing": {
            "started_utc": started_wall.isoformat(),
            "ended_utc": ended_wall.isoformat(),
            "wall_seconds": (ended_monotonic - started_monotonic) / 1_000_000_000,
            "user_seconds": usage_after.ru_utime - usage_before.ru_utime,
            "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
            "max_rss_kib": usage_after.ru_maxrss,
        },
        "exit_code": process.returncode,
        "outputs": {
            "stdout": artifact(args.stdout, args.hash_outputs),
            "stderr": artifact(args.stderr, args.hash_outputs),
        },
        "metadata": metadata,
    }
    atomic_json(args.output, record)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
