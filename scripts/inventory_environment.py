#!/usr/bin/env python3
"""Capture a reproducible, secret-free hardware/software inventory."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(argv: list[str], timeout: int = 30) -> dict[str, object]:
    """Run one inventory command without invoking a shell."""
    try:
        proc = subprocess.run(
            argv,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.rstrip(),
            "stderr": proc.stderr.rstrip(),
        }
    except FileNotFoundError as exc:
        return {"argv": argv, "exit_code": 127, "stdout": "", "stderr": str(exc)}
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "exit_code": 124,
            "stdout": (exc.stdout or "").rstrip(),
            "stderr": f"timed out after {timeout}s",
        }


def parse_csv_rows(command: dict[str, object], fields: list[str]) -> list[dict[str, str]]:
    if command["exit_code"] != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in str(command["stdout"]).splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values, strict=True)))
    return rows


def sysfs_numa() -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    for node_dir in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        entry = {"node": node_dir.name}
        for name in ("cpulist", "distance"):
            path = node_dir / name
            if path.exists():
                entry[name] = path.read_text(encoding="utf-8").strip()
        meminfo = node_dir / "meminfo"
        if meminfo.exists():
            entry["meminfo"] = meminfo.read_text(encoding="utf-8").strip()
        nodes.append(entry)
    return nodes


def first_line(command: dict[str, object]) -> str | None:
    output = str(command["stdout"])
    return output.splitlines()[0] if output else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "hardware.json",
    )
    args = parser.parse_args()

    gpu_query = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    lscpu = run(["lscpu", "--json"])
    try:
        cpu_structured = json.loads(str(lscpu["stdout"])) if lscpu["exit_code"] == 0 else None
    except json.JSONDecodeError:
        cpu_structured = None

    gpu_nodes = run(
        ["sinfo", "--Node", "--noheader", "--partition=gpu,gpu_h200", "--format=%N"]
    )
    node_names = sorted({name for name in str(gpu_nodes["stdout"]).splitlines() if name})

    software_commands = {
        "gcc": run(["gcc", "--version"]),
        "g++": run(["g++", "--version"]),
        "cmake": run(["cmake", "--version"]),
        "make": run(["make", "--version"]),
        "python": run(["python3", "--version"]),
        "uv": run(["uv", "--version"]),
        "slurm": run(["scontrol", "--version"]),
        "nsys": run(["nsys", "--version"]),
        "ncu": run(["ncu", "--version"]),
    }

    inventory = {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
        },
        "project_root": str(PROJECT_ROOT),
        "gpu": {
            "devices": parse_csv_rows(
                gpu_query,
                [
                    "index",
                    "name",
                    "uuid",
                    "compute_capability",
                    "memory_mib",
                    "driver_version",
                    "pci_bus_id",
                ],
            ),
            "query": gpu_query,
            "topology": run(["nvidia-smi", "topo", "-m"]),
        },
        "cuda": {"nvcc": run(["nvcc", "--version"])},
        "cpu": {"lscpu": cpu_structured, "command": lscpu},
        "numa": {
            "numactl": run(["numactl", "--hardware"]),
            "sysfs": sysfs_numa(),
        },
        "slurm": {
            "partitions": run(["sinfo", "--format=%P|%a|%l|%D|%G"]),
            "configuration": run(["scontrol", "show", "partition"]),
            "gpu_nodes": {
                name: run(["scontrol", "show", "node", name]) for name in node_names
            },
        },
        "software": {
            name: {"version": first_line(command), "command": command}
            for name, command in software_commands.items()
        },
        "process": {
            "python_executable": sys.executable,
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
