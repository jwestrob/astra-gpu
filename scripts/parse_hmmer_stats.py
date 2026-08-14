#!/usr/bin/env python3
"""Parse HMMER 3.4 internal pipeline summaries into strict JSON."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path


START = "Internal pipeline statistics summary:"
COUNT_RE = re.compile(r"^(Query model\(s\)|Query sequence\(s\)|Target sequences|Target model\(s\)):\s+(\d+)\s+\(([^)]*)\)")
PASS_RE = re.compile(
    r"^Passed (MSV|bias|Vit|Fwd) filter:\s+(\d+)\s+\(([^)]*)\); expected\s+([0-9.eE+-]+)\s+\(([^)]*)\)"
)
SPACE_RE = re.compile(
    r"^(Initial search space \(Z\)|Domain search space\s+\(domZ\)):\s+([0-9.eE+-]+)\s+\[([^]]*)\]"
)
CPU_RE = re.compile(
    r"^# CPU time:\s+([0-9.]+)u\s+([0-9.]+)s\s+(\S+)\s+Elapsed:\s+(\S+)"
)
MCSEC_RE = re.compile(r"^# Mc/sec:\s+([0-9.eE+-]+)")

COUNT_KEYS = {
    "Query model(s)": ("query_models", "nodes"),
    "Query sequence(s)": ("query_sequences", "residues_searched"),
    "Target sequences": ("target_sequences", "residues_searched"),
    "Target model(s)": ("target_models", "nodes"),
}
STAGE_KEYS = {"MSV": "msv", "bias": "bias", "Vit": "viterbi", "Fwd": "forward"}


def number(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def parse(text: str) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == START:
            current = {"inputs": {}, "stages": {}, "search_space": {}}
            summaries.append(current)
            continue
        if current is None:
            continue
        if line == "//":
            current = None
            continue

        if match := COUNT_RE.match(line):
            key, detail_key = COUNT_KEYS[match.group(1)]
            detail_text = match.group(3)
            detail_match = re.match(r"([0-9]+)\s+", detail_text)
            value: dict[str, object] = {"count": int(match.group(2)), "detail": detail_text}
            if detail_match:
                value[detail_key] = int(detail_match.group(1))
            inputs = current["inputs"]
            assert isinstance(inputs, dict)
            inputs[key] = value
            continue

        if match := PASS_RE.match(line):
            stages = current["stages"]
            assert isinstance(stages, dict)
            stages[STAGE_KEYS[match.group(1)]] = {
                "passed": int(match.group(2)),
                "fraction": float(match.group(3)),
                "expected": number(match.group(4)),
                "expected_fraction": float(match.group(5)),
            }
            continue

        if match := SPACE_RE.match(line):
            key = "Z" if match.group(1).startswith("Initial") else "domZ"
            spaces = current["search_space"]
            assert isinstance(spaces, dict)
            spaces[key] = {"value": number(match.group(2)), "set_by": match.group(3)}
            continue

        if match := CPU_RE.match(line):
            current["reported_timing"] = {
                "user_seconds": float(match.group(1)),
                "system_seconds": float(match.group(2)),
                "cpu_hms": match.group(3),
                "elapsed_hms": match.group(4),
            }
            continue

        if match := MCSEC_RE.match(line):
            current["mc_per_second"] = float(match.group(1))

    return summaries


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
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summaries = parse(args.input.read_text(encoding="utf-8", errors="replace"))
    if not summaries:
        parser.error(f"no HMMER pipeline summaries found in {args.input}")
    atomic_json(args.output, {"schema_version": 1, "summaries": summaries})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
