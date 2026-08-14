#!/usr/bin/env python3
"""Run a reproducible pristine-HMMER CPU scaling matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_benchmark.py"
PARSER = PROJECT_ROOT / "scripts" / "parse_hmmer_stats.py"
HMMSEARCH = PROJECT_ROOT / "refs" / "install" / "hmmer-3.4" / "bin" / "hmmsearch"


def parse_cpus(value: str) -> list[int]:
    cpus = [int(item) for item in value.split(",")]
    if not cpus or any(item < 0 for item in cpus) or len(set(cpus)) != len(cpus):
        raise argparse.ArgumentTypeError("CPU list must contain unique nonnegative integers")
    return cpus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=parse_cpus, default=parse_cpus("0,1,4,8,16,32"))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--run-status",
        choices=("reported", "pilot"),
        default="pilot",
        help="Pilot timings are never promoted to reported benchmark results.",
    )
    args = parser.parse_args()

    if not HMMSEARCH.is_file():
        parser.error(f"missing pristine HMMER executable: {HMMSEARCH}")
    for path in (args.hmm, args.fasta):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.warmups < 0 or args.replicates < 1:
        parser.error("warmups must be >= 0 and replicates must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for workers in args.cpus:
        run_kinds = [("warmup", index + 1) for index in range(args.warmups)]
        run_kinds += [("replicate", index + 1) for index in range(args.replicates)]
        for kind, index in run_kinds:
            stem = f"{args.dataset_id}-hmmer-cpu{workers}-{kind}{index}"
            metadata = {
                "dataset_id": args.dataset_id,
                "hmm": str(args.hmm.resolve()),
                "fasta": str(args.fasta.resolve()),
                "hmmer_cpu_workers": workers,
                "run_kind": kind,
                "run_index": index,
                "run_status": args.run_status,
                "note": "--cpu is HMMER worker count; its reader/master thread is additional when threading is enabled.",
            }
            output_json = args.output_dir / f"{stem}.json"
            stdout = args.output_dir / f"{stem}.out"
            stderr = args.output_dir / f"{stem}.err"
            stats = args.output_dir / f"{stem}-stats.json"
            tblout = args.output_dir / f"{stem}.tblout"
            domtblout = args.output_dir / f"{stem}.domtblout"

            command = [
                str(RUNNER),
                "--output",
                str(output_json),
                "--stdout",
                str(stdout),
                "--stderr",
                str(stderr),
                "--label",
                stem,
                "--metadata-json",
                json.dumps(metadata, sort_keys=True),
                "--",
                str(HMMSEARCH),
                "--cpu",
                str(workers),
                "--noali",
                "--tblout",
                str(tblout),
                "--domtblout",
                str(domtblout),
                str(args.hmm.resolve()),
                str(args.fasta.resolve()),
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            subprocess.run(
                [str(PARSER), str(stdout), str(stats)], cwd=PROJECT_ROOT, check=True
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
