"""H200-only, exact PFAM x first-1000 CPU semantic acceptance gate.

This is deliberately not a throughput benchmark.  It refuses every CUDA
target except the explicit H200/sm90 attestation and refuses FASTA inputs that
do not contain exactly 1,000 targets.  A whole-metagenome run must not be
started until this program writes a PASS record.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import struct
import subprocess
import sys
import time
from itertools import zip_longest
from pathlib import Path

import pyhmmer

from plan7_gpu import SequenceBatch, load_pressed_profiles
from plan7_gpu import _native
from plan7_gpu.astra_search import hmmsearch


SEQUENCE_COUNT = 1000
EXPECTED_TARGET = "sm90_h200"
_MISSING = object()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def git_revision(root: Path) -> dict[str, object]:
    revision = command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = command_output(["git", "-C", str(root), "status", "--porcelain"])
    return {"revision": revision, "dirty": bool(status)}


def pci_tuple(value: str) -> tuple[int, int, int, int]:
    domain, bus, tail = value.split(":")
    device, function = tail.split(".")
    return tuple(int(item, 16) for item in (domain, bus, device, function))


def visible_gpu_record(cuda: dict[str, object]) -> dict[str, object]:
    fields = "index,name,uuid,compute_cap,memory.total,driver_version,pci.bus_id"
    output = command_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    records = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 7:
            raise RuntimeError("nvidia-smi returned a malformed identity row")
        records.append(
            dict(
                zip(
                    (
                        "physical_index",
                        "name",
                        "uuid",
                        "compute_capability",
                        "memory_total_mib",
                        "driver_package_version",
                        "pci_bus_address",
                    ),
                    values,
                    strict=True,
                )
            )
        )
    matches = [record for record in records if record["uuid"] == cuda["uuid"]]
    if len(matches) != 1:
        raise RuntimeError(
            "CUDA runtime UUID does not identify exactly one nvidia-smi GPU"
        )
    match = matches[0]
    expected_capability = ".".join(str(value) for value in cuda["compute_capability"])
    if match["name"] != cuda["name"]:
        raise RuntimeError("CUDA runtime and nvidia-smi product names differ")
    if match["compute_capability"] != expected_capability:
        raise RuntimeError("CUDA runtime and nvidia-smi compute capabilities differ")
    if pci_tuple(match["pci_bus_address"]) != pci_tuple(cuda["pci_bus_address"]):
        raise RuntimeError("CUDA runtime and nvidia-smi PCI identities differ")
    return match


def load_exact_first1000(path: Path) -> pyhmmer.easel.DigitalSequenceBlock:
    alphabet = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(path, digital=True, alphabet=alphabet) as source:
        targets = source.read_block()
    if len(targets) != SEQUENCE_COUNT:
        raise RuntimeError(
            f"acceptance FASTA must contain exactly {SEQUENCE_COUNT} targets; "
            f"found {len(targets)}"
        )
    return targets


def table_bytes(hits: object, table_format: str) -> bytes:
    output = io.BytesIO()
    hits.write(output, format=table_format, header=True)
    return output.getvalue()


def exact_payload(hits: object) -> tuple[bytes, bytes, bytes, bytes]:
    query_name = hits.query.name or b""
    query_accession = hits.query.accession
    query_identity = (
        struct.pack(
            "=Iii",
            hits.query.M,
            len(query_name),
            -1 if query_accession is None else len(query_accession),
        )
        + query_name
        + (b"" if query_accession is None else query_accession)
    )
    accounting = struct.pack(
        "=QQQQdd",
        hits.searched_models,
        hits.searched_nodes,
        hits.searched_sequences,
        hits.searched_residues,
        hits.Z,
        hits.domZ,
    )
    return (
        query_identity,
        table_bytes(hits, "targets"),
        table_bytes(hits, "domains"),
        accounting,
    )


def update_digest(digest: object, payload: tuple[bytes, ...]) -> None:
    for item in payload:
        digest.update(struct.pack("=Q", len(item)))
        digest.update(item)


def compare_searches(pairs: tuple[object, ...], targets: object) -> dict[str, object]:
    options = {"bit_cutoffs": "gathering"}
    cpu_started = time.monotonic()
    cpu_results = pyhmmer.hmmsearch(
        (pair.hmm for pair in pairs), targets, cpus=1, **options
    )
    cpu_setup_seconds = time.monotonic() - cpu_started

    gpu_setup_started = time.monotonic()
    batch = SequenceBatch(targets)
    try:
        gpu_results = hmmsearch(pairs, batch, cpus=1, postfilter=True, **options)
        gpu_setup_seconds = time.monotonic() - gpu_setup_started
        comparison_started = time.monotonic()
        cpu_digest = hashlib.sha256()
        gpu_digest = hashlib.sha256()
        mismatch_examples = []
        mismatch_count = 0
        model_count = 0
        for ordinal, (cpu_hits, gpu_hits) in enumerate(
            zip_longest(cpu_results, gpu_results, fillvalue=_MISSING)
        ):
            model_count += 1
            if cpu_hits is _MISSING or gpu_hits is _MISSING:
                mismatch_count += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {"ordinal": ordinal, "kind": "result_count"}
                    )
                continue
            cpu_payload = exact_payload(cpu_hits)
            gpu_payload = exact_payload(gpu_hits)
            update_digest(cpu_digest, cpu_payload)
            update_digest(gpu_digest, gpu_payload)
            if cpu_payload != gpu_payload:
                mismatch_count += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(
                        {
                            "ordinal": ordinal,
                            "query_name": cpu_hits.query.name.decode(
                                "utf-8", "replace"
                            ),
                            "query_identity_equal": (cpu_payload[0] == gpu_payload[0]),
                            "targets_equal": cpu_payload[1] == gpu_payload[1],
                            "domains_equal": cpu_payload[2] == gpu_payload[2],
                            "accounting_equal": cpu_payload[3] == gpu_payload[3],
                        }
                    )
        comparison_seconds = time.monotonic() - comparison_started
    finally:
        batch.close()

    return {
        "models_compared": model_count,
        "expected_models": len(pairs),
        "mismatch_count": mismatch_count,
        "mismatch_examples": mismatch_examples,
        "cpu_payload_sha256": cpu_digest.hexdigest(),
        "gpu_payload_sha256": gpu_digest.hexdigest(),
        "cpu_iterator_setup_seconds": cpu_setup_seconds,
        "gpu_filter_setup_seconds": gpu_setup_seconds,
        "comparison_seconds": comparison_seconds,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--pressed-base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for label, path in (
        ("FASTA", args.fasta),
        ("pressed manifest", args.manifest),
    ):
        if not path.is_file():
            parser.error(f"{label} is not a file: {path}")
    for suffix in (".h3f", ".h3i", ".h3m", ".h3p"):
        if not Path(f"{args.pressed_base}{suffix}").is_file():
            parser.error(f"missing pressed artifact: {args.pressed_base}{suffix}")

    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != EXPECTED_TARGET:
        raise RuntimeError(
            "first-1000 acceptance requires the attested H200/sm90 target: "
            + (provenance["reason"] or provenance["target"])
        )
    nvidia_smi = visible_gpu_record(provenance["cuda"])
    source = git_revision(root)
    if source["dirty"]:
        raise RuntimeError("first-1000 acceptance requires a clean source revision")

    started = time.monotonic()
    fasta_sha256 = sha256_file(args.fasta)
    targets = load_exact_first1000(args.fasta)
    if sha256_file(args.fasta) != fasta_sha256:
        raise RuntimeError("acceptance FASTA changed while it was being loaded")
    pairs = load_pressed_profiles(args.pressed_base, manifest=args.manifest)
    comparison = compare_searches(pairs, targets)
    passed = (
        comparison["models_compared"] == comparison["expected_models"]
        and comparison["mismatch_count"] == 0
        and comparison["cpu_payload_sha256"] == comparison["gpu_payload_sha256"]
    )
    record = {
        "schema_version": 1,
        "verdict": "PASS" if passed else "FAIL",
        "scope": "PFAM x exactly first 1000 targets; semantic acceptance only",
        "warning": ("A whole-metagenome GPU run is forbidden unless verdict is PASS."),
        "command": sys.argv,
        "host": {
            "hostname": platform.node(),
            "pid": os.getpid(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_JOB_PARTITION": os.environ.get("SLURM_JOB_PARTITION"),
            "SLURM_JOB_NODELIST": os.environ.get("SLURM_JOB_NODELIST"),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
            "SLURM_STEP_GPUS": os.environ.get("SLURM_STEP_GPUS"),
        },
        "source": {
            **source,
            "native_extension": {
                "path": str(Path(_native.__file__).resolve()),
                "sha256": sha256_file(Path(_native.__file__).resolve()),
            },
        },
        "cuda_attestation": provenance,
        "nvidia_smi_selected_gpu": nvidia_smi,
        "workload": {
            "fasta": {
                "path": str(args.fasta.resolve()),
                "sha256": fasta_sha256,
                "sequences": len(targets),
                "residues": targets.total_length(),
            },
            "pressed_base": str(args.pressed_base.resolve()),
            "manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256_file(args.manifest),
            },
        },
        "comparison": comparison,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
