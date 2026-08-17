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
from plan7_gpu import _native, _pipeline
from plan7_gpu.adapter import _candidate_state, _sequence_native
from plan7_gpu.astra_search import _prepare_candidates, hmmsearch


SEQUENCE_COUNT = 1000
EXPECTED_TARGET = "sm90_h200"
_MISSING = object()
_POSITIVE_WORKSPACE_FIELDS = (
    "postfilter_device_bytes",
    "postfilter_dp_capacity_bytes",
    "postfilter_run_count",
    "forward_device_bytes",
    "forward_dp_capacity_bytes",
    "forward_xmx_capacity_bytes",
    "forward_gather_capacity_bytes",
    "forward_event_create_count",
    "forward_run_count",
)
_SEAM_PROBES = {
    "filter": "_filter_scores_seam_available",
    "forward": "_filter_and_forward_scores_seam_available",
    "simple_regions": "_simple_regions_seam_available",
    "compact_domains": "_compact_domains_seam_available",
}
_NATIVE_CAPABILITIES = {
    "ViterbiProfiles": (_native, "ViterbiProfiles"),
    "ForwardProfiles": (_native, "ForwardProfiles"),
    "SequenceBatch.postfilter_candidates_many_csr_raw": (
        _native.SequenceBatch,
        "postfilter_candidates_many_csr_raw",
    ),
    "SequenceBatch.forward_candidates_many_raw": (
        _native.SequenceBatch,
        "forward_candidates_many_raw",
    ),
}


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


def identifier_bytes(value: object | None) -> bytes | None:
    """Normalize PyHMMER identifier storage without changing its text."""
    if value is None or type(value) is bytes:
        return value
    if type(value) is str:
        return value.encode("utf-8")
    raise TypeError(f"unexpected PyHMMER identifier type: {type(value).__name__}")


def exact_payload(hits: object) -> tuple[bytes, bytes, bytes, bytes]:
    query_name = identifier_bytes(hits.query.name) or b""
    query_accession = identifier_bytes(hits.query.accession)
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


def gpu_capability_record() -> dict[str, object]:
    """Resolve and attest every native seam required by the V2 GPU route."""
    seams = {
        name: bool(getattr(_pipeline, attribute)())
        for name, attribute in _SEAM_PROBES.items()
    }
    cache = _pipeline._continuation_seam_cache_info()
    native = {
        name: callable(getattr(owner, attribute, None))
        for name, (owner, attribute) in _NATIVE_CAPABILITIES.items()
    }
    failures = [name for name, available in seams.items() if not available]
    failures.extend(name for name, available in native.items() if not available)
    for name in _SEAM_PROBES:
        state = cache.get(name)
        if not isinstance(state, dict):
            failures.append(f"{name}.cache_missing")
            continue
        for field in ("resolved", "available", "same_dso"):
            if state.get(field) is not True:
                failures.append(f"{name}.{field}")
        for field in ("resolutions", "dlopen_calls", "dlclose_calls"):
            if state.get(field) != 1:
                failures.append(f"{name}.{field}")
    return {
        "seams": seams,
        "native_capabilities": native,
        "continuation_seam_cache": cache,
        "attested": not failures,
        "failures": failures,
    }


def candidate_census(candidates: object) -> dict[str, object]:
    """Summarize every authentic post-filter row without exposing payloads."""
    counts = [candidates.candidate_count(row) for row in range(len(candidates))]
    digest = hashlib.sha256()
    for count in counts:
        digest.update(struct.pack("=q", count))
    out_of_range = [
        row for row, count in enumerate(counts) if not 0 <= count <= SEQUENCE_COUNT
    ]
    state = _candidate_state(candidates)
    routes = None
    routes_unavailable_reason = None
    if state.sealed_postfilter is not None:
        try:
            routes = _pipeline._sealed_continuation_statistics_bound(
                state.sealed_postfilter
            )
        except TypeError as error:
            # The direct first-1000 bridge seals filter/Forward provenance but
            # does not build the optional persistent-session domain journal.
            routes_unavailable_reason = str(error)
    return {
        "model_rows": len(counts),
        "total_retained_candidates": sum(counts),
        "nonempty_model_rows": sum(count > 0 for count in counts),
        "all_target_model_rows": sum(count == SEQUENCE_COUNT for count in counts),
        "minimum_row_candidates": min(counts, default=0),
        "maximum_row_candidates": max(counts, default=0),
        "out_of_range_model_rows": len(out_of_range),
        "out_of_range_model_row_examples": out_of_range[:20],
        "row_counts_sha256": digest.hexdigest(),
        "sealed_postfilter": state.sealed_postfilter is not None,
        "continuation_routes": routes,
        "continuation_routes_unavailable_reason": routes_unavailable_reason,
    }


def nonvacuous_gpu_work_failures(
    workspace: dict[str, object], candidates: dict[str, object]
) -> list[str]:
    """Return every reason the observed CUDA work is not demonstrably live."""
    failures = [
        f"workspace.{field} is not positive"
        for field in _POSITIVE_WORKSPACE_FIELDS
        if not isinstance(workspace.get(field), int) or workspace[field] <= 0
    ]
    for field in ("model_rows", "total_retained_candidates", "nonempty_model_rows"):
        if not isinstance(candidates.get(field), int) or candidates[field] <= 0:
            failures.append(f"candidates.{field} is not positive")
    if candidates.get("sealed_postfilter") is not True:
        failures.append("candidates.sealed_postfilter is not true")
    if candidates.get("out_of_range_model_rows") != 0:
        failures.append("candidate row counts fall outside [0, 1000]")
    routes = candidates.get("continuation_routes")
    if routes is not None:
        route_total = sum(
            int(routes[field])
            for field in (
                "cpu_required_count",
                "no_region_count",
                "simple_count",
            )
        )
        if int(routes["row_count"]) != route_total:
            failures.append("continuation route counts do not sum to row_count")
        if int(routes["row_count"]) != candidates.get(
            "total_retained_candidates"
        ):
            failures.append("continuation rows differ from retained candidates")
    return failures


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
        gpu_options = dict(options)
        candidates = _prepare_candidates(pairs, batch, gpu_options, True)
        candidates_record = candidate_census(candidates)
        gpu_results = hmmsearch(
            pairs, candidates, cpus=1, postfilter=True, **gpu_options
        )
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
        # The iterator is now exhausted and all queued native work is complete;
        # capture these counters while the underlying native batch is live.
        native_batch = _sequence_native(batch)
        workspace = dict(native_batch.workspace_statistics)
        memory_snapshot = dict(native_batch.memory_snapshot)
        gpu_work_failures = nonvacuous_gpu_work_failures(
            workspace, candidates_record
        )
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
        "candidate_census": candidates_record,
        "workspace_statistics_after_consumption": workspace,
        "memory_snapshot_after_consumption": memory_snapshot,
        "nonvacuous_gpu_work": not gpu_work_failures,
        "nonvacuous_gpu_work_failures": gpu_work_failures,
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
    capabilities = gpu_capability_record()
    if capabilities["attested"] is not True:
        raise RuntimeError(
            "first-1000 acceptance requires every V2 same-DSO GPU seam: "
            + ", ".join(capabilities["failures"])
        )

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
        and comparison["candidate_census"]["model_rows"]
        == comparison["expected_models"]
        and comparison["nonvacuous_gpu_work"] is True
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
            "pipeline_extension": {
                "path": str(Path(_pipeline.__file__).resolve()),
                "sha256": sha256_file(Path(_pipeline.__file__).resolve()),
            },
        },
        "gpu_capabilities": capabilities,
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
