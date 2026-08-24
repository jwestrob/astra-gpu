#!/usr/bin/env python3
"""Census certified GA pruning opportunity on PFAM x first 1000 targets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import resource
import struct
import subprocess
import time
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, _native, load_pressed_profiles
from plan7_gpu.adapter import _sequence_native


EXPECTED_TARGET = "sm90_h200"
EXPECTED_SEQUENCE_COUNT = 1000
EXPECTED_OUTPUT_SHA256 = (
    "a0d928902cd8f6c3342b05c71aedaf0190ac2e02478ff879988a41d250de85b1"
)
F1 = 0.02
F2 = 0.001
F3 = 0.00001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier_bytes(value: object | None) -> bytes | None:
    if value is None or type(value) is bytes:
        return value
    if type(value) is str:
        return value.encode("utf-8")
    raise TypeError(f"unexpected identifier type: {type(value).__name__}")


def table_bytes(hits: object, table_format: str) -> bytes:
    output = io.BytesIO()
    hits.write(output, format=table_format, header=True)
    return output.getvalue()


def update_output_digest(digest: object, hits: object) -> None:
    query_name = identifier_bytes(hits.query.name) or b""
    query_accession = identifier_bytes(hits.query.accession)
    payloads = (
        struct.pack(
            "=Iii",
            hits.query.M,
            len(query_name),
            -1 if query_accession is None else len(query_accession),
        )
        + query_name
        + (b"" if query_accession is None else query_accession),
        table_bytes(hits, "targets"),
        table_bytes(hits, "domains"),
        struct.pack(
            "=QQQQdd",
            hits.searched_models,
            hits.searched_nodes,
            hits.searched_sequences,
            hits.searched_residues,
            hits.Z,
            hits.domZ,
        ),
    )
    for payload in payloads:
        digest.update(struct.pack("=Q", len(payload)))
        digest.update(payload)


def pipeline(alphabet: object) -> object:
    return pyhmmer.plan7.Pipeline(alphabet, bit_cutoffs="gathering")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfam-base", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=333)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != EXPECTED_TARGET:
        raise RuntimeError("Phase 9 census requires attested sm90/H200")

    started = time.monotonic()
    pairs = load_pressed_profiles(arguments.pfam_base)
    alphabet = pairs[0].hmm.alphabet
    with pyhmmer.easel.SequenceFile(
        arguments.targets, digital=True, alphabet=alphabet
    ) as source:
        targets = source.read_block()
    if len(targets) != EXPECTED_SEQUENCE_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_SEQUENCE_COUNT} targets, found {len(targets)}"
        )
    target_names = [identifier_bytes(target.name) or b"" for target in targets]
    if len(set(target_names)) != len(target_names):
        raise RuntimeError("Phase 9 target names must be unique")

    aggregate_fields = (
        "source_domain_rows",
        "evaluable_target_rows",
        "compact_region_count",
        "compact_region_cells_per_downstream_pass",
        "whole_forward_upper_below_ga_count",
        "reconstruction_upper_below_ga_count",
        "certified_target_reject_count",
        "certified_target_reject_region_count",
        "certified_target_reject_region_cells_per_downstream_pass",
        "domain_upper_below_ga_count",
        "domain_upper_below_ga_region_cells_per_downstream_pass",
        "certified_target_index_payload_bytes",
        "native_temporary_bytes",
        "preflight_ns",
        "scan_ns",
    )
    aggregate = {field: 0 for field in aggregate_fields}
    profiles_with_certified_reject = 0
    reported_conflicts = []
    profile_records = []
    output_digest = hashlib.sha256()
    generation_seconds = 0.0
    continuation_seconds = 0.0
    census_seconds = 0.0
    maximum_candidate_resident_bytes = 0
    maximum_candidate_owned_host_bytes = 0
    maximum_candidate_owned_device_bytes = 0
    device_before = _native.device_memory_info()
    maximum_device_used_bytes = (
        device_before["cuda_total_bytes"] - device_before["cuda_free_bytes"]
    )

    with ProfileSession(pairs, pack_workers=4) as session, SequenceBatch(
        targets
    ) as batch:
        for begin in range(0, len(pairs), arguments.chunk_size):
            end = min(begin + arguments.chunk_size, len(pairs))
            with session.select(range(begin, end)) as selection:
                generation_pipeline = pipeline(alphabet)
                generation_started = time.monotonic()
                candidates = batch._postfilter_forward_selection(
                    selection,
                    F1,
                    F2,
                    F3,
                    True,
                    pipeline=generation_pipeline,
                    sparse_journal_v3=True,
                )
                generation_seconds += time.monotonic() - generation_started
                resident = candidates.resident_memory
                maximum_candidate_resident_bytes = max(
                    maximum_candidate_resident_bytes, resident["resident_bytes"]
                )
                maximum_candidate_owned_host_bytes = max(
                    maximum_candidate_owned_host_bytes,
                    resident["owned_host_bytes"],
                )
                maximum_candidate_owned_device_bytes = max(
                    maximum_candidate_owned_device_bytes,
                    resident["owned_device_bytes"],
                )
                device_now = _native.device_memory_info()
                maximum_device_used_bytes = max(
                    maximum_device_used_bytes,
                    device_now["cuda_total_bytes"]
                    - device_now["cuda_free_bytes"],
                )
                for local_row, pair in enumerate(pairs[begin:end]):
                    worker = pipeline(alphabet)
                    census_started = time.monotonic()
                    census = candidates.evaluate_ga_pruning(
                        local_row, worker, include_indices=True
                    )
                    census_seconds += time.monotonic() - census_started
                    for field in aggregate_fields:
                        aggregate[field] += int(census[field])
                    if census["certified_target_reject_count"]:
                        profiles_with_certified_reject += 1

                    continuation_started = time.monotonic()
                    hits = candidates.search(local_row, worker)
                    continuation_seconds += time.monotonic() - continuation_started
                    update_output_digest(output_digest, hits)
                    reported_names = {
                        identifier_bytes(hit.name) or b""
                        for hit in hits
                        if hit.reported
                    }
                    conflicts = [
                        target_index
                        for target_index in census["certified_target_indices"]
                        if target_names[target_index] in reported_names
                    ]
                    if conflicts:
                        reported_conflicts.append(
                            {
                                "profile_index": begin + local_row,
                                "target_indices": conflicts,
                            }
                        )
                    if census["certified_target_reject_count"]:
                        profile_records.append(
                            {
                                "profile_index": begin + local_row,
                                "profile_name": (
                                    identifier_bytes(pair.hmm.name) or b""
                                ).decode("utf-8", "replace"),
                                "model_length": pair.hmm.M,
                                "target_cutoff_bits": census[
                                    "target_cutoff_bits"
                                ],
                                "domain_cutoff_bits": census[
                                    "domain_cutoff_bits"
                                ],
                                "evaluable_target_rows": census[
                                    "evaluable_target_rows"
                                ],
                                "certified_target_reject_count": census[
                                    "certified_target_reject_count"
                                ],
                                "certified_target_reject_region_count": census[
                                    "certified_target_reject_region_count"
                                ],
                                "certified_target_reject_region_cells_per_downstream_pass": census[
                                    "certified_target_reject_region_cells_per_downstream_pass"
                                ],
                            }
                        )
                del candidates
        batch_memory = dict(_sequence_native(batch).memory_snapshot)
    device_after = _native.device_memory_info()

    exact_output_sha256 = output_digest.hexdigest()
    if exact_output_sha256 != EXPECTED_OUTPUT_SHA256:
        raise AssertionError(
            "current exact output digest changed: "
            f"{exact_output_sha256} != {EXPECTED_OUTPUT_SHA256}"
        )
    if reported_conflicts:
        raise AssertionError(
            f"certified GA rejects included reported hits: {reported_conflicts[:8]}"
        )
    if aggregate["evaluable_target_rows"] == 0:
        raise AssertionError("Phase 9 census had no evaluable compact rows")

    output = {
        "schema": "plan7_gpu.phase9_ga_census_h200.v1",
        "status": "PASS",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        ),
        "provenance": provenance,
        "input": {
            "pfam_base": str(arguments.pfam_base.resolve()),
            "profile_count": len(pairs),
            "targets": str(arguments.targets.resolve()),
            "target_sha256": sha256_file(arguments.targets),
            "target_count": len(targets),
            "target_residues": targets.total_length(),
            "chunk_size": arguments.chunk_size,
        },
        "exact_output_sha256": exact_output_sha256,
        "expected_output_sha256": EXPECTED_OUTPUT_SHA256,
        "reported_conflicts": reported_conflicts,
        "aggregate": {
            **aggregate,
            "profiles_with_certified_target_reject": (
                profiles_with_certified_reject
            ),
            "certified_target_fraction_of_evaluable": (
                aggregate["certified_target_reject_count"]
                / aggregate["evaluable_target_rows"]
            ),
            "certified_region_fraction_of_compact": (
                aggregate["certified_target_reject_region_count"]
                / aggregate["compact_region_count"]
            ),
            "certified_cell_fraction_of_compact": (
                aggregate[
                    "certified_target_reject_region_cells_per_downstream_pass"
                ]
                / aggregate["compact_region_cells_per_downstream_pass"]
            ),
            "domain_below_fraction_of_compact": (
                aggregate["domain_upper_below_ga_count"]
                / aggregate["compact_region_count"]
            ),
        },
        "timing": {
            "generation_seconds": generation_seconds,
            "census_seconds": census_seconds,
            "continuation_seconds": continuation_seconds,
            "wall_seconds": time.monotonic() - started,
        },
        "memory": {
            "device_before": device_before,
            "device_after": device_after,
            "batch_persistent": batch_memory,
            "maximum_candidate_resident_bytes": (
                maximum_candidate_resident_bytes
            ),
            "maximum_candidate_owned_host_bytes": (
                maximum_candidate_owned_host_bytes
            ),
            "maximum_candidate_owned_device_bytes": (
                maximum_candidate_owned_device_bytes
            ),
            "maximum_observed_cuda_used_bytes": maximum_device_used_bytes,
            "maximum_rss_kib": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
        "profiles_with_certified_reject": profile_records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(arguments.output.name + ".tmp")
    temporary.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, arguments.output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "input": output["input"],
                "exact_output_sha256": exact_output_sha256,
                "aggregate": output["aggregate"],
                "timing": output["timing"],
                "memory": output["memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
