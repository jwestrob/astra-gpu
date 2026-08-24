#!/usr/bin/env python3
"""Exact H200 oracle for the production Phase 9 GA early exit."""

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

from plan7_gpu import ProfileSession, SequenceBatch, _native, _pipeline
from plan7_gpu import load_pressed_profiles
from plan7_gpu.adapter import _candidate_state


EXPECTED_TARGET = "sm90_h200"
PROFILE_COUNT = 64
EXPECTED_GA_TARGET_ROWS = 11
EXPECTED_GA_REGIONS = 19
F1 = 0.02
F2 = 0.001
F3 = 0.00001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfam-base", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    return parser.parse_args()


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


def semantic_payload(hits: object) -> tuple[bytes, ...]:
    query_name = identifier_bytes(hits.query.name) or b""
    query_accession = identifier_bytes(hits.query.accession)
    return (
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


def payload_digest(payloads: list[tuple[bytes, ...]]) -> str:
    digest = hashlib.sha256()
    for row in payloads:
        for payload in row:
            digest.update(struct.pack("=Q", len(payload)))
            digest.update(payload)
    return digest.hexdigest()


def pipeline(alphabet: object) -> object:
    return pyhmmer.plan7.Pipeline(alphabet, bit_cutoffs="gathering")


def main() -> int:
    arguments = parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
    )
    if revision != arguments.expected_revision or dirty:
        raise RuntimeError("Phase 9 source revision is not the clean pin")
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != EXPECTED_TARGET:
        raise RuntimeError("Phase 9 production oracle requires sm90/H200")

    pairs = load_pressed_profiles(arguments.pfam_base)
    alphabet = pairs[0].hmm.alphabet
    with pyhmmer.easel.SequenceFile(
        arguments.targets, digital=True, alphabet=alphabet
    ) as source:
        targets = source.read_block()

    timing: dict[str, float] = {}
    started = time.monotonic()
    with ProfileSession(pairs, pack_workers=4) as session, SequenceBatch(
        targets
    ) as batch, session.select(range(PROFILE_COUNT)) as selection:
        before = _native.device_memory_info()
        stage = time.monotonic()
        baseline = batch._postfilter_forward_selection(
            selection,
            F1,
            F2,
            F3,
            True,
            pipeline=pipeline(alphabet),
            sparse_journal_v3=True,
        )
        timing["baseline_generation_seconds"] = time.monotonic() - stage
        stage = time.monotonic()
        optimized = batch._postfilter_forward_selection(
            selection,
            F1,
            F2,
            F3,
            True,
            pipeline=pipeline(alphabet),
            sparse_journal_v3=True,
            _ga_pruning=True,
            telemetry=True,
        )
        timing["optimized_generation_seconds"] = time.monotonic() - stage
        after_generation = _native.device_memory_info()

        baseline_payloads = []
        optimized_payloads = []
        stage = time.monotonic()
        for row in range(PROFILE_COUNT):
            baseline_payloads.append(
                semantic_payload(baseline.search(row, pipeline(alphabet)))
            )
        timing["baseline_continuation_seconds"] = time.monotonic() - stage
        stage = time.monotonic()
        for row in range(PROFILE_COUNT):
            optimized_payloads.append(
                semantic_payload(optimized.search(row, pipeline(alphabet)))
            )
        timing["optimized_continuation_seconds"] = time.monotonic() - stage
        if baseline_payloads != optimized_payloads:
            differing = [
                row
                for row, (left, right) in enumerate(
                    zip(baseline_payloads, optimized_payloads, strict=True)
                )
                if left != right
            ]
            raise AssertionError(
                f"GA early exit changed HMMER output rows: {differing[:8]}"
            )

        sealed = _candidate_state(optimized).sealed_postfilter
        continuation = _pipeline._sealed_continuation_statistics_bound(sealed)
        generation = optimized.generation_statistics
        ga_regions = continuation["compact_certified_ga_count"]
        ga_rows = continuation["no_region_count"]
        if ga_rows != EXPECTED_GA_TARGET_ROWS or ga_regions != EXPECTED_GA_REGIONS:
            raise AssertionError(
                f"GA census changed: rows={ga_rows}, regions={ga_regions}"
            )
        if (
            generation["native_totals"]["rescore"][
                "certified_ga_region_count"
            ]
            != ga_regions
        ):
            raise AssertionError("GA native/continuation counts disagree")
        memory = {
            "device_before": before,
            "device_after_generation": after_generation,
            "baseline_candidate": baseline.resident_memory,
            "optimized_candidate": optimized.resident_memory,
            "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }

    timing["wall_seconds"] = time.monotonic() - started
    baseline_digest = payload_digest(baseline_payloads)
    optimized_digest = payload_digest(optimized_payloads)
    result = {
        "schema": "plan7_gpu.phase9_ga_production_h200.v1",
        "status": "PASS",
        "source_revision": revision,
        "provenance": provenance,
        "profile_count": PROFILE_COUNT,
        "target_count": len(targets),
        "baseline_digest": baseline_digest,
        "optimized_digest": optimized_digest,
        "certified_ga_target_rows": ga_rows,
        "certified_ga_regions": ga_regions,
        "certified_ga_skipped_work_cells": generation["native_totals"][
            "rescore"
        ]["certified_ga_skipped_work_cells"],
        "timing": timing,
        "memory": memory,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(arguments.output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
