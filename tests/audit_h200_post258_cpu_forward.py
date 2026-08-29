"""Focused exact oracle for experimental CPU Forward ownership."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native, _pipeline
from plan7_gpu.adapter import _candidate_state, _sequence_state

from audit_h200_phase1a_sparse_v3 import OPTIONS, SELECTION, fixture, pipeline


def generate(selection, targets, alphabet, ownership: str):
    previous_domain = os.environ.get("PLAN7_GPU_DOMAIN_OWNERSHIP")
    previous_forward = os.environ.get("PLAN7_GPU_FORWARD_OWNERSHIP")
    try:
        os.environ["PLAN7_GPU_DOMAIN_OWNERSHIP"] = "cpu"
        os.environ["PLAN7_GPU_FORWARD_OWNERSHIP"] = ownership
        with SequenceBatch(targets) as batch:
            started = time.perf_counter_ns()
            candidates = batch._postfilter_forward_selection(
                selection,
                **OPTIONS,
                bias_filter=True,
                pipeline=pipeline(alphabet),
                sparse_journal_v3=True,
            )
            elapsed_ns = time.perf_counter_ns() - started
            sealed = _candidate_state(candidates).sealed_postfilter
            if sealed is None:
                raise AssertionError("fused generation did not produce a seal")
            statistics = _pipeline._sealed_continuation_statistics_bound(sealed)
            workspace = dict(_sequence_state(batch).native.workspace_statistics)
        return candidates, statistics, workspace, elapsed_ns
    finally:
        if previous_domain is None:
            os.environ.pop("PLAN7_GPU_DOMAIN_OWNERSHIP", None)
        else:
            os.environ["PLAN7_GPU_DOMAIN_OWNERSHIP"] = previous_domain
        if previous_forward is None:
            os.environ.pop("PLAN7_GPU_FORWARD_OWNERSHIP", None)
        else:
            os.environ["PLAN7_GPU_FORWARD_OWNERSHIP"] = previous_forward


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != "sm90_h200":
        raise RuntimeError("this oracle requires an attested H200/sm90 runtime")

    hmms, targets = fixture()
    alphabet = hmms[0].alphabet
    with tempfile.TemporaryDirectory(prefix="post258-p6-") as temporary:
        base = Path(temporary) / "models"
        pyhmmer.hmmer.hmmpress(hmms, base)
        pairs = load_pressed_profiles(base)
        with ProfileSession(pairs, pack_workers=1) as session, session.select(
            SELECTION
        ) as selection:
            reference, reference_stats, reference_workspace, reference_ns = generate(
                selection, targets, alphabet, "gpu"
            )
            cpu, cpu_stats, cpu_workspace, cpu_ns = generate(
                selection, targets, alphabet, "cpu"
            )

        rows = []
        for row in range(len(SELECTION)):
            reference_hits = reference.search(row, pipeline(alphabet))
            cpu_hits = cpu.search(row, pipeline(alphabet))
            reference_digest = _pipeline._semantic_tophits_fingerprint_bound(
                reference_hits
            ).hex()
            cpu_digest = _pipeline._semantic_tophits_fingerprint_bound(cpu_hits).hex()
            if cpu_digest != reference_digest:
                raise AssertionError(f"CPU Forward differs at profile {row}")
            rows.append(reference_digest)

    reference_sparse = reference_stats["sparse_journal_v3"]
    cpu_sparse = cpu_stats["sparse_journal_v3"]
    if reference_sparse["dense_forward_count"] == 0:
        raise AssertionError("reference fixture did not run GPU Forward")
    if cpu_sparse["dense_forward_count"] != 0:
        raise AssertionError("all-CPU mode retained GPU Forward rows")
    if cpu_sparse["exception_routes"]["filter_scores"] == 0:
        raise AssertionError("all-CPU mode produced no filter-score routes")

    result = {
        "status": "PASS",
        "profiles": len(SELECTION),
        "targets": len(targets),
        "reference_generation_ns": reference_ns,
        "cpu_generation_ns": cpu_ns,
        "reference_sparse": reference_sparse,
        "cpu_sparse": cpu_sparse,
        "reference_workspace": reference_workspace,
        "cpu_workspace": cpu_workspace,
        "tophits_sha256": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with temporary.open("x") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(hashlib.sha256(payload.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
