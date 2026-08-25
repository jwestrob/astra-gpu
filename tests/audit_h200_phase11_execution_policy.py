"""Exact H200 workload-shape matrix for the Phase 11 GPU policy."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native, _pipeline
from plan7_gpu.adapter import _candidate_state


F1 = 0.99
SOURCE_MODELS = (
    "RREFam.hmm",
    "PF02826.hmm",
    "Thioesterase.hmm",
    "KR.hmm",
    "LuxC.hmm",
)
SHAPES = (
    ("tiny", 1, 1),
    ("one_profile_large_targets", 1, 4096),
    ("ten_profiles_large_targets", 10, 4096),
    ("hundred_profiles_medium_targets", 100, 512),
    ("large_profiles_small_targets", 512, 16),
    ("large_by_large", 512, 4096),
)
POLICIES = ("auto", "simple", "throughput")
POLICY_ENVIRONMENT = (
    "PLAN7_GPU_SSV_PROFILE_POLICY",
    "PLAN7_GPU_SSV_LENGTH_METADATA",
    "PLAN7_GPU_FULL_MSV_POLICY",
    "PLAN7_GPU_FULL_MSV_ARITHMETIC",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_output(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments),
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def module_identity(module) -> dict[str, str]:
    path = Path(module.__file__).resolve()
    return {"path": str(path), "sha256": sha256(path.read_bytes())}


def mutate_copy(hmm: pyhmmer.plan7.HMM, ordinal: int):
    result = hmm.copy()
    result.name = f"phase11-{ordinal:04d}".encode()
    result.accession = None
    left = ordinal % 20
    right = (ordinal * 7 + 3) % 20
    if left == right:
        right = (right + 1) % 20
    for position in range(1, result.M + 1):
        row = result.match_emissions[position]
        row[left], row[right] = row[right], row[left]
    return result


def fixture():
    data = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
    templates = []
    for name in SOURCE_MODELS:
        with pyhmmer.plan7.HMMFile(data / name) as source:
            template = source.read()
        if template is None:
            raise RuntimeError(f"missing fixture HMM: {name}")
        templates.append(template)
    hmms = [
        mutate_copy(templates[(index // 32) % len(templates)], index)
        for index in range(512)
    ]
    alphabet = hmms[0].alphabet
    amino = "ACDEFGHIKLMNPQRSTVWY"
    lengths = (1, 7, 17, 31, 64, 127, 255, 511)
    sequences = []
    for index in range(4096):
        length = lengths[index % len(lengths)]
        text = "".join(
            amino[(index * 11 + position * 7) % len(amino)]
            for position in range(length)
        )
        sequences.append(
            pyhmmer.easel.TextSequence(
                name=f"phase11-target-{index:05d}".encode(), sequence=text
            ).digitize(alphabet)
        )
    return hmms, sequences


def hit_bytes(hits) -> bytes:
    output = io.BytesIO()
    hits.write(output, format="targets", header=True)
    hits.write(output, format="domains", header=True)
    return output.getvalue()


def pipeline(alphabet):
    return pyhmmer.plan7.Pipeline(
        alphabet,
        F1=F1,
        E=10.0,
        domE=10.0,
        incE=10.0,
        incdomE=10.0,
    )


def run_policy(selection, targets, policy: str):
    with SequenceBatch(targets, execution_policy=policy) as batch:
        cold_begin = time.perf_counter_ns()
        candidate = batch.postfilter_selection(selection, F1=F1)
        cold_ns = time.perf_counter_ns() - cold_begin
        cold_records = bytes(_candidate_state(candidate).postfilter_records)
        cold_offsets = bytes(_candidate_state(candidate).offsets)
        samples = []
        for _ in range(3):
            begin = time.perf_counter_ns()
            repeated = batch.postfilter_selection(selection, F1=F1)
            samples.append(time.perf_counter_ns() - begin)
            state = _candidate_state(repeated)
            if (
                bytes(state.postfilter_records) != cold_records
                or bytes(state.offsets) != cold_offsets
            ):
                raise AssertionError("repeated policy generation changed records")
            candidate = repeated
        row_indexes = sorted({0, len(selection) - 1})
        output = bytearray()
        for row in row_indexes:
            output.extend(hit_bytes(candidate.search(row, pipeline(targets.alphabet))))
        return {
            "policy": policy,
            "cold_ns": cold_ns,
            "warm_median_ns": int(statistics.median(samples)),
            "warm_samples_ns": samples,
            "postfilter_record_count": len(cold_records) // 16,
            "postfilter_records_sha256": sha256(cold_records),
            "postfilter_offsets_sha256": sha256(cold_offsets),
            "hmm_output_sha256": sha256(bytes(output)),
            "policy_statistics": batch.execution_policy_statistics,
            "memory_snapshot": batch.memory_snapshot,
        }


def validate_routes(shape: str, profile_count: int, target_count: int, rows):
    by_policy = {row["policy"]: row for row in rows}
    identity = {
        (row["postfilter_records_sha256"], row["postfilter_offsets_sha256"],
         row["hmm_output_sha256"])
        for row in rows
    }
    if len(identity) != 1:
        raise AssertionError(f"{shape}: execution policies changed exact output")
    simple = by_policy["simple"]["policy_statistics"]
    automatic = by_policy["auto"]["policy_statistics"]
    throughput = by_policy["throughput"]["policy_statistics"]
    has_candidates = by_policy["auto"]["postfilter_record_count"] != 0
    if (
        simple["profile_packed_run_count"] != 0
        or simple["length_class_run_count"] != 0
        or simple["full_msv_compaction_run_count"] != 0
        or simple["full_msv_packed_run_count"] != 0
        or (has_candidates and simple["full_msv_legacy_run_count"] == 0)
    ):
        raise AssertionError(f"{shape}: simple policy left the reference path")
    if bool(automatic["profile_packed_run_count"]) != (profile_count >= 32):
        raise AssertionError(f"{shape}: automatic profile-axis decision is wrong")
    expected_length = target_count >= 256 and 8 <= target_count // 2
    if bool(automatic["length_class_run_count"]) != expected_length:
        raise AssertionError(f"{shape}: automatic length decision is wrong")
    if profile_count >= 4 and throughput["profile_packed_run_count"] == 0:
        raise AssertionError(f"{shape}: throughput profile packing was not used")
    if throughput["length_class_run_count"] == 0:
        raise AssertionError(f"{shape}: throughput length compaction was not used")
    if has_candidates and throughput["full_msv_compaction_run_count"] == 0:
        raise AssertionError(f"{shape}: throughput MSV compaction was not used")
    for row in rows:
        if row["policy_statistics"]["forward_candidates_per_warp"] != 1:
            raise AssertionError(f"{shape}: rejected Forward width was selected")


def audit() -> dict[str, object]:
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != "sm90_h200":
        raise RuntimeError("Phase 11 matrix requires attested sm90/H200")
    if any(os.environ.get(name) is not None for name in POLICY_ENVIRONMENT):
        raise RuntimeError("legacy policy environment overrides must be absent")
    hmms, sequences = fixture()
    shape_results = []
    with tempfile.TemporaryDirectory(prefix="phase11-execution-policy-") as temporary:
        base = Path(temporary) / "models"
        pyhmmer.hmmer.hmmpress(hmms, base)
        pairs = load_pressed_profiles(base)
        with ProfileSession(pairs, pack_workers=8) as session:
            seal_factory = _pipeline._seal_postfilter_batch_bound
            _pipeline._seal_postfilter_batch_bound = None
            try:
                warm_targets = pyhmmer.easel.DigitalSequenceBlock(
                    pairs[0].hmm.alphabet, sequences[:1]
                )
                with (
                    session.select(range(1)) as warm_selection,
                    SequenceBatch(
                        warm_targets, execution_policy="simple"
                    ) as warm_batch,
                ):
                    warm_batch.postfilter_selection(warm_selection, F1=F1)
                gc.collect()
                for shape_index, (name, profile_count, target_count) in enumerate(SHAPES):
                    targets = pyhmmer.easel.DigitalSequenceBlock(
                        pairs[0].hmm.alphabet,
                        sequences[:target_count],
                    )
                    with session.select(range(profile_count)) as selection:
                        rotation = shape_index % len(POLICIES)
                        order = POLICIES[rotation:] + POLICIES[:rotation]
                        rows = [run_policy(selection, targets, policy) for policy in order]
                    validate_routes(name, profile_count, target_count, rows)
                    shape_results.append(
                        {
                            "name": name,
                            "profile_count": profile_count,
                            "target_count": target_count,
                            "rows": rows,
                        }
                    )
                    gc.collect()
            finally:
                _pipeline._seal_postfilter_batch_bound = seal_factory
    return {
        "schema": 1,
        "status": "PASS",
        "revision": git_output("rev-parse", "HEAD"),
        "tree": git_output("rev-parse", "HEAD^{tree}"),
        "dirty": bool(git_output("status", "--porcelain")),
        "device": provenance,
        "modules": {
            "native": module_identity(_native),
            "pipeline": module_identity(_pipeline),
        },
        "policy_version": _native.EXECUTION_POLICY_VERSION,
        "shapes": shape_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    arguments.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
