#!/usr/bin/env python3
"""Exact oracle and isolated timing sweep for Forward subwarp widths.

The production Forward entry point remains fixed at one candidate per warp.
This script uses its private diagnostic selector to force the 1/2/4/8
compile-time kernel variants, first compares complete result and XMX bytes,
then measures kernel-only CUDA event time with every benchmark row rejected at
F3 so gathering does not obscure the recurrence timing.
"""

import argparse
import hashlib
import json
import statistics
from array import array
from pathlib import Path

import pyhmmer

from plan7_gpu import _native


WIDTHS = (1, 2, 4, 8)
AMINO_MOTIF = "ACDEFGHIKLMNPQRSTVWY"
DATA = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"


def load_profile(name, target_length):
    with pyhmmer.plan7.HMMFile(DATA / name) as hmm_file:
        hmm = hmm_file.read()
    if hmm.alphabet != pyhmmer.easel.Alphabet.amino():
        raise RuntimeError(f"{name} is not a protein profile")
    background = pyhmmer.plan7.Background(hmm.alphabet)
    return hmm.to_profile(background, L=target_length).to_optimized()


def encoded_target(alphabet, length):
    text = (AMINO_MOTIF * ((length + len(AMINO_MOTIF) - 1) // len(AMINO_MOTIF)))[
        :length
    ]
    digital = pyhmmer.easel.TextSequence(sequence=text).digitize(alphabet)
    return memoryview(digital.sequence).cast("B").tobytes()


def make_batch(alphabet, lengths):
    encoded = {}
    residues = bytearray()
    offsets = array("Q", [0])
    for length in lengths:
        row = encoded.setdefault(length, encoded_target(alphabet, length))
        residues.extend(row)
        offsets.append(len(residues))
    return _native.SequenceBatch(residues, offsets, alphabet.Kp)


def run_forward(batch, resident, profiles, offsets, indices, filters, f3, width):
    return batch.forward_candidates_many_raw(
        offsets,
        indices,
        filters,
        f3,
        profiles,
        resident,
        _candidates_per_warp=width,
    )


def exact_payload(output):
    records, offsets, specials, stats = output
    payload = records + offsets.tobytes() + specials.tobytes()
    return payload, {
        "row_hash": stats["row_hash"],
        "special_hash": stats["special_hash"],
        "continuation_hash": stats["continuation_hash"],
        "pass_count": stats["pass_count"],
        "special_count": stats["special_count"],
    }


def run_exact_oracle():
    lengths = [
        1,
        3,
        7,
        15,
        31,
        63,
        127,
        255,
        511,
    ] * 8
    profiles = [
        load_profile("RREFam.hmm", max(lengths)),
        load_profile("Thioesterase.hmm", max(lengths)),
        load_profile("LuxC.hmm", max(lengths)),
    ]
    row_counts = (1, 17, 65)
    candidate_offsets = array("Q", [0])
    candidate_indices = array("I")
    for count in row_counts:
        candidate_indices.extend(range(count))
        candidate_offsets.append(len(candidate_indices))
    filters = array("f", [0.0]) * len(candidate_indices)

    with make_batch(profiles[0].alphabet, lengths) as batch:
        with _native.ForwardProfiles(profiles) as resident:
            outputs = {
                width: run_forward(
                    batch,
                    resident,
                    profiles,
                    candidate_offsets,
                    candidate_indices,
                    filters,
                    1.0,
                    width,
                )
                for width in WIDTHS
            }
            automatic = run_forward(
                batch,
                resident,
                profiles,
                candidate_offsets,
                candidate_indices,
                filters,
                1.0,
                0,
            )

    reference_payload, reference_seal = exact_payload(outputs[1])
    variants = {}
    for width in WIDTHS:
        payload, seal = exact_payload(outputs[width])
        if payload != reference_payload:
            raise RuntimeError(
                f"Forward width {width} changed result/offset/XMX bytes"
            )
        if seal != reference_seal:
            raise RuntimeError(f"Forward width {width} changed provenance hashes")
        stats = outputs[width][3]
        variants[str(width)] = {
            "kernel_ms": stats["kernel_ms"],
            "scheduled_warp_count": stats["scheduled_warp_count"],
            "active_lane_slots": stats["active_lane_slots"],
            "issued_lane_slots": stats["issued_lane_slots"],
        }
    automatic_payload, automatic_seal = exact_payload(automatic)
    if automatic_payload != reference_payload or automatic_seal != reference_seal:
        raise RuntimeError("automatic Forward policy changed exact output")
    return {
        "profile_model_lengths": [profile.M for profile in profiles],
        "profile_q": [(profile.M + 3) // 4 for profile in profiles],
        "target_lengths": sorted(set(lengths)),
        "row_candidate_counts": list(row_counts),
        "candidate_count": len(candidate_indices),
        "payload_sha256": hashlib.sha256(reference_payload).hexdigest(),
        "provenance": reference_seal,
        "automatic_policy": {
            key: automatic[3][key]
            for key in (
                "requested_candidates_per_warp",
                "candidates_per_warp",
                "subwarp_policy_reason",
                "multiprocessor_count",
                "l2_cache_bytes",
                "policy_tile_candidate_count",
                "average_work_cells",
                "policy_xmx_workspace_bytes",
                "minimum_cta_count",
                "width1_cta_count",
                "width2_cta_count",
                "width4_cta_count",
            )
        },
        "variants": variants,
    }


def benchmark_case(name, hmm_name, lengths, repeats):
    profile = load_profile(hmm_name, max(lengths))
    candidate_count = len(lengths)
    offsets = array("Q", [0, candidate_count])
    indices = array("I", range(candidate_count))
    # A very large filter score makes every finite Forward row an exact F3
    # reject at f3=0, avoiding survivor gather traffic in this kernel study.
    filters = array("f", [1.0e4]) * candidate_count
    samples = {width: [] for width in WIDTHS}
    policy_samples = []
    counters = {}

    with make_batch(profile.alphabet, lengths) as batch:
        with _native.ForwardProfiles([profile]) as resident:
            reference = run_forward(
                batch, resident, [profile], offsets, indices, filters, 0.0, 1
            )
            reference_payload, reference_seal = exact_payload(reference)

            for width in WIDTHS:
                warm = run_forward(
                    batch,
                    resident,
                    [profile],
                    offsets,
                    indices,
                    filters,
                    0.0,
                    width,
                )
                payload, seal = exact_payload(warm)
                if payload != reference_payload or seal != reference_seal:
                    raise RuntimeError(
                        f"{name}: Forward width {width} failed the exact warm oracle"
                    )
                counters[width] = {
                    key: warm[3][key]
                    for key in (
                        "kernel_launch_count",
                        "scheduled_warp_count",
                        "candidate_subwarp_count",
                        "active_lane_slots",
                        "issued_lane_slots",
                    )
                }

            automatic = run_forward(
                batch, resident, [profile], offsets, indices, filters, 0.0, 0
            )
            payload, seal = exact_payload(automatic)
            if payload != reference_payload or seal != reference_seal:
                raise RuntimeError(f"{name}: automatic policy changed exact output")
            policy = {
                key: automatic[3][key]
                for key in (
                    "subwarp_policy_version",
                    "requested_candidates_per_warp",
                    "candidates_per_warp",
                    "subwarp_policy_reason",
                    "multiprocessor_count",
                    "l2_cache_bytes",
                    "policy_tile_candidate_count",
                    "model_length_sum",
                    "target_length_sum",
                    "average_model_length",
                    "average_target_length",
                    "maximum_model_length",
                    "maximum_target_length",
                    "maximum_candidate_work_cells",
                    "average_work_cells",
                    "short_width4_workspace_limit_bytes",
                    "long_packed_workspace_limit_bytes",
                    "policy_xmx_workspace_bytes",
                    "minimum_cta_count",
                    "width1_cta_count",
                    "width2_cta_count",
                    "width4_cta_count",
                )
            }

            for repeat in range(repeats):
                order = WIDTHS[repeat % len(WIDTHS) :] + WIDTHS[: repeat % len(WIDTHS)]
                for width in order:
                    output = run_forward(
                        batch,
                        resident,
                        [profile],
                        offsets,
                        indices,
                        filters,
                        0.0,
                        width,
                    )
                    payload, seal = exact_payload(output)
                    if payload != reference_payload or seal != reference_seal:
                        raise RuntimeError(
                            f"{name}: Forward width {width} changed exact output"
                        )
                    samples[width].append(output[3]["kernel_ms"])
                automatic = run_forward(
                    batch, resident, [profile], offsets, indices, filters, 0.0, 0
                )
                payload, seal = exact_payload(automatic)
                if payload != reference_payload or seal != reference_seal:
                    raise RuntimeError(
                        f"{name}: automatic policy changed exact timed output"
                    )
                if automatic[3]["candidates_per_warp"] != policy[
                    "candidates_per_warp"
                ]:
                    raise RuntimeError(f"{name}: automatic policy is not deterministic")
                policy_samples.append(automatic[3]["kernel_ms"])

    medians = {width: statistics.median(samples[width]) for width in WIDTHS}
    reference_ms = medians[1]
    policy_median = statistics.median(policy_samples)
    return {
        "name": name,
        "hmm": hmm_name,
        "model_length": profile.M,
        "q": (profile.M + 3) // 4,
        "candidate_count": candidate_count,
        "target_length_min": min(lengths),
        "target_length_max": max(lengths),
        "target_length_mean": sum(lengths) / candidate_count,
        "work_cells": sum(profile.M * length for length in lengths),
        "exact_payload_sha256": hashlib.sha256(reference_payload).hexdigest(),
        "automatic_policy": {
            **policy,
            "kernel_ms_samples": policy_samples,
            "kernel_ms_median": policy_median,
            "speedup_vs_width1": reference_ms / policy_median,
        },
        "variants": {
            str(width): {
                "kernel_ms_samples": samples[width],
                "kernel_ms_median": medians[width],
                "speedup_vs_width1": reference_ms / medians[width],
                **counters[width],
            }
            for width in WIDTHS
        },
    }


def scaled_count(value, scale):
    return max(1, int(round(value * scale)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply benchmark candidate counts (the exact oracle is unchanged)",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be positive")
    if not (arguments.scale > 0.0):
        parser.error("--scale must be positive")
    if _native.device_count() < 1:
        raise RuntimeError("no CUDA device is visible")

    coherent_count = scaled_count(8192, arguments.scale)
    long_count = scaled_count(2048, arguments.scale)
    mixed_count = scaled_count(4096, arguments.scale)
    mixed_pattern = (31, 64, 127, 256, 511, 1024)
    cases = [
        (
            "short_coherent",
            "RREFam.hmm",
            [64] * coherent_count,
        ),
        (
            "medium_coherent",
            "Thioesterase.hmm",
            [256] * coherent_count,
        ),
        (
            "long_coherent",
            "LuxC.hmm",
            [1024] * long_count,
        ),
        (
            "mixed_lengths",
            "KR.hmm",
            [mixed_pattern[index % len(mixed_pattern)] for index in range(mixed_count)],
        ),
    ]
    report = {
        "schema": "plan7-forward-subwarp-microbenchmark-v1",
        "device": _native.bias_environment_provenance()["cuda"],
        "repeats": arguments.repeats,
        "scale": arguments.scale,
        "exact_oracle": run_exact_oracle(),
        "benchmarks": [
            benchmark_case(name, hmm, lengths, arguments.repeats)
            for name, hmm, lengths in cases
        ],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
