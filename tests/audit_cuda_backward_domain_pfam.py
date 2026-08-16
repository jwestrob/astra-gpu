"""Matched PFAM x first-1000 audit for CUDA Backward/domain decoding."""

import argparse
import json
import struct
import time
from array import array
from pathlib import Path

import numpy as np
import pyhmmer

from plan7_gpu import _native


RESULT_FORMAT = "=IIffIIIBBBB"


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def load_census(path):
    with path.open() as stream:
        for line in stream:
            if line.startswith("serial\t"):
                header = line.rstrip().split("\t")
                break
        return [
            dict(zip(header, line.rstrip().split("\t")))
            for line in stream
            if line.strip()
        ]


def quantiles(values):
    values = np.concatenate(values) if values else np.zeros(0, np.float32)
    if values.size == 0:
        return {"count": 0, "max": 0.0, "p99": 0.0, "p50": 0.0}
    return {
        "count": int(values.size),
        "max": float(np.max(values)),
        "p99": float(np.quantile(values, 0.99)),
        "p50": float(np.quantile(values, 0.50)),
    }


def ordered_float_bits(values):
    bits = values.view(np.uint32)
    return np.where(
        (bits & np.uint32(0x80000000)) != 0,
        np.bitwise_not(bits),
        bits | np.uint32(0x80000000),
    ).astype(np.uint64)


def replay_regions(values, rt1=np.float32(0.25), rt2=np.float32(0.10),
                   rt3=np.float32(0.20)):
    posterior = values.reshape((-1, 3))
    bocc = posterior[1:, 0] - posterior[:-1, 0]
    eocc = posterior[1:, 1] - posterior[:-1, 1]
    mocc = posterior[1:, 2]
    left = mocc - bocc
    right = mocc - eocc
    events = {"rt1": {}, "start_rt2": {}, "end_rt2": {}, "rt3": {}}
    regions = []
    begin = None
    triggered = False
    for offset in range(len(mocc)):
        position = offset + 1
        if not triggered:
            events["start_rt2"][position] = bool(left[offset] < rt2)
            events["rt1"][position] = bool(mocc[offset] >= rt1)
            if left[offset] < rt2 or begin is None:
                begin = position
            if mocc[offset] >= rt1:
                triggered = True
        else:
            events["end_rt2"][position] = bool(right[offset] < rt2)
            if right[offset] >= rt2:
                continue
            maximum = np.float32(-1.0)
            for z in range(begin, position + 1):
                left_expected = np.float32(
                    posterior[z, 1] - posterior[begin - 1, 1]
                )
                right_expected = np.float32(
                    posterior[position, 0] - posterior[z - 1, 0]
                )
                maximum = np.maximum(
                    maximum, np.minimum(left_expected, right_expected)
                )
            events["rt3"][(begin, position)] = bool(maximum >= rt3)
            regions.append((begin, position, maximum))
            begin = None
            triggered = False
    return {
        "bocc": bocc,
        "eocc": eocc,
        "mocc": mocc,
        "left": left,
        "right": right,
        "events": events,
        "regions": regions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--hmm", type=Path, required=True)
    parser.add_argument("--guard", type=float, default=2.0e-4)
    args = parser.parse_args()
    started = time.monotonic()
    rows = load_census(args.census)

    alphabet = pyhmmer.easel.Alphabet.amino()
    sequences = []
    with pyhmmer.easel.SequenceFile(
        args.fasta, digital=True, alphabet=alphabet
    ) as sequence_file:
        sequences.extend(sequence_file)
    sequence_indexes = {text(sequence.name): i for i, sequence in enumerate(sequences)}
    assert len(sequence_indexes) == len(sequences)

    profile_names = []
    previous = None
    for row in rows:
        if row["profile_name"] != previous:
            profile_names.append(row["profile_name"])
            previous = row["profile_name"]
    wanted = set(profile_names)
    profiles_by_name = {}
    with pyhmmer.plan7.HMMFile(args.hmm) as hmm_file:
        for hmm in hmm_file:
            name = text(hmm.name)
            if name not in wanted:
                continue
            background = pyhmmer.plan7.Background(hmm.alphabet)
            profiles_by_name[name] = hmm.to_profile(
                background, L=400
            ).to_optimized()
    assert set(profiles_by_name) == wanted
    profiles = [profiles_by_name[name] for name in profile_names]

    residues = bytearray()
    residue_offsets = array("Q", [0])
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        residue_offsets.append(len(residues))

    profile_indexes = []
    candidate_indexes = []
    candidate_offsets = array("Q", [0])
    row_index = 0
    for profile_index, profile_name in enumerate(profile_names):
        while row_index < len(rows) and rows[row_index]["profile_name"] == profile_name:
            profile_indexes.append(profile_index)
            candidate_indexes.append(sequence_indexes[rows[row_index]["sequence_name"]])
            assert profiles[profile_index].M == int(rows[row_index]["M"])
            row_index += 1
        candidate_offsets.append(row_index)
    assert row_index == len(rows)

    batch = _native.SequenceBatch(residues, residue_offsets, alphabet.Kp)
    resident = _native.ForwardProfiles(profiles)
    try:
        forward_records, forward_offsets, specials, forward_stats = (
            batch.forward_candidates_many_raw(
                candidate_offsets,
                array("I", candidate_indexes),
                array("f", [0.0]) * len(rows),
                1.0,
                profiles,
                resident,
            )
        )
        forward_results = list(struct.iter_unpack("=IfBBH", forward_records))
        assert all(
            result[3] == _native.FORWARD_DEFINITE_PASS
            for result in forward_results
        )
        provenance = forward_stats["_provenance"]

        gpu_records, gpu_offsets, gpu_flat, gpu_region_offsets, gpu_regions, gpu_stats = (
            batch.backward_domain_many_raw(
                array("I", profile_indexes),
                array("I", candidate_indexes),
                forward_offsets,
                specials,
                resident,
                provenance,
                guard_band=0.0,
            )
        )
        gpu_results = list(struct.iter_unpack(RESULT_FORMAT, gpu_records))

        absolute_errors = []
        relative_errors = []
        ulp_errors = []
        score_errors = []
        expected_errors = []
        derived_errors = {
            name: [] for name in ("bocc", "eocc", "mocc", "left", "right", "rt3")
        }
        threshold_crossings = {
            name: 0 for name in ("rt1", "start_rt2", "end_rt2", "rt3")
        }
        decision_mismatches = 0
        outside_guard_mismatches = 0
        region_journal_mismatches = 0
        replay_boundary_mismatches = 0
        fallback_boundary_mismatches = 0
        census_region_mismatches = 0
        census_cluster_mismatches = 0
        status_mismatches = 0
        own_scale_mismatches = 0
        cpu_boundaries_by_candidate = []
        for candidate, (profile_index, sequence_index) in enumerate(
            zip(profile_indexes, candidate_indexes)
        ):
            residue_begin = residue_offsets[sequence_index]
            residue_end = residue_offsets[sequence_index + 1]
            cpu_record, cpu_flat, cpu_regions = _native.backward_domain_cpu_oracle_raw(
                profiles[profile_index],
                memoryview(residues)[residue_begin:residue_end],
                specials[forward_offsets[candidate] : forward_offsets[candidate + 1]],
                guard_band=0.0,
            )
            cpu_result = struct.unpack(RESULT_FORMAT, cpu_record)
            gpu_result = gpu_results[candidate]
            gpu_values = np.asarray(
                gpu_flat[gpu_offsets[candidate] * 3 : gpu_offsets[candidate + 1] * 3]
            )
            cpu_values = np.asarray(cpu_flat)
            difference = np.abs(gpu_values - cpu_values)
            absolute_errors.append(difference)
            relative_errors.append(
                difference / np.maximum(np.abs(cpu_values), np.finfo(np.float32).tiny)
            )
            ulp_errors.append(
                np.abs(
                    ordered_float_bits(gpu_values).astype(np.int64)
                    - ordered_float_bits(cpu_values).astype(np.int64)
                ).astype(np.float32)
            )
            score_errors.append(abs(gpu_result[2] - cpu_result[2]))
            expected_errors.append(abs(gpu_result[3] - cpu_result[3]))
            gpu_replay = replay_regions(gpu_values)
            cpu_replay = replay_regions(cpu_values)
            for name in ("bocc", "eocc", "mocc", "left", "right"):
                derived_errors[name].append(
                    np.abs(gpu_replay[name] - cpu_replay[name])
                )
            gpu_boundaries = [region[:2] for region in gpu_replay["regions"]]
            cpu_boundaries = [region[:2] for region in cpu_replay["regions"]]
            cpu_boundaries_by_candidate.append(cpu_boundaries)
            if gpu_boundaries != cpu_boundaries:
                replay_boundary_mismatches += 1
                if gpu_result[8] == _native.BACKWARD_DOMAIN_CPU_REQUIRED:
                    fallback_boundary_mismatches += 1
            elif gpu_replay["regions"]:
                derived_errors["rt3"].append(np.abs(np.asarray(
                    [region[2] for region in gpu_replay["regions"]],
                    dtype=np.float32,
                ) - np.asarray(
                    [region[2] for region in cpu_replay["regions"]],
                    dtype=np.float32,
                )))
            for name in threshold_crossings:
                gpu_events = gpu_replay["events"][name]
                cpu_events = cpu_replay["events"][name]
                for key in set(gpu_events) | set(cpu_events):
                    threshold_crossings[name] += (
                        gpu_events.get(key) != cpu_events.get(key)
                    )
            if gpu_result[5:7] != cpu_result[5:7]:
                decision_mismatches += 1
                minimum_margin = min(
                    float(rows[candidate][field])
                    for field in (
                        "min_rt1_margin",
                        "min_start_rt2_margin",
                        "min_end_rt2_margin",
                        "min_rt3_margin",
                    )
                )
                if minimum_margin > args.guard:
                    outside_guard_mismatches += 1
            gpu_region_slice = gpu_regions[
                gpu_region_offsets[candidate] * 2 :
                gpu_region_offsets[candidate + 1] * 2
            ]
            if (
                gpu_result[8] == _native.BACKWARD_DOMAIN_SIMPLE
                and list(gpu_region_slice) != list(cpu_regions)
            ) or (
                gpu_result[8] != _native.BACKWARD_DOMAIN_SIMPLE
                and len(gpu_region_slice) != 0
            ):
                region_journal_mismatches += 1
            census_region_mismatches += (
                gpu_result[5] != int(rows[candidate]["replay_regions"])
            )
            census_cluster_mismatches += (
                gpu_result[6] != int(rows[candidate]["replay_clustered"])
            )
            status_mismatches += gpu_result[7] != cpu_result[7]
            own_scale_mismatches += gpu_result[9] != cpu_result[9]

        (
            guarded_records,
            guarded_posterior_offsets,
            guarded_posteriors,
            guarded_region_offsets,
            guarded_regions,
            guarded_stats,
        ) = batch.backward_domain_many_raw(
                array("I", profile_indexes),
                array("I", candidate_indexes),
                forward_offsets,
                specials,
                resident,
                provenance,
                guard_band=args.guard,
                posterior_byte_budget=0,
            )
        guarded_results = list(struct.iter_unpack(RESULT_FORMAT, guarded_records))
        assert len(guarded_posteriors) == 0
        assert guarded_stats["posterior_bytes"] == 0
        assert all(offset == 0 for offset in guarded_posterior_offsets)
        expected_guard_cpu = 0
        guard_route_mismatches = 0
        guarded_journal_mismatches = 0
        for candidate, (row, result) in enumerate(zip(rows, guarded_results)):
            ambiguous = any(
                float(row[field]) <= args.guard
                for field in (
                    "min_rt1_margin",
                    "min_start_rt2_margin",
                    "min_end_rt2_margin",
                    "min_rt3_margin",
                )
            )
            cpu_required = ambiguous or int(row["replay_clustered"]) != 0
            expected_guard_cpu += cpu_required
            guard_route_mismatches += (
                (result[8] == _native.BACKWARD_DOMAIN_CPU_REQUIRED)
                != cpu_required
            )
            begin = guarded_region_offsets[candidate] * 2
            end = guarded_region_offsets[candidate + 1] * 2
            observed = list(zip(
                guarded_regions[begin:end:2],
                guarded_regions[begin + 1:end:2],
            ))
            expected = (
                cpu_boundaries_by_candidate[candidate]
                if result[8] == _native.BACKWARD_DOMAIN_SIMPLE
                else []
            )
            guarded_journal_mismatches += observed != expected

        report = {
            "rows": len(rows),
            "profiles": len(profiles),
            "sequences": len(sequences),
            "posterior_absolute": quantiles(absolute_errors),
            "posterior_relative": quantiles(relative_errors),
            "posterior_ulp": quantiles(ulp_errors),
            "score_absolute": quantiles([np.asarray(score_errors)]),
            "nexpected_absolute": quantiles([np.asarray(expected_errors)]),
            "derived_absolute": {
                name: quantiles(values)
                for name, values in derived_errors.items()
            },
            "threshold_crossings": threshold_crossings,
            "decision_mismatches": decision_mismatches,
            "outside_guard_mismatches": outside_guard_mismatches,
            "region_journal_mismatches": region_journal_mismatches,
            "replay_boundary_mismatches": replay_boundary_mismatches,
            "fallback_boundary_mismatches": fallback_boundary_mismatches,
            "census_region_mismatches": census_region_mismatches,
            "census_cluster_mismatches": census_cluster_mismatches,
            "status_mismatches": status_mismatches,
            "own_scale_mismatches": own_scale_mismatches,
            "guard": args.guard,
            "guard_expected_cpu": expected_guard_cpu,
            "guard_observed_cpu": guarded_stats["cpu_required_count"],
            "guard_observed_device": guarded_stats["device_result_count"],
            "guard_route_mismatches": guard_route_mismatches,
            "guarded_posterior_bytes": guarded_stats["posterior_bytes"],
            "guarded_posterior_values": len(guarded_posteriors),
            "guarded_journal_mismatches": guarded_journal_mismatches,
            "gpu_kernel_ms": gpu_stats["kernel_ms"],
            "gpu_total_ms": gpu_stats["total_ms"],
            "elapsed_seconds": time.monotonic() - started,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        resident.close()
        batch.close()


if __name__ == "__main__":
    main()
