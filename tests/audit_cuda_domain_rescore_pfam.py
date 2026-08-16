"""Bounded PFAM x first-1000 audit for isolated-domain CUDA rescoring."""

import argparse
import hashlib
import json
import math
import struct
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native
from plan7_gpu.adapter import (
    _pair_state,
    _profile_selection_state,
    _sequence_state,
)


RESULT = struct.Struct("=9I5f4BI")
TRACE_STEP = struct.Struct("=IIfB3x")
UPSTREAM_RESULT = struct.Struct("=IIffIIIBBBB")
MARGIN_FIELDS = (
    "min_rt1_margin",
    "min_start_rt2_margin",
    "min_end_rt2_margin",
    "min_rt3_margin",
)


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_census(path):
    header = None
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("serial\t"):
                header = line.rstrip().split("\t")
                continue
            if header is not None and line.strip():
                rows.append(dict(zip(header, line.rstrip().split("\t"))))
    if header is None:
        raise ValueError("census header is missing")
    return rows


def quantiles(values):
    if not values:
        return {"count": 0, "max": 0.0, "p99": 0.0, "p50": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "max": float(np.max(array)),
        "p99": float(np.quantile(array, 0.99)),
        "p50": float(np.quantile(array, 0.50)),
    }


def ordered_float_bits(value):
    bits = struct.unpack("=I", struct.pack("=f", float(value)))[0]
    return (~bits & 0xFFFFFFFF) if bits & 0x80000000 else bits | 0x80000000


def ulp_distance(left, right):
    return abs(ordered_float_bits(left) - ordered_float_bits(right))


def selected_profile_names(rows, count):
    names = []
    seen = set()
    for row in rows:
        name = row["profile_name"]
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) == count:
            break
    if len(names) != count:
        raise ValueError(
            f"census contains only {len(names)} profiles, fewer than {count}"
        )
    return names


def read_hmms(path, names):
    wanted = set(names)
    found = {}
    with pyhmmer.plan7.HMMFile(path) as source:
        for hmm in source:
            name = text(hmm.name)
            if name in wanted:
                found[name] = hmm
                if len(found) == len(wanted):
                    break
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"PFAM source is missing profiles: {sorted(missing)!r}")
    return [found[name] for name in names]


def read_sequences(path, count):
    alphabet = pyhmmer.easel.Alphabet.amino()
    sequences = []
    with pyhmmer.easel.SequenceFile(
        path, digital=True, alphabet=alphabet
    ) as source:
        for sequence in source:
            sequences.append(sequence)
            if len(sequences) == count:
                break
    if len(sequences) != count:
        raise ValueError(
            f"FASTA contains only {len(sequences)} sequences, fewer than {count}"
        )
    return alphabet, sequences


def guarded_simple(row, guard):
    return (
        int(row["simple_regions"]) > 0
        and int(row["nclustered"]) == 0
        and all(float(row[field]) > guard for field in MARGIN_FIELDS)
    )


def reconcile_final_domains(
    census_by_key,
    upstream_keys,
    upstream_results,
    results,
    statistics,
    *,
    simple_route,
    device_action,
    cpu_action,
):
    """Reconcile final domains by whole-row ownership, not envelopes."""
    upstream_key_set = set(upstream_keys)
    front_half_cpu_retained = sum(
        int(row["ndom"])
        for key, row in census_by_key.items()
        if key not in upstream_key_set
    )
    simple_rows = {
        row_index
        for row_index, upstream_result in enumerate(upstream_results)
        if upstream_result[8] == simple_route
    }
    upstream_route_cpu_retained = sum(
        int(census_by_key[key]["ndom"])
        for row_index, key in enumerate(upstream_keys)
        if key in census_by_key and row_index not in simple_rows
    )

    actions_by_row = defaultdict(set)
    device_domains_by_row = defaultdict(int)
    for result in results:
        actions_by_row[result[0]].add(result[15])
        if result[15] == device_action:
            device_domains_by_row[result[0]] += 1
    stage_cpu_rows = {
        row_index
        for row_index, actions in actions_by_row.items()
        if cpu_action in actions
    } & simple_rows
    if statistics["global_cpu_fallback_count"]:
        stage_cpu_rows.update(simple_rows)
    device_rows = simple_rows - stage_cpu_rows

    stage_mixed_action_rows = sum(
        actions not in ({device_action}, {cpu_action})
        for row_index, actions in actions_by_row.items()
        if row_index in simple_rows
    )
    stage_cpu_retained_domains = sum(
        int(census_by_key[upstream_keys[row_index]]["ndom"])
        for row_index in stage_cpu_rows
        if upstream_keys[row_index] in census_by_key
    )
    observed_device_domains = sum(
        device_domains_by_row[row_index] for row_index in device_rows
    )
    device_domain_count_mismatches = sum(
        device_domains_by_row[row_index]
        != int(census_by_key[upstream_keys[row_index]]["ndom"])
        for row_index in device_rows
        if upstream_keys[row_index] in census_by_key
    )
    cpu_retained_final_domains = (
        front_half_cpu_retained
        + upstream_route_cpu_retained
        + stage_cpu_retained_domains
    )
    return {
        "front_half_cpu_retained_domains": front_half_cpu_retained,
        "upstream_route_cpu_retained_domains": (
            upstream_route_cpu_retained
        ),
        "stage_cpu_retained_domains": stage_cpu_retained_domains,
        "total_cpu_retained_domains": cpu_retained_final_domains,
        "device_final_domains": observed_device_domains,
        "accounted_final_domains": (
            cpu_retained_final_domains + observed_device_domains
        ),
        "stage_cpu_fallback_rows": len(stage_cpu_rows),
        "stage_mixed_action_rows": stage_mixed_action_rows,
        "device_domain_count_mismatches": device_domain_count_mismatches,
        "device_result_statistic_mismatches": (
            observed_device_domains != statistics["device_result_count"]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--hmm", type=Path, required=True)
    parser.add_argument("--profiles", type=int, default=16)
    parser.add_argument("--sequences", type=int, default=1000)
    parser.add_argument("--f1", type=float, default=0.02)
    parser.add_argument("--f2", type=float, default=0.001)
    parser.add_argument("--f3", type=float, default=1.0e-5)
    parser.add_argument("--guard", type=float, default=2.0e-4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.profiles <= 0 or args.sequences <= 0:
        parser.error("profile and sequence counts must be positive")
    if args.guard < 2.0e-4:
        parser.error("the production guard must be at least 2e-4")

    wall_start = time.monotonic()
    census = load_census(args.census)
    profile_names = selected_profile_names(census, args.profiles)
    profile_name_set = set(profile_names)
    selected_census = [
        row for row in census if row["profile_name"] in profile_name_set
    ]
    alphabet, sequences = read_sequences(args.fasta, args.sequences)
    sequence_names = [text(sequence.name) for sequence in sequences]
    sequence_name_set = set(sequence_names)
    selected_census = [
        row for row in selected_census
        if row["sequence_name"] in sequence_name_set
    ]
    hmms = read_hmms(args.hmm, profile_names)
    if any(hmm.alphabet != alphabet for hmm in hmms):
        raise ValueError("PFAM and FASTA alphabets differ")

    census_rows = len(selected_census)
    census_guarded_simple_rows = sum(
        guarded_simple(row, args.guard) for row in selected_census
    )
    census_guarded_stage_regions = sum(
        int(row["simple_regions"])
        for row in selected_census
        if guarded_simple(row, args.guard)
    )
    expected_final_domains = sum(int(row["ndom"]) for row in selected_census)
    census_by_key = {
        (row["profile_name"], row["sequence_name"]): row
        for row in selected_census
    }
    expected_by_key = {
        (row["profile_name"], row["sequence_name"]): row
        for row in selected_census
        if guarded_simple(row, args.guard)
    }
    if len(census_by_key) != census_rows:
        raise ValueError("selected census contains duplicate row keys")

    preparation_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="plan7-rescore-pfam-") as temporary:
        pressed = Path(temporary) / "PFAM-sample"
        pyhmmer.hmmer.hmmpress(hmms, pressed)
        pairs = load_pressed_profiles(pressed)
        targets = pyhmmer.easel.DigitalSequenceBlock(alphabet, sequences)
        preparation_seconds = time.monotonic() - preparation_start
        with (
            ProfileSession(pairs, pack_workers=1) as session,
            session.select(range(len(pairs))) as selection,
            SequenceBatch(targets) as batch,
        ):
            selection_state = _profile_selection_state(selection)
            sequence_state = _sequence_state(batch)
            stage_start = time.monotonic()
            journal, payload = (
                sequence_state.native
                ._postfilter_forward_domain_selection_sealed(
                    selection_state.native,
                    args.f1,
                    args.f2,
                    args.f3,
                    args.guard,
                    _native.FORWARD_MAX_GATHERED_BYTES,
                    True,
                )
            )
            stage_wall_seconds = time.monotonic() - stage_start
            del journal
            (
                result_bytes,
                trace_offsets,
                trace_bytes,
                null2,
                statistics,
                upstream_payload,
            ) = payload
            results = list(RESULT.iter_unpack(result_bytes))
            traces = list(TRACE_STEP.iter_unpack(trace_bytes))
            trace_offsets = list(trace_offsets)
            null2 = list(null2)
            (
                upstream_result_bytes,
                upstream_region_offsets,
                upstream_region_values,
            ) = upstream_payload
            upstream_results = list(
                UPSTREAM_RESULT.iter_unpack(upstream_result_bytes)
            )
            upstream_region_offsets = list(upstream_region_offsets)
            upstream_region_values = list(upstream_region_values)
            upstream_regions = list(zip(
                upstream_region_values[0::2],
                upstream_region_values[1::2],
                strict=True,
            ))
            if len(upstream_region_offsets) != len(upstream_results) + 1:
                raise RuntimeError("upstream region offsets have wrong length")
            if upstream_region_offsets[-1] != len(upstream_regions):
                raise RuntimeError("upstream region offsets do not span regions")

            upstream_by_key = {}
            upstream_unexpected_keys = 0
            upstream_duplicate_keys = 0
            upstream_route_mismatches = 0
            upstream_keys = []
            expected_owned_simple_rows = 0
            expected_owned_stage_regions = 0
            for row_index, upstream_result in enumerate(upstream_results):
                profile_index = upstream_result[0]
                sequence_index = upstream_result[1]
                if (
                    profile_index >= len(pairs)
                    or sequence_index >= len(sequences)
                ):
                    raise RuntimeError("upstream result index is out of range")
                key = (
                    profile_names[profile_index],
                    sequence_names[sequence_index],
                )
                upstream_keys.append(key)
                if key in upstream_by_key:
                    upstream_duplicate_keys += 1
                upstream_by_key[key] = row_index
                census_row = census_by_key.get(key)
                if census_row is None:
                    upstream_unexpected_keys += 1
                    continue
                ambiguous = any(
                    float(census_row[field]) <= args.guard
                    for field in MARGIN_FIELDS
                )
                if guarded_simple(census_row, args.guard):
                    expected_route = _native.BACKWARD_DOMAIN_SIMPLE
                    expected_owned_simple_rows += 1
                    expected_owned_stage_regions += int(
                        census_row["simple_regions"]
                    )
                elif int(census_row["nclustered"]) != 0 or ambiguous:
                    expected_route = _native.BACKWARD_DOMAIN_CPU_REQUIRED
                else:
                    expected_route = _native.BACKWARD_DOMAIN_NO_REGIONS
                upstream_route_mismatches += upstream_result[8] != expected_route

            score_errors = defaultdict(list)
            score_ulps = defaultdict(list)
            null2_errors = []
            null2_ulps = []
            posterior_errors = []
            posterior_ulps = []
            alignment_coordinate_mismatches = 0
            model_coordinate_mismatches = 0
            trace_length_mismatches = 0
            trace_state_coordinate_mismatches = 0
            nonfinite_values = 0
            envelope_count_mismatches = 0
            envelope_residue_mismatches = 0
            alignment_residue_mismatches = 0
            unexpected_result_keys = 0
            observed_by_key = defaultdict(list)
            observed_indexes_by_row = defaultdict(list)
            cpu_oracle_start = time.monotonic()

            for index, gpu_result in enumerate(results):
                profile_index = gpu_result[1]
                sequence_index = gpu_result[2]
                if (
                    profile_index >= len(pairs)
                    or sequence_index >= len(sequences)
                ):
                    raise RuntimeError("rescore result index is out of range")
                key = (
                    profile_names[profile_index],
                    sequence_names[sequence_index],
                )
                observed_by_key[key].append(gpu_result)
                observed_indexes_by_row[gpu_result[0]].append(index)
                unexpected_result_keys += key not in expected_by_key
                residues = memoryview(
                    sequences[sequence_index].sequence
                ).cast("B")
                cpu_bytes, cpu_null2, cpu_trace_bytes = (
                    _native.domain_rescore_cpu_oracle_raw(
                        _pair_state(pairs[profile_index]).optimized_profile,
                        residues,
                        gpu_result[3],
                        gpu_result[4],
                    )
                )
                cpu_result = RESULT.unpack(cpu_bytes)
                cpu_trace = list(TRACE_STEP.iter_unpack(cpu_trace_bytes))
                begin = trace_offsets[index]
                end = trace_offsets[index + 1]
                gpu_trace = traces[begin:end]
                gpu_null2 = null2[index * 29 : (index + 1) * 29]

                if gpu_result[15] != _native.DOMAIN_RESCORE_DEVICE_RESULT:
                    continue
                alignment_coordinate_mismatches += (
                    gpu_result[5:7] != cpu_result[5:7]
                )
                model_coordinate_mismatches += (
                    gpu_result[7:9] != cpu_result[7:9]
                )
                trace_length_mismatches += len(gpu_trace) != len(cpu_trace)
                trace_state_coordinate_mismatches += (
                    [(step[0], step[1], step[3]) for step in gpu_trace]
                    != [(step[0], step[1], step[3]) for step in cpu_trace]
                )
                for name, gpu, cpu in zip(
                    ("forward", "backward", "oa", "correction", "consistency"),
                    gpu_result[9:14],
                    cpu_result[9:14],
                    strict=True,
                ):
                    nonfinite_values += not (
                        math.isfinite(gpu) and math.isfinite(cpu)
                    )
                    score_errors[name].append(abs(gpu - cpu))
                    score_ulps[name].append(ulp_distance(gpu, cpu))
                for gpu, cpu in zip(gpu_null2, cpu_null2, strict=True):
                    nonfinite_values += not (
                        math.isfinite(gpu) and math.isfinite(cpu)
                    )
                    null2_errors.append(abs(gpu - cpu))
                    null2_ulps.append(ulp_distance(gpu, cpu))
                for gpu, cpu in zip(gpu_trace, cpu_trace):
                    nonfinite_values += not (
                        math.isfinite(gpu[2]) and math.isfinite(cpu[2])
                    )
                    posterior_errors.append(abs(gpu[2] - cpu[2]))
                    posterior_ulps.append(ulp_distance(gpu[2], cpu[2]))

            cpu_oracle_seconds = time.monotonic() - cpu_oracle_start
            upstream_region_journal_mismatches = 0
            non_simple_result_mismatches = 0
            for row_index, upstream_result in enumerate(upstream_results):
                begin = upstream_region_offsets[row_index]
                end = upstream_region_offsets[row_index + 1]
                expected_intervals = upstream_regions[begin:end]
                observed_indexes = observed_indexes_by_row.get(row_index, [])
                if upstream_result[8] == _native.BACKWARD_DOMAIN_SIMPLE:
                    observed_intervals = [
                        (results[index][3], results[index][4])
                        for index in observed_indexes
                    ]
                    identities_match = all(
                        results[index][1] == upstream_result[0]
                        and results[index][2] == upstream_result[1]
                        for index in observed_indexes
                    )
                    if not statistics["global_cpu_fallback_count"]:
                        upstream_region_journal_mismatches += (
                            observed_intervals != expected_intervals
                            or not identities_match
                        )
                else:
                    non_simple_result_mismatches += bool(observed_indexes)

            for key, observed in observed_by_key.items():
                row = expected_by_key.get(key)
                if row is None:
                    continue
                envelope_count_mismatches += (
                    len(observed) != int(row["simple_regions"])
                )
                envelope_residue_mismatches += (
                    sum(result[4] - result[3] + 1 for result in observed)
                    != int(row["region_residues"])
                )
                if observed and all(
                    result[15] == _native.DOMAIN_RESCORE_DEVICE_RESULT
                    for result in observed
                ):
                    alignment_residue_mismatches += (
                        sum(result[6] - result[5] + 1 for result in observed)
                        != int(row["alignment_residues"])
                    )

    observed_stage_regions = statistics["region_count"]
    reconciliation = reconcile_final_domains(
        census_by_key,
        upstream_keys,
        upstream_results,
        results,
        statistics,
        simple_route=_native.BACKWARD_DOMAIN_SIMPLE,
        device_action=_native.DOMAIN_RESCORE_DEVICE_RESULT,
        cpu_action=_native.DOMAIN_RESCORE_CPU_REQUIRED,
    )
    report = {
        "inputs": {
            "census": {
                "path": str(args.census.resolve()),
                "sha256": sha256(args.census),
            },
            "fasta": {
                "path": str(args.fasta.resolve()),
                "sha256": sha256(args.fasta),
            },
            "hmm": {
                "path": str(args.hmm.resolve()),
                "sha256": sha256(args.hmm),
            },
            "reference_options": {
                "engine": "PyHMMER 0.12.0 / HMMER 3.4",
                "cpus": 0,
                "bit_cutoffs": "gathering",
                "f1": args.f1,
                "f2": args.f2,
                "f3": args.f3,
            },
        },
        "scope": {
            "profiles": args.profiles,
            "sequences": args.sequences,
            "profile_sequence_pairs": args.profiles * args.sequences,
            "profile_names": profile_names,
            "f1": args.f1,
            "f2": args.f2,
            "f3": args.f3,
            "guard": args.guard,
        },
        "coverage": {
            "stock_upstream_rows": census_rows,
            "device_owned_upstream_rows": statistics["upstream_row_count"],
            "front_half_cpu_retained_rows": (
                census_rows - statistics["upstream_row_count"]
            ),
            "census_guarded_simple_rows": census_guarded_simple_rows,
            "expected_device_owned_simple_rows": expected_owned_simple_rows,
            "observed_simple_rows": statistics["simple_row_count"],
            "census_guarded_stage_regions": census_guarded_stage_regions,
            "expected_device_owned_stage_regions": (
                expected_owned_stage_regions
            ),
            "observed_stage_regions": observed_stage_regions,
            "emitted_rescore_records": len(results),
            "device_rescore_envelopes": statistics["device_result_count"],
            "stage_cpu_fallback_envelopes": statistics["cpu_required_count"],
            "stage_cpu_fallback_rows": reconciliation[
                "stage_cpu_fallback_rows"
            ],
            "front_half_cpu_retained_domains": reconciliation[
                "front_half_cpu_retained_domains"
            ],
            "upstream_route_cpu_retained_domains": reconciliation[
                "upstream_route_cpu_retained_domains"
            ],
            "stage_cpu_retained_domains": reconciliation[
                "stage_cpu_retained_domains"
            ],
            "total_cpu_retained_domains": reconciliation[
                "total_cpu_retained_domains"
            ],
            "device_final_domains": reconciliation[
                "device_final_domains"
            ],
            "expected_final_domains": expected_final_domains,
            "accounted_final_domains": reconciliation[
                "accounted_final_domains"
            ],
            "missed_final_domains": (
                expected_final_domains
                - reconciliation["accounted_final_domains"]
            ),
            "device_final_domain_fraction": (
                reconciliation["device_final_domains"]
                / expected_final_domains
                if expected_final_domains else 0.0
            ),
        },
        "coordinates": {
            "alignment_mismatches": alignment_coordinate_mismatches,
            "model_mismatches": model_coordinate_mismatches,
            "trace_length_mismatches": trace_length_mismatches,
            "trace_state_coordinate_mismatches": (
                trace_state_coordinate_mismatches
            ),
            "unexpected_result_keys": unexpected_result_keys,
            "envelope_count_mismatches": envelope_count_mismatches,
            "envelope_residue_mismatches": envelope_residue_mismatches,
            "alignment_residue_mismatches": alignment_residue_mismatches,
            "upstream_region_journal_mismatches": (
                upstream_region_journal_mismatches
            ),
            "non_simple_result_mismatches": non_simple_result_mismatches,
        },
        "routes": {
            "upstream_payload_count_mismatches": (
                len(upstream_results) != statistics["upstream_row_count"]
            ),
            "upstream_unexpected_keys": upstream_unexpected_keys,
            "upstream_duplicate_keys": upstream_duplicate_keys,
            "upstream_route_mismatches": upstream_route_mismatches,
            "stage_mixed_action_rows": reconciliation[
                "stage_mixed_action_rows"
            ],
            "device_domain_count_mismatches": (
                reconciliation["device_domain_count_mismatches"]
            ),
            "device_result_statistic_mismatches": (
                reconciliation["device_result_statistic_mismatches"]
            ),
        },
        "absolute_error": {
            **{name: quantiles(values) for name, values in score_errors.items()},
            "null2": quantiles(null2_errors),
            "trace_posterior": quantiles(posterior_errors),
        },
        "ulp_error": {
            **{name: quantiles(values) for name, values in score_ulps.items()},
            "null2": quantiles(null2_ulps),
            "trace_posterior": quantiles(posterior_ulps),
        },
        "nonfinite_values": nonfinite_values,
        "statistics": statistics,
        "memory": {
            "dense_workspace_bytes": (
                statistics["forward_matrix_bytes"]
                + statistics["posterior_matrix_bytes"]
                + statistics["special_workspace_bytes"]
                + statistics["trace_workspace_bytes"]
            ),
            "compact_output_bytes": statistics["compact_output_bytes"],
        },
        "timing": {
            "preparation_seconds": preparation_seconds,
            "end_to_end_stage_wall_seconds": stage_wall_seconds,
            "rescore_kernel_ms": statistics["kernel_ms"],
            "rescore_wrapper_total_ms": statistics["total_ms"],
            "cpu_oracle_validation_seconds": cpu_oracle_seconds,
            "audit_wall_seconds": time.monotonic() - wall_start,
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)

    failures = (
        report["coverage"]["expected_device_owned_simple_rows"]
        != report["coverage"]["observed_simple_rows"]
        or report["coverage"]["expected_device_owned_stage_regions"]
        != report["coverage"]["observed_stage_regions"]
        or report["coverage"]["missed_final_domains"] != 0
        or any(report["coordinates"].values())
        or any(report["routes"].values())
        or report["nonfinite_values"] != 0
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
