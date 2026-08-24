#!/usr/bin/env python3
"""Measure the certified reduced-alphabet F0 hypothesis on H200."""

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import time
from array import array
from pathlib import Path

import pyhmmer

from plan7_gpu import SequenceBatch, _native
from plan7_gpu.adapter import _pack_profiles, _sequence_native


F1 = 0.02
FIXED_GROUPS = {
    4: (b"AGPST", b"DENQHKR", b"ILMV", b"CFWY"),
    6: (b"AGPST", b"DENQ", b"HKR", b"ILMV", b"FWY", b"C"),
    8: (b"AG", b"P", b"ST", b"DENQ", b"HKR", b"ILMV", b"FWY", b"C"),
}
ALIASES = {
    ord("B"): ord("D"),
    ord("J"): ord("I"),
    ord("Z"): ord("E"),
    ord("U"): ord("C"),
    ord("O"): ord("K"),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_partition(symbols, groups):
    classes = tuple(map(set, groups))
    return bytearray(
        next(
            (
                index
                for index, group in enumerate(classes)
                if ALIASES.get(symbol, symbol) in group
            ),
            0,
        )
        for symbol in symbols
    )


def random_partition(alphabet_size, class_count, seed):
    rng = random.Random(seed)
    regular = list(range(20))
    rng.shuffle(regular)
    output = [0] * alphabet_size
    for rank, residue in enumerate(regular):
        output[residue] = rank % class_count
    for residue in range(20, alphabet_size):
        output[residue] = rng.randrange(class_count)
    return bytearray(output)


def compact_result(result):
    keys = (
        "logical_pair_count",
        "exact_candidate_count",
        "coarse_candidate_count",
        "certified_reject_count",
        "false_reject_count",
        "logical_cell_count",
        "survivor_exact_cell_count",
        "coarse_table_bytes",
        "temporary_device_bytes",
        "exact_generation_milliseconds",
        "coarse_table_build_milliseconds",
        "coarse_upload_milliseconds",
        "coarse_kernel_milliseconds",
        "analysis_milliseconds",
    )
    output = {key: result[key] for key in keys}
    output["certified_pair_fraction"] = (
        result["certified_reject_count"] / result["logical_pair_count"]
    )
    output["certified_cell_fraction"] = 1.0 - (
        result["survivor_exact_cell_count"] / result["logical_cell_count"]
    )
    output["f0_to_exact_time_ratio"] = (
        result["coarse_kernel_milliseconds"]
        / result["exact_generation_milliseconds"]
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfam-base", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-limit", type=int, default=0)
    parser.add_argument("--codebook-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != "sm90_h200":
        raise RuntimeError("Phase 8 evaluator requires attested sm90/H200")
    if args.codebook_size < 1 or args.codebook_size > 32:
        raise ValueError("codebook size must be in [1, 32]")

    started = time.monotonic()
    with pyhmmer.plan7.HMMPressedFile(args.pfam_base) as stream:
        if args.profile_limit:
            profiles = [next(stream) for _ in range(args.profile_limit)]
        else:
            profiles = list(stream)
    with pyhmmer.easel.SequenceFile(
        args.targets, digital=True, alphabet=profiles[0].alphabet
    ) as stream:
        targets = list(stream)
    packed = _pack_profiles(profiles)
    parameters = [profile.evalue_parameters.as_vector() for profile in profiles]
    m_mu = array("f", (value[0] for value in parameters))
    m_lambda = array("f", (value[1] for value in parameters))
    symbols = profiles[0].alphabet.symbols.encode()

    fixed = {}
    best_rows = None
    codebook_kernel_ms = []
    codebook_hashes = []
    exact_count = None
    memory_before = _native.device_memory_info()
    with SequenceBatch(targets) as batch:
        native = _sequence_native(batch)
        identity = native.evaluate_f0_many_raw(
            *packed,
            m_mu,
            m_lambda,
            F1,
            bytearray(range(len(symbols))),
            len(symbols),
        )
        if identity["false_reject_count"] != 0 or (
            identity["coarse_candidate_count"]
            != identity["exact_candidate_count"]
        ):
            raise AssertionError("singleton F0 partition differs from exact F1")
        exact_count = identity["exact_candidate_count"]

        fixed_results = {}
        for class_count, groups in FIXED_GROUPS.items():
            partition = fixed_partition(symbols, groups)
            result = native.evaluate_f0_many_raw(
                *packed, m_mu, m_lambda, F1, partition, class_count
            )
            if result["false_reject_count"] != 0:
                raise AssertionError(f"{class_count}-class F0 had a false reject")
            if result["exact_candidate_count"] != exact_count:
                raise AssertionError("exact F1 census changed between partitions")
            fixed[class_count] = compact_result(result)
            fixed_results[class_count] = result

        partitions = [fixed_partition(symbols, FIXED_GROUPS[8])]
        partitions.extend(
            random_partition(len(symbols), 8, 0xF000 + index)
            for index in range(1, args.codebook_size)
        )
        for index, partition in enumerate(partitions):
            if index == 0:
                result = fixed_results[8]
            else:
                result = native.evaluate_f0_many_raw(
                    *packed, m_mu, m_lambda, F1, partition, 8
                )
            if result["false_reject_count"] != 0:
                raise AssertionError(f"codebook partition {index} had a false reject")
            codebook_hashes.append(hashlib.sha256(partition).hexdigest())
            codebook_kernel_ms.append(result["coarse_kernel_milliseconds"])
            rows = result["profiles"]
            if best_rows is None:
                best_rows = [dict(row, partition_index=index) for row in rows]
            else:
                for profile_index, row in enumerate(rows):
                    if row["survivor_exact_cell_count"] < best_rows[profile_index][
                        "survivor_exact_cell_count"
                    ]:
                        best_rows[profile_index] = dict(
                            row, partition_index=index
                        )
        batch_memory = native.memory_snapshot
    memory_after = _native.device_memory_info()

    assert best_rows is not None
    logical_pairs = sum(row["logical_pair_count"] for row in best_rows)
    coarse_candidates = sum(row["coarse_candidate_count"] for row in best_rows)
    logical_cells = sum(row["logical_cell_count"] for row in best_rows)
    survivor_cells = sum(row["survivor_exact_cell_count"] for row in best_rows)
    profile_records = []
    for index, (profile, row) in enumerate(zip(profiles, best_rows, strict=True)):
        profile_records.append(
            {
                "profile_index": index,
                "name": profile.name.decode("utf-8", "replace"),
                "model_length": profile.M,
                "exact_candidate_count": row["exact_candidate_count"],
                "coarse_candidate_count": row["coarse_candidate_count"],
                "certified_reject_count": row["certified_reject_count"],
                "logical_cell_count": row["logical_cell_count"],
                "survivor_exact_cell_count": row["survivor_exact_cell_count"],
                "partition_index": row["partition_index"],
            }
        )

    output = {
        "schema": "plan7_gpu.phase8_f0_h200.v1",
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
            "pfam_base": str(args.pfam_base.resolve()),
            "profile_count": len(profiles),
            "targets": str(args.targets.resolve()),
            "target_sha256": sha256_file(args.targets),
            "target_count": len(targets),
            "target_residues": sum(len(target) for target in targets),
        },
        "identity_oracle": compact_result(identity),
        "fixed_partitions": {str(key): value for key, value in fixed.items()},
        "codebook": {
            "size": len(partitions),
            "partition_sha256": codebook_hashes,
            "logical_pair_count": logical_pairs,
            "exact_candidate_count": sum(
                row["exact_candidate_count"] for row in best_rows
            ),
            "coarse_candidate_count": coarse_candidates,
            "certified_reject_count": logical_pairs - coarse_candidates,
            "certified_pair_fraction": 1.0 - coarse_candidates / logical_pairs,
            "logical_cell_count": logical_cells,
            "survivor_exact_cell_count": survivor_cells,
            "certified_cell_fraction": 1.0 - survivor_cells / logical_cells,
            "coarse_kernel_milliseconds_min": min(codebook_kernel_ms),
            "coarse_kernel_milliseconds_max": max(codebook_kernel_ms),
            "coarse_kernel_milliseconds_mean": sum(codebook_kernel_ms)
            / len(codebook_kernel_ms),
            "profiles_with_any_reject": sum(
                row["certified_reject_count"] > 0 for row in best_rows
            ),
            "profiles_with_at_least_one_percent_reject": sum(
                row["certified_reject_count"] * 100
                >= row["logical_pair_count"]
                for row in best_rows
            ),
        },
        "memory": {
            "device_before": memory_before,
            "device_after": memory_after,
            "batch_persistent": batch_memory,
            "packed_profile_host_score_bytes": len(packed.scores),
            "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "profile_records": profile_records,
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, args.output)
    print(json.dumps({key: output[key] for key in ("status", "input", "fixed_partitions", "codebook", "memory", "wall_seconds")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
