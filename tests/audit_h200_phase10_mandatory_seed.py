#!/usr/bin/env python3
"""Measure the certified mandatory-word hypothesis on H200."""

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from array import array
from pathlib import Path

import pyhmmer

from plan7_gpu import SequenceBatch, _native
from plan7_gpu._mandatory_seed import (
    SSVSeedParameters,
    compile_required_gain,
    enumerate_window_seeds,
    score_gains,
    window_seed_threshold,
)
from plan7_gpu.adapter import _pack_profiles, _sequence_native


F1 = 0.02
WORD_LENGTHS = (1, 2, 4, 8, 16, 32)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(result):
    keys = (
        "logical_pair_count",
        "exact_candidate_count",
        "seed_candidate_count",
        "certified_reject_count",
        "false_reject_count",
        "unsupported_pair_count",
        "logical_cell_count",
        "survivor_exact_cell_count",
        "temporary_device_bytes",
        "exact_generation_milliseconds",
        "seed_kernel_milliseconds",
        "analysis_milliseconds",
    )
    output = {key: result[key] for key in keys}
    output["certified_pair_fraction"] = (
        result["certified_reject_count"] / result["logical_pair_count"]
    )
    output["certified_cell_fraction"] = 1.0 - (
        result["survivor_exact_cell_count"] / result["logical_cell_count"]
    )
    output["seed_to_exact_time_ratio"] = (
        result["seed_kernel_milliseconds"]
        / result["exact_generation_milliseconds"]
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pfam-base", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-limit", type=int, default=0)
    parser.add_argument("--dictionary-sample-size", type=int, default=16)
    return parser.parse_args()


def dictionary_probe(profiles, packed, targets, word_length, sample_size):
    if not profiles or sample_size <= 0:
        return {"sample_size": 0}
    sample_size = min(sample_size, len(profiles))
    indices = sorted(
        {index * (len(profiles) - 1) // max(1, sample_size - 1)
         for index in range(sample_size)}
    )
    target_lengths = sorted(len(target) for target in targets)
    representative_length = target_lengths[len(target_lengths) // 2]
    started = time.monotonic()
    records = []
    total_entries = 0
    total_unique_words = 0
    total_nodes = 0
    minimum_payload_bytes = 0
    for index in indices:
        profile = profiles[index]
        mu, lambda_ = profile.evalue_parameters.as_vector()[:2]
        mode, cutoff = _native.f1_cutoff(mu, lambda_, F1)
        tjb = _native.tjb_for_lengths(
            profile.scale_b, array("Q", [representative_length])
        )[0]
        required_gain = compile_required_gain(
            SSVSeedParameters(
                profile.tbm,
                profile.tec,
                profile.base_b,
                profile.bias_b,
                profile.scale_b,
                mode,
                0.0 if cutoff is None else cutoff,
            ),
            representative_length,
            tjb,
            _native.f1_cutoff_decision,
            status_ok=_native.STATUS_OK,
            cpu_required=_native.F1_CPU_REQUIRED,
        )
        if required_gain is None:
            records.append(
                {"profile_index": index, "supported": False, "truncated": True}
            )
            continue
        threshold = window_seed_threshold(
            required_gain,
            profile.M,
            representative_length,
            word_length,
        )
        offset = packed.score_offsets[index]
        count = packed.score_counts[index]
        gains = score_gains(
            memoryview(packed.scores)[offset : offset + count],
            profile.M,
            packed.score_strides[index],
            20,
        )
        plan = enumerate_window_seeds(
            gains,
            threshold,
            word_length,
            alphabet_size=20,
            association_limit=100_000,
            node_limit=1_000_000,
        )
        entry_bytes = sum(len(entry.word) + 16 for entry in plan.entries)
        total_entries += len(plan.entries)
        total_unique_words += len(plan.unique_words)
        total_nodes += plan.enumerated_nodes
        minimum_payload_bytes += entry_bytes
        records.append(
            {
                "profile_index": index,
                "model_length": profile.M,
                "required_gain": required_gain,
                "word_threshold": threshold,
                "entry_count": len(plan.entries),
                "unique_word_count": len(plan.unique_words),
                "enumerated_nodes": plan.enumerated_nodes,
                "truncated": plan.truncated,
                "minimum_payload_bytes": entry_bytes,
            }
        )
    return {
        "sample_size": len(indices),
        "representative_target_length": representative_length,
        "canonical_alphabet_size": 20,
        "association_limit_per_profile": 100_000,
        "node_limit_per_profile": 1_000_000,
        "truncated_profile_count": sum(row["truncated"] for row in records),
        "entry_count_lower_bound": total_entries,
        "unique_word_count_sum_lower_bound": total_unique_words,
        "enumerated_node_count": total_nodes,
        "minimum_payload_bytes_lower_bound": minimum_payload_bytes,
        "build_wall_seconds": time.monotonic() - started,
        "profiles": records,
    }


def main():
    args = parse_args()
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != "sm90_h200":
        raise RuntimeError("Phase 10 evaluator requires attested sm90/H200")

    started = time.monotonic()
    with pyhmmer.plan7.HMMPressedFile(args.pfam_base) as stream:
        profiles = (
            [next(stream) for _ in range(args.profile_limit)]
            if args.profile_limit
            else list(stream)
        )
    with pyhmmer.easel.SequenceFile(
        args.targets, digital=True, alphabet=profiles[0].alphabet
    ) as stream:
        targets = list(stream)
    packed = _pack_profiles(profiles)
    parameters = [profile.evalue_parameters.as_vector() for profile in profiles]
    m_mu = array("f", (value[0] for value in parameters))
    m_lambda = array("f", (value[1] for value in parameters))

    evaluations = {}
    profile_rows = {}
    exact_count = None
    memory_before = _native.device_memory_info()
    with SequenceBatch(targets) as batch:
        native = _sequence_native(batch)
        for word_length in WORD_LENGTHS:
            result = native.evaluate_mandatory_seed_many_raw(
                *packed,
                m_mu,
                m_lambda,
                F1,
                word_length,
                profiles[0].alphabet.Kp,
            )
            if result["false_reject_count"] != 0:
                raise AssertionError(
                    f"word length {word_length} produced a false reject"
                )
            if exact_count is None:
                exact_count = result["exact_candidate_count"]
            elif exact_count != result["exact_candidate_count"]:
                raise AssertionError("exact F1 census changed between seed arms")
            if (
                result["certified_reject_count"]
                + result["seed_candidate_count"]
                != result["logical_pair_count"]
            ):
                raise AssertionError("seed candidate partition does not reconcile")
            evaluations[str(word_length)] = compact(result)
            profile_rows[str(word_length)] = result["profiles"]
        batch_memory = native.memory_snapshot
    memory_after = _native.device_memory_info()

    best_word_length = max(
        WORD_LENGTHS,
        key=lambda value: evaluations[str(value)]["certified_cell_fraction"],
    )
    dictionary = dictionary_probe(
        profiles,
        packed,
        targets,
        best_word_length,
        args.dictionary_sample_size,
    )
    symbol_counts = [0] * profiles[0].alphabet.Kp
    sequence_symbol_counts = [0] * profiles[0].alphabet.Kp
    for target in targets:
        seen = set(target.sequence)
        for residue in target.sequence:
            symbol_counts[residue] += 1
        for residue in seen:
            sequence_symbol_counts[residue] += 1

    output = {
        "schema": "plan7_gpu.phase10_mandatory_seed_h200.v1",
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
            "alphabet_symbols": profiles[0].alphabet.symbols,
            "symbol_counts": symbol_counts,
            "sequence_symbol_counts": sequence_symbol_counts,
        },
        "evaluations": evaluations,
        "best_word_length_by_certified_cells": best_word_length,
        "dictionary_probe": dictionary,
        "memory": {
            "device_before": memory_before,
            "device_after": memory_after,
            "batch_persistent": batch_memory,
            "packed_profile_host_score_bytes": len(packed.scores),
            "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "profile_rows": profile_rows,
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "status",
                    "input",
                    "evaluations",
                    "best_word_length_by_certified_cells",
                    "dictionary_probe",
                    "memory",
                    "wall_seconds",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
