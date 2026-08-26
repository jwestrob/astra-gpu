#!/usr/bin/env python3
"""Focused exact/timing oracle for the experimental sequence-major bias path."""

import argparse
import hashlib
import json
import statistics
import subprocess
from array import array
from pathlib import Path

import pyhmmer

from plan7_gpu import _native
from plan7_gpu.adapter import _pack_profiles


ROOT = Path(__file__).resolve().parents[1]
PFAM = Path("/groups/banfield/users/jwestrob/.config/Astra/PFAM/PFAM")
FASTA = Path(
    "/groups/banfield/projects/environmental/sr/srvp2020/Jacob/hmmer_gpu/"
    "results/datasets/PLM2_5.first1000.faa"
)


def pack_bias(background, profiles, f1):
    packed = bytearray()
    mu = array("f")
    lambda_ = array("f")
    for profile in profiles:
        profile_mu, profile_lambda = profile.evalue_parameters.as_vector()[:2]
        mode, cutoff = _native.f1_cutoff(profile_mu, profile_lambda, f1)
        mu.append(profile_mu)
        lambda_.append(profile_lambda)
        packed.extend(
            _native.pack_bias_profile_raw(
                memoryview(background.residue_frequencies),
                memoryview(profile.compositions),
                profile.M,
                profile.scale_b,
                mode,
                float("nan") if cutoff is None else cutoff,
            )
        )
    return packed, mu, lambda_


def run_case(sequences, profiles, profile_count, f1):
    selected = profiles[:profile_count]
    packed_ssv = _pack_profiles(selected)
    background = pyhmmer.plan7.Background(selected[0].alphabet)
    packed_bias, mu, lambda_ = pack_bias(background, selected, f1)
    residues = bytearray()
    residue_offsets = array("Q", [0])
    for sequence in sequences:
        residues.extend(memoryview(sequence.sequence).cast("B"))
        residue_offsets.append(len(residues))

    retained_stats = []
    sequence_stats = []
    reference = None
    with _native.SequenceBatch(
        residues, residue_offsets, selected[0].alphabet.Kp
    ) as batch:
        for iteration in range(10):
            sequence_major = bool(iteration & 1)
            records, offsets, stats = (
                batch.bias_candidates_many_experimental_csr_raw(
                    *packed_ssv,
                    mu,
                    lambda_,
                    f1,
                    packed_bias,
                    sequence_major,
                )
            )
            payload = bytes(records) + array("Q", offsets).tobytes()
            if reference is None:
                reference = payload
            elif payload != reference:
                raise RuntimeError(
                    f"bias bytes differ for profiles={profile_count}, f1={f1}"
                )
            (sequence_stats if sequence_major else retained_stats).append(stats)

    retained_warm = retained_stats[1:]
    sequence_warm = sequence_stats[1:]
    retained_ms = statistics.median(
        row["kernel_milliseconds"] for row in retained_warm
    )
    sequence_kernel_ms = statistics.median(
        row["kernel_milliseconds"] for row in sequence_warm
    )
    grouping_ms = statistics.median(
        row["grouping_milliseconds"] for row in sequence_warm
    )
    sequence_total_ms = statistics.median(
        row["total_milliseconds"] for row in sequence_warm
    )
    return {
        "profile_count": profile_count,
        "sequence_count": len(sequences),
        "f1": f1,
        "candidate_count": retained_stats[-1]["candidate_count"],
        "candidates_per_sequence": (
            retained_stats[-1]["candidate_count"] / len(sequences)
        ),
        "payload_sha256": hashlib.sha256(reference).hexdigest(),
        "payload_bytes": len(reference),
        "retained_kernel_milliseconds": retained_ms,
        "sequence_grouping_milliseconds": grouping_ms,
        "sequence_kernel_milliseconds": sequence_kernel_ms,
        "sequence_total_milliseconds": sequence_total_ms,
        "speedup": retained_ms / sequence_total_ms,
        "temporary_device_bytes": sequence_stats[-1]["temporary_device_bytes"],
        "exact": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    args = parser.parse_args()

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if revision != args.expected_revision:
        raise RuntimeError(f"source revision {revision} != {args.expected_revision}")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("source worktree is dirty")
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"]:
        raise RuntimeError(provenance["reason"])
    if provenance["cuda"]["compute_capability"] != [9, 0]:
        raise RuntimeError("focused oracle did not run on sm90")
    if "H200" not in provenance["cuda"]["name"]:
        raise RuntimeError("focused oracle did not run on H200")

    with pyhmmer.plan7.HMMPressedFile(PFAM) as pressed:
        profiles = [next(pressed) for _ in range(384)]
    with pyhmmer.easel.SequenceFile(
        FASTA, digital=True, alphabet=profiles[0].alphabet
    ) as sequence_file:
        sequences = list(sequence_file)

    cases = [
        run_case(sequences, profiles, 1, 0.02),
        run_case(sequences, profiles, 16, 0.02),
        run_case(sequences, profiles, 64, 0.02),
        run_case(sequences, profiles, 384, 0.02),
        run_case(sequences, profiles, 384, 1.0),
        run_case(sequences[:16], profiles, 384, 1.0),
    ]
    result = {
        "status": "PASS",
        "revision": revision,
        "provenance": provenance,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
