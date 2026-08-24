"""Focused H200 old-vs-packed oracle for Phase 5 profile-axis SSV."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native, _pipeline
from plan7_gpu.adapter import _candidate_state, _sequence_state


F1 = 0.99
SOURCE_MODELS = (
    "RREFam.hmm",
    "PF02826.hmm",
    "Thioesterase.hmm",
    "KR.hmm",
    "LuxC.hmm",
)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def mutate_copy(hmm: pyhmmer.plan7.HMM, family: int, variant: int):
    result = hmm.copy()
    result.name = f"phase5-{family:02d}-{variant:02d}".encode()
    result.accession = None
    left = variant % 20
    right = (variant * 7 + 3) % 20
    if left == right:
        right = (right + 1) % 20
    for position in range(1, result.M + 1):
        row = result.match_emissions[position]
        row[left], row[right] = row[right], row[left]
    return result


def fixture() -> tuple[list[pyhmmer.plan7.HMM], object]:
    data = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
    hmms = []
    for family, name in enumerate(SOURCE_MODELS):
        with pyhmmer.plan7.HMMFile(data / name) as source:
            template = source.read()
        if template is None:
            raise RuntimeError(f"missing fixture HMM: {name}")
        for variant in range(8):
            hmms.append(mutate_copy(template, family, variant))

    alphabet = hmms[0].alphabet
    amino = "ACDEFGHIKLMNPQRSTVWY"
    sequences = []
    for index in range(65):
        length = 1 + (index * 37) % 511
        text = "".join(
            amino[(index * 11 + position * 7) % len(amino)]
            for position in range(length)
        )
        sequences.append(
            pyhmmer.easel.TextSequence(
                name=f"phase5-target-{index:03d}".encode(), sequence=text
            ).digitize(alphabet)
        )
    return hmms, pyhmmer.easel.DigitalSequenceBlock(alphabet, sequences)


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


def delta(after: dict[str, int], before: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in after}


def audit() -> dict[str, object]:
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != "sm90_h200":
        raise RuntimeError("Phase 5 oracle requires attested sm90/H200")
    hmms, targets = fixture()
    with tempfile.TemporaryDirectory(prefix="phase5-profile-packed-") as temporary:
        base = Path(temporary) / "models"
        pyhmmer.hmmer.hmmpress(hmms, base)
        pairs = load_pressed_profiles(base)
        with (
            ProfileSession(pairs, pack_workers=4) as session,
            session.select(range(len(pairs))) as selection,
            SequenceBatch(targets) as batch,
        ):
            native = _sequence_state(batch).native
            before = dict(native.workspace_statistics)
            seal_factory = _pipeline._seal_postfilter_batch_bound
            _pipeline._seal_postfilter_batch_bound = None
            try:
                os.environ["PLAN7_GPU_SSV_PROFILE_POLICY"] = "scalar"
                scalar = batch.postfilter_selection(selection, F1=F1)
                os.environ.pop("PLAN7_GPU_SSV_PROFILE_POLICY", None)
                after_scalar = dict(native.workspace_statistics)
                packed = batch.postfilter_selection(selection, F1=F1)
                after_packed = dict(native.workspace_statistics)
            finally:
                os.environ.pop("PLAN7_GPU_SSV_PROFILE_POLICY", None)
                _pipeline._seal_postfilter_batch_bound = seal_factory

            scalar_state = _candidate_state(scalar)
            packed_state = _candidate_state(packed)
            scalar_offsets = memoryview(scalar_state.offsets).cast("B").tobytes()
            packed_offsets = memoryview(packed_state.offsets).cast("B").tobytes()
            scalar_records = bytes(scalar_state.postfilter_records)
            packed_records = bytes(packed_state.postfilter_records)
            if scalar_offsets != packed_offsets or scalar_records != packed_records:
                raise AssertionError("packed F1 changed postfilter records or order")

            outputs = bytearray()
            for row in range(len(pairs)):
                expected = hit_bytes(scalar.search(row, pipeline(pairs[0].hmm.alphabet)))
                actual = hit_bytes(packed.search(row, pipeline(pairs[0].hmm.alphabet)))
                if actual != expected:
                    raise AssertionError(f"packed F1 changed HMMER output at row {row}")
                outputs.extend(actual)

            scalar_delta = delta(after_scalar, before)
            packed_delta = delta(after_packed, after_scalar)
            if scalar_delta["f1_profile_packed_run_count"] != 0:
                raise AssertionError("forced scalar reference used packed execution")
            if (
                packed_delta["f1_profile_packed_run_count"] != 1
                or packed_delta["f1_profile_packed_quartet_count"] != 10
                or packed_delta["f1_profile_packed_profile_count"] != 40
                or packed_delta["f1_profile_scalar_profile_count"] != 0
                or packed_delta["f1_profile_packed_score_bytes"] <= 0
            ):
                raise AssertionError(f"packed execution census is wrong: {packed_delta}")

            return {
                "schema": 1,
                "status": "PASS",
                "device": provenance,
                "profile_count": len(pairs),
                "sequence_count": len(targets),
                "model_lengths": [pair.hmm.M for pair in pairs],
                "postfilter_offsets_sha256": sha256(packed_offsets),
                "postfilter_records_sha256": sha256(packed_records),
                "hmm_output_sha256": sha256(bytes(outputs)),
                "scalar_delta": scalar_delta,
                "packed_delta": packed_delta,
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
