"""Focused H200 expanded-vs-length-class oracle for Phase 6."""

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
TARGET_LENGTHS = (1, 17, 64, 127, 511)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def mutate_copy(hmm: pyhmmer.plan7.HMM, family: int, variant: int):
    result = hmm.copy()
    result.name = f"phase6-{family:02d}-{variant:02d}".encode()
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
    for index in range(320):
        length = TARGET_LENGTHS[index % len(TARGET_LENGTHS)]
        text = "".join(
            amino[(index * 11 + position * 7) % len(amino)]
            for position in range(length)
        )
        sequences.append(
            pyhmmer.easel.TextSequence(
                name=f"phase6-target-{index:03d}".encode(), sequence=text
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
        raise RuntimeError("Phase 6 oracle requires attested sm90/H200")
    hmms, targets = fixture()
    with tempfile.TemporaryDirectory(prefix="phase6-length-metadata-") as temporary:
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
                os.environ["PLAN7_GPU_SSV_LENGTH_METADATA"] = "expanded"
                expanded = batch.postfilter_selection(selection, F1=F1)
                after_expanded = dict(native.workspace_statistics)
                os.environ["PLAN7_GPU_SSV_LENGTH_METADATA"] = "compact"
                compact = batch.postfilter_selection(selection, F1=F1)
                after_compact = dict(native.workspace_statistics)
            finally:
                os.environ.pop("PLAN7_GPU_SSV_LENGTH_METADATA", None)
                _pipeline._seal_postfilter_batch_bound = seal_factory

            expanded_state = _candidate_state(expanded)
            compact_state = _candidate_state(compact)
            expanded_offsets = memoryview(expanded_state.offsets).cast("B").tobytes()
            compact_offsets = memoryview(compact_state.offsets).cast("B").tobytes()
            expanded_records = bytes(expanded_state.postfilter_records)
            compact_records = bytes(compact_state.postfilter_records)
            if expanded_offsets != compact_offsets or expanded_records != compact_records:
                raise AssertionError("length-class metadata changed records or order")

            outputs = bytearray()
            for row in range(len(pairs)):
                expected = hit_bytes(
                    expanded.search(row, pipeline(pairs[0].hmm.alphabet))
                )
                actual = hit_bytes(compact.search(row, pipeline(pairs[0].hmm.alphabet)))
                if actual != expected:
                    raise AssertionError(
                        f"length-class metadata changed HMMER output at row {row}"
                    )
                outputs.extend(actual)

            expanded_delta = delta(after_expanded, before)
            compact_delta = delta(after_compact, after_expanded)
            if expanded_delta["f1_length_class_run_count"] != 0:
                raise AssertionError("expanded reference used length classes")
            if (
                compact_delta["f1_length_class_run_count"] != 1
                or compact_delta["f1_length_class_value_count"]
                != len(TARGET_LENGTHS)
                or compact_delta["f1_length_compact_h2d_bytes"] <= 0
                or compact_delta["f1_length_dense_h2d_bytes_avoided"]
                != compact_delta["f1_length_dense_materialized_bytes"]
                or compact_delta["f1_length_dense_h2d_bytes_avoided"]
                != compact_delta["f1_length_compact_h2d_bytes"]
                * len(targets)
                // len(TARGET_LENGTHS)
            ):
                raise AssertionError(
                    f"length-class execution census is wrong: {compact_delta}"
                )

            return {
                "schema": 1,
                "status": "PASS",
                "device": provenance,
                "profile_count": len(pairs),
                "sequence_count": len(targets),
                "length_classes": list(TARGET_LENGTHS),
                "postfilter_offsets_sha256": sha256(compact_offsets),
                "postfilter_records_sha256": sha256(compact_records),
                "hmm_output_sha256": sha256(bytes(outputs)),
                "expanded_delta": expanded_delta,
                "compact_delta": compact_delta,
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
