"""Focused exact/timing oracle for sealed bias-reject Viterbi skipping."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import statistics
import struct
import tempfile
from pathlib import Path

import pyhmmer

from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
from plan7_gpu import _native
from plan7_gpu.adapter import _profile_selection_state, _sequence_native


SOURCE_MODELS = (
    "RREFam.hmm",
    "PF02826.hmm",
    "Thioesterase.hmm",
    "KR.hmm",
    "LuxC.hmm",
)
POSTFILTER = struct.Struct("=IfhBBf")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def mutate_copy(hmm: pyhmmer.plan7.HMM, family: int, variant: int):
    result = hmm.copy()
    result.name = f"post258-p7-{family:02d}-{variant:02d}".encode()
    result.accession = None
    left = variant % 20
    right = (variant * 7 + 3) % 20
    if left == right:
        right = (right + 1) % 20
    for position in range(1, result.M + 1):
        row = result.match_emissions[position]
        row[left], row[right] = row[right], row[left]
    return result


def fixture(profile_count: int, target_count: int, maximum_length: int):
    data = Path(pyhmmer.__file__).parent / "tests" / "data" / "hmms" / "txt"
    templates = []
    for name in SOURCE_MODELS:
        with pyhmmer.plan7.HMMFile(data / name) as source:
            template = source.read()
        if template is None:
            raise RuntimeError(f"missing fixture HMM: {name}")
        templates.append(template)
    hmms = [
        mutate_copy(templates[index % len(templates)], index % len(templates), index)
        for index in range(profile_count)
    ]
    alphabet = hmms[0].alphabet
    amino = "ACDEFGHIKLMNPQRSTVWY"
    sequences = []
    for index in range(target_count):
        length = 32 + (index * 131) % (maximum_length - 31)
        text = "".join(
            amino[(index * 11 + position * 7) % len(amino)]
            for position in range(length)
        )
        sequences.append(
            pyhmmer.easel.TextSequence(
                name=f"post258-p7-target-{index:05d}".encode(), sequence=text
            ).digitize(alphabet)
        )
    return hmms, pyhmmer.easel.DigitalSequenceBlock(alphabet, sequences)


def pipeline(alphabet):
    return pyhmmer.plan7.Pipeline(
        alphabet,
        F1=0.99,
        F2=1.0e-3,
        F3=1.0e-5,
        E=10.0,
        domE=10.0,
        incE=10.0,
        incdomE=10.0,
    )


def hit_bytes(hits) -> bytes:
    output = io.BytesIO()
    hits.write(output, format="targets", header=True)
    hits.write(output, format="domains", header=True)
    return output.getvalue()


def compare_records(reference: bytes, fixed: bytes):
    if len(reference) != len(fixed) or len(reference) % POSTFILTER.size:
        raise AssertionError("postfilter record storage size changed")
    bias_rejects = 0
    for offset in range(0, len(reference), POSTFILTER.size):
        before = POSTFILTER.unpack_from(reference, offset)
        after = POSTFILTER.unpack_from(fixed, offset)
        # sequence, filtersc bits, numerator, MSV status, and action are the
        # complete fixed-bias semantics and must be bit-identical.
        if before[:1] != after[:1] or reference[offset + 4 : offset + 12] != fixed[
            offset + 4 : offset + 12
        ]:
            raise AssertionError(f"postfilter semantics changed at row {offset // 16}")
        filtersc = before[1]
        action = before[4]
        if action == _native.BIAS_DEFINITE_REJECT and math.isfinite(filtersc):
            bias_rejects += 1
            if not math.isinf(after[5]) or after[5] < 0.0:
                raise AssertionError("sealed bias reject lacks +infinity sentinel")
        elif reference[offset + 12 : offset + 16] != fixed[offset + 12 : offset + 16]:
            raise AssertionError(f"nonterminal Viterbi score changed at row {offset // 16}")
    return bias_rejects


def postfilter_call(native, native_selection, fixed: bool):
    before = native.workspace_statistics["generation_ledger"]["postfilter_native_ns"]
    records, offsets, _reasons, reason_statistics = (
        native.postfilter_profile_selection_csr_raw(
            native_selection,
            0.99,
            _return_reason_facts=True,
            _immutable_records=True,
            _sealed_bias_viterbi_skip=fixed,
        )
    )
    after = native.workspace_statistics["generation_ledger"]["postfilter_native_ns"]
    return {
        "records": bytes(records),
        "offsets": memoryview(offsets).cast("B").tobytes(),
        "statistics": tuple(reason_statistics),
        "wall_ms": (after - before) / 1.0e6,
    }


def fused_output(targets, selection, pairs, enabled: bool):
    if enabled:
        os.environ["PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP"] = "1"
    else:
        os.environ.pop("PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP", None)
    with SequenceBatch(targets) as batch:
        generation_pipeline = pipeline(pairs[0].hmm.alphabet)
        candidates = batch._postfilter_forward_selection(
            selection,
            0.99,
            1.0e-3,
            1.0e-5,
            True,
            pipeline=generation_pipeline,
            sparse_journal_v3=True,
        )
        output = bytearray()
        for row in range(len(pairs)):
            output.extend(
                hit_bytes(candidates.search(row, pipeline(pairs[0].hmm.alphabet)))
            )
        return bytes(output)


def audit():
    provenance = _native.bias_environment_provenance()
    if not provenance["attested"] or provenance["target"] != "sm90_h200":
        raise RuntimeError("sealed bias/Viterbi oracle requires attested sm90/H200")
    os.environ["PLAN7_GPU_GENERATION_LEDGER"] = "1"
    hmms, timing_targets = fixture(40, 4096, 512)
    _, output_targets = fixture(40, 256, 256)
    try:
        with tempfile.TemporaryDirectory(prefix="post258-p7-") as temporary:
            base = Path(temporary) / "models"
            pyhmmer.hmmer.hmmpress(hmms, base)
            pairs = load_pressed_profiles(base)
            with (
                ProfileSession(pairs, pack_workers=4) as session,
                session.select(range(len(pairs))) as selection,
            ):
                with SequenceBatch(timing_targets) as batch:
                    native = _sequence_native(batch)
                    native_selection = _profile_selection_state(selection).native
                    reference = postfilter_call(native, native_selection, False)
                    fixed = postfilter_call(native, native_selection, True)
                    if fixed["offsets"] != reference["offsets"]:
                        raise AssertionError("fixed-bias candidate offsets changed")
                    bias_rejects = compare_records(
                        reference["records"], fixed["records"]
                    )
                    if bias_rejects <= 0:
                        raise AssertionError("fixture produced no finite bias rejects")
                    reference_stats = reference["statistics"]
                    fixed_stats = fixed["statistics"]
                    skipped_rows = reference_stats[2] - fixed_stats[2]
                    skipped_cells = reference_stats[4] - fixed_stats[4]
                    if skipped_rows != bias_rejects or skipped_cells <= 0:
                        raise AssertionError(
                            f"Viterbi skip census does not reconcile: "
                            f"rejects={bias_rejects}, rows={skipped_rows}, "
                            f"cells={skipped_cells}"
                        )

                    reference_ms = []
                    fixed_ms = []
                    for repeat in range(5):
                        modes = (
                            (False, True) if repeat % 2 == 0 else (True, False)
                        )
                        paired = {}
                        for enabled in modes:
                            paired[enabled] = postfilter_call(
                                native, native_selection, enabled
                            )
                        if paired[True]["offsets"] != paired[False]["offsets"]:
                            raise AssertionError("timed candidate offsets changed")
                        compare_records(
                            paired[False]["records"], paired[True]["records"]
                        )
                        reference_ms.append(paired[False]["wall_ms"])
                        fixed_ms.append(paired[True]["wall_ms"])

                ordinary_output = fused_output(
                    output_targets, selection, pairs, False
                )
                fixed_output = fused_output(output_targets, selection, pairs, True)
                if fixed_output != ordinary_output:
                    raise AssertionError(
                        "sealed bias/Viterbi skip changed final HMMER output"
                    )
    finally:
        os.environ.pop("PLAN7_GPU_SEALED_BIAS_VITERBI_SKIP", None)
        os.environ.pop("PLAN7_GPU_GENERATION_LEDGER", None)

    reference_median = statistics.median(reference_ms)
    fixed_median = statistics.median(fixed_ms)
    return {
        "schema": "post258-sealed-bias-viterbi-v1",
        "status": "PASS",
        "device": provenance,
        "profile_count": len(hmms),
        "timing_target_count": len(timing_targets),
        "candidate_count": reference_stats[0],
        "bias_reject_count": bias_rejects,
        "viterbi_execution_count_before": reference_stats[2],
        "viterbi_execution_count_after": fixed_stats[2],
        "viterbi_cells_before": reference_stats[4],
        "viterbi_cells_after": fixed_stats[4],
        "viterbi_cells_skipped": skipped_cells,
        "reference_postfilter_ms": reference_ms,
        "fixed_postfilter_ms": fixed_ms,
        "reference_median_ms": reference_median,
        "fixed_median_ms": fixed_median,
        "ratio": fixed_median / reference_median,
        "speedup": reference_median / fixed_median,
        "output_sha256": sha256(fixed_output),
        "output_bytes": len(fixed_output),
    }


def main():
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
