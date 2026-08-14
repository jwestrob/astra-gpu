from __future__ import annotations

from array import array
from collections.abc import Iterable
from typing import Any

from . import _native


def filter_ssv(optimized_profile: Any, sequences: Iterable[Any]) -> list[dict[str, Any]]:
    """Run direct SSV without making the pipeline's F1 decision.

    ``eslENORESULT`` requires CPU full-MSV fallback, while ``eslERANGE`` is a
    guaranteed overflow that must be promoted. Empty targets are skipped by
    HMMER's comparison pipeline.
    """
    sequence_list = list(sequences)
    residues = bytearray()
    offsets = array("Q", [0])

    for sequence in sequence_list:
        if sequence.alphabet != optimized_profile.alphabet:
            raise ValueError("profile and sequence alphabets differ")
        encoded = memoryview(sequence.sequence).cast("B")
        if len(encoded) > 100_000:
            raise ValueError("sequence length exceeds HMMER's protein limit")
        residues.extend(encoded)
        offsets.append(len(residues))

    scores = memoryview(optimized_profile.sbv).cast("B")
    score_stride = optimized_profile.sbv.shape[1]
    raw_results = _native.filter_raw(
        scores,
        score_stride,
        optimized_profile.M,
        optimized_profile.alphabet.Kp,
        residues,
        offsets,
        optimized_profile.tbm,
        optimized_profile.tec,
        optimized_profile.base_b,
        optimized_profile.bias_b,
        optimized_profile.scale_b,
    )

    names = {
        _native.STATUS_OK: "eslOK",
        _native.STATUS_ERANGE: "eslERANGE",
        _native.STATUS_ENORESULT: "eslENORESULT",
        _native.STATUS_EMPTY: "empty",
    }
    actions = {
        _native.STATUS_OK: "threshold_score",
        _native.STATUS_ERANGE: "promote",
        _native.STATUS_ENORESULT: "cpu_msv",
        _native.STATUS_EMPTY: "skip_empty",
    }
    return [
        {
            "status": names[status],
            "action": actions[status],
            "xE_u8": xE,
            "tjb_u8": length_tjb,
            "numerator": numerator if status == _native.STATUS_OK else None,
            "score_bits": score_bits,
        }
        for status, xE, length_tjb, numerator, score_bits in raw_results
    ]
