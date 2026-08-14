from __future__ import annotations

from array import array
from collections.abc import Iterable
from threading import Lock
from typing import Any

from . import _native  # type: ignore[attr-defined]


def _format_results(raw_results: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
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


class SequenceBatch:
    """A packed target batch kept resident on one CUDA device."""

    def __init__(self, sequences: Iterable[Any], *, alphabet: Any | None = None):
        inherited_alphabet = getattr(sequences, "alphabet", None)
        sequence_list = list(sequences)
        if alphabet is None:
            alphabet = inherited_alphabet
        if alphabet is None and sequence_list:
            alphabet = sequence_list[0].alphabet
        if alphabet is None:
            raise ValueError("an alphabet is required for an empty sequence batch")

        residues = bytearray()
        offsets = array("Q", [0])
        for sequence in sequence_list:
            if sequence.alphabet != alphabet:
                raise ValueError("sequence alphabets differ")
            encoded = memoryview(sequence.sequence).cast("B")
            if len(encoded) > 100_000:
                raise ValueError("sequence length exceeds HMMER's protein limit")
            residues.extend(encoded)
            offsets.append(len(residues))

        self.alphabet = alphabet
        self._native = _native.SequenceBatch(residues, offsets, alphabet.Kp)
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._native)

    @property
    def closed(self) -> bool:
        with self._lock:
            return bool(self._native.closed)

    def close(self) -> None:
        with self._lock:
            self._native.close()

    def __enter__(self) -> SequenceBatch:
        if self.closed:
            raise RuntimeError("sequence batch is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _filter_raw(self, optimized_profile: Any) -> list[tuple[Any, ...]]:
        with self._lock:
            if optimized_profile.alphabet != self.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            scores = memoryview(optimized_profile.sbv).cast("B")
            return self._native.filter_raw(
                scores,
                optimized_profile.sbv.shape[1],
                optimized_profile.M,
                optimized_profile.alphabet.Kp,
                optimized_profile.tbm,
                optimized_profile.tec,
                optimized_profile.base_b,
                optimized_profile.bias_b,
                optimized_profile.scale_b,
            )

    def filter_ssv(self, optimized_profile: Any) -> list[dict[str, Any]]:
        return _format_results(self._filter_raw(optimized_profile))


def filter_ssv(
    optimized_profile: Any, sequences: Iterable[Any] | SequenceBatch
) -> list[dict[str, Any]]:
    """Run direct SSV without making the pipeline's F1 decision.

    ``eslENORESULT`` requires CPU full-MSV fallback, while ``eslERANGE`` is a
    guaranteed overflow that must be promoted. Empty targets are skipped by
    HMMER's comparison pipeline.
    """
    if isinstance(sequences, SequenceBatch):
        return sequences.filter_ssv(optimized_profile)
    with SequenceBatch(sequences, alphabet=optimized_profile.alphabet) as batch:
        return batch.filter_ssv(optimized_profile)
