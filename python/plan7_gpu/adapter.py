from __future__ import annotations

from array import array
from collections.abc import Iterable
import math
from threading import Lock
from typing import Any, NamedTuple, cast

from . import _native  # type: ignore[attr-defined]


class _PackedProfiles(NamedTuple):
    scores: bytearray
    score_offsets: array[int]
    score_counts: array[int]
    score_strides: array[int]
    model_lengths: array[int]
    constants: bytearray
    scales: array[float]


def _pack_profiles(profiles: list[Any]) -> _PackedProfiles:
    striped_score_buffers = []
    striped_score_strides = array("i")
    score_offsets = array("Q")
    score_counts = array("Q")
    score_strides = array("i")
    model_lengths = array("i")
    constants = bytearray()
    scales = array("f")
    score_offset = 0

    for profile in profiles:
        alphabet_size = profile.alphabet.Kp
        score_count = profile.M * alphabet_size
        striped_score_buffers.append(memoryview(profile.sbv).cast("B"))
        striped_score_strides.append(profile.sbv.shape[1])
        score_offsets.append(score_offset)
        score_counts.append(score_count)
        score_strides.append(alphabet_size)
        model_lengths.append(profile.M)
        constants.extend((profile.tbm, profile.tec, profile.base_b, profile.bias_b))
        scales.append(profile.scale_b)
        score_offset += score_count

    scores = (
        _native.pack_striped_scores(
            striped_score_buffers,
            striped_score_strides,
            model_lengths,
            profiles[0].alphabet.Kp,
        )
        if profiles
        else bytearray()
    )

    return _PackedProfiles(
        scores,
        score_offsets,
        score_counts,
        score_strides,
        model_lengths,
        constants,
        scales,
    )


def _f1_parameters(profile: Any) -> tuple[float, float, float] | None:
    try:
        parameters = profile.evalue_parameters.as_vector()
        m_mu = float(parameters[0])
        m_lambda = float(parameters[1])
        scale = float(profile.scale_b)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(m_mu)
        or not math.isfinite(m_lambda)
        or not math.isfinite(scale)
        or m_mu == -99999.0
        or m_lambda == -99999.0
        or m_lambda <= 0.0
        or scale <= 0.0
    ):
        return None
    return m_mu, m_lambda, scale


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
            return cast(
                list[tuple[Any, ...]],
                self._native.filter_raw(
                    scores,
                    optimized_profile.sbv.shape[1],
                    optimized_profile.M,
                    optimized_profile.alphabet.Kp,
                    optimized_profile.tbm,
                    optimized_profile.tec,
                    optimized_profile.base_b,
                    optimized_profile.bias_b,
                    optimized_profile.scale_b,
                ),
            )

    def filter_ssv(self, optimized_profile: Any) -> list[dict[str, Any]]:
        return _format_results(self._filter_raw(optimized_profile))

    def cpu_candidates(self, optimized_profile: Any, F1: float = 0.02) -> list[int]:
        """Return target indexes that still require HMMER's CPU pipeline.

        Only finite direct-SSV scores with a P-value strictly greater than
        ``F1`` are omitted. Every fallback, overflow, empty, or invalid case
        is retained conservatively.
        """
        with self._lock:
            if optimized_profile.alphabet != self.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            if self._native.closed:
                raise RuntimeError("sequence batch is closed")

            try:
                threshold = float(F1)
            except (TypeError, ValueError, OverflowError):
                return list(range(len(self)))

            if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
                return list(range(len(self)))
            parameters = _f1_parameters(optimized_profile)
            if parameters is None:
                return list(range(len(self)))
            m_mu, m_lambda, scale = parameters

            scores = memoryview(optimized_profile.sbv).cast("B")
            return cast(
                list[int],
                self._native.cpu_candidates_raw(
                    scores,
                    optimized_profile.sbv.shape[1],
                    optimized_profile.M,
                    optimized_profile.alphabet.Kp,
                    optimized_profile.tbm,
                    optimized_profile.tec,
                    optimized_profile.base_b,
                    optimized_profile.bias_b,
                    scale,
                    m_mu,
                    m_lambda,
                    threshold,
                ),
            )

    def filter_ssv_many(
        self, optimized_profiles: Iterable[Any]
    ) -> list[list[dict[str, Any]]]:
        profiles = list(optimized_profiles)
        with self._lock:
            if self._native.closed:
                raise RuntimeError("sequence batch is closed")
            for profile in profiles:
                if profile.alphabet != self.alphabet:
                    raise ValueError("profile and sequence alphabets differ")
            packed = _pack_profiles(profiles)
            raw = cast(
                list[list[tuple[Any, ...]]],
                self._native.filter_many_raw(*packed),
            )
        return [_format_results(profile_results) for profile_results in raw]

    def cpu_candidates_many(
        self, optimized_profiles: Iterable[Any], F1: float = 0.02
    ) -> list[list[int]]:
        profiles = list(optimized_profiles)
        with self._lock:
            if self._native.closed:
                raise RuntimeError("sequence batch is closed")
            for profile in profiles:
                if profile.alphabet != self.alphabet:
                    raise ValueError("profile and sequence alphabets differ")
            try:
                threshold = float(F1)
            except (TypeError, ValueError, OverflowError):
                return [list(range(len(self))) for _ in profiles]
            if not math.isfinite(threshold) or not 0.0 <= threshold < 1.0:
                return [list(range(len(self))) for _ in profiles]

            output: list[list[int] | None] = [None] * len(profiles)
            valid_profiles = []
            valid_indices = []
            m_mu = array("f")
            m_lambda = array("f")
            for index, profile in enumerate(profiles):
                parameters = _f1_parameters(profile)
                if parameters is None:
                    output[index] = list(range(len(self)))
                else:
                    profile_mu, profile_lambda, _ = parameters
                    valid_profiles.append(profile)
                    valid_indices.append(index)
                    m_mu.append(profile_mu)
                    m_lambda.append(profile_lambda)

            if valid_profiles:
                packed = _pack_profiles(valid_profiles)
                candidates = cast(
                    list[list[int]],
                    self._native.cpu_candidates_many_raw(
                        *packed, m_mu, m_lambda, threshold
                    ),
                )
                for index, profile_candidates in zip(
                    valid_indices, candidates, strict=True
                ):
                    output[index] = profile_candidates

            return cast(list[list[int]], output)


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


def cpu_candidates(
    optimized_profile: Any,
    sequences: Iterable[Any] | SequenceBatch,
    F1: float = 0.02,
) -> list[int]:
    """Return indexes that cannot be rejected safely by direct CUDA SSV."""
    if isinstance(sequences, SequenceBatch):
        return sequences.cpu_candidates(optimized_profile, F1)
    with SequenceBatch(sequences, alphabet=optimized_profile.alphabet) as batch:
        return batch.cpu_candidates(optimized_profile, F1)
