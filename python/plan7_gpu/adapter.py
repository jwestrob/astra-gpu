from __future__ import annotations

from array import array
from collections.abc import Iterable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import io
from itertools import zip_longest
import math
import operator
from pathlib import Path
import struct
from threading import Lock
from typing import Any, NamedTuple, cast
from weakref import WeakKeyDictionary

import pyhmmer

from . import _native  # type: ignore[attr-defined]

_MISSING = object()
_PIPELINE_LEASES_LOCK = Lock()


class _PipelineLease:
    __slots__ = ("pipeline", "lock", "refcount")

    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.lock = Lock()
        self.refcount = 0


_PIPELINE_LEASES: dict[int, _PipelineLease] = {}


@contextmanager
def _lease_pipeline(pipeline: Any):
    identity = id(pipeline)
    with _PIPELINE_LEASES_LOCK:
        lease = _PIPELINE_LEASES.get(identity)
        if lease is None:
            lease = _PipelineLease(pipeline)
            _PIPELINE_LEASES[identity] = lease
        elif lease.pipeline is not pipeline:
            raise RuntimeError("pipeline identity collision")
        lease.refcount += 1

    try:
        with lease.lock:
            yield
    finally:
        with _PIPELINE_LEASES_LOCK:
            lease.refcount -= 1
            if lease.refcount == 0:
                if _PIPELINE_LEASES.get(identity) is not lease:
                    raise RuntimeError("pipeline lease registry corruption")
                del _PIPELINE_LEASES[identity]


class _FileStat(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class _PressedStatToken(NamedTuple):
    h3m: _FileStat
    h3i: _FileStat
    h3f: _FileStat
    h3p: _FileStat


class _CutoffSnapshot(NamedTuple):
    gathering: tuple[float, float] | None
    noise: tuple[float, float] | None
    trusted: tuple[float, float] | None


class _PairState:
    __slots__ = ("hmm", "optimized_profile", "background_fingerprint", "lock")

    def __init__(
        self,
        hmm: Any,
        optimized_profile: Any,
        background_fingerprint: bytes,
    ) -> None:
        self.hmm = hmm
        self.optimized_profile = optimized_profile
        self.background_fingerprint = background_fingerprint
        self.lock = Lock()


def _background_fingerprint(background: Any) -> bytes:
    frequencies = memoryview(background.residue_frequencies)
    if frequencies.format != "f" or frequencies.ndim != 1:
        raise TypeError("background residue frequencies are not binary32")
    return frequencies.cast("B").tobytes() + struct.pack("=f", background.omega)


def _optimized_profile_bytes(profile: Any) -> tuple[bytes, bytes]:
    filter_output = io.BytesIO()
    profile_output = io.BytesIO()
    profile.write(filter_output, profile_output)
    return filter_output.getvalue(), profile_output.getvalue()


def _verify_pressed_profile(
    hmm: Any,
    optimized_profile: Any,
    canonical_background: Any,
    pipeline_module: Any,
) -> bool:
    reference_profile = hmm.to_profile(
        canonical_background, L=optimized_profile.L
    ).to_optimized()
    for offset_name in ("model", "filter", "profile"):
        setattr(
            reference_profile.offsets,
            offset_name,
            getattr(optimized_profile.offsets, offset_name),
        )
    return pipeline_module._oprofiles_equal_hmmer(
        reference_profile, optimized_profile
    ) and _optimized_profile_bytes(reference_profile) == _optimized_profile_bytes(
        optimized_profile
    )


@dataclass(
    frozen=True,
    slots=True,
    weakref_slot=True,
    init=False,
    eq=False,
    repr=False,
)
class PressedProfilePair:
    """One immutable HMM/optimized-profile pair read in pressed-file lockstep."""

    canonical_base: Path
    ordinal: int
    stat_token: _PressedStatToken
    cutoffs: _CutoffSnapshot

    def __init__(self) -> None:
        raise TypeError("PressedProfilePair objects come from load_pressed_profiles")

    @property
    def hmm(self) -> Any:
        return _pair_state(self).hmm.copy()

    def __repr__(self) -> str:
        return (
            f"PressedProfilePair(canonical_base={self.canonical_base!r}, "
            f"ordinal={self.ordinal})"
        )


_PAIR_STATES: WeakKeyDictionary[PressedProfilePair, _PairState] = WeakKeyDictionary()


def _pair_state(pair: PressedProfilePair) -> _PairState:
    try:
        return _PAIR_STATES[pair]
    except KeyError as error:
        raise TypeError(
            "candidate batches require pairs from load_pressed_profiles"
        ) from error


def _new_pressed_profile_pair(
    hmm: Any,
    optimized_profile: Any,
    canonical_base: Path,
    ordinal: int,
    stat_token: _PressedStatToken,
    cutoffs: _CutoffSnapshot,
    background_fingerprint: bytes,
) -> PressedProfilePair:
    pair = object.__new__(PressedProfilePair)
    object.__setattr__(pair, "canonical_base", canonical_base)
    object.__setattr__(pair, "ordinal", ordinal)
    object.__setattr__(pair, "stat_token", stat_token)
    object.__setattr__(pair, "cutoffs", cutoffs)
    _PAIR_STATES[pair] = _PairState(hmm, optimized_profile, background_fingerprint)
    return pair


def load_pressed_profiles(
    path: str | Path,
    *,
    manifest: str | Path | None = None,
) -> tuple[PressedProfilePair, ...]:
    """Load a pressed HMM database as provenance-bound lockstep pairs.

    ``manifest`` is an unsigned trust anchor and must come from a trusted
    source, such as this repository's reviewed ``results/pressed`` records.
    """
    from .pressed_manifest import (
        _pinned_pressed_database,
        validate_pressed_manifest,
    )

    base = Path(path).resolve(strict=False)
    amino_alphabet = pyhmmer.easel.Alphabet.amino()
    manifest_validation = None
    if manifest is not None:
        manifest_validation = validate_pressed_manifest(base, manifest)
        if manifest_validation.canonical_base != base:
            raise ValueError("pressed manifest canonical database mismatch")
        pipeline_module = None
    else:
        from . import _pipeline  # type: ignore[attr-defined]

        pipeline_module = _pipeline
    pairs = []
    with _pinned_pressed_database(base) as (pinned_base, pinned_token):
        before = _PressedStatToken(*(_FileStat(*member) for member in pinned_token))
        if (
            manifest_validation is not None
            and pinned_token != manifest_validation.stat_token
        ):
            raise RuntimeError(
                "pressed database changed after its manifest was validated"
            )
        with (
            pyhmmer.plan7.HMMFile(pinned_base) as hmm_file,
            pyhmmer.plan7.HMMPressedFile(pinned_base) as optimized_file,
        ):
            for ordinal, (hmm, optimized_profile) in enumerate(
                zip_longest(hmm_file, optimized_file, fillvalue=_MISSING)
            ):
                if hmm is _MISSING:
                    raise ValueError(
                        "pressed optimized-profile stream has more models than HMM stream"
                    )
                if optimized_profile is _MISSING:
                    raise ValueError(
                        "pressed HMM stream has more models than optimized-profile stream"
                    )
                hmm = cast(pyhmmer.plan7.HMM, hmm)
                optimized_profile = cast(
                    pyhmmer.plan7.OptimizedProfile, optimized_profile
                )
                if hmm.alphabet != amino_alphabet:
                    raise ValueError(
                        "bound candidate search only supports amino-acid profiles"
                    )
                if (
                    hmm.alphabet != optimized_profile.alphabet
                    or hmm.M != optimized_profile.M
                    or hmm.name != optimized_profile.name
                    or hmm.accession != optimized_profile.accession
                    or hmm.consensus != optimized_profile.consensus
                ):
                    raise ValueError(
                        f"pressed HMM/profile identity mismatch at ordinal {ordinal}"
                    )
                canonical_background = pyhmmer.plan7.Background(hmm.alphabet)
                if manifest_validation is None and not _verify_pressed_profile(
                    hmm,
                    optimized_profile,
                    canonical_background,
                    pipeline_module,
                ):
                    raise ValueError(
                        f"pressed HMM/profile score mismatch at ordinal {ordinal}"
                    )
                pairs.append(
                    _new_pressed_profile_pair(
                        hmm,
                        optimized_profile,
                        base,
                        ordinal,
                        before,
                        _CutoffSnapshot(
                            hmm.cutoffs.gathering,
                            hmm.cutoffs.noise,
                            hmm.cutoffs.trusted,
                        ),
                        _background_fingerprint(canonical_background),
                    )
                )

    if (
        manifest_validation is not None
        and len(pairs) != manifest_validation.model_count
    ):
        raise ValueError("pressed profile count does not match the validated manifest")
    return tuple(pairs)


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


class _CandidateState:
    __slots__ = (
        "pairs",
        "targets",
        "indices",
        "offsets",
        "all_rows",
        "all_targets",
        "f1",
    )

    def __init__(
        self,
        pairs: tuple[PressedProfilePair, ...],
        targets: Any,
        indices: bytes,
        offsets: bytes,
        all_rows: bytes,
        all_targets: bytes,
        f1: float,
    ) -> None:
        self.pairs = pairs
        self.targets = targets
        self.indices = indices
        self.offsets = offsets
        self.all_rows = all_rows
        self.all_targets = all_targets
        self.f1 = f1


@dataclass(
    frozen=True,
    slots=True,
    weakref_slot=True,
    init=False,
    eq=False,
    repr=False,
)
class CandidateBatch:
    """Opaque candidate rows bound to their queries, profiles, and targets."""

    def __init__(self) -> None:
        raise TypeError(
            "CandidateBatch objects come from SequenceBatch.candidate_batch"
        )

    def __len__(self) -> int:
        return len(_candidate_state(self).pairs)

    @property
    def F1(self) -> float:
        return _candidate_state(self).f1

    def __repr__(self) -> str:
        return f"CandidateBatch(rows={len(self)}, F1={self.F1!r})"

    def _row_index(self, row: int) -> int:
        if isinstance(row, bool):
            raise TypeError("candidate row must be an integer, not bool")
        try:
            row_index = operator.index(row)
        except TypeError as error:
            raise TypeError("candidate row must be an integer") from error
        if not 0 <= row_index < len(self):
            raise IndexError("candidate row out of range")
        return row_index

    def candidate_count(self, row: int) -> int:
        """Return the number of targets retained in one bound row."""
        row_index = self._row_index(row)
        state = _candidate_state(self)
        if state.all_rows[row_index]:
            return len(state.all_targets) // 4
        offsets = memoryview(state.offsets).cast("Q")
        return offsets[row_index + 1] - offsets[row_index]

    def search(self, row: int, pipeline: Any) -> Any:
        """Search one bound row using an exclusively owned matching pipeline.

        The caller must not inspect, mutate, clear, or use ``pipeline`` from
        another thread until this call returns.
        """
        row_index = self._row_index(row)
        if type(pipeline) is not pyhmmer.plan7.Pipeline:
            raise TypeError("pipeline must be exactly pyhmmer.plan7.Pipeline")
        candidate_state = _candidate_state(self)
        pair = candidate_state.pairs[row_index]
        if candidate_state.all_rows[row_index]:
            candidate_row = memoryview(candidate_state.all_targets).cast("I")
        else:
            offsets = memoryview(candidate_state.offsets).cast("Q")
            start = offsets[row_index]
            stop = offsets[row_index + 1]
            candidate_row = memoryview(candidate_state.indices).cast("I")[start:stop]
        from . import _pipeline  # type: ignore[attr-defined]

        state = _pair_state(pair)

        with _lease_pipeline(pipeline):
            if float(pipeline.F1) != candidate_state.f1:
                raise ValueError(
                    f"pipeline F1 {pipeline.F1!r} does not match "
                    f"candidate F1 {candidate_state.f1!r}"
                )
            if (
                _background_fingerprint(pipeline.background)
                != state.background_fingerprint
            ):
                raise ValueError(
                    "pipeline background does not match the canonical "
                    "hmmpress background"
                )
            with state.lock:
                return _pipeline._search_hmm_candidates(
                    pipeline,
                    state.hmm.copy(),
                    state.optimized_profile,
                    candidate_state.targets,
                    candidate_row,
                )


_CANDIDATE_STATES: WeakKeyDictionary[CandidateBatch, _CandidateState] = (
    WeakKeyDictionary()
)


def _candidate_state(candidates: CandidateBatch) -> _CandidateState:
    try:
        return _CANDIDATE_STATES[candidates]
    except KeyError as error:
        raise TypeError(
            "CandidateBatch objects come from SequenceBatch.candidate_batch"
        ) from error


def _new_candidate_batch(
    pairs: tuple[PressedProfilePair, ...],
    targets: Any,
    indices: array[int],
    offsets: array[int],
    all_rows: bytes,
    all_targets: array[int],
    f1: float,
) -> CandidateBatch:
    candidates = object.__new__(CandidateBatch)
    _CANDIDATE_STATES[candidates] = _CandidateState(
        pairs,
        targets,
        indices.tobytes(),
        offsets.tobytes(),
        all_rows,
        all_targets.tobytes(),
        f1,
    )
    return candidates


class _SequenceState:
    __slots__ = ("alphabet", "native", "lock", "targets")

    def __init__(self, alphabet: Any, native: Any, targets: Any) -> None:
        self.alphabet = alphabet
        self.native = native
        self.lock = Lock()
        self.targets = targets


_SEQUENCE_STATES: WeakKeyDictionary[Any, _SequenceState] = WeakKeyDictionary()


def _sequence_state(batch: Any) -> _SequenceState:
    try:
        return _SEQUENCE_STATES[batch]
    except KeyError as error:
        raise TypeError("invalid SequenceBatch object") from error


def _sequence_native(batch: Any) -> Any:
    """Return the concealed native batch for package diagnostics only."""
    return _sequence_state(batch).native


class SequenceBatch:
    """A packed target batch kept resident on one CUDA device."""

    __slots__ = ("__weakref__",)

    def __init__(self, sequences: Iterable[Any], *, alphabet: Any | None = None):
        inherited_alphabet = getattr(sequences, "alphabet", None)
        sequence_list = list(sequences)
        if alphabet is None:
            alphabet = inherited_alphabet
        if alphabet is None and sequence_list:
            alphabet = sequence_list[0].alphabet
        if alphabet is None:
            raise ValueError("an alphabet is required for an empty sequence batch")

        target_sequences = [sequence.copy() for sequence in sequence_list]
        residues = bytearray()
        offsets = array("Q", [0])
        for sequence in target_sequences:
            if sequence.alphabet != alphabet:
                raise ValueError("sequence alphabets differ")
            encoded = memoryview(sequence.sequence).cast("B")
            if len(encoded) > 100_000:
                raise ValueError("sequence length exceeds HMMER's protein limit")
            residues.extend(encoded)
            offsets.append(len(residues))
        targets = pyhmmer.easel.DigitalSequenceBlock(alphabet, target_sequences)

        native = _native.SequenceBatch(residues, offsets, alphabet.Kp)
        _SEQUENCE_STATES[self] = _SequenceState(alphabet, native, targets)

    @property
    def alphabet(self) -> Any:
        return _sequence_state(self).alphabet

    def __len__(self) -> int:
        return len(_sequence_state(self).native)

    @property
    def closed(self) -> bool:
        state = _sequence_state(self)
        with state.lock:
            return bool(state.native.closed)

    def close(self) -> None:
        state = _sequence_state(self)
        with state.lock:
            state.native.close()

    def __enter__(self) -> SequenceBatch:
        if self.closed:
            raise RuntimeError("sequence batch is closed")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _filter_raw(self, optimized_profile: Any) -> list[tuple[Any, ...]]:
        state = _sequence_state(self)
        with state.lock:
            if optimized_profile.alphabet != state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            scores = memoryview(optimized_profile.sbv).cast("B")
            return cast(
                list[tuple[Any, ...]],
                state.native.filter_raw(
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
        state = _sequence_state(self)
        with state.lock:
            if optimized_profile.alphabet != state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            if state.native.closed:
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
                state.native.cpu_candidates_raw(
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
        state = _sequence_state(self)
        with state.lock:
            if state.native.closed:
                raise RuntimeError("sequence batch is closed")
            for profile in profiles:
                if profile.alphabet != state.alphabet:
                    raise ValueError("profile and sequence alphabets differ")
            packed = _pack_profiles(profiles)
            raw = cast(
                list[list[tuple[Any, ...]]],
                state.native.filter_many_raw(*packed),
            )
        return [_format_results(profile_results) for profile_results in raw]

    def cpu_candidates_many(
        self, optimized_profiles: Iterable[Any], F1: float = 0.02
    ) -> list[list[int]]:
        profiles = list(optimized_profiles)
        state = _sequence_state(self)
        with state.lock:
            if state.native.closed:
                raise RuntimeError("sequence batch is closed")
            for profile in profiles:
                if profile.alphabet != state.alphabet:
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
                    state.native.cpu_candidates_many_raw(
                        *packed, m_mu, m_lambda, threshold
                    ),
                )
                for index, profile_candidates in zip(
                    valid_indices, candidates, strict=True
                ):
                    output[index] = profile_candidates

            return cast(list[list[int]], output)

    def _candidate_csr_locked(
        self, profiles: list[Any], threshold: float
    ) -> tuple[array[int], array[int], bytes, array[int]]:
        """Build complete CSR rows, promoting invalid profiles fail-closed."""
        state = _sequence_state(self)
        if not profiles:
            return array("I"), array("Q", [0]), b"", array("I")
        if threshold == 1.0:
            return (
                array("I"),
                array("Q", [0]) * (len(profiles) + 1),
                bytes([1]) * len(profiles),
                array("I", range(len(state.native))),
            )

        parameters = [_f1_parameters(profile) for profile in profiles]
        parameters = [
            profile_parameters
            if (
                profile_parameters is not None
                and _native.f1_cutoff(
                    profile_parameters[0], profile_parameters[1], threshold
                )[0]
                != _native.F1_CUTOFF_INVALID
            )
            else None
            for profile_parameters in parameters
        ]
        valid_profiles = [
            profile
            for profile, profile_parameters in zip(profiles, parameters, strict=True)
            if profile_parameters is not None
        ]
        valid_indices = array("I")
        valid_offsets = array("Q", [0])
        if valid_profiles:
            m_mu = array(
                "f",
                [
                    cast(tuple[float, float, float], profile_parameters)[0]
                    for profile_parameters in parameters
                    if profile_parameters is not None
                ],
            )
            m_lambda = array(
                "f",
                [
                    cast(tuple[float, float, float], profile_parameters)[1]
                    for profile_parameters in parameters
                    if profile_parameters is not None
                ],
            )
            packed = _pack_profiles(valid_profiles)
            valid_indices, valid_offsets = cast(
                tuple[array[int], array[int]],
                state.native.cpu_candidates_many_csr_raw(
                    *packed, m_mu, m_lambda, threshold
                ),
            )

        if len(valid_profiles) == len(profiles):
            return (
                valid_indices,
                valid_offsets,
                bytes(len(profiles)),
                array("I"),
            )

        all_targets = array("I", range(len(state.native)))
        indices = array("I")
        offsets = array("Q", [0])
        all_rows = bytearray()
        valid_row = 0
        for profile_parameters in parameters:
            if profile_parameters is None:
                all_rows.append(1)
            else:
                start = valid_offsets[valid_row]
                stop = valid_offsets[valid_row + 1]
                indices.frombytes(memoryview(valid_indices)[start:stop].cast("B"))
                valid_row += 1
                all_rows.append(0)
            offsets.append(len(indices))
        return indices, offsets, bytes(all_rows), all_targets

    def candidate_batch(
        self,
        profile_pairs: Iterable[PressedProfilePair],
        F1: float = 0.02,
    ) -> CandidateBatch:
        """Bind exact CUDA candidates to immutable pressed-profile pairs."""
        sequence_state = _sequence_state(self)
        pairs = tuple(profile_pairs)
        try:
            threshold = float(F1)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("F1 must be a finite number in [0, 1]") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("F1 must be a finite number in [0, 1]")

        profiles = []
        states = []
        for pair in pairs:
            if type(pair) is not PressedProfilePair:
                raise TypeError(
                    "candidate batches require pairs from load_pressed_profiles"
                )
            state = _pair_state(pair)
            if state.hmm.alphabet != sequence_state.alphabet:
                raise ValueError("profile and sequence alphabets differ")
            profiles.append(state.optimized_profile)
            states.append(state)

        unique_locks = {id(state.lock): state.lock for state in states}
        with ExitStack() as locks:
            for lock_id in sorted(unique_locks):
                locks.enter_context(unique_locks[lock_id])
            with sequence_state.lock:
                if sequence_state.native.closed:
                    raise RuntimeError("sequence batch is closed")
                indices, offsets, all_rows, all_targets = self._candidate_csr_locked(
                    profiles, threshold
                )

        return _new_candidate_batch(
            pairs,
            sequence_state.targets,
            indices,
            offsets,
            all_rows,
            all_targets,
            threshold,
        )


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
