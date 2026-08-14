# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Candidate-aware companion for PyHMMER 0.12.0's comparison pipeline.

This module deliberately cimports PyHMMER private state.  It must be rebuilt
for any PyHMMER version change; the runtime guard below prevents accidentally
loading it against an unsupported private ABI.
"""

from libc.stddef cimport size_t
from libc.stdint cimport uint32_t

from libeasel cimport eslERRBUFSIZE, eslEINVAL, eslERANGE, eslOK
from libeasel.sq cimport ESL_SQ
from libhmmer.impl.p7_oprofile cimport (
    P7_OPROFILE,
    p7_oprofile_Compare,
    p7_oprofile_ReconfigLength,
)
from libhmmer.p7_bg cimport P7_BG, p7_bg_SetLength
from libhmmer.p7_pipeline cimport (
    P7_PIPELINE,
    p7_SEARCH_SEQS,
    p7_ZSETBY_NTARGETS,
    p7_Pipeline,
    p7_pipeline_Reuse,
    p7_pli_NewModel,
)
from libhmmer.p7_tophits cimport P7_TOPHITS

from pyhmmer.easel cimport DigitalSequenceBlock
from pyhmmer.plan7 cimport HMM, OptimizedProfile, Pipeline, TopHits

import pyhmmer as _pyhmmer
from pyhmmer.errors import AlphabetMismatch, UnexpectedError
import importlib.util as _importlib_util
from pathlib import Path as _Path


_abi_spec = _importlib_util.spec_from_file_location(
    "_plan7_gpu_pyhmmer_abi", _Path(__file__).resolve().with_name("_abi.py")
)
if _abi_spec is None or _abi_spec.loader is None:
    raise ImportError("cannot load the plan7_gpu PyHMMER ABI verifier")
_abi_module = _importlib_util.module_from_spec(_abi_spec)
_abi_spec.loader.exec_module(_abi_module)


PYHMMER_PRIVATE_ABI = "0.12.0"
PYHMMER_PRIVATE_ABI_SHA256 = PYHMMER_ABI_SHA256
if _pyhmmer.__version__ != PYHMMER_PRIVATE_ABI:
    raise ImportError(
        f"plan7_gpu._pipeline requires PyHMMER {PYHMMER_PRIVATE_ABI}, "
        f"found {_pyhmmer.__version__}"
    )
_abi_module.validate_private_abi_platform()
_runtime_abi_sha256 = _abi_module.pyhmmer_abi_fingerprint()
if _runtime_abi_sha256 != PYHMMER_PRIVATE_ABI_SHA256:
    raise ImportError(
        "plan7_gpu._pipeline was built against a different PyHMMER private ABI "
        f"({PYHMMER_PRIVATE_ABI_SHA256} != {_runtime_abi_sha256})"
    )


cdef size_t HMMER_TARGET_LIMIT = 100000


def _oprofiles_equal_hmmer(
    OptimizedProfile expected,
    OptimizedProfile observed,
):
    """Run HMMER's own optimized-profile comparator at zero tolerance."""
    cdef char error[eslERRBUFSIZE]
    cdef int status

    error[0] = 0
    with nogil:
        status = p7_oprofile_Compare(expected._om, observed._om, 0.0, error)
    return status == eslOK


cdef int _search_loop_candidates(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint32_t* candidate_indexes,
    size_t candidate_count,
    P7_TOPHITS* th,
) except 1 nogil:
    """Mirror ``Pipeline._search_loop`` while skipping definite rejects."""
    cdef int status
    cdef size_t candidate_cursor = 0
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for t in range(n_targets):
        # This is exactly p7_pli_NewSeq's accounting for a search pipeline.
        # Doing it inline avoids a function call for every definite reject.
        if not pli.long_targets:
            pli.nseqs += 1
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += sq[t].n

        if (
            candidate_cursor < candidate_count
            and candidate_indexes[candidate_cursor] == t
        ):
            # A rejected prefix would have reused any residual workspaces
            # before this first candidate in HMMER's ordinary target loop.
            if candidate_cursor == 0 and t != 0:
                p7_pipeline_Reuse(pli)

            status = p7_bg_SetLength(bg, sq[t].n)
            if status != eslOK:
                raise UnexpectedError(status, "p7_bg_SetLength")
            status = p7_oprofile_ReconfigLength(om, sq[t].n)
            if status != eslOK:
                raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
            if status == eslEINVAL:
                Pipeline._missing_cutoffs(pli, om)
            elif status == eslERANGE:
                raise OverflowError(
                    "numerical overflow in the optimized vector implementation"
                )
            elif status != eslOK:
                raise UnexpectedError(status, "p7_Pipeline")
            candidate_cursor += 1

            p7_pipeline_Reuse(pli)

    # HMMER leaves both models configured for the final target, even when it
    # fails F1. Preserve that state with one configuration for a rejected tail.
    if (
        n_targets != 0
        and (
            candidate_count == 0
            or candidate_indexes[candidate_count - 1] != n_targets - 1
        )
    ):
        status = p7_bg_SetLength(bg, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[n_targets - 1].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")
        p7_pipeline_Reuse(pli)

    return 0


cdef void _validate_inputs(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
) except *:
    """Validate every input without mutating the supplied pipeline."""
    if not pipeline.alphabet._eq(query.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, query.alphabet)
    if not pipeline.alphabet._eq(optimized_profile.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, optimized_profile.alphabet)
    if not pipeline.alphabet._eq(sequences.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sequences.alphabet)

    if query._hmm.M != optimized_profile._om.M:
        raise ValueError(
            "HMM and optimized profile model lengths differ: "
            f"{query._hmm.M} != {optimized_profile._om.M}"
        )
    if (
        query.name != optimized_profile.name
        or query.accession != optimized_profile.accession
    ):
        raise ValueError("HMM and optimized profile metadata differ")

    if (
        sequences._length != 0
        and len(sequences.largest()) > HMMER_TARGET_LIMIT
    ):
        raise ValueError(
            f"sequence length over comparison pipeline limit "
            f"({HMMER_TARGET_LIMIT})"
        )


cdef TopHits _search_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
):
    cdef const uint32_t* candidate_ptr = NULL
    cdef TopHits hits = TopHits(query)

    if candidate_indexes.shape[0] != 0:
        candidate_ptr = &candidate_indexes[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_candidates(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            candidate_ptr,
            <size_t> candidate_indexes.shape[0],
            hits._th,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    # Preserve Astra/PyHMMER query metadata and object identity.  In
    # particular, never expose the lockstep OptimizedProfile as hits.query.
    hits._query = query
    hits._empty = False
    return hits


def _search_hmm_candidates(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
):
    """Internal primitive consuming one sorted native-endian uint32 CSR row.

    Structural validation cannot establish candidate provenance.  Callers
    outside the package must not use this function directly.  The candidate
    buffer and all PyHMMER inputs must remain alive and unmodified until the
    call returns; the provenance-bound adapter is responsible for enforcing
    that ownership and synchronization contract.
    """
    cdef size_t cursor
    cdef uint32_t index
    cdef uint32_t previous = 0

    _validate_inputs(pipeline, query, optimized_profile, sequences)

    # Fully validate the typed CSR row before changing pipeline state.
    for cursor in range(<size_t> candidate_indexes.shape[0]):
        index = candidate_indexes[cursor]
        if index >= sequences._length:
            raise IndexError(f"candidate index out of range: {index}")
        if cursor != 0 and index <= previous:
            raise ValueError(
                "candidate indexes must be strictly increasing and unique"
            )
        previous = index

    return _search_validated(
        pipeline, query, optimized_profile, sequences, candidate_indexes
    )
