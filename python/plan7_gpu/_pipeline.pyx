# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Candidate-aware companion for PyHMMER 0.12.0's comparison pipeline.

This module deliberately cimports PyHMMER private state.  It must be rebuilt
for any PyHMMER version change; the runtime guard below prevents accidentally
loading it against an unsupported private ABI.
"""

from libc.stddef cimport size_t
from libc.math cimport isfinite
from libc.stdint cimport int16_t, uint8_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy

from libeasel cimport eslERRBUFSIZE, eslEINVAL, eslERANGE, eslOK
from libeasel.sq cimport ESL_SQ
from libhmmer.impl.p7_oprofile cimport (
    P7_OPROFILE,
    p7_oprofile_Compare,
    p7_oprofile_ReconfigLength,
)
from libhmmer.p7_bg cimport (
    P7_BG,
    p7_bg_FilterScore,
    p7_bg_SetFilter,
    p7_bg_SetLength,
)
from libhmmer.p7_pipeline cimport (
    P7_PIPELINE,
    p7_SEARCH_SEQS,
    p7_ZSETBY_NTARGETS,
    p7_Pipeline,
    p7_pipeline_Reuse,
    p7_pli_NewModel,
)
from libhmmer.p7_tophits cimport P7_TOPHITS

from pyhmmer.easel cimport DigitalSequence, DigitalSequenceBlock
from pyhmmer.plan7 cimport HMM, OptimizedProfile, Pipeline, TopHits

import pyhmmer as _pyhmmer
from pyhmmer.errors import AlphabetMismatch, UnexpectedError
import importlib.util as _importlib_util
from pathlib import Path as _Path


cdef extern from "dlfcn.h" nogil:
    ctypedef struct Dl_info:
        const char* dli_fname
        void* dli_fbase

    int RTLD_NOLOAD
    int RTLD_NOW
    int dladdr(const void* address, Dl_info* info)
    int dlclose(void* handle)
    void* dlopen(const char* filename, int flags)
    void* dlsym(void* handle, const char* symbol)


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

cdef enum:
    BIAS_CPU_REQUIRED = 0
    BIAS_DEFINITE_REJECT = 1
    BIAS_DEFINITE_PASS = 2
    SSV_OK = 0
    P7_VIT_CPU = 2


cdef struct _bias_result:
    uint32_t sequence_index
    float filtersc
    int16_t ssv_numerator
    uint8_t ssv_status
    uint8_t action


ctypedef int (*_pipeline_from_filter_scores_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    int,
    float,
) noexcept nogil


cdef union _float_bits:
    float value
    uint32_t bits


def _bias_filter_score_bits(
    Pipeline pipeline,
    OptimizedProfile optimized_profile,
    DigitalSequence sequence,
):
    """Return HMMER 3.4 ``p7_bg_FilterScore`` as raw binary32 bits."""
    cdef float filtersc
    cdef _float_bits encoded
    cdef int status
    if not pipeline.alphabet._eq(optimized_profile.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, optimized_profile.alphabet)
    if not pipeline.alphabet._eq(sequence.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sequence.alphabet)
    if sequence._sq.n == 0 or sequence._sq.n > HMMER_TARGET_LIMIT:
        raise ValueError("bias oracle target length must be in [1, 100000]")
    with nogil:
        status = p7_bg_SetFilter(
            pipeline.background._bg,
            optimized_profile._om.M,
            optimized_profile._om.compo,
        )
        if status == eslOK:
            status = p7_bg_SetLength(pipeline.background._bg, sequence._sq.n)
        if status == eslOK:
            status = p7_bg_FilterScore(
                pipeline.background._bg,
                sequence._sq.dsq,
                sequence._sq.n,
                &filtersc,
            )
    if status != eslOK:
        raise UnexpectedError(status, "p7_bg_FilterScore")
    encoded.value = filtersc
    return encoded.bits


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


cdef _pipeline_from_filter_scores_f _resolve_filter_scores_seam() noexcept nogil:
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(handle, "p7_PipelineFromFilterScores")
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        dlclose(handle)
        return NULL
    dlclose(handle)
    return <_pipeline_from_filter_scores_f> symbol


def _filter_scores_seam_available():
    """Return whether the project-private HMMER continuation seam is loaded."""
    cdef _pipeline_from_filter_scores_f seam
    with nogil:
        seam = _resolve_filter_scores_seam()
    return seam != NULL


cdef int _search_loop_candidates(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint32_t* candidate_indexes,
    size_t candidate_count,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
) except 1 nogil:
    """Mirror ``Pipeline._search_loop`` while skipping definite rejects."""
    cdef int status
    cdef size_t candidate_cursor
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for candidate_cursor in range(candidate_count):
        t = candidate_indexes[candidate_cursor]
        target_end = t + 1

        # Advance exactly the p7_pli_NewSeq accounting through this target.
        # Prefix residue offsets make gaps between candidates O(1), while Z
        # still has the value HMMER would see at this target's ordinal.
        if not pli.long_targets:
            pli.nseqs += target_end - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[target_end] - residue_offsets[previous_end]

        # A rejected prefix would have reused any residual workspaces before
        # this first candidate in HMMER's ordinary target loop.
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

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    # Account the rejected tail in one step.
    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

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


cdef int _search_loop_bias(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint8_t* record_bytes,
    size_t record_count,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
    _pipeline_from_filter_scores_f filter_scores_seam,
) except 1 nogil:
    """Run one validated sparse bias row through HMMER's exact pipeline."""
    cdef _bias_result record
    cdef float usc
    cdef int status
    cdef size_t cursor
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for cursor in range(record_count):
        memcpy(
            &record,
            record_bytes + cursor * sizeof(_bias_result),
            sizeof(_bias_result),
        )
        t = record.sequence_index
        target_end = t + 1

        if not pli.long_targets:
            pli.nseqs += target_end - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[target_end] - residue_offsets[previous_end]

        if cursor == 0 and t != 0:
            p7_pipeline_Reuse(pli)

        status = p7_bg_SetLength(bg, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_bg_SetLength")
        status = p7_oprofile_ReconfigLength(om, sq[t].n)
        if status != eslOK:
            raise UnexpectedError(status, "p7_oprofile_ReconfigLength")

        if record.action == BIAS_CPU_REQUIRED:
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        else:
            usc = <float> record.ssv_numerator
            usc = usc / om.scale_b
            usc = usc - <float> 3.0
            status = filter_scores_seam(
                pli,
                om,
                bg,
                sq[t],
                NULL,
                th,
                usc,
                record.filtersc,
                P7_VIT_CPU,
                <float> 0.0,
            )
        if status == eslEINVAL:
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

    if (
        n_targets != 0
        and (
            record_count == 0
            or record.sequence_index != n_targets - 1
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
    bint validate_target_lengths,
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
        validate_target_lengths
        and sequences._length != 0
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
    const uint64_t* residue_offsets,
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
            residue_offsets,
            hits._th,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    # Preserve Astra/PyHMMER query metadata and object identity.  In
    # particular, never expose the lockstep OptimizedProfile as hits.query.
    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _search_bias_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
    const uint64_t* residue_offsets,
    _pipeline_from_filter_scores_f filter_scores_seam,
):
    cdef const uint8_t* record_ptr = NULL
    cdef size_t record_count = (
        <size_t> bias_records.shape[0] // sizeof(_bias_result)
    )
    cdef TopHits hits = TopHits(query)

    if bias_records.shape[0] != 0:
        record_ptr = &bias_records[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_bias(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            record_ptr,
            record_count,
            residue_offsets,
            hits._th,
            filter_scores_seam,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    hits._query = query
    hits._empty = False
    return hits


cdef void _validate_candidate_indexes(
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
) except *:
    cdef size_t cursor
    cdef uint32_t index
    cdef uint32_t previous = 0

    for cursor in range(<size_t> candidate_indexes.shape[0]):
        index = candidate_indexes[cursor]
        if index >= sequences._length:
            raise IndexError(f"candidate index out of range: {index}")
        if cursor != 0 and index <= previous:
            raise ValueError(
                "candidate indexes must be strictly increasing and unique"
            )
        previous = index


cdef bint _validate_bias_records(
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
) except -1:
    cdef _bias_result record
    cdef bint has_direct = False
    cdef size_t cursor
    cdef size_t record_count
    cdef uint32_t previous = 0

    if sizeof(_bias_result) != 12:
        raise RuntimeError("native bias result ABI is not 12 bytes")
    if <size_t> bias_records.shape[0] % sizeof(_bias_result) != 0:
        raise ValueError("bias result row has trailing bytes")
    record_count = <size_t> bias_records.shape[0] // sizeof(_bias_result)

    for cursor in range(record_count):
        memcpy(
            &record,
            &bias_records[cursor * sizeof(_bias_result)],
            sizeof(_bias_result),
        )
        if record.sequence_index >= sequences._length:
            raise IndexError(
                f"bias result sequence index out of range: "
                f"{record.sequence_index}"
            )
        if cursor != 0 and record.sequence_index <= previous:
            raise ValueError(
                "bias result sequence indexes must be strictly increasing "
                "and unique"
            )
        if record.action == BIAS_CPU_REQUIRED:
            pass
        elif (
            record.action == BIAS_DEFINITE_REJECT
            or record.action == BIAS_DEFINITE_PASS
        ):
            if record.ssv_status != SSV_OK:
                raise ValueError("direct bias result requires eslOK SSV status")
            if not isfinite(record.filtersc):
                raise ValueError("direct bias result requires finite filter score")
            if sequences._refs[record.sequence_index].n == 0:
                raise ValueError("empty target requires CPU bias fallback")
            has_direct = True
        else:
            raise ValueError(f"unknown bias result action: {record.action}")
        previous = record.sequence_index

    if has_direct and (
        not isfinite(optimized_profile._om.scale_b)
        or optimized_profile._om.scale_b <= 0.0
    ):
        raise ValueError("direct bias results require a positive finite SSV scale")
    return has_direct


cdef void _validate_bias_residue_offsets(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
    const uint64_t[::1] residue_offsets,
) except *:
    cdef _bias_result record
    cdef size_t cursor
    cdef size_t record_count = (
        <size_t> bias_records.shape[0] // sizeof(_bias_result)
    )
    cdef size_t target
    cdef size_t target_end
    cdef size_t previous_end = 0

    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")

    for cursor in range(record_count):
        memcpy(
            &record,
            &bias_records[cursor * sizeof(_bias_result)],
            sizeof(_bias_result),
        )
        target = record.sequence_index
        target_end = target + 1
        if residue_offsets[target_end] < residue_offsets[previous_end]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target_end] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")
        previous_end = target_end
    if residue_offsets[sequences._length] < residue_offsets[previous_end]:
        raise ValueError("target residue prefix is not monotone")
    if sequences._length != 0 and (
        residue_offsets[sequences._length]
        - residue_offsets[sequences._length - 1]
        != <uint64_t> sequences._refs[sequences._length - 1].n
    ):
        raise ValueError("target residue prefix differs from final target length")


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
    cdef uint64_t* residue_offsets = NULL
    cdef size_t target

    _validate_inputs(pipeline, query, optimized_profile, sequences, True)
    _validate_candidate_indexes(sequences, candidate_indexes)

    residue_offsets = <uint64_t*> malloc(
        (sequences._length + 1) * sizeof(uint64_t)
    )
    if residue_offsets == NULL:
        raise MemoryError("target residue-prefix allocation failed")
    residue_offsets[0] = 0
    for target in range(sequences._length):
        residue_offsets[target + 1] = (
            residue_offsets[target] + sequences._refs[target].n
        )
    try:
        return _search_validated(
            pipeline,
            query,
            optimized_profile,
            sequences,
            candidate_indexes,
            residue_offsets,
        )
    finally:
        free(residue_offsets)


def _search_hmm_candidates_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint32_t[::1] candidate_indexes,
    const uint64_t[::1] residue_offsets,
):
    """Consume one provenance-bound CSR row and cumulative residue offsets.

    This is a package-private fast path. ``SequenceBatch`` creates both
    buffers from the same concealed target snapshot and keeps them immutable;
    callers outside the package cannot establish that provenance safely.
    """
    cdef size_t cursor
    cdef size_t target
    cdef size_t target_end
    cdef size_t previous_end = 0

    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    _validate_candidate_indexes(sequences, candidate_indexes)
    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")

    # Validate every prefix entry the sparse loop will dereference. Full
    # target-by-target validation belongs to SequenceBatch construction.
    for cursor in range(<size_t> candidate_indexes.shape[0]):
        target = candidate_indexes[cursor]
        target_end = target + 1
        if residue_offsets[target_end] < residue_offsets[previous_end]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target_end] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")
        previous_end = target_end
    if residue_offsets[sequences._length] < residue_offsets[previous_end]:
        raise ValueError("target residue prefix is not monotone")
    if sequences._length != 0 and (
        residue_offsets[sequences._length]
        - residue_offsets[sequences._length - 1]
        != <uint64_t> sequences._refs[sequences._length - 1].n
    ):
        raise ValueError("target residue prefix differs from final target length")

    return _search_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        candidate_indexes,
        &residue_offsets[0],
    )


def _search_hmm_bias_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] bias_records,
    const uint64_t[::1] residue_offsets,
):
    """Consume one provenance-bound row of exact CUDA bias records.

    The caller must hold the same exclusive ``Pipeline`` and immutable target
    ownership used by ``_search_hmm_candidates_bound``. Records absent from
    the row are definite F1 rejects. CPU-required records run the ordinary
    HMMER pipeline; direct records enter the patched HMMER continuation seam,
    which rechecks F1, the current bias setting, and F2 before CPU Viterbi.
    """
    cdef bint has_direct
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly pyhmmer.plan7.Pipeline")
    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    has_direct = _validate_bias_records(
        optimized_profile, sequences, bias_records
    )
    _validate_bias_residue_offsets(sequences, bias_records, residue_offsets)

    if has_direct:
        with nogil:
            filter_scores_seam = _resolve_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct bias records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

    return _search_bias_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        bias_records,
        &residue_offsets[0],
        filter_scores_seam,
    )
