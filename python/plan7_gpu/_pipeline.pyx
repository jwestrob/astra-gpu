# cython: language_level=3
# cython: boundscheck=False, wraparound=False, initializedcheck=False

"""Candidate-aware companion for PyHMMER 0.12.0's comparison pipeline.

This module deliberately cimports PyHMMER private state.  It must be rebuilt
for any PyHMMER version change; the runtime guard below prevents accidentally
loading it against an unsupported private ABI.
"""

from libc.stddef cimport size_t
from libc.math cimport isfinite, isnan
from libc.stdint cimport int16_t, int32_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdlib cimport free, malloc
from libc.string cimport memcmp, memcpy
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.buffer cimport Py_buffer, PyBuffer_FillInfo
from cpython.pycapsule cimport (
    PyCapsule_Destructor,
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_SetDestructor,
    PyCapsule_SetName,
    PyCapsule_SetPointer,
)
from cpython.pyport cimport PY_SSIZE_T_MAX

from libeasel cimport (
    eslCONST_LOG2,
    eslERRBUFSIZE,
    eslEINACCURATE,
    eslEINVAL,
    eslERANGE,
    eslOK,
)
from libeasel.sq cimport ESL_SQ
from libhmmer cimport p7_MLAMBDA, p7_MMU, p7_VLAMBDA, p7_VMU
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
from array import array as _array
import importlib.util as _importlib_util
from pathlib import Path as _Path
from threading import Lock as _Lock

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


cdef extern from "esl_gumbel.h" nogil:
    double esl_gumbel_surv(double, double, double)


cdef extern from "continuation_journal.h":
    const char *PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    const char *PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME

    cdef enum plan7_continuation_journal_abi:
        PLAN7_CONTINUATION_JOURNAL_VERSION
        PLAN7_CONTINUATION_JOURNAL_MAGIC
        PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE

    cdef enum plan7_domain_rescore_abi:
        PLAN7_DOMAIN_RESCORE_RECORD_VERSION
        PLAN7_DOMAIN_RESCORE_RECORD_SIZE
        PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE
        PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES

    cdef enum plan7_domain_rescore_action:
        PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT

    cdef enum plan7_domain_rescore_status:
        PLAN7_DOMAIN_RESCORE_OK
        PLAN7_DOMAIN_RESCORE_ERANGE
        PLAN7_DOMAIN_RESCORE_ENORESULT
        PLAN7_DOMAIN_RESCORE_ECAP
        PLAN7_DOMAIN_RESCORE_EMPTY

    ctypedef struct plan7_forward_provenance:
        uint64_t database_generation
        uint64_t batch_generation
        uint64_t row_hash
        uint64_t special_hash
        uint64_t continuation_hash
        uint64_t pass_count
        uint64_t special_count
        uint64_t generation_f3_bits
        uint64_t integrity_tag

    ctypedef struct plan7_backward_domain_provenance:
        plan7_forward_provenance forward
        uint64_t threshold_hash
        uint64_t result_hash
        uint64_t region_hash
        uint64_t candidate_count
        uint64_t region_count

    ctypedef struct plan7_domain_rescore_result:
        uint32_t row_index
        uint32_t profile_index
        uint32_t sequence_index
        uint32_t envelope_begin
        uint32_t envelope_end
        uint32_t alignment_begin
        uint32_t alignment_end
        uint32_t model_begin
        uint32_t model_end
        float forward_score
        float backward_score
        float oa_score
        float domain_correction
        float score_consistency
        uint8_t status
        uint8_t action
        uint8_t has_own_scales
        uint8_t reserved
        uint32_t reserved2

    ctypedef struct plan7_domain_rescore_trace_step:
        uint32_t sequence_position
        uint32_t model_position
        float posterior
        uint8_t state
        uint8_t reserved[3]

    ctypedef struct plan7_domain_rescore_provenance:
        plan7_backward_domain_provenance backward
        uint64_t result_hash
        uint64_t trace_hash
        uint64_t null2_hash
        uint64_t result_count
        uint64_t trace_count
        uint64_t null2_count

    cdef enum plan7_continuation_compact_route:
        PLAN7_CONTINUATION_COMPACT_NONE
        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
        PLAN7_CONTINUATION_COMPACT_DEVICE

    ctypedef struct plan7_simple_region:
        uint32_t begin
        uint32_t end

    ctypedef struct plan7_continuation_journal_row:
        uint32_t profile_index
        uint32_t sequence_index
        float usc
        float filtersc
        float vfsc
        float fwdsc
        float backward_score
        float nexpected
        uint32_t uncertain_count
        uint32_t region_count
        uint32_t multidomain_count
        uint8_t postfilter_status
        uint8_t postfilter_action
        uint8_t forward_status
        uint8_t forward_action
        uint8_t domain_status
        uint8_t domain_route
        uint8_t has_own_scales
        uint8_t reserved
        uint32_t compact_result_count
        uint8_t compact_route
        uint8_t reserved2[3]
        uint32_t reserved3

    ctypedef struct plan7_continuation_journal:
        uint32_t magic
        uint16_t version
        uint16_t header_size
        uint32_t row_size
        uint32_t region_size
        uint32_t compact_result_size
        uint32_t compact_trace_step_size
        uint32_t compact_null2_stride
        uint64_t total_bytes
        uint64_t session_id
        uint64_t selection_id
        uint64_t profile_count
        uint64_t postfilter_count
        uint64_t forward_count
        uint64_t row_count
        uint64_t special_count
        uint64_t region_count
        uint64_t compact_result_count
        uint64_t compact_trace_offset_count
        uint64_t compact_trace_count
        uint64_t compact_null2_count
        uint64_t generation_tail_fingerprint
        uint64_t rescore_simple_row_count
        uint64_t rescore_device_result_count
        uint64_t rescore_cpu_required_count
        uint64_t rescore_numeric_fallback_count
        uint64_t rescore_cap_fallback_count
        uint64_t rescore_global_cpu_fallback_count
        uint64_t rescore_compact_output_byte_limit
        uint64_t rescore_compact_output_bytes
        uint64_t generation_f1_bits
        uint64_t generation_f2_bits
        uint64_t generation_f3_bits
        uint32_t rt1_bits
        uint32_t rt2_bits
        uint32_t rt3_bits
        uint32_t guard_band_bits
        uint8_t generation_bias_filter
        uint8_t generation_compact_domains
        uint8_t compact_global_fallback
        uint8_t sequence_content_fingerprint[32]
        uint64_t postfilter_offsets_offset
        uint64_t postfilter_records_offset
        uint64_t forward_offsets_offset
        uint64_t forward_records_offset
        uint64_t forward_special_offsets_offset
        uint64_t profile_offsets_offset
        uint64_t identity_tokens_offset
        uint64_t profile_fingerprints_offset
        uint64_t rows_offset
        uint64_t special_offsets_offset
        uint64_t specials_offset
        uint64_t region_offsets_offset
        uint64_t regions_offset
        uint64_t compact_row_offsets_offset
        uint64_t compact_results_offset
        uint64_t compact_trace_offsets_offset
        uint64_t compact_traces_offset
        uint64_t compact_null2_offset
        plan7_forward_provenance forward
        plan7_backward_domain_provenance backward
        plan7_domain_rescore_provenance rescore
        uint64_t integrity_tag

    uint64_t plan7_continuation_journal_integrity(
        const plan7_continuation_journal *journal,
    ) nogil

    int plan7_continuation_journal_rescore_hashes(
        const plan7_domain_rescore_result *results,
        uint64_t result_count,
        const uint64_t *trace_offsets,
        const plan7_domain_rescore_trace_step *traces,
        uint64_t trace_count,
        const float *null2,
        uint64_t null2_count,
        uint64_t *result_hash_out,
        uint64_t *trace_hash_out,
        uint64_t *null2_hash_out,
    ) nogil


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
    SSV_ENORESULT = 19
    FORWARD_CPU_REQUIRED = 0
    FORWARD_DEFINITE_REJECT = 1
    FORWARD_DEFINITE_PASS = 2
    FORWARD_OK = 0
    FORWARD_ERANGE = 16
    FORWARD_ENORESULT = 19
    FORWARD_EMPTY = 255
    DOMAIN_CPU_REQUIRED = 0
    DOMAIN_NO_REGIONS = 1
    DOMAIN_SIMPLE = 2
    DOMAIN_OK = 0
    DOMAIN_ERANGE = 16
    DOMAIN_ENORESULT = 19
    DOMAIN_EMPTY = 255
    P7_VIT_EXTERNAL = 1
    P7_VIT_CPU = 2


cdef struct _bias_result:
    uint32_t sequence_index
    float filtersc
    int16_t ssv_numerator
    uint8_t ssv_status
    uint8_t action


cdef struct _postfilter_result:
    uint32_t sequence_index
    float filtersc
    int16_t msv_numerator
    uint8_t msv_status
    uint8_t action
    float vfsc


cdef struct _forward_result:
    uint32_t sequence_index
    float fwdsc
    uint8_t status
    uint8_t action
    uint16_t reserved


cdef struct _compact_consumption_statistics:
    uint64_t attempt_count
    uint64_t accepted_count
    uint64_t invalid_retry_count
    uint64_t threshold_retry_count
    uint64_t first_row_index
    uint32_t first_profile_index
    uint32_t first_sequence_index
    uint64_t first_domain_count


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

ctypedef int (*_pipeline_from_filter_and_forward_scores_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    const float*,
    uint64_t,
) noexcept nogil

ctypedef int (*_pipeline_from_filter_and_forward_simple_regions_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    uint64_t,
    uint64_t,
    uint64_t,
    int,
    int,
    float,
    const plan7_simple_region*,
    uint64_t,
) noexcept nogil

ctypedef uint64_t (*_pipeline_compact_tail_fingerprint_f)(
    const P7_PIPELINE*,
) noexcept nogil

ctypedef int (*_pipeline_from_filter_forward_compact_domains_f)(
    P7_PIPELINE*,
    P7_OPROFILE*,
    P7_BG*,
    const ESL_SQ*,
    const ESL_SQ*,
    P7_TOPHITS*,
    float,
    float,
    float,
    float,
    uint64_t,
    uint64_t,
    uint32_t,
    uint32_t,
    uint32_t,
    float,
    const plan7_domain_rescore_result*,
    uint64_t,
    const uint64_t*,
    uint64_t,
    const plan7_domain_rescore_trace_step*,
    uint64_t,
    const float*,
    uint64_t,
) noexcept nogil


cdef union _float_bits:
    float value
    uint32_t bits


cdef union _double_bits:
    double value
    uint64_t bits


cdef struct _pipeline_tail_snapshot:
    uint64_t f1_bits
    uint64_t f2_bits
    uint64_t f3_bits
    uint64_t E_bits
    uint64_t T_bits
    uint64_t domE_bits
    uint64_t domT_bits
    uint64_t incE_bits
    uint64_t incT_bits
    uint64_t incdomE_bits
    uint64_t incdomT_bits
    uint64_t Z_bits
    uint64_t domZ_bits
    uint32_t rt1_bits
    uint32_t rt2_bits
    uint32_t rt3_bits
    int32_t do_biasfilter
    int32_t do_null2
    int32_t do_alignment_score_calc
    int32_t by_E
    int32_t dom_by_E
    int32_t inc_by_E
    int32_t incdom_by_E
    int32_t use_bit_cutoffs
    int32_t Z_setby
    int32_t domZ_setby
    int32_t mode
    int32_t long_targets


cdef class _SealedPostfilterBatch:
    """Opaque, lifetime-pinned continuation data validated as one batch."""

    cdef bint _ready
    cdef tuple _queries
    cdef tuple _optimized_profiles
    cdef DigitalSequenceBlock _sequences
    cdef const uint8_t[::1] _postfilter_records
    cdef const uint64_t[::1] _postfilter_offsets
    cdef const uint64_t[::1] _residue_offsets
    cdef const uint8_t[::1] _forward_records
    cdef const uint64_t[::1] _forward_offsets
    cdef const uint64_t[::1] _special_offsets
    cdef const float[::1] _specials
    cdef const uint8_t[::1] _row_has_external
    cdef const uint8_t[::1] _background_fingerprint
    cdef const uint8_t[::1] _journal_storage
    cdef const uint64_t[::1] _journal_profile_offsets
    cdef const uint8_t[::1] _journal_rows
    cdef const uint64_t[::1] _journal_region_offsets
    cdef const uint8_t[::1] _journal_regions
    cdef const uint64_t[::1] _journal_compact_row_offsets
    cdef const uint8_t[::1] _journal_compact_results
    cdef const uint64_t[::1] _journal_compact_trace_offsets
    cdef const uint8_t[::1] _journal_compact_traces
    cdef const float[::1] _journal_compact_null2
    cdef double _f1
    cdef uint64_t _generation_f2_bits
    cdef uint64_t _generation_f3_bits
    cdef bint _generation_bias_filter
    cdef uint32_t _journal_guard_bits
    cdef uint64_t _generation_tail_fingerprint
    cdef uint64_t _rescore_simple_row_count
    cdef uint64_t _rescore_device_result_count
    cdef uint64_t _rescore_cpu_required_count
    cdef uint64_t _rescore_numeric_fallback_count
    cdef uint64_t _rescore_cap_fallback_count
    cdef uint64_t _rescore_global_cpu_fallback_count
    cdef _pipeline_tail_snapshot _pipeline_options
    cdef _pipeline_from_filter_scores_f _filter_scores_seam
    cdef _pipeline_from_filter_and_forward_scores_f _forward_scores_seam
    cdef _pipeline_from_filter_and_forward_simple_regions_f _simple_regions_seam
    cdef _pipeline_compact_tail_fingerprint_f _compact_tail_fingerprint
    cdef _pipeline_from_filter_forward_compact_domains_f _compact_domains_seam

    def __cinit__(self):
        self._ready = False

    def __repr__(self):
        return "<opaque sealed post-filter batch>"

    def __reduce__(self):
        raise TypeError("sealed post-filter batches cannot be pickled")


cdef class _ContinuationJournalStorage:
    """Read-only buffer exporting one transferred native journal allocation."""

    cdef uint8_t *_data
    cdef Py_ssize_t _size
    cdef bint _owns

    def __cinit__(self):
        self._data = NULL
        self._size = 0
        self._owns = False

    def __dealloc__(self):
        if self._owns and self._data != NULL:
            free(self._data)

    def __getbuffer__(self, Py_buffer *view, int flags):
        if self._data == NULL:
            raise BufferError("continuation journal storage is unavailable")
        PyBuffer_FillInfo(
            view, self, self._data, self._size, 1, flags
        )

    def __releasebuffer__(self, Py_buffer *view):
        pass


cdef _pipeline_from_filter_scores_f _filter_scores_seam_cache = NULL
cdef _pipeline_from_filter_and_forward_scores_f _forward_scores_seam_cache = NULL
cdef _pipeline_from_filter_and_forward_simple_regions_f _simple_regions_seam_cache = NULL
cdef _pipeline_compact_tail_fingerprint_f _compact_tail_fingerprint_cache = NULL
cdef _pipeline_from_filter_forward_compact_domains_f _compact_domains_seam_cache = NULL
cdef bint _filter_scores_seam_resolved = False
cdef bint _forward_scores_seam_resolved = False
cdef bint _simple_regions_seam_resolved = False
cdef bint _compact_domains_seam_resolved = False
cdef bint _filter_scores_same_dso = False
cdef bint _forward_scores_same_dso = False
cdef bint _simple_regions_same_dso = False
cdef bint _compact_domains_same_dso = False
cdef uint64_t _filter_scores_resolutions = 0
cdef uint64_t _forward_scores_resolutions = 0
cdef uint64_t _filter_scores_dlopen_calls = 0
cdef uint64_t _forward_scores_dlopen_calls = 0
cdef uint64_t _filter_scores_dlclose_calls = 0
cdef uint64_t _forward_scores_dlclose_calls = 0
cdef uint64_t _simple_regions_resolutions = 0
cdef uint64_t _simple_regions_dlopen_calls = 0
cdef uint64_t _simple_regions_dlclose_calls = 0
cdef uint64_t _compact_domains_resolutions = 0
cdef uint64_t _compact_domains_dlopen_calls = 0
cdef uint64_t _compact_domains_dlclose_calls = 0

_continuation_seam_resolve_lock = _Lock()
cdef uint8_t _consumed_journal_sentinel = 0


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
    global _filter_scores_dlopen_calls
    global _filter_scores_dlclose_calls
    global _filter_scores_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _filter_scores_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(handle, "p7_PipelineFromFilterScores")
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _filter_scores_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _filter_scores_same_dso = True
    _filter_scores_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_scores_f> symbol


cdef _pipeline_from_filter_scores_f _cached_filter_scores_seam():
    global _filter_scores_resolutions
    global _filter_scores_seam_cache
    global _filter_scores_seam_resolved

    if not _filter_scores_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _filter_scores_seam_resolved:
                _filter_scores_resolutions += 1
                _filter_scores_seam_cache = _resolve_filter_scores_seam()
                _filter_scores_seam_resolved = True
    return _filter_scores_seam_cache


def _filter_scores_seam_available():
    """Return whether the project-private HMMER continuation seam is loaded."""
    return _cached_filter_scores_seam() != NULL


cdef _pipeline_from_filter_and_forward_scores_f _resolve_filter_and_forward_scores_seam() noexcept nogil:
    global _forward_scores_dlopen_calls
    global _forward_scores_dlclose_calls
    global _forward_scores_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _forward_scores_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(handle, "p7_PipelineFromFilterAndForwardScores")
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _forward_scores_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _forward_scores_same_dso = True
    _forward_scores_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_and_forward_scores_f> symbol


cdef _pipeline_from_filter_and_forward_scores_f _cached_filter_and_forward_scores_seam():
    global _forward_scores_resolutions
    global _forward_scores_seam_cache
    global _forward_scores_seam_resolved

    if not _forward_scores_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _forward_scores_seam_resolved:
                _forward_scores_resolutions += 1
                _forward_scores_seam_cache = (
                    _resolve_filter_and_forward_scores_seam()
                )
                _forward_scores_seam_resolved = True
    return _forward_scores_seam_cache


def _filter_and_forward_scores_seam_available():
    """Return whether the exact external-Forward seam is loaded."""
    return _cached_filter_and_forward_scores_seam() != NULL


cdef _pipeline_from_filter_and_forward_simple_regions_f _resolve_simple_regions_seam() noexcept nogil:
    global _simple_regions_dlopen_calls
    global _simple_regions_dlclose_calls
    global _simple_regions_same_dso
    cdef Dl_info info
    cdef Dl_info symbol_info
    cdef void* handle
    cdef void* symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return NULL
    _simple_regions_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return NULL
    symbol = dlsym(
        handle, "p7_PipelineFromFilterAndForwardSimpleRegions"
    )
    if (
        symbol == NULL
        or dladdr(symbol, &symbol_info) == 0
        or symbol_info.dli_fbase != info.dli_fbase
    ):
        _simple_regions_dlclose_calls += 1
        dlclose(handle)
        return NULL
    _simple_regions_same_dso = True
    _simple_regions_dlclose_calls += 1
    dlclose(handle)
    return <_pipeline_from_filter_and_forward_simple_regions_f> symbol


cdef _pipeline_from_filter_and_forward_simple_regions_f _cached_simple_regions_seam():
    global _simple_regions_resolutions
    global _simple_regions_seam_cache
    global _simple_regions_seam_resolved

    if not _simple_regions_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _simple_regions_seam_resolved:
                _simple_regions_resolutions += 1
                _simple_regions_seam_cache = _resolve_simple_regions_seam()
                _simple_regions_seam_resolved = True
    return _simple_regions_seam_cache


def _simple_regions_seam_available():
    """Return whether the guarded simple-region continuation seam is loaded."""
    return _cached_simple_regions_seam() != NULL


cdef void _resolve_compact_domain_seams() noexcept nogil:
    global _compact_domains_dlopen_calls
    global _compact_domains_dlclose_calls
    global _compact_domains_same_dso
    global _compact_domains_seam_cache
    global _compact_tail_fingerprint_cache
    cdef Dl_info info
    cdef Dl_info domain_info
    cdef Dl_info fingerprint_info
    cdef void* handle
    cdef void* domain_symbol
    cdef void* fingerprint_symbol

    if dladdr(<const void*> p7_Pipeline, &info) == 0 or info.dli_fname == NULL:
        return
    _compact_domains_dlopen_calls += 1
    handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_NOW)
    if handle == NULL:
        return
    domain_symbol = dlsym(
        handle, "p7_PipelineFromFilterForwardAndCompactDomainsV2"
    )
    fingerprint_symbol = dlsym(
        handle, "p7_pipeline_CompactTailFingerprintV2"
    )
    if (
        domain_symbol == NULL
        or fingerprint_symbol == NULL
        or dladdr(domain_symbol, &domain_info) == 0
        or dladdr(fingerprint_symbol, &fingerprint_info) == 0
        or domain_info.dli_fbase != info.dli_fbase
        or fingerprint_info.dli_fbase != info.dli_fbase
    ):
        _compact_domains_dlclose_calls += 1
        dlclose(handle)
        return
    _compact_domains_same_dso = True
    _compact_domains_seam_cache = (
        <_pipeline_from_filter_forward_compact_domains_f> domain_symbol
    )
    _compact_tail_fingerprint_cache = (
        <_pipeline_compact_tail_fingerprint_f> fingerprint_symbol
    )
    _compact_domains_dlclose_calls += 1
    dlclose(handle)


cdef _pipeline_from_filter_forward_compact_domains_f _cached_compact_domains_seam():
    global _compact_domains_resolutions
    global _compact_domains_seam_resolved

    if not _compact_domains_seam_resolved:
        with _continuation_seam_resolve_lock:
            if not _compact_domains_seam_resolved:
                _compact_domains_resolutions += 1
                with nogil:
                    _resolve_compact_domain_seams()
                _compact_domains_seam_resolved = True
    return _compact_domains_seam_cache


def _compact_domains_seam_available():
    """Return whether the same-DSO compact-domain seam pair is loaded."""
    return (
        _cached_compact_domains_seam() != NULL
        and _compact_tail_fingerprint_cache != NULL
    )


def _compact_tail_fingerprint_bound(Pipeline pipeline):
    """Capture the exact same-DSO compact-tail option fingerprint."""
    cdef uint64_t fingerprint
    if _cached_compact_domains_seam() == NULL:
        raise RuntimeError("the private compact-domain seam is unavailable")
    if _compact_tail_fingerprint_cache == NULL:
        raise RuntimeError("the compact-tail fingerprint seam is unavailable")
    with nogil:
        fingerprint = _compact_tail_fingerprint_cache(pipeline._pli)
    if fingerprint == 0:
        raise ValueError("pipeline compact-tail options are invalid")
    return fingerprint


def _continuation_seam_cache_info():
    """Return private resolver state for concurrency and lifetime tests."""
    return {
        "filter": {
            "resolved": bool(_filter_scores_seam_resolved),
            "available": _filter_scores_seam_cache != NULL,
            "same_dso": bool(_filter_scores_same_dso),
            "resolutions": _filter_scores_resolutions,
            "dlopen_calls": _filter_scores_dlopen_calls,
            "dlclose_calls": _filter_scores_dlclose_calls,
        },
        "forward": {
            "resolved": bool(_forward_scores_seam_resolved),
            "available": _forward_scores_seam_cache != NULL,
            "same_dso": bool(_forward_scores_same_dso),
            "resolutions": _forward_scores_resolutions,
            "dlopen_calls": _forward_scores_dlopen_calls,
            "dlclose_calls": _forward_scores_dlclose_calls,
        },
        "simple_regions": {
            "resolved": bool(_simple_regions_seam_resolved),
            "available": _simple_regions_seam_cache != NULL,
            "same_dso": bool(_simple_regions_same_dso),
            "resolutions": _simple_regions_resolutions,
            "dlopen_calls": _simple_regions_dlopen_calls,
            "dlclose_calls": _simple_regions_dlclose_calls,
        },
        "compact_domains": {
            "resolved": bool(_compact_domains_seam_resolved),
            "available": (
                _compact_domains_seam_cache != NULL
                and _compact_tail_fingerprint_cache != NULL
            ),
            "same_dso": bool(_compact_domains_same_dso),
            "resolutions": _compact_domains_resolutions,
            "dlopen_calls": _compact_domains_dlopen_calls,
            "dlclose_calls": _compact_domains_dlclose_calls,
        },
    }


cdef void _capture_pipeline_tail_options(
    Pipeline pipeline,
    _pipeline_tail_snapshot *snapshot,
) except *:
    cdef _double_bits encoded_double
    cdef _float_bits encoded_float
    cdef P7_PIPELINE *pli = pipeline._pli

    if pli == NULL or pli.ddef == NULL:
        raise RuntimeError("pipeline domain state is unavailable")
    encoded_double.value = pli.F1
    snapshot.f1_bits = encoded_double.bits
    encoded_double.value = pli.F2
    snapshot.f2_bits = encoded_double.bits
    encoded_double.value = pli.F3
    snapshot.f3_bits = encoded_double.bits
    encoded_double.value = pli.E
    snapshot.E_bits = encoded_double.bits
    encoded_double.value = pli.T
    snapshot.T_bits = encoded_double.bits
    encoded_double.value = pli.domE
    snapshot.domE_bits = encoded_double.bits
    encoded_double.value = pli.domT
    snapshot.domT_bits = encoded_double.bits
    encoded_double.value = pli.incE
    snapshot.incE_bits = encoded_double.bits
    encoded_double.value = pli.incT
    snapshot.incT_bits = encoded_double.bits
    encoded_double.value = pli.incdomE
    snapshot.incdomE_bits = encoded_double.bits
    encoded_double.value = pli.incdomT
    snapshot.incdomT_bits = encoded_double.bits
    encoded_double.value = pli.Z
    snapshot.Z_bits = encoded_double.bits
    encoded_double.value = pli.domZ
    snapshot.domZ_bits = encoded_double.bits
    encoded_float.value = pli.ddef.rt1
    snapshot.rt1_bits = encoded_float.bits
    encoded_float.value = pli.ddef.rt2
    snapshot.rt2_bits = encoded_float.bits
    encoded_float.value = pli.ddef.rt3
    snapshot.rt3_bits = encoded_float.bits
    snapshot.do_biasfilter = pli.do_biasfilter
    snapshot.do_null2 = pli.do_null2
    snapshot.do_alignment_score_calc = pli.do_alignment_score_calc
    snapshot.by_E = pli.by_E
    snapshot.dom_by_E = pli.dom_by_E
    snapshot.inc_by_E = pli.inc_by_E
    snapshot.incdom_by_E = pli.incdom_by_E
    snapshot.use_bit_cutoffs = pli.use_bit_cutoffs
    snapshot.Z_setby = pli.Z_setby
    snapshot.domZ_setby = pli.domZ_setby
    snapshot.mode = pli.mode
    snapshot.long_targets = pli.long_targets


cdef bint _pipeline_tail_options_match(
    const _pipeline_tail_snapshot *expected,
    Pipeline pipeline,
) except -1:
    cdef _pipeline_tail_snapshot observed
    _capture_pipeline_tail_options(pipeline, &observed)
    if (
        observed.f1_bits != expected.f1_bits
        or observed.f2_bits != expected.f2_bits
        or observed.f3_bits != expected.f3_bits
        or observed.E_bits != expected.E_bits
        or observed.domE_bits != expected.domE_bits
        or observed.incE_bits != expected.incE_bits
        or observed.incdomE_bits != expected.incdomE_bits
        or observed.rt1_bits != expected.rt1_bits
        or observed.rt2_bits != expected.rt2_bits
        or observed.rt3_bits != expected.rt3_bits
        or observed.do_biasfilter != expected.do_biasfilter
        or observed.do_null2 != expected.do_null2
        or observed.do_alignment_score_calc
        != expected.do_alignment_score_calc
        or observed.by_E != expected.by_E
        or observed.dom_by_E != expected.dom_by_E
        or observed.inc_by_E != expected.inc_by_E
        or observed.incdom_by_E != expected.incdom_by_E
        or observed.use_bit_cutoffs != expected.use_bit_cutoffs
        or observed.Z_setby != expected.Z_setby
        or observed.domZ_setby != expected.domZ_setby
        or observed.mode != expected.mode
        or observed.long_targets != expected.long_targets
    ):
        return False
    if expected.use_bit_cutoffs == 0 and (
        observed.T_bits != expected.T_bits
        or observed.domT_bits != expected.domT_bits
        or observed.incT_bits != expected.incT_bits
        or observed.incdomT_bits != expected.incdomT_bits
    ):
        return False
    if expected.Z_setby != p7_ZSETBY_NTARGETS and (
        observed.Z_bits != expected.Z_bits
    ):
        return False
    if expected.domZ_setby != p7_ZSETBY_NTARGETS and (
        observed.domZ_bits != expected.domZ_bits
    ):
        return False
    return True


def _validate_simple_region_generation_bound(
    Pipeline pipeline,
    double f1,
    double f2,
    double f3,
    bias_filter,
    double guard_band,
):
    """Validate product-domain options without mutating the pipeline."""
    cdef _pipeline_tail_snapshot snapshot
    cdef _double_bits encoded
    cdef _float_bits observed_float
    cdef _float_bits expected_float

    if type(bias_filter) is not bool or not bias_filter:
        raise ValueError("simple-region generation requires bias filtering")
    if (
        not isfinite(f1) or f1 < 0.0 or f1 > 1.0
        or not isfinite(f2) or f2 < 0.0 or f2 > 1.0
        or not isfinite(f3) or f3 < 0.0 or f3 > 1.0
        or not isfinite(guard_band)
        or guard_band < 2.0e-4 or guard_band > 1.0
    ):
        raise ValueError("invalid simple-region generation thresholds")
    _capture_pipeline_tail_options(pipeline, &snapshot)
    encoded.value = f1
    if encoded.bits != snapshot.f1_bits:
        raise ValueError("pipeline F1 differs from journal generation F1")
    encoded.value = f2
    if encoded.bits != snapshot.f2_bits:
        raise ValueError("pipeline F2 differs from journal generation F2")
    encoded.value = f3
    if encoded.bits != snapshot.f3_bits:
        raise ValueError("pipeline F3 differs from journal generation F3")
    if not snapshot.do_biasfilter:
        raise ValueError("pipeline bias filter is disabled")
    if snapshot.mode != p7_SEARCH_SEQS or snapshot.long_targets:
        raise ValueError("simple-region generation requires sequence-search mode")
    expected_float.value = <float> 0.25
    if snapshot.rt1_bits != expected_float.bits:
        raise ValueError("pipeline rt1 is not the calibrated default")
    expected_float.value = <float> 0.10
    if snapshot.rt2_bits != expected_float.bits:
        raise ValueError("pipeline rt2 is not the calibrated default")
    expected_float.value = <float> 0.20
    if snapshot.rt3_bits != expected_float.bits:
        raise ValueError("pipeline rt3 is not the calibrated default")
    observed_float.value = <float> guard_band
    return (0.25, 0.10, 0.20, observed_float.value)


def _select_forward_inputs_bound(
    profiles,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] postfilter_offsets,
    const uint64_t[::1] residue_offsets,
    double f2,
):
    """Select exact F2 survivors from trusted profile-major post-filter rows.

    Rows that fail F2 remain in the original post-filter stream and therefore
    take the ordinary CPU Forward continuation. The native Forward runner caps
    actual gathered F3-pass matrices; charging all F2 survivors here would
    discard work that ultimately needs no matrix.
    """
    cdef size_t profile_count = len(profiles)
    cdef size_t profile_index
    cdef size_t cursor
    cdef size_t start
    cdef size_t stop
    cdef size_t record_count
    cdef size_t sequence_count
    cdef uint32_t previous
    cdef bint have_previous
    cdef uint64_t sequence_length
    cdef float usc
    cdef float bit_score
    cdef double probability
    cdef _float_bits vfsc_bits
    cdef _postfilter_result record
    cdef OptimizedProfile profile
    cdef object value
    cdef object candidate_offsets = _array("Q", [0])
    cdef object candidate_indices = _array("I")
    cdef object filter_scores = _array("f")

    if sizeof(_postfilter_result) != 16:
        raise RuntimeError("native post-filter result ABI is not 16 bytes")
    if postfilter_offsets.shape[0] != profile_count + 1:
        raise ValueError("post-filter row-offset count differs from profiles")
    if postfilter_offsets[0] != 0:
        raise ValueError("post-filter row offsets must start at zero")
    if <size_t> postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result storage has trailing bytes")
    record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    if postfilter_offsets[profile_count] != record_count:
        raise ValueError("post-filter row offsets do not span result storage")
    if residue_offsets.shape[0] == 0 or residue_offsets[0] != 0:
        raise ValueError("target residue prefix must contain an initial zero")
    sequence_count = <size_t> residue_offsets.shape[0] - 1

    for profile_index in range(profile_count):
        start = <size_t> postfilter_offsets[profile_index]
        stop = <size_t> postfilter_offsets[profile_index + 1]
        if start > stop or stop > record_count:
            raise ValueError("post-filter row offsets are not monotone")
        value = profiles[profile_index]
        if not isinstance(value, _pyhmmer.plan7.OptimizedProfile):
            raise TypeError("Forward source profiles must be OptimizedProfile objects")
        profile = value
        previous = 0
        have_previous = False
        for cursor in range(start, stop):
            memcpy(
                &record,
                &postfilter_records[cursor * sizeof(_postfilter_result)],
                sizeof(_postfilter_result),
            )
            if record.sequence_index >= sequence_count:
                raise IndexError(
                    f"post-filter result sequence index out of range: "
                    f"{record.sequence_index}"
                )
            if have_previous and record.sequence_index <= previous:
                raise ValueError(
                    "post-filter result sequence indexes must be strictly "
                    "increasing and unique within each row"
                )
            previous = record.sequence_index
            have_previous = True

            if record.action != BIAS_DEFINITE_PASS:
                continue
            vfsc_bits.value = record.vfsc
            if (
                record.msv_status != SSV_OK
                or not isfinite(record.filtersc)
                or (
                    not isfinite(record.vfsc)
                    and vfsc_bits.bits != 0x7f800000
                )
                or not isfinite(profile._om.scale_b)
                or profile._om.scale_b <= 0.0
            ):
                continue

            usc = <float> record.msv_numerator
            usc = usc / profile._om.scale_b
            usc = usc - <float> 3.0
            bit_score = <float> ((usc - record.filtersc) / eslCONST_LOG2)
            probability = esl_gumbel_surv(
                bit_score,
                profile._om.evparam[<int> p7_MMU],
                profile._om.evparam[<int> p7_MLAMBDA],
            )
            if probability > f2:
                bit_score = <float> (
                    (record.vfsc - record.filtersc) / eslCONST_LOG2
                )
                probability = esl_gumbel_surv(
                    bit_score,
                    profile._om.evparam[<int> p7_VMU],
                    profile._om.evparam[<int> p7_VLAMBDA],
                )
                if probability > f2:
                    continue

            if residue_offsets[record.sequence_index + 1] < (
                residue_offsets[record.sequence_index]
            ):
                raise ValueError("target residue prefix is not monotone")
            sequence_length = (
                residue_offsets[record.sequence_index + 1]
                - residue_offsets[record.sequence_index]
            )
            if sequence_length > HMMER_TARGET_LIMIT:
                raise ValueError("Forward target exceeds HMMER's protein limit")
            candidate_indices.append(record.sequence_index)
            filter_scores.append(record.filtersc)
        candidate_offsets.append(len(candidate_indices))

    return candidate_offsets, candidate_indices, filter_scores


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


cdef int _search_loop_postfilter(
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
    cdef _postfilter_result record
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
            record_bytes + cursor * sizeof(_postfilter_result),
            sizeof(_postfilter_result),
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
        elif isnan(record.filtersc):
            # A fallback full-MSV score can reject at F1 after the public SSV
            # path conservatively retained it. Its row is accounted but does
            # not enter the pipeline continuation.
            status = eslOK
        else:
            usc = <float> record.msv_numerator
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
                P7_VIT_EXTERNAL,
                record.vfsc,
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


cdef int _search_loop_postfilter_forward(
    P7_PIPELINE* pli,
    P7_OPROFILE* om,
    P7_BG* bg,
    const ESL_SQ** sq,
    size_t n_targets,
    const uint8_t* postfilter_bytes,
    size_t postfilter_count,
    const uint8_t* forward_bytes,
    size_t forward_count,
    const uint64_t* special_offsets,
    const float* specials,
    const uint64_t* residue_offsets,
    P7_TOPHITS* th,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
    const uint8_t* journal_bytes,
    size_t journal_count,
    const uint64_t* journal_region_offsets,
    const uint8_t* journal_region_bytes,
    uint64_t generation_f1_bits,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    int generation_bias_filter,
    _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam,
    const uint64_t* journal_compact_row_offsets,
    const uint8_t* journal_compact_result_bytes,
    const uint64_t* compact_trace_offsets,
    const uint8_t* compact_trace_bytes,
    const float* compact_null2,
    uint64_t journal_row_base,
    uint64_t generation_tail_fingerprint,
    _pipeline_compact_tail_fingerprint_f compact_tail_fingerprint,
    _pipeline_from_filter_forward_compact_domains_f compact_domains_seam,
    uint64_t* compact_rebased_offsets,
    _compact_consumption_statistics* compact_statistics,
) except 1 nogil:
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef const float* xmx = NULL
    cdef uint64_t xmx_count = 0
    cdef float usc
    cdef int status
    cdef size_t cursor
    cdef size_t forward_cursor = 0
    cdef size_t journal_cursor = 0
    cdef size_t previous_end = 0
    cdef size_t target_end
    cdef size_t t
    cdef bint has_forward
    cdef bint used_forward_seam
    cdef bint used_simple_regions_seam
    cdef bint used_compact_domains_seam
    cdef bint compact_generation_matches = False
    cdef bint has_journal
    cdef plan7_continuation_journal_row journal
    cdef const plan7_simple_region* regions = NULL
    cdef uint64_t region_start
    cdef uint64_t region_stop
    cdef uint64_t compact_start
    cdef uint64_t compact_stop
    cdef uint64_t compact_count
    cdef uint64_t compact_index
    cdef uint64_t compact_trace_base
    cdef uint64_t compact_trace_stop
    cdef const plan7_domain_rescore_result* compact_domains = NULL
    cdef const plan7_domain_rescore_trace_step* compact_traces = NULL
    cdef const float* compact_row_null2 = NULL

    if (
        compact_tail_fingerprint != NULL
        and compact_domains_seam != NULL
        and generation_tail_fingerprint != 0
        and compact_tail_fingerprint(pli) == generation_tail_fingerprint
    ):
        compact_generation_matches = True

    status = p7_pli_NewModel(pli, om, bg)
    if status == eslEINVAL:
        Pipeline._missing_cutoffs(pli, om)
    elif status != eslOK:
        raise UnexpectedError(status, "p7_pli_NewModel")

    for cursor in range(postfilter_count):
        memcpy(
            &postfilter,
            postfilter_bytes + cursor * sizeof(_postfilter_result),
            sizeof(_postfilter_result),
        )
        t = postfilter.sequence_index
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

        has_forward = False
        if forward_cursor < forward_count:
            memcpy(
                &forward,
                forward_bytes + forward_cursor * sizeof(_forward_result),
                sizeof(_forward_result),
            )
            has_forward = forward.sequence_index == postfilter.sequence_index

        has_journal = False
        if journal_cursor < journal_count:
            memcpy(
                &journal,
                journal_bytes
                + journal_cursor * sizeof(plan7_continuation_journal_row),
                sizeof(plan7_continuation_journal_row),
            )
            has_journal = journal.sequence_index == postfilter.sequence_index

        used_forward_seam = False
        used_simple_regions_seam = False
        used_compact_domains_seam = False
        if postfilter.action == BIAS_CPU_REQUIRED:
            status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
        elif isnan(postfilter.filtersc):
            status = eslOK
        else:
            usc = <float> postfilter.msv_numerator
            usc = usc / om.scale_b
            usc = usc - <float> 3.0
            if (
                has_forward
                and has_journal
                and forward.action == FORWARD_DEFINITE_PASS
                and simple_regions_seam != NULL
                and journal.domain_status == DOMAIN_OK
                and (
                    journal.domain_route == DOMAIN_NO_REGIONS
                    or journal.domain_route == DOMAIN_SIMPLE
                )
                and not journal.has_own_scales
                and journal.uncertain_count == 0
                and journal.multidomain_count == 0
            ):
                region_start = journal_region_offsets[journal_cursor]
                region_stop = journal_region_offsets[journal_cursor + 1]
                if region_start == region_stop:
                    regions = NULL
                else:
                    regions = <const plan7_simple_region *> (
                        journal_region_bytes
                        + region_start * sizeof(plan7_simple_region)
                    )
                compact_start = journal_compact_row_offsets[journal_cursor]
                compact_stop = journal_compact_row_offsets[journal_cursor + 1]
                compact_count = compact_stop - compact_start
                if (
                    journal.domain_route == DOMAIN_SIMPLE
                    and journal.compact_route
                    == PLAN7_CONTINUATION_COMPACT_DEVICE
                    and compact_count != 0
                    and compact_generation_matches
                    and compact_rebased_offsets != NULL
                ):
                    if compact_statistics != NULL:
                        if compact_statistics.attempt_count == 0:
                            compact_statistics.first_row_index = (
                                journal_row_base + journal_cursor
                            )
                            compact_statistics.first_profile_index = (
                                journal.profile_index
                            )
                            compact_statistics.first_sequence_index = (
                                journal.sequence_index
                            )
                            compact_statistics.first_domain_count = (
                                compact_count
                            )
                        compact_statistics.attempt_count += 1
                    compact_trace_base = compact_trace_offsets[compact_start]
                    compact_trace_stop = compact_trace_offsets[compact_stop]
                    for compact_index in range(compact_count + 1):
                        compact_rebased_offsets[compact_index] = (
                            compact_trace_offsets[
                                compact_start + compact_index
                            ]
                            - compact_trace_base
                        )
                    compact_domains = (
                        <const plan7_domain_rescore_result *> (
                            journal_compact_result_bytes
                            + compact_start
                            * sizeof(plan7_domain_rescore_result)
                        )
                    )
                    compact_traces = (
                        <const plan7_domain_rescore_trace_step *> (
                            compact_trace_bytes
                            + compact_trace_base
                            * sizeof(plan7_domain_rescore_trace_step)
                        )
                    )
                    compact_row_null2 = (
                        compact_null2
                        + compact_start * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                    )
                    status = compact_domains_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        journal.usc,
                        journal.filtersc,
                        journal.vfsc,
                        journal.fwdsc,
                        generation_tail_fingerprint,
                        n_targets,
                        <uint32_t> (journal_row_base + journal_cursor),
                        journal.profile_index,
                        journal.sequence_index,
                        journal.nexpected,
                        compact_domains,
                        compact_count,
                        compact_rebased_offsets,
                        compact_count + 1,
                        compact_traces,
                        compact_trace_stop - compact_trace_base,
                        compact_row_null2,
                        compact_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
                    )
                    used_compact_domains_seam = True
                    if status == eslEINACCURATE:
                        if compact_statistics != NULL:
                            compact_statistics.threshold_retry_count += 1
                        # The guard covers uncertainty in both the compact
                        # domains and the upstream Forward score. Recompute
                        # the entire native pipeline so the retry is exact.
                        status = p7_Pipeline(pli, om, bg, sq[t], NULL, th)
                        used_compact_domains_seam = False
                    elif status == eslEINVAL:
                        if compact_statistics != NULL:
                            compact_statistics.invalid_retry_count += 1
                        xmx_count = (
                            special_offsets[forward_cursor + 1]
                            - special_offsets[forward_cursor]
                        )
                        if xmx_count == 0:
                            xmx = NULL
                        else:
                            xmx = specials + special_offsets[forward_cursor]
                        status = forward_scores_seam(
                            pli,
                            om,
                            bg,
                            sq[t],
                            NULL,
                            th,
                            usc,
                            postfilter.filtersc,
                            postfilter.vfsc,
                            forward.fwdsc,
                            xmx,
                            xmx_count,
                        )
                        used_compact_domains_seam = False
                        used_forward_seam = True
                    elif status == eslOK and compact_statistics != NULL:
                        compact_statistics.accepted_count += 1
                else:
                    status = simple_regions_seam(
                        pli,
                        om,
                        bg,
                        sq[t],
                        NULL,
                        th,
                        journal.usc,
                        journal.filtersc,
                        journal.vfsc,
                        journal.fwdsc,
                        generation_f1_bits,
                        generation_f2_bits,
                        generation_f3_bits,
                        generation_bias_filter,
                        journal.domain_route,
                        journal.nexpected,
                        regions,
                        region_stop - region_start,
                    )
                    used_simple_regions_seam = True
            elif has_forward and forward.action != FORWARD_CPU_REQUIRED:
                xmx_count = (
                    special_offsets[forward_cursor + 1]
                    - special_offsets[forward_cursor]
                )
                if xmx_count == 0:
                    xmx = NULL
                else:
                    xmx = specials + special_offsets[forward_cursor]
                status = forward_scores_seam(
                    pli,
                    om,
                    bg,
                    sq[t],
                    NULL,
                    th,
                    usc,
                    postfilter.filtersc,
                    postfilter.vfsc,
                    forward.fwdsc,
                    xmx,
                    xmx_count,
                )
                used_forward_seam = True
            else:
                status = filter_scores_seam(
                    pli,
                    om,
                    bg,
                    sq[t],
                    NULL,
                    th,
                    usc,
                    postfilter.filtersc,
                    P7_VIT_EXTERNAL,
                    postfilter.vfsc,
                )

        if has_forward:
            forward_cursor += 1
        if has_journal:
            journal_cursor += 1
        if status == eslEINVAL:
            if used_simple_regions_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterAndForwardSimpleRegions",
                )
            elif used_compact_domains_seam:
                raise UnexpectedError(
                    status,
                    "p7_PipelineFromFilterForwardAndCompactDomainsV2",
                )
            elif used_forward_seam:
                raise UnexpectedError(
                    status, "p7_PipelineFromFilterAndForwardScores"
                )
            Pipeline._missing_cutoffs(pli, om)
        elif status == eslERANGE:
            raise OverflowError(
                "numerical overflow in the optimized vector implementation"
            )
        elif status != eslOK:
            raise UnexpectedError(status, "p7_Pipeline")

        p7_pipeline_Reuse(pli)
        previous_end = target_end

    if journal_cursor != journal_count:
        raise ValueError("continuation journal row was not consumed")

    if previous_end < n_targets:
        if not pli.long_targets:
            pli.nseqs += n_targets - previous_end
            if pli.Z_setby == p7_ZSETBY_NTARGETS and pli.mode == p7_SEARCH_SEQS:
                pli.Z = pli.nseqs
        pli.nres += residue_offsets[n_targets] - residue_offsets[previous_end]

    if (
        n_targets != 0
        and (
            postfilter_count == 0
            or postfilter.sequence_index != n_targets - 1
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


cdef TopHits _search_postfilter_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t* residue_offsets,
    _pipeline_from_filter_scores_f filter_scores_seam,
):
    cdef const uint8_t* record_ptr = NULL
    cdef size_t record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef TopHits hits = TopHits(query)

    if postfilter_records.shape[0] != 0:
        record_ptr = &postfilter_records[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_postfilter(
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


cdef TopHits _search_postfilter_forward_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint64_t* residue_offsets,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
):
    cdef const uint8_t* postfilter_ptr = NULL
    cdef const uint8_t* forward_ptr = NULL
    cdef const float* special_ptr = NULL
    cdef size_t postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef size_t forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    cdef TopHits hits = TopHits(query)

    if postfilter_records.shape[0] != 0:
        postfilter_ptr = &postfilter_records[0]
    if forward_records.shape[0] != 0:
        forward_ptr = &forward_records[0]
    if specials.shape[0] != 0:
        special_ptr = &specials[0]

    with nogil:
        pipeline._pli.mode = p7_SEARCH_SEQS
        pipeline._pli.nseqs = 0
        _search_loop_postfilter_forward(
            pipeline._pli,
            optimized_profile._om,
            pipeline.background._bg,
            <const ESL_SQ**> sequences._refs,
            sequences._length,
            postfilter_ptr,
            postfilter_count,
            forward_ptr,
            forward_count,
            &special_offsets[0],
            special_ptr,
            residue_offsets,
            hits._th,
            filter_scores_seam,
            forward_scores_seam,
            NULL,
            0,
            NULL,
            NULL,
            0,
            0,
            0,
            0,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            NULL,
            0,
            0,
            NULL,
            NULL,
            NULL,
            NULL,
        )
        hits._sort_by_key()
        hits._threshold(pipeline)

    hits._query = query
    hits._empty = False
    return hits


cdef TopHits _search_postfilter_forward_journal_validated(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint8_t[::1] journal_rows,
    const uint64_t[::1] journal_region_offsets,
    const uint8_t[::1] journal_regions,
    const uint64_t[::1] journal_compact_row_offsets,
    const uint8_t[::1] journal_compact_results,
    const uint64_t[::1] journal_compact_trace_offsets,
    const uint8_t[::1] journal_compact_traces,
    const float[::1] journal_compact_null2,
    uint64_t journal_row_base,
    uint64_t generation_tail_fingerprint,
    const uint64_t* residue_offsets,
    uint64_t generation_f1_bits,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    int generation_bias_filter,
    _pipeline_from_filter_scores_f filter_scores_seam,
    _pipeline_from_filter_and_forward_scores_f forward_scores_seam,
    _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam,
    _pipeline_compact_tail_fingerprint_f compact_tail_fingerprint,
    _pipeline_from_filter_forward_compact_domains_f compact_domains_seam,
    _compact_consumption_statistics* compact_statistics,
):
    cdef const uint8_t* postfilter_ptr = NULL
    cdef const uint8_t* forward_ptr = NULL
    cdef const float* special_ptr = NULL
    cdef const uint8_t* journal_ptr = NULL
    cdef const uint8_t* region_ptr = NULL
    cdef const uint8_t* compact_result_ptr = NULL
    cdef const uint8_t* compact_trace_ptr = NULL
    cdef const float* compact_null2_ptr = NULL
    cdef uint64_t* compact_rebased_offsets = NULL
    cdef size_t compact_profile_count
    cdef size_t postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    cdef size_t forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    cdef size_t journal_count = (
        <size_t> journal_rows.shape[0]
        // sizeof(plan7_continuation_journal_row)
    )
    cdef TopHits hits = TopHits(query)

    if postfilter_records.shape[0]:
        postfilter_ptr = &postfilter_records[0]
    if forward_records.shape[0]:
        forward_ptr = &forward_records[0]
    if specials.shape[0]:
        special_ptr = &specials[0]
    if journal_rows.shape[0]:
        journal_ptr = &journal_rows[0]
    if journal_regions.shape[0]:
        region_ptr = &journal_regions[0]
    if journal_compact_results.shape[0]:
        compact_result_ptr = &journal_compact_results[0]
    if journal_compact_traces.shape[0]:
        compact_trace_ptr = &journal_compact_traces[0]
    if journal_compact_null2.shape[0]:
        compact_null2_ptr = &journal_compact_null2[0]
    compact_profile_count = <size_t> (
        journal_compact_row_offsets[journal_count]
        - journal_compact_row_offsets[0]
    )
    if compact_profile_count > (<size_t> -1) // sizeof(uint64_t) - 1:
        raise OverflowError("compact trace-offset scratch size overflows")
    compact_rebased_offsets = <uint64_t*> malloc(
        (compact_profile_count + 1) * sizeof(uint64_t)
    )
    if compact_rebased_offsets == NULL:
        raise MemoryError("compact trace-offset scratch allocation failed")

    try:
        with nogil:
            pipeline._pli.mode = p7_SEARCH_SEQS
            pipeline._pli.nseqs = 0
            _search_loop_postfilter_forward(
                pipeline._pli,
                optimized_profile._om,
                pipeline.background._bg,
                <const ESL_SQ**> sequences._refs,
                sequences._length,
                postfilter_ptr,
                postfilter_count,
                forward_ptr,
                forward_count,
                &special_offsets[0],
                special_ptr,
                residue_offsets,
                hits._th,
                filter_scores_seam,
                forward_scores_seam,
                journal_ptr,
                journal_count,
                &journal_region_offsets[0],
                region_ptr,
                generation_f1_bits,
                generation_f2_bits,
                generation_f3_bits,
                generation_bias_filter,
                simple_regions_seam,
                &journal_compact_row_offsets[0],
                compact_result_ptr,
                &journal_compact_trace_offsets[0],
                compact_trace_ptr,
                compact_null2_ptr,
                journal_row_base,
                generation_tail_fingerprint,
                compact_tail_fingerprint,
                compact_domains_seam,
                compact_rebased_offsets,
                compact_statistics,
            )
    finally:
        free(compact_rebased_offsets)
        compact_rebased_offsets = NULL
    with nogil:
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


cdef bint _validate_postfilter_records(
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
) except -1:
    cdef _postfilter_result record
    cdef _float_bits vfsc_bits
    cdef bint has_direct = False
    cdef size_t cursor
    cdef size_t record_count
    cdef uint32_t previous = 0

    if sizeof(_postfilter_result) != 16:
        raise RuntimeError("native post-filter result ABI is not 16 bytes")
    if <size_t> postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result row has trailing bytes")
    record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )

    for cursor in range(record_count):
        memcpy(
            &record,
            &postfilter_records[cursor * sizeof(_postfilter_result)],
            sizeof(_postfilter_result),
        )
        if record.sequence_index >= sequences._length:
            raise IndexError(
                f"post-filter result sequence index out of range: "
                f"{record.sequence_index}"
            )
        if cursor != 0 and record.sequence_index <= previous:
            raise ValueError(
                "post-filter result sequence indexes must be strictly "
                "increasing and unique"
            )
        if record.action == BIAS_CPU_REQUIRED:
            pass
        elif (
            record.action == BIAS_DEFINITE_REJECT
            and record.msv_status == SSV_OK
            and isnan(record.filtersc)
            and isnan(record.vfsc)
        ):
            pass
        elif (
            record.action == BIAS_DEFINITE_REJECT
            or record.action == BIAS_DEFINITE_PASS
        ):
            if record.msv_status != SSV_OK:
                raise ValueError(
                    "direct post-filter result requires eslOK MSV status"
                )
            if not isfinite(record.filtersc):
                raise ValueError(
                    "direct post-filter result requires finite filter score"
                )
            vfsc_bits.value = record.vfsc
            if not isfinite(record.vfsc) and vfsc_bits.bits != 0x7f800000:
                raise ValueError(
                    "direct post-filter result requires finite or +infinity "
                    "Viterbi score"
                )
            if sequences._refs[record.sequence_index].n == 0:
                raise ValueError("empty target requires CPU post-filter fallback")
            has_direct = True
        else:
            raise ValueError(f"unknown post-filter result action: {record.action}")
        previous = record.sequence_index

    if has_direct and (
        not isfinite(optimized_profile._om.scale_b)
        or optimized_profile._om.scale_b <= 0.0
    ):
        raise ValueError(
            "direct post-filter results require a positive finite SSV scale"
        )
    return has_direct


cdef void _validate_postfilter_residue_offsets(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] residue_offsets,
) except *:
    cdef _postfilter_result record
    cdef size_t cursor
    cdef size_t record_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
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
            &postfilter_records[cursor * sizeof(_postfilter_result)],
            sizeof(_postfilter_result),
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


cdef bint _validate_forward_augmentation(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
) except -1:
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t postfilter_cursor = 0
    cdef size_t forward_cursor
    cdef size_t special_cursor
    cdef uint32_t previous = 0
    cdef uint64_t special_start
    cdef uint64_t special_stop
    cdef uint64_t expected_count
    cdef bint has_external = False

    if sizeof(_forward_result) != 12:
        raise RuntimeError("native Forward result ABI is not 12 bytes")
    if <size_t> forward_records.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("Forward result row has trailing bytes")
    forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    if special_offsets.shape[0] != forward_count + 1:
        raise ValueError("Forward special-offset count differs from result count")
    if special_offsets[0] > <uint64_t> specials.shape[0]:
        raise ValueError("Forward special offsets start outside storage")
    postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )

    for forward_cursor in range(forward_count):
        memcpy(
            &forward,
            &forward_records[forward_cursor * sizeof(_forward_result)],
            sizeof(_forward_result),
        )
        if forward.sequence_index >= sequences._length:
            raise IndexError(
                f"Forward result sequence index out of range: "
                f"{forward.sequence_index}"
            )
        if forward_cursor != 0 and forward.sequence_index <= previous:
            raise ValueError(
                "Forward result sequence indexes must be strictly increasing "
                "and unique"
            )
        previous = forward.sequence_index
        if forward.reserved != 0:
            raise ValueError("Forward result reserved field is nonzero")

        while postfilter_cursor < postfilter_count:
            memcpy(
                &postfilter,
                &postfilter_records[
                    postfilter_cursor * sizeof(_postfilter_result)
                ],
                sizeof(_postfilter_result),
            )
            if postfilter.sequence_index >= forward.sequence_index:
                break
            postfilter_cursor += 1
        if (
            postfilter_cursor == postfilter_count
            or postfilter.sequence_index != forward.sequence_index
            or postfilter.action != BIAS_DEFINITE_PASS
        ):
            raise ValueError(
                "Forward results must be a subset of direct post-filter passes"
            )

        special_start = special_offsets[forward_cursor]
        special_stop = special_offsets[forward_cursor + 1]
        if special_start > special_stop:
            raise ValueError("Forward special offsets are not monotone")
        if special_stop > <uint64_t> specials.shape[0]:
            raise ValueError("Forward special offsets exceed storage")

        if forward.action == FORWARD_CPU_REQUIRED:
            if forward.status not in (
                FORWARD_OK,
                FORWARD_ERANGE,
                FORWARD_ENORESULT,
                FORWARD_EMPTY,
            ):
                raise ValueError("unknown CPU-required Forward status")
            if forward.status == FORWARD_OK:
                if not isfinite(forward.fwdsc):
                    raise ValueError(
                        "eslOK CPU-required Forward result needs a finite score"
                    )
            elif not isnan(forward.fwdsc):
                raise ValueError(
                    "failed CPU-required Forward result must have NaN score"
                )
            if special_start != special_stop:
                raise ValueError(
                    "CPU-required Forward result must not have special states"
                )
        elif (
            forward.action == FORWARD_DEFINITE_REJECT
            or forward.action == FORWARD_DEFINITE_PASS
        ):
            if forward.status != FORWARD_OK:
                raise ValueError("direct Forward result requires eslOK status")
            if not isfinite(forward.fwdsc):
                raise ValueError("direct Forward result requires finite score")
            if forward.action == FORWARD_DEFINITE_REJECT:
                if special_start != special_stop:
                    raise ValueError(
                        "rejected Forward result must not have special states"
                    )
            else:
                expected_count = 6 * (
                    <uint64_t> sequences._refs[forward.sequence_index].n + 1
                )
                if special_stop - special_start != expected_count:
                    raise ValueError(
                        "passing Forward result has wrong special-state span"
                    )
                for special_cursor in range(
                    <size_t> special_start, <size_t> special_stop
                ):
                    if (
                        not isfinite(specials[special_cursor])
                        or specials[special_cursor] < 0.0
                    ):
                        raise ValueError(
                            "Forward special states must be finite and nonnegative"
                        )
            has_external = True
        else:
            raise ValueError(f"unknown Forward result action: {forward.action}")

    return has_external


def _validate_forward_batch_bound(
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] postfilter_offsets,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] forward_offsets,
    const uint32_t[::1] expected_indices,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
):
    """Validate one complete native Forward result before it is retained."""
    cdef _forward_result forward
    cdef size_t profile_count
    cdef size_t profile_index
    cdef size_t cursor
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop

    if postfilter_offsets.shape[0] == 0:
        raise ValueError("post-filter row offsets need an initial zero")
    if postfilter_offsets.shape[0] != forward_offsets.shape[0]:
        raise ValueError("Forward row-offset count differs from post-filter rows")
    profile_count = <size_t> postfilter_offsets.shape[0] - 1
    if postfilter_offsets[0] != 0 or forward_offsets[0] != 0:
        raise ValueError("Forward batch row offsets must start at zero")
    if <size_t> postfilter_records.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result storage has trailing bytes")
    if <size_t> forward_records.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("Forward result storage has trailing bytes")
    postfilter_count = (
        <size_t> postfilter_records.shape[0] // sizeof(_postfilter_result)
    )
    forward_count = (
        <size_t> forward_records.shape[0] // sizeof(_forward_result)
    )
    if postfilter_offsets[profile_count] != postfilter_count:
        raise ValueError("post-filter row offsets do not span result storage")
    if forward_offsets[profile_count] != forward_count:
        raise ValueError("Forward row offsets do not span result storage")
    if expected_indices.shape[0] != forward_count:
        raise ValueError("Forward result count differs from selected inputs")
    if special_offsets.shape[0] != forward_count + 1:
        raise ValueError("Forward special-offset count differs from result count")
    if special_offsets[0] != 0:
        raise ValueError("Forward global special offsets must start at zero")
    if special_offsets[forward_count] != <uint64_t> specials.shape[0]:
        raise ValueError("Forward special offsets do not span matrix storage")

    for cursor in range(forward_count):
        memcpy(
            &forward,
            &forward_records[cursor * sizeof(_forward_result)],
            sizeof(_forward_result),
        )
        if forward.sequence_index != expected_indices[cursor]:
            raise ValueError("Forward result order differs from selected inputs")

    for profile_index in range(profile_count):
        postfilter_start = <size_t> postfilter_offsets[profile_index]
        postfilter_stop = <size_t> postfilter_offsets[profile_index + 1]
        forward_start = <size_t> forward_offsets[profile_index]
        forward_stop = <size_t> forward_offsets[profile_index + 1]
        if postfilter_start > postfilter_stop or postfilter_stop > postfilter_count:
            raise ValueError("post-filter row offsets are not monotone")
        if forward_start > forward_stop or forward_stop > forward_count:
            raise ValueError("Forward row offsets are not monotone")
        _validate_forward_augmentation(
            sequences,
            postfilter_records[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            forward_records[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            special_offsets[forward_start:forward_stop + 1],
            specials,
        )


cdef void _validate_sealed_pair(
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
) except *:
    if not query.alphabet._eq(optimized_profile.alphabet):
        raise AlphabetMismatch(query.alphabet, optimized_profile.alphabet)
    if not query.alphabet._eq(sequences.alphabet):
        raise AlphabetMismatch(query.alphabet, sequences.alphabet)
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


cdef void _validate_sealed_residue_offsets(
    DigitalSequenceBlock sequences,
    const uint64_t[::1] residue_offsets,
) except *:
    cdef size_t target

    if residue_offsets.shape[0] != sequences._length + 1:
        raise ValueError("target residue-prefix length differs from target count")
    if residue_offsets[0] != 0:
        raise ValueError("target residue prefix must start at zero")
    for target in range(sequences._length):
        if sequences._refs[target].n > HMMER_TARGET_LIMIT:
            raise ValueError("target exceeds HMMER's protein limit")
        if residue_offsets[target + 1] < residue_offsets[target]:
            raise ValueError("target residue prefix is not monotone")
        if (
            residue_offsets[target + 1] - residue_offsets[target]
            != <uint64_t> sequences._refs[target].n
        ):
            raise ValueError("target residue prefix differs from target length")


cdef object _immutable_owned_view(object value, str expected_format):
    """Return a one-dimensional native view whose ultimate owner is bytes."""
    cdef object view = memoryview(value)
    cdef object owner

    if view.ndim != 1 or not view.c_contiguous:
        raise ValueError("sealed buffers must be one-dimensional and contiguous")
    if view.format != expected_format:
        raise TypeError(
            f"sealed buffer format {view.format!r} is not {expected_format!r}"
        )
    owner = view.obj
    while type(owner) is memoryview:
        owner = owner.obj
    if type(owner) is bytes and view.readonly:
        return view
    return memoryview(view.cast("B").tobytes()).cast(expected_format)


cdef bint _journal_expected_segment(
    size_t *cursor,
    uint64_t observed_offset,
    uint64_t count,
    size_t item_size,
) noexcept:
    cdef size_t aligned
    cdef size_t native_count
    cdef size_t byte_count
    if count > <uint64_t> (<size_t> -1):
        return False
    native_count = <size_t> count
    if cursor[0] > (<size_t> -1) - 7:
        return False
    aligned = (cursor[0] + 7) & ~<size_t> 7
    if observed_offset != <uint64_t> aligned:
        return False
    if native_count != 0 and item_size > (<size_t> -1) // native_count:
        return False
    byte_count = native_count * item_size
    if aligned > (<size_t> -1) - byte_count:
        return False
    cursor[0] = aligned + byte_count
    return True


cdef tuple _consume_validate_continuation_journal(
    object capsule,
    tuple profiles,
    DigitalSequenceBlock sequences,
    uint64_t expected_session_id,
    uint64_t expected_selection_id,
    const uint64_t[::1] expected_identity_tokens,
    const uint8_t[::1] expected_profile_fingerprints,
    uint64_t expected_batch_generation,
    const uint8_t[::1] expected_sequence_fingerprint,
    double f1,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    bint generation_bias_filter,
    uint64_t expected_tail_fingerprint,
):
    cdef plan7_continuation_journal *journal
    cdef size_t cursor
    cdef size_t profile_count = len(profiles)
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t profile
    cdef size_t row
    cdef size_t row_start
    cdef size_t row_stop
    cdef size_t postfilter_cursor
    cdef size_t postfilter_stop
    cdef size_t forward_cursor
    cdef size_t forward_stop
    cdef size_t region
    cdef size_t region_start
    cdef size_t region_stop
    cdef size_t compact_start
    cdef size_t compact_stop
    cdef size_t compact_index
    cdef size_t trace_start
    cdef size_t trace_stop
    cdef size_t simple_row_count = 0
    cdef size_t device_result_count = 0
    cdef size_t cpu_required_count = 0
    cdef size_t compact_bytes
    cdef size_t residue
    cdef uint8_t compact_row_action
    cdef uint8_t expected_compact_route
    cdef float null2_value
    cdef uint32_t previous_sequence
    cdef uint32_t previous_end
    cdef uint32_t guard_bits
    cdef _postfilter_result postfilter
    cdef _forward_result forward
    cdef plan7_continuation_journal_row journal_row
    cdef plan7_simple_region interval
    cdef plan7_domain_rescore_result compact_result
    cdef OptimizedProfile optimized_profile
    cdef float computed_usc
    cdef _float_bits expected_float
    cdef _float_bits observed_float
    cdef _double_bits expected_double
    cdef object tokens = set()
    cdef _ContinuationJournalStorage storage
    cdef object storage_view
    cdef const uint8_t[::1] postfilter_view
    cdef const uint64_t[::1] postfilter_offset_view
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint64_t[::1] profile_offset_view
    cdef const uint64_t[::1] identity_token_view
    cdef const uint8_t[::1] profile_fingerprint_view
    cdef const uint8_t[::1] row_view
    cdef const uint64_t[::1] journal_special_offset_view
    cdef const uint64_t[::1] region_offset_view
    cdef const uint8_t[::1] region_view
    cdef const uint64_t[::1] compact_row_offset_view
    cdef const uint8_t[::1] compact_result_view
    cdef const uint64_t[::1] compact_trace_offset_view
    cdef const uint8_t[::1] compact_trace_view
    cdef const float[::1] compact_null2_view
    cdef uint64_t observed_result_hash = 0
    cdef uint64_t observed_trace_hash = 0
    cdef uint64_t observed_null2_hash = 0

    if not PyCapsule_IsValid(
        capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    ):
        raise TypeError("continuation journal capsule is invalid or consumed")
    journal = <plan7_continuation_journal *> PyCapsule_GetPointer(
        capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
    )
    if journal == NULL:
        raise TypeError("continuation journal capsule has no storage")
    if (
        journal.magic != PLAN7_CONTINUATION_JOURNAL_MAGIC
        or journal.version != PLAN7_CONTINUATION_JOURNAL_VERSION
        or journal.header_size != sizeof(plan7_continuation_journal)
        or journal.row_size != sizeof(plan7_continuation_journal_row)
        or journal.region_size != sizeof(plan7_simple_region)
        or journal.compact_result_size
        != sizeof(plan7_domain_rescore_result)
        or journal.compact_trace_step_size
        != sizeof(plan7_domain_rescore_trace_step)
        or journal.compact_null2_stride
        != PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.total_bytes < sizeof(plan7_continuation_journal)
        or journal.total_bytes > <uint64_t> (<size_t> -1)
        or journal.total_bytes > <uint64_t> PY_SSIZE_T_MAX
        or journal.profile_count != profile_count
        or journal.profile_count > <uint64_t> (<size_t> -1) - 1
        or journal.postfilter_count > <uint64_t> (<size_t> -1)
        or journal.forward_count > <uint64_t> (<size_t> -1) - 1
        or journal.row_count > <uint64_t> (<size_t> -1) - 1
        or journal.special_count > <uint64_t> (<size_t> -1)
        or journal.region_count > <uint64_t> (<size_t> -1)
        or journal.compact_result_count > <uint64_t> (<size_t> -1) - 1
        or journal.compact_trace_offset_count
        > <uint64_t> (<size_t> -1)
        or journal.compact_trace_count > <uint64_t> (<size_t> -1)
        or journal.compact_null2_count > <uint64_t> (<size_t> -1)
    ):
        raise ValueError("continuation journal ABI header is invalid")
    cursor = sizeof(plan7_continuation_journal)
    if not (
        _journal_expected_segment(
            &cursor, journal.postfilter_offsets_offset,
            journal.profile_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.postfilter_records_offset,
            journal.postfilter_count, sizeof(_postfilter_result),
        )
        and _journal_expected_segment(
            &cursor, journal.forward_offsets_offset,
            journal.profile_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.forward_records_offset,
            journal.forward_count, sizeof(_forward_result),
        )
        and _journal_expected_segment(
            &cursor, journal.forward_special_offsets_offset,
            journal.forward_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.profile_offsets_offset,
            journal.profile_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.identity_tokens_offset,
            journal.profile_count, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.profile_fingerprints_offset,
            journal.profile_count,
            PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
        )
        and _journal_expected_segment(
            &cursor, journal.rows_offset,
            journal.row_count, sizeof(plan7_continuation_journal_row),
        )
        and _journal_expected_segment(
            &cursor, journal.special_offsets_offset,
            journal.row_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.specials_offset,
            journal.special_count, sizeof(float),
        )
        and _journal_expected_segment(
            &cursor, journal.region_offsets_offset,
            journal.row_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.regions_offset,
            journal.region_count, sizeof(plan7_simple_region),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_row_offsets_offset,
            journal.row_count + 1, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_results_offset,
            journal.compact_result_count,
            sizeof(plan7_domain_rescore_result),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_trace_offsets_offset,
            journal.compact_trace_offset_count, sizeof(uint64_t),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_traces_offset,
            journal.compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step),
        )
        and _journal_expected_segment(
            &cursor, journal.compact_null2_offset,
            journal.compact_null2_count, sizeof(float),
        )
        and cursor == <size_t> journal.total_bytes
    ):
        raise ValueError("continuation journal storage layout is invalid")
    if (
        journal.integrity_tag == 0
        or journal.integrity_tag
        != plan7_continuation_journal_integrity(journal)
    ):
        raise ValueError("continuation journal integrity check failed")
    if (
        journal.session_id == 0
        or journal.selection_id == 0
        or journal.session_id != expected_session_id
        or journal.selection_id != expected_selection_id
        or journal.profile_count != profile_count
    ):
        raise ValueError("continuation journal selection identity differs")
    if expected_identity_tokens.shape[0] != profile_count:
        raise ValueError("continuation journal profile identity count differs")
    if expected_profile_fingerprints.shape[0] != (
        profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
    ):
        raise ValueError("continuation journal profile fingerprint count differs")
    if expected_sequence_fingerprint.shape[0] != 32:
        raise ValueError("sequence content fingerprint has the wrong size")
    if memcmp(
        journal.sequence_content_fingerprint,
        &expected_sequence_fingerprint[0],
        32,
    ) != 0:
        raise ValueError("continuation journal target content differs")
    expected_double.value = f1
    if journal.generation_f1_bits != expected_double.bits:
        raise ValueError("continuation journal F1 provenance differs")
    if (
        journal.generation_f2_bits != generation_f2_bits
        or journal.generation_f3_bits != generation_f3_bits
        or journal.generation_bias_filter != <uint8_t> generation_bias_filter
        or not generation_bias_filter
    ):
        raise ValueError("continuation journal filter provenance differs")
    expected_float.value = <float> 0.25
    if journal.rt1_bits != expected_float.bits:
        raise ValueError("continuation journal rt1 differs")
    expected_float.value = <float> 0.10
    if journal.rt2_bits != expected_float.bits:
        raise ValueError("continuation journal rt2 differs")
    expected_float.value = <float> 0.20
    if journal.rt3_bits != expected_float.bits:
        raise ValueError("continuation journal rt3 differs")
    observed_float.bits = journal.guard_band_bits
    guard_bits = journal.guard_band_bits
    if (
        not isfinite(observed_float.value)
        or observed_float.value < <float> 2.0e-4
        or observed_float.value > <float> 1.0
    ):
        raise ValueError("continuation journal guard band is invalid")
    if (
        journal.forward.database_generation == 0
        or journal.forward.batch_generation == 0
        or journal.forward.batch_generation != expected_batch_generation
        or journal.forward.pass_count != journal.row_count
        or journal.forward.special_count != journal.special_count
        or journal.forward.generation_f3_bits != generation_f3_bits
        or journal.backward.candidate_count != journal.row_count
        or journal.backward.region_count != journal.region_count
        or memcmp(
            &journal.forward,
            &journal.backward.forward,
            sizeof(plan7_forward_provenance),
        ) != 0
    ):
        raise ValueError("continuation journal native provenance differs")
    if journal.generation_compact_domains > 1:
        raise ValueError("continuation journal compact-domain flag is invalid")
    if not journal.generation_compact_domains:
        if (
            journal.generation_tail_fingerprint != 0
            or journal.compact_result_count != 0
            or journal.compact_trace_offset_count != 1
            or journal.compact_trace_count != 0
            or journal.compact_null2_count != 0
            or journal.rescore_simple_row_count != 0
            or journal.rescore_device_result_count != 0
            or journal.rescore_cpu_required_count != 0
            or journal.rescore_numeric_fallback_count != 0
            or journal.rescore_cap_fallback_count != 0
            or journal.rescore_global_cpu_fallback_count != 0
            or journal.rescore_compact_output_byte_limit != 0
            or journal.rescore_compact_output_bytes != 0
            or journal.compact_global_fallback != 0
            or journal.rescore.result_hash != 0
            or journal.rescore.trace_hash != 0
            or journal.rescore.null2_hash != 0
            or journal.rescore.result_count != 0
            or journal.rescore.trace_count != 0
            or journal.rescore.null2_count != 0
        ):
            raise ValueError("disabled compact-domain journal carries output")
    elif (
        journal.generation_tail_fingerprint == 0
        or expected_tail_fingerprint == 0
        or journal.generation_tail_fingerprint != expected_tail_fingerprint
        or journal.compact_trace_offset_count
        != journal.compact_result_count + 1
        or journal.compact_global_fallback > 1
        or memcmp(
            &journal.rescore.backward,
            &journal.backward,
            sizeof(plan7_backward_domain_provenance),
        ) != 0
        or journal.rescore.result_count != journal.compact_result_count
        or journal.rescore.trace_count != journal.compact_trace_count
        or journal.rescore.null2_count != journal.compact_null2_count
        or journal.rescore_simple_row_count > journal.row_count
        or journal.rescore_device_result_count
        + journal.rescore_cpu_required_count != journal.region_count
        or journal.rescore_numeric_fallback_count
        + journal.rescore_cap_fallback_count
        != journal.rescore_cpu_required_count
        or journal.rescore_global_cpu_fallback_count
        not in (0, journal.region_count)
        or bool(journal.compact_global_fallback)
        != (journal.rescore_global_cpu_fallback_count != 0)
        or journal.rescore_compact_output_byte_limit < sizeof(uint64_t)
        or journal.rescore_compact_output_byte_limit
        > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        or journal.rescore_compact_output_bytes
        > journal.rescore_compact_output_byte_limit
        or journal.compact_result_count
        > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        // sizeof(plan7_domain_rescore_result)
        or journal.compact_trace_count
        > PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES
        // sizeof(plan7_domain_rescore_trace_step)
        or journal.compact_result_count
        > (<uint64_t> -1) // PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.compact_null2_count
        != journal.compact_result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        or journal.compact_result_count not in (0, journal.region_count)
    ):
        raise ValueError("continuation journal compact provenance differs")

    storage = _ContinuationJournalStorage.__new__(_ContinuationJournalStorage)
    storage._data = <uint8_t *> journal
    storage._size = <Py_ssize_t> journal.total_bytes
    storage_view = memoryview(storage).cast("B")
    postfilter_offset_view = storage_view[
        journal.postfilter_offsets_offset:
        journal.postfilter_offsets_offset
        + (profile_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    postfilter_view = storage_view[
        journal.postfilter_records_offset:
        journal.postfilter_records_offset
        + <size_t> journal.postfilter_count * sizeof(_postfilter_result)
    ]
    forward_offset_view = storage_view[
        journal.forward_offsets_offset:
        journal.forward_offsets_offset
        + (profile_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    forward_view = storage_view[
        journal.forward_records_offset:
        journal.forward_records_offset
        + <size_t> journal.forward_count * sizeof(_forward_result)
    ]
    special_offset_view = storage_view[
        journal.forward_special_offsets_offset:
        journal.forward_special_offsets_offset
        + (<size_t> journal.forward_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    profile_offset_view = storage_view[
        journal.profile_offsets_offset:
        journal.profile_offsets_offset + (profile_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    identity_token_view = storage_view[
        journal.identity_tokens_offset:
        journal.identity_tokens_offset + profile_count * sizeof(uint64_t)
    ].cast("Q")
    profile_fingerprint_view = storage_view[
        journal.profile_fingerprints_offset:
        journal.profile_fingerprints_offset
        + profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
    ]
    row_view = storage_view[
        journal.rows_offset:
        journal.rows_offset
        + <size_t> journal.row_count * sizeof(plan7_continuation_journal_row)
    ]
    journal_special_offset_view = storage_view[
        journal.special_offsets_offset:
        journal.special_offsets_offset
        + (<size_t> journal.row_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    special_view = storage_view[
        journal.specials_offset:
        journal.specials_offset
        + <size_t> journal.special_count * sizeof(float)
    ].cast("f")
    region_offset_view = storage_view[
        journal.region_offsets_offset:
        journal.region_offsets_offset
        + (<size_t> journal.row_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    region_view = storage_view[
        journal.regions_offset:
        journal.regions_offset
        + <size_t> journal.region_count * sizeof(plan7_simple_region)
    ]
    compact_row_offset_view = storage_view[
        journal.compact_row_offsets_offset:
        journal.compact_row_offsets_offset
        + (<size_t> journal.row_count + 1) * sizeof(uint64_t)
    ].cast("Q")
    compact_result_view = storage_view[
        journal.compact_results_offset:
        journal.compact_results_offset
        + <size_t> journal.compact_result_count
        * sizeof(plan7_domain_rescore_result)
    ]
    compact_trace_offset_view = storage_view[
        journal.compact_trace_offsets_offset:
        journal.compact_trace_offsets_offset
        + <size_t> journal.compact_trace_offset_count * sizeof(uint64_t)
    ].cast("Q")
    compact_trace_view = storage_view[
        journal.compact_traces_offset:
        journal.compact_traces_offset
        + <size_t> journal.compact_trace_count
        * sizeof(plan7_domain_rescore_trace_step)
    ]
    compact_null2_view = storage_view[
        journal.compact_null2_offset:
        journal.compact_null2_offset
        + <size_t> journal.compact_null2_count * sizeof(float)
    ].cast("f")

    if (
        postfilter_offset_view[0] != 0
        or postfilter_offset_view[profile_count] != journal.postfilter_count
    ):
        raise ValueError("continuation journal post-filter offsets do not span rows")
    if (
        forward_offset_view[0] != 0
        or forward_offset_view[profile_count] != journal.forward_count
    ):
        raise ValueError("continuation journal Forward offsets do not span rows")
    if (
        special_offset_view[0] != 0
        or special_offset_view[journal.forward_count] != journal.special_count
    ):
        raise ValueError("continuation journal Forward specials do not span storage")
    if profile_offset_view[0] != 0 or (
        profile_offset_view[profile_count] != journal.row_count
    ):
        raise ValueError("continuation journal profile offsets do not span rows")
    if journal_special_offset_view[0] != 0 or (
        journal_special_offset_view[journal.row_count] != journal.special_count
    ):
        raise ValueError("continuation journal special offsets do not span storage")
    if region_offset_view[0] != 0 or (
        region_offset_view[journal.row_count] != journal.region_count
    ):
        raise ValueError("continuation journal region offsets do not span storage")
    if compact_row_offset_view[0] != 0 or (
        compact_row_offset_view[journal.row_count]
        != journal.compact_result_count
    ):
        raise ValueError("continuation journal compact rows do not span storage")
    if compact_trace_offset_view[0] != 0 or (
        compact_trace_offset_view[journal.compact_trace_offset_count - 1]
        != journal.compact_trace_count
    ):
        raise ValueError("continuation journal compact traces do not span storage")
    if journal.generation_compact_domains:
        compact_bytes = (
            <size_t> journal.compact_result_count
            * sizeof(plan7_domain_rescore_result)
            + (<size_t> journal.compact_result_count + 1) * sizeof(uint64_t)
            + <size_t> journal.compact_trace_count
            * sizeof(plan7_domain_rescore_trace_step)
            + <size_t> journal.compact_null2_count * sizeof(float)
        )
        if compact_bytes != journal.rescore_compact_output_bytes:
            raise ValueError("continuation journal compact byte count differs")
        if not plan7_continuation_journal_rescore_hashes(
            (
                <const plan7_domain_rescore_result *> &compact_result_view[0]
                if journal.compact_result_count
                else NULL
            ),
            journal.compact_result_count,
            &compact_trace_offset_view[0],
            (
                <const plan7_domain_rescore_trace_step *> &compact_trace_view[0]
                if journal.compact_trace_count
                else NULL
            ),
            journal.compact_trace_count,
            &compact_null2_view[0] if journal.compact_null2_count else NULL,
            journal.compact_null2_count,
            &observed_result_hash,
            &observed_trace_hash,
            &observed_null2_hash,
        ):
            raise ValueError("continuation journal compact hashes are invalid")
        if (
            observed_result_hash != journal.rescore.result_hash
            or observed_trace_hash != journal.rescore.trace_hash
            or observed_null2_hash != journal.rescore.null2_hash
        ):
            raise ValueError("continuation journal compact hashes differ")
        if journal.compact_result_count == 0:
            if (
                journal.compact_trace_count != 0
                or journal.rescore_device_result_count != 0
                or journal.rescore_cpu_required_count != journal.region_count
                or journal.rescore_global_cpu_fallback_count
                != journal.region_count
            ):
                raise ValueError(
                    "continuation journal global compact fallback differs"
                )
        elif journal.rescore_global_cpu_fallback_count != 0:
            raise ValueError("continuation journal compact fallback differs")
    else:
        for row in range(<size_t> journal.row_count + 1):
            if compact_row_offset_view[row] != 0:
                raise ValueError("disabled compact journal has row offsets")
        if compact_trace_offset_view[0] != 0:
            raise ValueError("disabled compact journal has trace offsets")
    for profile in range(profile_count):
        if (
            identity_token_view[profile] == 0
            or identity_token_view[profile] in tokens
        ):
            raise ValueError("continuation journal profile identity is invalid")
        if identity_token_view[profile] != expected_identity_tokens[profile]:
            raise ValueError("continuation journal profile identity differs")
        tokens.add(identity_token_view[profile])
    if profile_count and memcmp(
        &profile_fingerprint_view[0],
        &expected_profile_fingerprints[0],
        profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
    ) != 0:
        raise ValueError("continuation journal optimized-profile snapshot differs")

    postfilter_count = <size_t> journal.postfilter_count
    forward_count = <size_t> journal.forward_count

    for profile in range(profile_count):
        row_start = <size_t> profile_offset_view[profile]
        row_stop = <size_t> profile_offset_view[profile + 1]
        if row_start > row_stop or row_stop > journal.row_count:
            raise ValueError("continuation journal profile offsets are not monotone")
        postfilter_cursor = <size_t> postfilter_offset_view[profile]
        postfilter_stop = <size_t> postfilter_offset_view[profile + 1]
        forward_cursor = <size_t> forward_offset_view[profile]
        forward_stop = <size_t> forward_offset_view[profile + 1]
        if (
            postfilter_cursor > postfilter_stop
            or postfilter_stop > postfilter_count
            or forward_cursor > forward_stop
            or forward_stop > forward_count
        ):
            raise ValueError("continuation journal row offsets are not monotone")
        row = row_start
        while forward_cursor < forward_stop:
            memcpy(
                &forward,
                &forward_view[forward_cursor * sizeof(_forward_result)],
                sizeof(_forward_result),
            )
            if forward.action != FORWARD_DEFINITE_PASS:
                forward_cursor += 1
                continue
            if row >= row_stop:
                raise ValueError("continuation journal omits a Forward pass")
            memcpy(
                &journal_row,
                &row_view[row * sizeof(plan7_continuation_journal_row)],
                sizeof(plan7_continuation_journal_row),
            )
            if (
                journal_row.profile_index != profile
                or journal_row.sequence_index >= sequences._length
                or journal_row.sequence_index != forward.sequence_index
                or journal_row.forward_status != forward.status
                or journal_row.forward_action != forward.action
            ):
                raise ValueError("continuation journal Forward identity differs")
            expected_float.value = forward.fwdsc
            observed_float.value = journal_row.fwdsc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal Forward score differs")
            while postfilter_cursor < postfilter_stop:
                memcpy(
                    &postfilter,
                    &postfilter_view[
                        postfilter_cursor * sizeof(_postfilter_result)
                    ],
                    sizeof(_postfilter_result),
                )
                if postfilter.sequence_index >= journal_row.sequence_index:
                    break
                postfilter_cursor += 1
            if (
                postfilter_cursor == postfilter_stop
                or postfilter.sequence_index != journal_row.sequence_index
                or postfilter.msv_status != journal_row.postfilter_status
                or postfilter.action != journal_row.postfilter_action
                or postfilter.action != BIAS_DEFINITE_PASS
            ):
                raise ValueError("continuation journal post-filter identity differs")
            expected_float.value = postfilter.filtersc
            observed_float.value = journal_row.filtersc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal bias score differs")
            expected_float.value = postfilter.vfsc
            observed_float.value = journal_row.vfsc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal Viterbi score differs")
            optimized_profile = <OptimizedProfile> profiles[profile]
            computed_usc = <float> postfilter.msv_numerator
            computed_usc = computed_usc / optimized_profile._om.scale_b
            computed_usc = computed_usc - <float> 3.0
            expected_float.value = computed_usc
            observed_float.value = journal_row.usc
            if expected_float.bits != observed_float.bits:
                raise ValueError("continuation journal MSV score differs")
            if (
                journal_special_offset_view[row]
                != special_offset_view[forward_cursor]
                or journal_special_offset_view[row + 1]
                != special_offset_view[forward_cursor + 1]
            ):
                raise ValueError("continuation journal Forward span differs")

            region_start = <size_t> region_offset_view[row]
            region_stop = <size_t> region_offset_view[row + 1]
            if (
                region_start > region_stop
                or region_stop > journal.region_count
                or journal_row.reserved != 0
                or journal_row.domain_status not in (
                    DOMAIN_OK, DOMAIN_ERANGE, DOMAIN_ENORESULT, DOMAIN_EMPTY
                )
                or journal_row.domain_route not in (
                    DOMAIN_CPU_REQUIRED, DOMAIN_NO_REGIONS, DOMAIN_SIMPLE
                )
                or journal_row.has_own_scales > 1
                or journal_row.compact_route not in (
                    PLAN7_CONTINUATION_COMPACT_NONE,
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED,
                    PLAN7_CONTINUATION_COMPACT_DEVICE,
                )
                or journal_row.reserved2[0] != 0
                or journal_row.reserved2[1] != 0
                or journal_row.reserved2[2] != 0
                or journal_row.reserved3 != 0
            ):
                raise ValueError("continuation journal domain record is invalid")
            if journal_row.domain_route == DOMAIN_CPU_REQUIRED:
                if region_start != region_stop:
                    raise ValueError("CPU domain route carries simple regions")
            else:
                if (
                    journal_row.domain_status != DOMAIN_OK
                    or journal_row.uncertain_count != 0
                    or journal_row.multidomain_count != 0
                    or not isfinite(journal_row.backward_score)
                    or not isfinite(journal_row.nexpected)
                    or journal_row.nexpected < 0.0
                    or journal_row.nexpected
                    > <float> sequences._refs[journal_row.sequence_index].n
                ):
                    raise ValueError("domain route is not continuation-safe")
                if journal_row.domain_route == DOMAIN_NO_REGIONS:
                    if journal_row.region_count != 0 or region_start != region_stop:
                        raise ValueError("no-region route carries intervals")
                elif (
                    region_start == region_stop
                    or journal_row.region_count != region_stop - region_start
                ):
                    raise ValueError("simple domain route has invalid intervals")

            previous_end = 0
            for region in range(region_start, region_stop):
                memcpy(
                    &interval,
                    &region_view[region * sizeof(plan7_simple_region)],
                    sizeof(plan7_simple_region),
                )
                if (
                    interval.begin < 1
                    or interval.begin > interval.end
                    or interval.end
                    > <uint32_t> sequences._refs[journal_row.sequence_index].n
                    or (region != region_start and interval.begin <= previous_end)
                ):
                    raise ValueError("continuation journal intervals are invalid")
                previous_end = interval.end

            compact_start = <size_t> compact_row_offset_view[row]
            compact_stop = <size_t> compact_row_offset_view[row + 1]
            if (
                compact_start > compact_stop
                or compact_stop > journal.compact_result_count
                or journal_row.compact_result_count
                != compact_stop - compact_start
            ):
                raise ValueError(
                    "continuation journal compact row offsets are not monotone"
                )
            if journal_row.domain_route == DOMAIN_SIMPLE:
                simple_row_count += 1
            if journal.compact_result_count == 0:
                if compact_start != 0 or compact_stop != 0:
                    raise ValueError("empty compact output carries row results")
            elif (
                compact_start != region_start
                or compact_stop != region_stop
            ):
                raise ValueError("compact results differ from simple regions")

            expected_compact_route = PLAN7_CONTINUATION_COMPACT_NONE
            if (
                journal.compact_global_fallback
                and journal_row.domain_route == DOMAIN_SIMPLE
                and region_start != region_stop
            ):
                expected_compact_route = (
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                )
            if compact_start != compact_stop:
                compact_row_action = 255
            for compact_index in range(compact_start, compact_stop):
                memcpy(
                    &compact_result,
                    &compact_result_view[
                        compact_index * sizeof(plan7_domain_rescore_result)
                    ],
                    sizeof(plan7_domain_rescore_result),
                )
                memcpy(
                    &interval,
                    &region_view[
                        compact_index * sizeof(plan7_simple_region)
                    ],
                    sizeof(plan7_simple_region),
                )
                trace_start = <size_t> compact_trace_offset_view[compact_index]
                trace_stop = <size_t> compact_trace_offset_view[
                    compact_index + 1
                ]
                if (
                    trace_start > trace_stop
                    or trace_stop > journal.compact_trace_count
                    or compact_result.row_index != row
                    or compact_result.profile_index != profile
                    or compact_result.sequence_index
                    != journal_row.sequence_index
                    or compact_result.envelope_begin != interval.begin
                    or compact_result.envelope_end != interval.end
                    or compact_result.action not in (
                        PLAN7_DOMAIN_RESCORE_CPU_REQUIRED,
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT,
                    )
                    or compact_result.status not in (
                        PLAN7_DOMAIN_RESCORE_OK,
                        PLAN7_DOMAIN_RESCORE_ERANGE,
                        PLAN7_DOMAIN_RESCORE_ENORESULT,
                        PLAN7_DOMAIN_RESCORE_ECAP,
                        PLAN7_DOMAIN_RESCORE_EMPTY,
                    )
                    or compact_result.has_own_scales > 1
                    or compact_result.reserved != 0
                    or compact_result.reserved2 != 0
                ):
                    raise ValueError("continuation compact result is invalid")
                if compact_index == compact_start:
                    compact_row_action = compact_result.action
                elif compact_result.action != compact_row_action:
                    raise ValueError("continuation compact row is not atomic")

                if compact_result.action == (
                    PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                ):
                    if (
                        compact_result.status != PLAN7_DOMAIN_RESCORE_OK
                        or compact_result.has_own_scales
                        or not isfinite(compact_result.forward_score)
                        or not isfinite(compact_result.backward_score)
                        or not isfinite(compact_result.oa_score)
                        or not isfinite(compact_result.domain_correction)
                        or not isfinite(compact_result.score_consistency)
                        or compact_result.score_consistency < 0.0
                        or compact_result.score_consistency > 2.0e-3
                        or compact_result.alignment_begin
                        < compact_result.envelope_begin
                        or compact_result.alignment_begin
                        > compact_result.alignment_end
                        or compact_result.alignment_end
                        > compact_result.envelope_end
                        or compact_result.model_begin < 1
                        or compact_result.model_begin
                        > compact_result.model_end
                        or compact_result.model_end
                        > <uint32_t> optimized_profile._om.M
                        or trace_start == trace_stop
                    ):
                        raise ValueError(
                            "continuation device compact result is invalid"
                        )
                    for residue in range(PLAN7_DOMAIN_RESCORE_NULL2_COUNT):
                        null2_value = compact_null2_view[
                            compact_index
                            * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                            + residue
                        ]
                        if not isfinite(null2_value) or null2_value <= 0.0:
                            raise ValueError(
                                "continuation compact null2 is invalid"
                            )
                    device_result_count += 1
                else:
                    if trace_start != trace_stop:
                        raise ValueError(
                            "CPU compact fallback carries a trace"
                        )
                    for residue in range(PLAN7_DOMAIN_RESCORE_NULL2_COUNT):
                        if not isnan(compact_null2_view[
                            compact_index
                            * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
                            + residue
                        ]):
                            raise ValueError(
                                "CPU compact fallback carries null2"
                            )
                    cpu_required_count += 1
            if compact_start != compact_stop:
                if compact_row_action == PLAN7_DOMAIN_RESCORE_DEVICE_RESULT:
                    expected_compact_route = PLAN7_CONTINUATION_COMPACT_DEVICE
                else:
                    expected_compact_route = (
                        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                    )
            if journal_row.compact_route != expected_compact_route:
                raise ValueError("continuation compact row route differs")
            row += 1
            forward_cursor += 1
        if row != row_stop:
            raise ValueError("continuation journal has an extra Forward row")

    if journal.generation_compact_domains and (
        simple_row_count != journal.rescore_simple_row_count
        or (
            journal.compact_result_count != 0
            and (
                device_result_count != journal.rescore_device_result_count
                or cpu_required_count != journal.rescore_cpu_required_count
            )
        )
    ):
        raise ValueError("continuation compact route accounting differs")

    if PyCapsule_SetDestructor(
        capsule, <PyCapsule_Destructor> NULL
    ) != 0:
        raise RuntimeError("cannot consume continuation journal capsule")
    if PyCapsule_SetPointer(capsule, &_consumed_journal_sentinel) != 0:
        storage._data = NULL
        storage._size = 0
        free(journal)
        PyCapsule_SetName(capsule, PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME)
        raise RuntimeError("cannot retire continuation journal capsule")
    storage._owns = True
    if PyCapsule_SetName(
        capsule, PLAN7_CONTINUATION_JOURNAL_CONSUMED_NAME
    ) != 0:
        raise RuntimeError("cannot mark continuation journal consumed")
    return (
        storage_view,
        postfilter_view,
        postfilter_offset_view,
        forward_view,
        forward_offset_view,
        special_offset_view,
        special_view,
        profile_offset_view,
        row_view,
        region_offset_view,
        region_view,
        compact_row_offset_view,
        compact_result_view,
        compact_trace_offset_view,
        compact_trace_view,
        compact_null2_view,
        guard_bits,
        journal.generation_tail_fingerprint,
        journal.generation_compact_domains,
        journal.rescore_simple_row_count,
        journal.rescore_device_result_count,
        journal.rescore_cpu_required_count,
        journal.rescore_numeric_fallback_count,
        journal.rescore_cap_fallback_count,
        journal.rescore_global_cpu_fallback_count,
    )


def _seal_postfilter_batch_bound(
    queries,
    optimized_profiles,
    DigitalSequenceBlock sequences,
    postfilter_records,
    postfilter_offsets,
    residue_offsets,
    double f1,
    background_fingerprint,
    forward_records=None,
    forward_offsets=None,
    special_offsets=None,
    specials=None,
    expected_forward_indices=None,
    uint64_t generation_f2_bits=0,
    uint64_t generation_f3_bits=0,
    generation_bias_filter=False,
    continuation_journal=None,
    selection_identity=None,
    selection_identity_tokens=None,
    domain_pipeline=None,
):
    """Seal one already-owned post-filter batch after one complete validation.

    This private factory's HMM, profile, and target arguments must come from
    the adapter's hidden registries while its pair and sequence locks are
    held. It defensively freezes every caller buffer before validating it.
    The returned object exposes neither those buffers nor native pointers;
    the provenance adapter keeps the matching pair and pipeline locks around
    every search.
    """
    cdef tuple query_tuple = tuple(queries)
    cdef tuple profile_tuple = tuple(optimized_profiles)
    cdef object postfilter_owner = _immutable_owned_view(
        postfilter_records, "B"
    )
    cdef object postfilter_offset_owner = _immutable_owned_view(
        postfilter_offsets, "Q"
    )
    cdef object residue_offset_owner = _immutable_owned_view(
        residue_offsets, "Q"
    )
    cdef object background_owner = _immutable_owned_view(
        background_fingerprint, "B"
    )
    cdef const uint8_t[::1] postfilter_view = postfilter_owner
    cdef const uint64_t[::1] postfilter_offset_view = postfilter_offset_owner
    cdef const uint64_t[::1] residue_offset_view = residue_offset_owner
    cdef const uint8_t[::1] background_view = background_owner
    cdef size_t profile_count
    cdef size_t profile_index
    cdef size_t postfilter_count
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_count
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t forward_cursor
    cdef _forward_result forward_record
    cdef bint has_direct = False
    cdef bint has_forward_storage
    cdef bint row_has_external
    cdef bytearray external_rows
    cdef object empty_forward_records
    cdef object empty_forward_offsets
    cdef object empty_special_offsets
    cdef object empty_specials
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint32_t[::1] expected_forward_view
    cdef _SealedPostfilterBatch sealed
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL
    cdef _pipeline_from_filter_and_forward_scores_f forward_scores_seam = NULL
    cdef _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam = NULL
    cdef _pipeline_tail_snapshot pipeline_options
    cdef object journal_values = None
    cdef object empty_journal_storage = b""
    cdef object empty_journal_profile_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_rows = b""
    cdef object empty_journal_region_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_regions = b""
    cdef object empty_journal_compact_row_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_compact_results = b""
    cdef object empty_journal_compact_trace_offsets = bytes(sizeof(uint64_t))
    cdef object empty_journal_compact_traces = b""
    cdef object empty_journal_compact_null2 = b""
    cdef const uint8_t[::1] journal_storage_view = empty_journal_storage
    cdef const uint64_t[::1] journal_profile_offset_view = memoryview(
        empty_journal_profile_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_row_view = empty_journal_rows
    cdef const uint64_t[::1] journal_region_offset_view = memoryview(
        empty_journal_region_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_region_view = empty_journal_regions
    cdef const uint64_t[::1] journal_compact_row_offset_view = memoryview(
        empty_journal_compact_row_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_compact_result_view = (
        empty_journal_compact_results
    )
    cdef const uint64_t[::1] journal_compact_trace_offset_view = memoryview(
        empty_journal_compact_trace_offsets
    ).cast("Q")
    cdef const uint8_t[::1] journal_compact_trace_view = (
        empty_journal_compact_traces
    )
    cdef const float[::1] journal_compact_null2_view = memoryview(
        empty_journal_compact_null2
    ).cast("f")
    cdef uint32_t journal_guard_bits = 0
    cdef uint64_t expected_session_id = 0
    cdef uint64_t expected_selection_id = 0
    cdef size_t frequency_bytes
    cdef _double_bits generation_f1
    cdef _float_bits expected_rt
    cdef Pipeline journal_pipeline
    cdef object expected_identity_owner
    cdef const uint64_t[::1] expected_identity_view

    if postfilter_offset_view.shape[0] == 0:
        raise ValueError("post-filter row offsets need an initial zero")
    profile_count = <size_t> postfilter_offset_view.shape[0] - 1
    if len(query_tuple) != profile_count:
        raise ValueError("HMM count differs from post-filter row count")
    if len(profile_tuple) != profile_count:
        raise ValueError(
            "optimized-profile count differs from post-filter row count"
        )
    if postfilter_offset_view[0] != 0:
        raise ValueError("post-filter row offsets must start at zero")
    if <size_t> postfilter_view.shape[0] % sizeof(_postfilter_result) != 0:
        raise ValueError("post-filter result storage has trailing bytes")
    postfilter_count = (
        <size_t> postfilter_view.shape[0] // sizeof(_postfilter_result)
    )
    if postfilter_offset_view[profile_count] != postfilter_count:
        raise ValueError("post-filter row offsets do not span result storage")
    if not isfinite(f1) or f1 < 0.0 or f1 > 1.0:
        raise ValueError("sealed F1 must be a finite number in [0, 1]")
    if background_view.shape[0] != (
        (<size_t> sequences.alphabet.K + 1) * sizeof(float)
    ):
        raise ValueError("canonical background fingerprint has the wrong size")
    if type(generation_bias_filter) is not bool:
        raise TypeError("generation bias filter must be bool")

    _validate_sealed_residue_offsets(sequences, residue_offset_view)

    has_forward_storage = any(
        value is not None
        for value in (forward_records, forward_offsets, special_offsets, specials)
    )
    if has_forward_storage and any(
        value is None
        for value in (forward_records, forward_offsets, special_offsets, specials)
    ):
        raise ValueError("Forward storage must be supplied as one complete batch")
    if has_forward_storage:
        forward_view = _immutable_owned_view(forward_records, "B")
        forward_offset_view = _immutable_owned_view(forward_offsets, "Q")
        special_offset_view = _immutable_owned_view(special_offsets, "Q")
        special_view = _immutable_owned_view(specials, "f")
    else:
        empty_forward_records = b""
        empty_forward_offsets = bytes((profile_count + 1) * sizeof(uint64_t))
        empty_special_offsets = bytes(sizeof(uint64_t))
        empty_specials = b""
        forward_view = empty_forward_records
        forward_offset_view = memoryview(empty_forward_offsets).cast("Q")
        special_offset_view = memoryview(empty_special_offsets).cast("Q")
        special_view = memoryview(empty_specials).cast("f")

    if expected_forward_indices is not None:
        if not has_forward_storage:
            raise ValueError(
                "expected Forward indexes require complete Forward storage"
            )
        expected_forward_view = expected_forward_indices

    if forward_offset_view.shape[0] != profile_count + 1:
        raise ValueError("Forward row-offset count differs from post-filter rows")
    if forward_offset_view[0] != 0:
        raise ValueError("Forward row offsets must start at zero")
    if <size_t> forward_view.shape[0] % sizeof(_forward_result) != 0:
        raise ValueError("Forward result storage has trailing bytes")
    forward_count = <size_t> forward_view.shape[0] // sizeof(_forward_result)
    if forward_offset_view[profile_count] != forward_count:
        raise ValueError("Forward row offsets do not span result storage")
    if special_offset_view.shape[0] != forward_count + 1:
        raise ValueError("Forward special-offset count differs from result count")
    if special_offset_view[0] != 0:
        raise ValueError("Forward global special offsets must start at zero")
    if special_offset_view[forward_count] != <uint64_t> special_view.shape[0]:
        raise ValueError("Forward special offsets do not span matrix storage")
    if expected_forward_indices is not None:
        if expected_forward_view.shape[0] != forward_count:
            raise ValueError("Forward result count differs from selected inputs")
        for forward_cursor in range(forward_count):
            memcpy(
                &forward_record,
                &forward_view[forward_cursor * sizeof(_forward_result)],
                sizeof(_forward_result),
            )
            if forward_record.sequence_index != expected_forward_view[forward_cursor]:
                raise ValueError("Forward result order differs from selected inputs")

    external_rows = bytearray(profile_count)
    for profile_index in range(profile_count):
        if type(query_tuple[profile_index]) is not _pyhmmer.plan7.HMM:
            raise TypeError("sealed queries must be exactly pyhmmer.plan7.HMM")
        if (
            type(profile_tuple[profile_index])
            is not _pyhmmer.plan7.OptimizedProfile
        ):
            raise TypeError(
                "sealed profiles must be exactly pyhmmer.plan7.OptimizedProfile"
            )
        _validate_sealed_pair(
            <HMM> query_tuple[profile_index],
            <OptimizedProfile> profile_tuple[profile_index],
            sequences,
        )

        postfilter_start = <size_t> postfilter_offset_view[profile_index]
        postfilter_stop = <size_t> postfilter_offset_view[profile_index + 1]
        if (
            postfilter_start > postfilter_stop
            or postfilter_stop > postfilter_count
        ):
            raise ValueError("post-filter row offsets are not monotone")
        if _validate_postfilter_records(
            <OptimizedProfile> profile_tuple[profile_index],
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
        ):
            has_direct = True

        forward_start = <size_t> forward_offset_view[profile_index]
        forward_stop = <size_t> forward_offset_view[profile_index + 1]
        if forward_start > forward_stop or forward_stop > forward_count:
            raise ValueError("Forward row offsets are not monotone")
        row_has_external = _validate_forward_augmentation(
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            forward_view[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            special_offset_view[forward_start:forward_stop + 1],
            special_view,
        )
        external_rows[profile_index] = <uint8_t> row_has_external

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct post-filter records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )
    if has_forward_storage:
        forward_scores_seam = _cached_filter_and_forward_scores_seam()

    if (
        continuation_journal is not None
        or selection_identity is not None
        or selection_identity_tokens is not None
        or domain_pipeline is not None
    ):
        raise TypeError(
            "continuation journals require the fused profile-selection factory"
        )
    sealed = _SealedPostfilterBatch()
    sealed._queries = query_tuple
    sealed._optimized_profiles = profile_tuple
    sealed._sequences = sequences
    sealed._postfilter_records = postfilter_view
    sealed._postfilter_offsets = postfilter_offset_view
    sealed._residue_offsets = residue_offset_view
    sealed._forward_records = forward_view
    sealed._forward_offsets = forward_offset_view
    sealed._special_offsets = special_offset_view
    sealed._specials = special_view
    sealed._row_has_external = bytes(external_rows)
    sealed._background_fingerprint = background_view
    sealed._journal_storage = journal_storage_view
    sealed._journal_profile_offsets = journal_profile_offset_view
    sealed._journal_rows = journal_row_view
    sealed._journal_region_offsets = journal_region_offset_view
    sealed._journal_regions = journal_region_view
    sealed._journal_compact_row_offsets = journal_compact_row_offset_view
    sealed._journal_compact_results = journal_compact_result_view
    sealed._journal_compact_trace_offsets = (
        journal_compact_trace_offset_view
    )
    sealed._journal_compact_traces = journal_compact_trace_view
    sealed._journal_compact_null2 = journal_compact_null2_view
    sealed._f1 = f1
    sealed._generation_f2_bits = generation_f2_bits
    sealed._generation_f3_bits = generation_f3_bits
    sealed._generation_bias_filter = generation_bias_filter
    sealed._journal_guard_bits = journal_guard_bits
    sealed._generation_tail_fingerprint = 0
    sealed._rescore_simple_row_count = 0
    sealed._rescore_device_result_count = 0
    sealed._rescore_cpu_required_count = 0
    sealed._rescore_numeric_fallback_count = 0
    sealed._rescore_cap_fallback_count = 0
    sealed._rescore_global_cpu_fallback_count = 0
    if continuation_journal is not None:
        sealed._pipeline_options = pipeline_options
    sealed._filter_scores_seam = filter_scores_seam
    sealed._forward_scores_seam = forward_scores_seam
    sealed._simple_regions_seam = simple_regions_seam
    sealed._compact_tail_fingerprint = NULL
    sealed._compact_domains_seam = NULL
    sealed._ready = True
    return sealed


def _seal_profile_selection_continuation_bound(
    queries,
    optimized_profiles,
    DigitalSequenceBlock sequences,
    residue_offsets,
    double f1,
    background_fingerprint,
    continuation_journal,
    selection_identity,
    selection_identity_tokens,
    profile_fingerprints,
    uint64_t batch_generation,
    sequence_content_fingerprint,
    Pipeline pipeline,
    double guard_band,
):
    """Consume one package-internal journal into an opaque continuation batch.

    The capsule and underscore factory are trusted in-process transport. Their
    integrity checks reject accidental corruption and cross-binding, not a
    caller deliberately reaching private extension APIs or ctypes.
    """
    cdef tuple query_tuple = tuple(queries)
    cdef tuple profile_tuple = tuple(optimized_profiles)
    cdef size_t profile_count = len(profile_tuple)
    cdef object residue_offset_owner = _immutable_owned_view(
        residue_offsets, "Q"
    )
    cdef object background_owner = _immutable_owned_view(
        background_fingerprint, "B"
    )
    cdef object identity_owner = _immutable_owned_view(
        selection_identity_tokens, "Q"
    )
    cdef object profile_fingerprint_owner = _immutable_owned_view(
        profile_fingerprints, "B"
    )
    cdef object sequence_fingerprint_owner = _immutable_owned_view(
        sequence_content_fingerprint, "B"
    )
    cdef const uint64_t[::1] residue_offset_view = residue_offset_owner
    cdef const uint8_t[::1] background_view = background_owner
    cdef const uint64_t[::1] identity_view = identity_owner
    cdef const uint8_t[::1] profile_fingerprint_view = (
        profile_fingerprint_owner
    )
    cdef const uint8_t[::1] sequence_fingerprint_view = (
        sequence_fingerprint_owner
    )
    cdef object journal_values
    cdef const uint8_t[::1] journal_storage_view
    cdef const uint8_t[::1] postfilter_view
    cdef const uint64_t[::1] postfilter_offset_view
    cdef const uint8_t[::1] forward_view
    cdef const uint64_t[::1] forward_offset_view
    cdef const uint64_t[::1] special_offset_view
    cdef const float[::1] special_view
    cdef const uint64_t[::1] journal_profile_offset_view
    cdef const uint8_t[::1] journal_row_view
    cdef const uint64_t[::1] journal_region_offset_view
    cdef const uint8_t[::1] journal_region_view
    cdef const uint64_t[::1] journal_compact_row_offset_view
    cdef const uint8_t[::1] journal_compact_result_view
    cdef const uint64_t[::1] journal_compact_trace_offset_view
    cdef const uint8_t[::1] journal_compact_trace_view
    cdef const float[::1] journal_compact_null2_view
    cdef _pipeline_tail_snapshot pipeline_options
    cdef _SealedPostfilterBatch sealed
    cdef _pipeline_from_filter_scores_f filter_scores_seam
    cdef _pipeline_from_filter_and_forward_scores_f forward_scores_seam
    cdef _pipeline_from_filter_and_forward_simple_regions_f simple_regions_seam
    cdef _pipeline_from_filter_forward_compact_domains_f compact_domains_seam
    cdef _pipeline_compact_tail_fingerprint_f compact_tail_fingerprint
    cdef size_t profile
    cdef size_t postfilter_count
    cdef size_t forward_count
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef bint has_direct = False
    cdef bint row_has_external
    cdef bytearray external_rows = bytearray(profile_count)
    cdef uint64_t expected_session_id
    cdef uint64_t expected_selection_id
    cdef uint32_t journal_guard_bits
    cdef _float_bits expected_guard
    cdef size_t frequency_bytes
    cdef uint64_t expected_tail_fingerprint = 0
    cdef uint64_t generation_tail_fingerprint = 0
    cdef bint generation_compact_domains = False
    cdef uint64_t rescore_simple_row_count = 0
    cdef uint64_t rescore_device_result_count = 0
    cdef uint64_t rescore_cpu_required_count = 0
    cdef uint64_t rescore_numeric_fallback_count = 0
    cdef uint64_t rescore_cap_fallback_count = 0
    cdef uint64_t rescore_global_cpu_fallback_count = 0

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly pyhmmer.plan7.Pipeline")
    if len(query_tuple) != profile_count:
        raise ValueError("HMM count differs from optimized-profile count")
    if (
        type(selection_identity) is not tuple
        or len(selection_identity) != 2
        or type(selection_identity[0]) is not int
        or type(selection_identity[1]) is not int
    ):
        raise TypeError("continuation selection identity is invalid")
    expected_session_id = selection_identity[0]
    expected_selection_id = selection_identity[1]
    if expected_session_id == 0 or expected_selection_id == 0:
        raise ValueError("continuation selection identity is zero")
    if batch_generation == 0:
        raise ValueError("continuation target generation is zero")
    if sequence_fingerprint_view.shape[0] != 32:
        raise ValueError("sequence content fingerprint has the wrong size")
    if profile_fingerprint_view.shape[0] != (
        profile_count * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
    ):
        raise ValueError("optimized-profile fingerprint storage has the wrong size")
    if background_view.shape[0] != (
        (<size_t> sequences.alphabet.K + 1) * sizeof(float)
    ):
        raise ValueError("canonical background fingerprint has the wrong size")
    _validate_sealed_residue_offsets(sequences, residue_offset_view)
    _validate_simple_region_generation_bound(
        pipeline,
        f1,
        pipeline._pli.F2,
        pipeline._pli.F3,
        True,
        guard_band,
    )
    _capture_pipeline_tail_options(pipeline, &pipeline_options)
    compact_domains_seam = _cached_compact_domains_seam()
    compact_tail_fingerprint = _compact_tail_fingerprint_cache
    if compact_domains_seam != NULL and compact_tail_fingerprint != NULL:
        with nogil:
            expected_tail_fingerprint = compact_tail_fingerprint(
                pipeline._pli
            )
    frequency_bytes = <size_t> pipeline.background._bg.abc.K * sizeof(float)
    if (
        background_view.shape[0] != frequency_bytes + sizeof(float)
        or memcmp(
            &background_view[0],
            pipeline.background._bg.f,
            frequency_bytes,
        ) != 0
        or memcmp(
            &background_view[frequency_bytes],
            &pipeline.background._bg.omega,
            sizeof(float),
        ) != 0
    ):
        raise ValueError(
            "pipeline background differs from journal generation"
        )

    # Reject every live PyHMMER object before consuming the one-shot capsule.
    # The journal validator may safely dereference only after this completes.
    for profile in range(profile_count):
        if type(query_tuple[profile]) is not _pyhmmer.plan7.HMM:
            raise TypeError("sealed queries must be exactly pyhmmer.plan7.HMM")
        if (
            type(profile_tuple[profile])
            is not _pyhmmer.plan7.OptimizedProfile
        ):
            raise TypeError(
                "sealed profiles must be exactly pyhmmer.plan7.OptimizedProfile"
            )
        if (
            (<HMM> query_tuple[profile])._hmm == NULL
            or (<OptimizedProfile> profile_tuple[profile])._om == NULL
        ):
            raise ValueError("sealed HMM/profile storage is unavailable")
        if (
            not profile_tuple[profile].local
            or not profile_tuple[profile].multihit
        ):
            raise ValueError(
                "sealed optimized profiles must be local multihit"
            )
        _validate_sealed_pair(
            <HMM> query_tuple[profile],
            <OptimizedProfile> profile_tuple[profile],
            sequences,
        )

    journal_values = _consume_validate_continuation_journal(
        continuation_journal,
        profile_tuple,
        sequences,
        expected_session_id,
        expected_selection_id,
        identity_view,
        profile_fingerprint_view,
        batch_generation,
        sequence_fingerprint_view,
        f1,
        pipeline_options.f2_bits,
        pipeline_options.f3_bits,
        True,
        expected_tail_fingerprint,
    )
    journal_storage_view = journal_values[0]
    postfilter_view = journal_values[1]
    postfilter_offset_view = journal_values[2]
    forward_view = journal_values[3]
    forward_offset_view = journal_values[4]
    special_offset_view = journal_values[5]
    special_view = journal_values[6]
    journal_profile_offset_view = journal_values[7]
    journal_row_view = journal_values[8]
    journal_region_offset_view = journal_values[9]
    journal_region_view = journal_values[10]
    journal_compact_row_offset_view = journal_values[11]
    journal_compact_result_view = journal_values[12]
    journal_compact_trace_offset_view = journal_values[13]
    journal_compact_trace_view = journal_values[14]
    journal_compact_null2_view = journal_values[15]
    journal_guard_bits = journal_values[16]
    generation_tail_fingerprint = journal_values[17]
    generation_compact_domains = journal_values[18]
    rescore_simple_row_count = journal_values[19]
    rescore_device_result_count = journal_values[20]
    rescore_cpu_required_count = journal_values[21]
    rescore_numeric_fallback_count = journal_values[22]
    rescore_cap_fallback_count = journal_values[23]
    rescore_global_cpu_fallback_count = journal_values[24]
    expected_guard.value = <float> guard_band
    if journal_guard_bits != expected_guard.bits:
        raise ValueError("continuation journal guard band differs")

    postfilter_count = (
        <size_t> postfilter_view.shape[0] // sizeof(_postfilter_result)
    )
    forward_count = <size_t> forward_view.shape[0] // sizeof(_forward_result)
    for profile in range(profile_count):
        postfilter_start = <size_t> postfilter_offset_view[profile]
        postfilter_stop = <size_t> postfilter_offset_view[profile + 1]
        forward_start = <size_t> forward_offset_view[profile]
        forward_stop = <size_t> forward_offset_view[profile + 1]
        if (
            postfilter_start > postfilter_stop
            or postfilter_stop > postfilter_count
            or forward_start > forward_stop
            or forward_stop > forward_count
        ):
            raise ValueError("continuation row offsets are not monotone")
        if _validate_postfilter_records(
            <OptimizedProfile> profile_tuple[profile],
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
        ):
            has_direct = True
        row_has_external = _validate_forward_augmentation(
            sequences,
            postfilter_view[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            forward_view[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            special_offset_view[forward_start:forward_stop + 1],
            special_view,
        )
        external_rows[profile] = <uint8_t> row_has_external

    filter_scores_seam = _cached_filter_scores_seam()
    forward_scores_seam = _cached_filter_and_forward_scores_seam()
    simple_regions_seam = _cached_simple_regions_seam()
    if has_direct and filter_scores_seam == NULL:
        raise RuntimeError(
            "continuation batches require the private filter-score seam"
        )
    if forward_count and forward_scores_seam == NULL:
        raise RuntimeError(
            "continuation batches require the private Forward seam"
        )
    if journal_row_view.shape[0] and simple_regions_seam == NULL:
        raise RuntimeError(
            "continuation batches require the private simple-region seam"
        )

    sealed = _SealedPostfilterBatch()
    sealed._queries = query_tuple
    sealed._optimized_profiles = profile_tuple
    sealed._sequences = sequences
    sealed._postfilter_records = postfilter_view
    sealed._postfilter_offsets = postfilter_offset_view
    sealed._residue_offsets = residue_offset_view
    sealed._forward_records = forward_view
    sealed._forward_offsets = forward_offset_view
    sealed._special_offsets = special_offset_view
    sealed._specials = special_view
    sealed._row_has_external = bytes(external_rows)
    sealed._background_fingerprint = background_view
    sealed._journal_storage = journal_storage_view
    sealed._journal_profile_offsets = journal_profile_offset_view
    sealed._journal_rows = journal_row_view
    sealed._journal_region_offsets = journal_region_offset_view
    sealed._journal_regions = journal_region_view
    sealed._journal_compact_row_offsets = journal_compact_row_offset_view
    sealed._journal_compact_results = journal_compact_result_view
    sealed._journal_compact_trace_offsets = (
        journal_compact_trace_offset_view
    )
    sealed._journal_compact_traces = journal_compact_trace_view
    sealed._journal_compact_null2 = journal_compact_null2_view
    sealed._f1 = f1
    sealed._generation_f2_bits = pipeline_options.f2_bits
    sealed._generation_f3_bits = pipeline_options.f3_bits
    sealed._generation_bias_filter = True
    sealed._journal_guard_bits = journal_guard_bits
    sealed._generation_tail_fingerprint = generation_tail_fingerprint
    sealed._rescore_simple_row_count = rescore_simple_row_count
    sealed._rescore_device_result_count = rescore_device_result_count
    sealed._rescore_cpu_required_count = rescore_cpu_required_count
    sealed._rescore_numeric_fallback_count = rescore_numeric_fallback_count
    sealed._rescore_cap_fallback_count = rescore_cap_fallback_count
    sealed._rescore_global_cpu_fallback_count = (
        rescore_global_cpu_fallback_count
    )
    sealed._pipeline_options = pipeline_options
    sealed._filter_scores_seam = filter_scores_seam
    sealed._forward_scores_seam = forward_scores_seam
    sealed._simple_regions_seam = simple_regions_seam
    sealed._compact_tail_fingerprint = (
        compact_tail_fingerprint if generation_compact_domains else NULL
    )
    sealed._compact_domains_seam = (
        compact_domains_seam if generation_compact_domains else NULL
    )
    sealed._ready = True
    return sealed


cdef bint _sealed_background_matches(
    _SealedPostfilterBatch sealed,
    Pipeline pipeline,
) noexcept nogil:
    cdef size_t frequency_bytes = (
        <size_t> pipeline.background._bg.abc.K * sizeof(float)
    )
    if sealed._background_fingerprint.shape[0] != frequency_bytes + sizeof(float):
        return False
    if memcmp(
        &sealed._background_fingerprint[0],
        pipeline.background._bg.f,
        frequency_bytes,
    ) != 0:
        return False
    return memcmp(
        &sealed._background_fingerprint[frequency_bytes],
        &pipeline.background._bg.omega,
        sizeof(float),
    ) == 0


cdef object _sealed_search_result(
    TopHits hits,
    const _compact_consumption_statistics* statistics,
    bint return_compact_statistics,
):
    if not return_compact_statistics:
        return hits
    return hits, {
        "attempt_count": statistics.attempt_count,
        "accepted_count": statistics.accepted_count,
        "invalid_retry_count": statistics.invalid_retry_count,
        "threshold_retry_count": statistics.threshold_retry_count,
        "first_attempt": (
            None
            if statistics.attempt_count == 0
            else (
                statistics.first_row_index,
                statistics.first_profile_index,
                statistics.first_sequence_index,
                statistics.first_domain_count,
            )
        ),
    }


def _search_hmm_sealed_postfilter_bound(
    sealed_object,
    Py_ssize_t row,
    Pipeline pipeline,
    bint _return_compact_statistics=False,
):
    """Search one row whose complete immutable batch was already validated."""
    cdef _SealedPostfilterBatch sealed
    cdef HMM query
    cdef OptimizedProfile optimized_profile
    cdef size_t postfilter_start
    cdef size_t postfilter_stop
    cdef size_t forward_start
    cdef size_t forward_stop
    cdef size_t journal_start
    cdef size_t journal_stop
    cdef _double_bits live_f2
    cdef _double_bits live_f3
    cdef _double_bits generation_f1
    cdef bint use_forward
    cdef bint use_journal
    cdef _compact_consumption_statistics statistics
    cdef _compact_consumption_statistics* compact_statistics = NULL

    if _return_compact_statistics:
        statistics.attempt_count = 0
        statistics.accepted_count = 0
        statistics.invalid_retry_count = 0
        statistics.threshold_retry_count = 0
        statistics.first_row_index = 0
        statistics.first_profile_index = 0
        statistics.first_sequence_index = 0
        statistics.first_domain_count = 0
        compact_statistics = &statistics

    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly pyhmmer.plan7.Pipeline")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")
    if not pipeline.alphabet._eq(sealed._sequences.alphabet):
        raise AlphabetMismatch(pipeline.alphabet, sealed._sequences.alphabet)
    if pipeline._pli.F1 != sealed._f1:
        raise ValueError(
            f"pipeline F1 {pipeline.F1!r} does not match "
            f"candidate F1 {sealed._f1!r}"
        )
    if not _sealed_background_matches(sealed, pipeline):
        raise ValueError(
            "pipeline background does not match the canonical hmmpress background"
        )

    # Make the query copy before any pipeline mutation. Each successful call
    # consequently owns an independent TopHits.query, even when a row is reused.
    query = (<HMM> sealed._queries[row]).copy()
    optimized_profile = <OptimizedProfile> sealed._optimized_profiles[row]
    postfilter_start = <size_t> sealed._postfilter_offsets[row]
    postfilter_stop = <size_t> sealed._postfilter_offsets[row + 1]
    forward_start = <size_t> sealed._forward_offsets[row]
    forward_stop = <size_t> sealed._forward_offsets[row + 1]

    live_f2.value = pipeline._pli.F2
    live_f3.value = pipeline._pli.F3
    use_forward = (
        sealed._row_has_external[row]
        and sealed._forward_scores_seam != NULL
        and live_f2.bits == sealed._generation_f2_bits
        and live_f3.bits == sealed._generation_f3_bits
        and pipeline._pli.do_biasfilter == sealed._generation_bias_filter
    )
    if not use_forward:
        return _sealed_search_result(
            _search_postfilter_validated(
                pipeline,
                query,
                optimized_profile,
                sealed._sequences,
                sealed._postfilter_records[
                    postfilter_start * sizeof(_postfilter_result):
                    postfilter_stop * sizeof(_postfilter_result)
                ],
                &sealed._residue_offsets[0],
                sealed._filter_scores_seam,
            ),
            &statistics,
            _return_compact_statistics,
        )
    use_journal = (
        sealed._journal_storage.shape[0] != 0
        and sealed._simple_regions_seam != NULL
        and _pipeline_tail_options_match(&sealed._pipeline_options, pipeline)
    )
    if use_journal:
        journal_start = <size_t> sealed._journal_profile_offsets[row]
        journal_stop = <size_t> sealed._journal_profile_offsets[row + 1]
        generation_f1.value = sealed._f1
        return _sealed_search_result(
            _search_postfilter_forward_journal_validated(
                pipeline,
                query,
                optimized_profile,
                sealed._sequences,
                sealed._postfilter_records[
                    postfilter_start * sizeof(_postfilter_result):
                    postfilter_stop * sizeof(_postfilter_result)
                ],
                sealed._forward_records[
                    forward_start * sizeof(_forward_result):
                    forward_stop * sizeof(_forward_result)
                ],
                sealed._special_offsets[forward_start:forward_stop + 1],
                sealed._specials,
                sealed._journal_rows[
                    journal_start * sizeof(plan7_continuation_journal_row):
                    journal_stop * sizeof(plan7_continuation_journal_row)
                ],
                sealed._journal_region_offsets[journal_start:journal_stop + 1],
                sealed._journal_regions,
                sealed._journal_compact_row_offsets[
                    journal_start:journal_stop + 1
                ],
                sealed._journal_compact_results,
                sealed._journal_compact_trace_offsets,
                sealed._journal_compact_traces,
                sealed._journal_compact_null2,
                journal_start,
                sealed._generation_tail_fingerprint,
                &sealed._residue_offsets[0],
                generation_f1.bits,
                sealed._generation_f2_bits,
                sealed._generation_f3_bits,
                sealed._generation_bias_filter,
                sealed._filter_scores_seam,
                sealed._forward_scores_seam,
                sealed._simple_regions_seam,
                sealed._compact_tail_fingerprint,
                sealed._compact_domains_seam,
                compact_statistics,
            ),
            &statistics,
            _return_compact_statistics,
        )
    return _sealed_search_result(
        _search_postfilter_forward_validated(
            pipeline,
            query,
            optimized_profile,
            sealed._sequences,
            sealed._postfilter_records[
                postfilter_start * sizeof(_postfilter_result):
                postfilter_stop * sizeof(_postfilter_result)
            ],
            sealed._forward_records[
                forward_start * sizeof(_forward_result):
                forward_stop * sizeof(_forward_result)
            ],
            sealed._special_offsets[forward_start:forward_stop + 1],
            sealed._specials,
            &sealed._residue_offsets[0],
            sealed._filter_scores_seam,
            sealed._forward_scores_seam,
        ),
        &statistics,
        _return_compact_statistics,
    )


def _sealed_postfilter_candidate_count_bound(sealed_object, Py_ssize_t row):
    """Return one opaque batch's authentic post-filter row count."""
    cdef _SealedPostfilterBatch sealed
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready:
        raise TypeError("sealed batch was not created by the provenance adapter")
    if row < 0 or row >= len(sealed._queries):
        raise IndexError("sealed post-filter row out of range")
    return sealed._postfilter_offsets[row + 1] - sealed._postfilter_offsets[row]


def _sealed_continuation_statistics_bound(sealed_object):
    """Return immutable route counts without exposing journal rows."""
    cdef _SealedPostfilterBatch sealed
    cdef plan7_continuation_journal_row journal_row
    cdef size_t row
    cdef size_t row_count
    cdef size_t cpu_required = 0
    cdef size_t no_regions = 0
    cdef size_t simple = 0
    if type(sealed_object) is not _SealedPostfilterBatch:
        raise TypeError("sealed batch has the wrong extension type")
    sealed = <_SealedPostfilterBatch> sealed_object
    if not sealed._ready or sealed._journal_storage.shape[0] == 0:
        raise TypeError("sealed batch has no continuation journal")
    row_count = (
        <size_t> sealed._journal_rows.shape[0]
        // sizeof(plan7_continuation_journal_row)
    )
    for row in range(row_count):
        memcpy(
            &journal_row,
            &sealed._journal_rows[
                row * sizeof(plan7_continuation_journal_row)
            ],
            sizeof(plan7_continuation_journal_row),
        )
        if (
            journal_row.domain_status != DOMAIN_OK
            or journal_row.domain_route == DOMAIN_CPU_REQUIRED
            or journal_row.has_own_scales
            or journal_row.uncertain_count != 0
            or journal_row.multidomain_count != 0
        ):
            cpu_required += 1
        elif journal_row.domain_route == DOMAIN_NO_REGIONS:
            no_regions += 1
        elif journal_row.domain_route == DOMAIN_SIMPLE:
            simple += 1
        else:
            cpu_required += 1
    return {
        "row_count": row_count,
        "cpu_required_count": cpu_required,
        "no_region_count": no_regions,
        "simple_count": simple,
        "journal_bytes": sealed._journal_storage.shape[0],
        "guard_band_bits": sealed._journal_guard_bits,
        "compact_enabled": sealed._compact_domains_seam != NULL,
        "compact_simple_row_count": sealed._rescore_simple_row_count,
        "compact_device_result_count": (
            sealed._rescore_device_result_count
        ),
        "compact_cpu_required_count": (
            sealed._rescore_cpu_required_count
        ),
        "compact_numeric_fallback_count": (
            sealed._rescore_numeric_fallback_count
        ),
        "compact_cap_fallback_count": sealed._rescore_cap_fallback_count,
        "compact_global_cpu_fallback_count": (
            sealed._rescore_global_cpu_fallback_count
        ),
    }


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
        filter_scores_seam = _cached_filter_scores_seam()
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


def _search_hmm_postfilter_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint64_t[::1] residue_offsets,
):
    """Consume one provenance-bound row of exact CUDA post-filter records."""
    cdef bint has_direct
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly pyhmmer.plan7.Pipeline")
    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    has_direct = _validate_postfilter_records(
        optimized_profile, sequences, postfilter_records
    )
    _validate_postfilter_residue_offsets(
        sequences, postfilter_records, residue_offsets
    )

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct post-filter records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

    return _search_postfilter_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        postfilter_records,
        &residue_offsets[0],
        filter_scores_seam,
    )


def _search_hmm_postfilter_forward_bound(
    Pipeline pipeline,
    HMM query,
    OptimizedProfile optimized_profile,
    DigitalSequenceBlock sequences,
    const uint8_t[::1] postfilter_records,
    const uint8_t[::1] forward_records,
    const uint64_t[::1] special_offsets,
    const float[::1] specials,
    const uint64_t[::1] residue_offsets,
    uint64_t generation_f2_bits,
    uint64_t generation_f3_bits,
    bint generation_bias_filter,
):
    """Consume one provenance-bound post-filter row augmented by Forward.

    The original post-filter row remains authoritative. If the live pipeline
    options differ bit-for-bit from generation, the complete row takes the
    existing exact continuation and computes Forward on the CPU.
    """
    cdef bint has_direct
    cdef bint has_external
    cdef _double_bits live_f2
    cdef _double_bits live_f3
    cdef _pipeline_from_filter_scores_f filter_scores_seam = NULL
    cdef _pipeline_from_filter_and_forward_scores_f forward_scores_seam = NULL

    if type(pipeline) is not _pyhmmer.plan7.Pipeline:
        raise TypeError("pipeline must be exactly pyhmmer.plan7.Pipeline")
    _validate_inputs(pipeline, query, optimized_profile, sequences, False)
    has_direct = _validate_postfilter_records(
        optimized_profile, sequences, postfilter_records
    )
    _validate_postfilter_residue_offsets(
        sequences, postfilter_records, residue_offsets
    )
    has_external = _validate_forward_augmentation(
        sequences,
        postfilter_records,
        forward_records,
        special_offsets,
        specials,
    )

    if has_direct:
        filter_scores_seam = _cached_filter_scores_seam()
        if filter_scores_seam == NULL:
            raise RuntimeError(
                "direct post-filter records require the project-private "
                "p7_PipelineFromFilterScores HMMER seam"
            )

    live_f2.value = pipeline._pli.F2
    live_f3.value = pipeline._pli.F3
    if (
        not has_external
        or live_f2.bits != generation_f2_bits
        or live_f3.bits != generation_f3_bits
        or pipeline._pli.do_biasfilter != generation_bias_filter
    ):
        return _search_postfilter_validated(
            pipeline,
            query,
            optimized_profile,
            sequences,
            postfilter_records,
            &residue_offsets[0],
            filter_scores_seam,
        )

    forward_scores_seam = _cached_filter_and_forward_scores_seam()
    if forward_scores_seam == NULL:
        return _search_postfilter_validated(
            pipeline,
            query,
            optimized_profile,
            sequences,
            postfilter_records,
            &residue_offsets[0],
            filter_scores_seam,
        )

    return _search_postfilter_forward_validated(
        pipeline,
        query,
        optimized_profile,
        sequences,
        postfilter_records,
        forward_records,
        special_offsets,
        specials,
        &residue_offsets[0],
        filter_scores_seam,
        forward_scores_seam,
    )
