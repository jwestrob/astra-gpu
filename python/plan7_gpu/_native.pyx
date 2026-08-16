# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, int32_t, uintptr_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.math cimport isfinite
from libc.stdlib cimport calloc, free
from libc.string cimport memcmp, memcpy
from cpython.array cimport array as carray, clone
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_FromStringAndSize
from cpython.pycapsule cimport (
    PyCapsule_GetPointer,
    PyCapsule_IsValid,
    PyCapsule_New,
)
from cpython.pyport cimport PY_SSIZE_T_MAX
from libcpp.vector cimport vector
from libeasel cimport eslCONST_LOG2
from pyhmmer.plan7 cimport OptimizedProfile

import array as _array
import pyhmmer as _pyhmmer

from . import _abi as _abi_module
from ._fingerprint import (
    optimized_profile_fingerprint as _profile_fingerprint,
    sequence_content_fingerprint as _sequence_content_fingerprint,
)


PYHMMER_PRIVATE_ABI = "0.12.0"
PYHMMER_PRIVATE_ABI_SHA256 = PYHMMER_ABI_SHA256
if _pyhmmer.__version__ != PYHMMER_PRIVATE_ABI:
    raise ImportError(
        f"plan7_gpu._native requires PyHMMER {PYHMMER_PRIVATE_ABI}, "
        f"found {_pyhmmer.__version__}"
    )
_abi_module.validate_private_abi_platform()
_runtime_abi_sha256 = _abi_module.pyhmmer_abi_fingerprint()
if _runtime_abi_sha256 != PYHMMER_PRIVATE_ABI_SHA256:
    raise ImportError(
        "plan7_gpu._native was built against a different PyHMMER private ABI "
        f"({PYHMMER_PRIVATE_ABI_SHA256} != {_runtime_abi_sha256})"
    )


cdef carray _UINT32_ARRAY_TEMPLATE = _array.array("I")
cdef carray _UINT64_ARRAY_TEMPLATE = _array.array("Q")
cdef carray _FLOAT_ARRAY_TEMPLATE = _array.array("f")
cdef uint64_t _sealed_journal_build_count = 0
cdef uint64_t _sealed_journal_payload_bytes = 0
cdef uint64_t _sealed_journal_duplicate_python_bytes = 0

_DEVICE_CAPACITY_NAMES = (
    "input_residues",
    "input_offsets",
    "input_null_scores",
    "length_tjb",
    "results",
    "compact_scores",
    "profiles",
    "f1_profiles",
    "candidate_words",
    "bias_profiles",
    "bias_candidates",
    "bias_ssv_inputs",
    "bias_results",
    "bias_logp",
    "bias_log1mp",
    "postfilter_states",
    "postfilter_bias_inputs",
    "postfilter_bias_results",
    "postfilter_viterbi_results",
    "postfilter_length_transitions",
    "postfilter_msv_offsets",
    "postfilter_viterbi_offsets",
    "postfilter_dp",
    "postfilter_results",
    "forward_candidate_profiles",
    "forward_candidate_sequences",
    "forward_length_transitions",
    "forward_dp_offsets",
    "forward_x_offsets",
    "forward_dp",
    "forward_xmx",
    "forward_results",
    "forward_survivor_candidates",
    "forward_survivor_offsets",
    "forward_gathered",
)


cdef extern from * nogil:
    """
    extern "C" {
    #include <esl_gumbel.h>
    }
    static inline unsigned plan7_popcount_u32(uint32_t value) {
      return (unsigned) __builtin_popcount(value);
    }
    static inline unsigned plan7_ctz_u32(uint32_t value) {
      return (unsigned) __builtin_ctz(value);
    }
    """
    unsigned plan7_popcount_u32(uint32_t value)
    unsigned plan7_ctz_u32(uint32_t value)
    double esl_gumbel_surv(double, double, double)


cdef extern from "bias_cuda.h" nogil:
    cdef enum plan7_bias_action:
        PLAN7_BIAS_CPU_REQUIRED
        PLAN7_BIAS_DEFINITE_REJECT
        PLAN7_BIAS_DEFINITE_PASS

    cdef enum plan7_bias_cutoff_mode:
        PLAN7_BIAS_CUTOFF_INVALID
        PLAN7_BIAS_CUTOFF_SCORE
        PLAN7_BIAS_CUTOFF_ALWAYS_REJECT
        PLAN7_BIAS_CUTOFF_ALWAYS_PASS

    ctypedef struct plan7_bias_profile:
        float t10
        float t11
        float scale
        float cutoff_bit_score
        int32_t cutoff_mode
        uint32_t reserved
        float pi0
        float pi1
        float t02
        float t12
        float eo[29][2]

    ctypedef struct plan7_bias_result:
        uint32_t sequence_index
        float filtersc
        int16_t ssv_numerator
        uint8_t ssv_status
        uint8_t action

    int plan7_bias_pack_amino_profile(
        const float *background,
        const float *composition,
        int model_length,
        float scale,
        int cutoff_mode,
        float cutoff_bit_score,
        plan7_bias_profile *profile,
        char *error,
        size_t error_size,
    )

    int plan7_bias_filter_score_host(
        const plan7_bias_profile *profile,
        const uint8_t *residues,
        uint64_t length,
        float *filtersc,
    )

    int plan7_bias_rebias_decision(
        uint8_t ssv_status,
        int16_t ssv_numerator,
        float scale,
        float filtersc,
        int cutoff_mode,
        float cutoff_bit_score,
        float *bit_score,
    )

    int plan7_bias_host_environment_attested()
    int plan7_bias_environment_attested(char *reason, size_t reason_size)


cdef extern from "ssv_cuda.h" nogil:
    cdef enum plan7_ssv_status:
        PLAN7_SSV_OK
        PLAN7_SSV_ERANGE
        PLAN7_SSV_ENORESULT
        PLAN7_SSV_EMPTY

    cdef enum plan7_f1_action:
        PLAN7_F1_CPU_REQUIRED
        PLAN7_F1_DEFINITE_REJECT

    cdef enum plan7_f1_cutoff_mode:
        PLAN7_F1_CUTOFF_INVALID
        PLAN7_F1_CUTOFF_SCORE
        PLAN7_F1_CUTOFF_ALWAYS_REJECT
        PLAN7_F1_CUTOFF_ALWAYS_CPU

    ctypedef struct plan7_ssv_result:
        uint8_t xE
        uint8_t status
        uint8_t tjb
        uint8_t reserved
        int16_t numerator

    ctypedef struct plan7_ssv_profile:
        uint64_t score_offset
        uint64_t score_count
        int32_t score_stride
        int32_t model_length
        uint8_t tbm
        uint8_t tec
        uint8_t base
        uint8_t bias
        float scale

    ctypedef struct plan7_ssv_sequence_batch:
        pass

    ctypedef struct plan7_ssv_sequence_batch_view:
        uint64_t generation_id
        int device_ordinal
        int alphabet_size
        int host_float_environment_valid
        size_t sequence_count
        const uint64_t *host_lengths
        const uint8_t *device_residues
        const uint64_t *device_offsets
        uint64_t input_device_bytes

    ctypedef struct plan7_ssv_workspace_statistics:
        uint64_t postfilter_device_bytes
        uint64_t postfilter_dp_capacity_bytes
        uint64_t postfilter_growth_count
        uint64_t postfilter_run_count
        uint64_t forward_device_bytes
        uint64_t forward_dp_capacity_bytes
        uint64_t forward_xmx_capacity_bytes
        uint64_t forward_gather_capacity_bytes
        uint64_t forward_growth_count
        uint64_t forward_event_create_count
        uint64_t forward_run_count

    ctypedef struct plan7_profile_footprint:
        uint64_t profile_count
        uint64_t ssv_device_bytes
        uint64_t viterbi_device_bytes
        uint64_t viterbi_exact_rbv_upper_bytes
        uint64_t forward_device_bytes
        uint64_t bias_device_bytes
        uint64_t minimum_device_bytes
        uint64_t maximum_device_bytes

    ctypedef struct plan7_allocation_simulation:
        uint64_t peak_additional_bytes
        uint64_t final_additional_bytes
        uint64_t final_free_bytes
        uint64_t growth_count
        uint64_t first_unfit_index
        int32_t fits
        int32_t reserved

    cdef enum plan7_ssv_device_capacity:
        PLAN7_SSV_DEVICE_CAPACITY_COUNT

    ctypedef struct plan7_ssv_memory_snapshot:
        int32_t device_ordinal
        int32_t reserved
        uint64_t cuda_free_bytes
        uint64_t cuda_total_bytes
        uint64_t persistent_device_bytes
        uint64_t device_capacity_bytes[35]

    int plan7_cuda_device_count(char *error, size_t error_size)
    int plan7_cuda_memory_info(
        int *device_ordinal,
        uint64_t *free_bytes,
        uint64_t *total_bytes,
        char *error,
        size_t error_size,
    )
    int plan7_validate_device_ordinal(
        int owner_device,
        int current_device,
        char *error,
        size_t error_size,
    )
    int plan7_profile_footprint_compute(
        const uint32_t *model_lengths,
        size_t profile_count,
        plan7_profile_footprint *footprint,
        char *error,
        size_t error_size,
    )
    int plan7_profile_slice_cell_count(
        uint64_t profile_count,
        uint64_t target_count,
        uint64_t cell_limit,
        uint64_t *cell_count,
        char *error,
        size_t error_size,
    )
    int plan7_simulate_allocate_before_free(
        const uint64_t *current_capacities,
        const uint64_t *required_capacities,
        size_t capacity_count,
        uint64_t free_bytes,
        uint64_t *final_capacities,
        plan7_allocation_simulation *simulation,
        char *error,
        size_t error_size,
    )
    int plan7_tjb_for_length(float scale, uint64_t length)
    int plan7_ssv_f1_decision(
        uint8_t status,
        int16_t numerator,
        uint64_t length,
        float scale,
        float m_mu,
        float m_lambda,
        double f1,
        double *ret_p,
    )

    int plan7_ssv_f1_cutoff(
        float m_mu,
        float m_lambda,
        double f1,
        float *ret_bit_score,
    )

    int plan7_ssv_f1_cutoff_decision(
        uint8_t status,
        int16_t numerator,
        uint64_t length,
        float scale,
        int cutoff_mode,
        float cutoff_bit_score,
    )

    int plan7_ssv_sequence_batch_create(
        const uint8_t *residues,
        size_t residue_count,
        const uint64_t *offsets,
        size_t offset_count,
        int alphabet_size,
        plan7_ssv_sequence_batch **batch,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_destroy(
        plan7_ssv_sequence_batch **batch,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_view(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_sequence_batch_view *view,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_workspace_statistics(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_workspace_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_get_memory_snapshot(
        const plan7_ssv_sequence_batch *batch,
        plan7_ssv_memory_snapshot *snapshot,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_filter(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *striped_scores,
        size_t striped_score_count,
        int score_stride,
        int model_length,
        int alphabet_size,
        uint8_t tbm,
        uint8_t tec,
        uint8_t base,
        uint8_t bias,
        float scale,
        plan7_ssv_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_filter_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        plan7_ssv_result *profile_major_results,
        size_t result_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_f1_candidates_many(
        const plan7_ssv_sequence_batch *batch,
        const plan7_ssv_result *profile_major_results,
        size_t result_count,
        const float *scales,
        const float *m_mu,
        const float *m_lambda,
        size_t profile_count,
        double f1,
        const size_t *candidate_offsets,
        uint32_t *candidate_indices,
        size_t candidate_index_count,
        size_t *candidate_counts,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_f1_mask_many(
        plan7_ssv_sequence_batch *batch,
        const uint8_t *packed_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        const float *m_mu,
        const float *m_lambda,
        double f1,
        uint32_t *profile_major_candidate_words,
        size_t candidate_word_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_bias_candidates_many(
        plan7_ssv_sequence_batch *batch,
        const plan7_bias_profile *bias_profiles,
        size_t profile_count,
        const size_t *candidate_offsets,
        const uint32_t *candidate_indices,
        size_t candidate_count,
        plan7_bias_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )


cdef extern from "postfilter_cuda.h" nogil:
    cdef enum plan7_postfilter_abi:
        PLAN7_POSTFILTER_RECORD_VERSION
        PLAN7_POSTFILTER_RECORD_SIZE

    ctypedef struct plan7_postfilter_result:
        uint32_t sequence_index
        float filtersc
        int16_t msv_numerator
        uint8_t msv_status
        uint8_t action
        float vfsc

    ctypedef struct plan7_viterbi_database:
        pass

    ctypedef struct plan7_profile_session:
        pass

    ctypedef struct plan7_profile_selection:
        pass

    ctypedef struct plan7_forward_database:
        pass

    ctypedef struct plan7_profile_selection_view:
        uint64_t session_id
        uint64_t selection_id
        size_t profile_count
        const uint8_t *packed_scores
        size_t packed_score_count
        const plan7_ssv_profile *profiles
        const float *m_mu
        const float *m_lambda
        const float *v_mu
        const float *v_lambda
        const plan7_bias_profile *bias_templates
        const uintptr_t *identity_tokens
        uint64_t host_bytes

    ctypedef struct plan7_profile_session_statistics:
        uint64_t session_id
        uint64_t profile_count
        uint64_t worker_count
        uint64_t build_worker_count
        uint64_t selection_worker_count
        uint64_t selection_count
        uint64_t parallel_run_count
        uint64_t build_parallel_run_count
        uint64_t selection_parallel_run_count
        uint64_t host_bytes
        uint64_t ssv_score_bytes
        uint64_t bias_profile_bytes
        uint64_t viterbi_descriptor_bytes
        uint64_t viterbi_emission_bytes
        uint64_t viterbi_transition_bytes
        uint64_t viterbi_exact_rbv_bytes
        uint64_t forward_descriptor_bytes
        uint64_t forward_emission_bytes
        uint64_t forward_transition_bytes

    int plan7_viterbi_database_create(
        const uintptr_t *profile_pointers,
        size_t profile_count,
        plan7_viterbi_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_viterbi_database_destroy(
        plan7_viterbi_database **database,
        char *error,
        size_t error_size,
    )

    size_t plan7_viterbi_database_profile_count(
        const plan7_viterbi_database *database,
    )

    int plan7_profile_session_create(
        const uintptr_t *profile_pointers,
        size_t profile_count,
        const float *background,
        size_t background_count,
        size_t build_worker_count,
        size_t selection_worker_count,
        plan7_profile_session **session,
        char *error,
        size_t error_size,
    )

    int plan7_profile_session_destroy(
        plan7_profile_session **session,
        char *error,
        size_t error_size,
    )

    int plan7_profile_session_get_statistics(
        const plan7_profile_session *session,
        plan7_profile_session_statistics *statistics,
        char *error,
        size_t error_size,
    )

    int plan7_profile_session_select(
        plan7_profile_session *session,
        const size_t *profile_indices,
        size_t profile_count,
        plan7_profile_selection **selection,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_destroy(
        plan7_profile_selection **selection,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_get_view(
        const plan7_profile_selection *selection,
        plan7_profile_selection_view *view,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_stage_viterbi(
        const plan7_profile_selection *selection,
        plan7_viterbi_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_profile_selection_stage_forward(
        const plan7_profile_selection *selection,
        plan7_forward_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_sequence_batch_postfilter_candidates_many(
        plan7_ssv_sequence_batch *batch,
        const plan7_bias_profile *bias_profiles,
        size_t profile_count,
        const size_t *candidate_offsets,
        const uint32_t *candidate_indices,
        size_t candidate_count,
        const uintptr_t *source_profile_pointers,
        const plan7_viterbi_database *viterbi_database,
        plan7_postfilter_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )

    int plan7_ssv_filter_cuda(
        const uint8_t *striped_scores,
        size_t striped_score_count,
        int score_stride,
        int model_length,
        int alphabet_size,
        const uint8_t *residues,
        size_t residue_count,
        const uint64_t *offsets,
        size_t offset_count,
        size_t sequence_count,
        uint8_t tbm,
        uint8_t tec,
        uint8_t base,
        uint8_t bias,
        float scale,
        plan7_ssv_result *results,
        size_t result_count,
        char *error,
        size_t error_size,
    )


cdef extern from "forward_cuda.h" nogil:
    cdef enum plan7_forward_abi:
        PLAN7_FORWARD_RECORD_VERSION
        PLAN7_FORWARD_RECORD_SIZE
        PLAN7_FORWARD_MAX_GATHERED_BYTES

    cdef enum plan7_forward_action:
        PLAN7_FORWARD_CPU_REQUIRED
        PLAN7_FORWARD_DEFINITE_REJECT
        PLAN7_FORWARD_DEFINITE_PASS

    cdef enum plan7_forward_status:
        PLAN7_FORWARD_OK
        PLAN7_FORWARD_ERANGE
        PLAN7_FORWARD_ENORESULT
        PLAN7_FORWARD_EMPTY

    ctypedef struct plan7_forward_result:
        uint32_t sequence_index
        float fwdsc
        uint8_t status
        uint8_t action
        uint16_t reserved

    ctypedef struct plan7_forward_statistics:
        uint64_t generation_f3_bits
        uint64_t candidate_count
        uint64_t survivor_count
        uint64_t work_cells
        uint64_t dp_workspace_bytes
        uint64_t xmx_workspace_bytes
        uint64_t gather_workspace_bytes
        uint64_t gathered_xmx_bytes
        uint64_t output_byte_limit
        uint64_t output_cap_fallback_count
        float kernel_milliseconds
        float classification_milliseconds
        float gather_milliseconds
        float download_milliseconds
        float total_milliseconds

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

    ctypedef struct plan7_forward_output:
        pass

    int plan7_forward_database_create(
        const uintptr_t *profile_pointers,
        size_t profile_count,
        plan7_forward_database **database,
        char *error,
        size_t error_size,
    )

    int plan7_forward_database_destroy(
        plan7_forward_database **database,
        char *error,
        size_t error_size,
    )

    size_t plan7_forward_database_profile_count(
        const plan7_forward_database *database,
    )

    uint64_t plan7_forward_database_device_bytes(
        const plan7_forward_database *database,
    )

    float plan7_forward_database_pack_milliseconds(
        const plan7_forward_database *database,
    )

    float plan7_forward_database_upload_milliseconds(
        const plan7_forward_database *database,
    )

    int plan7_forward_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_run_batch_workspace(
        const plan7_forward_database *database,
        plan7_ssv_sequence_batch *batch,
        const uintptr_t *source_profile_pointers,
        size_t profile_count,
        const uint64_t *candidate_offsets,
        const uint32_t *candidate_indices,
        const float *filter_scores,
        size_t candidate_count,
        double f3,
        uint64_t gathered_byte_budget,
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_forward_output_destroy(
        plan7_forward_output **output,
        char *error,
        size_t error_size,
    )

    size_t plan7_forward_output_result_count(
        const plan7_forward_output *output,
    )

    const plan7_forward_result *plan7_forward_output_results(
        const plan7_forward_output *output,
    )

    const uint64_t *plan7_forward_output_special_offsets(
        const plan7_forward_output *output,
    )

    size_t plan7_forward_output_special_count(
        const plan7_forward_output *output,
    )

    const float *plan7_forward_output_specials(
        const plan7_forward_output *output,
    )

    const plan7_forward_statistics *plan7_forward_output_statistics(
        const plan7_forward_output *output,
    )

    const plan7_forward_provenance *plan7_forward_output_provenance(
        const plan7_forward_output *output,
    )


cdef extern from "backward_domain_cuda.h" nogil:
    cdef enum plan7_backward_domain_abi:
        PLAN7_BACKWARD_DOMAIN_RECORD_VERSION
        PLAN7_BACKWARD_DOMAIN_RECORD_SIZE
        PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE
        PLAN7_BACKWARD_DOMAIN_REGION_SIZE
        PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES
        PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS
        PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS

    cdef enum plan7_backward_domain_route:
        PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED
        PLAN7_BACKWARD_DOMAIN_NO_REGIONS
        PLAN7_BACKWARD_DOMAIN_SIMPLE

    cdef enum plan7_backward_domain_status:
        PLAN7_BACKWARD_DOMAIN_OK
        PLAN7_BACKWARD_DOMAIN_ERANGE
        PLAN7_BACKWARD_DOMAIN_ENORESULT
        PLAN7_BACKWARD_DOMAIN_EMPTY

    cdef enum plan7_backward_domain_test_fault:
        PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH
        PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH
        PLAN7_BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE

    ctypedef struct plan7_backward_domain_candidate:
        uint32_t profile_index
        uint32_t sequence_index

    ctypedef struct plan7_backward_domain_result:
        uint32_t profile_index
        uint32_t sequence_index
        float backward_score
        float nexpected
        uint32_t uncertain_count
        uint32_t region_count
        uint32_t multidomain_count
        uint8_t status
        uint8_t route
        uint8_t has_own_scales
        uint8_t reserved

    ctypedef struct plan7_domain_posterior:
        float btot
        float etot
        float mocc

    ctypedef struct plan7_simple_region:
        uint32_t begin
        uint32_t end

    ctypedef struct plan7_backward_domain_provenance:
        plan7_forward_provenance forward
        uint64_t threshold_hash
        uint64_t result_hash
        uint64_t region_hash
        uint64_t candidate_count
        uint64_t region_count

    ctypedef struct plan7_backward_domain_statistics:
        uint64_t candidate_count
        uint64_t device_result_count
        uint64_t cpu_required_count
        uint64_t work_cells
        uint64_t dp_workspace_bytes
        uint64_t backward_special_workspace_bytes
        uint64_t forward_special_workspace_bytes
        uint64_t posterior_bytes
        uint64_t simple_region_bytes
        uint64_t output_byte_limit
        uint64_t output_cap_fallback_count
        uint64_t work_cap_fallback_count
        uint64_t posterior_omitted_count
        uint64_t own_scale_count
        uint64_t threshold_uncertain_count
        uint64_t no_region_count
        uint64_t simple_count
        uint64_t multidomain_fallback_count
        float kernel_milliseconds
        float upload_milliseconds
        float download_milliseconds
        float total_milliseconds

    ctypedef struct plan7_backward_domain_output:
        pass

    int plan7_backward_domain_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const plan7_forward_provenance *provenance,
        const uint64_t *forward_offsets,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_unsealed_test_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_candidate *candidates,
        size_t candidate_count,
        const uint64_t *forward_offsets,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        uint64_t posterior_byte_budget,
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_output_destroy(
        plan7_backward_domain_output **output,
        char *error,
        size_t error_size,
    )

    size_t plan7_backward_domain_output_result_count(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_result *plan7_backward_domain_output_results(
        const plan7_backward_domain_output *output,
    )

    const uint64_t *plan7_backward_domain_output_posterior_offsets(
        const plan7_backward_domain_output *output,
    )

    size_t plan7_backward_domain_output_posterior_count(
        const plan7_backward_domain_output *output,
    )

    const plan7_domain_posterior *plan7_backward_domain_output_posteriors(
        const plan7_backward_domain_output *output,
    )

    const uint64_t *plan7_backward_domain_output_region_offsets(
        const plan7_backward_domain_output *output,
    )

    size_t plan7_backward_domain_output_region_count(
        const plan7_backward_domain_output *output,
    )

    const plan7_simple_region *plan7_backward_domain_output_regions(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_provenance *plan7_backward_domain_output_provenance(
        const plan7_backward_domain_output *output,
    )

    const plan7_backward_domain_statistics *plan7_backward_domain_output_statistics(
        const plan7_backward_domain_output *output,
    )

    int plan7_backward_domain_output_apply_test_fault(
        plan7_backward_domain_output *output,
        int fault,
        char *error,
        size_t error_size,
    )

    int plan7_backward_domain_cpu_oracle(
        uintptr_t source_profile_pointer,
        const uint8_t *residues,
        size_t residue_count,
        const float *forward_specials,
        size_t forward_special_count,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
        plan7_backward_domain_result *result,
        plan7_domain_posterior *posteriors,
        size_t posterior_count,
        plan7_simple_region *regions,
        size_t region_capacity,
        size_t *region_count,
        char *error,
        size_t error_size,
    )


cdef extern from "domain_rescore_cuda.h" nogil:
    cdef enum plan7_domain_rescore_abi:
        PLAN7_DOMAIN_RESCORE_RECORD_VERSION
        PLAN7_DOMAIN_RESCORE_RECORD_SIZE
        PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE
        PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
        PLAN7_DOMAIN_RESCORE_MAX_MATRIX_BYTES
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

    ctypedef struct plan7_domain_rescore_statistics:
        uint64_t upstream_row_count
        uint64_t simple_row_count
        uint64_t region_count
        uint64_t device_result_count
        uint64_t cpu_required_count
        uint64_t numeric_fallback_count
        uint64_t cap_fallback_count
        uint64_t global_cpu_fallback_count
        uint64_t work_cells
        uint64_t forward_matrix_bytes
        uint64_t posterior_matrix_bytes
        uint64_t special_workspace_bytes
        uint64_t trace_workspace_bytes
        uint64_t compact_output_byte_limit
        uint64_t compact_output_bytes
        float kernel_milliseconds
        float upload_milliseconds
        float download_milliseconds
        float total_milliseconds

    ctypedef struct plan7_domain_rescore_output:
        pass

    int plan7_domain_rescore_run(
        const plan7_forward_database *database,
        const plan7_ssv_sequence_batch *batch,
        const plan7_backward_domain_output *upstream,
        uint64_t compact_byte_budget,
        uint64_t matrix_byte_budget,
        uint64_t trace_byte_budget,
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    int plan7_domain_rescore_output_destroy(
        plan7_domain_rescore_output **output,
        char *error,
        size_t error_size,
    )

    size_t plan7_domain_rescore_output_result_count(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_result *plan7_domain_rescore_output_results(
        const plan7_domain_rescore_output *output,
    )

    const uint64_t *plan7_domain_rescore_output_trace_offsets(
        const plan7_domain_rescore_output *output,
    )

    size_t plan7_domain_rescore_output_trace_count(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_trace_step *plan7_domain_rescore_output_traces(
        const plan7_domain_rescore_output *output,
    )

    size_t plan7_domain_rescore_output_null2_count(
        const plan7_domain_rescore_output *output,
    )

    const float *plan7_domain_rescore_output_null2(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_provenance *plan7_domain_rescore_output_provenance(
        const plan7_domain_rescore_output *output,
    )

    const plan7_domain_rescore_statistics *plan7_domain_rescore_output_statistics(
        const plan7_domain_rescore_output *output,
    )

    int c_plan7_domain_rescore_own_scale_required_for_test \
            "plan7_domain_rescore_own_scale_required_for_test"(float xB)
    int c_plan7_domain_rescore_oatrace_j_predecessor_for_test \
            "plan7_domain_rescore_oatrace_j_predecessor_for_test"(
        float jpath,
        float epath,
        int j_loop_enabled,
        int e_loop_enabled,
    )

    int plan7_domain_rescore_cpu_oracle(
        uintptr_t source_profile_pointer,
        const uint8_t *residues,
        size_t residue_count,
        uint32_t envelope_begin,
        uint32_t envelope_end,
        plan7_domain_rescore_result *result,
        float *null2,
        size_t null2_count,
        plan7_domain_rescore_trace_step *trace,
        size_t trace_capacity,
        size_t *trace_count,
        char *error,
        size_t error_size,
    )


cdef extern from "continuation_journal.h":
    const char *PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME

    cdef enum plan7_continuation_journal_abi:
        PLAN7_CONTINUATION_JOURNAL_VERSION
        PLAN7_CONTINUATION_JOURNAL_MAGIC
        PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE

    cdef enum plan7_continuation_compact_route:
        PLAN7_CONTINUATION_COMPACT_NONE
        PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
        PLAN7_CONTINUATION_COMPACT_DEVICE

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


cdef union float_bits:
    float value
    uint32_t bits


cdef union double_bits:
    double value
    uint64_t bits


cdef void _continuation_journal_capsule_destroy(object capsule) noexcept:
    cdef void *pointer
    if PyCapsule_IsValid(capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME):
        pointer = PyCapsule_GetPointer(
            capsule, PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME
        )
        if pointer != NULL:
            free(pointer)


cdef bint _journal_append_storage(
    size_t *cursor,
    size_t count,
    size_t item_size,
    uint64_t *offset,
) noexcept:
    cdef size_t aligned
    cdef size_t byte_count
    if cursor[0] > (<size_t> -1) - 7:
        return False
    aligned = (cursor[0] + 7) & ~<size_t> 7
    if count != 0 and item_size > (<size_t> -1) // count:
        return False
    byte_count = count * item_size
    if aligned > (<size_t> -1) - byte_count:
        return False
    offset[0] = <uint64_t> aligned
    cursor[0] = aligned + byte_count
    return True


cdef object _build_continuation_journal_capsule(
    const plan7_profile_selection_view *view,
    const uint8_t *profile_fingerprints,
    const uint8_t *sequence_content_fingerprint,
    const plan7_postfilter_result *postfilter_records,
    size_t postfilter_count,
    const uint64_t *postfilter_offsets,
    const plan7_postfilter_result *candidate_records,
    const float *uncorrected_scores,
    const uint32_t *candidate_profiles,
    const size_t *pass_sources,
    const uint64_t *forward_profile_offsets,
    const uint64_t *profile_offsets,
    size_t pass_count,
    const uint64_t *pass_special_offsets,
    const plan7_forward_output *forward_output,
    const plan7_backward_domain_output *domain_output,
    const plan7_domain_rescore_output *rescore_output,
    uint64_t generation_tail_fingerprint,
    double f1,
    double f2,
    double f3,
    bint bias_filter,
    float rt1,
    float rt2,
    float rt3,
    float guard_band,
):
    cdef const plan7_forward_result *forward_results
    cdef const float *forward_specials
    cdef const plan7_forward_provenance *forward_provenance
    cdef const plan7_backward_domain_result *domain_results
    cdef const uint64_t *domain_region_offsets
    cdef const plan7_simple_region *domain_regions
    cdef const plan7_backward_domain_provenance *domain_provenance
    cdef const plan7_domain_rescore_result *rescore_results = NULL
    cdef const uint64_t *rescore_trace_offsets = NULL
    cdef const plan7_domain_rescore_trace_step *rescore_traces = NULL
    cdef const float *rescore_null2 = NULL
    cdef const plan7_domain_rescore_provenance *rescore_provenance = NULL
    cdef const plan7_domain_rescore_statistics *rescore_statistics = NULL
    cdef size_t forward_count
    cdef size_t special_count
    cdef size_t domain_count
    cdef size_t region_count
    cdef size_t compact_result_count = 0
    cdef size_t compact_trace_count = 0
    cdef size_t compact_null2_count = 0
    cdef size_t cursor = sizeof(plan7_continuation_journal)
    cdef size_t row
    cdef size_t source
    cdef size_t profile
    cdef size_t simple_row_count = 0
    cdef size_t device_result_count = 0
    cdef size_t cpu_required_count = 0
    cdef size_t compact_bytes = 0
    cdef size_t region_begin
    cdef size_t region_end
    cdef uint8_t row_action
    cdef const plan7_domain_rescore_result *compact_result
    cdef uint64_t profile_offsets_offset = 0
    cdef uint64_t identity_tokens_offset = 0
    cdef uint64_t profile_fingerprints_offset = 0
    cdef uint64_t postfilter_offsets_offset = 0
    cdef uint64_t postfilter_records_offset = 0
    cdef uint64_t forward_offsets_offset = 0
    cdef uint64_t forward_records_offset = 0
    cdef uint64_t forward_special_offsets_offset = 0
    cdef uint64_t rows_offset = 0
    cdef uint64_t special_offsets_offset = 0
    cdef uint64_t specials_offset = 0
    cdef uint64_t region_offsets_offset = 0
    cdef uint64_t regions_offset = 0
    cdef uint64_t compact_row_offsets_offset = 0
    cdef uint64_t compact_results_offset = 0
    cdef uint64_t compact_trace_offsets_offset = 0
    cdef uint64_t compact_traces_offset = 0
    cdef uint64_t compact_null2_offset = 0
    cdef plan7_continuation_journal *journal = NULL
    cdef plan7_continuation_journal_row *rows
    cdef uint64_t *target_u64
    cdef double_bits double_encoded
    cdef float_bits float_encoded
    cdef object capsule
    global _sealed_journal_build_count
    global _sealed_journal_payload_bytes

    forward_count = plan7_forward_output_result_count(forward_output)
    special_count = plan7_forward_output_special_count(forward_output)
    domain_count = plan7_backward_domain_output_result_count(domain_output)
    region_count = plan7_backward_domain_output_region_count(domain_output)
    if domain_count != pass_count:
        raise RuntimeError("Backward/domain journal result count changed")
    if rescore_output != NULL:
        compact_result_count = plan7_domain_rescore_output_result_count(
            rescore_output
        )
        compact_trace_count = plan7_domain_rescore_output_trace_count(
            rescore_output
        )
        compact_null2_count = plan7_domain_rescore_output_null2_count(
            rescore_output
        )
        rescore_results = plan7_domain_rescore_output_results(rescore_output)
        rescore_trace_offsets = (
            plan7_domain_rescore_output_trace_offsets(rescore_output)
        )
        rescore_traces = plan7_domain_rescore_output_traces(rescore_output)
        rescore_null2 = plan7_domain_rescore_output_null2(rescore_output)
        rescore_provenance = (
            plan7_domain_rescore_output_provenance(rescore_output)
        )
        rescore_statistics = (
            plan7_domain_rescore_output_statistics(rescore_output)
        )
        if generation_tail_fingerprint == 0:
            raise RuntimeError("compact-domain tail fingerprint is zero")
    elif generation_tail_fingerprint != 0:
        raise RuntimeError("compact-domain tail fingerprint has no output")
    if sizeof(uintptr_t) != sizeof(uint64_t):
        raise RuntimeError("profile identity tokens require a 64-bit process")
    if not (
        _journal_append_storage(
            &cursor, view.profile_count + 1, sizeof(uint64_t),
            &postfilter_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, postfilter_count, sizeof(plan7_postfilter_result),
            &postfilter_records_offset,
        )
        and _journal_append_storage(
            &cursor, view.profile_count + 1, sizeof(uint64_t),
            &forward_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, forward_count, sizeof(plan7_forward_result),
            &forward_records_offset,
        )
        and _journal_append_storage(
            &cursor, forward_count + 1, sizeof(uint64_t),
            &forward_special_offsets_offset,
        )
        and
        _journal_append_storage(
            &cursor, view.profile_count + 1, sizeof(uint64_t),
            &profile_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, view.profile_count, sizeof(uint64_t),
            &identity_tokens_offset,
        )
        and _journal_append_storage(
            &cursor,
            view.profile_count,
            PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
            &profile_fingerprints_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count, sizeof(plan7_continuation_journal_row),
            &rows_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count + 1, sizeof(uint64_t),
            &special_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, special_count, sizeof(float), &specials_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count + 1, sizeof(uint64_t),
            &region_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, region_count, sizeof(plan7_simple_region),
            &regions_offset,
        )
        and _journal_append_storage(
            &cursor, pass_count + 1, sizeof(uint64_t),
            &compact_row_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, compact_result_count,
            sizeof(plan7_domain_rescore_result),
            &compact_results_offset,
        )
        and _journal_append_storage(
            &cursor, compact_result_count + 1, sizeof(uint64_t),
            &compact_trace_offsets_offset,
        )
        and _journal_append_storage(
            &cursor, compact_trace_count,
            sizeof(plan7_domain_rescore_trace_step),
            &compact_traces_offset,
        )
        and _journal_append_storage(
            &cursor, compact_null2_count, sizeof(float),
            &compact_null2_offset,
        )
    ):
        raise OverflowError("continuation journal size overflows size_t")

    forward_results = plan7_forward_output_results(forward_output)
    forward_specials = plan7_forward_output_specials(forward_output)
    forward_provenance = plan7_forward_output_provenance(forward_output)
    domain_results = plan7_backward_domain_output_results(domain_output)
    domain_region_offsets = plan7_backward_domain_output_region_offsets(
        domain_output
    )
    domain_regions = plan7_backward_domain_output_regions(domain_output)
    domain_provenance = plan7_backward_domain_output_provenance(domain_output)
    if (
        forward_provenance == NULL
        or domain_provenance == NULL
        or domain_region_offsets == NULL
        or (forward_count and forward_results == NULL)
        or (special_count and forward_specials == NULL)
        or (pass_count and domain_results == NULL)
        or (region_count and domain_regions == NULL)
    ):
        raise RuntimeError("continuation journal source storage is incomplete")
    if memcmp(
        forward_provenance,
        &domain_provenance.forward,
        sizeof(plan7_forward_provenance),
    ) != 0:
        raise RuntimeError("Forward and Backward/domain provenance differ")
    if (
        forward_provenance.pass_count != pass_count
        or forward_provenance.special_count != special_count
        or domain_provenance.candidate_count != pass_count
        or domain_provenance.region_count != region_count
        or pass_special_offsets[pass_count] != special_count
        or domain_region_offsets[pass_count] != region_count
    ):
        raise RuntimeError("continuation journal provenance counts differ")

    for row in range(pass_count):
        if domain_results[row].route == PLAN7_BACKWARD_DOMAIN_SIMPLE:
            simple_row_count += 1
    if rescore_output != NULL:
        if (
            rescore_provenance == NULL
            or rescore_statistics == NULL
            or rescore_trace_offsets == NULL
            or (compact_result_count and rescore_results == NULL)
            or (compact_trace_count and rescore_traces == NULL)
            or (compact_null2_count and rescore_null2 == NULL)
            or compact_result_count
            > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
            // sizeof(plan7_domain_rescore_result)
            or compact_trace_count
            > PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES
            // sizeof(plan7_domain_rescore_trace_step)
            or compact_result_count
            > (<size_t> -1) // PLAN7_DOMAIN_RESCORE_NULL2_COUNT
            or compact_null2_count
            != compact_result_count * PLAN7_DOMAIN_RESCORE_NULL2_COUNT
            or compact_result_count not in (0, region_count)
            or memcmp(
                &rescore_provenance.backward,
                domain_provenance,
                sizeof(plan7_backward_domain_provenance),
            ) != 0
            or rescore_provenance.result_count != compact_result_count
            or rescore_provenance.trace_count != compact_trace_count
            or rescore_provenance.null2_count != compact_null2_count
            or rescore_statistics.upstream_row_count != pass_count
            or rescore_statistics.simple_row_count != simple_row_count
            or rescore_statistics.region_count != region_count
            or rescore_statistics.device_result_count
            + rescore_statistics.cpu_required_count != region_count
            or rescore_statistics.numeric_fallback_count
            + rescore_statistics.cap_fallback_count
            != rescore_statistics.cpu_required_count
            or rescore_statistics.global_cpu_fallback_count
            not in (0, region_count)
            or rescore_statistics.compact_output_byte_limit < sizeof(uint64_t)
            or rescore_statistics.compact_output_byte_limit
            > PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
            or rescore_statistics.compact_output_bytes
            > rescore_statistics.compact_output_byte_limit
        ):
            raise RuntimeError("compact-domain provenance or caps differ")
        compact_bytes = (
            compact_result_count * sizeof(plan7_domain_rescore_result)
            + (compact_result_count + 1) * sizeof(uint64_t)
            + compact_trace_count * sizeof(plan7_domain_rescore_trace_step)
            + compact_null2_count * sizeof(float)
        )
        if rescore_statistics.compact_output_bytes != compact_bytes:
            raise RuntimeError("compact-domain output byte count differs")
        if compact_result_count == 0:
            if (
                rescore_statistics.device_result_count != 0
                or rescore_statistics.cpu_required_count != region_count
                or rescore_statistics.global_cpu_fallback_count != region_count
                or compact_trace_count != 0
            ):
                raise RuntimeError("compact-domain global fallback differs")
        else:
            if rescore_statistics.global_cpu_fallback_count != 0:
                raise RuntimeError("compact-domain fallback accounting differs")
            for row in range(pass_count):
                region_begin = domain_region_offsets[row]
                region_end = domain_region_offsets[row + 1]
                if region_begin == region_end:
                    continue
                row_action = rescore_results[region_begin].action
                for source in range(region_begin, region_end):
                    compact_result = &rescore_results[source]
                    if (
                        compact_result.row_index != row
                        or compact_result.profile_index
                        != domain_results[row].profile_index
                        or compact_result.sequence_index
                        != domain_results[row].sequence_index
                        or compact_result.envelope_begin
                        != domain_regions[source].begin
                        or compact_result.envelope_end
                        != domain_regions[source].end
                        or compact_result.action != row_action
                        or compact_result.action not in (
                            PLAN7_DOMAIN_RESCORE_CPU_REQUIRED,
                            PLAN7_DOMAIN_RESCORE_DEVICE_RESULT,
                        )
                    ):
                        raise RuntimeError(
                            "compact-domain result order or row atomicity differs"
                        )
                    if compact_result.action == (
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                    ):
                        device_result_count += 1
                    else:
                        cpu_required_count += 1
            if (
                device_result_count
                != rescore_statistics.device_result_count
                or cpu_required_count
                != rescore_statistics.cpu_required_count
            ):
                raise RuntimeError("compact-domain action counts differ")

    journal = <plan7_continuation_journal *> calloc(1, cursor)
    if journal == NULL:
        raise MemoryError("continuation journal allocation failed")
    try:
        journal.magic = PLAN7_CONTINUATION_JOURNAL_MAGIC
        journal.version = PLAN7_CONTINUATION_JOURNAL_VERSION
        journal.header_size = sizeof(plan7_continuation_journal)
        journal.row_size = sizeof(plan7_continuation_journal_row)
        journal.region_size = sizeof(plan7_simple_region)
        journal.compact_result_size = sizeof(plan7_domain_rescore_result)
        journal.compact_trace_step_size = sizeof(
            plan7_domain_rescore_trace_step
        )
        journal.compact_null2_stride = PLAN7_DOMAIN_RESCORE_NULL2_COUNT
        journal.total_bytes = cursor
        journal.session_id = view.session_id
        journal.selection_id = view.selection_id
        journal.profile_count = view.profile_count
        journal.postfilter_count = postfilter_count
        journal.forward_count = forward_count
        journal.row_count = pass_count
        journal.special_count = special_count
        journal.region_count = region_count
        journal.compact_result_count = compact_result_count
        journal.compact_trace_offset_count = compact_result_count + 1
        journal.compact_trace_count = compact_trace_count
        journal.compact_null2_count = compact_null2_count
        journal.generation_tail_fingerprint = generation_tail_fingerprint
        if rescore_output != NULL:
            journal.rescore_simple_row_count = (
                rescore_statistics.simple_row_count
            )
            journal.rescore_device_result_count = (
                rescore_statistics.device_result_count
            )
            journal.rescore_cpu_required_count = (
                rescore_statistics.cpu_required_count
            )
            journal.rescore_numeric_fallback_count = (
                rescore_statistics.numeric_fallback_count
            )
            journal.rescore_cap_fallback_count = (
                rescore_statistics.cap_fallback_count
            )
            journal.rescore_global_cpu_fallback_count = (
                rescore_statistics.global_cpu_fallback_count
            )
            journal.rescore_compact_output_byte_limit = (
                rescore_statistics.compact_output_byte_limit
            )
            journal.rescore_compact_output_bytes = (
                rescore_statistics.compact_output_bytes
            )
        double_encoded.value = f1
        journal.generation_f1_bits = double_encoded.bits
        double_encoded.value = f2
        journal.generation_f2_bits = double_encoded.bits
        double_encoded.value = f3
        journal.generation_f3_bits = double_encoded.bits
        float_encoded.value = rt1
        journal.rt1_bits = float_encoded.bits
        float_encoded.value = rt2
        journal.rt2_bits = float_encoded.bits
        float_encoded.value = rt3
        journal.rt3_bits = float_encoded.bits
        float_encoded.value = guard_band
        journal.guard_band_bits = float_encoded.bits
        journal.generation_bias_filter = <uint8_t> bias_filter
        journal.generation_compact_domains = <uint8_t> (
            rescore_output != NULL
        )
        if rescore_output != NULL:
            journal.compact_global_fallback = <uint8_t> (
                rescore_statistics.global_cpu_fallback_count != 0
            )
        memcpy(
            journal.sequence_content_fingerprint,
            sequence_content_fingerprint,
            32,
        )
        journal.postfilter_offsets_offset = postfilter_offsets_offset
        journal.postfilter_records_offset = postfilter_records_offset
        journal.forward_offsets_offset = forward_offsets_offset
        journal.forward_records_offset = forward_records_offset
        journal.forward_special_offsets_offset = forward_special_offsets_offset
        journal.profile_offsets_offset = profile_offsets_offset
        journal.identity_tokens_offset = identity_tokens_offset
        journal.profile_fingerprints_offset = profile_fingerprints_offset
        journal.rows_offset = rows_offset
        journal.special_offsets_offset = special_offsets_offset
        journal.specials_offset = specials_offset
        journal.region_offsets_offset = region_offsets_offset
        journal.regions_offset = regions_offset
        journal.compact_row_offsets_offset = compact_row_offsets_offset
        journal.compact_results_offset = compact_results_offset
        journal.compact_trace_offsets_offset = compact_trace_offsets_offset
        journal.compact_traces_offset = compact_traces_offset
        journal.compact_null2_offset = compact_null2_offset
        journal.forward = forward_provenance[0]
        journal.backward = domain_provenance[0]
        if rescore_output != NULL:
            journal.rescore = rescore_provenance[0]

        memcpy(
            <uint8_t *> journal + postfilter_offsets_offset,
            postfilter_offsets,
            (view.profile_count + 1) * sizeof(uint64_t),
        )
        if postfilter_count:
            memcpy(
                <uint8_t *> journal + postfilter_records_offset,
                postfilter_records,
                postfilter_count * sizeof(plan7_postfilter_result),
            )
        memcpy(
            <uint8_t *> journal + forward_offsets_offset,
            forward_profile_offsets,
            (view.profile_count + 1) * sizeof(uint64_t),
        )
        if forward_count:
            memcpy(
                <uint8_t *> journal + forward_records_offset,
                forward_results,
                forward_count * sizeof(plan7_forward_result),
            )
        memcpy(
            <uint8_t *> journal + forward_special_offsets_offset,
            plan7_forward_output_special_offsets(forward_output),
            (forward_count + 1) * sizeof(uint64_t),
        )
        memcpy(
            <uint8_t *> journal + profile_offsets_offset,
            profile_offsets,
            (view.profile_count + 1) * sizeof(uint64_t),
        )
        target_u64 = <uint64_t *> (
            <uint8_t *> journal + identity_tokens_offset
        )
        for profile in range(view.profile_count):
            target_u64[profile] = <uint64_t> view.identity_tokens[profile]
        if view.profile_count:
            memcpy(
                <uint8_t *> journal + profile_fingerprints_offset,
                profile_fingerprints,
                view.profile_count
                * PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE,
            )
        memcpy(
            <uint8_t *> journal + special_offsets_offset,
            pass_special_offsets,
            (pass_count + 1) * sizeof(uint64_t),
        )
        if special_count:
            memcpy(
                <uint8_t *> journal + specials_offset,
                forward_specials,
                special_count * sizeof(float),
            )
        memcpy(
            <uint8_t *> journal + region_offsets_offset,
            domain_region_offsets,
            (pass_count + 1) * sizeof(uint64_t),
        )
        if region_count:
            memcpy(
                <uint8_t *> journal + regions_offset,
                domain_regions,
                region_count * sizeof(plan7_simple_region),
            )
        if compact_result_count:
            memcpy(
                <uint8_t *> journal + compact_row_offsets_offset,
                domain_region_offsets,
                (pass_count + 1) * sizeof(uint64_t),
            )
            memcpy(
                <uint8_t *> journal + compact_results_offset,
                rescore_results,
                compact_result_count * sizeof(plan7_domain_rescore_result),
            )
        if rescore_output != NULL:
            memcpy(
                <uint8_t *> journal + compact_trace_offsets_offset,
                rescore_trace_offsets,
                (compact_result_count + 1) * sizeof(uint64_t),
            )
        if compact_trace_count:
            memcpy(
                <uint8_t *> journal + compact_traces_offset,
                rescore_traces,
                compact_trace_count
                * sizeof(plan7_domain_rescore_trace_step),
            )
        if compact_null2_count:
            memcpy(
                <uint8_t *> journal + compact_null2_offset,
                rescore_null2,
                compact_null2_count * sizeof(float),
            )

        rows = <plan7_continuation_journal_row *> (
            <uint8_t *> journal + rows_offset
        )
        for row in range(pass_count):
            source = pass_sources[row]
            if source >= forward_count:
                raise RuntimeError("continuation journal source index changed")
            if (
                forward_results[source].action
                != PLAN7_FORWARD_DEFINITE_PASS
                or domain_results[row].profile_index
                != candidate_profiles[source]
                or domain_results[row].sequence_index
                != forward_results[source].sequence_index
            ):
                raise RuntimeError("continuation journal row identity changed")
            rows[row].profile_index = domain_results[row].profile_index
            rows[row].sequence_index = domain_results[row].sequence_index
            rows[row].usc = uncorrected_scores[source]
            rows[row].filtersc = candidate_records[source].filtersc
            rows[row].vfsc = candidate_records[source].vfsc
            rows[row].fwdsc = forward_results[source].fwdsc
            rows[row].backward_score = domain_results[row].backward_score
            rows[row].nexpected = domain_results[row].nexpected
            rows[row].uncertain_count = domain_results[row].uncertain_count
            rows[row].region_count = domain_results[row].region_count
            rows[row].multidomain_count = domain_results[row].multidomain_count
            rows[row].postfilter_status = candidate_records[source].msv_status
            rows[row].postfilter_action = candidate_records[source].action
            rows[row].forward_status = forward_results[source].status
            rows[row].forward_action = forward_results[source].action
            rows[row].domain_status = domain_results[row].status
            rows[row].domain_route = domain_results[row].route
            rows[row].has_own_scales = domain_results[row].has_own_scales
            rows[row].reserved = domain_results[row].reserved
            region_begin = domain_region_offsets[row]
            region_end = domain_region_offsets[row + 1]
            if region_end - region_begin > <size_t> 0xffffffff:
                raise RuntimeError("compact-domain row count exceeds uint32")
            if compact_result_count:
                rows[row].compact_result_count = <uint32_t> (
                    region_end - region_begin
                )
                if region_begin != region_end:
                    if rescore_results[region_begin].action == (
                        PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
                    ):
                        rows[row].compact_route = (
                            PLAN7_CONTINUATION_COMPACT_DEVICE
                        )
                    else:
                        rows[row].compact_route = (
                            PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                        )
            elif (
                rescore_output != NULL
                and rescore_statistics.global_cpu_fallback_count != 0
                and region_begin != region_end
            ):
                rows[row].compact_route = (
                    PLAN7_CONTINUATION_COMPACT_CPU_REQUIRED
                )

        journal.integrity_tag = plan7_continuation_journal_integrity(journal)
        capsule = PyCapsule_New(
            journal,
            PLAN7_CONTINUATION_JOURNAL_CAPSULE_NAME,
            _continuation_journal_capsule_destroy,
        )
        _sealed_journal_build_count += 1
        _sealed_journal_payload_bytes += <uint64_t> cursor
        journal = NULL
        return capsule
    finally:
        if journal != NULL:
            free(journal)


def bias_environment_attested():
    cdef char reason[512]
    cdef int attested
    reason[0] = 0
    with nogil:
        attested = plan7_bias_environment_attested(reason, sizeof(reason))
    return bool(attested), reason.decode("utf-8", "replace")


def _sealed_journal_transport_statistics():
    """Return implementation-only fused transport allocation counters."""
    return {
        "build_count": _sealed_journal_build_count,
        "payload_bytes": _sealed_journal_payload_bytes,
        "duplicate_python_bytes": _sealed_journal_duplicate_python_bytes,
    }


def bias_host_environment_attested():
    cdef int attested
    with nogil:
        attested = plan7_bias_host_environment_attested()
    return bool(attested)


def pack_bias_profile_raw(
    const float[::1] background,
    const float[::1] composition,
    int model_length,
    float scale,
    int cutoff_mode,
    float cutoff_bit_score,
):
    cdef char error[512]
    cdef int status
    cdef plan7_bias_profile profile
    cdef bytearray output
    cdef uint8_t[::1] output_view
    if background.shape[0] != 20 or composition.shape[0] != 20:
        raise ValueError("bias background and composition must contain 20 values")
    error[0] = 0
    with nogil:
        status = plan7_bias_pack_amino_profile(
            &background[0],
            &composition[0],
            model_length,
            scale,
            cutoff_mode,
            cutoff_bit_score,
            &profile,
            error,
            sizeof(error),
        )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    output = bytearray(sizeof(plan7_bias_profile))
    output_view = output
    memcpy(&output_view[0], &profile, sizeof(plan7_bias_profile))
    return bytes(output)


def bias_filter_score_host_raw(
    const uint8_t[::1] packed_profile,
    const uint8_t[::1] residues,
):
    cdef plan7_bias_profile profile
    cdef float score
    cdef float_bits score_bits
    cdef int status
    if packed_profile.shape[0] != sizeof(plan7_bias_profile):
        raise ValueError("packed bias profile has the wrong size")
    if residues.shape[0] == 0:
        raise ValueError("bias filter does not score empty targets")
    memcpy(&profile, &packed_profile[0], sizeof(plan7_bias_profile))
    with nogil:
        status = plan7_bias_filter_score_host(
            &profile,
            &residues[0],
            <uint64_t> residues.shape[0],
            &score,
        )
    if status != 0:
        raise ValueError("invalid bias profile or target")
    score_bits.value = score
    return score_bits.bits


def bias_rebias_decision_raw(
    int ssv_status,
    int ssv_numerator,
    float scale,
    float filtersc,
    int cutoff_mode,
    float cutoff_bit_score,
):
    cdef float bit_score
    cdef float_bits encoded
    cdef int action
    if not 0 <= ssv_status <= 255:
        raise ValueError("SSV status must fit in uint8")
    if not -32768 <= ssv_numerator <= 32767:
        raise ValueError("SSV numerator must fit in int16")
    action = plan7_bias_rebias_decision(
        <uint8_t> ssv_status,
        <int16_t> ssv_numerator,
        scale,
        filtersc,
        cutoff_mode,
        cutoff_bit_score,
        &bit_score,
    )
    if isfinite(bit_score):
        encoded.value = bit_score
        return action, encoded.bits
    return action, None


cdef list _format_results(vector[plan7_ssv_result]& results, float scale):
    cdef size_t i
    cdef plan7_ssv_result result
    cdef float_bits score
    cdef object score_value
    cdef list output = []

    for i in range(results.size()):
        result = results[i]
        if result.status == PLAN7_SSV_OK:
            score.value = <float> result.numerator
            score.value /= scale
            score.value -= 3.0
            score_value = score.bits
        else:
            score_value = None
        output.append(
            (result.status, result.xE, result.tjb, result.numerator, score_value)
        )
    return output


cdef list _format_many_results(
    vector[plan7_ssv_result]& results,
    const float[::1] scales,
    size_t sequence_count,
):
    cdef size_t profile_index
    cdef size_t sequence_index
    cdef size_t result_index
    cdef plan7_ssv_result result
    cdef float_bits score
    cdef object score_value
    cdef list profile_results
    cdef list output = []

    for profile_index in range(<size_t> scales.shape[0]):
        profile_results = []
        for sequence_index in range(sequence_count):
            result_index = profile_index * sequence_count + sequence_index
            result = results[result_index]
            if result.status == PLAN7_SSV_OK:
                score.value = <float> result.numerator
                score.value /= scales[profile_index]
                score.value -= 3.0
                score_value = score.bits
            else:
                score_value = None
            profile_results.append(
                (result.status, result.xE, result.tjb, result.numerator, score_value)
            )
        output.append(profile_results)
    return output


def device_count():
    cdef char error[512]
    cdef int count
    error[0] = 0
    with nogil:
        count = plan7_cuda_device_count(error, sizeof(error))
    if count < 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    return count


def device_memory_info():
    cdef char error[512]
    cdef int device_ordinal = -1
    cdef uint64_t free_bytes = 0
    cdef uint64_t total_bytes = 0
    cdef int status
    error[0] = 0
    with nogil:
        status = plan7_cuda_memory_info(
            &device_ordinal, &free_bytes, &total_bytes, error, sizeof(error)
        )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    return {
        "device_ordinal": device_ordinal,
        "cuda_free_bytes": free_bytes,
        "cuda_total_bytes": total_bytes,
    }


def _validate_device_ordinal(int owner_device, int current_device):
    """Exercise the pure owner/current-device validation seam."""
    cdef char error[512]
    cdef int status
    error[0] = 0
    status = plan7_validate_device_ordinal(
        owner_device, current_device, error, sizeof(error)
    )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))


def profile_footprint(model_lengths):
    """Return amino profile byte bounds without packing or allocating."""
    cdef object lengths_array
    cdef const uint32_t[::1] lengths
    cdef plan7_profile_footprint footprint
    cdef char error[512]
    cdef int status
    try:
        lengths_array = _array.array("I", model_lengths)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("model lengths must be unsigned 32-bit integers") from exc
    lengths = lengths_array
    error[0] = 0
    with nogil:
        status = plan7_profile_footprint_compute(
            &lengths[0] if lengths.shape[0] else NULL,
            <size_t> lengths.shape[0],
            &footprint,
            error,
            sizeof(error),
        )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    return {
        "profile_count": footprint.profile_count,
        "ssv_device_bytes": footprint.ssv_device_bytes,
        "viterbi_device_bytes": footprint.viterbi_device_bytes,
        "viterbi_exact_rbv_upper_bytes": (
            footprint.viterbi_exact_rbv_upper_bytes
        ),
        "forward_device_bytes": footprint.forward_device_bytes,
        "bias_device_bytes": footprint.bias_device_bytes,
        "minimum_device_bytes": footprint.minimum_device_bytes,
        "maximum_device_bytes": footprint.maximum_device_bytes,
    }


def profile_slice_cell_count(
    uint64_t profile_count,
    uint64_t target_count,
    uint64_t cell_limit=100_000_000,
):
    """Validate and return the profile-major planner cell count."""
    cdef uint64_t cell_count = 0
    cdef char error[512]
    cdef int status
    error[0] = 0
    status = plan7_profile_slice_cell_count(
        profile_count,
        target_count,
        cell_limit,
        &cell_count,
        error,
        sizeof(error),
    )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    return cell_count


def simulate_allocate_before_free(
    current_capacities,
    required_capacities,
    uint64_t free_bytes,
):
    """Simulate ordered high-water growth by allocate-new, then free-old."""
    cdef object current_array
    cdef object required_array
    cdef carray final_array
    cdef const uint64_t[::1] current
    cdef const uint64_t[::1] required
    cdef uint64_t[::1] final
    cdef plan7_allocation_simulation simulation
    cdef char error[512]
    cdef int status
    try:
        current_array = _array.array("Q", current_capacities)
        required_array = _array.array("Q", required_capacities)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("capacities must be unsigned 64-bit integers") from exc
    current = current_array
    required = required_array
    if current.shape[0] != required.shape[0]:
        raise ValueError("current and required capacities differ in length")
    final_array = clone(
        _UINT64_ARRAY_TEMPLATE, <Py_ssize_t> current.shape[0], zero=False
    )
    final = final_array
    error[0] = 0
    with nogil:
        status = plan7_simulate_allocate_before_free(
            &current[0] if current.shape[0] else NULL,
            &required[0] if required.shape[0] else NULL,
            <size_t> current.shape[0],
            free_bytes,
            &final[0] if final.shape[0] else NULL,
            &simulation,
            error,
            sizeof(error),
        )
    if status != 0:
        raise ValueError(error.decode("utf-8", "replace"))
    return {
        "fits": bool(simulation.fits),
        "peak_additional_bytes": simulation.peak_additional_bytes,
        "final_additional_bytes": simulation.final_additional_bytes,
        "final_free_bytes": (
            simulation.final_free_bytes if simulation.fits else None
        ),
        "growth_count": simulation.growth_count,
        "first_unfit_index": (
            None if simulation.fits else simulation.first_unfit_index
        ),
        "capacities": tuple(final_array),
    }


def tjb_for_lengths(float scale, const uint64_t[::1] lengths):
    cdef bytearray output = bytearray(lengths.shape[0])
    cdef uint8_t[::1] view = output
    cdef size_t i
    cdef int value
    if not isfinite(scale) or scale <= 0:
        raise ValueError("profile scale must be finite and positive")
    for i in range(<size_t> lengths.shape[0]):
        if lengths[i] > 100_000:
            raise ValueError("sequence length exceeds HMMER's protein limit")
    with nogil:
        for i in range(<size_t> lengths.shape[0]):
            value = plan7_tjb_for_length(scale, lengths[i])
            view[i] = <uint8_t> value
    return output


def pack_striped_scores(
    list striped_score_buffers,
    const int32_t[::1] score_strides,
    const int32_t[::1] model_lengths,
    int alphabet_size,
):
    """Transpose striped HMMER scores into compact ``[k][residue]`` rows."""
    cdef size_t profile_count = <size_t> len(striped_score_buffers)
    cdef size_t profile_index
    cdef size_t model_length
    cdef size_t profile_score_count
    cdef size_t total_score_count = 0
    cdef size_t output_offset = 0
    cdef size_t k
    cdef int residue
    cdef int q_count
    cdef int column
    cdef int score_stride
    cdef const uint8_t[::1] striped_scores
    cdef bytearray packed_scores
    cdef uint8_t[::1] packed_view

    if alphabet_size < 1:
        raise ValueError("alphabet size must be positive")
    if (
        <size_t> score_strides.shape[0] != profile_count
        or <size_t> model_lengths.shape[0] != profile_count
    ):
        raise ValueError("profile score metadata lengths differ")

    for profile_index in range(profile_count):
        if not 1 <= model_lengths[profile_index] <= 100_000:
            raise ValueError("invalid model length")
        model_length = <size_t> model_lengths[profile_index]
        if model_length > (<size_t> -1) // <size_t> alphabet_size:
            raise OverflowError("compact profile score count overflows size_t")
        profile_score_count = model_length * <size_t> alphabet_size
        if profile_score_count > (<size_t> -1) - total_score_count:
            raise OverflowError("packed profile score count overflows size_t")
        total_score_count += profile_score_count

    packed_scores = bytearray(total_score_count)
    packed_view = packed_scores
    for profile_index in range(profile_count):
        striped_scores = striped_score_buffers[profile_index]
        score_stride = score_strides[profile_index]
        model_length = <size_t> model_lengths[profile_index]
        q_count = max(2, (model_lengths[profile_index] + 15) // 16)
        if score_stride < 16 * (q_count + 17):
            raise ValueError("striped score stride is too short")
        if (
            <size_t> score_stride > (<size_t> -1) // <size_t> alphabet_size
            or <size_t> striped_scores.shape[0]
            != <size_t> score_stride * <size_t> alphabet_size
        ):
            raise ValueError("striped score buffer has the wrong size")
        with nogil:
            for k in range(model_length):
                column = 16 * (<int> k % q_count) + <int> k // q_count
                for residue in range(alphabet_size):
                    packed_view[
                        output_offset + k * <size_t> alphabet_size + residue
                    ] = striped_scores[
                        <size_t> residue * <size_t> score_stride + column
                    ]
        output_offset += model_length * <size_t> alphabet_size
    return packed_scores


cdef class ForwardProvenance:
    """Opaque identity binding gathered Forward rows to resident inputs."""

    cdef plan7_forward_provenance _value

    def __cinit__(self):
        self._value.database_generation = 0
        self._value.batch_generation = 0
        self._value.row_hash = 0
        self._value.special_hash = 0
        self._value.continuation_hash = 0
        self._value.pass_count = 0
        self._value.special_count = 0
        self._value.generation_f3_bits = 0
        self._value.integrity_tag = 0

    @property
    def pass_count(self):
        return self._value.pass_count

    @property
    def special_count(self):
        return self._value.special_count

    def __reduce__(self):
        raise TypeError("Forward provenance tokens cannot be serialized")

    def _tampered_for_test(self, field):
        """Return a corrupt copy for native provenance rejection tests."""
        cdef ForwardProvenance token
        token = ForwardProvenance.__new__(ForwardProvenance)
        token._value = self._value
        if field == "continuation_hash":
            token._value.continuation_hash ^= 1
        elif field == "generation_f3_bits":
            token._value.generation_f3_bits ^= 1
        elif field == "integrity_tag":
            token._value.integrity_tag ^= 1
        else:
            raise ValueError("unknown provenance field")
        return token


cdef ForwardProvenance _forward_provenance_from_output(
    const plan7_forward_output *output,
):
    cdef const plan7_forward_provenance *native
    cdef ForwardProvenance token
    native = plan7_forward_output_provenance(output)
    if native == NULL:
        raise RuntimeError("Forward provenance is null")
    token = ForwardProvenance.__new__(ForwardProvenance)
    token._value = native[0]
    return token


cdef class BackwardDomainProvenance:
    """Opaque seal for one compact Backward/domain continuation journal."""

    cdef plan7_backward_domain_provenance _value

    def __cinit__(self):
        self._value.candidate_count = 0
        self._value.region_count = 0

    @property
    def candidate_count(self):
        return self._value.candidate_count

    @property
    def region_count(self):
        return self._value.region_count

    def __reduce__(self):
        raise TypeError("Backward/domain provenance tokens cannot be serialized")


cdef BackwardDomainProvenance _backward_provenance_from_output(
    const plan7_backward_domain_output *output,
):
    cdef const plan7_backward_domain_provenance *native
    cdef BackwardDomainProvenance token
    native = plan7_backward_domain_output_provenance(output)
    if native == NULL:
        raise RuntimeError("Backward/domain provenance is null")
    token = BackwardDomainProvenance.__new__(BackwardDomainProvenance)
    token._value = native[0]
    return token


cdef object _backward_domain_route_payload_from_output(
    const plan7_backward_domain_output *output,
):
    """Copy the compact route/region journal for diagnostic audits only."""
    cdef size_t result_count
    cdef size_t result_bytes
    cdef size_t offset_bytes
    cdef size_t region_count
    cdef size_t region_bytes
    cdef bytes result_storage
    cdef bytes offset_storage
    cdef bytes region_storage
    cdef const plan7_backward_domain_result *results
    cdef const uint64_t *offsets
    cdef const plan7_simple_region *regions

    if output == NULL:
        raise RuntimeError("Backward/domain route output is null")
    result_count = plan7_backward_domain_output_result_count(output)
    region_count = plan7_backward_domain_output_region_count(output)
    if result_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_backward_domain_result)
    ):
        raise OverflowError("Backward/domain route results exceed Python limits")
    if result_count > (<size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)) - 1:
        raise OverflowError("Backward/domain route offsets exceed Python limits")
    if region_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_simple_region)
    ):
        raise OverflowError("Backward/domain route regions exceed Python limits")
    result_bytes = result_count * sizeof(plan7_backward_domain_result)
    offset_bytes = (result_count + 1) * sizeof(uint64_t)
    region_bytes = region_count * sizeof(plan7_simple_region)
    results = plan7_backward_domain_output_results(output)
    offsets = plan7_backward_domain_output_region_offsets(output)
    regions = plan7_backward_domain_output_regions(output)
    if (
        offsets == NULL
        or (result_count and results == NULL)
        or (region_count and regions == NULL)
    ):
        raise RuntimeError("Backward/domain route storage is incomplete")
    result_storage = PyBytes_FromStringAndSize(NULL, result_bytes)
    if result_bytes:
        memcpy(PyBytes_AS_STRING(result_storage), results, result_bytes)
    offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
    memcpy(PyBytes_AS_STRING(offset_storage), offsets, offset_bytes)
    region_storage = PyBytes_FromStringAndSize(NULL, region_bytes)
    if region_bytes:
        memcpy(PyBytes_AS_STRING(region_storage), regions, region_bytes)
    return (
        result_storage,
        memoryview(offset_storage).cast("Q"),
        memoryview(region_storage).cast("I"),
    )


cdef object _domain_rescore_payload_from_output(
    const plan7_domain_rescore_output *output,
):
    cdef size_t result_count
    cdef size_t result_bytes
    cdef size_t offset_bytes
    cdef size_t trace_count
    cdef size_t trace_bytes
    cdef size_t null2_count
    cdef size_t null2_bytes
    cdef bytes result_storage
    cdef bytes offset_storage
    cdef bytes trace_storage
    cdef bytes null2_storage
    cdef const plan7_domain_rescore_result *results
    cdef const uint64_t *offsets
    cdef const plan7_domain_rescore_trace_step *traces
    cdef const float *null2
    cdef const plan7_domain_rescore_statistics *statistics
    cdef const plan7_domain_rescore_provenance *provenance

    if output == NULL:
        raise RuntimeError("isolated-domain output is null")
    result_count = plan7_domain_rescore_output_result_count(output)
    trace_count = plan7_domain_rescore_output_trace_count(output)
    null2_count = plan7_domain_rescore_output_null2_count(output)
    if result_count > (<size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_rescore_result)):
        raise OverflowError("isolated-domain results exceed Python limits")
    if trace_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_rescore_trace_step)
    ):
        raise OverflowError("isolated-domain traces exceed Python limits")
    if null2_count > (<size_t> PY_SSIZE_T_MAX // sizeof(float)):
        raise OverflowError("isolated-domain null2 output exceeds Python limits")
    if result_count > (
        <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
    ) - 1:
        raise OverflowError("isolated-domain offsets exceed Python limits")
    result_bytes = result_count * sizeof(plan7_domain_rescore_result)
    offset_bytes = (result_count + 1) * sizeof(uint64_t)
    trace_bytes = trace_count * sizeof(plan7_domain_rescore_trace_step)
    null2_bytes = null2_count * sizeof(float)
    results = plan7_domain_rescore_output_results(output)
    offsets = plan7_domain_rescore_output_trace_offsets(output)
    traces = plan7_domain_rescore_output_traces(output)
    null2 = plan7_domain_rescore_output_null2(output)
    statistics = plan7_domain_rescore_output_statistics(output)
    provenance = plan7_domain_rescore_output_provenance(output)
    if (
        offsets == NULL
        or statistics == NULL
        or provenance == NULL
        or (result_count and results == NULL)
        or (trace_count and traces == NULL)
        or (null2_count and null2 == NULL)
    ):
        raise RuntimeError("isolated-domain output storage is incomplete")
    result_storage = PyBytes_FromStringAndSize(NULL, result_bytes)
    if result_bytes:
        memcpy(PyBytes_AS_STRING(result_storage), results, result_bytes)
    offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
    memcpy(PyBytes_AS_STRING(offset_storage), offsets, offset_bytes)
    trace_storage = PyBytes_FromStringAndSize(NULL, trace_bytes)
    if trace_bytes:
        memcpy(PyBytes_AS_STRING(trace_storage), traces, trace_bytes)
    null2_storage = PyBytes_FromStringAndSize(NULL, null2_bytes)
    if null2_bytes:
        memcpy(PyBytes_AS_STRING(null2_storage), null2, null2_bytes)
    return (
        result_storage,
        memoryview(offset_storage).cast("Q"),
        trace_storage,
        memoryview(null2_storage).cast("f"),
        {
            "upstream_row_count": statistics.upstream_row_count,
            "simple_row_count": statistics.simple_row_count,
            "region_count": statistics.region_count,
            "device_result_count": statistics.device_result_count,
            "cpu_required_count": statistics.cpu_required_count,
            "numeric_fallback_count": statistics.numeric_fallback_count,
            "cap_fallback_count": statistics.cap_fallback_count,
            "global_cpu_fallback_count": (
                statistics.global_cpu_fallback_count
            ),
            "work_cells": statistics.work_cells,
            "forward_matrix_bytes": statistics.forward_matrix_bytes,
            "posterior_matrix_bytes": statistics.posterior_matrix_bytes,
            "special_workspace_bytes": statistics.special_workspace_bytes,
            "trace_workspace_bytes": statistics.trace_workspace_bytes,
            "compact_output_byte_limit": (
                statistics.compact_output_byte_limit
            ),
            "compact_output_bytes": statistics.compact_output_bytes,
            "kernel_ms": statistics.kernel_milliseconds,
            "upload_ms": statistics.upload_milliseconds,
            "download_ms": statistics.download_milliseconds,
            "total_ms": statistics.total_milliseconds,
            "result_hash": provenance.result_hash,
            "trace_hash": provenance.trace_hash,
            "null2_hash": provenance.null2_hash,
            "result_count": provenance.result_count,
            "trace_count": provenance.trace_count,
            "null2_count": provenance.null2_count,
        },
    )


def backward_domain_cpu_oracle_raw(
    OptimizedProfile profile,
    const uint8_t[::1] residues,
    const float[::1] forward_specials,
    float rt1=0.25,
    float rt2=0.10,
    float rt3=0.20,
    float guard_band=2.0e-4,
):
    """Run pristine HMMER BackwardParser + DomainDecoding for one row."""
    cdef plan7_backward_domain_result result
    cdef size_t posterior_count
    cdef size_t posterior_bytes
    cdef bytes result_storage
    cdef bytes posterior_storage
    cdef plan7_domain_posterior *posterior_pointer
    cdef size_t region_capacity
    cdef size_t region_count = 0
    cdef size_t region_bytes
    cdef bytes region_storage
    cdef plan7_simple_region *region_pointer
    cdef char error[512]
    cdef int status

    if residues.shape[0] == 0:
        raise ValueError("Backward/domain CPU oracle requires a nonempty target")
    if residues.shape[0] > (<size_t> -1) - 1:
        raise OverflowError("Backward/domain posterior length overflows size_t")
    posterior_count = <size_t> residues.shape[0] + 1
    if posterior_count > (<size_t> -1) // sizeof(plan7_domain_posterior):
        raise OverflowError("Backward/domain posterior size overflows size_t")
    posterior_bytes = posterior_count * sizeof(plan7_domain_posterior)
    if posterior_bytes > <size_t> PY_SSIZE_T_MAX:
        raise OverflowError("Backward/domain posterior exceeds Python limits")
    posterior_storage = PyBytes_FromStringAndSize(NULL, posterior_bytes)
    posterior_pointer = <plan7_domain_posterior *> PyBytes_AS_STRING(
        posterior_storage
    )
    region_capacity = <size_t> residues.shape[0]
    if region_capacity > (<size_t> PY_SSIZE_T_MAX // sizeof(plan7_simple_region)):
        raise OverflowError("Backward/domain region output exceeds Python limits")
    region_bytes = region_capacity * sizeof(plan7_simple_region)
    region_storage = PyBytes_FromStringAndSize(NULL, region_bytes)
    region_pointer = <plan7_simple_region *> PyBytes_AS_STRING(region_storage)
    error[0] = 0
    with nogil:
        status = plan7_backward_domain_cpu_oracle(
            <uintptr_t> profile._om,
            &residues[0],
            <size_t> residues.shape[0],
            &forward_specials[0] if forward_specials.shape[0] else NULL,
            <size_t> forward_specials.shape[0],
            rt1,
            rt2,
            rt3,
            guard_band,
            &result,
            posterior_pointer,
            posterior_count,
            region_pointer if region_capacity else NULL,
            region_capacity,
            &region_count,
            error,
            sizeof(error),
        )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    result_storage = PyBytes_FromStringAndSize(
        <const char *> &result, sizeof(plan7_backward_domain_result)
    )
    if region_count > region_capacity:
        raise RuntimeError("Backward/domain oracle region count exceeds capacity")
    region_storage = region_storage[: region_count * sizeof(plan7_simple_region)]
    return (
        result_storage,
        memoryview(posterior_storage).cast("f"),
        memoryview(region_storage).cast("I"),
    )


def domain_rescore_cpu_oracle_raw(
    OptimizedProfile profile,
    const uint8_t[::1] residues,
    uint32_t envelope_begin,
    uint32_t envelope_end,
):
    """Run pristine HMMER's complete isolated-envelope rescore path."""
    cdef plan7_domain_rescore_result result
    cdef size_t trace_capacity
    cdef size_t trace_count = 0
    cdef size_t trace_bytes
    cdef bytes result_storage
    cdef bytes null2_storage
    cdef bytes trace_storage
    cdef float *null2_pointer
    cdef plan7_domain_rescore_trace_step *trace_pointer
    cdef char error[512]
    cdef int status

    if residues.shape[0] == 0:
        raise ValueError("isolated-domain CPU oracle requires a nonempty target")
    if envelope_begin == 0 or envelope_begin > envelope_end:
        raise ValueError("isolated-domain envelope is invalid")
    if envelope_end > <uint32_t> residues.shape[0]:
        raise ValueError("isolated-domain envelope exceeds the target")
    if sizeof(plan7_domain_rescore_result) != PLAN7_DOMAIN_RESCORE_RECORD_SIZE:
        raise RuntimeError("isolated-domain result ABI size mismatch")
    if sizeof(plan7_domain_rescore_trace_step) != (
        PLAN7_DOMAIN_RESCORE_TRACE_STEP_SIZE
    ):
        raise RuntimeError("isolated-domain trace ABI size mismatch")
    trace_capacity = (
        <size_t> (envelope_end - envelope_begin + 1)
        + <size_t> profile.M
        + 16
    )
    if trace_capacity > (
        <size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_rescore_trace_step)
    ):
        raise OverflowError("isolated-domain trace exceeds Python limits")
    trace_bytes = trace_capacity * sizeof(plan7_domain_rescore_trace_step)
    trace_storage = PyBytes_FromStringAndSize(NULL, trace_bytes)
    trace_pointer = <plan7_domain_rescore_trace_step *> PyBytes_AS_STRING(
        trace_storage
    )
    null2_storage = PyBytes_FromStringAndSize(
        NULL, PLAN7_DOMAIN_RESCORE_NULL2_COUNT * sizeof(float)
    )
    null2_pointer = <float *> PyBytes_AS_STRING(null2_storage)
    error[0] = 0
    with nogil:
        status = plan7_domain_rescore_cpu_oracle(
            <uintptr_t> profile._om,
            &residues[0],
            <size_t> residues.shape[0],
            envelope_begin,
            envelope_end,
            &result,
            null2_pointer,
            PLAN7_DOMAIN_RESCORE_NULL2_COUNT,
            trace_pointer,
            trace_capacity,
            &trace_count,
            error,
            sizeof(error),
        )
    if status != 0:
        message = error.decode("utf-8", "replace")
        raise RuntimeError(message or f"isolated-domain oracle status {status}")
    if trace_count > trace_capacity:
        raise RuntimeError("isolated-domain oracle trace count changed")
    result_storage = PyBytes_FromStringAndSize(
        <const char *> &result, sizeof(plan7_domain_rescore_result)
    )
    trace_storage = trace_storage[
        : trace_count * sizeof(plan7_domain_rescore_trace_step)
    ]
    return (
        result_storage,
        memoryview(null2_storage).cast("f"),
        trace_storage,
    )


def domain_rescore_own_scale_required_for_test(float xB):
    """Exercise the exact stock binary32/double scaling boundary."""
    return bool(c_plan7_domain_rescore_own_scale_required_for_test(xB))


def domain_rescore_oatrace_j_predecessor_for_test(
    float jpath,
    float epath,
    bint j_loop_enabled,
    bint e_loop_enabled,
):
    """Return whether stock OATrace's strict comparison chooses J."""
    return bool(c_plan7_domain_rescore_oatrace_j_predecessor_for_test(
        jpath,
        epath,
        j_loop_enabled,
        e_loop_enabled,
    ))


cdef class ProfileSelection:
    """Immutable pointer-free host pack for an ordered profile slice.

    Operations and ``close`` must be serialized by the public adapter.
    """

    cdef plan7_profile_selection *_selection
    cdef object _owner
    cdef tuple _fingerprints

    def __cinit__(self):
        self._selection = NULL
        self._owner = None
        self._fingerprints = ()

    def __dealloc__(self):
        if self._selection != NULL:
            plan7_profile_selection_destroy(&self._selection, NULL, 0)

    cdef plan7_profile_selection_view _view(self) except *:
        cdef plan7_profile_selection_view view
        cdef char error[512]
        cdef int status
        if self._selection == NULL:
            raise RuntimeError("profile selection is closed")
        error[0] = 0
        status = plan7_profile_selection_get_view(
            self._selection, &view, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return view

    def __len__(self):
        cdef plan7_profile_selection_view view = self._view()
        return view.profile_count

    @property
    def closed(self):
        return self._selection == NULL

    @property
    def identity(self):
        cdef plan7_profile_selection_view view = self._view()
        return view.session_id, view.selection_id

    def _identity_tokens_for_seal(self):
        """Return an immutable adapter-only identity snapshot."""
        cdef plan7_profile_selection_view view = self._view()
        cdef size_t byte_count
        if sizeof(uintptr_t) != sizeof(uint64_t):
            raise RuntimeError("profile identity tokens require a 64-bit process")
        if view.profile_count and view.identity_tokens == NULL:
            raise RuntimeError("profile selection identity storage is null")
        if view.profile_count > (<size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)):
            raise OverflowError("profile identity snapshot exceeds Python limits")
        byte_count = view.profile_count * sizeof(uint64_t)
        return PyBytes_FromStringAndSize(
            <const char *> view.identity_tokens, <Py_ssize_t> byte_count
        )

    def _fingerprints_for_seal(self):
        """Return adapter-only ordered immutable profile fingerprints."""
        return b"".join(self._fingerprints)

    @property
    def host_bytes(self):
        cdef plan7_profile_selection_view view = self._view()
        return view.host_bytes

    def close(self):
        cdef plan7_profile_selection *selection = NULL
        cdef char error[512]
        cdef int status = 0
        if self._selection != NULL:
            selection = self._selection
            self._selection = NULL
            self._owner = None
            error[0] = 0
            with nogil:
                status = plan7_profile_selection_destroy(
                    &selection, error, sizeof(error)
                )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))


cdef class ProfileSession:
    """Host-owned immutable SSV, bias, Viterbi, and Forward snapshots.

    Session operations and ``close`` must be serialized by the public adapter.
    Construction does not create a CUDA context or allocate device memory.
    """

    cdef plan7_profile_session *_session
    cdef tuple _fingerprints

    def __cinit__(self):
        self._session = NULL
        self._fingerprints = ()

    def __init__(
        self,
        profiles,
        const float[::1] background,
        size_t build_worker_count,
        size_t selection_worker_count,
        profile_fingerprints=None,
    ):
        cdef tuple owners = tuple(profiles)
        cdef tuple fingerprints
        cdef vector[uintptr_t] pointers
        cdef OptimizedProfile profile
        cdef object value
        cdef char error[512]
        cdef int status

        if self._session != NULL:
            raise RuntimeError("profile session is already initialized")
        if background.shape[0] != 20:
            raise ValueError("profile session background must have 20 residues")
        if (
            build_worker_count > <size_t> len(owners)
            or selection_worker_count > <size_t> len(owners)
        ):
            raise ValueError("profile session worker count exceeds profile count")
        pointers.reserve(len(owners))
        for value in owners:
            if not isinstance(value, OptimizedProfile):
                raise TypeError("profile sessions require OptimizedProfile objects")
            profile = value
            pointers.push_back(<uintptr_t> profile._om)
        error[0] = 0
        with nogil:
            status = plan7_profile_session_create(
                pointers.data() if pointers.size() else NULL,
                pointers.size(),
                &background[0],
                <size_t> background.shape[0],
                build_worker_count,
                selection_worker_count,
                &self._session,
                error,
                sizeof(error),
            )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        try:
            if profile_fingerprints is None:
                fingerprints = tuple(
                    _profile_fingerprint(value) for value in owners
                )
            else:
                fingerprints = tuple(profile_fingerprints)
                if len(fingerprints) != len(owners):
                    raise ValueError(
                        "profile fingerprint count differs from profile count"
                    )
            if any(
                type(value) is not bytes
                or len(value)
                != PLAN7_CONTINUATION_JOURNAL_PROFILE_FINGERPRINT_SIZE
                for value in fingerprints
            ):
                raise RuntimeError(
                    "optimized-profile fingerprint generation failed"
                )
            self._fingerprints = fingerprints
        except:
            plan7_profile_session_destroy(&self._session, NULL, 0)
            raise

    def __dealloc__(self):
        if self._session != NULL:
            plan7_profile_session_destroy(&self._session, NULL, 0)

    def __len__(self):
        return self.statistics["profile_count"]

    @property
    def closed(self):
        return self._session == NULL

    @property
    def statistics(self):
        cdef plan7_profile_session_statistics statistics
        cdef char error[512]
        cdef int status
        if self._session == NULL:
            raise RuntimeError("profile session is closed")
        error[0] = 0
        status = plan7_profile_session_get_statistics(
            self._session, &statistics, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return {
            "session_id": statistics.session_id,
            "profile_count": statistics.profile_count,
            "worker_count": statistics.worker_count,
            "build_worker_count": statistics.build_worker_count,
            "selection_worker_count": statistics.selection_worker_count,
            "selection_count": statistics.selection_count,
            "parallel_run_count": statistics.parallel_run_count,
            "build_parallel_run_count": (
                statistics.build_parallel_run_count
            ),
            "selection_parallel_run_count": (
                statistics.selection_parallel_run_count
            ),
            "host_bytes": statistics.host_bytes,
            "ssv_score_bytes": statistics.ssv_score_bytes,
            "bias_profile_bytes": statistics.bias_profile_bytes,
            "viterbi_descriptor_bytes": statistics.viterbi_descriptor_bytes,
            "viterbi_emission_bytes": statistics.viterbi_emission_bytes,
            "viterbi_transition_bytes": statistics.viterbi_transition_bytes,
            "viterbi_exact_rbv_bytes": statistics.viterbi_exact_rbv_bytes,
            "forward_descriptor_bytes": statistics.forward_descriptor_bytes,
            "forward_emission_bytes": statistics.forward_emission_bytes,
            "forward_transition_bytes": statistics.forward_transition_bytes,
        }

    def _fingerprints_for_seal(self):
        """Return adapter-only immutable database profile fingerprints."""
        if self._session == NULL:
            raise RuntimeError("profile session is closed")
        return b"".join(self._fingerprints)

    def select(self, values):
        cdef tuple requested = tuple(values)
        cdef vector[size_t] indices
        cdef object value
        cdef object indexed
        cdef list selected_fingerprints = []
        cdef size_t index
        cdef ProfileSelection selection
        cdef char error[512]
        cdef int status

        if self._session == NULL:
            raise RuntimeError("profile session is closed")
        indices.reserve(len(requested))
        for value in requested:
            if isinstance(value, bool):
                raise TypeError("profile selection index must not be bool")
            try:
                indexed = value.__index__()
            except (AttributeError, TypeError) as exc:
                raise TypeError("profile selection index must be an integer") from exc
            if indexed < 0:
                raise IndexError("profile selection index is out of range")
            indices.push_back(<size_t> indexed)
        selection = ProfileSelection.__new__(ProfileSelection)
        error[0] = 0
        with nogil:
            status = plan7_profile_session_select(
                self._session,
                indices.data() if indices.size() else NULL,
                indices.size(),
                &selection._selection,
                error,
                sizeof(error),
            )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        selection._owner = self
        for index in range(indices.size()):
            selected_fingerprints.append(self._fingerprints[indices[index]])
        selection._fingerprints = tuple(selected_fingerprints)
        return selection

    def close(self):
        cdef plan7_profile_session *session = NULL
        cdef char error[512]
        cdef int status = 0
        if self._session != NULL:
            session = self._session
            self._session = NULL
            error[0] = 0
            with nogil:
                status = plan7_profile_session_destroy(
                    &session, error, sizeof(error)
                )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))


cdef class ViterbiProfiles:
    """Device-resident exact Viterbi profiles for the raw post-filter API.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_viterbi_database *_database
    cdef tuple _owners

    def __cinit__(self):
        self._database = NULL
        self._owners = ()

    def __init__(self, profiles):
        cdef tuple owners = tuple(profiles)
        cdef vector[uintptr_t] pointers
        cdef OptimizedProfile profile
        cdef object value
        cdef char error[512]
        cdef int status

        if self._database != NULL:
            raise RuntimeError("Viterbi profiles are already initialized")
        pointers.reserve(len(owners))
        for value in owners:
            if not isinstance(value, OptimizedProfile):
                raise TypeError("Viterbi profiles must be OptimizedProfile objects")
            profile = value
            pointers.push_back(<uintptr_t> profile._om)
        error[0] = 0
        status = plan7_viterbi_database_create(
            pointers.data() if pointers.size() else NULL,
            pointers.size(),
            &self._database,
            error,
            sizeof(error),
        )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        self._owners = owners

    def __dealloc__(self):
        if self._database != NULL:
            plan7_viterbi_database_destroy(&self._database, NULL, 0)

    def __len__(self):
        if self._database == NULL:
            return 0
        return plan7_viterbi_database_profile_count(self._database)

    @property
    def closed(self):
        return self._database == NULL

    def __enter__(self):
        if self._database == NULL:
            raise RuntimeError("Viterbi profiles are closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        cdef plan7_viterbi_database *database = NULL
        cdef char error[512]
        cdef int status = 0
        if self._database != NULL:
            database = self._database
            self._database = NULL
            self._owners = ()
            error[0] = 0
            with nogil:
                status = plan7_viterbi_database_destroy(
                    &database, error, sizeof(error)
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))


cdef class ForwardProfiles:
    """Device-resident exact Forward snapshots for F3 and later stages.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_forward_database *_database
    cdef tuple _owners

    def __cinit__(self):
        self._database = NULL
        self._owners = ()

    def __init__(self, profiles):
        cdef tuple owners
        cdef vector[uintptr_t] pointers
        cdef OptimizedProfile profile
        cdef ProfileSelection selection
        cdef object value
        cdef char error[512]
        cdef int status

        if self._database != NULL:
            raise RuntimeError("Forward profiles are already initialized")
        if isinstance(profiles, ProfileSelection):
            selection = profiles
            if selection._selection == NULL:
                raise RuntimeError("profile selection is closed")
            error[0] = 0
            with nogil:
                status = plan7_profile_selection_stage_forward(
                    selection._selection,
                    &self._database,
                    error,
                    sizeof(error),
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            self._owners = (selection,)
            return
        owners = tuple(profiles)
        pointers.reserve(len(owners))
        for value in owners:
            if not isinstance(value, OptimizedProfile):
                raise TypeError("Forward profiles must be OptimizedProfile objects")
            profile = value
            pointers.push_back(<uintptr_t> profile._om)
        error[0] = 0
        # Keep the GIL while native code takes its exact private-array snapshot.
        status = plan7_forward_database_create(
            pointers.data() if pointers.size() else NULL,
            pointers.size(),
            &self._database,
            error,
            sizeof(error),
        )
        if status != 0:
            raise ValueError(error.decode("utf-8", "replace"))
        self._owners = owners

    def __dealloc__(self):
        if self._database != NULL:
            plan7_forward_database_destroy(&self._database, NULL, 0)

    def __len__(self):
        if self._database == NULL:
            return 0
        return plan7_forward_database_profile_count(self._database)

    @property
    def closed(self):
        return self._database == NULL

    @property
    def statistics(self):
        if self._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        return {
            "device_bytes": plan7_forward_database_device_bytes(self._database),
            "pack_ms": plan7_forward_database_pack_milliseconds(self._database),
            "upload_ms": plan7_forward_database_upload_milliseconds(self._database),
        }

    def __enter__(self):
        if self._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        cdef plan7_forward_database *database = NULL
        cdef char error[512]
        cdef int status = 0
        if self._database != NULL:
            database = self._database
            self._database = NULL
            self._owners = ()
            error[0] = 0
            with nogil:
                status = plan7_forward_database_destroy(
                    &database, error, sizeof(error)
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))


cdef class SequenceBatch:
    """Device-resident target batch for raw CUDA operations.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_ssv_sequence_batch *_batch
    cdef vector[plan7_ssv_result] _results
    cdef vector[plan7_ssv_result] _many_results
    cdef vector[plan7_ssv_profile] _profiles
    cdef vector[uint32_t] _candidate_words
    cdef vector[uint32_t] _candidate_indices
    cdef vector[size_t] _candidate_offsets
    cdef vector[size_t] _candidate_counts
    cdef vector[size_t] _bias_candidate_offsets
    cdef vector[plan7_bias_profile] _bias_profiles
    cdef vector[plan7_bias_result] _bias_results
    cdef vector[plan7_postfilter_result] _postfilter_results
    cdef vector[uint64_t] _lengths
    cdef size_t _sequence_count
    cdef int _alphabet_size
    cdef bytes _content_fingerprint

    def __cinit__(
        self,
        const uint8_t[::1] residues,
        const uint64_t[::1] offsets,
        int alphabet_size,
    ):
        cdef char error[512]
        cdef int status
        cdef size_t i

        self._batch = NULL
        self._sequence_count = 0
        self._alphabet_size = alphabet_size
        self._content_fingerprint = b""
        if offsets.shape[0] == 0:
            raise ValueError("offsets must contain an initial zero")
        if alphabet_size < 1:
            raise ValueError("alphabet size must be positive")
        self._content_fingerprint = _sequence_content_fingerprint(
            alphabet_size, residues, offsets
        )
        if len(self._content_fingerprint) != 32:
            raise RuntimeError("sequence content fingerprint generation failed")
        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_create(
                &residues[0] if residues.shape[0] else NULL,
                <size_t> residues.shape[0],
                &offsets[0],
                <size_t> offsets.shape[0],
                alphabet_size,
                &self._batch,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        self._sequence_count = <size_t> offsets.shape[0] - 1
        self._results.resize(self._sequence_count)
        self._lengths.resize(self._sequence_count)
        for i in range(self._sequence_count):
            self._lengths[i] = offsets[i + 1] - offsets[i]

    def __dealloc__(self):
        if self._batch != NULL:
            plan7_ssv_sequence_batch_destroy(&self._batch, NULL, 0)

    def __len__(self):
        return self._sequence_count

    @property
    def closed(self):
        return self._batch == NULL

    def _generation_and_content_for_seal(self):
        """Return adapter-only native batch identity and residue digest."""
        cdef plan7_ssv_sequence_batch_view view
        cdef char error[512]
        cdef int status
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_view(
            self._batch, &view, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return view.generation_id, self._content_fingerprint

    @property
    def workspace_statistics(self):
        """Return post-filter and Forward cache capacities for this batch.

        These values exclude the batch's input, SSV, and bias allocations.
        """
        cdef plan7_ssv_workspace_statistics statistics
        cdef char error[512]
        cdef int status
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_workspace_statistics(
            self._batch, &statistics, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return {
            "postfilter_device_bytes": statistics.postfilter_device_bytes,
            "postfilter_dp_capacity_bytes": (
                statistics.postfilter_dp_capacity_bytes
            ),
            "postfilter_growth_count": statistics.postfilter_growth_count,
            "postfilter_run_count": statistics.postfilter_run_count,
            "forward_device_bytes": statistics.forward_device_bytes,
            "forward_dp_capacity_bytes": statistics.forward_dp_capacity_bytes,
            "forward_xmx_capacity_bytes": statistics.forward_xmx_capacity_bytes,
            "forward_gather_capacity_bytes": (
                statistics.forward_gather_capacity_bytes
            ),
            "forward_growth_count": statistics.forward_growth_count,
            "forward_event_create_count": statistics.forward_event_create_count,
            "forward_run_count": statistics.forward_run_count,
        }

    @property
    def memory_snapshot(self):
        """Return current CUDA availability and persistent device capacities."""
        cdef plan7_ssv_memory_snapshot snapshot
        cdef char error[512]
        cdef int status
        cdef size_t i
        cdef dict capacities = {}
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if len(_DEVICE_CAPACITY_NAMES) != PLAN7_SSV_DEVICE_CAPACITY_COUNT:
            raise RuntimeError("native device capacity ABI changed")
        error[0] = 0
        status = plan7_ssv_sequence_batch_get_memory_snapshot(
            self._batch, &snapshot, error, sizeof(error)
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        for i in range(PLAN7_SSV_DEVICE_CAPACITY_COUNT):
            capacities[_DEVICE_CAPACITY_NAMES[i]] = (
                snapshot.device_capacity_bytes[i]
            )
        return {
            "device_ordinal": snapshot.device_ordinal,
            "cuda_free_bytes": snapshot.cuda_free_bytes,
            "cuda_total_bytes": snapshot.cuda_total_bytes,
            "persistent_device_bytes": snapshot.persistent_device_bytes,
            "capacity_bytes": capacities,
        }

    def __enter__(self):
        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        cdef plan7_ssv_sequence_batch *batch = NULL
        cdef char ssv_error[512]
        cdef int ssv_status = 0
        if self._batch != NULL:
            batch = self._batch
            self._batch = NULL
            ssv_error[0] = 0
            with nogil:
                ssv_status = plan7_ssv_sequence_batch_destroy(
                    &batch, ssv_error, sizeof(ssv_error)
                )
        if ssv_status != 0:
            raise RuntimeError(ssv_error.decode("utf-8", "replace"))

    cdef int _run_filter(
        self,
        const uint8_t[::1] striped_scores,
        int score_stride,
        int model_length,
        int alphabet_size,
        int tbm,
        int tec,
        int base,
        int bias,
        float scale,
    ) except -1:
        cdef char error[512]
        cdef int status

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if score_stride < 1 or alphabet_size < 1 or not 1 <= model_length <= 100_000:
            raise ValueError("invalid score dimensions")
        if alphabet_size != self._alphabet_size:
            raise ValueError("profile and sequence alphabets differ")
        if <size_t> striped_scores.shape[0] < <size_t> score_stride * alphabet_size:
            raise ValueError("striped score buffer is too short")
        if not 0 <= tbm <= 255 or not 0 <= tec <= 255:
            raise ValueError("transition costs must fit in uint8")
        if not 0 <= base <= 255 or not 0 <= bias <= 255:
            raise ValueError("profile byte constants must fit in uint8")
        if not isfinite(scale) or scale <= 0:
            raise ValueError("profile scale must be finite and positive")

        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_filter(
                self._batch,
                &striped_scores[0] if striped_scores.shape[0] else NULL,
                <size_t> striped_scores.shape[0],
                score_stride,
                model_length,
                alphabet_size,
                <uint8_t> tbm,
                <uint8_t> tec,
                <uint8_t> base,
                <uint8_t> bias,
                scale,
                self._results.data() if self._sequence_count else NULL,
                self._sequence_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return 0

    def filter_raw(
        self,
        const uint8_t[::1] striped_scores,
        int score_stride,
        int model_length,
        int alphabet_size,
        int tbm,
        int tec,
        int base,
        int bias,
        float scale,
    ):
        self._run_filter(
            striped_scores,
            score_stride,
            model_length,
            alphabet_size,
            tbm,
            tec,
            base,
            bias,
            scale,
        )
        return _format_results(self._results, scale)

    def cpu_candidates_raw(
        self,
        const uint8_t[::1] striped_scores,
        int score_stride,
        int model_length,
        int alphabet_size,
        int tbm,
        int tec,
        int base,
        int bias,
        float scale,
        float m_mu,
        float m_lambda,
        double f1,
    ):
        cdef size_t i
        cdef int action
        cdef list output = []

        self._run_filter(
            striped_scores,
            score_stride,
            model_length,
            alphabet_size,
            tbm,
            tec,
            base,
            bias,
            scale,
        )
        for i in range(self._sequence_count):
            action = plan7_ssv_f1_decision(
                self._results[i].status,
                self._results[i].numerator,
                self._lengths[i],
                scale,
                m_mu,
                m_lambda,
                f1,
                NULL,
            )
            if action == PLAN7_F1_CPU_REQUIRED:
                output.append(i)
        return output

    cdef size_t _prepare_profiles(
        self,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
    ) except? 0:
        cdef size_t i
        cdef size_t profile_count = <size_t> score_offsets.shape[0]

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if (
            <size_t> score_counts.shape[0] != profile_count
            or <size_t> score_strides.shape[0] != profile_count
            or <size_t> model_lengths.shape[0] != profile_count
            or <size_t> scales.shape[0] != profile_count
            or <size_t> constants.shape[0] != profile_count * 4
        ):
            raise ValueError("profile metadata lengths differ")

        self._profiles.resize(profile_count)
        for i in range(profile_count):
            self._profiles[i].score_offset = score_offsets[i]
            self._profiles[i].score_count = score_counts[i]
            self._profiles[i].score_stride = score_strides[i]
            self._profiles[i].model_length = model_lengths[i]
            self._profiles[i].tbm = constants[4 * i]
            self._profiles[i].tec = constants[4 * i + 1]
            self._profiles[i].base = constants[4 * i + 2]
            self._profiles[i].bias = constants[4 * i + 3]
            self._profiles[i].scale = scales[i]
        return profile_count

    cdef size_t _run_filter_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
    ) except? 0:
        cdef char error[512]
        cdef size_t profile_count
        cdef size_t result_count
        cdef int status

        profile_count = self._prepare_profiles(
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        if self._sequence_count and profile_count > (<size_t> -1) / self._sequence_count:
            raise OverflowError("multi-profile result count overflows size_t")
        result_count = profile_count * self._sequence_count
        self._many_results.resize(result_count)

        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_filter_many(
                self._batch,
                &packed_scores[0] if packed_scores.shape[0] else NULL,
                <size_t> packed_scores.shape[0],
                self._profiles.data() if profile_count else NULL,
                profile_count,
                self._many_results.data() if result_count else NULL,
                result_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return profile_count

    def filter_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
    ):
        self._run_filter_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        return _format_many_results(self._many_results, scales, self._sequence_count)

    cdef size_t _run_candidates_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
    ) except? 0:
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t words_per_profile
        cdef size_t candidate_word_count
        cdef size_t word_index
        cdef size_t sequence_index
        cdef size_t candidate_count = 0
        cdef size_t output_index
        cdef uint32_t word
        cdef unsigned bit
        cdef int status
        cdef char error[512]

        profile_count = <size_t> score_offsets.shape[0]
        if (
            <size_t> m_mu.shape[0] != profile_count
            or <size_t> m_lambda.shape[0] != profile_count
        ):
            raise ValueError("e-value parameter lengths differ")
        profile_count = self._prepare_profiles(
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        if self._sequence_count > (<size_t> -1) - 31:
            raise OverflowError("candidate mask size overflows size_t")
        words_per_profile = (self._sequence_count + 31) // 32
        if words_per_profile and profile_count > (<size_t> -1) // words_per_profile:
            raise OverflowError("candidate mask size overflows size_t")
        candidate_word_count = profile_count * words_per_profile
        self._candidate_words.resize(candidate_word_count)
        self._candidate_counts.resize(profile_count)
        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_f1_mask_many(
                self._batch,
                &packed_scores[0] if packed_scores.shape[0] else NULL,
                <size_t> packed_scores.shape[0],
                self._profiles.data() if profile_count else NULL,
                profile_count,
                &m_mu[0] if profile_count else NULL,
                &m_lambda[0] if profile_count else NULL,
                f1,
                self._candidate_words.data() if candidate_word_count else NULL,
                candidate_word_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))

        self._candidate_offsets.resize(profile_count)
        for profile_index in range(profile_count):
            self._candidate_counts[profile_index] = 0
            for word_index in range(words_per_profile):
                self._candidate_counts[profile_index] += plan7_popcount_u32(
                    self._candidate_words[
                        profile_index * words_per_profile + word_index
                    ]
                )
            self._candidate_offsets[profile_index] = candidate_count
            if self._candidate_counts[profile_index] > (<size_t> -1) - candidate_count:
                raise OverflowError("candidate count overflows size_t")
            candidate_count += self._candidate_counts[profile_index]

        self._candidate_indices.resize(candidate_count)
        for profile_index in range(profile_count):
            output_index = self._candidate_offsets[profile_index]
            for word_index in range(words_per_profile):
                word = self._candidate_words[
                    profile_index * words_per_profile + word_index
                ]
                while word:
                    bit = plan7_ctz_u32(word)
                    sequence_index = word_index * 32 + bit
                    if sequence_index >= self._sequence_count:
                        raise RuntimeError("candidate mask has trailing bits set")
                    self._candidate_indices[output_index] = <uint32_t> sequence_index
                    output_index += 1
                    word &= word - 1
            if output_index != (
                self._candidate_offsets[profile_index]
                + self._candidate_counts[profile_index]
            ):
                raise RuntimeError("candidate mask count changed")
        return profile_count

    def cpu_candidates_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
    ):
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t sequence_index
        cdef list candidates
        cdef list output = []

        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
        )
        for profile_index in range(profile_count):
            candidates = []
            for sequence_index in range(self._candidate_counts[profile_index]):
                candidates.append(
                    self._candidate_indices[
                        self._candidate_offsets[profile_index] + sequence_index
                    ]
                )
            output.append(candidates)
        return output

    def cpu_candidates_many_csr_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
    ):
        """Return compact candidate rows as native uint32 data and uint64 offsets."""
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef carray indices
        cdef carray offsets

        if _UINT32_ARRAY_TEMPLATE.itemsize != sizeof(uint32_t):
            raise RuntimeError("array('I') is not native uint32")
        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")

        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
        )
        candidate_count = self._candidate_indices.size()
        indices = clone(_UINT32_ARRAY_TEMPLATE, candidate_count, False)
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        if candidate_count:
            memcpy(
                indices.data.as_uints,
                self._candidate_indices.data(),
                candidate_count * sizeof(uint32_t),
            )
        for profile_index in range(profile_count):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._candidate_offsets[profile_index]
            )
        offsets.data.as_ulonglongs[profile_count] = <uint64_t> candidate_count
        return indices, offsets

    cdef size_t _run_bias_candidates_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
    ) except? 0:
        cdef char error[512]
        cdef size_t profile_count
        cdef size_t candidate_count
        cdef size_t profile_index
        cdef size_t expected_bias_bytes
        cdef int status

        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
        )
        if profile_count == 0:
            if packed_bias_profiles.shape[0] != 0:
                raise ValueError("packed bias profiles have trailing bytes")
            self._bias_profiles.clear()
            self._bias_results.clear()
            self._bias_candidate_offsets.resize(1)
            self._bias_candidate_offsets[0] = 0
            return 0
        if profile_count > (<size_t> -1) // sizeof(plan7_bias_profile):
            raise OverflowError("packed bias profile size overflows size_t")
        expected_bias_bytes = profile_count * sizeof(plan7_bias_profile)
        if <size_t> packed_bias_profiles.shape[0] != expected_bias_bytes:
            raise ValueError("packed bias profile buffer has the wrong size")

        self._bias_profiles.resize(profile_count)
        memcpy(
            self._bias_profiles.data(),
            &packed_bias_profiles[0],
            expected_bias_bytes,
        )
        candidate_count = self._candidate_indices.size()
        self._bias_results.resize(candidate_count)
        self._bias_candidate_offsets.resize(profile_count + 1)
        for profile_index in range(profile_count):
            self._bias_candidate_offsets[profile_index] = (
                self._candidate_offsets[profile_index]
            )
        self._bias_candidate_offsets[profile_count] = candidate_count

        if candidate_count == 0:
            return profile_count

        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_bias_candidates_many(
                self._batch,
                self._bias_profiles.data(),
                profile_count,
                self._bias_candidate_offsets.data(),
                self._candidate_indices.data() if candidate_count else NULL,
                candidate_count,
                self._bias_results.data() if candidate_count else NULL,
                candidate_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return profile_count

    def bias_candidates_many_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
    ):
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_index
        cdef plan7_bias_result result
        cdef float_bits score
        cdef object score_bits
        cdef list row
        cdef list output = []

        profile_count = self._run_bias_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            packed_bias_profiles,
        )
        for profile_index in range(profile_count):
            row = []
            for candidate_index in range(
                self._bias_candidate_offsets[profile_index],
                self._bias_candidate_offsets[profile_index + 1],
            ):
                result = self._bias_results[candidate_index]
                if isfinite(result.filtersc):
                    score.value = result.filtersc
                    score_bits = score.bits
                else:
                    score_bits = None
                row.append(
                    (
                        result.sequence_index,
                        result.action,
                        result.ssv_status,
                        result.ssv_numerator,
                        score_bits,
                    )
                )
            output.append(row)
        return output

    def bias_candidates_many_csr_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
    ):
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef size_t result_bytes
        cdef bytearray records
        cdef uint8_t[::1] record_view
        cdef carray offsets

        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        profile_count = self._run_bias_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            packed_bias_profiles,
        )
        candidate_count = self._bias_results.size()
        if candidate_count > (<size_t> -1) // sizeof(plan7_bias_result):
            raise OverflowError("bias result size overflows size_t")
        result_bytes = candidate_count * sizeof(plan7_bias_result)
        records = bytearray(result_bytes)
        if result_bytes:
            record_view = records
            memcpy(&record_view[0], self._bias_results.data(), result_bytes)
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._bias_candidate_offsets[profile_index]
            )
        return records, offsets

    def forward_candidates_many_raw(
        self,
        const uint64_t[::1] candidate_offsets,
        const uint32_t[::1] candidate_indices,
        const float[::1] filter_scores,
        double f3,
        source_profiles,
        ForwardProfiles forward_profiles,
        uint64_t gathered_byte_budget=PLAN7_FORWARD_MAX_GATHERED_BYTES,
    ):
        """Classify F3 and return only passing parser special-state rows."""
        cdef char error[512]
        cdef int status
        cdef size_t profile_count
        cdef size_t candidate_count = <size_t> candidate_indices.shape[0]
        cdef size_t result_count
        cdef size_t result_bytes
        cdef size_t offset_bytes
        cdef size_t special_count
        cdef size_t special_bytes
        cdef size_t profile_index
        cdef bint sealed_source = source_profiles is None
        cdef tuple owners = () if sealed_source else tuple(source_profiles)
        cdef vector[uintptr_t] source_pointers
        cdef OptimizedProfile source_profile
        cdef object value
        cdef plan7_forward_output *output = NULL
        cdef const plan7_forward_result *native_results
        cdef const uint64_t *native_offsets
        cdef const float *native_specials
        cdef const plan7_forward_statistics *native_statistics
        cdef bytes records
        cdef bytes offset_storage
        cdef bytes special_storage
        cdef object offsets
        cdef object specials
        cdef dict statistics
        cdef ForwardProvenance provenance

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if forward_profiles._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        if candidate_offsets.shape[0] == 0:
            raise ValueError("candidate offsets must contain an initial zero")
        profile_count = <size_t> candidate_offsets.shape[0] - 1
        if <size_t> filter_scores.shape[0] != candidate_count:
            raise ValueError("Forward candidate score lengths differ")
        if not sealed_source and len(owners) != profile_count:
            raise ValueError("source and Forward profile counts differ")
        if plan7_forward_database_profile_count(
            forward_profiles._database
        ) != profile_count:
            raise ValueError("Forward profile and candidate row counts differ")
        if not sealed_source:
            source_pointers.reserve(profile_count)
            for profile_index in range(profile_count):
                value = owners[profile_index]
                if not isinstance(value, OptimizedProfile):
                    raise TypeError(
                        "source profiles must be OptimizedProfile objects"
                    )
                if value is not forward_profiles._owners[profile_index]:
                    raise ValueError(
                        "source profile identity differs from Forward profile row"
                    )
                source_profile = value
                source_pointers.push_back(<uintptr_t> source_profile._om)
        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if _FLOAT_ARRAY_TEMPLATE.itemsize != sizeof(float):
            raise RuntimeError("array('f') is not native float32")
        if sizeof(plan7_forward_result) != PLAN7_FORWARD_RECORD_SIZE:
            raise RuntimeError("Forward result ABI size mismatch")

        error[0] = 0
        # Keep the GIL while native code validates live private profile arrays.
        status = plan7_forward_run_batch_workspace(
            forward_profiles._database,
            self._batch,
            source_pointers.data() if source_pointers.size() else NULL,
            profile_count,
            &candidate_offsets[0],
            &candidate_indices[0] if candidate_count else NULL,
            &filter_scores[0] if candidate_count else NULL,
            candidate_count,
            f3,
            gathered_byte_budget,
            &output,
            error,
            sizeof(error),
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        try:
            result_count = plan7_forward_output_result_count(output)
            if result_count != candidate_count:
                raise RuntimeError("Forward result count changed")
            if result_count > (<size_t> -1) // sizeof(plan7_forward_result):
                raise OverflowError("Forward result size overflows size_t")
            result_bytes = result_count * sizeof(plan7_forward_result)
            if result_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("Forward result size exceeds Python limits")
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            native_results = plan7_forward_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Forward result storage is null")
                memcpy(PyBytes_AS_STRING(records), native_results, result_bytes)

            native_offsets = plan7_forward_output_special_offsets(output)
            if native_offsets == NULL:
                raise RuntimeError("Forward special offsets are null")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
            ) - 1:
                raise OverflowError("Forward special offsets exceed Python limits")
            offset_bytes = (result_count + 1) * sizeof(uint64_t)
            offset_storage = PyBytes_FromStringAndSize(
                NULL,
                offset_bytes,
            )
            memcpy(
                PyBytes_AS_STRING(offset_storage),
                native_offsets,
                offset_bytes,
            )
            offsets = memoryview(offset_storage).cast("Q")

            special_count = plan7_forward_output_special_count(output)
            if special_count > (<size_t> -1) // sizeof(float):
                raise OverflowError("Forward special matrix size overflows size_t")
            special_bytes = special_count * sizeof(float)
            if special_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError(
                    "Forward special matrix size exceeds Python limits"
                )
            special_storage = PyBytes_FromStringAndSize(NULL, special_bytes)
            native_specials = plan7_forward_output_specials(output)
            if special_bytes:
                if native_specials == NULL:
                    raise RuntimeError("Forward special matrix is null")
                memcpy(
                    PyBytes_AS_STRING(special_storage),
                    native_specials,
                    special_bytes,
                )
            specials = memoryview(special_storage).cast("f")

            native_statistics = plan7_forward_output_statistics(output)
            if native_statistics == NULL:
                raise RuntimeError("Forward statistics are null")
            statistics = {
                "generation_f3_bits": native_statistics.generation_f3_bits,
                "candidate_count": native_statistics.candidate_count,
                "survivor_count": native_statistics.survivor_count,
                "work_cells": native_statistics.work_cells,
                "dp_workspace_bytes": native_statistics.dp_workspace_bytes,
                "xmx_workspace_bytes": native_statistics.xmx_workspace_bytes,
                "gather_workspace_bytes": (
                    native_statistics.gather_workspace_bytes
                ),
                "gathered_xmx_bytes": native_statistics.gathered_xmx_bytes,
                "output_byte_limit": native_statistics.output_byte_limit,
                "output_cap_fallback_count": (
                    native_statistics.output_cap_fallback_count
                ),
                "kernel_ms": native_statistics.kernel_milliseconds,
                "classification_ms": (
                    native_statistics.classification_milliseconds
                ),
                "gather_ms": native_statistics.gather_milliseconds,
                "download_ms": native_statistics.download_milliseconds,
                "total_ms": native_statistics.total_milliseconds,
            }
            provenance = _forward_provenance_from_output(output)
            statistics.update({
                "database_generation": provenance._value.database_generation,
                "batch_generation": provenance._value.batch_generation,
                "row_hash": provenance._value.row_hash,
                "special_hash": provenance._value.special_hash,
                "continuation_hash": provenance._value.continuation_hash,
                "pass_count": provenance._value.pass_count,
                "special_count": provenance._value.special_count,
                "_provenance": provenance,
            })
            return records, offsets, specials, statistics
        finally:
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)

    def backward_domain_many_raw(
        self,
        const uint32_t[::1] candidate_profiles,
        const uint32_t[::1] candidate_indices,
        const uint64_t[::1] forward_offsets,
        const float[::1] forward_specials,
        ForwardProfiles forward_profiles,
        ForwardProvenance provenance,
        float rt1=0.25,
        float rt2=0.10,
        float rt3=0.20,
        float guard_band=2.0e-4,
        uint64_t posterior_byte_budget=(384 * 1024 * 1024),
        bint _unsealed_test=False,
    ):
        """Run CUDA BackwardParser and domain posterior decoding."""
        cdef size_t candidate_count = <size_t> candidate_indices.shape[0]
        cdef vector[plan7_backward_domain_candidate] candidates
        cdef size_t candidate
        cdef plan7_backward_domain_output *output = NULL
        cdef const plan7_backward_domain_result *native_results
        cdef const uint64_t *native_offsets
        cdef const plan7_domain_posterior *native_posteriors
        cdef const uint64_t *native_region_offsets
        cdef const plan7_simple_region *native_regions
        cdef const plan7_backward_domain_statistics *native_statistics
        cdef size_t result_count
        cdef size_t result_bytes
        cdef size_t offset_bytes
        cdef size_t posterior_count
        cdef size_t posterior_bytes
        cdef size_t region_count
        cdef size_t region_bytes
        cdef bytes records
        cdef bytes offset_storage
        cdef bytes posterior_storage
        cdef bytes region_offset_storage
        cdef bytes region_storage
        cdef object offsets
        cdef object posteriors
        cdef object region_offsets
        cdef object regions
        cdef dict statistics
        cdef BackwardDomainProvenance output_provenance
        cdef char error[512]
        cdef int status

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if forward_profiles._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        if <size_t> candidate_profiles.shape[0] != candidate_count:
            raise ValueError("Backward/domain candidate lengths differ")
        if <size_t> forward_offsets.shape[0] != candidate_count + 1:
            raise ValueError("Backward/domain Forward offsets have wrong length")
        if sizeof(plan7_backward_domain_result) != (
            PLAN7_BACKWARD_DOMAIN_RECORD_SIZE
        ):
            raise RuntimeError("Backward/domain result ABI size mismatch")
        if sizeof(plan7_domain_posterior) != (
            PLAN7_BACKWARD_DOMAIN_POSTERIOR_SIZE
        ):
            raise RuntimeError("Backward/domain posterior ABI size mismatch")
        if sizeof(plan7_simple_region) != PLAN7_BACKWARD_DOMAIN_REGION_SIZE:
            raise RuntimeError("Backward/domain region ABI size mismatch")
        candidates.resize(candidate_count)
        for candidate in range(candidate_count):
            candidates[candidate].profile_index = candidate_profiles[candidate]
            candidates[candidate].sequence_index = candidate_indices[candidate]

        error[0] = 0
        if _unsealed_test:
            with nogil:
                status = plan7_backward_domain_unsealed_test_run(
                    forward_profiles._database,
                    self._batch,
                    candidates.data() if candidate_count else NULL,
                    candidate_count,
                    &forward_offsets[0],
                    &forward_specials[0] if forward_specials.shape[0] else NULL,
                    <size_t> forward_specials.shape[0],
                    rt1,
                    rt2,
                    rt3,
                    guard_band,
                    posterior_byte_budget,
                    &output,
                    error,
                    sizeof(error),
                )
        else:
            with nogil:
                status = plan7_backward_domain_run(
                    forward_profiles._database,
                    self._batch,
                    candidates.data() if candidate_count else NULL,
                    candidate_count,
                    &provenance._value,
                    &forward_offsets[0],
                    &forward_specials[0] if forward_specials.shape[0] else NULL,
                    <size_t> forward_specials.shape[0],
                    rt1,
                    rt2,
                    rt3,
                    guard_band,
                    posterior_byte_budget,
                    &output,
                    error,
                    sizeof(error),
                )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        try:
            result_count = plan7_backward_domain_output_result_count(output)
            if result_count != candidate_count:
                raise RuntimeError("Backward/domain result count changed")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(plan7_backward_domain_result)
            ):
                raise OverflowError("Backward/domain results exceed Python limits")
            result_bytes = result_count * sizeof(plan7_backward_domain_result)
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            native_results = plan7_backward_domain_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Backward/domain result storage is null")
                memcpy(PyBytes_AS_STRING(records), native_results, result_bytes)

            native_offsets = (
                plan7_backward_domain_output_posterior_offsets(output)
            )
            if native_offsets == NULL:
                raise RuntimeError("Backward/domain posterior offsets are null")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
            ) - 1:
                raise OverflowError(
                    "Backward/domain posterior offsets exceed Python limits"
                )
            offset_bytes = (result_count + 1) * sizeof(uint64_t)
            offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
            memcpy(
                PyBytes_AS_STRING(offset_storage),
                native_offsets,
                offset_bytes,
            )
            offsets = memoryview(offset_storage).cast("Q")

            posterior_count = (
                plan7_backward_domain_output_posterior_count(output)
            )
            if posterior_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(plan7_domain_posterior)
            ):
                raise OverflowError(
                    "Backward/domain posterior output exceeds Python limits"
                )
            posterior_bytes = (
                posterior_count * sizeof(plan7_domain_posterior)
            )
            posterior_storage = PyBytes_FromStringAndSize(NULL, posterior_bytes)
            native_posteriors = plan7_backward_domain_output_posteriors(output)
            if posterior_bytes:
                if native_posteriors == NULL:
                    raise RuntimeError("Backward/domain posterior storage is null")
                memcpy(
                    PyBytes_AS_STRING(posterior_storage),
                    native_posteriors,
                    posterior_bytes,
                )
            posteriors = memoryview(posterior_storage).cast("f")

            native_region_offsets = (
                plan7_backward_domain_output_region_offsets(output)
            )
            if native_region_offsets == NULL:
                raise RuntimeError("Backward/domain region offsets are null")
            region_offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
            memcpy(
                PyBytes_AS_STRING(region_offset_storage),
                native_region_offsets,
                offset_bytes,
            )
            region_offsets = memoryview(region_offset_storage).cast("Q")

            region_count = plan7_backward_domain_output_region_count(output)
            if region_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(plan7_simple_region)
            ):
                raise OverflowError(
                    "Backward/domain region output exceeds Python limits"
                )
            region_bytes = region_count * sizeof(plan7_simple_region)
            region_storage = PyBytes_FromStringAndSize(NULL, region_bytes)
            native_regions = plan7_backward_domain_output_regions(output)
            if region_bytes:
                if native_regions == NULL:
                    raise RuntimeError("Backward/domain region storage is null")
                memcpy(
                    PyBytes_AS_STRING(region_storage),
                    native_regions,
                    region_bytes,
                )
            regions = memoryview(region_storage).cast("I")

            native_statistics = plan7_backward_domain_output_statistics(output)
            if native_statistics == NULL:
                raise RuntimeError("Backward/domain statistics are null")
            statistics = {
                "candidate_count": native_statistics.candidate_count,
                "device_result_count": native_statistics.device_result_count,
                "cpu_required_count": native_statistics.cpu_required_count,
                "work_cells": native_statistics.work_cells,
                "dp_workspace_bytes": native_statistics.dp_workspace_bytes,
                "backward_special_workspace_bytes": (
                    native_statistics.backward_special_workspace_bytes
                ),
                "forward_special_workspace_bytes": (
                    native_statistics.forward_special_workspace_bytes
                ),
                "posterior_bytes": native_statistics.posterior_bytes,
                "simple_region_bytes": native_statistics.simple_region_bytes,
                "output_byte_limit": native_statistics.output_byte_limit,
                "output_cap_fallback_count": (
                    native_statistics.output_cap_fallback_count
                ),
                "work_cap_fallback_count": (
                    native_statistics.work_cap_fallback_count
                ),
                "posterior_omitted_count": (
                    native_statistics.posterior_omitted_count
                ),
                "own_scale_count": native_statistics.own_scale_count,
                "threshold_uncertain_count": (
                    native_statistics.threshold_uncertain_count
                ),
                "no_region_count": native_statistics.no_region_count,
                "simple_count": native_statistics.simple_count,
                "multidomain_fallback_count": (
                    native_statistics.multidomain_fallback_count
                ),
                "kernel_ms": native_statistics.kernel_milliseconds,
                "upload_ms": native_statistics.upload_milliseconds,
                "download_ms": native_statistics.download_milliseconds,
                "total_ms": native_statistics.total_milliseconds,
                "unsealed_test": bool(_unsealed_test),
            }
            output_provenance = _backward_provenance_from_output(output)
            statistics.update({
                "threshold_hash": output_provenance._value.threshold_hash,
                "result_hash": output_provenance._value.result_hash,
                "region_hash": output_provenance._value.region_hash,
                "_provenance": output_provenance,
            })
            return (
                records,
                offsets,
                posteriors,
                region_offsets,
                regions,
                statistics,
            )
        finally:
            if output != NULL:
                plan7_backward_domain_output_destroy(&output, NULL, 0)

    cdef object _forward_profile_selection_raw(
        self,
        ProfileSelection selection,
        const uint8_t[::1] postfilter_records,
        const uint64_t[::1] postfilter_offsets,
        const uint64_t[::1] residue_offsets,
        double f2,
        double f3,
        uint64_t gathered_byte_budget,
        bint sealed_domain_journal,
        bint rescore_simple_diagnostic,
        uint64_t rescore_compact_byte_budget,
        uint64_t rescore_matrix_byte_budget,
        uint64_t rescore_trace_byte_budget,
        int rescore_test_fault,
        uint64_t generation_tail_fingerprint,
        double generation_f1,
        bint generation_bias_filter,
        float rt1,
        float rt2,
        float rt3,
        float guard_band,
    ):
        """Run selection-aware F2/F3 without reading a live optimized profile."""
        cdef plan7_profile_selection_view view = selection._view()
        cdef vector[uint64_t] candidate_offsets
        cdef vector[uint32_t] candidate_indices
        cdef vector[float] filter_scores
        cdef vector[float] uncorrected_scores
        cdef vector[plan7_postfilter_result] candidate_records
        cdef vector[uint32_t] candidate_profiles
        cdef vector[plan7_backward_domain_candidate] domain_candidates
        cdef vector[size_t] pass_sources
        cdef vector[uint64_t] pass_special_offsets
        cdef vector[uint64_t] journal_profile_offsets
        cdef plan7_backward_domain_candidate domain_candidate
        cdef plan7_postfilter_result record
        cdef float_bits vfsc_bits
        cdef double_bits generation_f3
        cdef float usc
        cdef float bit_score
        cdef double probability
        cdef uint32_t previous
        cdef bint have_previous
        cdef bint host_attested
        cdef size_t profile_count = view.profile_count
        cdef size_t profile_index
        cdef size_t cursor
        cdef size_t start
        cdef size_t stop
        cdef size_t record_count
        cdef size_t candidate_count
        cdef uint64_t sequence_length
        cdef plan7_forward_database *database = NULL
        cdef plan7_forward_output *output = NULL
        cdef plan7_backward_domain_output *domain_output = NULL
        cdef plan7_domain_rescore_output *rescore_output = NULL
        cdef const plan7_forward_result *native_results
        cdef const uint64_t *native_offsets
        cdef const float *native_specials
        cdef const plan7_forward_provenance *native_provenance
        cdef const plan7_forward_statistics *native_statistics
        cdef char error[512]
        cdef char destroy_error[512]
        cdef int status = 0
        cdef int destroy_status = 0
        cdef size_t result_count
        cdef size_t result_bytes
        cdef size_t offset_bytes
        cdef size_t special_count
        cdef size_t special_bytes
        cdef size_t pass_count
        cdef size_t source
        cdef bytes records
        cdef bytes offset_storage
        cdef bytes special_storage
        cdef object special_offsets
        cdef object specials
        cdef carray row_offsets
        cdef carray expected_indices
        cdef dict statistics
        cdef ForwardProvenance provenance
        cdef object journal_capsule = None
        cdef object rescore_payload = None
        cdef object upstream_payload = None
        cdef bytes profile_fingerprint_storage = b""
        cdef const uint8_t[::1] profile_fingerprint_view
        cdef const uint8_t[::1] sequence_fingerprint_view
        cdef double_bits threshold_bits
        cdef float_bits threshold_float_bits

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if sizeof(plan7_postfilter_result) != PLAN7_POSTFILTER_RECORD_SIZE:
            raise RuntimeError("post-filter result ABI size mismatch")
        if sizeof(plan7_forward_result) != PLAN7_FORWARD_RECORD_SIZE:
            raise RuntimeError("Forward result ABI size mismatch")
        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if _UINT32_ARRAY_TEMPLATE.itemsize != sizeof(uint32_t):
            raise RuntimeError("array('I') is not native uint32")
        if sealed_domain_journal:
            if (
                not generation_bias_filter
                or not isfinite(generation_f1)
                or generation_f1 < 0.0
                or generation_f1 > 1.0
                or not isfinite(f2)
                or f2 < 0.0
                or f2 > 1.0
                or not isfinite(f3)
                or f3 < 0.0
                or f3 > 1.0
                or not isfinite(rt1)
                or not isfinite(rt2)
                or not isfinite(rt3)
                or not isfinite(guard_band)
                or guard_band < <float> 2.0e-4
                or guard_band > <float> 1.0
                or rt1 != <float> 0.25
                or rt2 != <float> 0.10
                or rt3 != <float> 0.20
            ):
                raise ValueError(
                    "sealed domain journals require finite canonical thresholds "
                    "with bias filtering and guard_band >= 2e-4"
                )
        if postfilter_offsets.shape[0] != profile_count + 1:
            raise ValueError("post-filter row-offset count differs from selection")
        if postfilter_offsets[0] != 0:
            raise ValueError("post-filter row offsets must start at zero")
        if <size_t> postfilter_records.shape[0] % sizeof(plan7_postfilter_result):
            raise ValueError("post-filter result storage has trailing bytes")
        record_count = (
            <size_t> postfilter_records.shape[0]
            // sizeof(plan7_postfilter_result)
        )
        if postfilter_offsets[profile_count] != record_count:
            raise ValueError("post-filter row offsets do not span result storage")
        if residue_offsets.shape[0] != self._sequence_count + 1:
            raise ValueError("target residue-prefix length differs from targets")
        if residue_offsets[0] != 0:
            raise ValueError("target residue prefix must start at zero")
        if profile_count and (
            view.profiles == NULL
            or view.m_mu == NULL
            or view.m_lambda == NULL
            or view.v_mu == NULL
            or view.v_lambda == NULL
            or view.identity_tokens == NULL
        ):
            raise RuntimeError("profile selection storage is incomplete")

        host_attested = plan7_bias_host_environment_attested() == 1
        candidate_offsets.reserve(profile_count + 1)
        candidate_offsets.push_back(0)
        for profile_index in range(profile_count):
            start = <size_t> postfilter_offsets[profile_index]
            stop = <size_t> postfilter_offsets[profile_index + 1]
            if start > stop or stop > record_count:
                raise ValueError("post-filter row offsets are not monotone")
            previous = 0
            have_previous = False
            for cursor in range(start, stop):
                memcpy(
                    &record,
                    &postfilter_records[cursor * sizeof(plan7_postfilter_result)],
                    sizeof(plan7_postfilter_result),
                )
                if record.sequence_index >= self._sequence_count:
                    raise IndexError("post-filter result sequence index out of range")
                if have_previous and record.sequence_index <= previous:
                    raise ValueError(
                        "post-filter result sequence indexes are not increasing"
                    )
                previous = record.sequence_index
                have_previous = True
                if not host_attested or record.action != PLAN7_BIAS_DEFINITE_PASS:
                    continue
                vfsc_bits.value = record.vfsc
                if (
                    record.msv_status != PLAN7_SSV_OK
                    or not isfinite(record.filtersc)
                    or (
                        not isfinite(record.vfsc)
                        and vfsc_bits.bits != 0x7f800000
                    )
                    or not isfinite(view.profiles[profile_index].scale)
                    or view.profiles[profile_index].scale <= 0.0
                ):
                    continue
                usc = <float> record.msv_numerator
                usc = usc / view.profiles[profile_index].scale
                usc = usc - <float> 3.0
                bit_score = <float> ((usc - record.filtersc) / eslCONST_LOG2)
                probability = esl_gumbel_surv(
                    bit_score,
                    view.m_mu[profile_index],
                    view.m_lambda[profile_index],
                )
                if probability > f2:
                    bit_score = <float> (
                        (record.vfsc - record.filtersc) / eslCONST_LOG2
                    )
                    probability = esl_gumbel_surv(
                        bit_score,
                        view.v_mu[profile_index],
                        view.v_lambda[profile_index],
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
                if sequence_length > 100000:
                    raise ValueError("Forward target exceeds HMMER's protein limit")
                candidate_indices.push_back(record.sequence_index)
                filter_scores.push_back(record.filtersc)
                uncorrected_scores.push_back(usc)
                candidate_records.push_back(record)
                candidate_profiles.push_back(<uint32_t> profile_index)
            candidate_offsets.push_back(candidate_indices.size())

        candidate_count = candidate_indices.size()
        row_offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            row_offsets.data.as_ulonglongs[profile_index] = (
                candidate_offsets[profile_index]
            )
        expected_indices = clone(_UINT32_ARRAY_TEMPLATE, candidate_count, False)
        for cursor in range(candidate_count):
            expected_indices.data.as_uints[cursor] = candidate_indices[cursor]
        error[0] = 0
        with nogil:
            status = plan7_profile_selection_stage_forward(
                selection._selection, &database, error, sizeof(error)
            )
        if status == 0:
            with nogil:
                status = plan7_forward_run_batch_workspace(
                    database,
                    self._batch,
                    view.identity_tokens,
                    profile_count,
                    candidate_offsets.data(),
                    candidate_indices.data(),
                    filter_scores.data(),
                    candidate_count,
                    f3,
                    gathered_byte_budget,
                    &output,
                    error,
                    sizeof(error),
                )
        if status != 0:
            if database != NULL:
                plan7_forward_database_destroy(&database, NULL, 0)
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)
            raise RuntimeError(error.decode("utf-8", "replace"))

        if sealed_domain_journal:
            try:
                result_count = plan7_forward_output_result_count(output)
                native_results = plan7_forward_output_results(output)
                native_offsets = plan7_forward_output_special_offsets(output)
                native_specials = plan7_forward_output_specials(output)
                native_provenance = plan7_forward_output_provenance(output)
                special_count = plan7_forward_output_special_count(output)
                if result_count != candidate_count:
                    raise RuntimeError("Forward result count changed")
                if (
                    native_offsets == NULL
                    or native_provenance == NULL
                    or (result_count and native_results == NULL)
                    or (special_count and native_specials == NULL)
                ):
                    raise RuntimeError("Forward journal storage is incomplete")

                pass_special_offsets.push_back(0)
                journal_profile_offsets.push_back(0)
                pass_count = 0
                for profile_index in range(profile_count):
                    start = <size_t> candidate_offsets[profile_index]
                    stop = <size_t> candidate_offsets[profile_index + 1]
                    for source in range(start, stop):
                        if (
                            native_results[source].action
                            != PLAN7_FORWARD_DEFINITE_PASS
                        ):
                            continue
                        if native_offsets[source] != pass_special_offsets.back():
                            raise RuntimeError(
                                "Forward pass special rows are not contiguous"
                            )
                        domain_candidate.profile_index = candidate_profiles[source]
                        domain_candidate.sequence_index = candidate_indices[source]
                        domain_candidates.push_back(domain_candidate)
                        pass_sources.push_back(source)
                        pass_special_offsets.push_back(native_offsets[source + 1])
                        pass_count += 1
                    journal_profile_offsets.push_back(pass_count)
                if (
                    pass_count != native_provenance.pass_count
                    or pass_special_offsets[pass_count] != special_count
                ):
                    raise RuntimeError("Forward pass provenance count changed")

                error[0] = 0
                with nogil:
                    status = plan7_backward_domain_run(
                        database,
                        self._batch,
                        domain_candidates.data() if pass_count else NULL,
                        pass_count,
                        native_provenance,
                        pass_special_offsets.data(),
                        native_specials if special_count else NULL,
                        special_count,
                        rt1,
                        rt2,
                        rt3,
                        guard_band,
                        0,
                        &domain_output,
                        error,
                        sizeof(error),
                    )
                if status != 0:
                    raise RuntimeError(error.decode("utf-8", "replace"))
                if (
                    rescore_simple_diagnostic
                    or generation_tail_fingerprint != 0
                ):
                    if rescore_test_fault != 0:
                        error[0] = 0
                        with nogil:
                            status = (
                                plan7_backward_domain_output_apply_test_fault(
                                    domain_output,
                                    rescore_test_fault,
                                    error,
                                    sizeof(error),
                                )
                            )
                        if status != 0:
                            raise RuntimeError(
                                error.decode("utf-8", "replace")
                            )
                    error[0] = 0
                    with nogil:
                        status = plan7_domain_rescore_run(
                            database,
                            self._batch,
                            domain_output,
                            rescore_compact_byte_budget,
                            rescore_matrix_byte_budget,
                            rescore_trace_byte_budget,
                            &rescore_output,
                            error,
                            sizeof(error),
                        )
                    if status != 0:
                        raise RuntimeError(error.decode("utf-8", "replace"))
                    if rescore_simple_diagnostic:
                        upstream_payload = (
                            _backward_domain_route_payload_from_output(
                                domain_output
                            )
                        )
                        rescore_payload = _domain_rescore_payload_from_output(
                            rescore_output
                        )
                        rescore_payload = rescore_payload + (upstream_payload,)
                profile_fingerprint_storage = b"".join(selection._fingerprints)
                profile_fingerprint_view = profile_fingerprint_storage
                sequence_fingerprint_view = self._content_fingerprint
                journal_capsule = _build_continuation_journal_capsule(
                    &view,
                    (
                        &profile_fingerprint_view[0]
                        if profile_fingerprint_view.shape[0]
                        else NULL
                    ),
                    &sequence_fingerprint_view[0],
                    (
                        <const plan7_postfilter_result *> &postfilter_records[0]
                        if record_count
                        else NULL
                    ),
                    record_count,
                    &postfilter_offsets[0],
                    candidate_records.data() if candidate_count else NULL,
                    uncorrected_scores.data() if candidate_count else NULL,
                    candidate_profiles.data() if candidate_count else NULL,
                    pass_sources.data() if pass_count else NULL,
                    candidate_offsets.data(),
                    journal_profile_offsets.data(),
                    pass_count,
                    pass_special_offsets.data(),
                    output,
                    domain_output,
                    (
                        rescore_output
                        if generation_tail_fingerprint != 0
                        else NULL
                    ),
                    generation_tail_fingerprint,
                    generation_f1,
                    f2,
                    f3,
                    generation_bias_filter,
                    rt1,
                    rt2,
                    rt3,
                    guard_band,
                )
            except:
                if rescore_output != NULL:
                    plan7_domain_rescore_output_destroy(
                        &rescore_output, NULL, 0
                    )
                if domain_output != NULL:
                    plan7_backward_domain_output_destroy(
                        &domain_output, NULL, 0
                    )
                if database != NULL:
                    plan7_forward_database_destroy(&database, NULL, 0)
                if output != NULL:
                    plan7_forward_output_destroy(&output, NULL, 0)
                raise

        destroy_error[0] = 0
        if database != NULL:
            with nogil:
                destroy_status = plan7_forward_database_destroy(
                    &database, destroy_error, sizeof(destroy_error)
                )
        if destroy_status != 0:
            if rescore_output != NULL:
                plan7_domain_rescore_output_destroy(
                    &rescore_output, NULL, 0
                )
            if domain_output != NULL:
                plan7_backward_domain_output_destroy(&domain_output, NULL, 0)
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)
            raise RuntimeError(destroy_error.decode("utf-8", "replace"))

        if sealed_domain_journal:
            if rescore_output != NULL:
                plan7_domain_rescore_output_destroy(
                    &rescore_output, NULL, 0
                )
            if domain_output != NULL:
                plan7_backward_domain_output_destroy(
                    &domain_output, NULL, 0
                )
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)
            if rescore_simple_diagnostic:
                return journal_capsule, rescore_payload
            return journal_capsule

        try:
            result_count = plan7_forward_output_result_count(output)
            if result_count != candidate_count:
                raise RuntimeError("Forward result count changed")
            if result_count > (<size_t> -1) // sizeof(plan7_forward_result):
                raise OverflowError("Forward result size overflows size_t")
            result_bytes = result_count * sizeof(plan7_forward_result)
            if result_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("Forward result size exceeds Python limits")
            records = PyBytes_FromStringAndSize(NULL, result_bytes)
            native_results = plan7_forward_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Forward result storage is null")
                memcpy(PyBytes_AS_STRING(records), native_results, result_bytes)

            native_offsets = plan7_forward_output_special_offsets(output)
            if native_offsets == NULL:
                raise RuntimeError("Forward special offsets are null")
            if result_count > (
                <size_t> PY_SSIZE_T_MAX // sizeof(uint64_t)
            ) - 1:
                raise OverflowError("Forward special offsets exceed Python limits")
            offset_bytes = (result_count + 1) * sizeof(uint64_t)
            offset_storage = PyBytes_FromStringAndSize(NULL, offset_bytes)
            memcpy(PyBytes_AS_STRING(offset_storage), native_offsets, offset_bytes)
            special_offsets = memoryview(offset_storage).cast("Q")

            special_count = plan7_forward_output_special_count(output)
            if special_count > (<size_t> -1) // sizeof(float):
                raise OverflowError("Forward special matrix size overflows size_t")
            special_bytes = special_count * sizeof(float)
            if special_bytes > <size_t> PY_SSIZE_T_MAX:
                raise OverflowError("Forward special matrix exceeds Python limits")
            special_storage = PyBytes_FromStringAndSize(NULL, special_bytes)
            native_specials = plan7_forward_output_specials(output)
            if special_bytes:
                if native_specials == NULL:
                    raise RuntimeError("Forward special matrix is null")
                memcpy(
                    PyBytes_AS_STRING(special_storage),
                    native_specials,
                    special_bytes,
                )
            specials = memoryview(special_storage).cast("f")

            native_statistics = plan7_forward_output_statistics(output)
            if native_statistics == NULL:
                raise RuntimeError("Forward statistics are null")
            statistics = {
                "generation_f3_bits": native_statistics.generation_f3_bits,
                "candidate_count": native_statistics.candidate_count,
                "survivor_count": native_statistics.survivor_count,
                "work_cells": native_statistics.work_cells,
                "dp_workspace_bytes": native_statistics.dp_workspace_bytes,
                "xmx_workspace_bytes": native_statistics.xmx_workspace_bytes,
                "gather_workspace_bytes": native_statistics.gather_workspace_bytes,
                "gathered_xmx_bytes": native_statistics.gathered_xmx_bytes,
                "output_byte_limit": native_statistics.output_byte_limit,
                "output_cap_fallback_count": (
                    native_statistics.output_cap_fallback_count
                ),
                "kernel_ms": native_statistics.kernel_milliseconds,
                "classification_ms": native_statistics.classification_milliseconds,
                "gather_ms": native_statistics.gather_milliseconds,
                "download_ms": native_statistics.download_milliseconds,
                "total_ms": native_statistics.total_milliseconds,
            }
            provenance = _forward_provenance_from_output(output)
            statistics.update({
                "database_generation": provenance._value.database_generation,
                "batch_generation": provenance._value.batch_generation,
                "row_hash": provenance._value.row_hash,
                "special_hash": provenance._value.special_hash,
                "continuation_hash": provenance._value.continuation_hash,
                "pass_count": provenance._value.pass_count,
                "special_count": provenance._value.special_count,
                "_provenance": provenance,
            })
            result = (
                records,
                row_offsets,
                expected_indices,
                special_offsets,
                specials,
                statistics,
            )
            return result
        finally:
            if rescore_output != NULL:
                plan7_domain_rescore_output_destroy(
                    &rescore_output, NULL, 0
                )
            if domain_output != NULL:
                plan7_backward_domain_output_destroy(&domain_output, NULL, 0)
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)

    def forward_profile_selection_raw(
        self,
        ProfileSelection selection,
        const uint8_t[::1] postfilter_records,
        const uint64_t[::1] postfilter_offsets,
        const uint64_t[::1] residue_offsets,
        double f2,
        double f3,
        uint64_t gathered_byte_budget=PLAN7_FORWARD_MAX_GATHERED_BYTES,
    ):
        """Run diagnostic selection-aware F2/F3 without minting a seal."""
        return self._forward_profile_selection_raw(
            selection,
            postfilter_records,
            postfilter_offsets,
            residue_offsets,
            f2,
            f3,
            gathered_byte_budget,
            False,
            False,
            0,
            0,
            0,
            0,
            0,
            0.02,
            True,
            <float> 0.25,
            <float> 0.10,
            <float> 0.20,
            <float> 2.0e-4,
        )

    def _postfilter_forward_domain_selection_sealed(
        self,
        ProfileSelection selection,
        double f1,
        double f2,
        double f3,
        float guard_band=2.0e-4,
        uint64_t gathered_byte_budget=PLAN7_FORWARD_MAX_GATHERED_BYTES,
        bint rescore_simple_diagnostic=False,
        uint64_t rescore_matrix_byte_budget=0,
        uint64_t rescore_trace_byte_budget=0,
        uint64_t rescore_compact_byte_budget=0,
        int _rescore_test_fault=0,
        uint64_t generation_tail_fingerprint=0,
    ):
        """Run the fused package-internal path and return one opaque seal.

        Underscore native entry points are trusted implementation details, not
        a security boundary against deliberate same-process private API use.
        """
        cdef object postfilter_records
        cdef object postfilter_offsets
        cdef object result
        cdef carray residue_offsets
        cdef size_t index

        postfilter_records, postfilter_offsets = (
            self.postfilter_profile_selection_csr_raw(selection, f1)
        )
        residue_offsets = clone(
            _UINT64_ARRAY_TEMPLATE, self._sequence_count + 1, False
        )
        residue_offsets.data.as_ulonglongs[0] = 0
        for index in range(self._sequence_count):
            if self._lengths[index] > (
                (<uint64_t> -1)
                - residue_offsets.data.as_ulonglongs[index]
            ):
                raise OverflowError("target residue prefix overflows uint64")
            residue_offsets.data.as_ulonglongs[index + 1] = (
                residue_offsets.data.as_ulonglongs[index]
                + self._lengths[index]
            )
        result = self._forward_profile_selection_raw(
            selection,
            memoryview(postfilter_records),
            memoryview(postfilter_offsets),
            memoryview(residue_offsets),
            f2,
            f3,
            gathered_byte_budget,
            True,
            rescore_simple_diagnostic,
            rescore_compact_byte_budget,
            rescore_matrix_byte_budget,
            rescore_trace_byte_budget,
            _rescore_test_fault,
            generation_tail_fingerprint,
            f1,
            True,
            <float> 0.25,
            <float> 0.10,
            <float> 0.20,
            guard_band,
        )
        return result

    cdef size_t _run_profile_selection_candidates(
        self,
        const plan7_profile_selection_view *view,
        double f1,
    ) except? 0:
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t words_per_profile
        cdef size_t candidate_word_count
        cdef size_t word_index
        cdef size_t sequence_index
        cdef size_t candidate_count = 0
        cdef size_t output_index
        cdef uint32_t word
        cdef unsigned bit
        cdef int status
        cdef char error[512]

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        profile_count = view.profile_count
        if profile_count and (
            view.packed_scores == NULL
            or view.profiles == NULL
            or view.m_mu == NULL
            or view.m_lambda == NULL
            or view.v_mu == NULL
            or view.v_lambda == NULL
            or view.bias_templates == NULL
            or view.identity_tokens == NULL
        ):
            raise RuntimeError("profile selection storage is incomplete")
        if self._sequence_count > (<size_t> -1) - 31:
            raise OverflowError("candidate mask size overflows size_t")
        words_per_profile = (self._sequence_count + 31) // 32
        if words_per_profile and profile_count > (<size_t> -1) // words_per_profile:
            raise OverflowError("candidate mask size overflows size_t")
        candidate_word_count = profile_count * words_per_profile
        self._candidate_words.resize(candidate_word_count)
        self._candidate_counts.resize(profile_count)
        error[0] = 0
        with nogil:
            status = plan7_ssv_sequence_batch_f1_mask_many(
                self._batch,
                view.packed_scores,
                view.packed_score_count,
                view.profiles,
                profile_count,
                view.m_mu,
                view.m_lambda,
                f1,
                self._candidate_words.data() if candidate_word_count else NULL,
                candidate_word_count,
                error,
                sizeof(error),
            )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))

        self._candidate_offsets.resize(profile_count)
        for profile_index in range(profile_count):
            self._candidate_counts[profile_index] = 0
            for word_index in range(words_per_profile):
                self._candidate_counts[profile_index] += plan7_popcount_u32(
                    self._candidate_words[
                        profile_index * words_per_profile + word_index
                    ]
                )
            self._candidate_offsets[profile_index] = candidate_count
            if self._candidate_counts[profile_index] > (
                (<size_t> -1) - candidate_count
            ):
                raise OverflowError("candidate count overflows size_t")
            candidate_count += self._candidate_counts[profile_index]

        self._candidate_indices.resize(candidate_count)
        for profile_index in range(profile_count):
            output_index = self._candidate_offsets[profile_index]
            for word_index in range(words_per_profile):
                word = self._candidate_words[
                    profile_index * words_per_profile + word_index
                ]
                while word:
                    bit = plan7_ctz_u32(word)
                    sequence_index = word_index * 32 + bit
                    if sequence_index >= self._sequence_count:
                        raise RuntimeError("candidate mask has trailing bits set")
                    self._candidate_indices[output_index] = <uint32_t> sequence_index
                    output_index += 1
                    word &= word - 1
            if output_index != (
                self._candidate_offsets[profile_index]
                + self._candidate_counts[profile_index]
            ):
                raise RuntimeError("candidate mask count changed")
        return profile_count

    def postfilter_profile_selection_csr_raw(
        self,
        ProfileSelection selection,
        double f1,
    ):
        """Run a sealed selection without reading any live optimized profile."""
        cdef plan7_profile_selection_view view = selection._view()
        cdef plan7_viterbi_database *database = NULL
        cdef char error[512]
        cdef char destroy_error[512]
        cdef int status = 0
        cdef int destroy_status = 0
        cdef int cutoff_mode
        cdef float cutoff
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef size_t result_bytes
        cdef bytearray records
        cdef uint8_t[::1] record_view
        cdef carray offsets

        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if sizeof(plan7_postfilter_result) != PLAN7_POSTFILTER_RECORD_SIZE:
            raise RuntimeError("post-filter result ABI size mismatch")
        profile_count = self._run_profile_selection_candidates(&view, f1)
        self._bias_profiles.resize(profile_count)
        self._bias_candidate_offsets.resize(profile_count + 1)
        if profile_count:
            memcpy(
                self._bias_profiles.data(),
                view.bias_templates,
                profile_count * sizeof(plan7_bias_profile),
            )
        for profile_index in range(profile_count):
            cutoff_mode = plan7_ssv_f1_cutoff(
                view.m_mu[profile_index],
                view.m_lambda[profile_index],
                f1,
                &cutoff,
            )
            self._bias_profiles[profile_index].cutoff_mode = cutoff_mode
            if cutoff_mode == PLAN7_F1_CUTOFF_SCORE:
                self._bias_profiles[profile_index].cutoff_bit_score = cutoff
            self._bias_candidate_offsets[profile_index] = (
                self._candidate_offsets[profile_index]
            )
        candidate_count = self._candidate_indices.size()
        self._bias_candidate_offsets[profile_count] = candidate_count
        self._postfilter_results.resize(candidate_count)

        if candidate_count:
            error[0] = 0
            with nogil:
                status = plan7_profile_selection_stage_viterbi(
                    selection._selection, &database, error, sizeof(error)
                )
            if status == 0:
                with nogil:
                    status = plan7_ssv_sequence_batch_postfilter_candidates_many(
                        self._batch,
                        self._bias_profiles.data(),
                        profile_count,
                        self._bias_candidate_offsets.data(),
                        self._candidate_indices.data(),
                        candidate_count,
                        view.identity_tokens,
                        database,
                        self._postfilter_results.data(),
                        candidate_count,
                        error,
                        sizeof(error),
                    )
                destroy_error[0] = 0
                with nogil:
                    destroy_status = plan7_viterbi_database_destroy(
                        &database, destroy_error, sizeof(destroy_error)
                    )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))
            if destroy_status != 0:
                raise RuntimeError(destroy_error.decode("utf-8", "replace"))

        if candidate_count > (<size_t> -1) // sizeof(plan7_postfilter_result):
            raise OverflowError("post-filter result size overflows size_t")
        result_bytes = candidate_count * sizeof(plan7_postfilter_result)
        records = bytearray(result_bytes)
        if result_bytes:
            record_view = records
            memcpy(
                &record_view[0], self._postfilter_results.data(), result_bytes
            )
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._bias_candidate_offsets[profile_index]
            )
        return records, offsets

    cdef size_t _run_postfilter_candidates_many(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
        source_profiles,
        ViterbiProfiles viterbi_profiles,
    ) except? 0:
        cdef char error[512]
        cdef size_t profile_count
        cdef size_t candidate_count
        cdef size_t profile_index
        cdef size_t expected_bias_bytes
        cdef int status
        cdef tuple owners = tuple(source_profiles)
        cdef vector[uintptr_t] source_pointers
        cdef OptimizedProfile source_profile
        cdef object value

        if viterbi_profiles._database == NULL:
            raise RuntimeError("Viterbi profiles are closed")
        profile_count = self._run_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
        )
        if plan7_viterbi_database_profile_count(
            viterbi_profiles._database
        ) != profile_count:
            raise ValueError("Viterbi and SSV profile counts differ")
        if len(owners) != profile_count:
            raise ValueError("source and SSV profile counts differ")
        source_pointers.reserve(profile_count)
        for profile_index in range(profile_count):
            value = owners[profile_index]
            if not isinstance(value, OptimizedProfile):
                raise TypeError(
                    "source profiles must be OptimizedProfile objects"
                )
            if value is not viterbi_profiles._owners[profile_index]:
                raise ValueError(
                    "source profile identity differs from Viterbi profile row"
                )
            source_profile = value
            source_pointers.push_back(<uintptr_t> source_profile._om)
        if profile_count == 0:
            if packed_bias_profiles.shape[0] != 0:
                raise ValueError("packed bias profiles have trailing bytes")
            self._bias_profiles.clear()
            self._postfilter_results.clear()
            self._bias_candidate_offsets.resize(1)
            self._bias_candidate_offsets[0] = 0
            return 0
        if profile_count > (<size_t> -1) // sizeof(plan7_bias_profile):
            raise OverflowError("packed bias profile size overflows size_t")
        expected_bias_bytes = profile_count * sizeof(plan7_bias_profile)
        if <size_t> packed_bias_profiles.shape[0] != expected_bias_bytes:
            raise ValueError("packed bias profile buffer has the wrong size")

        self._bias_profiles.resize(profile_count)
        memcpy(
            self._bias_profiles.data(),
            &packed_bias_profiles[0],
            expected_bias_bytes,
        )
        candidate_count = self._candidate_indices.size()
        self._postfilter_results.resize(candidate_count)
        self._bias_candidate_offsets.resize(profile_count + 1)
        for profile_index in range(profile_count):
            self._bias_candidate_offsets[profile_index] = (
                self._candidate_offsets[profile_index]
            )
        self._bias_candidate_offsets[profile_count] = candidate_count
        if candidate_count == 0:
            return profile_count

        error[0] = 0
        # Hold the GIL while native code validates live P7_OPROFILE snapshots;
        # PyHMMER mutation APIs must not race those private-array reads.
        status = plan7_ssv_sequence_batch_postfilter_candidates_many(
            self._batch,
            self._bias_profiles.data(),
            profile_count,
            self._bias_candidate_offsets.data(),
            self._candidate_indices.data(),
            candidate_count,
            source_pointers.data(),
            viterbi_profiles._database,
            self._postfilter_results.data(),
            candidate_count,
            error,
            sizeof(error),
        )
        if status != 0:
            raise RuntimeError(error.decode("utf-8", "replace"))
        return profile_count

    def postfilter_candidates_many_csr_raw(
        self,
        const uint8_t[::1] packed_scores,
        const uint64_t[::1] score_offsets,
        const uint64_t[::1] score_counts,
        const int32_t[::1] score_strides,
        const int32_t[::1] model_lengths,
        const uint8_t[::1] constants,
        const float[::1] scales,
        const float[::1] m_mu,
        const float[::1] m_lambda,
        double f1,
        const uint8_t[::1] packed_bias_profiles,
        source_profiles,
        ViterbiProfiles viterbi_profiles,
    ):
        """Return version-1 post-filter records and native uint64 row offsets."""
        cdef size_t profile_count
        cdef size_t profile_index
        cdef size_t candidate_count
        cdef size_t result_bytes
        cdef bytearray records
        cdef uint8_t[::1] record_view
        cdef carray offsets

        if _UINT64_ARRAY_TEMPLATE.itemsize != sizeof(uint64_t):
            raise RuntimeError("array('Q') is not native uint64")
        if sizeof(plan7_postfilter_result) != PLAN7_POSTFILTER_RECORD_SIZE:
            raise RuntimeError("post-filter result ABI size mismatch")
        profile_count = self._run_postfilter_candidates_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
            m_mu,
            m_lambda,
            f1,
            packed_bias_profiles,
            source_profiles,
            viterbi_profiles,
        )
        candidate_count = self._postfilter_results.size()
        if candidate_count > (<size_t> -1) // sizeof(plan7_postfilter_result):
            raise OverflowError("post-filter result size overflows size_t")
        result_bytes = candidate_count * sizeof(plan7_postfilter_result)
        records = bytearray(result_bytes)
        if result_bytes:
            record_view = records
            memcpy(
                &record_view[0], self._postfilter_results.data(), result_bytes
            )
        offsets = clone(_UINT64_ARRAY_TEMPLATE, profile_count + 1, False)
        for profile_index in range(profile_count + 1):
            offsets.data.as_ulonglongs[profile_index] = <uint64_t> (
                self._bias_candidate_offsets[profile_index]
            )
        return records, offsets


def filter_raw(
    const uint8_t[::1] striped_scores,
    int score_stride,
    int model_length,
    int alphabet_size,
    const uint8_t[::1] residues,
    const uint64_t[::1] offsets,
    int tbm,
    int tec,
    int base,
    int bias,
    float scale,
):
    cdef SequenceBatch batch = SequenceBatch(residues, offsets, alphabet_size)
    try:
        return batch.filter_raw(
            striped_scores,
            score_stride,
            model_length,
            alphabet_size,
            tbm,
            tec,
            base,
            bias,
            scale,
        )
    finally:
        batch.close()


def f1_decision(
    int status,
    int numerator,
    uint64_t length,
    float scale,
    float m_mu,
    float m_lambda,
    double f1,
):
    cdef double p
    cdef int action
    if not 0 <= status <= 255:
        raise ValueError("status must fit in uint8")
    if not -32768 <= numerator <= 32767:
        raise ValueError("numerator must fit in int16")
    action = plan7_ssv_f1_decision(
        <uint8_t> status,
        <int16_t> numerator,
        length,
        scale,
        m_mu,
        m_lambda,
        f1,
        &p,
    )
    return action, p


def f1_cutoff(float m_mu, float m_lambda, double f1):
    cdef float cutoff
    cdef int mode = plan7_ssv_f1_cutoff(m_mu, m_lambda, f1, &cutoff)
    return mode, cutoff if mode == PLAN7_F1_CUTOFF_SCORE else None


def f1_cutoff_decision(
    int status,
    int numerator,
    uint64_t length,
    float scale,
    int cutoff_mode,
    float cutoff_bit_score,
):
    if not 0 <= status <= 255:
        raise ValueError("status must fit in uint8")
    if not -32768 <= numerator <= 32767:
        raise ValueError("numerator must fit in int16")
    return plan7_ssv_f1_cutoff_decision(
        <uint8_t> status,
        <int16_t> numerator,
        length,
        scale,
        cutoff_mode,
        cutoff_bit_score,
    )


STATUS_OK = PLAN7_SSV_OK
STATUS_ERANGE = PLAN7_SSV_ERANGE
STATUS_ENORESULT = PLAN7_SSV_ENORESULT
STATUS_EMPTY = PLAN7_SSV_EMPTY
F1_CPU_REQUIRED = PLAN7_F1_CPU_REQUIRED
F1_DEFINITE_REJECT = PLAN7_F1_DEFINITE_REJECT
F1_CUTOFF_INVALID = PLAN7_F1_CUTOFF_INVALID
F1_CUTOFF_SCORE = PLAN7_F1_CUTOFF_SCORE
F1_CUTOFF_ALWAYS_REJECT = PLAN7_F1_CUTOFF_ALWAYS_REJECT
F1_CUTOFF_ALWAYS_CPU = PLAN7_F1_CUTOFF_ALWAYS_CPU
BIAS_CPU_REQUIRED = PLAN7_BIAS_CPU_REQUIRED
BIAS_DEFINITE_REJECT = PLAN7_BIAS_DEFINITE_REJECT
BIAS_DEFINITE_PASS = PLAN7_BIAS_DEFINITE_PASS
BIAS_CUTOFF_INVALID = PLAN7_BIAS_CUTOFF_INVALID
BIAS_CUTOFF_SCORE = PLAN7_BIAS_CUTOFF_SCORE
BIAS_CUTOFF_ALWAYS_REJECT = PLAN7_BIAS_CUTOFF_ALWAYS_REJECT
BIAS_CUTOFF_ALWAYS_PASS = PLAN7_BIAS_CUTOFF_ALWAYS_PASS
BIAS_PROFILE_SIZE = sizeof(plan7_bias_profile)
BIAS_RESULT_SIZE = sizeof(plan7_bias_result)
POSTFILTER_RECORD_VERSION = PLAN7_POSTFILTER_RECORD_VERSION
POSTFILTER_RESULT_SIZE = sizeof(plan7_postfilter_result)
FORWARD_RECORD_VERSION = PLAN7_FORWARD_RECORD_VERSION
FORWARD_RESULT_SIZE = sizeof(plan7_forward_result)
FORWARD_MAX_GATHERED_BYTES = PLAN7_FORWARD_MAX_GATHERED_BYTES
FORWARD_CPU_REQUIRED = PLAN7_FORWARD_CPU_REQUIRED
FORWARD_DEFINITE_REJECT = PLAN7_FORWARD_DEFINITE_REJECT
FORWARD_DEFINITE_PASS = PLAN7_FORWARD_DEFINITE_PASS
FORWARD_STATUS_OK = PLAN7_FORWARD_OK
FORWARD_STATUS_ERANGE = PLAN7_FORWARD_ERANGE
FORWARD_STATUS_ENORESULT = PLAN7_FORWARD_ENORESULT
FORWARD_STATUS_EMPTY = PLAN7_FORWARD_EMPTY
BACKWARD_DOMAIN_RECORD_VERSION = PLAN7_BACKWARD_DOMAIN_RECORD_VERSION
BACKWARD_DOMAIN_RESULT_SIZE = sizeof(plan7_backward_domain_result)
BACKWARD_DOMAIN_POSTERIOR_SIZE = sizeof(plan7_domain_posterior)
BACKWARD_DOMAIN_REGION_SIZE = sizeof(plan7_simple_region)
BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES = (
    PLAN7_BACKWARD_DOMAIN_MAX_POSTERIOR_BYTES
)
BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS = (
    PLAN7_BACKWARD_DOMAIN_MAX_ROW_WORK_CELLS
)
BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS = (
    PLAN7_BACKWARD_DOMAIN_MAX_RUN_WORK_CELLS
)
BACKWARD_DOMAIN_CPU_REQUIRED = PLAN7_BACKWARD_DOMAIN_CPU_REQUIRED
BACKWARD_DOMAIN_NO_REGIONS = PLAN7_BACKWARD_DOMAIN_NO_REGIONS
BACKWARD_DOMAIN_SIMPLE = PLAN7_BACKWARD_DOMAIN_SIMPLE
BACKWARD_DOMAIN_STATUS_OK = PLAN7_BACKWARD_DOMAIN_OK
BACKWARD_DOMAIN_STATUS_ERANGE = PLAN7_BACKWARD_DOMAIN_ERANGE
BACKWARD_DOMAIN_STATUS_ENORESULT = PLAN7_BACKWARD_DOMAIN_ENORESULT
BACKWARD_DOMAIN_STATUS_EMPTY = PLAN7_BACKWARD_DOMAIN_EMPTY
BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH = (
    PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_RESULT_HASH
)
BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH = (
    PLAN7_BACKWARD_DOMAIN_TEST_TAMPER_THRESHOLD_HASH
)
BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE = (
    PLAN7_BACKWARD_DOMAIN_TEST_FORCE_SIMPLE_OWN_SCALE
)
DOMAIN_RESCORE_RECORD_VERSION = PLAN7_DOMAIN_RESCORE_RECORD_VERSION
DOMAIN_RESCORE_RESULT_SIZE = sizeof(plan7_domain_rescore_result)
DOMAIN_RESCORE_TRACE_STEP_SIZE = sizeof(plan7_domain_rescore_trace_step)
DOMAIN_RESCORE_NULL2_COUNT = PLAN7_DOMAIN_RESCORE_NULL2_COUNT
DOMAIN_RESCORE_MAX_COMPACT_BYTES = PLAN7_DOMAIN_RESCORE_MAX_COMPACT_BYTES
DOMAIN_RESCORE_MAX_MATRIX_BYTES = PLAN7_DOMAIN_RESCORE_MAX_MATRIX_BYTES
DOMAIN_RESCORE_MAX_TRACE_BYTES = PLAN7_DOMAIN_RESCORE_MAX_TRACE_BYTES
DOMAIN_RESCORE_CPU_REQUIRED = PLAN7_DOMAIN_RESCORE_CPU_REQUIRED
DOMAIN_RESCORE_DEVICE_RESULT = PLAN7_DOMAIN_RESCORE_DEVICE_RESULT
DOMAIN_RESCORE_STATUS_OK = PLAN7_DOMAIN_RESCORE_OK
DOMAIN_RESCORE_STATUS_ERANGE = PLAN7_DOMAIN_RESCORE_ERANGE
DOMAIN_RESCORE_STATUS_ENORESULT = PLAN7_DOMAIN_RESCORE_ENORESULT
DOMAIN_RESCORE_STATUS_ECAP = PLAN7_DOMAIN_RESCORE_ECAP
DOMAIN_RESCORE_STATUS_EMPTY = PLAN7_DOMAIN_RESCORE_EMPTY
