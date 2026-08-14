# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, int32_t, uintptr_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.math cimport isfinite
from libc.string cimport memcpy
from cpython.array cimport array as carray, clone
from libcpp.vector cimport vector
from pyhmmer.plan7 cimport OptimizedProfile

import array as _array
import pyhmmer as _pyhmmer

from . import _abi as _abi_module


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
    static inline unsigned plan7_popcount_u32(uint32_t value) {
      return (unsigned) __builtin_popcount(value);
    }
    static inline unsigned plan7_ctz_u32(uint32_t value) {
      return (unsigned) __builtin_ctz(value);
    }
    """
    unsigned plan7_popcount_u32(uint32_t value)
    unsigned plan7_ctz_u32(uint32_t value)


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

    ctypedef struct plan7_forward_database:
        pass

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


cdef union float_bits:
    float value
    uint32_t bits


def bias_environment_attested():
    cdef char reason[512]
    cdef int attested
    reason[0] = 0
    with nogil:
        attested = plan7_bias_environment_attested(reason, sizeof(reason))
    return bool(attested), reason.decode("utf-8", "replace")


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
    """Device-resident exact Forward profiles for F3 classification.

    Operations must not overlap each other or ``close``. Concurrent ``close``
    calls are safe.
    """

    cdef plan7_forward_database *_database
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
            raise RuntimeError("Forward profiles are already initialized")
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
        if offsets.shape[0] == 0:
            raise ValueError("offsets must contain an initial zero")
        if alphabet_size < 1:
            raise ValueError("alphabet size must be positive")
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
        cdef size_t special_count
        cdef size_t special_bytes
        cdef size_t profile_index
        cdef tuple owners = tuple(source_profiles)
        cdef vector[uintptr_t] source_pointers
        cdef OptimizedProfile source_profile
        cdef object value
        cdef plan7_forward_output *output = NULL
        cdef const plan7_forward_result *native_results
        cdef const uint64_t *native_offsets
        cdef const float *native_specials
        cdef const plan7_forward_statistics *native_statistics
        cdef bytearray records
        cdef uint8_t[::1] record_view
        cdef carray offsets
        cdef carray specials
        cdef dict statistics

        if self._batch == NULL:
            raise RuntimeError("sequence batch is closed")
        if forward_profiles._database == NULL:
            raise RuntimeError("Forward profiles are closed")
        if candidate_offsets.shape[0] == 0:
            raise ValueError("candidate offsets must contain an initial zero")
        profile_count = <size_t> candidate_offsets.shape[0] - 1
        if <size_t> filter_scores.shape[0] != candidate_count:
            raise ValueError("Forward candidate score lengths differ")
        if len(owners) != profile_count:
            raise ValueError("source and Forward profile counts differ")
        if plan7_forward_database_profile_count(
            forward_profiles._database
        ) != profile_count:
            raise ValueError("Forward profile and candidate row counts differ")
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
            source_pointers.data() if profile_count else NULL,
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
            records = bytearray(result_bytes)
            native_results = plan7_forward_output_results(output)
            if result_bytes:
                if native_results == NULL:
                    raise RuntimeError("Forward result storage is null")
                record_view = records
                memcpy(&record_view[0], native_results, result_bytes)

            native_offsets = plan7_forward_output_special_offsets(output)
            if native_offsets == NULL:
                raise RuntimeError("Forward special offsets are null")
            offsets = clone(_UINT64_ARRAY_TEMPLATE, result_count + 1, False)
            memcpy(
                offsets.data.as_ulonglongs,
                native_offsets,
                (result_count + 1) * sizeof(uint64_t),
            )

            special_count = plan7_forward_output_special_count(output)
            if special_count > (<size_t> -1) // sizeof(float):
                raise OverflowError("Forward special matrix size overflows size_t")
            special_bytes = special_count * sizeof(float)
            specials = clone(_FLOAT_ARRAY_TEMPLATE, special_count, False)
            native_specials = plan7_forward_output_specials(output)
            if special_bytes:
                if native_specials == NULL:
                    raise RuntimeError("Forward special matrix is null")
                memcpy(
                    specials.data.as_floats,
                    native_specials,
                    special_bytes,
                )

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
            return records, offsets, specials, statistics
        finally:
            if output != NULL:
                plan7_forward_output_destroy(&output, NULL, 0)

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
