# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, uint8_t, uint32_t, uint64_t
from libc.math cimport isfinite
from libcpp.vector cimport vector


cdef extern from "ssv_cuda.h" nogil:
    cdef enum plan7_ssv_status:
        PLAN7_SSV_OK
        PLAN7_SSV_ERANGE
        PLAN7_SSV_ENORESULT
        PLAN7_SSV_EMPTY

    ctypedef struct plan7_ssv_result:
        uint8_t xE
        uint8_t status
        uint8_t tjb
        int16_t numerator

    int plan7_cuda_device_count(char *error, size_t error_size)
    int plan7_tjb_for_length(float scale, uint64_t length)

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


cdef union float_bits:
    float value
    uint32_t bits


def device_count():
    cdef char error[512]
    cdef int count
    error[0] = 0
    with nogil:
        count = plan7_cuda_device_count(error, sizeof(error))
    if count < 0:
        raise RuntimeError(error.decode("utf-8", "replace"))
    return count


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
    cdef size_t sequence_count
    cdef vector[plan7_ssv_result] results
    cdef char error[512]
    cdef int status
    cdef size_t i
    cdef plan7_ssv_result result
    cdef float_bits score
    cdef object score_value
    cdef list output = []

    if offsets.shape[0] == 0:
        raise ValueError("offsets must contain an initial zero")
    sequence_count = <size_t> offsets.shape[0] - 1
    if score_stride < 1 or alphabet_size < 1 or not 1 <= model_length <= 100_000:
        raise ValueError("invalid score dimensions")
    if <size_t> striped_scores.shape[0] < <size_t> score_stride * alphabet_size:
        raise ValueError("striped score buffer is too short")
    if not 0 <= tbm <= 255 or not 0 <= tec <= 255:
        raise ValueError("transition costs must fit in uint8")
    if not 0 <= base <= 255 or not 0 <= bias <= 255:
        raise ValueError("profile byte constants must fit in uint8")
    if not isfinite(scale) or scale <= 0:
        raise ValueError("profile scale must be finite and positive")

    results.resize(sequence_count)
    error[0] = 0
    with nogil:
        status = plan7_ssv_filter_cuda(
            &striped_scores[0] if striped_scores.shape[0] else NULL,
            <size_t> striped_scores.shape[0],
            score_stride,
            model_length,
            alphabet_size,
            &residues[0] if residues.shape[0] else NULL,
            <size_t> residues.shape[0],
            &offsets[0],
            <size_t> offsets.shape[0],
            sequence_count,
            <uint8_t> tbm,
            <uint8_t> tec,
            <uint8_t> base,
            <uint8_t> bias,
            scale,
            results.data() if sequence_count else NULL,
            sequence_count,
            error,
            sizeof(error),
        )
    if status != 0:
        raise RuntimeError(error.decode("utf-8", "replace"))

    for i in range(sequence_count):
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


STATUS_OK = PLAN7_SSV_OK
STATUS_ERANGE = PLAN7_SSV_ERANGE
STATUS_ENORESULT = PLAN7_SSV_ENORESULT
STATUS_EMPTY = PLAN7_SSV_EMPTY
