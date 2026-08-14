# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, int32_t, uint8_t, uint32_t, uint64_t
from libc.math cimport isfinite
from libcpp.vector cimport vector


cdef extern from "ssv_cuda.h" nogil:
    cdef enum plan7_ssv_status:
        PLAN7_SSV_OK
        PLAN7_SSV_ERANGE
        PLAN7_SSV_ENORESULT
        PLAN7_SSV_EMPTY

    cdef enum plan7_f1_action:
        PLAN7_F1_CPU_REQUIRED
        PLAN7_F1_DEFINITE_REJECT

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

    int plan7_cuda_device_count(char *error, size_t error_size)
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
        const uint8_t *packed_striped_scores,
        size_t packed_score_count,
        const plan7_ssv_profile *profiles,
        size_t profile_count,
        plan7_ssv_result *profile_major_results,
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


cdef union float_bits:
    float value
    uint32_t bits


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


cdef class SequenceBatch:
    cdef plan7_ssv_sequence_batch *_batch
    cdef vector[plan7_ssv_result] _results
    cdef vector[plan7_ssv_result] _many_results
    cdef vector[plan7_ssv_profile] _profiles
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

    def close(self):
        cdef char error[512]
        cdef int status
        if self._batch != NULL:
            error[0] = 0
            with nogil:
                status = plan7_ssv_sequence_batch_destroy(
                    &self._batch, error, sizeof(error)
                )
            if status != 0:
                raise RuntimeError(error.decode("utf-8", "replace"))

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
        cdef size_t i
        cdef size_t profile_count = <size_t> score_offsets.shape[0]
        cdef size_t result_count
        cdef int status

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
        if self._sequence_count and profile_count > (<size_t> -1) / self._sequence_count:
            raise OverflowError("multi-profile result count overflows size_t")

        result_count = profile_count * self._sequence_count
        self._profiles.resize(profile_count)
        self._many_results.resize(result_count)
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
        cdef size_t result_index
        cdef int action
        cdef list candidates
        cdef list output = []

        profile_count = <size_t> score_offsets.shape[0]
        if (
            <size_t> m_mu.shape[0] != profile_count
            or <size_t> m_lambda.shape[0] != profile_count
        ):
            raise ValueError("e-value parameter lengths differ")
        profile_count = self._run_filter_many(
            packed_scores,
            score_offsets,
            score_counts,
            score_strides,
            model_lengths,
            constants,
            scales,
        )
        for profile_index in range(profile_count):
            candidates = []
            for sequence_index in range(self._sequence_count):
                result_index = profile_index * self._sequence_count + sequence_index
                action = plan7_ssv_f1_decision(
                    self._many_results[result_index].status,
                    self._many_results[result_index].numerator,
                    self._lengths[sequence_index],
                    scales[profile_index],
                    m_mu[profile_index],
                    m_lambda[profile_index],
                    f1,
                    NULL,
                )
                if action == PLAN7_F1_CPU_REQUIRED:
                    candidates.append(sequence_index)
            output.append(candidates)
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


STATUS_OK = PLAN7_SSV_OK
STATUS_ERANGE = PLAN7_SSV_ERANGE
STATUS_ENORESULT = PLAN7_SSV_ENORESULT
STATUS_EMPTY = PLAN7_SSV_EMPTY
F1_CPU_REQUIRED = PLAN7_F1_CPU_REQUIRED
F1_DEFINITE_REJECT = PLAN7_F1_DEFINITE_REJECT
