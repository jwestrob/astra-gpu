# cython: language_level=3, boundscheck=False, wraparound=False

from libc.stddef cimport size_t
from libc.stdint cimport int16_t, int32_t, uint8_t, uint32_t, uint64_t
from libc.math cimport isfinite
from libc.string cimport memcpy
from cpython.array cimport array as carray, clone
from libcpp.vector cimport vector

import array as _array


cdef carray _UINT32_ARRAY_TEMPLATE = _array.array("I")
cdef carray _UINT64_ARRAY_TEMPLATE = _array.array("Q")


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


cdef class SequenceBatch:
    cdef plan7_ssv_sequence_batch *_batch
    cdef vector[plan7_ssv_result] _results
    cdef vector[plan7_ssv_result] _many_results
    cdef vector[plan7_ssv_profile] _profiles
    cdef vector[uint32_t] _candidate_words
    cdef vector[uint32_t] _candidate_indices
    cdef vector[size_t] _candidate_offsets
    cdef vector[size_t] _candidate_counts
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
