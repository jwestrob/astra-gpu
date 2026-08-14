#include "ssv_cuda.h"

#include <cuda_runtime.h>

extern "C" {
#include <easel.h>
#include <esl_gumbel.h>
}

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct plan7_ssv_sequence_batch {
  int device_ordinal;
  int alphabet_size;
  size_t sequence_count;
  uint64_t *host_lengths;
  float *host_null_scores;
  float *host_tjb_log_terms;
  uint8_t *host_tjb;
  size_t host_tjb_capacity;
  uint8_t *device_residues;
  uint64_t *device_offsets;
  uint8_t *device_tjb;
  size_t device_tjb_capacity;
  plan7_ssv_result *device_results;
  size_t device_result_capacity;
  uint8_t *device_scores;
  size_t device_score_capacity;
  plan7_ssv_profile *device_profiles;
  size_t device_profile_capacity;
  float cached_tjb_scale;
  int tjb_cache_valid;
};

static_assert(sizeof(plan7_ssv_result) == 6,
              "plan7_ssv_result ABI size changed");
static_assert(offsetof(plan7_ssv_result, numerator) == 4,
              "plan7_ssv_result ABI layout changed");
static_assert(sizeof(float) == 4, "plan7 profile ABI requires binary32 float");
static_assert(sizeof(plan7_ssv_profile) == 32,
              "plan7_ssv_profile ABI size changed");
static_assert(offsetof(plan7_ssv_profile, score_offset) == 0 &&
              offsetof(plan7_ssv_profile, score_count) == 8 &&
              offsetof(plan7_ssv_profile, score_stride) == 16 &&
              offsetof(plan7_ssv_profile, model_length) == 20 &&
              offsetof(plan7_ssv_profile, tbm) == 24 &&
              offsetof(plan7_ssv_profile, scale) == 28,
              "plan7_ssv_profile ABI layout changed");

namespace {

constexpr int kThreads = 256;
constexpr int kExtraScoreVectors = 17;
constexpr uint64_t kMaximumTargetLength = 100000;
constexpr float kEvparamUnset = -99999.0f;

__device__ __forceinline__ int
saturating_signed_subtract(int left, int right)
{
  int value = left - right;
  return value > INT8_MAX ? INT8_MAX : (value < INT8_MIN ? INT8_MIN : value);
}

template<bool CompactScores>
__device__ __forceinline__ void
ssv_filter_block(const uint8_t *scores,
                 int score_stride,
                 int model_length,
                 const uint8_t *residues,
                 const uint64_t *offsets,
                 size_t sequence,
                 uint8_t length_tjb,
                 uint8_t tbm,
                 uint8_t tec,
                 uint8_t base,
                 uint8_t bias,
                 plan7_ssv_result *result_out,
                 unsigned *maxima)
{
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  const int Q = max(2, (model_length + 15) / 16);
  unsigned local_maximum = 128;

  if (length == 0) {
    if (threadIdx.x == 0)
      *result_out = {128, PLAN7_SSV_EMPTY, length_tjb, 0, 0};
    return;
  }

  const int diagonal_count = model_length + length - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    int value = INT8_MIN;

    while (i < length && k < model_length) {
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      unsigned raw_cost;
      if constexpr (CompactScores) {
        raw_cost = scores[static_cast<size_t>(k) * score_stride + residue];
      } else {
        const int q = k % Q;
        const int lane = k / Q;
        raw_cost =
          scores[static_cast<size_t>(residue) * score_stride + 16 * q + lane];
      }
      const int cost = raw_cost < 128 ? static_cast<int>(raw_cost)
                                     : static_cast<int>(raw_cost) - 256;
      value = saturating_signed_subtract(value, cost);
      const unsigned raw_value = value < 0 ? static_cast<unsigned>(value + 256)
                                           : static_cast<unsigned>(value);
      local_maximum = max(local_maximum, raw_value);
      ++i;
      ++k;
    }
  }

  maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (int width = blockDim.x / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width)
      maxima[threadIdx.x] = max(maxima[threadIdx.x], maxima[threadIdx.x + width]);
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const unsigned raw_xE = maxima[0];
    plan7_ssv_result result = {
      static_cast<uint8_t>(raw_xE), PLAN7_SSV_OK,
      static_cast<uint8_t>(length_tjb), 0, 0
    };

    if (length_tjb + tbm + tec + bias >= 127) {
      result.status = PLAN7_SSV_ENORESULT;
    } else if (raw_xE >= 255U - bias) {
      result.status =
        static_cast<int>(base) - static_cast<int>(length_tjb) -
            static_cast<int>(tbm) < 128
          ? PLAN7_SSV_ENORESULT
          : PLAN7_SSV_ERANGE;
    } else {
      unsigned adjusted = raw_xE + base - length_tjb - tbm;
      adjusted -= 128;
      if (adjusted >= 255U - bias) {
        result.status = PLAN7_SSV_ERANGE;
      } else {
        const unsigned xJ = adjusted - tec;
        if (xJ > base) {
          result.status = PLAN7_SSV_ENORESULT;
        } else {
          result.numerator = static_cast<int16_t>(
            static_cast<int>(xJ) - static_cast<int>(length_tjb) -
            static_cast<int>(base));
        }
      }
    }
    *result_out = result;
  }
}

__global__ void
ssv_kernel(const uint8_t *scores,
           int score_stride,
           int model_length,
           const uint8_t *residues,
           const uint64_t *offsets,
           const uint8_t *tjb,
           uint8_t tbm,
           uint8_t tec,
           uint8_t base,
           uint8_t bias,
           plan7_ssv_result *results)
{
  __shared__ unsigned maxima[kThreads];
  const size_t sequence = static_cast<size_t>(blockIdx.x);
  ssv_filter_block<false>(scores, score_stride, model_length, residues, offsets,
                          sequence, tjb[sequence], tbm, tec, base, bias,
                          &results[sequence], maxima);
}

__global__ void
ssv_many_kernel(const uint8_t *packed_scores,
                const plan7_ssv_profile *profiles,
                size_t sequence_count,
                const uint8_t *residues,
                const uint64_t *offsets,
                const uint8_t *profile_major_tjb,
                plan7_ssv_result *profile_major_results)
{
  __shared__ unsigned maxima[kThreads];
  __shared__ plan7_ssv_profile profile_descriptor;
  const size_t sequence = static_cast<size_t>(blockIdx.x);
  const size_t profile = static_cast<size_t>(blockIdx.y);
  const size_t result_index = profile * sequence_count + sequence;

  if (threadIdx.x == 0) profile_descriptor = profiles[profile];
  __syncthreads();
  ssv_filter_block<true>(
    packed_scores + profile_descriptor.score_offset,
    profile_descriptor.score_stride,
    profile_descriptor.model_length,
    residues,
    offsets,
    sequence,
    profile_major_tjb[result_index],
    profile_descriptor.tbm,
    profile_descriptor.tec,
    profile_descriptor.base,
    profile_descriptor.bias,
    &profile_major_results[result_index],
    maxima);
}

void
set_error(char *error, size_t error_size, const char *message)
{
  if (error != nullptr && error_size != 0)
    snprintf(error, error_size, "%s", message);
}

void
set_cuda_error(char *error, size_t error_size, const char *operation,
               cudaError_t status)
{
  if (error != nullptr && error_size != 0)
    snprintf(error, error_size, "%s: %s", operation, cudaGetErrorString(status));
}

bool
checked_product(size_t left, size_t right, size_t *product)
{
  if (right != 0 && left > SIZE_MAX / right) return false;
  *product = left * right;
  return true;
}

bool
compute_null_score(uint64_t length, float *ret_null_score)
{
  float length_f;
  float p1;
  float null_score;

  if (length == 0 || length > kMaximumTargetLength ||
      ret_null_score == nullptr)
    return false;
  length_f = static_cast<float>(length);
  p1 = length_f / static_cast<float>(length + 1);
  null_score = static_cast<float>(
    static_cast<double>(length_f) * log(static_cast<double>(p1)) +
    log(1.0 - static_cast<double>(p1)));
  if (!isfinite(null_score)) return false;
  *ret_null_score = null_score;
  return true;
}

bool
compute_bit_score(int16_t numerator,
                  float null_score,
                  float scale,
                  float *ret_bit_score)
{
  float score;
  float delta;
  float bit_score;

  if (!isfinite(null_score) || !isfinite(scale) || scale <= 0.0f ||
      ret_bit_score == nullptr)
    return false;
  score = static_cast<float>(numerator);
  score /= scale;
  score -= 3.0;
  delta = score - null_score;
  bit_score = static_cast<float>(
    static_cast<double>(delta) / eslCONST_LOG2);
  if (!isfinite(score) || !isfinite(bit_score)) return false;
  *ret_bit_score = bit_score;
  return true;
}

float
ordered_to_float(uint32_t ordered)
{
  const uint32_t bits = (ordered & UINT32_C(0x80000000)) != 0
                          ? ordered ^ UINT32_C(0x80000000)
                          : ~ordered;
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

bool
valid_f1_parameters(float m_mu, float m_lambda, double f1)
{
  return isfinite(m_mu) && isfinite(m_lambda) && m_lambda > 0.0f &&
         m_mu != kEvparamUnset && m_lambda != kEvparamUnset && isfinite(f1) &&
         f1 >= 0.0 && f1 <= 1.0;
}

bool
f1_probability(float bit_score,
               float m_mu,
               float m_lambda,
               double *ret_probability)
{
  const double probability = esl_gumbel_surv(
    static_cast<double>(bit_score), static_cast<double>(m_mu),
    static_cast<double>(m_lambda));
  if (!isfinite(probability) || probability < 0.0 || probability > 1.0)
    return false;
  *ret_probability = probability;
  return true;
}

bool
uses_smallx_survivor_branch(float bit_score,
                            float m_mu,
                            float m_lambda,
                            bool *ret_uses_smallx)
{
  const double y = static_cast<double>(m_lambda) *
                   (static_cast<double>(bit_score) -
                    static_cast<double>(m_mu));
  const double tail_term = exp(-y);
  if (isnan(tail_term)) return false;
  *ret_uses_smallx = tail_term < eslSMALLX1;
  return true;
}

bool
f1_branch_is_nonmonotone(float m_mu,
                         float m_lambda,
                         double f1,
                         bool *ret_nonmonotone)
{
  constexpr uint32_t kLowestFinite = UINT32_C(0x00800000);
  constexpr uint32_t kHighestFinite = UINT32_C(0xff7fffff);
  const double lowest_jump_probability = 1.0 - exp(-eslSMALLX1);
  bool low_uses_smallx;
  bool high_uses_smallx;
  uint32_t low = kLowestFinite;
  uint32_t high = kHighestFinite;

  *ret_nonmonotone = false;
  if (f1 < lowest_jump_probability || f1 >= eslSMALLX1) return true;
  if (!uses_smallx_survivor_branch(-FLT_MAX, m_mu, m_lambda,
                                   &low_uses_smallx) ||
      !uses_smallx_survivor_branch(FLT_MAX, m_mu, m_lambda,
                                   &high_uses_smallx))
    return false;
  if (low_uses_smallx == high_uses_smallx) return true;
  if (low_uses_smallx || !high_uses_smallx) return false;

  while (high - low > 1) {
    const uint32_t middle = low + (high - low) / 2;
    bool uses_smallx;
    if (!uses_smallx_survivor_branch(
          ordered_to_float(middle), m_mu, m_lambda, &uses_smallx))
      return false;
    if (uses_smallx)
      high = middle;
    else
      low = middle;
  }

  double before_probability;
  double after_probability;
  if (!f1_probability(ordered_to_float(low), m_mu, m_lambda,
                      &before_probability) ||
      !f1_probability(ordered_to_float(high), m_mu, m_lambda,
                      &after_probability))
    return false;
  *ret_nonmonotone =
    before_probability <= f1 && after_probability > f1;
  return true;
}

int
derive_f1_cutoff(float m_mu,
                 float m_lambda,
                 double f1,
                 float *ret_bit_score)
{
  constexpr uint32_t kLowestFinite = UINT32_C(0x00800000);
  constexpr uint32_t kHighestFinite = UINT32_C(0xff7fffff);
  double low_probability;
  double high_probability;
  bool branch_is_nonmonotone;
  uint32_t low = kLowestFinite;
  uint32_t high = kHighestFinite;

  if (ret_bit_score != nullptr) *ret_bit_score = NAN;
  if (!valid_f1_parameters(m_mu, m_lambda, f1) ||
      !f1_probability(-FLT_MAX, m_mu, m_lambda, &low_probability) ||
      !f1_probability(FLT_MAX, m_mu, m_lambda, &high_probability) ||
      !f1_branch_is_nonmonotone(m_mu, m_lambda, f1,
                                &branch_is_nonmonotone) ||
      branch_is_nonmonotone ||
      low_probability < high_probability)
    return PLAN7_F1_CUTOFF_INVALID;

  /* The survivor is nonincreasing for lambda > 0. */
  if (low_probability <= f1) return PLAN7_F1_CUTOFF_ALWAYS_CPU;
  if (high_probability > f1) return PLAN7_F1_CUTOFF_ALWAYS_REJECT;

  while (high - low > 1) {
    const uint32_t middle = low + (high - low) / 2;
    double probability;
    if (!f1_probability(ordered_to_float(middle), m_mu, m_lambda,
                        &probability))
      return PLAN7_F1_CUTOFF_INVALID;
    if (probability > low_probability || probability < high_probability)
      return PLAN7_F1_CUTOFF_INVALID;
    if (probability > f1) {
      low = middle;
      low_probability = probability;
    } else {
      high = middle;
      high_probability = probability;
    }
  }

  const float cutoff = ordered_to_float(high);
  const float predecessor = ordered_to_float(high - 1);
  double cutoff_probability;
  double predecessor_probability;
  if (!isfinite(cutoff) ||
      !f1_probability(cutoff, m_mu, m_lambda, &cutoff_probability) ||
      !f1_probability(predecessor, m_mu, m_lambda,
                      &predecessor_probability) ||
      predecessor_probability < cutoff_probability ||
      predecessor_probability <= f1 || cutoff_probability > f1)
    return PLAN7_F1_CUTOFF_INVALID;
  if (ret_bit_score != nullptr) *ret_bit_score = cutoff;
  return PLAN7_F1_CUTOFF_SCORE;
}

int
cutoff_decision_with_null_score(uint8_t status,
                                int16_t numerator,
                                float null_score,
                                float scale,
                                int cutoff_mode,
                                float cutoff_bit_score)
{
  float bit_score;
  if (status != PLAN7_SSV_OK ||
      !compute_bit_score(numerator, null_score, scale, &bit_score))
    return PLAN7_F1_CPU_REQUIRED;

  switch (cutoff_mode) {
    case PLAN7_F1_CUTOFF_SCORE:
      if (!isfinite(cutoff_bit_score)) return PLAN7_F1_CPU_REQUIRED;
      return bit_score < cutoff_bit_score ? PLAN7_F1_DEFINITE_REJECT
                                          : PLAN7_F1_CPU_REQUIRED;
    case PLAN7_F1_CUTOFF_ALWAYS_REJECT:
      return PLAN7_F1_DEFINITE_REJECT;
    case PLAN7_F1_CUTOFF_ALWAYS_CPU:
    default:
      return PLAN7_F1_CPU_REQUIRED;
  }
}

template<typename T>
int
grow_device_buffer(T **buffer,
                   size_t *capacity,
                   size_t required_bytes,
                   const char *malloc_name,
                   const char *free_name,
                   char *error,
                   size_t error_size)
{
  T *replacement = nullptr;
  cudaError_t status;

  if (required_bytes <= *capacity) return 0;
  status = cudaMalloc(&replacement, required_bytes);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, malloc_name, status);
    return -1;
  }
  status = cudaFree(*buffer);
  if (status != cudaSuccess) {
    cudaFree(replacement);
    set_cuda_error(error, error_size, free_name, status);
    return -1;
  }
  *buffer = replacement;
  *capacity = required_bytes;
  return 0;
}

bool
validate_compact_profile(const plan7_ssv_profile *profile,
                         int alphabet_size,
                         size_t packed_score_count,
                         size_t expected_offset,
                         char *error,
                         size_t error_size)
{
  size_t expected_score_count;

  if (profile->model_length < 1 || profile->model_length > 100000 ||
      profile->score_stride < 1 || !isfinite(profile->scale) ||
      profile->scale <= 0.0f) {
    set_error(error, error_size, "invalid packed profile dimensions or scale");
    return false;
  }
  if (profile->score_stride != alphabet_size ||
      !checked_product(static_cast<size_t>(profile->model_length),
                       static_cast<size_t>(alphabet_size),
                       &expected_score_count) ||
      profile->score_count != expected_score_count ||
      profile->score_offset != expected_offset ||
      expected_offset > packed_score_count ||
      profile->score_count >
        static_cast<uint64_t>(packed_score_count - expected_offset)) {
    set_error(error, error_size, "invalid compact profile score range");
    return false;
  }
  return true;
}

uint8_t
compute_tjb_from_log_term(float scale, float log_term)
{
  const float cost = -1.0f * roundf(scale * log_term);
  return cost > 255.0f ? 255 : static_cast<uint8_t>(cost);
}

float
compute_tjb_log_term(uint64_t length)
{
  return logf(3.0f / static_cast<float>(length + 3));
}

uint8_t
compute_tjb(float scale, uint64_t length)
{
  return compute_tjb_from_log_term(scale, compute_tjb_log_term(length));
}

int
destroy_sequence_batch(plan7_ssv_sequence_batch *batch,
                       char *error,
                       size_t error_size)
{
  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool device_ready = true;
  bool restore_device = false;

  if (batch == nullptr) return 0;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess && original_device != batch->device_ordinal) {
    status = cudaSetDevice(batch->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice(batch->device_ordinal);
    if (status != cudaSuccess) {
      first_error = status;
      device_ready = false;
    }
  }

#define CUDA_FREE(pointer)                                                    \
  do {                                                                        \
    if (device_ready) {                                                       \
      status = cudaFree(pointer);                                             \
      if (first_error == cudaSuccess && status != cudaSuccess)                \
        first_error = status;                                                 \
    }                                                                         \
  } while (0)

  CUDA_FREE(batch->device_scores);
  CUDA_FREE(batch->device_profiles);
  CUDA_FREE(batch->device_results);
  CUDA_FREE(batch->device_tjb);
  CUDA_FREE(batch->device_offsets);
  CUDA_FREE(batch->device_residues);
#undef CUDA_FREE
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (first_error == cudaSuccess && status != cudaSuccess)
      first_error = status;
  }
  free(batch->host_tjb);
  free(batch->host_tjb_log_terms);
  free(batch->host_null_scores);
  free(batch->host_lengths);
  free(batch);
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy CUDA sequence batch",
                   first_error);
    return -1;
  }
  return 0;
}

}  // namespace

extern "C" int
plan7_cuda_device_count(char *error, size_t error_size)
{
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDeviceCount", status);
    return -1;
  }
  return count;
}

extern "C" int
plan7_tjb_for_length(float scale, uint64_t length)
{
  if (!isfinite(scale) || scale <= 0.0f || length > kMaximumTargetLength)
    return -1;
  return compute_tjb(scale, length);
}

extern "C" int
plan7_ssv_f1_decision(uint8_t status,
                      int16_t numerator,
                      uint64_t length,
                      float scale,
                      float m_mu,
                      float m_lambda,
                      double f1,
                      double *ret_p)
{
  float null_score;
  float bit_score;
  double probability;

  if (ret_p != nullptr) *ret_p = NAN;
  if (status != PLAN7_SSV_OK || length == 0 ||
      length > kMaximumTargetLength || !isfinite(scale) || scale <= 0.0f ||
      !valid_f1_parameters(m_mu, m_lambda, f1))
    return PLAN7_F1_CPU_REQUIRED;

  /* Preserve HMMER 3.4's float/double evaluation order exactly. */
  if (!compute_null_score(length, &null_score) ||
      !compute_bit_score(numerator, null_score, scale, &bit_score) ||
      !f1_probability(bit_score, m_mu, m_lambda, &probability))
    return PLAN7_F1_CPU_REQUIRED;
  if (ret_p != nullptr) *ret_p = probability;
  return probability > f1 ? PLAN7_F1_DEFINITE_REJECT
                          : PLAN7_F1_CPU_REQUIRED;
}

extern "C" int
plan7_ssv_f1_cutoff(float m_mu,
                    float m_lambda,
                    double f1,
                    float *ret_bit_score)
{
  return derive_f1_cutoff(m_mu, m_lambda, f1, ret_bit_score);
}

extern "C" int
plan7_ssv_f1_cutoff_decision(uint8_t status,
                             int16_t numerator,
                             uint64_t length,
                             float scale,
                             int cutoff_mode,
                             float cutoff_bit_score)
{
  float null_score;
  if (!compute_null_score(length, &null_score))
    return PLAN7_F1_CPU_REQUIRED;
  return cutoff_decision_with_null_score(status, numerator, null_score, scale,
                                         cutoff_mode, cutoff_bit_score);
}

extern "C" int
plan7_ssv_sequence_batch_create(const uint8_t *residues,
                                size_t residue_count,
                                const uint64_t *offsets,
                                size_t offset_count,
                                int alphabet_size,
                                plan7_ssv_sequence_batch **batch_out,
                                char *error,
                                size_t error_size)
{
  plan7_ssv_sequence_batch *batch = nullptr;
  cudaError_t status;
  size_t sequence_count;
  size_t offset_bytes;
  size_t result_bytes;
  int rc = -1;
  int device_ordinal;

  if (batch_out == nullptr) {
    set_error(error, error_size, "sequence batch output is null");
    return -1;
  }
  *batch_out = nullptr;
  status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (alphabet_size < 1 || offset_count == 0 || offsets == nullptr ||
      (residue_count != 0 && residues == nullptr)) {
    set_error(error, error_size, "invalid sequence batch buffers");
    return -1;
  }
  sequence_count = offset_count - 1;
  if (sequence_count > INT_MAX) {
    set_error(error, error_size, "too many sequences in CUDA batch");
    return -1;
  }
  if (offsets[0] != 0 || offsets[sequence_count] != residue_count) {
    set_error(error, error_size, "invalid sequence offsets");
    return -1;
  }
  for (size_t i = 0; i < sequence_count; ++i) {
    if (offsets[i] > offsets[i + 1] ||
        offsets[i + 1] - offsets[i] > kMaximumTargetLength) {
      set_error(error, error_size, "invalid target length or offsets");
      return -1;
    }
  }
  for (size_t i = 0; i < residue_count; ++i) {
    if (residues[i] >= alphabet_size) {
      set_error(error, error_size, "digital residue is outside the alphabet");
      return -1;
    }
  }
  if (!checked_product(offset_count, sizeof(*offsets), &offset_bytes) ||
      !checked_product(sequence_count, sizeof(plan7_ssv_result),
                       &result_bytes)) {
    set_error(error, error_size, "sequence batch size overflow");
    return -1;
  }

  batch = static_cast<plan7_ssv_sequence_batch *>(calloc(1, sizeof(*batch)));
  if (batch == nullptr) {
    set_error(error, error_size, "host sequence batch allocation failed");
    return -1;
  }
  batch->alphabet_size = alphabet_size;
  batch->device_ordinal = device_ordinal;
  batch->sequence_count = sequence_count;
  if (sequence_count == 0) {
    *batch_out = batch;
    return 0;
  }
  batch->host_lengths =
    static_cast<uint64_t *>(malloc(sequence_count * sizeof(uint64_t)));
  batch->host_null_scores =
    static_cast<float *>(malloc(sequence_count * sizeof(float)));
  batch->host_tjb_log_terms =
    static_cast<float *>(malloc(sequence_count * sizeof(float)));
  batch->host_tjb = static_cast<uint8_t *>(malloc(sequence_count));
  if (batch->host_lengths == nullptr || batch->host_null_scores == nullptr ||
      batch->host_tjb_log_terms == nullptr || batch->host_tjb == nullptr) {
    set_error(error, error_size, "host sequence metadata allocation failed");
    goto cleanup;
  }
  batch->host_tjb_capacity = sequence_count;
  for (size_t i = 0; i < sequence_count; ++i) {
    batch->host_lengths[i] = offsets[i + 1] - offsets[i];
    if (!compute_null_score(batch->host_lengths[i],
                            &batch->host_null_scores[i]))
      batch->host_null_scores[i] = NAN;
    batch->host_tjb_log_terms[i] =
      compute_tjb_log_term(batch->host_lengths[i]);
  }

#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      goto cleanup;                                                           \
    }                                                                         \
  } while (0)

  if (residue_count != 0)
    CUDA_TRY(cudaMalloc(&batch->device_residues, residue_count));
  CUDA_TRY(cudaMalloc(&batch->device_offsets, offset_bytes));
  CUDA_TRY(cudaMalloc(&batch->device_tjb, sequence_count));
  batch->device_tjb_capacity = sequence_count;
  CUDA_TRY(cudaMalloc(&batch->device_results, result_bytes));
  batch->device_result_capacity = result_bytes;
  if (residue_count != 0)
    CUDA_TRY(cudaMemcpy(batch->device_residues, residues, residue_count,
                        cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(batch->device_offsets, offsets, offset_bytes,
                      cudaMemcpyHostToDevice));
  *batch_out = batch;
  rc = 0;

cleanup:
  if (rc != 0) destroy_sequence_batch(batch, nullptr, 0);
#undef CUDA_TRY
  return rc;
}

extern "C" int
plan7_ssv_sequence_batch_destroy(plan7_ssv_sequence_batch **batch_out,
                                 char *error,
                                 size_t error_size)
{
  plan7_ssv_sequence_batch *batch;

  if (batch_out == nullptr) {
    set_error(error, error_size, "sequence batch pointer is null");
    return -1;
  }
  batch = *batch_out;
  *batch_out = nullptr;
  return destroy_sequence_batch(batch, error, error_size);
}

extern "C" int
plan7_ssv_sequence_batch_filter(plan7_ssv_sequence_batch *batch,
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
                                size_t error_size)
{
  uint8_t *new_device_scores = nullptr;
  cudaError_t status;
  size_t score_bytes;
  int current_device;

  if (batch == nullptr || model_length < 1 || model_length > 100000 ||
      alphabet_size < 1 || alphabet_size != batch->alphabet_size) {
    set_error(error, error_size, "invalid SSV batch dimensions");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  const int Q = model_length < 17 ? 2 : (model_length + 15) / 16;
  const int minimum_stride = 16 * (Q + kExtraScoreVectors);
  if (score_stride < minimum_stride || striped_scores == nullptr ||
      !checked_product(static_cast<size_t>(score_stride),
                       static_cast<size_t>(alphabet_size), &score_bytes) ||
      striped_score_count < score_bytes) {
    set_error(error, error_size, "invalid striped score buffer");
    return -1;
  }
  if (!isfinite(scale) || scale <= 0.0f ||
      (batch->sequence_count != 0 &&
       (results == nullptr || result_count < batch->sequence_count))) {
    set_error(error, error_size, "invalid SSV result buffer or scale");
    return -1;
  }
  if (batch->sequence_count == 0) return 0;

#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  if (score_bytes > batch->device_score_capacity) {
    CUDA_TRY(cudaMalloc(&new_device_scores, score_bytes));
    status = cudaFree(batch->device_scores);
    if (status != cudaSuccess) {
      cudaFree(new_device_scores);
      set_cuda_error(error, error_size, "cudaFree(device_scores)", status);
      return -1;
    }
    batch->device_scores = new_device_scores;
    batch->device_score_capacity = score_bytes;
  }
  CUDA_TRY(cudaMemcpy(batch->device_scores, striped_scores, score_bytes,
                      cudaMemcpyHostToDevice));
  if (!batch->tjb_cache_valid || scale != batch->cached_tjb_scale) {
    for (size_t i = 0; i < batch->sequence_count; ++i)
      batch->host_tjb[i] =
        compute_tjb_from_log_term(scale, batch->host_tjb_log_terms[i]);
    batch->tjb_cache_valid = 0;
    CUDA_TRY(cudaMemcpy(batch->device_tjb, batch->host_tjb,
                        batch->sequence_count, cudaMemcpyHostToDevice));
    batch->cached_tjb_scale = scale;
    batch->tjb_cache_valid = 1;
  }

  ssv_kernel<<<static_cast<unsigned>(batch->sequence_count), kThreads>>>(
    batch->device_scores, score_stride, model_length, batch->device_residues,
    batch->device_offsets, batch->device_tjb, tbm, tec, base, bias,
    batch->device_results);
  CUDA_TRY(cudaGetLastError());
  CUDA_TRY(cudaMemcpy(results, batch->device_results,
                      batch->sequence_count * sizeof(*results),
                      cudaMemcpyDeviceToHost));
#undef CUDA_TRY
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_filter_many(
  plan7_ssv_sequence_batch *batch,
  const uint8_t *packed_scores,
  size_t packed_score_count,
  const plan7_ssv_profile *profiles,
  size_t profile_count,
  plan7_ssv_result *profile_major_results,
  size_t result_count,
  char *error,
  size_t error_size)
{
  cudaError_t status;
  size_t cell_count;
  size_t result_bytes;
  size_t profile_bytes;
  uint8_t *new_host_tjb;
  int current_device;
  int maximum_grid_x;
  int maximum_grid_y;
  size_t expected_score_offset = 0;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != batch->device_ordinal) {
    set_error(error, error_size,
              "CUDA sequence batch belongs to a different device");
    return -1;
  }
  if (profile_count == 0) return 0;

  status = cudaDeviceGetAttribute(
    &maximum_grid_x, cudaDevAttrMaxGridDimX, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid x)", status);
    return -1;
  }
  status = cudaDeviceGetAttribute(
    &maximum_grid_y, cudaDevAttrMaxGridDimY, current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size,
                   "cudaDeviceGetAttribute(maximum grid y)", status);
    return -1;
  }
  if (batch->sequence_count > static_cast<size_t>(maximum_grid_x) ||
      profile_count > static_cast<size_t>(maximum_grid_y)) {
    set_error(error, error_size, "multi-profile CUDA grid is too large");
    return -1;
  }
  if (packed_scores == nullptr || profiles == nullptr) {
    set_error(error, error_size, "packed profile buffers are null");
    return -1;
  }
  for (size_t profile = 0; profile < profile_count; ++profile) {
    if (!validate_compact_profile(
          &profiles[profile], batch->alphabet_size, packed_score_count,
          expected_score_offset, error, error_size))
      return -1;
    expected_score_offset += static_cast<size_t>(profiles[profile].score_count);
  }
  if (expected_score_offset != packed_score_count) {
    set_error(error, error_size, "compact profile scores have trailing bytes");
    return -1;
  }

  if (!checked_product(profile_count, batch->sequence_count, &cell_count) ||
      !checked_product(cell_count, sizeof(plan7_ssv_result), &result_bytes) ||
      !checked_product(profile_count, sizeof(plan7_ssv_profile),
                       &profile_bytes)) {
    set_error(error, error_size, "multi-profile batch size overflow");
    return -1;
  }
  if (cell_count != 0 &&
      (profile_major_results == nullptr || result_count < cell_count)) {
    set_error(error, error_size, "multi-profile result buffer is too short");
    return -1;
  }
  if (batch->sequence_count == 0) return 0;

  if (cell_count > batch->host_tjb_capacity) {
    new_host_tjb = static_cast<uint8_t *>(
      realloc(batch->host_tjb, cell_count));
    if (new_host_tjb == nullptr) {
      set_error(error, error_size,
                "host multi-profile transition allocation failed");
      return -1;
    }
    batch->host_tjb = new_host_tjb;
    batch->host_tjb_capacity = cell_count;
  }
  if (grow_device_buffer(&batch->device_scores,
                         &batch->device_score_capacity,
                         packed_score_count,
                         "cudaMalloc(packed profile scores)",
                         "cudaFree(packed profile scores)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_profiles,
                         &batch->device_profile_capacity,
                         profile_bytes,
                         "cudaMalloc(profile descriptors)",
                         "cudaFree(profile descriptors)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_tjb,
                         &batch->device_tjb_capacity,
                         cell_count,
                         "cudaMalloc(profile transitions)",
                         "cudaFree(profile transitions)",
                         error,
                         error_size) != 0 ||
      grow_device_buffer(&batch->device_results,
                         &batch->device_result_capacity,
                         result_bytes,
                         "cudaMalloc(profile results)",
                         "cudaFree(profile results)",
                         error,
                         error_size) != 0)
    return -1;

#define CUDA_TRY_MANY(call)                                                   \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  CUDA_TRY_MANY(cudaMemcpy(batch->device_scores,
                           packed_scores,
                           packed_score_count,
                           cudaMemcpyHostToDevice));
  CUDA_TRY_MANY(cudaMemcpy(batch->device_profiles,
                           profiles,
                           profile_bytes,
                           cudaMemcpyHostToDevice));
  for (size_t profile = 0; profile < profile_count; ++profile) {
    const size_t row = profile * batch->sequence_count;
    if (profile != 0 && profiles[profile].scale == profiles[profile - 1].scale) {
      memcpy(batch->host_tjb + row,
             batch->host_tjb + row - batch->sequence_count,
             batch->sequence_count);
    } else {
      for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence)
        batch->host_tjb[row + sequence] = compute_tjb_from_log_term(
          profiles[profile].scale, batch->host_tjb_log_terms[sequence]);
    }
  }
  batch->tjb_cache_valid = 0;
  CUDA_TRY_MANY(cudaMemcpy(batch->device_tjb,
                           batch->host_tjb,
                           cell_count,
                           cudaMemcpyHostToDevice));

  const dim3 grid(static_cast<unsigned>(batch->sequence_count),
                  static_cast<unsigned>(profile_count));
  ssv_many_kernel<<<grid, kThreads>>>(
    batch->device_scores,
    batch->device_profiles,
    batch->sequence_count,
    batch->device_residues,
    batch->device_offsets,
    batch->device_tjb,
    batch->device_results);
  CUDA_TRY_MANY(cudaGetLastError());
  CUDA_TRY_MANY(cudaMemcpy(profile_major_results,
                           batch->device_results,
                           result_bytes,
                           cudaMemcpyDeviceToHost));
#undef CUDA_TRY_MANY
  return 0;
}

extern "C" int
plan7_ssv_sequence_batch_f1_candidates_many(
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
  size_t error_size)
{
  size_t cell_count;

  if (batch == nullptr) {
    set_error(error, error_size, "sequence batch is null");
    return -1;
  }
  if (profile_count == 0) return 0;
  if (scales == nullptr || m_mu == nullptr || m_lambda == nullptr ||
      candidate_counts == nullptr) {
    set_error(error, error_size, "F1 profile buffers are null");
    return -1;
  }
  if ((candidate_offsets == nullptr) != (candidate_indices == nullptr)) {
    set_error(error, error_size, "F1 candidate output buffers differ");
    return -1;
  }
  if (!checked_product(profile_count, batch->sequence_count, &cell_count)) {
    set_error(error, error_size, "F1 candidate batch size overflow");
    return -1;
  }
  if (cell_count != 0 &&
      (profile_major_results == nullptr || result_count < cell_count ||
       batch->host_lengths == nullptr || batch->host_null_scores == nullptr)) {
    set_error(error, error_size, "F1 candidate buffer is too short");
    return -1;
  }
  if (candidate_indices != nullptr) {
    size_t expected_offset = 0;
    for (size_t profile = 0; profile < profile_count; ++profile) {
      if (candidate_offsets[profile] != expected_offset ||
          expected_offset > candidate_index_count ||
          candidate_counts[profile] > candidate_index_count - expected_offset) {
        set_error(error, error_size, "invalid F1 candidate offsets");
        return -1;
      }
      expected_offset += candidate_counts[profile];
    }
    if (expected_offset != candidate_index_count) {
      set_error(error, error_size, "invalid F1 candidate count");
      return -1;
    }
  }

  for (size_t profile = 0; profile < profile_count; ++profile) {
    float cutoff = NAN;
    const int cutoff_mode =
      derive_f1_cutoff(m_mu[profile], m_lambda[profile], f1, &cutoff);
    size_t candidate_count = 0;
    const size_t row = profile * batch->sequence_count;
    const size_t expected_candidate_count = candidate_indices != nullptr
                                              ? candidate_counts[profile]
                                              : 0;

    for (size_t sequence = 0; sequence < batch->sequence_count; ++sequence) {
      const plan7_ssv_result result = profile_major_results[row + sequence];
      int action;
      if (cutoff_mode == PLAN7_F1_CUTOFF_INVALID) {
        action = plan7_ssv_f1_decision(
          result.status, result.numerator, batch->host_lengths[sequence],
          scales[profile], m_mu[profile], m_lambda[profile], f1, nullptr);
      } else {
        action = cutoff_decision_with_null_score(
          result.status, result.numerator, batch->host_null_scores[sequence],
          scales[profile], cutoff_mode, cutoff);
      }
      if (action == PLAN7_F1_CPU_REQUIRED) {
        if (candidate_indices != nullptr) {
          if (candidate_count >= expected_candidate_count) {
            set_error(error, error_size, "F1 candidate count changed");
            return -1;
          }
          candidate_indices[candidate_offsets[profile] + candidate_count] =
            static_cast<uint32_t>(sequence);
        }
        ++candidate_count;
      }
    }
    if (candidate_indices != nullptr &&
        candidate_count != expected_candidate_count) {
      set_error(error, error_size, "F1 candidate count changed");
      return -1;
    }
    candidate_counts[profile] = candidate_count;
  }
  return 0;
}

extern "C" int
plan7_ssv_filter_cuda(const uint8_t *striped_scores,
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
                      size_t error_size)
{
  plan7_ssv_sequence_batch *batch = nullptr;
  int rc;
  int destroy_rc;

  if (offset_count != sequence_count + 1) {
    set_error(error, error_size, "invalid sequence offsets");
    return -1;
  }
  rc = plan7_ssv_sequence_batch_create(
    residues, residue_count, offsets, offset_count, alphabet_size, &batch,
    error, error_size);
  if (rc == 0)
    rc = plan7_ssv_sequence_batch_filter(
      batch, striped_scores, striped_score_count, score_stride, model_length,
      alphabet_size, tbm, tec, base, bias, scale, results, result_count, error,
      error_size);
  destroy_rc = plan7_ssv_sequence_batch_destroy(
    &batch, rc == 0 ? error : nullptr, rc == 0 ? error_size : 0);
  if (rc == 0 && destroy_rc != 0) rc = -1;
  return rc;
}
