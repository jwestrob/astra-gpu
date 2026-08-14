#include "ssv_cuda.h"

#include <cuda_runtime.h>

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace {

constexpr int kThreads = 256;
constexpr int kExtraScoreVectors = 17;
constexpr uint64_t kMaximumTargetLength = 100000;

__device__ __forceinline__ int
saturating_signed_subtract(int left, int right)
{
  int value = left - right;
  return value > INT8_MAX ? INT8_MAX : (value < INT8_MIN ? INT8_MIN : value);
}

__global__ void
ssv_kernel(const uint8_t *scores,
           int score_stride,
           int model_length,
           int alphabet_size,
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
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  const int Q = max(2, (model_length + 15) / 16);
  unsigned local_maximum = 128;

  if (length == 0) {
    if (threadIdx.x == 0)
      results[sequence] = {128, PLAN7_SSV_EMPTY, tjb[sequence], 0};
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
      const int q = k % Q;
      const int lane = k / Q;
      const unsigned raw_cost =
        scores[static_cast<size_t>(residue) * score_stride + 16 * q + lane];
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
    const unsigned length_tjb = tjb[sequence];
    plan7_ssv_result result = {
      static_cast<uint8_t>(raw_xE), PLAN7_SSV_OK,
      static_cast<uint8_t>(length_tjb), 0
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
    results[sequence] = result;
  }
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

uint8_t
compute_tjb(float scale, uint64_t length)
{
  float cost = -1.0f * roundf(scale * logf(3.0f / (float) (length + 3)));
  return cost > 255.0f ? 255 : static_cast<uint8_t>(cost);
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
  uint8_t *device_scores = nullptr;
  uint8_t *device_residues = nullptr;
  uint64_t *device_offsets = nullptr;
  uint8_t *device_tjb = nullptr;
  uint8_t *host_tjb = nullptr;
  plan7_ssv_result *device_results = nullptr;
  cudaError_t status;
  size_t score_bytes;
  size_t offset_bytes;
  size_t result_bytes;
  int rc = -1;

  if (model_length < 1 || model_length > 100000 || alphabet_size < 1 ||
      sequence_count > INT_MAX) {
    set_error(error, error_size, "invalid SSV dimensions");
    return -1;
  }
  const int Q = model_length < 17 ? 2 : (model_length + 15) / 16;
  const int minimum_stride = 16 * (Q + kExtraScoreVectors);
  if (score_stride < minimum_stride || striped_scores == nullptr ||
      offset_count != sequence_count + 1 || offsets == nullptr ||
      (residue_count != 0 && residues == nullptr)) {
    set_error(error, error_size, "invalid SSV buffers");
    return -1;
  }
  if (offsets[0] != 0 || offsets[sequence_count] != residue_count) {
    set_error(error, error_size, "invalid sequence offsets");
    return -1;
  }
  if (sequence_count == 0) return 0;
  if (result_count < sequence_count || results == nullptr ||
      !isfinite(scale) || scale <= 0.0f) {
    set_error(error, error_size, "invalid SSV result buffer or scale");
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
  if (!checked_product(static_cast<size_t>(score_stride),
                       static_cast<size_t>(alphabet_size), &score_bytes) ||
      !checked_product(sequence_count + 1, sizeof(*offsets), &offset_bytes) ||
      !checked_product(sequence_count, sizeof(*results), &result_bytes)) {
    set_error(error, error_size, "SSV buffer size overflow");
    return -1;
  }
  if (striped_score_count < score_bytes) {
    set_error(error, error_size, "striped score buffer is too short");
    return -1;
  }

#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      goto cleanup;                                                           \
    }                                                                         \
  } while (0)

  CUDA_TRY(cudaMalloc(&device_scores, score_bytes));
  if (residue_count != 0) CUDA_TRY(cudaMalloc(&device_residues, residue_count));
  CUDA_TRY(cudaMalloc(&device_offsets, offset_bytes));
  CUDA_TRY(cudaMalloc(&device_tjb, sequence_count));
  CUDA_TRY(cudaMalloc(&device_results, result_bytes));
  CUDA_TRY(cudaMemcpy(device_scores, striped_scores, score_bytes,
                      cudaMemcpyHostToDevice));
  if (residue_count != 0)
    CUDA_TRY(cudaMemcpy(device_residues, residues, residue_count,
                        cudaMemcpyHostToDevice));
  CUDA_TRY(cudaMemcpy(device_offsets, offsets, offset_bytes,
                      cudaMemcpyHostToDevice));
  host_tjb = static_cast<uint8_t *>(malloc(sequence_count));
  if (host_tjb == nullptr) {
    set_error(error, error_size, "host tjb allocation failed");
    goto cleanup;
  }
  for (size_t i = 0; i < sequence_count; ++i)
    host_tjb[i] = compute_tjb(scale, offsets[i + 1] - offsets[i]);
  CUDA_TRY(cudaMemcpy(device_tjb, host_tjb, sequence_count,
                      cudaMemcpyHostToDevice));

  ssv_kernel<<<static_cast<unsigned>(sequence_count), kThreads>>>(
    device_scores, score_stride, model_length, alphabet_size, device_residues,
    device_offsets, device_tjb, tbm, tec, base, bias, device_results);
  CUDA_TRY(cudaGetLastError());
  CUDA_TRY(cudaMemcpy(results, device_results, result_bytes,
                      cudaMemcpyDeviceToHost));
  rc = 0;

cleanup:
  free(host_tjb);
  cudaFree(device_results);
  cudaFree(device_tjb);
  cudaFree(device_offsets);
  cudaFree(device_residues);
  cudaFree(device_scores);
#undef CUDA_TRY
  return rc;
}
