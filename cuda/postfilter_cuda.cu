#include "postfilter_cuda.h"
#include "forward_cuda.h"

#include <cuda_runtime.h>

extern "C" {
#include <easel.h>
#include <hmmer.h>
#include <impl_sse/impl_sse.h>
}

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include <utility>
#include <vector>

static_assert(sizeof(plan7_postfilter_result) == PLAN7_POSTFILTER_RECORD_SIZE,
              "post-filter record ABI size changed");
static_assert(offsetof(plan7_postfilter_result, sequence_index) == 0 &&
              offsetof(plan7_postfilter_result, filtersc) == 4 &&
              offsetof(plan7_postfilter_result, msv_numerator) == 8 &&
              offsetof(plan7_postfilter_result, msv_status) == 10 &&
              offsetof(plan7_postfilter_result, action) == 11 &&
              offsetof(plan7_postfilter_result, vfsc) == 12,
              "post-filter record ABI layout changed");
static_assert(sizeof(plan7_forward_snapshot_profile) == 48,
              "Forward snapshot descriptor footprint changed");

namespace {

constexpr int kWarpSize = 32;
constexpr int kForwardSubwarp = 4;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpSize * kWarpsPerBlock;
constexpr int kNegInf = -32768;
constexpr uint64_t kDpByteLimit = UINT64_C(256) << 20;
constexpr double kLog2 = 0.69314718055994529;

enum CandidateState : uint8_t {
  kCandidateCpu = 0,
  kCandidateRawReject = 1,
  kCandidateFinite = 2,
  kCandidateMsvRange = 3
};

struct VitProfile {
  uint64_t ssv_offset;
  uint64_t rbv_offset;
  uint64_t emission_offset;
  uint64_t transition_offset;
  uint32_t q;
  uint32_t model_length;
  int32_t mode;
  int32_t base;
  int32_t ddbound;
  int32_t e_move;
  int32_t e_loop;
  int32_t n_loop;
  int32_t j_loop;
  int32_t c_loop;
  float scale;
  float nj;
  uint8_t msv_tbm;
  uint8_t msv_tec;
  uint8_t msv_base;
  uint8_t msv_bias;
  float msv_scale;
  uint64_t reserved;
};

static_assert(sizeof(VitProfile) == 96,
              "Viterbi descriptor footprint changed");
static_assert(sizeof(uintptr_t) >= sizeof(uint64_t),
              "profile identity tokens require 64-bit uintptr_t");
static_assert(p7O_NTRANS == 8,
              "Viterbi transition-row footprint changed");
static_assert((29 + p7O_NTRANS) * kWarpSize * sizeof(int16_t) == 2368,
              "Viterbi packed-row footprint changed");

struct VitLengthTransitions {
  int16_t n_move;
  int16_t j_move;
  int16_t c_move;
};

struct VitResult {
  int32_t status;
  uint32_t score_bits;
  int32_t numerator;
};

union FloatBits {
  float value;
  uint32_t bits;
};

void set_error(char *error, size_t error_size, const char *message) {
  if (error != nullptr && error_size != 0)
    std::snprintf(error, error_size, "%s", message);
}

void set_cuda_error(char *error, size_t error_size, const char *operation,
                    cudaError_t status) {
  if (error != nullptr && error_size != 0)
    std::snprintf(error, error_size, "%s: %s", operation,
                  cudaGetErrorString(status));
}

bool checked_add(uint64_t left, uint64_t right, uint64_t *sum) {
  if (right > UINT64_MAX - left) return false;
  *sum = left + right;
  return true;
}

bool checked_multiply(uint64_t left, uint64_t right, uint64_t *product) {
  if (left != 0 && right > UINT64_MAX / left) return false;
  *product = left * right;
  return true;
}

bool checked_bytes(size_t count, size_t size, size_t *bytes) {
  if (size != 0 && count > SIZE_MAX / size) return false;
  *bytes = count * size;
  return true;
}

bool aligned_vector_address(const void *allocation, uintptr_t *address) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(allocation);
  if (raw == 0 || raw > UINTPTR_MAX - 15) return false;
  *address = (raw + 15) & ~static_cast<uintptr_t>(15);
  return true;
}

bool valid_profile_storage(const P7_OPROFILE *profile) {
  if (profile == nullptr || profile->abc == nullptr ||
      profile->abc->type != eslAMINO || profile->abc->K != 20 ||
      profile->abc->Kp != 29 || profile->M < 1 || profile->M > 100000 ||
      profile->allocM < profile->M ||
      profile->allocM > 100000 || profile->allocQ16 != p7O_NQB(profile->allocM) ||
      profile->allocQ8 != p7O_NQW(profile->allocM) ||
      profile->allocQ16 < p7O_NQB(profile->M) ||
      profile->allocQ8 < p7O_NQW(profile->M) || profile->rbv == nullptr ||
      profile->sbv == nullptr || profile->rwv == nullptr ||
      profile->twv == nullptr || profile->rbv_mem == nullptr ||
      profile->sbv_mem == nullptr || profile->rwv_mem == nullptr ||
      profile->twv_mem == nullptr)
    return false;

  uintptr_t rbv_base;
  uintptr_t sbv_base;
  uintptr_t rwv_base;
  uintptr_t twv_base;
  if (!aligned_vector_address(profile->rbv_mem, &rbv_base) ||
      !aligned_vector_address(profile->sbv_mem, &sbv_base) ||
      !aligned_vector_address(profile->rwv_mem, &rwv_base) ||
      !aligned_vector_address(profile->twv_mem, &twv_base) ||
      reinterpret_cast<uintptr_t>(profile->twv) != twv_base)
    return false;

  const size_t rbv_stride = static_cast<size_t>(profile->allocQ16);
  const size_t sbv_stride = rbv_stride + p7O_EXTRA_SB;
  const size_t rwv_stride = static_cast<size_t>(profile->allocQ8);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const uintptr_t rbv_expected = rbv_base +
        static_cast<size_t>(residue) * rbv_stride * sizeof(__m128i);
    const uintptr_t sbv_expected = sbv_base +
        static_cast<size_t>(residue) * sbv_stride * sizeof(__m128i);
    const uintptr_t rwv_expected = rwv_base +
        static_cast<size_t>(residue) * rwv_stride * sizeof(__m128i);
    if (profile->rbv[residue] == nullptr || profile->sbv[residue] == nullptr ||
        profile->rwv[residue] == nullptr ||
        reinterpret_cast<uintptr_t>(profile->rbv[residue]) != rbv_expected ||
        reinterpret_cast<uintptr_t>(profile->sbv[residue]) != sbv_expected ||
        reinterpret_cast<uintptr_t>(profile->rwv[residue]) != rwv_expected)
      return false;
  }
  return true;
}

bool valid_viterbi_specials(const P7_OPROFILE *profile) {
  if (profile->base_w != 12000 ||
      (profile->nj != 0.0f && profile->nj != 1.0f) ||
      profile->xw[p7O_N][p7O_LOOP] != 0 ||
      profile->xw[p7O_J][p7O_LOOP] != 0 ||
      profile->xw[p7O_C][p7O_LOOP] != 0)
    return false;
  if (profile->nj == 0.0f)
    return profile->xw[p7O_E][p7O_MOVE] == 0 &&
           profile->xw[p7O_E][p7O_LOOP] == kNegInf;
  return profile->xw[p7O_E][p7O_MOVE] < 0 &&
         profile->xw[p7O_E][p7O_MOVE] > kNegInf &&
         profile->xw[p7O_E][p7O_LOOP] ==
             profile->xw[p7O_E][p7O_MOVE];
}

bool valid_forward_storage(const P7_OPROFILE *profile) {
  if (profile == nullptr || profile->abc == nullptr ||
      profile->abc->type != eslAMINO || profile->abc->K != 20 ||
      profile->abc->Kp != 29 || profile->M < 1 || profile->M > 100000 ||
      profile->allocM < profile->M || profile->allocM > 100000 ||
      profile->allocQ4 != p7O_NQF(profile->allocM) ||
      profile->allocQ4 < p7O_NQF(profile->M) ||
      profile->rfv == nullptr || profile->tfv == nullptr ||
      profile->rfv_mem == nullptr || profile->tfv_mem == nullptr)
    return false;

  uintptr_t rfv_base;
  uintptr_t tfv_base;
  if (!aligned_vector_address(profile->rfv_mem, &rfv_base) ||
      !aligned_vector_address(profile->tfv_mem, &tfv_base) ||
      reinterpret_cast<uintptr_t>(profile->tfv) != tfv_base)
    return false;
  const size_t row_stride = static_cast<size_t>(profile->allocQ4);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const uintptr_t expected = rfv_base +
        static_cast<size_t>(residue) * row_stride * sizeof(__m128);
    if (profile->rfv[residue] == nullptr ||
        reinterpret_cast<uintptr_t>(profile->rfv[residue]) != expected)
      return false;
  }
  return true;
}

bool valid_forward_specials(const P7_OPROFILE *profile) {
  if ((profile->mode != p7_LOCAL && profile->mode != p7_UNILOCAL) ||
      (profile->nj != 0.0f && profile->nj != 1.0f))
    return false;
  if (profile->nj == 0.0f)
    return profile->xf[p7O_E][p7O_MOVE] == 1.0f &&
           profile->xf[p7O_E][p7O_LOOP] == 0.0f;
  return profile->xf[p7O_E][p7O_MOVE] == 0.5f &&
         profile->xf[p7O_E][p7O_LOOP] == 0.5f;
}

bool valid_forward_probability_rows(const P7_OPROFILE *profile) {
  const int q_count = p7O_NQF(profile->M);
  for (int residue = 0; residue < profile->abc->Kp; ++residue) {
    const auto *values = reinterpret_cast<const float *>(profile->rfv[residue]);
    for (int cell = 0; cell < q_count * kForwardSubwarp; ++cell)
      if (!std::isfinite(values[cell]) || values[cell] < 0.0f) return false;
  }
  const auto *transitions = reinterpret_cast<const float *>(profile->tfv);
  for (int cell = 0;
       cell < q_count * p7O_NTRANS * kForwardSubwarp; ++cell)
    if (!std::isfinite(transitions[cell]) || transitions[cell] < 0.0f)
      return false;
  return true;
}

VitLengthTransitions length_transitions_for(const VitProfile &profile,
                                            int length) {
  const float numerator = 2.0f + profile.nj;
  const float denominator = static_cast<float>(length) + 2.0f + profile.nj;
  const float pmove = numerator / denominator;
  const float score = roundf(profile.scale * logf(pmove));
  int16_t move;
  if (score >= 32767.0f)
    move = 32767;
  else if (score <= -32768.0f)
    move = -32768;
  else
    move = static_cast<int16_t>(score);
  return {move, move, move};
}

__device__ __forceinline__ unsigned sat_add_u8(unsigned left,
                                                unsigned right) {
  const unsigned value = left + right;
  return value > UINT8_MAX ? UINT8_MAX : value;
}

__device__ __forceinline__ unsigned sat_sub_u8(unsigned left,
                                                unsigned right) {
  return left > right ? left - right : 0;
}

__device__ __forceinline__ int sat_add_i16(int left, int right) {
  const int value = left + right;
  return value > 32767 ? 32767 : (value < kNegInf ? kNegInf : value);
}

template<typename T>
__device__ __forceinline__ T warp_max(T value) {
  for (int delta = 16; delta != 0; delta >>= 1)
    value = max(value, __shfl_xor_sync(UINT32_MAX, value, delta));
  return value;
}

template<typename T>
__device__ __forceinline__ T previous_lane(T value, int lane, T first) {
  value = __shfl_up_sync(UINT32_MAX, value, 1);
  return lane == 0 ? first : value;
}

__device__ __forceinline__ bool raw_f1_survives(
    const plan7_bias_ssv_input result, float null_score,
    const plan7_ssv_f1_profile profile) {
  if (result.status == PLAN7_SSV_ERANGE) return true;
  if (result.status != PLAN7_SSV_OK) return false;
  if (profile.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_REJECT) return false;
  if (profile.cutoff_mode != PLAN7_F1_CUTOFF_SCORE ||
      !isfinite(null_score) || !isfinite(profile.profile.scale) ||
      profile.profile.scale <= 0.0f ||
      !isfinite(profile.cutoff_bit_score))
    return false;
  float score = __int2float_rn(static_cast<int>(result.numerator));
  score = __fdiv_rn(score, profile.profile.scale);
  score = __fsub_rn(score, 3.0f);
  const float delta = __fsub_rn(score, null_score);
  const float bit_score = __double2float_rn(__ddiv_rn(
      static_cast<double>(delta), kLog2));
  return isfinite(score) && isfinite(bit_score) &&
         bit_score >= profile.cutoff_bit_score;
}

__global__ void full_msv_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const uint8_t *compact_scores, const uint8_t *exact_rbv,
    const plan7_ssv_f1_profile *msv_profiles,
    const VitProfile *vit_profiles, const uint8_t *tjb,
    const plan7_bias_candidate *candidates, const uint64_t *dp_offsets,
    size_t candidate_begin, size_t tile_count, uint64_t tile_dp_begin,
    uint8_t *dp_storage, plan7_bias_ssv_input *msv_results) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (tile_candidate >= tile_count) return;
  const size_t candidate = candidate_begin + tile_candidate;
  if (msv_results[candidate].status != PLAN7_SSV_ENORESULT) return;

  const plan7_bias_candidate mapping = candidates[candidate];
  const plan7_ssv_f1_profile profile =
      msv_profiles[mapping.profile_index];
  const VitProfile vit_profile = vit_profiles[mapping.profile_index];
  const int q_count = static_cast<int>(vit_profile.q);
  const uint64_t sequence_start = sequence_offsets[mapping.sequence_index];
  const int sequence_length = static_cast<int>(
      sequence_offsets[mapping.sequence_index + 1] - sequence_start);
  const unsigned candidate_tjb =
      tjb[profile.tjb_offset + mapping.sequence_index];
  uint8_t *dp = dp_storage + dp_offsets[candidate] - tile_dp_begin;
  for (int q = 0; q < q_count; ++q)
    dp[q * kWarpSize + lane] = 0;

  const int signed_tjb = candidate_tjb < 128
      ? static_cast<int>(candidate_tjb)
      : static_cast<int>(candidate_tjb) - 256;
  const int signed_tbm = profile.profile.tbm < 128
      ? static_cast<int>(profile.profile.tbm)
      : static_cast<int>(profile.profile.tbm) - 256;
  const unsigned tjbm =
      static_cast<unsigned>(signed_tjb + signed_tbm) & 255U;
  unsigned xJ = 0;
  unsigned xB = sat_sub_u8(profile.profile.base, tjbm);

  for (int i = 0; i < sequence_length; ++i) {
    const unsigned residue = residues[sequence_start + i];
    unsigned xE = 0;
    unsigned mpv = previous_lane(
        static_cast<unsigned>(dp[(q_count - 1) * kWarpSize + lane]),
        lane, 0U);
    for (int q = 0; q < q_count; ++q) {
      unsigned score = max(mpv, xB);
      score = sat_add_u8(score, profile.profile.bias);
      const int model_position = q + q_count * lane;
      unsigned cost = UINT8_MAX;
      if (model_position < profile.profile.model_length) {
        if (vit_profile.rbv_offset != UINT64_MAX) {
          cost = exact_rbv[
              vit_profile.rbv_offset +
              static_cast<uint64_t>(model_position) * 29 + residue];
        } else if (residue != 20 && residue != 27 && residue != 28) {
          const unsigned raw = compact_scores[
              profile.profile.score_offset +
              static_cast<uint64_t>(model_position) *
                  profile.profile.score_stride + residue];
          const int signed_score = raw < 128 ? static_cast<int>(raw)
                                             : static_cast<int>(raw) - 256;
          const int decoded =
              signed_score + static_cast<int>(profile.profile.bias);
          cost = static_cast<unsigned>(
              decoded < 0 ? 0 : (decoded > 255 ? 255 : decoded));
        }
      }
      score = sat_sub_u8(score, cost);
      xE = max(xE, score);
      const unsigned old = dp[q * kWarpSize + lane];
      dp[q * kWarpSize + lane] = static_cast<uint8_t>(score);
      mpv = old;
    }
    xE = warp_max(xE);
    if (sat_add_u8(xE, profile.profile.bias) == UINT8_MAX) {
      if (lane == 0)
        msv_results[candidate] = {0, PLAN7_SSV_ERANGE, 0};
      return;
    }
    xE = sat_sub_u8(xE, profile.profile.tec);
    xJ = max(xJ, xE);
    xB = sat_sub_u8(max(static_cast<unsigned>(profile.profile.base), xJ),
                    tjbm);
  }
  if (lane == 0) {
    const int numerator = static_cast<int>(xJ) -
                          static_cast<int>(candidate_tjb) -
                          static_cast<int>(profile.profile.base);
    msv_results[candidate] = {
      static_cast<int16_t>(numerator), PLAN7_SSV_OK, 0
    };
  }
}

__global__ void prepare_bias_inputs_kernel(
    const float *null_scores, const plan7_ssv_f1_profile *profiles,
    const plan7_bias_candidate *candidates,
    const plan7_bias_ssv_input *msv_results, size_t candidate_count,
    uint8_t *states, plan7_bias_ssv_input *bias_inputs) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const plan7_bias_candidate mapping = candidates[candidate];
  const plan7_bias_ssv_input msv = msv_results[candidate];
  CandidateState state = kCandidateCpu;
  plan7_bias_ssv_input bias = {0, PLAN7_SSV_ENORESULT, 0};
  if (msv.status == PLAN7_SSV_ERANGE) {
    state = kCandidateMsvRange;
    bias = {INT16_MAX, PLAN7_SSV_OK, 0};
  } else if (msv.status == PLAN7_SSV_OK) {
    const plan7_ssv_f1_profile profile = profiles[mapping.profile_index];
    if (profile.cutoff_mode == PLAN7_F1_CUTOFF_ALWAYS_REJECT ||
        (profile.cutoff_mode == PLAN7_F1_CUTOFF_SCORE &&
         !raw_f1_survives(msv, null_scores[mapping.sequence_index],
                          profile))) {
      state = kCandidateRawReject;
    } else if (raw_f1_survives(
                   msv, null_scores[mapping.sequence_index], profile)) {
      state = kCandidateFinite;
      bias = msv;
    }
  }
  states[candidate] = static_cast<uint8_t>(state);
  bias_inputs[candidate] = bias;
}

__global__ void viterbi_kernel(
    const uint8_t *residues, const uint64_t *sequence_offsets,
    const VitProfile *profiles, const int16_t *emissions,
    const int16_t *transitions, const plan7_bias_candidate *candidates,
    const uint8_t *states, const plan7_bias_result *bias_results,
    const VitLengthTransitions *length_transitions,
    const uint64_t *dp_offsets, size_t candidate_begin, size_t tile_count,
    uint64_t tile_dp_begin, int16_t *dp_storage, VitResult *results) {
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const size_t tile_candidate =
      static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (tile_candidate >= tile_count) return;
  const size_t candidate = candidate_begin + tile_candidate;
  if (states[candidate] != kCandidateFinite ||
      bias_results[candidate].action == PLAN7_BIAS_CPU_REQUIRED)
    return;

  const plan7_bias_candidate mapping = candidates[candidate];
  const VitProfile profile = profiles[mapping.profile_index];
  const int q_count = static_cast<int>(profile.q);
  const uint64_t sequence_start = sequence_offsets[mapping.sequence_index];
  const int sequence_length = static_cast<int>(
      sequence_offsets[mapping.sequence_index + 1] - sequence_start);
  const VitLengthTransitions moves = length_transitions[candidate];
  int16_t *mmx = dp_storage + dp_offsets[candidate] - tile_dp_begin;
  int16_t *imx = mmx + static_cast<uint64_t>(q_count) * kWarpSize;
  int16_t *dmx = imx + static_cast<uint64_t>(q_count) * kWarpSize;
  for (int q = 0; q < q_count; ++q) {
    mmx[q * kWarpSize + lane] = kNegInf;
    imx[q * kWarpSize + lane] = kNegInf;
    dmx[q * kWarpSize + lane] = kNegInf;
  }

  int xN = profile.base;
  int xB = xN + moves.n_move;
  int xJ = kNegInf;
  int xC = kNegInf;
  for (int i = 0; i < sequence_length; ++i) {
    const unsigned residue = residues[sequence_start + i];
    const uint64_t emission_base = profile.emission_offset +
        static_cast<uint64_t>(residue) * q_count * kWarpSize;
    int dcv = kNegInf;
    int xE = kNegInf;
    int dmax = kNegInf;
    int mpv = previous_lane(
        static_cast<int>(mmx[(q_count - 1) * kWarpSize + lane]), lane,
        kNegInf);
    int ipv = previous_lane(
        static_cast<int>(imx[(q_count - 1) * kWarpSize + lane]), lane,
        kNegInf);
    int dpv = previous_lane(
        static_cast<int>(dmx[(q_count - 1) * kWarpSize + lane]), lane,
        kNegInf);
    for (int q = 0; q < q_count; ++q) {
      const uint64_t transition_base = profile.transition_offset +
          static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize + lane;
      int score = sat_add_i16(
          xB, transitions[transition_base + p7O_BM * kWarpSize]);
      score = max(score, sat_add_i16(
          mpv, transitions[transition_base + p7O_MM * kWarpSize]));
      score = max(score, sat_add_i16(
          ipv, transitions[transition_base + p7O_IM * kWarpSize]));
      score = max(score, sat_add_i16(
          dpv, transitions[transition_base + p7O_DM * kWarpSize]));
      score = sat_add_i16(
          score, emissions[emission_base + q * kWarpSize + lane]);
      xE = max(xE, score);
      const int old_m = mmx[q * kWarpSize + lane];
      const int old_i = imx[q * kWarpSize + lane];
      const int old_d = dmx[q * kWarpSize + lane];
      mmx[q * kWarpSize + lane] = static_cast<int16_t>(score);
      dmx[q * kWarpSize + lane] = static_cast<int16_t>(dcv);
      dcv = sat_add_i16(
          score, transitions[transition_base + p7O_MD * kWarpSize]);
      dmax = max(dmax, dcv);
      int insert = sat_add_i16(
          old_m, transitions[transition_base + p7O_MI * kWarpSize]);
      insert = max(insert, sat_add_i16(
          old_i, transitions[transition_base + p7O_II * kWarpSize]));
      imx[q * kWarpSize + lane] = static_cast<int16_t>(insert);
      mpv = old_m;
      ipv = old_i;
      dpv = old_d;
    }
    xE = warp_max(xE);
    if (xE >= 32767) {
      if (lane == 0)
        results[candidate] = {
          PLAN7_SSV_ERANGE, UINT32_C(0x7f800000), INT32_MAX
        };
      return;
    }
    xN += profile.n_loop;
    xC = max(xC + profile.c_loop, xE + profile.e_move);
    xJ = max(xJ + profile.j_loop, xE + profile.e_loop);
    xB = max(xJ + moves.j_move, xN + moves.n_move);

    dmax = warp_max(dmax);
    if (dmax + profile.ddbound > xB) {
      dcv = previous_lane(dcv, lane, kNegInf);
      for (int q = 0; q < q_count; ++q) {
        const uint64_t transition_base = profile.transition_offset +
            static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize + lane;
        const int updated = max(
            dcv, static_cast<int>(dmx[q * kWarpSize + lane]));
        dmx[q * kWarpSize + lane] = static_cast<int16_t>(updated);
        dcv = sat_add_i16(
            updated, transitions[transition_base + p7O_DD * kWarpSize]);
      }
      while (true) {
        dcv = previous_lane(dcv, lane, kNegInf);
        bool complete = true;
        for (int q = 0; q < q_count; ++q) {
          const int current = dmx[q * kWarpSize + lane];
          if (__any_sync(UINT32_MAX, dcv > current) == 0) {
            complete = false;
            break;
          }
          const int updated = max(dcv, current);
          dmx[q * kWarpSize + lane] = static_cast<int16_t>(updated);
          const uint64_t transition_base = profile.transition_offset +
              static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize + lane;
          dcv = sat_add_i16(
              updated, transitions[transition_base + p7O_DD * kWarpSize]);
        }
        if (!complete) break;
      }
    } else {
      dcv = previous_lane(dcv, lane, kNegInf);
      dmx[lane] = static_cast<int16_t>(dcv);
    }
  }

  if (lane == 0) {
    FloatBits score{};
    int numerator = INT32_MIN;
    if (xC > kNegInf) {
      numerator = xC + moves.c_move - profile.base;
      float value = __fadd_rn(__int2float_rn(xC),
                              __int2float_rn(moves.c_move));
      value = __fsub_rn(value, __int2float_rn(profile.base));
      value = __fdiv_rn(value, profile.scale);
      score.value = __fsub_rn(value, 3.0f);
    } else {
      score.bits = UINT32_C(0xff800000);
    }
    results[candidate] = {PLAN7_SSV_OK, score.bits, numerator};
  }
}

__global__ void merge_results_kernel(
    const plan7_bias_candidate *candidates,
    const plan7_bias_ssv_input *msv_results, const uint8_t *states,
    const plan7_bias_result *bias_results, const VitResult *vit_results,
    size_t candidate_count, plan7_postfilter_result *results) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const plan7_bias_ssv_input msv = msv_results[candidate];
  plan7_postfilter_result output = {
    candidates[candidate].sequence_index, NAN, msv.numerator,
    msv.status, PLAN7_BIAS_CPU_REQUIRED, NAN
  };
  const CandidateState state =
      static_cast<CandidateState>(states[candidate]);
  if (state == kCandidateRawReject && msv.status == PLAN7_SSV_OK) {
    output.action = PLAN7_BIAS_DEFINITE_REJECT;
  } else if (state == kCandidateFinite &&
             msv.status == PLAN7_SSV_OK &&
             isfinite(bias_results[candidate].filtersc) &&
             (bias_results[candidate].action == PLAN7_BIAS_DEFINITE_REJECT ||
              bias_results[candidate].action == PLAN7_BIAS_DEFINITE_PASS)) {
    FloatBits vfsc{};
    vfsc.bits = vit_results[candidate].score_bits;
    if (vit_results[candidate].status == PLAN7_SSV_ERANGE) {
      output.filtersc = bias_results[candidate].filtersc;
      output.action = bias_results[candidate].action;
      output.vfsc = __int_as_float(0x7f800000);
    } else if (vit_results[candidate].status == PLAN7_SSV_OK &&
               isfinite(vfsc.value)) {
      output.filtersc = bias_results[candidate].filtersc;
      output.action = bias_results[candidate].action;
      output.vfsc = vfsc.value;
    }
  }
  results[candidate] = output;
}

__global__ void postfilter_reason_facts_kernel(
    const plan7_bias_profile *profiles,
    const plan7_bias_candidate *candidates,
    const plan7_bias_ssv_input *msv_results,
    const uint8_t *states,
    const plan7_bias_ssv_input *bias_inputs,
    const plan7_bias_result *bias_results,
    const VitResult *vit_results,
    const plan7_postfilter_result *results,
    size_t candidate_count,
    uint16_t *reason_facts) {
  const size_t candidate =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;

  const CandidateState state = static_cast<CandidateState>(states[candidate]);
  const plan7_bias_result bias = bias_results[candidate];
  const plan7_postfilter_result output = results[candidate];
  uint16_t facts = 0;
  if (state == kCandidateRawReject)
    facts |= PLAN7_POSTFILTER_REASON_RAW_F1_REJECT;
  else if (state == kCandidateMsvRange)
    facts |= PLAN7_POSTFILTER_REASON_FULL_MSV_ERANGE;
  else if (state == kCandidateCpu)
    facts |= PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU;

  if (state == kCandidateFinite) {
    const plan7_bias_ssv_input bias_input = bias_inputs[candidate];
    if (bias_input.status != PLAN7_SSV_OK) {
      facts |= PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO;
    } else if (!isfinite(bias.filtersc)) {
      /* bias_filter_score writes filtersc only after its complete finite
       * recurrence, so a nonfinite retained value is its exact false path. */
      facts |= PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED;
    } else {
      const plan7_bias_candidate mapping = candidates[candidate];
      const plan7_bias_profile profile = profiles[mapping.profile_index];
      float usc = __int2float_rn(static_cast<int>(bias_input.numerator));
      usc = __fdiv_rn(usc, profile.scale);
      usc = __fsub_rn(usc, 3.0f);
      const float bit_score = __double2float_rn(__ddiv_rn(
          static_cast<double>(__fsub_rn(usc, bias.filtersc)), kLog2));
      if (!isfinite(usc) || !isfinite(bit_score))
        facts |= PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE;
      else if (bias.action == PLAN7_BIAS_CPU_REQUIRED)
        facts |= PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED;
    }

    if (bias.action != PLAN7_BIAS_CPU_REQUIRED) {
      if (vit_results[candidate].status == PLAN7_SSV_ERANGE)
        facts |= PLAN7_POSTFILTER_REASON_VITERBI_ERANGE;
      else if (vit_results[candidate].status != PLAN7_SSV_OK)
        facts |= PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS;
    }
  }

  if (output.action == PLAN7_BIAS_CPU_REQUIRED) {
    facts |= PLAN7_POSTFILTER_REASON_FINAL_CPU_REQUIRED;
    constexpr uint16_t explained =
        PLAN7_POSTFILTER_REASON_FULL_MSV_ERANGE |
        PLAN7_POSTFILTER_REASON_CANDIDATE_STATE_CPU |
        PLAN7_POSTFILTER_REASON_BIAS_INPUT_STATUS_NONZERO |
        PLAN7_POSTFILTER_REASON_BIAS_FILTER_SCORE_FAILED |
        PLAN7_POSTFILTER_REASON_BIAS_SCORE_NONFINITE |
        PLAN7_POSTFILTER_REASON_BIAS_CUTOFF_UNRESOLVED |
        PLAN7_POSTFILTER_REASON_VITERBI_ERANGE |
        PLAN7_POSTFILTER_REASON_VITERBI_NO_RESULT_OR_OTHER_STATUS;
    if ((facts & explained) == 0)
      facts |= PLAN7_POSTFILTER_REASON_OTHER_CPU_REQUIRED;
  } else if (output.action == PLAN7_BIAS_DEFINITE_REJECT) {
    facts |= PLAN7_POSTFILTER_REASON_FINAL_REJECT;
  } else if (output.action == PLAN7_BIAS_DEFINITE_PASS) {
    facts |= PLAN7_POSTFILTER_REASON_FINAL_PASS;
  }
  reason_facts[candidate] = facts;
}

}  // namespace

namespace {

class PackWorkerPool {
 public:
  using Task = void (*)(void *, size_t) noexcept;

  explicit PackWorkerPool(size_t worker_count) {
    try {
      workers_.reserve(worker_count);
      for (size_t i = 0; i < worker_count; ++i)
        workers_.emplace_back([this]() { worker_loop(); });
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
      }
      ready_.notify_all();
      for (auto &worker : workers_)
        if (worker.joinable()) worker.join();
      throw;
    }
  }

  PackWorkerPool(const PackWorkerPool &) = delete;
  PackWorkerPool &operator=(const PackWorkerPool &) = delete;

  ~PackWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
  }

  void parallel_for(size_t task_count, void *context, Task task) {
    if (task_count == 0) return;
    if (workers_.empty()) {
      for (size_t task_index = 0; task_index < task_count; ++task_index)
        task(context, task_index);
      parallel_run_count_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    completed_.wait(lock, [this]() { return !active_; });
    task_count_ = task_count;
    task_context_ = context;
    task_ = task;
    next_task_.store(0, std::memory_order_relaxed);
    finished_workers_ = 0;
    active_ = true;
    ++generation_;
    parallel_run_count_.fetch_add(1, std::memory_order_relaxed);
    ready_.notify_all();
    completed_.wait(lock, [this]() { return !active_; });
  }

  size_t worker_count() const noexcept { return workers_.size(); }

  uint64_t parallel_run_count() const noexcept {
    return parallel_run_count_.load(std::memory_order_relaxed);
  }

 private:
  void worker_loop() noexcept {
    uint64_t observed_generation = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      ready_.wait(lock, [this, observed_generation]() {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_) return;
      observed_generation = generation_;
      const size_t task_count = task_count_;
      void *context = task_context_;
      Task task = task_;
      lock.unlock();
      for (;;) {
        const size_t task_index =
            next_task_.fetch_add(1, std::memory_order_relaxed);
        if (task_index >= task_count) break;
        task(context, task_index);
      }
      lock.lock();
      ++finished_workers_;
      if (finished_workers_ == workers_.size()) {
        active_ = false;
        completed_.notify_all();
      }
    }
  }

  std::vector<std::thread> workers_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::condition_variable completed_;
  std::atomic<size_t> next_task_{0};
  std::atomic<uint64_t> parallel_run_count_{0};
  size_t task_count_ = 0;
  void *task_context_ = nullptr;
  Task task_ = nullptr;
  size_t finished_workers_ = 0;
  uint64_t generation_ = 0;
  bool active_ = false;
  bool stopping_ = false;
};

struct HostProfilePack {
  int alphabet_size = 29;
  std::vector<VitProfile> vit_profiles;
  std::vector<uint8_t> ssv_scores;
  std::vector<uint8_t> exact_rbv;
  std::vector<int16_t> emissions;
  std::vector<int16_t> transitions;
  std::vector<plan7_ssv_profile> ssv_profiles;
  std::vector<float> m_mu;
  std::vector<float> m_lambda;
  std::vector<float> v_mu;
  std::vector<float> v_lambda;
  std::vector<plan7_bias_profile> bias_templates;
  std::vector<plan7_forward_snapshot_profile> forward_profiles;
  std::vector<float> forward_emissions;
  std::vector<float> forward_transitions;
  std::vector<uintptr_t> identity_tokens;
};

uint64_t host_pack_bytes(const HostProfilePack &pack) {
  const uint64_t counts[] = {
      static_cast<uint64_t>(pack.vit_profiles.size()) * sizeof(VitProfile),
      static_cast<uint64_t>(pack.ssv_scores.size()),
      static_cast<uint64_t>(pack.exact_rbv.size()),
      static_cast<uint64_t>(pack.emissions.size()) * sizeof(int16_t),
      static_cast<uint64_t>(pack.transitions.size()) * sizeof(int16_t),
      static_cast<uint64_t>(pack.ssv_profiles.size()) *
          sizeof(plan7_ssv_profile),
      static_cast<uint64_t>(pack.m_mu.size()) * sizeof(float),
      static_cast<uint64_t>(pack.m_lambda.size()) * sizeof(float),
      static_cast<uint64_t>(pack.v_mu.size()) * sizeof(float),
      static_cast<uint64_t>(pack.v_lambda.size()) * sizeof(float),
      static_cast<uint64_t>(pack.bias_templates.size()) *
          sizeof(plan7_bias_profile),
      static_cast<uint64_t>(pack.forward_profiles.size()) *
          sizeof(plan7_forward_snapshot_profile),
      static_cast<uint64_t>(pack.forward_emissions.size()) * sizeof(float),
      static_cast<uint64_t>(pack.forward_transitions.size()) * sizeof(float),
      static_cast<uint64_t>(pack.identity_tokens.size()) * sizeof(uintptr_t)};
  uint64_t total = 0;
  for (const uint64_t count : counts) {
    if (count > UINT64_MAX - total) return UINT64_MAX;
    total += count;
  }
  return total;
}

struct SnapshotTask {
  const uintptr_t *profile_pointers;
  HostProfilePack *pack;
};

void snapshot_profile_task(void *opaque, size_t profile_index) noexcept {
  auto *task = static_cast<SnapshotTask *>(opaque);
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      task->profile_pointers[profile_index]);
  HostProfilePack &pack = *task->pack;
  const VitProfile descriptor = pack.vit_profiles[profile_index];
  const int source_q = p7O_NQW(source->M);
  const int source_qb = std::max(2, (source->M + 15) / 16);
  const int q_count = static_cast<int>(descriptor.q);
  for (int q = 0; q < q_count; ++q) {
    for (int lane = 0; lane < kWarpSize; ++lane) {
      const int model_position = q + q_count * lane + 1;
      if (model_position > source->M) continue;
      const int source_stripe = (model_position - 1) % source_q;
      const int source_lane = (model_position - 1) / source_q;
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const auto *source_bytes = reinterpret_cast<const uint8_t *>(
            source->sbv[residue] + (model_position - 1) % source_qb);
        pack.ssv_scores[
            descriptor.ssv_offset +
            static_cast<uint64_t>(model_position - 1) *
                source->abc->Kp + residue] =
            source_bytes[(model_position - 1) / source_qb];
        if (descriptor.rbv_offset != UINT64_MAX) {
          const auto *source_rbv = reinterpret_cast<const uint8_t *>(
              source->rbv[residue] + (model_position - 1) % source_qb);
          pack.exact_rbv[
              descriptor.rbv_offset +
              static_cast<uint64_t>(model_position - 1) *
                  source->abc->Kp + residue] =
              source_rbv[(model_position - 1) / source_qb];
        }
        const uint64_t destination = descriptor.emission_offset +
            (static_cast<uint64_t>(residue) * q_count + q) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->rwv[residue] + source_stripe);
        pack.emissions[destination] = source_words[source_lane];
      }
      for (int transition = p7O_BM; transition <= p7O_II; ++transition) {
        const uint64_t destination = descriptor.transition_offset +
            (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->twv + source_stripe * 7 + transition);
        pack.transitions[destination] = source_words[source_lane];
      }
      const uint64_t dd_destination = descriptor.transition_offset +
          (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) *
              kWarpSize + lane;
      const auto *dd_words = reinterpret_cast<const int16_t *>(
          source->twv + 7 * source_q + source_stripe);
      pack.transitions[dd_destination] = dd_words[source_lane];
    }
  }

  if (pack.forward_profiles.empty()) return;
  const plan7_forward_snapshot_profile forward =
      pack.forward_profiles[profile_index];
  const int forward_q = static_cast<int>(forward.q);
  for (int residue = 0; residue < source->abc->Kp; ++residue) {
    const auto *source_values =
        reinterpret_cast<const float *>(source->rfv[residue]);
    float *destination = pack.forward_emissions.data() +
        forward.emission_offset +
        static_cast<uint64_t>(residue) * forward_q * kForwardSubwarp;
    std::memcpy(destination, source_values,
                static_cast<size_t>(forward_q) * kForwardSubwarp *
                    sizeof(float));
  }
  for (int q = 0; q < forward_q; ++q) {
    for (int transition = p7O_BM; transition <= p7O_II; ++transition) {
      const auto *source_values = reinterpret_cast<const float *>(
          source->tfv + q * 7 + transition);
      float *destination = pack.forward_transitions.data() +
          forward.transition_offset +
          (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
              kForwardSubwarp;
      std::memcpy(destination, source_values,
                  kForwardSubwarp * sizeof(float));
    }
    const auto *source_dd = reinterpret_cast<const float *>(
        source->tfv + 7 * forward_q + q);
    float *destination_dd = pack.forward_transitions.data() +
        forward.transition_offset +
        (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) *
            kForwardSubwarp;
    std::memcpy(destination_dd, source_dd,
                kForwardSubwarp * sizeof(float));
  }
}

int snapshot_profiles(const uintptr_t *profile_pointers, size_t profile_count,
                      const float *background, PackWorkerPool *workers,
                      HostProfilePack *output, char *error,
                      size_t error_size) {
  if (output == nullptr || workers == nullptr ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid host profile snapshot arguments");
    return -1;
  }
  HostProfilePack pack;
  uint64_t emission_total = 0;
  uint64_t transition_total = 0;
  uint64_t ssv_total = 0;
  uint64_t exact_rbv_total = 0;
  uint64_t forward_emission_total = 0;
  uint64_t forward_transition_total = 0;
  try {
    pack.vit_profiles.resize(profile_count);
    pack.ssv_profiles.resize(profile_count);
    pack.m_mu.resize(profile_count);
    pack.m_lambda.resize(profile_count);
    if (background != nullptr) {
      pack.v_mu.resize(profile_count);
      pack.v_lambda.resize(profile_count);
      pack.bias_templates.resize(profile_count);
      pack.forward_profiles.resize(profile_count);
    }
  } catch (...) {
    set_error(error, error_size, "host profile descriptor allocation failed");
    return -1;
  }

  for (size_t p = 0; p < profile_count; ++p) {
    const auto *source = reinterpret_cast<const P7_OPROFILE *>(
        profile_pointers[p]);
    if (!valid_profile_storage(source) || !valid_viterbi_specials(source) ||
        (background != nullptr &&
         (!valid_forward_storage(source) || !valid_forward_specials(source) ||
          !valid_forward_probability_rows(source))) ||
        source->abc == nullptr || source->abc->Kp != 29 ||
        !isfinite(source->scale_b) || source->scale_b <= 0.0f ||
        !isfinite(source->scale_w) || source->scale_w <= 0.0f ||
        (source->mode != p7_LOCAL && source->mode != p7_UNILOCAL)) {
      set_error(error, error_size, "invalid optimized Viterbi profile");
      return -1;
    }
    const int q = (source->M + kWarpSize - 1) / kWarpSize;
    const uint64_t emission_count =
        static_cast<uint64_t>(source->abc->Kp) * q * kWarpSize;
    const uint64_t ssv_count =
        static_cast<uint64_t>(source->abc->Kp) * source->M;
    const uint64_t transition_count =
        static_cast<uint64_t>(q) * p7O_NTRANS * kWarpSize;
    const uint64_t forward_q = background == nullptr
        ? 0 : static_cast<uint64_t>(p7O_NQF(source->M));
    uint64_t forward_emission_count = 0;
    uint64_t forward_transition_count = 0;
    if (background != nullptr &&
        (!checked_multiply(static_cast<uint64_t>(source->abc->Kp),
                           forward_q * kForwardSubwarp,
                           &forward_emission_count) ||
         !checked_multiply(forward_q,
                           p7O_NTRANS * kForwardSubwarp,
                           &forward_transition_count))) {
      set_error(error, error_size, "Forward packed profile size overflow");
      return -1;
    }
    bool needs_exact_rbv = false;
    const int source_qb = std::max(2, (source->M + 15) / 16);
    for (int model_position = 0;
         model_position < source->M && !needs_exact_rbv;
         ++model_position) {
      const int source_stripe = model_position % source_qb;
      const int source_lane = model_position / source_qb;
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const auto *sbv = reinterpret_cast<const uint8_t *>(
            source->sbv[residue] + source_stripe);
        const auto *rbv = reinterpret_cast<const uint8_t *>(
            source->rbv[residue] + source_stripe);
        const unsigned raw = sbv[source_lane];
        const int signed_score = raw < 128 ? static_cast<int>(raw)
                                           : static_cast<int>(raw) - 256;
        const int decoded_value =
            signed_score + static_cast<int>(source->bias_b);
        const unsigned decoded = residue == 20 || residue == 27 ||
                                 residue == 28
            ? UINT8_MAX
            : static_cast<unsigned>(decoded_value < 0 ? 0 :
                                    (decoded_value > 255 ? 255 :
                                     decoded_value));
        if (decoded != rbv[source_lane]) {
          needs_exact_rbv = true;
          break;
        }
      }
    }
    const uint64_t exact_rbv_count = needs_exact_rbv ? ssv_count : 0;
    if (!checked_add(ssv_total, ssv_count, &ssv_total) ||
        !checked_add(exact_rbv_total, exact_rbv_count, &exact_rbv_total) ||
        !checked_add(emission_total, emission_count, &emission_total) ||
        !checked_add(transition_total, transition_count, &transition_total) ||
        !checked_add(forward_emission_total, forward_emission_count,
                     &forward_emission_total) ||
        !checked_add(forward_transition_total, forward_transition_count,
                     &forward_transition_total) ||
        ssv_total > SIZE_MAX || exact_rbv_total > SIZE_MAX ||
        emission_total > SIZE_MAX || transition_total > SIZE_MAX ||
        forward_emission_total > SIZE_MAX ||
        forward_transition_total > SIZE_MAX) {
      set_error(error, error_size, "Viterbi packed profile size overflow");
      return -1;
    }
    VitProfile descriptor{};
    descriptor.ssv_offset = ssv_total - ssv_count;
    descriptor.rbv_offset = needs_exact_rbv
        ? exact_rbv_total - exact_rbv_count
        : UINT64_MAX;
    descriptor.emission_offset = emission_total - emission_count;
    descriptor.transition_offset = transition_total - transition_count;
    descriptor.q = static_cast<uint32_t>(q);
    descriptor.model_length = static_cast<uint32_t>(source->M);
    descriptor.mode = source->mode;
    descriptor.base = source->base_w;
    descriptor.ddbound = source->ddbound_w;
    descriptor.e_move = source->xw[p7O_E][p7O_MOVE];
    descriptor.e_loop = source->xw[p7O_E][p7O_LOOP];
    descriptor.n_loop = source->xw[p7O_N][p7O_LOOP];
    descriptor.j_loop = source->xw[p7O_J][p7O_LOOP];
    descriptor.c_loop = source->xw[p7O_C][p7O_LOOP];
    descriptor.scale = source->scale_w;
    descriptor.nj = source->nj;
    descriptor.msv_tbm = source->tbm_b;
    descriptor.msv_tec = source->tec_b;
    descriptor.msv_base = source->base_b;
    descriptor.msv_bias = source->bias_b;
    descriptor.msv_scale = source->scale_b;
    pack.vit_profiles[p] = descriptor;
    pack.ssv_profiles[p] = {
        descriptor.ssv_offset, ssv_count, 29, source->M,
        source->tbm_b, source->tec_b, source->base_b, source->bias_b,
        source->scale_b};
    pack.m_mu[p] = source->evparam[p7_MMU];
    pack.m_lambda[p] = source->evparam[p7_MLAMBDA];
    if (background != nullptr) {
      pack.v_mu[p] = source->evparam[p7_VMU];
      pack.v_lambda[p] = source->evparam[p7_VLAMBDA];
      pack.forward_profiles[p] = {
        forward_emission_total - forward_emission_count,
        forward_transition_total - forward_transition_count,
        static_cast<uint32_t>(forward_q),
        static_cast<uint32_t>(source->M),
        source->xf[p7O_E][p7O_MOVE],
        source->xf[p7O_E][p7O_LOOP],
        source->evparam[p7_FTAU],
        source->evparam[p7_FLAMBDA],
        source->nj,
        source->mode
      };
    }
    if (background != nullptr &&
        plan7_bias_pack_amino_profile(
            background, source->compo, source->M, source->scale_b,
            PLAN7_BIAS_CUTOFF_INVALID, NAN, &pack.bias_templates[p],
            error, error_size) != 0)
      return -1;
  }

  try {
    pack.ssv_scores.resize(static_cast<size_t>(ssv_total));
    pack.exact_rbv.resize(static_cast<size_t>(exact_rbv_total));
    pack.emissions.assign(static_cast<size_t>(emission_total),
                          static_cast<int16_t>(kNegInf));
    pack.transitions.assign(static_cast<size_t>(transition_total),
                            static_cast<int16_t>(kNegInf));
    pack.forward_emissions.resize(
        static_cast<size_t>(forward_emission_total));
    pack.forward_transitions.resize(
        static_cast<size_t>(forward_transition_total));
  } catch (...) {
    set_error(error, error_size, "Viterbi packed data allocation failed");
    return -1;
  }
  SnapshotTask task{profile_pointers, &pack};
  workers->parallel_for(profile_count, &task, snapshot_profile_task);
  *output = std::move(pack);
  return 0;
}

struct SelectionCopyTask {
  const HostProfilePack *source;
  HostProfilePack *destination;
  const size_t *indices;
};

void copy_selection_profile(void *opaque, size_t output_index) noexcept {
  auto *task = static_cast<SelectionCopyTask *>(opaque);
  const HostProfilePack &source = *task->source;
  HostProfilePack &destination = *task->destination;
  const size_t source_index = task->indices[output_index];
  const VitProfile &from = source.vit_profiles[source_index];
  const VitProfile &to = destination.vit_profiles[output_index];
  const size_t ssv_count =
      static_cast<size_t>(to.model_length) * destination.alphabet_size;
  const size_t emission_count = static_cast<size_t>(to.q) * kWarpSize *
                                destination.alphabet_size;
  const size_t transition_count =
      static_cast<size_t>(to.q) * kWarpSize * p7O_NTRANS;
  std::memcpy(destination.ssv_scores.data() + to.ssv_offset,
              source.ssv_scores.data() + from.ssv_offset, ssv_count);
  if (to.rbv_offset != UINT64_MAX)
    std::memcpy(destination.exact_rbv.data() + to.rbv_offset,
                source.exact_rbv.data() + from.rbv_offset, ssv_count);
  std::memcpy(destination.emissions.data() + to.emission_offset,
              source.emissions.data() + from.emission_offset,
              emission_count * sizeof(int16_t));
  std::memcpy(destination.transitions.data() + to.transition_offset,
              source.transitions.data() + from.transition_offset,
              transition_count * sizeof(int16_t));

  const plan7_forward_snapshot_profile &forward_from =
      source.forward_profiles[source_index];
  const plan7_forward_snapshot_profile &forward_to =
      destination.forward_profiles[output_index];
  const size_t forward_emission_count =
      static_cast<size_t>(forward_to.q) * kForwardSubwarp *
      destination.alphabet_size;
  const size_t forward_transition_count =
      static_cast<size_t>(forward_to.q) * kForwardSubwarp * p7O_NTRANS;
  std::memcpy(
      destination.forward_emissions.data() + forward_to.emission_offset,
      source.forward_emissions.data() + forward_from.emission_offset,
      forward_emission_count * sizeof(float));
  std::memcpy(
      destination.forward_transitions.data() + forward_to.transition_offset,
      source.forward_transitions.data() + forward_from.transition_offset,
      forward_transition_count * sizeof(float));
}

std::atomic<uint64_t> next_profile_session_id{1};
std::atomic<uint64_t> next_profile_identity_token{1};

bool claim_profile_session_id(uint64_t *session_id) {
  uint64_t current = next_profile_session_id.load(std::memory_order_relaxed);
  while (current != 0) {
    const uint64_t next = current == UINT64_MAX ? 0 : current + 1;
    if (next_profile_session_id.compare_exchange_weak(
          current, next, std::memory_order_relaxed,
          std::memory_order_relaxed)) {
      *session_id = current;
      return true;
    }
  }
  return false;
}

bool claim_profile_identity_tokens(size_t token_count,
                                   uint64_t *first_token) {
  if (token_count == 0) {
    *first_token = 0;
    return true;
  }
  const uint64_t count = static_cast<uint64_t>(token_count);
  if (static_cast<size_t>(count) != token_count) return false;
  uint64_t current = next_profile_identity_token.load(
      std::memory_order_relaxed);
  while (current != 0) {
    if (count - 1 > UINT64_MAX - current) return false;
    const uint64_t last = current + count - 1;
    const uint64_t next = last == UINT64_MAX ? 0 : last + 1;
    if (next_profile_identity_token.compare_exchange_weak(
          current, next, std::memory_order_relaxed,
          std::memory_order_relaxed)) {
      *first_token = current;
      return true;
    }
  }
  return false;
}

}  // namespace

struct plan7_viterbi_database {
  int device_ordinal;
  int alphabet_size;
  bool sealed_source;
  std::vector<VitProfile> host_profiles;
  std::vector<uintptr_t> source_profile_pointers;
  std::vector<uintptr_t> source_alphabet_pointers;
  std::vector<uint8_t> host_ssv_scores;
  std::vector<uint8_t> host_exact_rbv;
  std::vector<int16_t> host_emissions;
  std::vector<int16_t> host_transitions;
  std::shared_ptr<const HostProfilePack> sealed_pack;
  VitProfile *device_profiles;
  int16_t *device_emissions;
  int16_t *device_transitions;
  uint8_t *device_exact_rbv;
  size_t emission_count;
  size_t transition_count;
  size_t exact_rbv_count;
};

struct plan7_profile_session {
  uint64_t session_id;
  uint64_t build_worker_count;
  uint64_t build_parallel_run_count;
  uint64_t selection_count;
  std::shared_ptr<const HostProfilePack> pack;
  std::unique_ptr<PackWorkerPool> selection_workers;
  std::mutex operation_mutex;
};

struct plan7_profile_selection {
  uint64_t session_id;
  uint64_t selection_id;
  std::shared_ptr<const HostProfilePack> pack;
};

struct plan7_postfilter_workspace {
  int device_ordinal;
  std::vector<uint64_t> host_msv_offsets;
  std::vector<uint64_t> host_vit_offsets;
  std::vector<VitLengthTransitions> host_moves;
  std::vector<size_t> msv_tiles;
  std::vector<size_t> vit_tiles;
  uint8_t *device_states;
  plan7_bias_ssv_input *device_bias_inputs;
  plan7_bias_result *device_bias_results;
  VitResult *device_vit_results;
  VitLengthTransitions *device_moves;
  uint64_t *device_msv_offsets;
  uint64_t *device_vit_offsets;
  void *device_dp;
  plan7_postfilter_result *device_results;
  size_t states_capacity;
  size_t bias_inputs_capacity;
  size_t bias_results_capacity;
  size_t vit_results_capacity;
  size_t moves_capacity;
  size_t msv_offsets_capacity;
  size_t vit_offsets_capacity;
  size_t dp_capacity;
  size_t results_capacity;
  uint64_t growth_count;
  uint64_t run_count;
};

namespace {

const std::vector<VitProfile> &database_host_profiles(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->vit_profiles
      : database->host_profiles;
}

const std::vector<uint8_t> &database_host_ssv_scores(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->ssv_scores
      : database->host_ssv_scores;
}

const std::vector<uint8_t> &database_host_exact_rbv(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->exact_rbv
      : database->host_exact_rbv;
}

const std::vector<int16_t> &database_host_emissions(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->emissions
      : database->host_emissions;
}

const std::vector<int16_t> &database_host_transitions(
    const plan7_viterbi_database *database) {
  return database->sealed_pack != nullptr
      ? database->sealed_pack->transitions
      : database->host_transitions;
}

template <typename T>
int grow_workspace_buffer(T **buffer, size_t *capacity, size_t required_bytes,
                          uint64_t *growth_count, const char *name,
                          char *error, size_t error_size) {
  if (required_bytes <= *capacity) return 0;
  T *replacement = nullptr;
  cudaError_t status = cudaMalloc(&replacement, required_bytes);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, name, status);
    return -1;
  }
  status = cudaFree(*buffer);
  if (status != cudaSuccess) {
    cudaFree(replacement);
    set_cuda_error(error, error_size, "cudaFree(post-filter workspace)", status);
    return -1;
  }
  *buffer = replacement;
  *capacity = required_bytes;
  ++*growth_count;
  return 0;
}

uint64_t postfilter_workspace_device_bytes(
    const plan7_postfilter_workspace *workspace) {
  if (workspace == nullptr) return 0;
  const size_t capacities[] = {
      workspace->states_capacity,       workspace->bias_inputs_capacity,
      workspace->bias_results_capacity, workspace->vit_results_capacity,
      workspace->moves_capacity,        workspace->msv_offsets_capacity,
      workspace->vit_offsets_capacity,  workspace->dp_capacity,
      workspace->results_capacity};
  uint64_t total = 0;
  for (const size_t capacity : capacities) {
    if (capacity > UINT64_MAX - total) return UINT64_MAX;
    total += static_cast<uint64_t>(capacity);
  }
  return total;
}

int destroy_postfilter_workspace_device(plan7_postfilter_workspace *workspace,
                                        char *error, size_t error_size) {
  if (workspace == nullptr) return 0;
  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool restore_device = false;
  bool device_ready = true;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess && original_device != workspace->device_ordinal) {
    status = cudaSetDevice(workspace->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice(workspace->device_ordinal);
    if (status != cudaSuccess) {
      first_error = status;
      device_ready = false;
    }
  }
#define CUDA_DESTROY_WORKSPACE(pointer)                                        \
  do {                                                                         \
    if (device_ready) {                                                        \
      status = cudaFree(pointer);                                              \
      if (status != cudaSuccess && first_error == cudaSuccess)                \
        first_error = status;                                                  \
    }                                                                          \
  } while (0)
  CUDA_DESTROY_WORKSPACE(workspace->device_results);
  CUDA_DESTROY_WORKSPACE(workspace->device_dp);
  CUDA_DESTROY_WORKSPACE(workspace->device_vit_offsets);
  CUDA_DESTROY_WORKSPACE(workspace->device_msv_offsets);
  CUDA_DESTROY_WORKSPACE(workspace->device_moves);
  CUDA_DESTROY_WORKSPACE(workspace->device_vit_results);
  CUDA_DESTROY_WORKSPACE(workspace->device_bias_results);
  CUDA_DESTROY_WORKSPACE(workspace->device_bias_inputs);
  CUDA_DESTROY_WORKSPACE(workspace->device_states);
#undef CUDA_DESTROY_WORKSPACE
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (status != cudaSuccess && first_error == cudaSuccess)
      first_error = status;
  }
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy post-filter workspace",
                   first_error);
    return -1;
  }
  return 0;
}

}  // namespace

bool live_profile_matches_snapshot(const plan7_viterbi_database *database,
                                   size_t profile_index) {
  if (database->sealed_source) return true;
  const auto *source = reinterpret_cast<const P7_OPROFILE *>(
      database->source_profile_pointers[profile_index]);
  const VitProfile &descriptor = database_host_profiles(database)[profile_index];
  const auto &host_scores = database_host_ssv_scores(database);
  const auto &host_exact_rbv = database_host_exact_rbv(database);
  const auto &host_emissions = database_host_emissions(database);
  const auto &host_transitions = database_host_transitions(database);
  if (!valid_profile_storage(source) || !valid_viterbi_specials(source) ||
      source->abc->Kp != database->alphabet_size ||
      reinterpret_cast<uintptr_t>(source->abc) !=
          database->source_alphabet_pointers[profile_index] ||
      source->M != static_cast<int>(descriptor.model_length) ||
      source->mode != descriptor.mode ||
      source->tbm_b != descriptor.msv_tbm ||
      source->tec_b != descriptor.msv_tec ||
      source->base_b != descriptor.msv_base ||
      source->bias_b != descriptor.msv_bias ||
      source->scale_b != descriptor.msv_scale ||
      source->base_w != descriptor.base ||
      source->ddbound_w != descriptor.ddbound ||
      source->scale_w != descriptor.scale || source->nj != descriptor.nj ||
      source->xw[p7O_E][p7O_MOVE] != descriptor.e_move ||
      source->xw[p7O_E][p7O_LOOP] != descriptor.e_loop ||
      source->xw[p7O_N][p7O_LOOP] != descriptor.n_loop ||
      source->xw[p7O_J][p7O_LOOP] != descriptor.j_loop ||
      source->xw[p7O_C][p7O_LOOP] != descriptor.c_loop)
    return false;
  for (int residue = 0; residue < source->abc->Kp; ++residue)
    if (source->sbv[residue] == nullptr || source->rbv[residue] == nullptr ||
        source->rwv[residue] == nullptr)
      return false;

  const int source_qw = p7O_NQW(source->M);
  const int source_qb = std::max(2, (source->M + 15) / 16);
  const int q_count = static_cast<int>(descriptor.q);
  for (int q = 0; q < q_count; ++q) {
    for (int lane = 0; lane < kWarpSize; ++lane) {
      const int model_position = q + q_count * lane + 1;
      if (model_position > source->M) continue;
      const int source_word_stripe = (model_position - 1) % source_qw;
      const int source_word_lane = (model_position - 1) / source_qw;
      const int source_byte_stripe = (model_position - 1) % source_qb;
      const int source_byte_lane = (model_position - 1) / source_qb;
      for (int residue = 0; residue < source->abc->Kp; ++residue) {
        const uint64_t ssv_index = descriptor.ssv_offset +
            static_cast<uint64_t>(model_position - 1) *
                source->abc->Kp + residue;
        const auto *source_sbv = reinterpret_cast<const uint8_t *>(
            source->sbv[residue] + source_byte_stripe);
        const auto *source_rbv = reinterpret_cast<const uint8_t *>(
            source->rbv[residue] + source_byte_stripe);
        if (source_sbv[source_byte_lane] !=
            host_scores[ssv_index])
          return false;
        if (descriptor.rbv_offset != UINT64_MAX) {
          const uint64_t rbv_index = descriptor.rbv_offset +
              static_cast<uint64_t>(model_position - 1) *
                  source->abc->Kp + residue;
          if (source_rbv[source_byte_lane] !=
              host_exact_rbv[rbv_index])
            return false;
        } else {
          const unsigned raw = source_sbv[source_byte_lane];
          const int signed_score = raw < 128 ? static_cast<int>(raw)
                                             : static_cast<int>(raw) - 256;
          const int decoded_value =
              signed_score + static_cast<int>(descriptor.msv_bias);
          const unsigned decoded = residue == 20 || residue == 27 ||
                                   residue == 28
              ? UINT8_MAX
              : static_cast<unsigned>(decoded_value < 0 ? 0 :
                                      (decoded_value > 255 ? 255 :
                                       decoded_value));
          if (source_rbv[source_byte_lane] != decoded) return false;
        }
        const uint64_t emission_index = descriptor.emission_offset +
            (static_cast<uint64_t>(residue) * q_count + q) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->rwv[residue] + source_word_stripe);
        if (source_words[source_word_lane] !=
            host_emissions[emission_index])
          return false;
      }
      for (int transition = p7O_BM; transition <= p7O_II;
           ++transition) {
        const uint64_t transition_index = descriptor.transition_offset +
            (static_cast<uint64_t>(q) * p7O_NTRANS + transition) *
                kWarpSize + lane;
        const auto *source_words = reinterpret_cast<const int16_t *>(
            source->twv + source_word_stripe * 7 + transition);
        if (source_words[source_word_lane] !=
            host_transitions[transition_index])
          return false;
      }
      const uint64_t dd_index = descriptor.transition_offset +
          (static_cast<uint64_t>(q) * p7O_NTRANS + p7O_DD) *
              kWarpSize + lane;
      const auto *dd_words = reinterpret_cast<const int16_t *>(
          source->twv + 7 * source_qw + source_word_stripe);
      if (dd_words[source_word_lane] !=
          host_transitions[dd_index])
        return false;
    }
  }
  return true;
}

namespace {

void initialize_viterbi_database(plan7_viterbi_database *database) {
  database->device_ordinal = -1;
  database->alphabet_size = 29;
  database->sealed_source = false;
  database->device_profiles = nullptr;
  database->device_emissions = nullptr;
  database->device_transitions = nullptr;
  database->device_exact_rbv = nullptr;
  database->emission_count = 0;
  database->transition_count = 0;
  database->exact_rbv_count = 0;
}

void release_partial_viterbi_upload(plan7_viterbi_database *database) {
  cudaFree(database->device_exact_rbv);
  cudaFree(database->device_transitions);
  cudaFree(database->device_emissions);
  cudaFree(database->device_profiles);
  database->device_exact_rbv = nullptr;
  database->device_transitions = nullptr;
  database->device_emissions = nullptr;
  database->device_profiles = nullptr;
}

int upload_viterbi_database(plan7_viterbi_database *database, char *error,
                            size_t error_size) {
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  database->device_ordinal = current_device;
  const auto &profiles = database_host_profiles(database);
  const auto &emissions = database_host_emissions(database);
  const auto &transitions = database_host_transitions(database);
  const auto &exact_rbv = database_host_exact_rbv(database);
  database->emission_count = emissions.size();
  database->transition_count = transitions.size();
  database->exact_rbv_count = exact_rbv.size();
  if (profiles.empty()) return 0;

  size_t profile_bytes;
  size_t emission_bytes;
  size_t transition_bytes;
  if (!checked_bytes(profiles.size(), sizeof(VitProfile), &profile_bytes) ||
      !checked_bytes(emissions.size(), sizeof(int16_t), &emission_bytes) ||
      !checked_bytes(transitions.size(), sizeof(int16_t), &transition_bytes)) {
    set_error(error, error_size, "Viterbi device allocation size overflow");
    return -1;
  }
#define CUDA_UPLOAD(call)                                                     \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      release_partial_viterbi_upload(database);                               \
      return -1;                                                              \
    }                                                                         \
  } while (0)
  CUDA_UPLOAD(cudaMalloc(&database->device_profiles, profile_bytes));
  CUDA_UPLOAD(cudaMalloc(&database->device_emissions, emission_bytes));
  CUDA_UPLOAD(cudaMalloc(&database->device_transitions, transition_bytes));
  if (!exact_rbv.empty())
    CUDA_UPLOAD(cudaMalloc(&database->device_exact_rbv, exact_rbv.size()));
  CUDA_UPLOAD(cudaMemcpy(database->device_profiles, profiles.data(),
                         profile_bytes, cudaMemcpyHostToDevice));
  CUDA_UPLOAD(cudaMemcpy(database->device_emissions, emissions.data(),
                         emission_bytes, cudaMemcpyHostToDevice));
  CUDA_UPLOAD(cudaMemcpy(database->device_transitions, transitions.data(),
                         transition_bytes, cudaMemcpyHostToDevice));
  if (!exact_rbv.empty())
    CUDA_UPLOAD(cudaMemcpy(database->device_exact_rbv, exact_rbv.data(),
                           exact_rbv.size(), cudaMemcpyHostToDevice));
#undef CUDA_UPLOAD
  return 0;
}

}  // namespace

extern "C" int plan7_viterbi_database_create(
    const uintptr_t *profile_pointers, size_t profile_count,
    plan7_viterbi_database **database, char *error, size_t error_size) {
  if (database == nullptr || *database != nullptr ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid Viterbi database arguments");
    return -1;
  }
  std::unique_ptr<PackWorkerPool> workers;
  try {
    workers = std::make_unique<PackWorkerPool>(
        std::min<size_t>(16, profile_count));
  } catch (...) {
    set_error(error, error_size, "Viterbi profile worker launch failed");
    return -1;
  }
  HostProfilePack pack;
  if (snapshot_profiles(profile_pointers, profile_count, nullptr,
                        workers.get(), &pack, error, error_size) != 0)
    return -1;
  auto *created = new (std::nothrow) plan7_viterbi_database{};
  if (created == nullptr) {
    set_error(error, error_size, "Viterbi database allocation failed");
    return -1;
  }
  initialize_viterbi_database(created);
  try {
    created->host_profiles = std::move(pack.vit_profiles);
    created->host_ssv_scores = std::move(pack.ssv_scores);
    created->host_exact_rbv = std::move(pack.exact_rbv);
    created->host_emissions = std::move(pack.emissions);
    created->host_transitions = std::move(pack.transitions);
    if (profile_count != 0) {
      created->source_profile_pointers.assign(
          profile_pointers, profile_pointers + profile_count);
      created->source_alphabet_pointers.resize(profile_count);
      for (size_t profile_index = 0; profile_index < profile_count;
           ++profile_index) {
        const auto *source = reinterpret_cast<const P7_OPROFILE *>(
            profile_pointers[profile_index]);
        created->source_alphabet_pointers[profile_index] =
            reinterpret_cast<uintptr_t>(source->abc);
      }
    }
  } catch (...) {
    delete created;
    set_error(error, error_size, "Viterbi descriptor allocation failed");
    return -1;
  }
  if (upload_viterbi_database(created, error, error_size) != 0) {
    delete created;
    return -1;
  }
  *database = created;
  return 0;
}

extern "C" int plan7_viterbi_database_destroy(
    plan7_viterbi_database **database, char *error, size_t error_size) {
  if (database == nullptr) {
    set_error(error, error_size, "Viterbi database handle is null");
    return -1;
  }
  if (*database == nullptr) return 0;
  cudaError_t first_error = cudaSuccess;
  cudaError_t status;
  int original_device = -1;
  bool device_ready = true;
  bool restore_device = false;
  status = cudaGetDevice(&original_device);
  if (status == cudaSuccess &&
      original_device != (*database)->device_ordinal) {
    status = cudaSetDevice((*database)->device_ordinal);
    if (status == cudaSuccess)
      restore_device = true;
    else {
      first_error = status;
      device_ready = false;
    }
  } else if (status != cudaSuccess) {
    status = cudaSetDevice((*database)->device_ordinal);
    if (status != cudaSuccess) {
      first_error = status;
      device_ready = false;
    }
  }
#define CUDA_DESTROY(pointer)                                                 \
  do {                                                                        \
    if (device_ready) {                                                       \
      status = cudaFree(pointer);                                             \
      if (status != cudaSuccess && first_error == cudaSuccess)                \
        first_error = status;                                                 \
    }                                                                         \
  } while (0)
  CUDA_DESTROY((*database)->device_exact_rbv);
  CUDA_DESTROY((*database)->device_transitions);
  CUDA_DESTROY((*database)->device_emissions);
  CUDA_DESTROY((*database)->device_profiles);
#undef CUDA_DESTROY
  if (restore_device) {
    status = cudaSetDevice(original_device);
    if (status != cudaSuccess && first_error == cudaSuccess)
      first_error = status;
  }
  delete *database;
  *database = nullptr;
  if (first_error != cudaSuccess) {
    set_cuda_error(error, error_size, "destroy Viterbi database", first_error);
    return -1;
  }
  return 0;
}

extern "C" size_t plan7_viterbi_database_profile_count(
    const plan7_viterbi_database *database) {
  return database == nullptr ? 0 : database_host_profiles(database).size();
}

extern "C" int plan7_profile_session_create(
    const uintptr_t *profile_pointers, size_t profile_count,
    const float *background, size_t background_count,
    size_t build_worker_count, size_t selection_worker_count,
    plan7_profile_session **session, char *error, size_t error_size) {
  if (session == nullptr || *session != nullptr || background == nullptr ||
      background_count != 20 ||
      build_worker_count > profile_count ||
      selection_worker_count > profile_count ||
      (profile_count != 0 && profile_pointers == nullptr)) {
    set_error(error, error_size, "invalid profile session arguments");
    return -1;
  }
  if (plan7_bias_host_environment_attested() != 1) {
    set_error(error, error_size,
              "profile session requires the attested host environment");
    return -1;
  }
  for (size_t residue = 0; residue < background_count; ++residue) {
    if (!isfinite(background[residue]) || background[residue] <= 0.0f) {
      set_error(error, error_size, "invalid profile session background");
      return -1;
    }
  }

  std::unique_ptr<PackWorkerPool> build_workers;
  try {
    build_workers = std::make_unique<PackWorkerPool>(build_worker_count);
  } catch (...) {
    set_error(error, error_size, "profile session build worker launch failed");
    return -1;
  }
  HostProfilePack pack;
  if (snapshot_profiles(profile_pointers, profile_count, background,
                        build_workers.get(), &pack, error, error_size) != 0)
    return -1;
  const uint64_t build_parallel_run_count =
      build_workers->parallel_run_count();
  build_workers.reset();

  uint64_t session_id;
  if (!claim_profile_session_id(&session_id)) {
    set_error(error, error_size, "profile session identity exhausted");
    return -1;
  }
  uint64_t first_identity_token;
  if (!claim_profile_identity_tokens(profile_count, &first_identity_token)) {
    set_error(error, error_size, "profile identity token space exhausted");
    return -1;
  }
  try {
    pack.identity_tokens.resize(profile_count);
    for (size_t profile_index = 0; profile_index < profile_count;
         ++profile_index)
      pack.identity_tokens[profile_index] =
          static_cast<uintptr_t>(first_identity_token + profile_index);
  } catch (...) {
    set_error(error, error_size, "profile session identity allocation failed");
    return -1;
  }

  std::unique_ptr<PackWorkerPool> selection_workers;
  try {
    selection_workers =
        std::make_unique<PackWorkerPool>(selection_worker_count);
  } catch (...) {
    set_error(error, error_size,
              "profile session selection worker launch failed");
    return -1;
  }

  auto *created = new (std::nothrow) plan7_profile_session{};
  if (created == nullptr) {
    set_error(error, error_size, "profile session allocation failed");
    return -1;
  }
  try {
    created->session_id = session_id;
    created->build_worker_count = build_worker_count;
    created->build_parallel_run_count = build_parallel_run_count;
    created->selection_count = 0;
    created->pack = std::make_shared<const HostProfilePack>(std::move(pack));
    created->selection_workers = std::move(selection_workers);
  } catch (...) {
    delete created;
    set_error(error, error_size, "profile session allocation failed");
    return -1;
  }
  *session = created;
  return 0;
}

extern "C" int plan7_profile_session_destroy(
    plan7_profile_session **session, char *error, size_t error_size) {
  if (session == nullptr) {
    set_error(error, error_size, "profile session handle is null");
    return -1;
  }
  plan7_profile_session *value = *session;
  *session = nullptr;
  delete value;
  return 0;
}

extern "C" int plan7_profile_session_get_statistics(
    const plan7_profile_session *session,
    plan7_profile_session_statistics *statistics,
    char *error, size_t error_size) {
  if (session == nullptr || statistics == nullptr || session->pack == nullptr ||
      session->selection_workers == nullptr) {
    set_error(error, error_size, "invalid profile session statistics");
    return -1;
  }
  const HostProfilePack &pack = *session->pack;
  *statistics = {};
  statistics->session_id = session->session_id;
  statistics->profile_count = pack.vit_profiles.size();
  statistics->worker_count = session->selection_workers->worker_count();
  statistics->build_worker_count = session->build_worker_count;
  statistics->selection_worker_count =
      session->selection_workers->worker_count();
  statistics->selection_count = session->selection_count;
  statistics->build_parallel_run_count =
      session->build_parallel_run_count;
  statistics->selection_parallel_run_count =
      session->selection_workers->parallel_run_count();
  statistics->parallel_run_count =
      statistics->selection_parallel_run_count >
              UINT64_MAX - statistics->build_parallel_run_count
          ? UINT64_MAX
          : statistics->build_parallel_run_count +
                statistics->selection_parallel_run_count;
  statistics->host_bytes = host_pack_bytes(pack);
  statistics->ssv_score_bytes = pack.ssv_scores.size();
  statistics->bias_profile_bytes =
      static_cast<uint64_t>(pack.bias_templates.size()) *
      sizeof(plan7_bias_profile);
  statistics->viterbi_descriptor_bytes =
      static_cast<uint64_t>(pack.vit_profiles.size()) * sizeof(VitProfile);
  statistics->viterbi_emission_bytes =
      static_cast<uint64_t>(pack.emissions.size()) * sizeof(int16_t);
  statistics->viterbi_transition_bytes =
      static_cast<uint64_t>(pack.transitions.size()) * sizeof(int16_t);
  statistics->viterbi_exact_rbv_bytes = pack.exact_rbv.size();
  statistics->forward_descriptor_bytes =
      static_cast<uint64_t>(pack.forward_profiles.size()) *
      sizeof(plan7_forward_snapshot_profile);
  statistics->forward_emission_bytes =
      static_cast<uint64_t>(pack.forward_emissions.size()) * sizeof(float);
  statistics->forward_transition_bytes =
      static_cast<uint64_t>(pack.forward_transitions.size()) * sizeof(float);
  return 0;
}

extern "C" int plan7_profile_session_select(
    plan7_profile_session *session, const size_t *profile_indices,
    size_t profile_count, plan7_profile_selection **selection,
    char *error, size_t error_size) {
  if (session == nullptr || session->pack == nullptr ||
      session->selection_workers == nullptr || selection == nullptr ||
      *selection != nullptr ||
      (profile_count != 0 && profile_indices == nullptr)) {
    set_error(error, error_size, "invalid profile selection arguments");
    return -1;
  }
  std::lock_guard<std::mutex> operation(session->operation_mutex);
  const HostProfilePack &source = *session->pack;
  HostProfilePack selected;
  selected.alphabet_size = source.alphabet_size;
  std::vector<uint8_t> seen;
  uint64_t ssv_total = 0;
  uint64_t rbv_total = 0;
  uint64_t emission_total = 0;
  uint64_t transition_total = 0;
  uint64_t forward_emission_total = 0;
  uint64_t forward_transition_total = 0;
  try {
    seen.assign(source.vit_profiles.size(), 0);
    selected.vit_profiles.resize(profile_count);
    selected.ssv_profiles.resize(profile_count);
    selected.m_mu.resize(profile_count);
    selected.m_lambda.resize(profile_count);
    selected.v_mu.resize(profile_count);
    selected.v_lambda.resize(profile_count);
    selected.bias_templates.resize(profile_count);
    selected.forward_profiles.resize(profile_count);
    selected.identity_tokens.resize(profile_count);
  } catch (...) {
    set_error(error, error_size, "profile selection allocation failed");
    return -1;
  }

  for (size_t output_index = 0; output_index < profile_count;
       ++output_index) {
    const size_t source_index = profile_indices[output_index];
    if (source_index >= source.vit_profiles.size()) {
      set_error(error, error_size, "profile selection index is out of range");
      return -1;
    }
    if (seen[source_index] != 0) {
      set_error(error, error_size, "profile selection indexes must be unique");
      return -1;
    }
    seen[source_index] = 1;
    const VitProfile &from = source.vit_profiles[source_index];
    const uint64_t ssv_count =
        static_cast<uint64_t>(from.model_length) * source.alphabet_size;
    const uint64_t rbv_count =
        from.rbv_offset == UINT64_MAX ? 0 : ssv_count;
    const uint64_t emission_count =
        static_cast<uint64_t>(from.q) * kWarpSize * source.alphabet_size;
    const uint64_t transition_count =
        static_cast<uint64_t>(from.q) * kWarpSize * p7O_NTRANS;
    const plan7_forward_snapshot_profile &forward_from =
        source.forward_profiles[source_index];
    const uint64_t forward_emission_count =
        static_cast<uint64_t>(forward_from.q) * kForwardSubwarp *
        source.alphabet_size;
    const uint64_t forward_transition_count =
        static_cast<uint64_t>(forward_from.q) * kForwardSubwarp * p7O_NTRANS;
    VitProfile to = from;
    to.ssv_offset = ssv_total;
    to.rbv_offset = rbv_count == 0 ? UINT64_MAX : rbv_total;
    to.emission_offset = emission_total;
    to.transition_offset = transition_total;
    if (!checked_add(ssv_total, ssv_count, &ssv_total) ||
        !checked_add(rbv_total, rbv_count, &rbv_total) ||
        !checked_add(emission_total, emission_count, &emission_total) ||
        !checked_add(transition_total, transition_count, &transition_total) ||
        !checked_add(forward_emission_total, forward_emission_count,
                     &forward_emission_total) ||
        !checked_add(forward_transition_total, forward_transition_count,
                     &forward_transition_total) ||
        ssv_total > SIZE_MAX || rbv_total > SIZE_MAX ||
        emission_total > SIZE_MAX || transition_total > SIZE_MAX ||
        forward_emission_total > SIZE_MAX ||
        forward_transition_total > SIZE_MAX) {
      set_error(error, error_size, "profile selection size overflow");
      return -1;
    }
    selected.vit_profiles[output_index] = to;
    selected.ssv_profiles[output_index] = source.ssv_profiles[source_index];
    selected.ssv_profiles[output_index].score_offset = to.ssv_offset;
    selected.m_mu[output_index] = source.m_mu[source_index];
    selected.m_lambda[output_index] = source.m_lambda[source_index];
    selected.v_mu[output_index] = source.v_mu[source_index];
    selected.v_lambda[output_index] = source.v_lambda[source_index];
    selected.bias_templates[output_index] =
        source.bias_templates[source_index];
    selected.forward_profiles[output_index] = forward_from;
    selected.forward_profiles[output_index].emission_offset =
        forward_emission_total - forward_emission_count;
    selected.forward_profiles[output_index].transition_offset =
        forward_transition_total - forward_transition_count;
    selected.identity_tokens[output_index] =
        source.identity_tokens[source_index];
  }
  try {
    selected.ssv_scores.resize(static_cast<size_t>(ssv_total));
    selected.exact_rbv.resize(static_cast<size_t>(rbv_total));
    selected.emissions.resize(static_cast<size_t>(emission_total));
    selected.transitions.resize(static_cast<size_t>(transition_total));
    selected.forward_emissions.resize(
        static_cast<size_t>(forward_emission_total));
    selected.forward_transitions.resize(
        static_cast<size_t>(forward_transition_total));
  } catch (...) {
    set_error(error, error_size, "profile selection data allocation failed");
    return -1;
  }
  SelectionCopyTask task{&source, &selected, profile_indices};
  session->selection_workers->parallel_for(
      profile_count, &task, copy_selection_profile);

  if (session->selection_count == UINT64_MAX) {
    set_error(error, error_size, "profile selection identity exhausted");
    return -1;
  }
  auto *created = new (std::nothrow) plan7_profile_selection{};
  if (created == nullptr) {
    set_error(error, error_size, "profile selection allocation failed");
    return -1;
  }
  try {
    created->session_id = session->session_id;
    created->selection_id = session->selection_count + 1;
    created->pack =
        std::make_shared<const HostProfilePack>(std::move(selected));
  } catch (...) {
    delete created;
    set_error(error, error_size, "profile selection allocation failed");
    return -1;
  }
  ++session->selection_count;
  *selection = created;
  return 0;
}

extern "C" int plan7_profile_selection_destroy(
    plan7_profile_selection **selection, char *error, size_t error_size) {
  if (selection == nullptr) {
    set_error(error, error_size, "profile selection handle is null");
    return -1;
  }
  plan7_profile_selection *value = *selection;
  *selection = nullptr;
  delete value;
  return 0;
}

extern "C" int plan7_profile_selection_get_view(
    const plan7_profile_selection *selection,
    plan7_profile_selection_view *view,
    char *error, size_t error_size) {
  if (selection == nullptr || selection->pack == nullptr || view == nullptr) {
    set_error(error, error_size, "invalid profile selection view");
    return -1;
  }
  const HostProfilePack &pack = *selection->pack;
  *view = {};
  view->session_id = selection->session_id;
  view->selection_id = selection->selection_id;
  view->profile_count = pack.ssv_profiles.size();
  view->packed_scores = pack.ssv_scores.empty()
      ? nullptr : pack.ssv_scores.data();
  view->packed_score_count = pack.ssv_scores.size();
  view->profiles = pack.ssv_profiles.empty()
      ? nullptr : pack.ssv_profiles.data();
  view->m_mu = pack.m_mu.empty() ? nullptr : pack.m_mu.data();
  view->m_lambda = pack.m_lambda.empty() ? nullptr : pack.m_lambda.data();
  view->v_mu = pack.v_mu.empty() ? nullptr : pack.v_mu.data();
  view->v_lambda = pack.v_lambda.empty() ? nullptr : pack.v_lambda.data();
  view->bias_templates = pack.bias_templates.empty()
      ? nullptr : pack.bias_templates.data();
  view->identity_tokens = pack.identity_tokens.empty()
      ? nullptr : pack.identity_tokens.data();
  view->host_bytes = host_pack_bytes(pack);
  return 0;
}

extern "C" int plan7_profile_selection_stage_viterbi(
    const plan7_profile_selection *selection,
    plan7_viterbi_database **database,
    char *error, size_t error_size) {
  if (selection == nullptr || selection->pack == nullptr ||
      database == nullptr || *database != nullptr) {
    set_error(error, error_size, "invalid staged Viterbi arguments");
    return -1;
  }
  auto *created = new (std::nothrow) plan7_viterbi_database{};
  if (created == nullptr) {
    set_error(error, error_size, "staged Viterbi allocation failed");
    return -1;
  }
  initialize_viterbi_database(created);
  created->sealed_source = true;
  created->sealed_pack = selection->pack;
  created->alphabet_size = selection->pack->alphabet_size;
  try {
    created->source_profile_pointers = selection->pack->identity_tokens;
  } catch (...) {
    delete created;
    set_error(error, error_size, "staged Viterbi identity allocation failed");
    return -1;
  }
  if (upload_viterbi_database(created, error, error_size) != 0) {
    delete created;
    return -1;
  }
  *database = created;
  return 0;
}

extern "C" int plan7_profile_selection_stage_forward(
    const plan7_profile_selection *selection,
    plan7_forward_database **database,
    char *error, size_t error_size) {
  if (selection == nullptr || selection->pack == nullptr) {
    set_error(error, error_size, "invalid staged Forward selection");
    return -1;
  }
  const HostProfilePack &pack = *selection->pack;
  return plan7_forward_database_create_snapshot(
      pack.alphabet_size,
      pack.forward_profiles.empty() ? nullptr : pack.forward_profiles.data(),
      pack.forward_profiles.size(),
      pack.forward_emissions.empty() ? nullptr : pack.forward_emissions.data(),
      pack.forward_emissions.size(),
      pack.forward_transitions.empty()
          ? nullptr : pack.forward_transitions.data(),
      pack.forward_transitions.size(),
      pack.identity_tokens.empty() ? nullptr : pack.identity_tokens.data(),
      database, error, error_size);
}

extern "C" int plan7_postfilter_workspace_create(
    plan7_postfilter_workspace **workspace, char *error, size_t error_size) {
  if (workspace == nullptr || *workspace != nullptr) {
    set_error(error, error_size, "invalid post-filter workspace output");
    return -1;
  }
  int current_device = -1;
  const cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  auto *created = new (std::nothrow) plan7_postfilter_workspace{};
  if (created == nullptr) {
    set_error(error, error_size, "post-filter workspace allocation failed");
    return -1;
  }
  created->device_ordinal = current_device;
  *workspace = created;
  return 0;
}

extern "C" int plan7_postfilter_workspace_destroy(
    plan7_postfilter_workspace **workspace, char *error, size_t error_size) {
  if (workspace == nullptr) {
    set_error(error, error_size, "post-filter workspace handle is null");
    return -1;
  }
  plan7_postfilter_workspace *value = *workspace;
  *workspace = nullptr;
  if (value == nullptr) return 0;
  const int status = destroy_postfilter_workspace_device(
      value, error, error_size);
  delete value;
  return status;
}

extern "C" int plan7_postfilter_workspace_get_statistics(
    const plan7_postfilter_workspace *workspace,
    plan7_postfilter_workspace_statistics *statistics,
    char *error, size_t error_size) {
  if (workspace == nullptr || statistics == nullptr) {
    set_error(error, error_size, "invalid post-filter workspace statistics");
    return -1;
  }
  *statistics = {};
  statistics->device_bytes = postfilter_workspace_device_bytes(workspace);
  statistics->dp_capacity_bytes =
      static_cast<uint64_t>(workspace->dp_capacity);
  statistics->growth_count = workspace->growth_count;
  statistics->run_count = workspace->run_count;
  const size_t capacities[PLAN7_POSTFILTER_CAPACITY_COUNT] = {
      workspace->states_capacity,
      workspace->bias_inputs_capacity,
      workspace->bias_results_capacity,
      workspace->vit_results_capacity,
      workspace->moves_capacity,
      workspace->msv_offsets_capacity,
      workspace->vit_offsets_capacity,
      workspace->dp_capacity,
      workspace->results_capacity};
  for (size_t i = 0; i < PLAN7_POSTFILTER_CAPACITY_COUNT; ++i)
    statistics->capacity_bytes[i] = static_cast<uint64_t>(capacities[i]);
  return 0;
}

extern "C" int plan7_viterbi_database_matches_ssv(
    const plan7_viterbi_database *database,
    const uint8_t *packed_scores, size_t packed_score_count,
    const uintptr_t *source_profile_pointers,
    const plan7_ssv_f1_profile *profiles, size_t profile_count,
    char *error, size_t error_size) {
  if (database == nullptr ||
      (profile_count != 0 &&
       (profiles == nullptr || packed_scores == nullptr ||
        source_profile_pointers == nullptr)) ||
      database_host_profiles(database).size() != profile_count) {
    set_error(error, error_size, "Viterbi and SSV profile counts differ");
    return -1;
  }
  const auto &host_profiles = database_host_profiles(database);
  const auto &host_scores = database_host_ssv_scores(database);
  for (size_t p = 0; p < profile_count; ++p) {
    const VitProfile &vit = host_profiles[p];
    const plan7_ssv_profile &msv = profiles[p].profile;
    const uint64_t expected_count =
        static_cast<uint64_t>(vit.model_length) * database->alphabet_size;
    if (source_profile_pointers[p] !=
          database->source_profile_pointers[p] ||
        (!database->sealed_source &&
         !live_profile_matches_snapshot(database, p)) ||
        vit.model_length != static_cast<uint32_t>(msv.model_length) ||
        msv.score_offset != vit.ssv_offset ||
        msv.score_count != expected_count ||
        msv.score_stride != database->alphabet_size ||
        msv.score_offset > packed_score_count ||
        msv.score_count > packed_score_count - msv.score_offset ||
        vit.msv_tbm != msv.tbm || vit.msv_tec != msv.tec ||
        vit.msv_base != msv.base || vit.msv_bias != msv.bias ||
        vit.msv_scale != msv.scale ||
        std::memcmp(host_scores.data() + vit.ssv_offset,
                    packed_scores + msv.score_offset,
                    static_cast<size_t>(expected_count)) != 0) {
      set_error(error, error_size,
                "Viterbi database does not match the SSV profile row");
      return -1;
    }
  }
  if (packed_score_count != host_scores.size()) {
    set_error(error, error_size,
              "SSV profile scores have trailing bytes");
    return -1;
  }
  return 0;
}

namespace {

struct ScopedReasonFacts {
  uint16_t *device = nullptr;
  ~ScopedReasonFacts() { cudaFree(device); }
};

int postfilter_candidates_device_with_workspace_impl(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, uint16_t *host_reason_facts,
    char *error, size_t error_size) {
  if (workspace == nullptr || database == nullptr || device_residues == nullptr ||
      device_sequence_offsets == nullptr || host_sequence_lengths == nullptr ||
      device_null_scores == nullptr || device_compact_scores == nullptr ||
      device_f1_profiles == nullptr || device_tjb == nullptr ||
      device_length_logp == nullptr || device_length_log1mp == nullptr ||
      device_bias_profiles == nullptr || device_candidates == nullptr ||
      host_candidates == nullptr || device_msv_inputs == nullptr ||
      (candidate_count != 0 && host_results == nullptr)) {
    set_error(error, error_size, "invalid post-filter device buffers");
    return -1;
  }
  if (candidate_count == 0) return 0;
  int current_device = -1;
  cudaError_t status = cudaGetDevice(&current_device);
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "cudaGetDevice", status);
    return -1;
  }
  if (current_device != database->device_ordinal) {
    set_error(error, error_size,
              "Viterbi database belongs to a different CUDA device");
    return -1;
  }
  if (current_device != workspace->device_ordinal) {
    set_error(error, error_size,
              "post-filter workspace belongs to a different CUDA device");
    return -1;
  }
  ++workspace->run_count;

  if (candidate_count == SIZE_MAX) {
    set_error(error, error_size, "post-filter candidate count overflow");
    return -1;
  }
  std::vector<uint64_t> &host_msv_offsets = workspace->host_msv_offsets;
  std::vector<uint64_t> &host_vit_offsets = workspace->host_vit_offsets;
  std::vector<VitLengthTransitions> &host_moves = workspace->host_moves;
  std::vector<size_t> &msv_tiles = workspace->msv_tiles;
  std::vector<size_t> &vit_tiles = workspace->vit_tiles;
  uint64_t maximum_msv_bytes = 0;
  uint64_t maximum_vit_cells = 0;
  constexpr uint64_t kVitCellLimit = kDpByteLimit / sizeof(int16_t);
  try {
    host_msv_offsets.assign(candidate_count + 1, 0);
    host_vit_offsets.assign(candidate_count + 1, 0);
    host_moves.resize(candidate_count);
    msv_tiles.clear();
    vit_tiles.clear();
    msv_tiles.push_back(0);
    vit_tiles.push_back(0);
    for (size_t c = 0; c < candidate_count; ++c) {
      const plan7_bias_candidate mapping = host_candidates[c];
      const auto &profiles = database_host_profiles(database);
      if (mapping.profile_index >= profiles.size() ||
          mapping.sequence_index >= sequence_count ||
          host_sequence_lengths[mapping.sequence_index] > 100000) {
        set_error(error, error_size, "invalid post-filter candidate mapping");
        return -1;
      }
      const VitProfile &profile = profiles[mapping.profile_index];
      const int length = static_cast<int>(
          host_sequence_lengths[mapping.sequence_index]);
      host_moves[c] = length_transitions_for(profile, length);
      const uint64_t msv_cells =
          static_cast<uint64_t>(profile.q) * kWarpSize;
      const uint64_t vit_cells = msv_cells * 3;
      if (!checked_add(host_msv_offsets[c], msv_cells,
                       &host_msv_offsets[c + 1]) ||
          !checked_add(host_vit_offsets[c], vit_cells,
                       &host_vit_offsets[c + 1])) {
        set_error(error, error_size, "post-filter DP offset overflow");
        return -1;
      }
    }
    for (size_t begin = 0; begin < candidate_count;) {
      size_t end = begin + 1;
      while (end < candidate_count &&
             host_msv_offsets[end + 1] - host_msv_offsets[begin] <=
                 kDpByteLimit)
        ++end;
      maximum_msv_bytes = std::max(
          maximum_msv_bytes,
          host_msv_offsets[end] - host_msv_offsets[begin]);
      msv_tiles.push_back(end);
      begin = end;
    }
    for (size_t begin = 0; begin < candidate_count;) {
      size_t end = begin + 1;
      while (end < candidate_count &&
             host_vit_offsets[end + 1] - host_vit_offsets[begin] <=
                 kVitCellLimit)
        ++end;
      maximum_vit_cells = std::max(
          maximum_vit_cells,
          host_vit_offsets[end] - host_vit_offsets[begin]);
      vit_tiles.push_back(end);
      begin = end;
    }
  } catch (...) {
    set_error(error, error_size, "post-filter host workspace allocation failed");
    return -1;
  }
  const uint64_t dp_bytes_u64 = std::max(
      maximum_msv_bytes, maximum_vit_cells * sizeof(int16_t));
  if (dp_bytes_u64 > kDpByteLimit || dp_bytes_u64 > SIZE_MAX) {
    set_error(error, error_size, "post-filter DP tile exceeds 256 MiB");
    return -1;
  }

  size_t candidate_bytes;
  size_t offset_bytes;
  size_t move_bytes;
  size_t bias_input_bytes;
  size_t bias_result_bytes;
  size_t vit_result_bytes;
  size_t post_result_bytes;
  size_t reason_fact_bytes = 0;
  if (!checked_bytes(candidate_count + 1, sizeof(uint64_t), &offset_bytes) ||
      !checked_bytes(candidate_count, sizeof(VitLengthTransitions),
                     &move_bytes) ||
      !checked_bytes(candidate_count, sizeof(plan7_bias_ssv_input),
                     &bias_input_bytes) ||
      !checked_bytes(candidate_count, sizeof(plan7_bias_result),
                     &bias_result_bytes) ||
      !checked_bytes(candidate_count, sizeof(VitResult), &vit_result_bytes) ||
      !checked_bytes(candidate_count, sizeof(plan7_postfilter_result),
                     &post_result_bytes) ||
      (host_reason_facts != nullptr &&
       !checked_bytes(candidate_count, sizeof(uint16_t), &reason_fact_bytes)) ||
      !checked_bytes(candidate_count, 1, &candidate_bytes)) {
    set_error(error, error_size, "post-filter workspace size overflow");
    return -1;
  }

  const unsigned blocks = static_cast<unsigned>(
      (candidate_count - 1) / kThreads + 1);
  ScopedReasonFacts reason_storage;

#define CUDA_RUN(call)                                                        \
  do {                                                                        \
    status = (call);                                                          \
    if (status != cudaSuccess) {                                              \
      set_cuda_error(error, error_size, #call, status);                       \
      return -1;                                                              \
    }                                                                         \
  } while (0)

  if (grow_workspace_buffer(&workspace->device_states,
                            &workspace->states_capacity, candidate_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter states)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_bias_inputs,
                            &workspace->bias_inputs_capacity, bias_input_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter bias inputs)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_bias_results,
                            &workspace->bias_results_capacity,
                            bias_result_bytes, &workspace->growth_count,
                            "cudaMalloc(post-filter bias results)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_vit_results,
                            &workspace->vit_results_capacity, vit_result_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter Viterbi results)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_moves,
                            &workspace->moves_capacity, move_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter length transitions)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_msv_offsets,
                            &workspace->msv_offsets_capacity, offset_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter MSV offsets)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_vit_offsets,
                            &workspace->vit_offsets_capacity, offset_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter Viterbi offsets)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_dp, &workspace->dp_capacity,
                            static_cast<size_t>(dp_bytes_u64),
                            &workspace->growth_count,
                            "cudaMalloc(post-filter DP workspace)", error,
                            error_size) != 0 ||
      grow_workspace_buffer(&workspace->device_results,
                            &workspace->results_capacity, post_result_bytes,
                            &workspace->growth_count,
                            "cudaMalloc(post-filter results)", error,
                            error_size) != 0)
    return -1;
  if (host_reason_facts != nullptr)
    CUDA_RUN(cudaMalloc(&reason_storage.device, reason_fact_bytes));
  CUDA_RUN(cudaMemcpy(workspace->device_moves, host_moves.data(), move_bytes,
                      cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(workspace->device_msv_offsets, host_msv_offsets.data(),
                      offset_bytes, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(workspace->device_vit_offsets, host_vit_offsets.data(),
                      offset_bytes, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemset(workspace->device_vit_results, 0xff, vit_result_bytes));

  for (size_t tile = 0; tile + 1 < msv_tiles.size(); ++tile) {
    const size_t tile_begin = msv_tiles[tile];
    const size_t tile_count = msv_tiles[tile + 1] - tile_begin;
    full_msv_kernel<<<
        static_cast<unsigned>((tile_count - 1) / kWarpsPerBlock + 1),
        kThreads>>>(
        device_residues, device_sequence_offsets, device_compact_scores,
        database->device_exact_rbv, device_f1_profiles,
        database->device_profiles, device_tjb,
        device_candidates, workspace->device_msv_offsets, tile_begin, tile_count,
        host_msv_offsets[tile_begin],
        static_cast<uint8_t *>(workspace->device_dp),
        device_msv_inputs);
    CUDA_RUN(cudaGetLastError());
  }
  prepare_bias_inputs_kernel<<<blocks, kThreads>>>(
      device_null_scores, device_f1_profiles, device_candidates,
      device_msv_inputs, candidate_count, workspace->device_states,
      workspace->device_bias_inputs);
  CUDA_RUN(cudaGetLastError());
  if (plan7_bias_filter_candidates_device(
        device_residues, device_sequence_offsets, device_length_logp,
        device_length_log1mp, device_bias_profiles, device_candidates,
        workspace->device_bias_inputs, candidate_count,
        workspace->device_bias_results,
        error, error_size) != 0)
    return -1;

  for (size_t tile = 0; tile + 1 < vit_tiles.size(); ++tile) {
    const size_t tile_begin = vit_tiles[tile];
    const size_t tile_count = vit_tiles[tile + 1] - tile_begin;
    viterbi_kernel<<<
        static_cast<unsigned>((tile_count - 1) / kWarpsPerBlock + 1),
        kThreads>>>(
        device_residues, device_sequence_offsets, database->device_profiles,
        database->device_emissions, database->device_transitions,
        device_candidates, workspace->device_states,
        workspace->device_bias_results, workspace->device_moves,
        workspace->device_vit_offsets, tile_begin, tile_count,
        host_vit_offsets[tile_begin],
        static_cast<int16_t *>(workspace->device_dp),
        workspace->device_vit_results);
    CUDA_RUN(cudaGetLastError());
  }
  merge_results_kernel<<<blocks, kThreads>>>(
      device_candidates, device_msv_inputs, workspace->device_states,
      workspace->device_bias_results, workspace->device_vit_results,
      candidate_count, workspace->device_results);
  CUDA_RUN(cudaGetLastError());
  if (host_reason_facts != nullptr) {
    postfilter_reason_facts_kernel<<<blocks, kThreads>>>(
        device_bias_profiles, device_candidates, device_msv_inputs,
        workspace->device_states, workspace->device_bias_inputs,
        workspace->device_bias_results, workspace->device_vit_results,
        workspace->device_results, candidate_count, reason_storage.device);
    CUDA_RUN(cudaGetLastError());
    CUDA_RUN(cudaMemcpy(host_reason_facts, reason_storage.device,
                        reason_fact_bytes, cudaMemcpyDeviceToHost));
  }
  CUDA_RUN(cudaMemcpy(host_results, workspace->device_results, post_result_bytes,
                      cudaMemcpyDeviceToHost));
  return 0;
#undef CUDA_RUN
}

}  // namespace

extern "C" int plan7_postfilter_candidates_device_with_workspace(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, char *error, size_t error_size) {
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, nullptr, error, error_size);
}

extern "C" int plan7_postfilter_candidates_device_with_workspace_reason_facts(
    plan7_postfilter_workspace *workspace,
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, uint16_t *reason_facts,
    char *error, size_t error_size) {
  if (candidate_count != 0 && reason_facts == nullptr) {
    set_error(error, error_size, "post-filter reason output is null");
    return -1;
  }
  return postfilter_candidates_device_with_workspace_impl(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, reason_facts, error, error_size);
}

extern "C" int plan7_postfilter_candidates_device(
    const plan7_viterbi_database *database, const uint8_t *device_residues,
    const uint64_t *device_sequence_offsets,
    const uint64_t *host_sequence_lengths, size_t sequence_count,
    const float *device_null_scores, const uint8_t *device_compact_scores,
    const plan7_ssv_f1_profile *device_f1_profiles,
    const uint8_t *device_tjb, const float *device_length_logp,
    const float *device_length_log1mp,
    const plan7_bias_profile *device_bias_profiles,
    const plan7_bias_candidate *device_candidates,
    const plan7_bias_candidate *host_candidates,
    plan7_bias_ssv_input *device_msv_inputs, size_t candidate_count,
    plan7_postfilter_result *host_results, char *error, size_t error_size) {
  plan7_postfilter_workspace *workspace = nullptr;
  if (plan7_postfilter_workspace_create(&workspace, error, error_size) != 0)
    return -1;
  const int run_status = plan7_postfilter_candidates_device_with_workspace(
      workspace, database, device_residues, device_sequence_offsets,
      host_sequence_lengths, sequence_count, device_null_scores,
      device_compact_scores, device_f1_profiles, device_tjb,
      device_length_logp, device_length_log1mp, device_bias_profiles,
      device_candidates, host_candidates, device_msv_inputs, candidate_count,
      host_results, error, error_size);
  char destroy_error[512] = {0};
  const int destroy_status = plan7_postfilter_workspace_destroy(
      &workspace, destroy_error, sizeof(destroy_error));
  if (run_status != 0) return run_status;
  if (destroy_status != 0) {
    set_error(error, error_size, destroy_error);
    return -1;
  }
  return 0;
}
