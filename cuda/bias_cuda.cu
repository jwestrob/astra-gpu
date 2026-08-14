#include "bias_cuda.h"

#include <cuda_runtime.h>

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <link.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <dlfcn.h>

#if defined(__GLIBC__)
#include <cpuid.h>
#include <gnu/libc-version.h>
#include <xmmintrin.h>
#endif

static_assert(sizeof(float) == 4,
              "bias ABI requires binary32 float");
static_assert(sizeof(plan7_bias_profile) == 272,
              "plan7_bias_profile ABI size changed");
static_assert(sizeof(plan7_bias_candidate) == 8,
              "plan7_bias_candidate ABI size changed");
static_assert(sizeof(plan7_bias_ssv_input) == 4,
              "plan7_bias_ssv_input ABI size changed");
static_assert(sizeof(plan7_bias_result) == 12,
              "plan7_bias_result ABI size changed");

namespace {

constexpr int kThreads = 256;
constexpr uint64_t kMaximumTargetLength = 100000;
constexpr double kLog2 = 0.69314718055994529;

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

float
host_add(float left, float right)
{
  const float result = left + right;
  return result;
}

float
host_mul(float left, float right)
{
  const float result = left * right;
  return result;
}

float
host_div(float left, float right)
{
  const float result = left / right;
  return result;
}

bool
valid_profile_inputs(const float *background,
                     const float *composition,
                     int model_length,
                     float scale,
                     int cutoff_mode,
                     float cutoff_bit_score)
{
  if (background == nullptr || composition == nullptr || model_length < 1 ||
      model_length > static_cast<int>(kMaximumTargetLength) ||
      !isfinite(scale) || scale <= 0.0f ||
      cutoff_mode < PLAN7_BIAS_CUTOFF_INVALID ||
      cutoff_mode > PLAN7_BIAS_CUTOFF_ALWAYS_PASS ||
      (cutoff_mode == PLAN7_BIAS_CUTOFF_SCORE &&
       !isfinite(cutoff_bit_score)))
    return false;
  for (int residue = 0; residue < 20; ++residue) {
    if (!isfinite(background[residue]) || background[residue] <= 0.0f ||
        !isfinite(composition[residue]) || composition[residue] < 0.0f)
      return false;
  }
  return true;
}

struct build_id_search {
  uintptr_t symbol;
  bool module_found;
  bool build_id_matches;
};

int
find_attested_build_id(struct dl_phdr_info *info, size_t, void *opaque)
{
  constexpr unsigned char kExpectedBuildId[20] = {
    0xe6, 0xf0, 0x50, 0x69, 0x61, 0x20, 0xae, 0xb5, 0x13, 0x4a,
    0x14, 0xe3, 0x8b, 0xc5, 0x4e, 0x8c, 0xe1, 0xbd, 0xc0, 0xb5
  };
  build_id_search *search = static_cast<build_id_search *>(opaque);
  bool contains_symbol = false;
  for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
    const ElfW(Phdr) &header = info->dlpi_phdr[index];
    if (header.p_type != PT_LOAD) continue;
    const uintptr_t first =
      static_cast<uintptr_t>(info->dlpi_addr) + header.p_vaddr;
    const uintptr_t last = first + header.p_memsz;
    if (search->symbol >= first && search->symbol < last) {
      contains_symbol = true;
      break;
    }
  }
  if (!contains_symbol) return 0;
  search->module_found = true;

  for (ElfW(Half) index = 0; index < info->dlpi_phnum; ++index) {
    const ElfW(Phdr) &header = info->dlpi_phdr[index];
    if (header.p_type != PT_NOTE) continue;
    const unsigned char *cursor = reinterpret_cast<const unsigned char *>(
      static_cast<uintptr_t>(info->dlpi_addr) + header.p_vaddr);
    const unsigned char *end = cursor + header.p_memsz;
    while (cursor + sizeof(ElfW(Nhdr)) <= end) {
      const ElfW(Nhdr) *note = reinterpret_cast<const ElfW(Nhdr) *>(cursor);
      cursor += sizeof(*note);
      const size_t name_size = (note->n_namesz + 3U) & ~size_t(3U);
      const size_t descriptor_size = (note->n_descsz + 3U) & ~size_t(3U);
      if (name_size > static_cast<size_t>(end - cursor)) break;
      const unsigned char *name = cursor;
      cursor += name_size;
      if (descriptor_size > static_cast<size_t>(end - cursor)) break;
      const unsigned char *descriptor = cursor;
      cursor += descriptor_size;
      if (note->n_type == NT_GNU_BUILD_ID && note->n_namesz == 4 &&
          memcmp(name, "GNU", 4) == 0 &&
          note->n_descsz == sizeof(kExpectedBuildId) &&
          memcmp(descriptor, kExpectedBuildId, sizeof(kExpectedBuildId)) == 0) {
        search->build_id_matches = true;
        return 1;
      }
    }
  }
  return 1;
}

bool
loaded_libm_is_attested()
{
  void *symbol = dlsym(RTLD_DEFAULT, "log");
  if (symbol == nullptr) return false;
  build_id_search search = {
    reinterpret_cast<uintptr_t>(symbol), false, false
  };
  dl_iterate_phdr(find_attested_build_id, &search);
  return search.module_found && search.build_id_matches;
}

void
set_degenerate_odds(const float *background,
                    const float *composition,
                    uint32_t mask,
                    float odds[2])
{
  float normal_numerator = 0.0f;
  float bias_numerator = 0.0f;
  float denominator = 0.0f;
  for (int residue = 0; residue < 20; ++residue) {
    if ((mask & (UINT32_C(1) << residue)) == 0) continue;
    normal_numerator = host_add(normal_numerator, background[residue]);
    bias_numerator = host_add(bias_numerator, composition[residue]);
    denominator = host_add(denominator, background[residue]);
  }
  odds[0] = denominator > 0.0f
              ? host_div(normal_numerator, denominator)
              : 0.0f;
  odds[1] = denominator > 0.0f
              ? host_div(bias_numerator, denominator)
              : 0.0f;
}

__device__ __forceinline__ float
device_add(float left, float right)
{
  return __fadd_rn(left, right);
}

__device__ __forceinline__ float
device_mul(float left, float right)
{
  return __fmul_rn(left, right);
}

__device__ __forceinline__ float
device_div(float left, float right)
{
  return __fdiv_rn(left, right);
}

__device__ __forceinline__ float
device_log_to_float(float value)
{
  return __double2float_rn(log(static_cast<double>(value)));
}

__device__ __forceinline__ bool
bias_filter_score(const uint8_t *residues,
                  const uint64_t *offsets,
                  uint32_t sequence,
                  const plan7_bias_profile &profile,
                  float length_logp,
                  float length_log1mp,
                  float *filtersc)
{
  const uint64_t start = offsets[sequence];
  const uint64_t length_u64 = offsets[sequence + 1] - start;
  if (length_u64 == 0 || length_u64 > kMaximumTargetLength) return false;
  const int length = static_cast<int>(length_u64);
  const float length_f = __int2float_rn(length);
  const float p1 = device_div(length_f, __int2float_rn(length + 1));
  const float t00 = p1;
  const float t01 = __fsub_rn(1.0f, p1);

  unsigned residue = residues[start];
  float d0 = device_mul(profile.eo[residue][0], profile.pi0);
  float d1 = device_mul(profile.eo[residue][1], profile.pi1);
  float maximum = d0 > 0.0f ? d0 : 0.0f;
  maximum = d1 > maximum ? d1 : maximum;
  if (!(maximum > 0.0f) || !isfinite(maximum)) return false;
  d0 = device_div(d0, maximum);
  d1 = device_div(d1, maximum);
  float score = device_log_to_float(maximum);

  for (int position = 1; position < length; ++position) {
    residue = residues[start + static_cast<uint64_t>(position)];
    float next0 = device_add(device_mul(d0, t00),
                             device_mul(d1, profile.t10));
    next0 = device_mul(next0, profile.eo[residue][0]);
    float next1 = device_add(device_mul(d0, t01),
                             device_mul(d1, profile.t11));
    next1 = device_mul(next1, profile.eo[residue][1]);
    maximum = next0 > 0.0f ? next0 : 0.0f;
    maximum = next1 > maximum ? next1 : maximum;
    if (!(maximum > 0.0f) || !isfinite(maximum)) return false;
    d0 = device_div(next0, maximum);
    d1 = device_div(next1, maximum);
    score = device_add(score, device_log_to_float(maximum));
  }

  const float end = device_add(device_mul(d0, profile.t02),
                               device_mul(d1, profile.t12));
  if (!(end > 0.0f) || !isfinite(end)) return false;
  score = device_add(score, device_log_to_float(end));
  score = device_add(score, length_logp);
  score = device_add(score, length_log1mp);
  if (!isfinite(score)) return false;
  *filtersc = score;
  return true;
}

__global__ void
bias_candidates_kernel(const uint8_t *residues,
                       const uint64_t *offsets,
                       const float *length_logp,
                       const float *length_log1mp,
                       const plan7_bias_profile *profiles,
                       const plan7_bias_candidate *candidates,
                       const plan7_bias_ssv_input *ssv_inputs,
                       size_t candidate_count,
                       plan7_bias_result *results)
{
  const size_t candidate_index =
    static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (candidate_index >= candidate_count) return;

  const plan7_bias_candidate candidate = candidates[candidate_index];
  const plan7_bias_ssv_input ssv = ssv_inputs[candidate_index];
  const plan7_bias_profile &profile = profiles[candidate.profile_index];
  plan7_bias_result result = {
    candidate.sequence_index, NAN, ssv.numerator, ssv.status,
    PLAN7_BIAS_CPU_REQUIRED
  };
  if (ssv.status != 0) {
    results[candidate_index] = result;
    return;
  }

  float filtersc;
  if (!bias_filter_score(residues, offsets, candidate.sequence_index, profile,
                         length_logp[candidate.sequence_index],
                         length_log1mp[candidate.sequence_index], &filtersc)) {
    results[candidate_index] = result;
    return;
  }
  result.filtersc = filtersc;

  float usc = __int2float_rn(static_cast<int>(ssv.numerator));
  usc = device_div(usc, profile.scale);
  usc = __fsub_rn(usc, 3.0f);
  const float delta = __fsub_rn(usc, filtersc);
  const float bit_score = __double2float_rn(
    __ddiv_rn(static_cast<double>(delta), kLog2));
  if (!isfinite(usc) || !isfinite(bit_score)) {
    results[candidate_index] = result;
    return;
  }

  switch (profile.cutoff_mode) {
    case PLAN7_BIAS_CUTOFF_SCORE:
      if (isfinite(profile.cutoff_bit_score))
        result.action = bit_score < profile.cutoff_bit_score
                          ? PLAN7_BIAS_DEFINITE_REJECT
                          : PLAN7_BIAS_DEFINITE_PASS;
      break;
    case PLAN7_BIAS_CUTOFF_ALWAYS_REJECT:
      result.action = PLAN7_BIAS_DEFINITE_REJECT;
      break;
    case PLAN7_BIAS_CUTOFF_ALWAYS_PASS:
      result.action = PLAN7_BIAS_DEFINITE_PASS;
      break;
    default:
      break;
  }
  results[candidate_index] = result;
}

}  // namespace

extern "C" int
plan7_bias_pack_amino_profile(const float *background,
                              const float *composition,
                              int model_length,
                              float scale,
                              int cutoff_mode,
                              float cutoff_bit_score,
                              plan7_bias_profile *profile,
                              char *error,
                              size_t error_size)
{
  if (plan7_bias_host_environment_attested() != 1) {
    set_error(error, error_size,
              "bias host floating-point environment is not attested");
    return -1;
  }
  if (profile == nullptr ||
      !valid_profile_inputs(background, composition, model_length, scale,
                            cutoff_mode, cutoff_bit_score)) {
    set_error(error, error_size, "invalid amino bias profile");
    return -1;
  }

  memset(profile, 0, sizeof(*profile));
  const float biased_length = host_div(static_cast<float>(model_length), 8.0f);
  const float biased_denominator = host_add(biased_length, 1.0f);
  profile->t10 = host_div(1.0f, biased_denominator);
  profile->t11 = host_div(biased_length, biased_denominator);
  profile->scale = scale;
  profile->cutoff_mode = cutoff_mode;
  profile->cutoff_bit_score = cutoff_bit_score;
  profile->pi0 = 0x1.ff7ceep-1f;
  profile->pi1 = 0x1.0624dep-10f;
  profile->t02 = 1.0f;
  profile->t12 = 1.0f;

  for (int residue = 0; residue < 20; ++residue) {
    profile->eo[residue][0] = host_div(background[residue],
                                       background[residue]);
    profile->eo[residue][1] = host_div(composition[residue],
                                       background[residue]);
  }

  profile->eo[20][0] = profile->eo[20][1] = 1.0f;
  set_degenerate_odds(background, composition,
                      (UINT32_C(1) << 2) | (UINT32_C(1) << 11),
                      profile->eo[21]);
  set_degenerate_odds(background, composition,
                      (UINT32_C(1) << 7) | (UINT32_C(1) << 9),
                      profile->eo[22]);
  set_degenerate_odds(background, composition,
                      (UINT32_C(1) << 3) | (UINT32_C(1) << 13),
                      profile->eo[23]);
  set_degenerate_odds(background, composition, UINT32_C(1) << 8,
                      profile->eo[24]);
  set_degenerate_odds(background, composition, UINT32_C(1) << 1,
                      profile->eo[25]);
  set_degenerate_odds(background, composition, UINT32_C(0x000fffff),
                      profile->eo[26]);
  profile->eo[27][0] = profile->eo[27][1] = 1.0f;
  profile->eo[28][0] = profile->eo[28][1] = 1.0f;
  return 0;
}

extern "C" int
plan7_bias_length_terms(uint64_t length,
                        float *length_logp,
                        float *length_log1mp)
{
  if (plan7_bias_host_environment_attested() != 1 ||
      length == 0 || length > kMaximumTargetLength ||
      length_logp == nullptr || length_log1mp == nullptr)
    return -1;
  const float length_f = static_cast<float>(length);
  const float p1 = host_div(length_f, static_cast<float>(length + 1));
  *length_logp = host_mul(length_f, logf(p1));
  *length_log1mp = logf(1.0f - p1);
  return isfinite(*length_logp) && isfinite(*length_log1mp) ? 0 : -1;
}

extern "C" int
plan7_bias_filter_score_host(const plan7_bias_profile *profile,
                             const uint8_t *residues,
                             uint64_t length,
                             float *filtersc)
{
  if (plan7_bias_host_environment_attested() != 1 ||
      profile == nullptr || residues == nullptr || filtersc == nullptr ||
      length == 0 || length > kMaximumTargetLength)
    return -1;
  for (uint64_t position = 0; position < length; ++position)
    if (residues[position] >= 29) return -1;

  const float length_f = static_cast<float>(length);
  const float p1 = host_div(length_f, static_cast<float>(length + 1));
  const float t00 = p1;
  const float t01 = 1.0f - p1;
  unsigned residue = residues[0];
  if (!isfinite(profile->pi0) || !isfinite(profile->pi1) ||
      !isfinite(profile->t02) || !isfinite(profile->t12) ||
      profile->pi0 < 0.0f || profile->pi1 < 0.0f ||
      profile->t02 < 0.0f || profile->t12 < 0.0f)
    return -1;
  float d0 = host_mul(profile->eo[residue][0], profile->pi0);
  float d1 = host_mul(profile->eo[residue][1], profile->pi1);
  float maximum = d0 > 0.0f ? d0 : 0.0f;
  maximum = d1 > maximum ? d1 : maximum;
  if (!(maximum > 0.0f) || !isfinite(maximum)) return -1;
  d0 = host_div(d0, maximum);
  d1 = host_div(d1, maximum);
  float score = static_cast<float>(log(static_cast<double>(maximum)));

  for (uint64_t position = 1; position < length; ++position) {
    residue = residues[position];
    float next0 = 0.0f;
    next0 = host_add(next0, host_mul(d0, t00));
    next0 = host_add(next0, host_mul(d1, profile->t10));
    next0 = host_mul(next0, profile->eo[residue][0]);
    float next1 = 0.0f;
    next1 = host_add(next1, host_mul(d0, t01));
    next1 = host_add(next1, host_mul(d1, profile->t11));
    next1 = host_mul(next1, profile->eo[residue][1]);
    maximum = next0 > 0.0f ? next0 : 0.0f;
    maximum = next1 > maximum ? next1 : maximum;
    if (!(maximum > 0.0f) || !isfinite(maximum)) return -1;
    d0 = host_div(next0, maximum);
    d1 = host_div(next1, maximum);
    score = host_add(score,
                     static_cast<float>(log(static_cast<double>(maximum))));
  }

  float end = 0.0f;
  end = host_add(end, host_mul(d0, profile->t02));
  end = host_add(end, host_mul(d1, profile->t12));
  if (!(end > 0.0f) || !isfinite(end)) return -1;
  score = host_add(score, static_cast<float>(log(static_cast<double>(end))));
  score = host_add(score, host_mul(length_f, logf(p1)));
  score = host_add(score, logf(1.0f - p1));
  if (!isfinite(score)) return -1;
  *filtersc = score;
  return 0;
}

extern "C" int
plan7_bias_rebias_decision(uint8_t ssv_status,
                           int16_t ssv_numerator,
                           float scale,
                           float filtersc,
                           int cutoff_mode,
                           float cutoff_bit_score,
                           float *bit_score)
{
  if (bit_score != nullptr) *bit_score = NAN;
  if (plan7_bias_host_environment_attested() != 1 ||
      ssv_status != 0 || !isfinite(scale) || scale <= 0.0f ||
      !isfinite(filtersc))
    return PLAN7_BIAS_CPU_REQUIRED;
  float usc = static_cast<float>(ssv_numerator);
  usc = host_div(usc, scale);
  usc = usc - 3.0f;
  const float delta = usc - filtersc;
  const float score = static_cast<float>(static_cast<double>(delta) / kLog2);
  if (!isfinite(usc) || !isfinite(score)) return PLAN7_BIAS_CPU_REQUIRED;
  if (bit_score != nullptr) *bit_score = score;
  switch (cutoff_mode) {
    case PLAN7_BIAS_CUTOFF_SCORE:
      if (!isfinite(cutoff_bit_score)) return PLAN7_BIAS_CPU_REQUIRED;
      return score < cutoff_bit_score ? PLAN7_BIAS_DEFINITE_REJECT
                                      : PLAN7_BIAS_DEFINITE_PASS;
    case PLAN7_BIAS_CUTOFF_ALWAYS_REJECT:
      return PLAN7_BIAS_DEFINITE_REJECT;
    case PLAN7_BIAS_CUTOFF_ALWAYS_PASS:
      return PLAN7_BIAS_DEFINITE_PASS;
    default:
      return PLAN7_BIAS_CPU_REQUIRED;
  }
}

extern "C" int
plan7_bias_host_environment_attested(void)
{
#if !defined(__x86_64__) || !defined(__GLIBC__)
  return 0;
#else
  constexpr unsigned kMxcsrRoundingFtzDazMask = UINT32_C(0x0000e040);
  return fegetround() == FE_TONEAREST &&
         (_mm_getcsr() & kMxcsrRoundingFtzDazMask) == 0;
#endif
}

extern "C" int
plan7_bias_environment_attested(char *reason, size_t reason_size)
{
#if !defined(__x86_64__) || !defined(__GLIBC__) || CUDART_VERSION != 12050 || \
    __CUDACC_VER_MAJOR__ != 12 || __CUDACC_VER_MINOR__ != 5 ||             \
    __CUDACC_VER_BUILD__ != 82
  set_error(reason, reason_size,
            "bias math is not attested for this host or CUDA toolkit");
  return 0;
#else
  if (plan7_bias_host_environment_attested() != 1) {
    set_error(reason, reason_size,
              "bias host floating-point environment is not attested");
    return 0;
  }
  if (strcmp(gnu_get_libc_version(), "2.35") != 0) {
    set_error(reason, reason_size,
              "bias math is not attested for this glibc version");
    return 0;
  }
  if (!loaded_libm_is_attested()) {
    set_error(reason, reason_size,
              "bias math is not attested for this loaded libm build");
    return 0;
  }
  unsigned vendor_max;
  unsigned vendor_ebx;
  unsigned vendor_ecx;
  unsigned vendor_edx;
  unsigned leaf1_eax;
  unsigned leaf1_ebx;
  unsigned leaf1_ecx;
  unsigned leaf1_edx;
  unsigned leaf7_eax;
  unsigned leaf7_ebx;
  unsigned leaf7_ecx;
  unsigned leaf7_edx;
  vendor_max = __get_cpuid_max(0, nullptr);
  if (vendor_max < 7 ||
      !__get_cpuid(0, &vendor_max, &vendor_ebx, &vendor_ecx, &vendor_edx) ||
      vendor_ebx != UINT32_C(0x756e6547) ||
      vendor_edx != UINT32_C(0x49656e69) ||
      vendor_ecx != UINT32_C(0x6c65746e) ||
      !__get_cpuid(1, &leaf1_eax, &leaf1_ebx, &leaf1_ecx, &leaf1_edx) ||
      !__get_cpuid_count(7, 0, &leaf7_eax, &leaf7_ebx,
                         &leaf7_ecx, &leaf7_edx)) {
    set_error(reason, reason_size,
              "bias math is not attested for this CPU vendor");
    return 0;
  }
  const unsigned stepping = leaf1_eax & UINT32_C(0x0f);
  const unsigned base_model = (leaf1_eax >> 4) & UINT32_C(0x0f);
  const unsigned base_family = (leaf1_eax >> 8) & UINT32_C(0x0f);
  const unsigned extended_model = (leaf1_eax >> 16) & UINT32_C(0x0f);
  const unsigned family = base_family;
  const unsigned model = base_model | (extended_model << 4);
  const uint32_t required_leaf1_ecx =
    bit_FMA | bit_SSE4_1 | bit_OSXSAVE | bit_AVX;
  const uint32_t required_leaf7_ebx =
    bit_AVX2 | bit_AVX512F | bit_AVX512DQ | bit_AVX512CD |
    bit_AVX512BW | bit_AVX512VL;
  uint32_t xcr0_low;
  uint32_t xcr0_high;
  __asm__ volatile("xgetbv" : "=a"(xcr0_low), "=d"(xcr0_high) : "c"(0));
  if (family != 6 || model != 85 || stepping != 4 ||
      (leaf1_edx & bit_SSE2) == 0 ||
      (leaf1_ecx & required_leaf1_ecx) != required_leaf1_ecx ||
      (leaf7_ebx & required_leaf7_ebx) != required_leaf7_ebx ||
      (xcr0_low & UINT32_C(0x000000e6)) != UINT32_C(0x000000e6) ||
      xcr0_high != 0) {
    set_error(reason, reason_size,
              "bias math is not attested for this CPU model or feature set");
    return 0;
  }
  int device;
  cudaDeviceProp properties;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "cudaGetDevice", status);
    return 0;
  }
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    set_cuda_error(reason, reason_size, "cudaGetDeviceProperties", status);
    return 0;
  }
  if (properties.major != 7 || properties.minor != 5 ||
      strcmp(properties.name, "NVIDIA GeForce RTX 2080 Ti") != 0) {
    set_error(reason, reason_size,
              "bias math is not attested for this CUDA architecture");
    return 0;
  }
  if (reason != nullptr && reason_size != 0) reason[0] = '\0';
  return 1;
#endif
}

extern "C" int
plan7_bias_filter_candidates_device(
  const uint8_t *device_residues,
  const uint64_t *device_offsets,
  const float *device_length_logp,
  const float *device_length_log1mp,
  const plan7_bias_profile *device_profiles,
  const plan7_bias_candidate *device_candidates,
  const plan7_bias_ssv_input *device_ssv_inputs,
  size_t candidate_count,
  plan7_bias_result *device_results,
  char *error,
  size_t error_size)
{
  if (candidate_count == 0) return 0;
  if (device_residues == nullptr || device_offsets == nullptr ||
      device_length_logp == nullptr || device_length_log1mp == nullptr ||
      device_profiles == nullptr || device_candidates == nullptr ||
      device_ssv_inputs == nullptr || device_results == nullptr) {
    set_error(error, error_size, "bias CUDA device buffer is null");
    return -1;
  }
  if (plan7_bias_environment_attested(error, error_size) != 1) return -1;
  if (candidate_count >
      static_cast<size_t>(UINT_MAX) * static_cast<size_t>(kThreads)) {
    set_error(error, error_size, "bias CUDA candidate grid is too large");
    return -1;
  }

  const unsigned blocks = static_cast<unsigned>(
    (candidate_count + kThreads - 1) / kThreads);
  bias_candidates_kernel<<<blocks, kThreads>>>(
    device_residues, device_offsets, device_length_logp,
    device_length_log1mp, device_profiles, device_candidates,
    device_ssv_inputs, candidate_count, device_results);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    set_cuda_error(error, error_size, "bias candidate kernel", status);
    return -1;
  }
  return 0;
}
