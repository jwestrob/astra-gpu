#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kAlphabet = 29;
constexpr int kProfilesPerWord = 4;

struct Result {
  uint8_t xE;
  uint8_t status;
  uint8_t tjb;
  uint8_t reserved;
  int16_t numerator;
};

static_assert(sizeof(Result) == 6, "result ABI must match plan7_ssv_result");

enum Status : uint8_t {
  kOk = 0,
  kRange = 16,
  kNoResult = 19,
  kEmpty = 255,
};

struct ScalarProfile {
  uint64_t score_offset;
  int32_t model_length;
  uint8_t tbm;
  uint8_t tec;
  uint8_t base;
  uint8_t bias;
};

struct PackedQuartet {
  uint64_t score_offset;
  int32_t maximum_model_length;
  int32_t model_length[4];
  uint8_t tbm[4];
  uint8_t tec[4];
  uint8_t base[4];
  uint8_t bias[4];
};

void cuda_check(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << operation << ": " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
  }
}

__device__ __forceinline__ int saturating_signed_subtract(int left, int right)
{
  const int value = left - right;
  return value > INT8_MAX ? INT8_MAX : (value < INT8_MIN ? INT8_MIN : value);
}

__device__ __forceinline__ Result finalize_result(unsigned raw_xE,
                                                  uint8_t length_tjb,
                                                  uint8_t tbm,
                                                  uint8_t tec,
                                                  uint8_t base,
                                                  uint8_t bias)
{
  Result result = {
    static_cast<uint8_t>(raw_xE), kOk, length_tjb, 0, 0
  };
  if (static_cast<unsigned>(length_tjb) + tbm + tec + bias >= 127U) {
    result.status = kNoResult;
  } else if (raw_xE >= 255U - bias) {
    result.status = static_cast<int>(base) - static_cast<int>(length_tjb) -
                        static_cast<int>(tbm) < 128
                      ? kNoResult
                      : kRange;
  } else {
    unsigned adjusted = raw_xE + base - length_tjb - tbm;
    adjusted -= 128;
    if (adjusted >= 255U - bias) {
      result.status = kRange;
    } else {
      const unsigned xJ = adjusted - tec;
      if (xJ > base) {
        result.status = kNoResult;
      } else {
        result.numerator = static_cast<int16_t>(
          static_cast<int>(xJ) - static_cast<int>(length_tjb) -
          static_cast<int>(base));
      }
    }
  }
  return result;
}

__device__ __forceinline__ unsigned reduce_scalar_maximum(unsigned value,
                                                          unsigned *scratch)
{
  for (int width = 16; width > 0; width >>= 1)
    value = max(value, __shfl_down_sync(UINT32_MAX, value, width));
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kThreads / 32 ? scratch[lane] : 128U;
    for (int width = 16; width > 0; width >>= 1)
      value = max(value, __shfl_down_sync(UINT32_MAX, value, width));
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ uint32_t reduce_packed_maximum(uint32_t value,
                                                          uint32_t *scratch)
{
  for (int width = 16; width > 0; width >>= 1)
    value = __vmaxu4(value, __shfl_down_sync(UINT32_MAX, value, width));
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kThreads / 32 ? scratch[lane] : 0x80808080U;
    for (int width = 16; width > 0; width >>= 1)
      value = __vmaxu4(value,
                       __shfl_down_sync(UINT32_MAX, value, width));
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void scalar_ssv_many_kernel(const uint8_t *scores,
                                       const ScalarProfile *profiles,
                                       size_t profile_count,
                                       size_t sequence_count,
                                       const uint8_t *residues,
                                       const uint64_t *offsets,
                                       const uint8_t *profile_major_tjb,
                                       Result *results)
{
  __shared__ unsigned maxima[kThreads / 32];
  const size_t sequence = blockIdx.x;
  const size_t profile_index = blockIdx.y;
  if (sequence >= sequence_count || profile_index >= profile_count) return;

  const ScalarProfile profile = profiles[profile_index];
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  const size_t result_index = profile_index * sequence_count + sequence;
  const uint8_t tjb = profile_major_tjb[result_index];
  if (length == 0) {
    if (threadIdx.x == 0) results[result_index] = {128, kEmpty, tjb, 0, 0};
    return;
  }

  unsigned local_maximum = 128;
  const int diagonal_count = profile.model_length + length - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    int value = INT8_MIN;
    while (i < length && k < profile.model_length) {
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      const uint8_t raw_cost = scores[
        profile.score_offset + static_cast<uint64_t>(k) * kAlphabet + residue];
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

  const unsigned raw_xE = reduce_scalar_maximum(local_maximum, maxima);
  if (threadIdx.x == 0)
    results[result_index] = finalize_result(raw_xE, tjb, profile.tbm,
                                            profile.tec, profile.base,
                                            profile.bias);
}

__global__ void packed_profile_ssv_many_kernel(
  const uint32_t *packed_scores,
  const PackedQuartet *quartets,
  size_t profile_count,
  size_t sequence_count,
  const uint8_t *residues,
  const uint64_t *offsets,
  const uint8_t *profile_major_tjb,
  Result *results)
{
  __shared__ uint32_t maxima[kThreads / 32];
  const size_t sequence = blockIdx.x;
  const size_t quartet_index = blockIdx.y;
  if (sequence >= sequence_count) return;
  const size_t first_profile = quartet_index * kProfilesPerWord;
  if (first_profile >= profile_count) return;

  const PackedQuartet quartet = quartets[quartet_index];
  const uint64_t start = offsets[sequence];
  const int length = static_cast<int>(offsets[sequence + 1] - start);
  if (length == 0) {
    if (threadIdx.x == 0) {
      for (int lane = 0; lane < kProfilesPerWord; ++lane) {
        const size_t profile = first_profile + static_cast<size_t>(lane);
        if (profile < profile_count) {
          const size_t index = profile * sequence_count + sequence;
          results[index] = {128, kEmpty, profile_major_tjb[index], 0, 0};
        }
      }
    }
    return;
  }

  uint32_t local_maximum = 0x80808080U;
  const int diagonal_count = quartet.maximum_model_length + length - 1;
  for (int diagonal = threadIdx.x; diagonal < diagonal_count;
       diagonal += blockDim.x) {
    const int delta = diagonal - (length - 1);
    int i = delta < 0 ? -delta : 0;
    int k = i + delta;
    uint32_t value = 0x80808080U;
    while (i < length && k < quartet.maximum_model_length) {
      const unsigned residue = residues[start + static_cast<uint64_t>(i)];
      const uint32_t cost = packed_scores[
        quartet.score_offset + static_cast<uint64_t>(k) * kAlphabet + residue];
      uint32_t active = 0;
#pragma unroll
      for (int lane = 0; lane < kProfilesPerWord; ++lane)
        if (k < quartet.model_length[lane]) active |= 0xffU << (lane * 8);
      const uint32_t next = __vsubss4(value, cost);
      value = (next & active) | (value & ~active);
      local_maximum = __vmaxu4(local_maximum, value);
      ++i;
      ++k;
    }
  }

  const uint32_t packed_xE = reduce_packed_maximum(local_maximum, maxima);
  if (threadIdx.x == 0) {
    for (int lane = 0; lane < kProfilesPerWord; ++lane) {
      const size_t profile = first_profile + static_cast<size_t>(lane);
      if (profile >= profile_count) continue;
      const size_t index = profile * sequence_count + sequence;
      const unsigned raw_xE = (packed_xE >> (lane * 8)) & 0xffU;
      results[index] = finalize_result(raw_xE, profile_major_tjb[index],
                                       quartet.tbm[lane], quartet.tec[lane],
                                       quartet.base[lane], quartet.bias[lane]);
    }
  }
}

__global__ void packed_byte_primitive_kernel(uint8_t *subtract_results,
                                             uint8_t *maximum_results)
{
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 65536U) return;
  const uint8_t left = static_cast<uint8_t>(index >> 8);
  const uint8_t right = static_cast<uint8_t>(index);
  const uint32_t left4 = static_cast<uint32_t>(left) * 0x01010101U;
  const uint32_t right4 = static_cast<uint32_t>(right) * 0x01010101U;
  subtract_results[index] = static_cast<uint8_t>(__vsubss4(left4, right4));
  maximum_results[index] = static_cast<uint8_t>(__vmaxu4(left4, right4));
}

template<typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) : count_(count)
  {
    if (count_ != 0)
      cuda_check(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                            count_ * sizeof(T)), "cudaMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
    : pointer_(other.pointer_), count_(other.count_)
  {
    other.pointer_ = nullptr;
    other.count_ = 0;
  }
  ~DeviceBuffer()
  {
    if (pointer_ != nullptr) cudaFree(pointer_);
  }
  T *get() const { return pointer_; }
  size_t size() const { return count_; }

 private:
  T *pointer_ = nullptr;
  size_t count_ = 0;
};

template<typename T>
DeviceBuffer<T> upload(const std::vector<T> &source)
{
  DeviceBuffer<T> destination(source.size());
  if (!source.empty())
    cuda_check(cudaMemcpy(destination.get(), source.data(),
                          source.size() * sizeof(T), cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");
  return destination;
}

struct HostCase {
  std::string name;
  size_t profile_count = 0;
  size_t sequence_count = 0;
  std::vector<ScalarProfile> scalar_profiles;
  std::vector<PackedQuartet> packed_quartets;
  std::vector<uint8_t> scalar_scores;
  std::vector<uint32_t> packed_scores;
  std::vector<uint8_t> residues;
  std::vector<uint64_t> offsets;
  std::vector<uint8_t> tjb;
  uint64_t logical_cells = 0;
};

uint8_t score_value(size_t profile, int k, int residue)
{
  uint32_t value = static_cast<uint32_t>(profile * 0x9e3779b1U) ^
                   static_cast<uint32_t>((k + 1) * 0x85ebca6bU) ^
                   static_cast<uint32_t>((residue + 3) * 0xc2b2ae35U);
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  const int signed_cost = static_cast<int>(value % 31U) - 15;
  return static_cast<uint8_t>(static_cast<int8_t>(signed_cost));
}

HostCase make_case(const std::string &name,
                   const std::vector<int> &model_lengths,
                   const std::vector<int> &sequence_lengths)
{
  if (model_lengths.empty() || sequence_lengths.empty())
    throw std::runtime_error("empty benchmark shape");
  HostCase result;
  result.name = name;
  result.profile_count = model_lengths.size();
  result.sequence_count = sequence_lengths.size();
  result.scalar_profiles.resize(result.profile_count);
  result.offsets.push_back(0);
  for (size_t sequence = 0; sequence < result.sequence_count; ++sequence) {
    const int length = sequence_lengths[sequence];
    for (int i = 0; i < length; ++i)
      result.residues.push_back(static_cast<uint8_t>(
        (sequence * 17U + static_cast<size_t>(i) * 11U + 3U) % kAlphabet));
    result.offsets.push_back(result.residues.size());
  }

  for (size_t profile = 0; profile < result.profile_count; ++profile) {
    ScalarProfile descriptor{};
    descriptor.score_offset = result.scalar_scores.size();
    descriptor.model_length = model_lengths[profile];
    descriptor.tbm = static_cast<uint8_t>(2U + profile % 5U);
    descriptor.tec = static_cast<uint8_t>(1U + profile % 3U);
    descriptor.base = static_cast<uint8_t>(180U + profile % 31U);
    descriptor.bias = static_cast<uint8_t>(profile % 7U);
    result.scalar_profiles[profile] = descriptor;
    for (int k = 0; k < descriptor.model_length; ++k)
      for (int residue = 0; residue < kAlphabet; ++residue)
        result.scalar_scores.push_back(score_value(profile, k, residue));
  }

  result.tjb.resize(result.profile_count * result.sequence_count);
  for (size_t profile = 0; profile < result.profile_count; ++profile)
    for (size_t sequence = 0; sequence < result.sequence_count; ++sequence)
      result.tjb[profile * result.sequence_count + sequence] =
        static_cast<uint8_t>(1U + (profile * 3U + sequence * 5U) % 23U);

  const size_t quartet_count =
    (result.profile_count + kProfilesPerWord - 1) / kProfilesPerWord;
  result.packed_quartets.resize(quartet_count);
  for (size_t quartet_index = 0; quartet_index < quartet_count;
       ++quartet_index) {
    PackedQuartet quartet{};
    quartet.score_offset = result.packed_scores.size();
    for (int lane = 0; lane < kProfilesPerWord; ++lane) {
      const size_t profile = quartet_index * kProfilesPerWord + lane;
      if (profile < result.profile_count) {
        const ScalarProfile scalar = result.scalar_profiles[profile];
        quartet.model_length[lane] = scalar.model_length;
        quartet.maximum_model_length =
          std::max(quartet.maximum_model_length, scalar.model_length);
        quartet.tbm[lane] = scalar.tbm;
        quartet.tec[lane] = scalar.tec;
        quartet.base[lane] = scalar.base;
        quartet.bias[lane] = scalar.bias;
      }
    }
    for (int k = 0; k < quartet.maximum_model_length; ++k) {
      for (int residue = 0; residue < kAlphabet; ++residue) {
        uint32_t packed = 0;
        for (int lane = 0; lane < kProfilesPerWord; ++lane) {
          const size_t profile = quartet_index * kProfilesPerWord + lane;
          uint8_t cost = 0x80U;  // Hostile padding; active masks must suppress it.
          if (profile < result.profile_count &&
              k < result.scalar_profiles[profile].model_length) {
            const ScalarProfile scalar = result.scalar_profiles[profile];
            cost = result.scalar_scores[
              scalar.score_offset + static_cast<uint64_t>(k) * kAlphabet +
              residue];
          }
          packed |= static_cast<uint32_t>(cost) << (lane * 8);
        }
        result.packed_scores.push_back(packed);
      }
    }
    result.packed_quartets[quartet_index] = quartet;
  }

  const uint64_t total_residues = std::accumulate(
    sequence_lengths.begin(), sequence_lengths.end(), uint64_t{0});
  for (int model_length : model_lengths)
    result.logical_cells += static_cast<uint64_t>(model_length) * total_residues;
  return result;
}

double median(std::vector<float> values)
{
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 != 0
           ? values[middle]
           : 0.5 * (static_cast<double>(values[middle - 1]) + values[middle]);
}

uint64_t fnv1a(const void *data, size_t byte_count)
{
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  uint64_t hash = 1469598103934665603ULL;
  for (size_t index = 0; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

void verify_primitives()
{
  DeviceBuffer<uint8_t> device_subtract(65536);
  DeviceBuffer<uint8_t> device_maximum(65536);
  packed_byte_primitive_kernel<<<256, 256>>>(device_subtract.get(),
                                             device_maximum.get());
  cuda_check(cudaGetLastError(), "packed primitive launch");
  std::vector<uint8_t> subtract(65536);
  std::vector<uint8_t> maximum(65536);
  cuda_check(cudaMemcpy(subtract.data(), device_subtract.get(), subtract.size(),
                        cudaMemcpyDeviceToHost), "primitive subtract D2H");
  cuda_check(cudaMemcpy(maximum.data(), device_maximum.get(), maximum.size(),
                        cudaMemcpyDeviceToHost), "primitive maximum D2H");
  for (unsigned index = 0; index < 65536U; ++index) {
    const uint8_t left_raw = static_cast<uint8_t>(index >> 8);
    const uint8_t right_raw = static_cast<uint8_t>(index);
    const int left = static_cast<int>(static_cast<int8_t>(left_raw));
    const int right = static_cast<int>(static_cast<int8_t>(right_raw));
    const int difference = std::max(-128, std::min(127, left - right));
    if (subtract[index] != static_cast<uint8_t>(static_cast<int8_t>(difference)) ||
        maximum[index] != std::max(left_raw, right_raw))
      throw std::runtime_error("packed byte primitive mismatch");
  }
}

struct CaseResult {
  std::string name;
  size_t profiles;
  size_t sequences;
  uint64_t cells;
  size_t result_bytes;
  uint64_t result_hash;
  double scalar_ms;
  double packed_ms;
};

CaseResult run_case(const HostCase &test_case, int samples)
{
  auto scalar_profiles = upload(test_case.scalar_profiles);
  auto packed_quartets = upload(test_case.packed_quartets);
  auto scalar_scores = upload(test_case.scalar_scores);
  auto packed_scores = upload(test_case.packed_scores);
  auto residues = upload(test_case.residues);
  auto offsets = upload(test_case.offsets);
  auto tjb = upload(test_case.tjb);
  const size_t result_count = test_case.profile_count * test_case.sequence_count;
  DeviceBuffer<Result> scalar_results(result_count);
  DeviceBuffer<Result> packed_results(result_count);
  const dim3 scalar_grid(static_cast<unsigned>(test_case.sequence_count),
                         static_cast<unsigned>(test_case.profile_count));
  const dim3 packed_grid(
    static_cast<unsigned>(test_case.sequence_count),
    static_cast<unsigned>(test_case.packed_quartets.size()));

  auto launch_scalar = [&]() {
    scalar_ssv_many_kernel<<<scalar_grid, kThreads>>>(
      scalar_scores.get(), scalar_profiles.get(), test_case.profile_count,
      test_case.sequence_count, residues.get(), offsets.get(), tjb.get(),
      scalar_results.get());
  };
  auto launch_packed = [&]() {
    packed_profile_ssv_many_kernel<<<packed_grid, kThreads>>>(
      packed_scores.get(), packed_quartets.get(), test_case.profile_count,
      test_case.sequence_count, residues.get(), offsets.get(), tjb.get(),
      packed_results.get());
  };

  launch_scalar();
  launch_packed();
  cuda_check(cudaGetLastError(), "SSV oracle launch");
  cuda_check(cudaDeviceSynchronize(), "SSV oracle synchronize");
  std::vector<Result> host_scalar(result_count);
  std::vector<Result> host_packed(result_count);
  cuda_check(cudaMemcpy(host_scalar.data(), scalar_results.get(),
                        result_count * sizeof(Result), cudaMemcpyDeviceToHost),
             "scalar results D2H");
  cuda_check(cudaMemcpy(host_packed.data(), packed_results.get(),
                        result_count * sizeof(Result), cudaMemcpyDeviceToHost),
             "packed results D2H");
  if (std::memcmp(host_scalar.data(), host_packed.data(),
                  result_count * sizeof(Result)) != 0) {
    for (size_t index = 0; index < result_count; ++index)
      if (std::memcmp(&host_scalar[index], &host_packed[index], sizeof(Result)) !=
          0) {
        std::ostringstream message;
        message << "SSV mismatch in " << test_case.name << " at profile "
                << index / test_case.sequence_count << ", sequence "
                << index % test_case.sequence_count;
        throw std::runtime_error(message.str());
      }
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_check(cudaEventCreate(&start), "cudaEventCreate start");
  cuda_check(cudaEventCreate(&stop), "cudaEventCreate stop");
  auto time_kernel = [&](auto launch) {
    std::vector<float> timings;
    timings.reserve(samples);
    launch();
    launch();
    cuda_check(cudaDeviceSynchronize(), "timing warmup synchronize");
    for (int sample = 0; sample < samples; ++sample) {
      cuda_check(cudaEventRecord(start), "cudaEventRecord start");
      launch();
      cuda_check(cudaEventRecord(stop), "cudaEventRecord stop");
      cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
      float elapsed = 0;
      cuda_check(cudaEventElapsedTime(&elapsed, start, stop),
                 "cudaEventElapsedTime");
      timings.push_back(elapsed);
    }
    return median(timings);
  };
  const double scalar_ms = time_kernel(launch_scalar);
  const double packed_ms = time_kernel(launch_packed);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);

  return {
    test_case.name,
    test_case.profile_count,
    test_case.sequence_count,
    test_case.logical_cells,
    result_count * sizeof(Result),
    fnv1a(host_scalar.data(), result_count * sizeof(Result)),
    scalar_ms,
    packed_ms,
  };
}

std::vector<int> repeat_lengths(size_t count, const std::vector<int> &pattern)
{
  std::vector<int> lengths;
  lengths.reserve(count);
  for (size_t index = 0; index < count; ++index)
    lengths.push_back(pattern[index % pattern.size()]);
  return lengths;
}

std::string json_escape(const std::string &value)
{
  std::ostringstream output;
  for (char character : value) {
    if (character == '"' || character == '\\') output << '\\';
    output << character;
  }
  return output.str();
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    if (argc != 2) {
      std::cerr << "usage: " << argv[0] << " OUTPUT.json\n";
      return 2;
    }
    int device = -1;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties");
    if (properties.major != 9 || std::string(properties.name).find("H200") ==
                                   std::string::npos)
      throw std::runtime_error("Phase 5 benchmark requires an attested H200");

    verify_primitives();
    std::vector<CaseResult> results;

    // Untimed edge oracle: partial quartet, empty target, boundary lengths,
    // unequal models, and hostile 0x80 padding in inactive packed lanes.
    const HostCase edge = make_case(
      "edge_oracle", {1, 7, 31, 32, 33, 95, 257}, {0, 1, 2, 31, 65, 257});
    results.push_back(run_case(edge, 3));

    results.push_back(run_case(make_case(
      "equal_m96_l96", std::vector<int>(1024, 96), std::vector<int>(64, 96)),
      7));

    std::vector<int> sorted_models(1024);
    for (size_t index = 0; index < sorted_models.size(); ++index)
      sorted_models[index] = 32 + static_cast<int>((352 * index) /
                                                    sorted_models.size());
    results.push_back(run_case(make_case(
      "sorted_m32_383_l256", sorted_models, std::vector<int>(64, 256)), 7));

    results.push_back(run_case(make_case(
      "divergent_quartets_l512", repeat_lengths(512, {32, 64, 128, 384}),
      std::vector<int>(32, 512)), 7));

    std::ostringstream json;
    json << std::setprecision(12)
         << "{\n  \"schema\": 1,\n  \"status\": \"PASS\",\n"
         << "  \"device\": \"" << json_escape(properties.name) << "\",\n"
         << "  \"compute_capability\": \"" << properties.major << "."
         << properties.minor << "\",\n"
         << "  \"primitive_pairs_checked\": 65536,\n"
         << "  \"packed_profiles_per_word\": 4,\n"
         << "  \"padding_byte\": 128,\n"
         << "  \"cases\": [\n";
    for (size_t index = 0; index < results.size(); ++index) {
      const CaseResult &result = results[index];
      json << "    {\"name\": \"" << result.name << "\", \"profiles\": "
           << result.profiles << ", \"sequences\": " << result.sequences
           << ", \"logical_cells\": " << result.cells
           << ", \"result_bytes\": " << result.result_bytes
           << ", \"result_hash_fnv1a64\": \"" << std::hex
           << result.result_hash << std::dec << "\", \"scalar_ms\": "
           << result.scalar_ms << ", \"packed_ms\": " << result.packed_ms
           << ", \"speedup\": " << result.scalar_ms / result.packed_ms
           << "}" << (index + 1 == results.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";

    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output path");
    output << json.str();
    output.close();
    if (!output) throw std::runtime_error("cannot write output path");
    std::cout << json.str();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "phase5_profile_packed_ssv: " << error.what() << "\n";
    return 1;
  }
}
