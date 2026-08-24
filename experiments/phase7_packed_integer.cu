#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kSteps = 128;
constexpr size_t kLogicalValues = size_t{1} << 20;
constexpr int16_t kShortCorners[] = {
  INT16_MIN, INT16_MIN + 1, -1, 0, 1, INT16_MAX - 1, INT16_MAX};
__device__ __constant__ int16_t kDeviceShortCorners[] = {
  INT16_MIN, INT16_MIN + 1, -1, 0, 1, INT16_MAX - 1, INT16_MAX};

void check(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
}

template<typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count)
  {
    check(cudaMalloc(&data_, std::max<size_t>(count, 1) * sizeof(T)),
          "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept
    : data_(other.data_), count_(other.count_)
  {
    other.data_ = nullptr;
    other.count_ = 0;
  }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
  {
    if (this != &other) {
      cudaFree(data_);
      data_ = other.data_;
      count_ = other.count_;
      other.data_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }
  T *get() { return data_; }
  const T *get() const { return data_; }
  size_t count() const { return count_; }

 private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

template<typename T>
DeviceBuffer<T> upload(const std::vector<T> &values)
{
  DeviceBuffer<T> result(values.size());
  if (!values.empty())
    check(cudaMemcpy(result.get(), values.data(), values.size() * sizeof(T),
                     cudaMemcpyHostToDevice), "upload");
  return result;
}

__device__ __forceinline__ uint8_t scalar_add_u8(uint8_t a, uint8_t b)
{
  const unsigned value = static_cast<unsigned>(a) + b;
  return static_cast<uint8_t>(value > UINT8_MAX ? UINT8_MAX : value);
}

__device__ __forceinline__ uint8_t scalar_sub_u8(uint8_t a, uint8_t b)
{
  return a > b ? static_cast<uint8_t>(a - b) : 0;
}

__device__ __forceinline__ int16_t scalar_add_i16(int16_t a, int16_t b)
{
  const int value = static_cast<int>(a) + static_cast<int>(b);
  return static_cast<int16_t>(value > INT16_MAX ? INT16_MAX :
                              (value < INT16_MIN ? INT16_MIN : value));
}

struct BytePrimitiveResult {
  uint32_t add;
  uint32_t subtract;
  uint32_t maximum;
};

struct ShortPrimitiveResult {
  uint32_t add;
  uint32_t maximum;
  uint32_t dpx_add_max;
};

__global__ void byte_primitive_kernel(BytePrimitiveResult *results)
{
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 65536) return;
  const uint32_t left = (index >> 8) * UINT32_C(0x01010101);
  const uint32_t right = (index & 255) * UINT32_C(0x01010101);
  results[index] = {
    __vaddus4(left, right), __vsubus4(left, right), __vmaxu4(left, right)};
}

__global__ void short_primitive_kernel(ShortPrimitiveResult *results)
{
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  constexpr unsigned corner_count = sizeof(kShortCorners) / sizeof(int16_t);
  if (index >= 65536U * corner_count) return;
  const int16_t left = static_cast<int16_t>(index / corner_count + INT16_MIN);
  const int16_t right = kDeviceShortCorners[index % corner_count];
  const uint32_t packed_left = static_cast<uint16_t>(left) |
    (static_cast<uint32_t>(static_cast<uint16_t>(left)) << 16);
  const uint32_t packed_right = static_cast<uint16_t>(right) |
    (static_cast<uint32_t>(static_cast<uint16_t>(right)) << 16);
  const uint32_t negative_infinity = UINT32_C(0x80008000);
  results[index] = {
    __vaddss2(packed_left, packed_right),
    __vmaxs2(packed_left, packed_right),
    __viaddmax_s16x2(packed_left, packed_right, negative_infinity)};
}

__global__ void msv_scalar_kernel(const uint8_t *seed,
                                  const uint8_t *cost_seed,
                                  const uint8_t *bias,
                                  size_t count,
                                  uint8_t *output)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (index >= count) return;
  uint8_t value = seed[index];
  uint8_t best = 0;
  const uint8_t add = bias[index];
  for (int step = 0; step < kSteps; ++step) {
    const uint8_t cost = static_cast<uint8_t>(
      cost_seed[index] ^ static_cast<uint8_t>(step * 73));
    value = scalar_sub_u8(
      scalar_add_u8(value > best ? value : best, add), cost);
    best = best > value ? best : value;
  }
  output[index] = static_cast<uint8_t>(value ^ best);
}

__global__ void msv_packed_kernel(const uint32_t *seed,
                                  const uint32_t *cost_seed,
                                  const uint32_t *bias,
                                  size_t packed_count,
                                  uint32_t *output)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (index >= packed_count) return;
  uint32_t value = seed[index];
  uint32_t best = 0;
  const uint32_t add = bias[index];
  for (int step = 0; step < kSteps; ++step) {
    const uint32_t pattern = static_cast<uint8_t>(step * 73) *
                             UINT32_C(0x01010101);
    const uint32_t cost = cost_seed[index] ^ pattern;
    value = __vsubus4(__vaddus4(__vmaxu4(value, best), add), cost);
    best = __vmaxu4(best, value);
  }
  output[index] = value ^ best;
}

__global__ void viterbi_scalar_kernel(const int16_t *seed,
                                      const int16_t *transition_seed,
                                      const int16_t *alternate_seed,
                                      size_t count,
                                      int16_t *output)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (index >= count) return;
  int16_t value = seed[index];
  for (int step = 0; step < kSteps; ++step) {
    const uint16_t pattern = static_cast<uint16_t>(step * 977);
    const int16_t transition = static_cast<int16_t>(
      static_cast<uint16_t>(transition_seed[index]) ^ pattern);
    const int16_t alternate = static_cast<int16_t>(
      static_cast<uint16_t>(alternate_seed[index]) ^
      static_cast<uint16_t>(pattern * 3U));
    const int16_t advanced = scalar_add_i16(value, transition);
    value = advanced > alternate ? advanced : alternate;
  }
  output[index] = value;
}

template<bool UseDpx>
__global__ void viterbi_packed_kernel(const uint32_t *seed,
                                      const uint32_t *transition_seed,
                                      const uint32_t *alternate_seed,
                                      size_t packed_count,
                                      uint32_t *output)
{
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
  if (index >= packed_count) return;
  uint32_t value = seed[index];
  for (int step = 0; step < kSteps; ++step) {
    const uint32_t pattern = static_cast<uint16_t>(step * 977) *
                             UINT32_C(0x00010001);
    const uint32_t alternate_pattern =
      static_cast<uint16_t>(static_cast<uint16_t>(step * 977) * 3U) *
      UINT32_C(0x00010001);
    const uint32_t transition = transition_seed[index] ^ pattern;
    const uint32_t alternate = alternate_seed[index] ^ alternate_pattern;
    if constexpr (UseDpx)
      value = __viaddmax_s16x2(value, transition, alternate);
    else
      value = __vmaxs2(__vaddss2(value, transition), alternate);
  }
  output[index] = value;
}

uint8_t host_add_u8(uint8_t a, uint8_t b)
{
  const unsigned value = static_cast<unsigned>(a) + b;
  return static_cast<uint8_t>(value > UINT8_MAX ? UINT8_MAX : value);
}

uint8_t host_sub_u8(uint8_t a, uint8_t b)
{
  return a > b ? static_cast<uint8_t>(a - b) : 0;
}

int16_t host_add_i16(int16_t a, int16_t b)
{
  const int value = static_cast<int>(a) + static_cast<int>(b);
  return static_cast<int16_t>(value > INT16_MAX ? INT16_MAX :
                              (value < INT16_MIN ? INT16_MIN : value));
}

uint32_t repeat_byte(uint8_t value)
{
  return static_cast<uint32_t>(value) * UINT32_C(0x01010101);
}

uint32_t repeat_short(int16_t value)
{
  const uint32_t low = static_cast<uint16_t>(value);
  return low | (low << 16);
}

template<typename Launch>
double median_ms(Launch launch)
{
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check(cudaEventCreate(&start), "cudaEventCreate start");
  check(cudaEventCreate(&stop), "cudaEventCreate stop");
  launch();
  launch();
  check(cudaDeviceSynchronize(), "timing warmup");
  std::vector<float> samples;
  for (int sample = 0; sample < 9; ++sample) {
    check(cudaEventRecord(start), "cudaEventRecord start");
    launch();
    check(cudaEventRecord(stop), "cudaEventRecord stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    float elapsed = 0;
    check(cudaEventElapsedTime(&elapsed, start, stop), "cudaEventElapsedTime");
    samples.push_back(elapsed);
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

template<typename T>
std::vector<T> download(const DeviceBuffer<T> &source)
{
  std::vector<T> result(source.count());
  if (!result.empty())
    check(cudaMemcpy(result.data(), source.get(), result.size() * sizeof(T),
                     cudaMemcpyDeviceToHost), "download");
  return result;
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
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.major != 9 ||
        std::string(properties.name).find("H200") == std::string::npos)
      throw std::runtime_error("Phase 7 benchmark requires an H200");

    DeviceBuffer<BytePrimitiveResult> byte_device(65536);
    byte_primitive_kernel<<<256, kThreads>>>(byte_device.get());
    check(cudaGetLastError(), "byte primitive launch");
    const auto byte_results = download(byte_device);
    for (unsigned left = 0; left < 256; ++left)
      for (unsigned right = 0; right < 256; ++right) {
        const auto &result = byte_results[left * 256 + right];
        if (result.add != repeat_byte(host_add_u8(left, right)) ||
            result.subtract != repeat_byte(host_sub_u8(left, right)) ||
            result.maximum != repeat_byte(std::max(left, right)))
          throw std::runtime_error("packed byte primitive mismatch");
      }

    constexpr size_t short_count =
      65536 * (sizeof(kShortCorners) / sizeof(int16_t));
    DeviceBuffer<ShortPrimitiveResult> short_device(short_count);
    short_primitive_kernel<<<
      static_cast<unsigned>((short_count + kThreads - 1) / kThreads),
      kThreads>>>(short_device.get());
    check(cudaGetLastError(), "short primitive launch");
    const auto short_results = download(short_device);
    size_t dpx_saturation_mismatches = 0;
    for (size_t index = 0; index < short_count; ++index) {
      const int16_t left = static_cast<int16_t>(
        index / (sizeof(kShortCorners) / sizeof(int16_t)) + INT16_MIN);
      const int16_t right =
        kShortCorners[index % (sizeof(kShortCorners) / sizeof(int16_t))];
      const uint32_t expected_add = repeat_short(host_add_i16(left, right));
      const uint32_t expected_max = repeat_short(std::max(left, right));
      if (short_results[index].add != expected_add ||
          short_results[index].maximum != expected_max)
        throw std::runtime_error("packed short primitive mismatch");
      if (short_results[index].dpx_add_max != expected_add)
        ++dpx_saturation_mismatches;
    }
    if (dpx_saturation_mismatches == 0)
      throw std::runtime_error("DPX saturation incompatibility was not exposed");

    std::vector<uint8_t> msv_seed(kLogicalValues);
    std::vector<uint8_t> msv_cost(kLogicalValues);
    std::vector<uint8_t> msv_bias(kLogicalValues);
    for (size_t i = 0; i < kLogicalValues; ++i) {
      msv_seed[i] = static_cast<uint8_t>(i * 17 + 3);
      msv_cost[i] = static_cast<uint8_t>(i * 29 + 11);
      msv_bias[i] = static_cast<uint8_t>(i * 7 + 1);
    }
    auto d_msv_seed = upload(msv_seed);
    auto d_msv_cost = upload(msv_cost);
    auto d_msv_bias = upload(msv_bias);
    DeviceBuffer<uint8_t> d_msv_scalar(kLogicalValues);
    DeviceBuffer<uint8_t> d_msv_packed(kLogicalValues);
    const unsigned scalar_blocks = static_cast<unsigned>(
      (kLogicalValues + kThreads - 1) / kThreads);
    const size_t packed_byte_count = kLogicalValues / 4;
    const unsigned packed_byte_blocks = static_cast<unsigned>(
      (packed_byte_count + kThreads - 1) / kThreads);
    auto launch_msv_scalar = [&]() {
      msv_scalar_kernel<<<scalar_blocks, kThreads>>>(
        d_msv_seed.get(), d_msv_cost.get(), d_msv_bias.get(),
        kLogicalValues, d_msv_scalar.get());
    };
    auto launch_msv_packed = [&]() {
      msv_packed_kernel<<<packed_byte_blocks, kThreads>>>(
        reinterpret_cast<const uint32_t *>(d_msv_seed.get()),
        reinterpret_cast<const uint32_t *>(d_msv_cost.get()),
        reinterpret_cast<const uint32_t *>(d_msv_bias.get()),
        packed_byte_count,
        reinterpret_cast<uint32_t *>(d_msv_packed.get()));
    };
    launch_msv_scalar();
    launch_msv_packed();
    check(cudaDeviceSynchronize(), "MSV oracle synchronize");
    if (download(d_msv_scalar) != download(d_msv_packed))
      throw std::runtime_error("packed MSV recurrence mismatch");
    const double msv_scalar_ms = median_ms(launch_msv_scalar);
    const double msv_packed_ms = median_ms(launch_msv_packed);

    std::vector<int16_t> vit_seed(kLogicalValues);
    std::vector<int16_t> vit_transition(kLogicalValues);
    std::vector<int16_t> vit_alternate(kLogicalValues);
    for (size_t i = 0; i < kLogicalValues; ++i) {
      vit_seed[i] = static_cast<int16_t>(i * 101 + 7);
      vit_transition[i] = static_cast<int16_t>(i * 313 + 19);
      vit_alternate[i] = static_cast<int16_t>(i * 911 + 23);
    }
    auto d_vit_seed = upload(vit_seed);
    auto d_vit_transition = upload(vit_transition);
    auto d_vit_alternate = upload(vit_alternate);
    DeviceBuffer<int16_t> d_vit_scalar(kLogicalValues);
    DeviceBuffer<int16_t> d_vit_packed(kLogicalValues);
    DeviceBuffer<int16_t> d_vit_dpx(kLogicalValues);
    const size_t packed_short_count = kLogicalValues / 2;
    const unsigned packed_short_blocks = static_cast<unsigned>(
      (packed_short_count + kThreads - 1) / kThreads);
    auto launch_vit_scalar = [&]() {
      viterbi_scalar_kernel<<<scalar_blocks, kThreads>>>(
        d_vit_seed.get(), d_vit_transition.get(), d_vit_alternate.get(),
        kLogicalValues, d_vit_scalar.get());
    };
    auto launch_vit_packed = [&]() {
      viterbi_packed_kernel<false><<<packed_short_blocks, kThreads>>>(
        reinterpret_cast<const uint32_t *>(d_vit_seed.get()),
        reinterpret_cast<const uint32_t *>(d_vit_transition.get()),
        reinterpret_cast<const uint32_t *>(d_vit_alternate.get()),
        packed_short_count,
        reinterpret_cast<uint32_t *>(d_vit_packed.get()));
    };
    auto launch_vit_dpx = [&]() {
      viterbi_packed_kernel<true><<<packed_short_blocks, kThreads>>>(
        reinterpret_cast<const uint32_t *>(d_vit_seed.get()),
        reinterpret_cast<const uint32_t *>(d_vit_transition.get()),
        reinterpret_cast<const uint32_t *>(d_vit_alternate.get()),
        packed_short_count,
        reinterpret_cast<uint32_t *>(d_vit_dpx.get()));
    };
    launch_vit_scalar();
    launch_vit_packed();
    launch_vit_dpx();
    check(cudaDeviceSynchronize(), "Viterbi oracle synchronize");
    const auto vit_scalar = download(d_vit_scalar);
    const auto vit_packed = download(d_vit_packed);
    const auto vit_dpx = download(d_vit_dpx);
    if (vit_scalar != vit_packed)
      throw std::runtime_error("packed Viterbi recurrence mismatch");
    size_t dpx_recurrence_mismatches = 0;
    for (size_t index = 0; index < vit_dpx.size(); ++index)
      if (vit_dpx[index] != vit_scalar[index])
        ++dpx_recurrence_mismatches;
    const double vit_scalar_ms = median_ms(launch_vit_scalar);
    const double vit_packed_ms = median_ms(launch_vit_packed);
    const double vit_dpx_ms = median_ms(launch_vit_dpx);

    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    output << std::setprecision(12)
           << "{\n  \"schema\": 1,\n  \"status\": \"PASS\",\n"
           << "  \"device\": \"" << properties.name << "\",\n"
           << "  \"byte_pairs_checked\": 65536,\n"
           << "  \"short_corner_pairs_checked\": " << short_count << ",\n"
           << "  \"dpx_saturation_mismatches\": "
           << dpx_saturation_mismatches << ",\n"
           << "  \"logical_values\": " << kLogicalValues << ",\n"
           << "  \"steps\": " << kSteps << ",\n"
           << "  \"msv_scalar_ms\": " << msv_scalar_ms << ",\n"
           << "  \"msv_packed_ms\": " << msv_packed_ms << ",\n"
           << "  \"msv_speedup\": " << msv_scalar_ms / msv_packed_ms << ",\n"
           << "  \"viterbi_scalar_ms\": " << vit_scalar_ms << ",\n"
           << "  \"viterbi_packed_ms\": " << vit_packed_ms << ",\n"
           << "  \"viterbi_packed_speedup\": "
           << vit_scalar_ms / vit_packed_ms << ",\n"
           << "  \"viterbi_dpx_ms\": " << vit_dpx_ms << ",\n"
           << "  \"viterbi_dpx_speedup\": " << vit_scalar_ms / vit_dpx_ms
           << ",\n  \"dpx_recurrence_mismatches\": "
           << dpx_recurrence_mismatches << "\n}\n";
    output.close();
    if (!output) throw std::runtime_error("cannot write output");
    std::ifstream result(argv[1]);
    std::cout << result.rdbuf();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "phase7_packed_integer: " << error.what() << "\n";
    return 1;
  }
}
