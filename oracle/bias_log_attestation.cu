#include <cuda_runtime.h>
#include <omp.h>

#include <cpuid.h>
#include <gnu/libc-version.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void
gpu_logs(uint32_t first, float *outputs, uint32_t count)
{
  const uint32_t index =
    static_cast<uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const uint32_t raw = first + index;
  float input;
  memcpy(&input, &raw, sizeof(input));
  outputs[index] = static_cast<float>(log(static_cast<double>(input)));
}

uint32_t
bits(float value)
{
  uint32_t raw;
  memcpy(&raw, &value, sizeof(raw));
  return raw;
}

struct mismatch {
  uint32_t input;
  uint32_t host;
  uint32_t device;
};

int
main(int argc, char **argv)
{
  constexpr uint32_t kLast = UINT32_C(0x7f7fffff);
  const uint32_t chunk = argc > 1
                           ? static_cast<uint32_t>(strtoul(argv[1], nullptr, 0))
                           : UINT32_C(1) << 24;
  const int thread_count = argc > 2 ? atoi(argv[2]) : omp_get_max_threads();
  if (chunk == 0 || thread_count < 1) {
    fprintf(stderr, "chunk and host thread count must be positive\n");
    return 2;
  }

  int device_ordinal;
  cudaDeviceProp properties;
  cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess ||
      (status = cudaGetDeviceProperties(&properties, device_ordinal)) !=
        cudaSuccess) {
    fprintf(stderr, "CUDA identity failure: %s\n", cudaGetErrorString(status));
    return 2;
  }
  unsigned eax;
  unsigned ebx;
  unsigned ecx;
  unsigned edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    fprintf(stderr, "CPUID identity failure\n");
    return 2;
  }
  const unsigned stepping = eax & UINT32_C(0x0f);
  const unsigned family = (eax >> 8) & UINT32_C(0x0f);
  const unsigned model = ((eax >> 4) & UINT32_C(0x0f)) |
                         (((eax >> 16) & UINT32_C(0x0f)) << 4);
  fprintf(stderr,
          "identity cpu_family=%u cpu_model=%u cpu_stepping=%u glibc=%s "
          "nvcc=%d.%d.%d cudart=%d gpu=%s cc=%d.%d\n",
          family, model, stepping, gnu_get_libc_version(),
          __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__,
          CUDART_VERSION, properties.name, properties.major, properties.minor);

  omp_set_num_threads(thread_count);
  std::vector<float> device_outputs(chunk);
  float *device_buffer = nullptr;
  status = cudaMalloc(&device_buffer, static_cast<size_t>(chunk) * sizeof(float));
  if (status != cudaSuccess) {
    fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(status));
    return 2;
  }

  unsigned long long total_mismatches = 0;
  unsigned maximum_ulp = 0;
  std::vector<mismatch> examples;
  const double started = omp_get_wtime();
  for (uint64_t first64 = 1; first64 <= kLast; first64 += chunk) {
    const uint32_t first = static_cast<uint32_t>(first64);
    const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(
      chunk, static_cast<uint64_t>(kLast) - first64 + 1));
    gpu_logs<<<(count + 255) / 256, 256>>>(first, device_buffer, count);
    status = cudaMemcpy(device_outputs.data(), device_buffer,
                        static_cast<size_t>(count) * sizeof(float),
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      fprintf(stderr, "CUDA failure at 0x%08x: %s\n", first,
              cudaGetErrorString(status));
      cudaFree(device_buffer);
      return 2;
    }

    unsigned long long chunk_mismatches = 0;
    unsigned chunk_maximum_ulp = 0;
#pragma omp parallel for schedule(static) reduction(+:chunk_mismatches) reduction(max:chunk_maximum_ulp)
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t raw = first + index;
      float input;
      memcpy(&input, &raw, sizeof(input));
      const uint32_t host_bits = bits(
        static_cast<float>(std::log(static_cast<double>(input))));
      const uint32_t device_bits = bits(device_outputs[index]);
      const unsigned difference = host_bits > device_bits
                                    ? host_bits - device_bits
                                    : device_bits - host_bits;
      chunk_mismatches += difference != 0;
      chunk_maximum_ulp = std::max(chunk_maximum_ulp, difference);
      if (difference != 0) {
#pragma omp critical
        {
          if (examples.size() < 128)
            examples.push_back({raw, host_bits, device_bits});
        }
      }
    }
    total_mismatches += chunk_mismatches;
    maximum_ulp = std::max(maximum_ulp, chunk_maximum_ulp);
    if ((first64 & UINT64_C(0x0fffffff)) == 1 ||
        static_cast<uint64_t>(first) + count > kLast)
      fprintf(stderr, "through=0x%08x mismatches=%llu elapsed=%.3f\n",
              first + count - 1, total_mismatches,
              omp_get_wtime() - started);
  }

  printf("inputs=%u mismatches=%llu max_ulp=%u seconds=%.6f threads=%d\n",
         kLast, total_mismatches, maximum_ulp, omp_get_wtime() - started,
         thread_count);
  for (const mismatch &item : examples)
    printf("input=0x%08x host=0x%08x device=0x%08x\n",
           item.input, item.host, item.device);
  cudaFree(device_buffer);
  return total_mismatches == 0 ? 0 : 1;
}
