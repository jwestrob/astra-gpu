# Environment inventory

Inventory captured on 2026-08-14. The complete command output is stored in
[`results/hardware.json`](results/hardware.json) and can be regenerated with
`scripts/inventory_environment.py`.

## Interactive development host (`biotite`)

- GPU: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5, 11,264 MiB VRAM
- GPU driver: 555.42.06
- CUDA toolkit: 12.5 (`nvcc` 12.5.82)
- CPU: 4 x Intel Xeon Platinum 8176M sockets
- CPU topology: 28 physical cores/socket, 112 physical cores total, SMT2
- NUMA: four nodes; the GPU is attached to NUMA node 1 over PCIe
- RAM: approximately 3 TiB across the four NUMA nodes
- OS: Ubuntu 22.04 userland, Linux 5.15 kernel
- Slurm: 23.02.1
- Compiler: GCC/G++ 11.4.0
- CMake: 3.24.0
- Nsight Systems: 2024.2.3
- Nsight Compute: installed with CUDA 12.5

`numactl` is not installed. CPU and memory placement must therefore be
controlled with Slurm binding, `taskset`, or equivalent facilities. Unbound
2080 Ti comparisons are invalid because remote-socket memory traffic can
dominate transfer measurements.

The upstream HMMER 3.4 configure probe selects its SSE implementation on this
host. It does not select AVX2/AVX-512 for the Plan7 filters even though the CPU
supports those instruction sets.

## Slurm GPU resources

- `gpu`: two nodes, eight GPUs configured on each node
- `gpu_h200`: one node (`node-224-2t-8gpu-1`), eight GPUs configured

The `gpu_h200` name indicates the intended Hopper target, but the precise GPU,
driver, interconnect, CPU, and NUMA topology must be captured from inside an
allocation before reporting H200 results. The interactive 2080 Ti is for rapid
development; the H200 node is the intended modern performance target.

## Benchmark controls

- Count physical cores, not SMT threads, for the primary scaling series.
- Record affinity and NUMA placement with every run.
- Use exclusive or otherwise uncontended allocations for reported timings.
- Report the pristine HMMER build separately from instrumented/oracle builds.
- Warm the relevant filesystem/cache state consistently and report whether
  compute-only or end-to-end I/O is being measured.
