# Phase 4: packed Forward subwarps

## Hypothesis

The exact Forward recurrence uses four CUDA lanes to reproduce one HMMER SSE
row.  The production kernel issued only one such recurrence per 32-lane warp.
Packing 2, 4, or 8 independent candidates into the otherwise idle four-lane
groups should increase throughput without changing arithmetic within a
candidate.

## Implementation

`forward_kernel<CandidatesPerWarp>` is instantiated for 1, 2, 4, and 8
candidates.  Each candidate retains its original profile-major ordinal and its
own DP/XMX spans.  Its subwarp mask is still exactly four adjacent lanes, and
the sequence of explicit round-to-nearest additions, multiplications,
divisions, shuffles, and horizontal sums is unchanged.

Production dispatch remains fixed at `CandidatesPerWarp=1`.  A separate
diagnostic native entry point and the private Python argument
`_candidates_per_warp` can force another specialization for exact comparison
and benchmarking.  The existing result, statistics, journal, and provenance
ABIs are unchanged.  Additive opaque-output counters record launches,
candidate subwarps, scheduled warps, and active/issued lane slots.

`scripts/benchmark_forward_subwarps.py` first performs a complete byte oracle,
then measures CUDA-event kernel time.  Timing rows use an exact F3 rejection so
survivor gathering does not obscure Forward recurrence time.

## Correctness evidence

- The complete extension built with native `sm_75` and `sm_90` cubins.
- `cuobjdump --dump-resource-usage` reports the same register allocation for
  every width: 80 registers/thread on `sm_75` and 64 registers/thread on
  `sm_90`.
- H200 job 1182698 compared widths 1/2/4/8 on 83 ordered candidates across
  model lengths 87, 243, and 400 (`Q=22,61,100`), per-profile candidate counts
  1, 17, and 65, and target lengths 1 through 511.  Result records, special
  offsets, every returned XMX bit, and all provenance hashes were identical.
  The combined payload SHA-256 was
  `6e424630681c23b2d18a6b928f605a64ed25ced470cd78ced50f75986a6eed8d`.
- H200 job 1182701 passed the existing independent HMMER known-row oracle
  `CudaForwardTests.test_versioned_abi_and_exact_known_rows`.

The raw H200 report was
`build/h200-forward-subwarps/result.json`, SHA-256
`4eb10de8e36f3f7a952bad8ce833ca8ccdba34041819fb3aa0a93aa6ce444749`.

## H200 microbenchmark

Five timed samples were collected per specialization.  Values below are the
median Forward kernel time; speedup is relative to width 1.

| Workload | Width 1 | Width 2 | Width 4 | Width 8 | Best |
|---|---:|---:|---:|---:|---:|
| M87, L64, 8,192 candidates | 1.872 ms | 1.459 ms / 1.283x | 1.415 ms / 1.323x | 1.434 ms / 1.305x | width 4 |
| M243, L256, 8,192 candidates | 17.672 ms | 15.822 ms / 1.117x | 15.068 ms / 1.173x | 15.470 ms / 1.142x | width 4 |
| M400, L1024, 2,048 candidates | 35.076 ms | 33.779 ms / 1.038x | 37.214 ms / 0.943x | 105.087 ms / 0.334x | width 2 |
| M262, L31-1024, 4,096 candidates | 23.839 ms | 23.112 ms / 1.031x | 20.954 ms / 1.138x | 25.348 ms / 0.940x | width 4 |

Width 8 is not a universal winner.  For the 2,048-candidate long-row case it
launches only 32 CTAs, leaving much of the H200 grid under-subscribed; width 2
launches 128 CTAs.  Mixed target lengths also penalize the widest warp because
all subwarps remain occupied until the longest candidate in the warp finishes.

## Decision

Retain the four exact specializations and diagnostic harness.  Do not change
the production width yet.  The evidence supports a small shape policy rather
than a fixed width: preserve width 1 for sparse work, prefer width 4 for dense
short/medium or length-compatible rows, and consider width 2 for long rows
unless the candidate count provides enough CTAs.  Length cohorts should be
tested before width 8 is considered for production.
