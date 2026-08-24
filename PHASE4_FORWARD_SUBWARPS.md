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

## Production policy v1

The follow-up candidate-count sweep showed that wider is not monotonically
better.  Once XMX/DP traffic substantially exceeds cache, the larger supply of
independent width-1 warps can hide memory latency better.  The initial
production policy therefore uses the largest physical tile, not total request
size, and is deliberately bounded by both CTA coverage and working-set size.

Definitions:

- minimum coverage is `ceil(3 * SM_count / 4)` CTAs;
- a long row set has average exact `M * L >= 131,072` cells/candidate;
- high length divergence means `max(L) > 2 * average(L)`;
- the width-4 short/medium working-set limit is the device L2 size;
- the packed long-row working-set limit is twice the device L2 size.

Dispatch:

1. If width 2 cannot provide minimum CTA coverage, use width 1.
2. For short/medium work, use width 4 when it has minimum coverage and the
   tile XMX allocation fits in L2.  If width 4 lacks coverage and target
   lengths are highly divergent, use width 1.  Otherwise use width 2.
3. For long work whose XMX allocation fits in twice L2, use width 4 when it
   has minimum coverage, otherwise width 2.  If it exceeds twice L2, use
   width 1.
4. Width 8 is never selected automatically.

The private debug override accepts forced widths 1, 2, 4, and 8.  Auto is
encoded as zero.  Additive counters expose the request and selected width,
reason code, SM/L2 inputs, M/L sums and maxima, average work, XMX bytes, CTA
counts for widths 1/2/4, launches, and active/issued lane slots.

H200 policy job 1182708 used 132 SMs and 60 MiB L2.  Auto output was identical
to every forced exact output and selected the measured winner in all four
representative cases:

| Workload | Auto width | Auto median | Speedup vs width 1 |
|---|---:|---:|---:|
| M87, L64, 8,192 candidates | 4 | 1.424 ms | 1.311x |
| M243, L256, 8,192 candidates | 4 | 15.106 ms | 1.170x |
| M400, L1024, 2,048 candidates | 2 | 33.899 ms | 1.025x |
| M262, L31-1024, 4,096 candidates | 4 | 20.968 ms | 1.154x |

The exact report SHA-256 is
`813cce619f03a9decf13ab6e54ad6a88d46bde26e2a4893f78cb0e3a211df593`.

H200 boundary job 1182710 then exercised the remaining routes with exact
forced-width comparison: divergent sparse width 1, short/medium width 2,
long width 4, and long working-set-saturated width 1.  Across the twelve
boundary workloads the selected-policy timing ranged from 0.993x (noise-level
for the same forced width-1 kernel) to 1.485x versus width 1.  Its scale
0.5/2/4 report hashes are, respectively:

- `25548ef53ef8074b3ea2676126c6fe1449d853a14a2bda50f9db8e58daffc89d`
- `12d95f0c4a32b79c1edbd0e62ded990e3c22878810c0a98971e3809fed5c24b6`
- `33a4dfb0cfd9db7e1699617af47e6103cd7c3fd6e7b727a05736db9d87179ed6`

That microbenchmark-only promotion decision was provisional and is superseded
by the full-workload result below.

## Full-workload result and final decision

H200 job `1182718` ran the complete 300,186-target by 27,481-profile workload.
It reproduced the established CPU64 output exactly: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.  Policy v1 selected width 2 for all 83
full-workload chunks.

The policy was slower than the otherwise identical combined Phase 3 run:

| Measurement | Phase 3 width 1 | Policy v1 | Change |
|---|---:|---:|---:|
| Forward kernel total | 32.823 s | 33.892 s | +1.069 s / +3.26% |
| Generation | 463.471 s | 469.576 s | +6.105 s / +1.32% |
| Request wall | 535.213 s | 539.052 s | +3.839 s / +0.72% |

The full-workload evidence rejects automatic promotion.  Ordinary production
entry points are fixed at width 1 again.  Widths 2/4/8 and the policy selector
remain available only through the private diagnostic variant for future
length-cohort research; width 1 is the retained production implementation.

The full worker and raw-validation SHA-256 values are, respectively,
`cd59de5ffed5a5f3c4622844adea9504b96421c767ba9b152e62ca7cdca565e3`
and `62df054b02c81475af43e03960a4f9163df01239463af003bfdd64b393878bb5`.
