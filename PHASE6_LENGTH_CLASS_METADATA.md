# Phase 6: compiled target-length metadata

## Hypothesis

F1 currently computes and uploads one `tjb` transition byte for every unique
profile scale times every target, even though `tjb` depends only on the profile
scale and target length. The full workload has 300,186 targets but only 1,465
distinct lengths, so the host table contains extensive exact duplication.

## Implementation

`SequenceBatch` now builds one stable target-length class map at creation and
uploads that map once. During fused F1, the host computes one transition byte
per unique profile scale and length class. A small CUDA expansion kernel writes
the existing dense profile-scale-by-sequence table on device. All existing F1,
bias, postfilter, and continuation kernels continue to consume the same dense
layout and the same original sequence ordinals.

Automatic use requires at least 256 targets and at least twofold length-class
compression. `PLAN7_GPU_SSV_LENGTH_METADATA=expanded` forces the unchanged
reference path and `compact` forces the new path for exact audit. Counters
report executions, class values, compact H2D bytes, counterfactual dense H2D
bytes avoided, and dense bytes materialized on device. Device allocation
accounting includes both the persistent class map and compact transition table.

No filter arithmetic, cutoff, profile ordering, sequence ordering, or
downstream ABI changes in this phase.

## Acceptance

- Forced-expanded and forced-compact postfilter offsets and complete records
  must be byte-identical.
- Final target/domain output must be byte-identical.
- The full workload must use the length-class path on all profile chunks while
  retaining the scalar/small-workload path.
- Report request, generation, continuation, overlap, compact H2D bytes, dense
  H2D bytes avoided, materialized bytes, and peak memory.

H200 evidence and the retain/reject decision will be appended after execution.
