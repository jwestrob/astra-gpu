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

## H200 result

Focused job `1182742` forced the expanded and compact paths over 40 profiles
and 320 targets spanning five length classes. Postfilter offsets, complete
records, and final HMMER output were byte-identical. The compact path uploaded
5 bytes instead of the counterfactual 320-byte dense table for its single
profile scale.

Full job `1182743` reproduced the established output exactly: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines. All 83 chunks used length-class metadata.
They uploaded 121,595 compact bytes, avoided 24,915,438 bytes of dense H2D,
and materialized the same 24,915,438-byte dense device layout consumed by the
unchanged kernels.

Request wall was 454.247490 seconds, generation 342.647949 seconds, CPU
continuation/output 402.232751 seconds, pipeline wall 446.652565 seconds, and
overlap 298.384871 seconds. Against Phase 5, request wall improved by 0.778958
seconds (0.171%) and generation improved by 2.546776 seconds (0.738%); CPU
continuation varied upward by 0.548329 seconds. Peak sampled H200 memory was
3,376 MiB and maximum RSS was 13,305,384 KiB.

The result is retained. It is a small end-to-end gain, but it removes exact
duplicate host computation and transfer without changing downstream layout or
penalizing small workloads.
