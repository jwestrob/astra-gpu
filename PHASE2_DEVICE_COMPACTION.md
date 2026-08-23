# Phase 2: device-side stable F1 candidate compaction

## Hypothesis

The exact fused F1 kernel can retain its profile-major bit mask on the GPU.
Stable device compaction should remove the dense mask download, host popcount/
expansion, and candidate mapping upload without changing any F1 decision or
candidate order.

## Implementation

The production path now applies a per-word popcount, CUB exclusive scan, and
stable word/bit scatter into the existing `plan7_bias_candidate` device
buffer. It downloads only profile CSR offsets and compact candidate mappings
needed by the current host-side postfilter/Forward planner. Bias and postfilter
reuse the authenticated cached mapping instead of uploading it again.

The old host-mask expansion remains selectable with
`_host_candidate_expansion=True` and is automatic when the mask word count
exceeds CUB's `INT_MAX` item-count limit. Scan inputs, offsets, profile offsets,
and CUB temporary storage are persistent high-water buffers included exactly
in the batch memory snapshot.

## Correctness evidence

The CUDA oracle compares old and new CSR offsets, indices, and a combined hash
for nonadjacent profile selections at 33- and 65-target word boundaries. It
also covers empty, all-reject, and near-all-pass mappings. A downstream bias
test asserts that the cached device mapping is consumed with zero candidate
uploads. The CUDA 12.5 build and 15 CUDA-hidden host/ABI tests pass.

GPU execution of the new oracle is deliberately pending; this commit was
built and host-tested only so it cannot interfere with the running benchmark.

## Status

Provisionally retained for the focused GPU oracle and benchmark. No SSV
arithmetic, statistical decision, or Phase 3 host planning was changed.
