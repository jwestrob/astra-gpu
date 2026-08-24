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
uploads. The CUDA 12.5 build and 15 CUDA-hidden host/ABI tests pass. Full H200
job `1182619` also passed its first-1,000 gate and reproduced the established
full output exactly: SHA-256
`3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`,
39,010,327 bytes, and 383,235 lines.

The full request used device compaction for all 83 profile chunks, with zero
host expansions, zero candidate-mapping uploads, and 83 uploads avoided.

## Benchmark result

Job `1182619` completed in 538.822140 seconds: 468.243264 seconds native
generation, 401.386855 seconds continuation/output, and 338.474725 seconds
overlap. This was 0.495% slower than the one-pass sparse-v3 run (536.168411
seconds), but 1.354% faster than the original dense baseline (546.220705
seconds). A single-run difference of this size does not establish a Phase 2
speedup.

Evidence is under
`build/phase1b-benchmark-harness/build/h200-phase2-device-compaction-20260823/attempt-02-full/runs/h200-full`.
The worker record SHA-256 is
`dcc3da2eafdcf076ab76a867472282cc290a529b8e7b10e64906e4a5709363f9` and
the raw validation SHA-256 is
`7428eb2fd2c4cf152564f51b512b6f6ec0636b21cf1bed2945e98f7b8e022f2b`.

## Status

Retained as a structural prerequisite for the device-resident pipeline, not
as a standalone performance win. No SSV arithmetic, statistical decision, or
Phase 3 host planning was changed.
