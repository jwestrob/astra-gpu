# Phase 3 slice: resident Backward-to-rescore handoff

## Hypothesis

Isolated-domain rescore should not upload a host reconstruction of the SIMPLE
region identity that Backward/domain just produced on the same CUDA device.
The host result and interval arrays remain necessary for the current semantic
seal, planning caps, audit journal, and conservative CPU fallback, but they do
not need to be the production kernel input.

## Implementation

Backward/domain now attempts a bounded, fail-soft allocation for one compact
device descriptor per emitted SIMPLE interval.  The existing region-gather
kernel writes this descriptor beside the unchanged host-bound interval.  Each
descriptor records the profile, target, exact interval, target length, and
own-scale state.  The allocation is owned by the opaque Backward output and is
released on its originating CUDA device.

The resident view is accepted only when its database and batch generations,
CUDA device, result and region provenance hashes, row and region counts, and
materialized byte counts all match the sealed host output.  Allocation failure
leaves the view unavailable and rescore follows the old host replay unchanged.
The test-only resealed own-scale mutation explicitly invalidates the old
device generation and likewise uses the legacy path.

Rescore still performs the same host seal validation and cap planning.  For
admitted regions it uploads an ordered 8-byte `(region,row)` selection instead
of the previous 32-byte work descriptor plus 64-byte initial result.  A small
device kernel expands the selection from the authenticated resident
descriptors before the existing isolated Forward, Backward/decode, and
OA/null2/trace kernels run.  Final downloads, host identity validation,
row-atomic fallback, output provenance, and journal semantics are unchanged.

Additive counters report resident descriptor allocation/materialization,
legacy and eliminated upstream H2D bytes, compact selection H2D bytes, and the
resident preparation time.  For `N` active intervals the handoff changes from
`96*N` upstream bytes to `8*N` selection bytes; other rescore workspace
transfers are unchanged in this slice.

## Correctness evidence

The complete native extension builds with both sm_75 and sm_90 code
generation.  With CUDA hidden, the extension imports, the additive counter ABI
is present, and the focused Backward/rescore sparse-source remap tests pass.
Exact GPU and output-oracle validation remains required on H200 before this
slice is promoted.

## Status

Retained after the combined exact H200 full-workload job `1182713`, recorded
in `PHASE3_EXACT_F3_THRESHOLD.md`.  All 83 chunks used the resident rescore
handoff with zero legacy upstream H2D bytes and zero allocation fallbacks;
21,425,952 bytes of replay were eliminated.  This does not remove the host
Backward journal, move rescore cap planning to the device, retain rescore
outputs for final egress, or introduce a persistent workspace.
