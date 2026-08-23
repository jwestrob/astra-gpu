# Phase 1A production sparse-v3 opt-in

## Hypothesis

A fused `CandidateBatch` can plan and validate one sparse journal-v3 packet at
seal time, then reuse immutable per-profile spans from concurrent row searches.
This removes dense replay from an opted-in search without changing the default
journal-v2 path or treating a batch-wide packet as a one-shot row result.

## Implementation

`SequenceBatch._postfilter_forward_selection(..., sparse_journal_v3=True)`
builds the packet once while the selected profile pairs are locked. The sealed
batch owns and frees it. `CandidateBatch.search(row, pipeline)` dispatches only
that profile through row-scoped preflight, counter-capacity checks, scratch, and
continuation. The packet is structurally validated, integrity checked, and
bound to its v2 provenance and dense source spans without rebuilding it.
Capsule validation and the dual oracle retain the independent rebuild and full
byte comparison. The opt-in defaults to false.

Planning and validation time are reported once per sealed batch by
`_sealed_continuation_statistics_bound`; ordinary continuation `wall_ns` is the
native time for one row consumer and excludes adapter lock wait. Retained packet
bytes are included in resident-memory accounting.

## Correctness evidence

CUDA-hidden focused tests verify default-v2 versus opt-in dispatch, reusable
sparse output against dense output, canonical route telemetry, and failure
before pipeline mutation for incompatible options. Existing focused direct and
dual journal-v3 tests still pass for terminal, F3, no-region, and multi-profile
fixtures.

## Benchmark status

No GPU or full workload benchmark was run for this commit. The implementation
is retained behind the explicit opt-in pending the authenticated H200 oracle
and representative continuation timing.
