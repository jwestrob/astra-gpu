# Phase 1B: direct sparse-v3 source

## Hypothesis

When sparse journal v3 is explicitly enabled, the native generation outputs
already contain every fact needed by the validated Phase 1A planner. Packing
those facts into a dense v2 allocation, validating that allocation, and then
copying only its exceptions into v3 is redundant.

## Implementation

The opt-in native path now returns a transient, segmented generation bundle
instead of allocating v2. The bundle carries immutable selection identity,
content fingerprints, Forward/Backward/rescore provenance, exact generation
options, and the live dense views required by the Phase 1A planner. The planner
emits an authenticated `NATIVE_DIRECT` v3 packet, validates it against those
facts, and immediately drops every dense planning view. Production continuation
and Phase 0 route statistics then read the v3 certificates and exceptions.

The default path and explicit dense dual-audit path still use journal v2.
Calling a dense replay/audit consumer on a direct-v3 seal fails closed.

Counters report dense-v2 validation/emit time and bytes, direct source
validation/staging time and bytes, eliminated v2 bytes, retained dense bytes,
v3 certificate/exception counts, and sparse-consumer preflight/core/statistics
time plus certificate/exception visits.

## Correctness evidence

- Both Cython extensions compile against the patched PyHMMER private ABI.
- Focused host tests preserve the v2 default, sparse-v3 dispatch, reusable
  sparse continuation, dense-vs-sparse state/output oracle, and resident-memory
  accounting.
- A visible-GPU first-1000 oracle and full H200 output comparison remain the
  required acceptance gates for the new native-direct source.

## Benchmark result

Pending the visible-GPU gates. This change removes dense-v2 construction and
retention, but deliberately retains the Phase 1A two-pass sparse planner. Its
measured planning cost therefore remains a separate optimization target.

## Decision

Retain provisionally behind explicit `sparse_journal_v3=True`. Do not switch
the default or remove v2 until the first-1000 and full-workload equivalence and
performance gates pass.
