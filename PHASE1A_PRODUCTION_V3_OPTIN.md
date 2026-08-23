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

The production opt-in passed on an attested H200 in Slurm job `1182344`
(exit 0) at source revision
`9a078c22f5ba386fdaeb726a688ef16179bb277c`. The loaded native and pipeline
extension SHA-256 values were respectively
`8067fb336e3ef7412a0b01972177d94a867b01a8307c70396ad9b8bc6fd35b03`
and `1042c2ff578d8426b54f4eaa7a5d5d98cacf3d7e01bee3cc55cc0286499b8b3d`.
Real CUDA work produced 10 Forward-score, 3 simple-region, and 3
compact-domain routes, including 3 accepted compact results. All six
profile/case comparisons had exact route reconciliation, semantic pipeline
state, TopHits, and target/domain table bytes. Direct production
`CandidateBatch.search()` results also matched the dense oracle for every
profile. The result JSON SHA-256 is
`caf9eaca0ad5cd2a2b910f8c313fdb1ba2caaae3eed5f29a31deaf94ade8b101`.

## Benchmark result

The current-code first-1000 gate passed exactly in Slurm job `1182348`. The
full 300,186-target by 27,481-profile request then completed in job `1182349`
with byte-identical output:

- SHA-256 `3d7cda45ab1fca27fbb3b03a58bc501936666b7419fe0b6670fe46947e9f18e6`;
- 39,010,327 bytes and 383,235 lines;
- all candidate, continuation-route, compact-accept, and retry counts exact.

The performance hypothesis failed in its present form. Outer request time was
567.080 seconds versus 546.221 seconds for sealed dense v2: 20.859 seconds, or
3.82%, slower. Generation was 502.286 seconds, continuation/output 404.272
seconds, and measured overlap 347.054 seconds. Dense-v2 payload fell from
9,890,721,120 to 5,801,342,068 bytes (41.35%), but building the sparse packet
from the completed dense history and structurally validating it cost 28.945
seconds. CPU continuation itself did not materially improve.

## Decision

Do not enable this Phase 1A implementation by default. Retain it as the exact
audit/reference path and as the semantic foundation for Phase 1B. The next
experiment must generate v3 directly and avoid constructing and retaining the
dense replay history; merely compacting v2 after the fact is rejected.
