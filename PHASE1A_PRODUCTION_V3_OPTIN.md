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

## Benchmark status

The authenticated H200 correctness gate passed. The tiny fixture reduced the
production packet from 85,248 to 63,700 bytes and the forced simple-fallback
packet from 80,912 to 30,348 bytes; those values are correctness-scale
evidence, not a performance claim. The implementation remains behind the
explicit opt-in pending the representative first-1000 and full-workload wall
time and output oracle.
