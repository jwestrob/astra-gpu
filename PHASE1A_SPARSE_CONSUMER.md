# Phase 1A sparse continuation consumer

## Hypothesis

The CPU continuation does not need to visit every dense postfilter record.
Rows that are already resolved by the GPU can be replaced by exact accounting
certificates, leaving only ordered exception rows for the existing HMMER
continuation seams.

## Implementation

This change adds a private journal-v3 consumer and a dual-consumer audit entry
point.  The production journal-v2 path is unchanged.

For each profile, the sparse consumer:

1. applies checked certificate deltas for omitted target and residue spans;
2. applies the exact F1/F2/F3 promotion-counter contributions of those spans;
3. configures each exceptional target and invokes the same continuation seam
   used by the dense path;
4. reuses the pipeline after each exception;
5. applies the tail certificate and restores the exact final target-length
   configuration.

Before mutation it validates the packet, live pipeline state, profile and
sequence identities, route/certificate partitions, counter overflow, and the
host floating-point environment.  A packet is one-shot: preflight failures do
not claim it, while failures after execution starts retire it.

The dual audit executes dense journal-v2 and sparse journal-v3 continuations on
independent pipeline, background, profile, query, and TopHits state.  It
requires exact equality of:

- the canonical semantic pipeline fingerprint;
- the canonical TopHits fingerprint;
- target and domain table bytes;
- compact retry/acceptance counts;
- dense routes after adding the certified sparse terminal stages.

## Correctness evidence

With CUDA hidden and the pinned private PyHMMER ABI, the focused continuation
module passes 61 tests (3 expected CUDA-only skips).  Executed dual-consumer
cases include:

- rejected prefixes and tails, consecutive exceptions, and no exceptions;
- F1, bias, F2, and F3 terminal certificates;
- an authenticated domain/no-region certificate;
- first and final target exceptions, including final target length zero;
- multiple profiles;
- fixed and dynamic Z/domZ behavior and cumulative pipeline reuse;
- overflow, floating-point-environment drift, malformed state, and one-shot
  ownership failures.

An independent source review found no remaining host-semantic blocker.

The authenticated real-CUDA gate passed on an attested H200 in Slurm job
1182321 (exit 0). The source revision was
`4649a91e4b3fe9ffea1519feca9ff577a79005d0`; the loaded native and pipeline
extension SHA-256 values were respectively
`8067fb336e3ef7412a0b01972177d94a867b01a8307c70396ad9b8bc6fd35b03`
and `fa3dff21bdf35f7f9e8f1fee10748f3aa3e51e77726bfcf191c1a382339f66`.
The fixture exercised 10 Forward-score, 3 simple-region, and 3 compact-domain
routes, with 3 compact results accepted. Dense and sparse pipeline/TopHits
fingerprints and target/domain table bytes matched exactly in every profile.
The result JSON SHA-256 is
`ebb10bf2cf665f8efc120725f0ba7693585c231f019df325a68e4ed2d79f463b`.

## Benchmark status

No production performance claim is made by this commit.  The private planner
and validator deliberately rebuild and byte-compare an expected packet before
execution.  That work is useful for proof but must be removed from, or timed
separately from, the production path before measuring continuation speed.

For scale only, the production-route fixture's v2 packet was 85,248 bytes and
its v3 packet was 63,700 bytes. A forced genuine simple-fallback case changed
80,912 bytes to 30,348 bytes. These tiny-fixture byte reductions are
correctness evidence, not a full-workload forecast.

## Decision

Retain this as the Phase 1A equivalence/reference implementation. Keep journal
v2 as production until the proof-validation cost is separated and the sparse
path's total wall time is measured. The next optimization step is to switch
the CPU continuation to validated v3, then generate the sparse certificate
before dense host materialization while preserving this dual oracle as the
audit path.
