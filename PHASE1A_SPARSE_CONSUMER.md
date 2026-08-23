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

An independent source review found no remaining host-semantic blocker.  The
Forward, simple-domain, compact-result, and compact-retry routes still require
the authenticated real-CUDA dual oracle before production activation.

## Benchmark status

No production performance claim is made by this commit.  The private planner
and validator deliberately rebuild and byte-compare an expected packet before
execution.  That work is useful for proof but must be removed from, or timed
separately from, the production path before measuring continuation speed.

## Decision

Retain this as the Phase 1A equivalence/reference implementation.  Keep
journal v2 as production until the real-route CUDA oracle passes and the proof
validation cost is separated.  The next optimization step is to generate the
sparse certificate before dense host materialization while preserving this
dual oracle as the audit path.
