# Phase 3 slice: resident Forward-to-Backward handoff

## Hypothesis

Backward/domain should not re-upload Forward special-state trajectories that
were produced on the same CUDA device moments earlier.  The existing Forward
host classification and host copy can remain unchanged while this first slice
removes the redundant active-row H2D replay.

## Implementation

The sealed production Forward entry point optionally retains its gathered F3
pass trajectories in a generation-owned device allocation.  The ordinary
Forward entry points remain host-only.  Allocation is bounded by the existing
384 MiB output cap and the total possible candidate trajectory bytes; it is
attempted only after all persistent Forward work buffers are viable.  Device
allocation pressure falls back to the unchanged host path.

Backward/domain has a new opaque-output seam.  It validates the Forward
database generation, sequence-batch generation, pass/special counts, CUDA
device, device pointer, and the existing complete host provenance before
launch.  Active rows use absolute offsets into the retained buffer, removing
the compact active-special allocation, host repack, and H2D copy.  All other
Backward inputs, kernels, classification, output, and provenance are
unchanged.

The retained buffer is owned by `plan7_forward_output`, so it cannot be
invalidated by reuse of the batch's persistent tile workspace.  It is released
on the originating CUDA device when that output dies.  This ownership boundary
prevents reuse of the trajectory allocation across generations; a later
request-scoped arena can pool such allocations once it owns both stage
lifetimes.

Additive counters report requested/allocated/materialized resident bytes,
allocation fallback, resident allocation/materialization time, legacy Forward
special H2D bytes, eliminated H2D bytes, and Forward-special upload time.  The
Python native module exposes cumulative production counters through
`_forward_backward_residency_statistics()`.

## Correctness evidence

The complete CUDA/Cython extension builds for sm_75 and sm_90 with CUDA hidden.
The resident route deliberately uses the same host trajectory bytes for
provenance and final journal construction, and the same device trajectory bits
already emitted by the existing gather kernel.  Exact GPU/output-oracle and
transfer-counter validation are pending on the next available H200 run.

## Status

Provisionally retained as the smallest Forward-to-Backward residency slice.
It does not yet remove the Forward D2H copy needed by the current journal, make
Backward workspaces persistent, retain Backward output into rescore, or compile
host F3 classification onto the device.
