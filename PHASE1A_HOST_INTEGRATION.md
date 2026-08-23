# Phase 1A host integration record

## Scope

This record covers the host-only integration of the semantic-state fingerprint
and continuation-journal-v3 planner.  It does not switch production away from
journal v2, change CUDA transport, or claim a GPU performance result.

Integrated source before this record:

- commit: `2b7f926de0698993a64d416adac179259aa95df4`
- tree: `08d48f3298bd7746f2b7fafea10e5241ae850afa`
- PyHMMER private ABI: `d4867ff865e9b8a7acdbbf9106e3d7e1223336d374cb0f46d7e352427b990689`

The integration contains these independently reviewed chains:

- semantic fingerprint: `583dce2`, `6f5fac6`, `1635f6f`
- journal-v3 planner: `7a9344c`, `6529a6f`

The only textual merge conflict was the additive Cython import set.  The
resolution retained `calloc` for v3 and `int64_t`, `uintptr_t`, and `strlen`
for the semantic fingerprint.  An independent post-merge audit found every
Phase-0, fingerprint, and v3 API exactly once and no lost symbols.

## Correctness evidence

Using the isolated patched PyHMMER 0.12.0 runtime, with CUDA hidden and Python
bytecode disabled:

- fresh native and pipeline build: PASS
- concrete native and pipeline `make -q`: PASS
- complete test discovery: 303 tests PASS, 129 expected CUDA/oracle skips
- native and pipeline runtime ABI checks: exact `d4867ff...689`
- Git conflict-marker, diff, object, and worktree checks: PASS
- Python bytecode in the worktree after validation: zero

Built extension hashes for this validation were:

- `_native`: `b3c13cd39f413bc8a7ecf48e037ba655336f9e737918d39b10a498b30f596ccf`
- `_pipeline`: `b68df236d62413c92d2d720fc467c8911e9a4e3d06c1b36fc23e51017769129c`

No GPU kernel and no Slurm job was run.

## Decision

Retain both host-side foundations.  Journal v2 remains the production and
audit reference.  The next gate is an actual sparse-v3 consumer plus a dual
dense/sparse execution mode that proves exact TopHits and semantic pipeline
state equality before any production switch.

The initially integrated Phase-0 telemetry commit is compatibility-tested here
but is not accepted as final measurement evidence; its independent audit fixes
remain a separate follow-up.
